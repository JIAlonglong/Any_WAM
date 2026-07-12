"""Offline Stage 2 rollout metrics.

This script compares student rollouts with teacher/reference states on the
training dataset. It does not step a LIBERO environment and therefore does not
produce real robot execution videos. Use evaluation/libero/run_eval_new.sh for
LIBERO environment videos and success-rate evaluation.
"""

import argparse
import gc
import importlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
sys.path.append(str(REPO_ROOT / "wan_va"))
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "distillation_flowmap"))

from distillation.patches import install_flash_attn_stub
install_flash_attn_stub()

from distributed.util import init_distributed, dist_mean
from utils import init_logger, logger
from distillation_flowmap.flowmap_trainer import FlowMapDistiller
from distillation_flowmap.flowmap_step import _downsample_action_grid_id
from distillation_flowmap.rollout_eval_steps import normalize_rollout_steps
from distillation_flowmap.rollout_masking import (
    masked_video_mse_l1,
    masked_video_rms,
    video_frame_mask_from_batch,
)
from distillation_flowmap.ablation.robotwin_mini_protocol import (
    dataset_records_for_manifest,
    eval_seed_for_pair,
    load_eval_pairs,
)


def parse_pair(text):
    t, r = text.split(",")
    return float(t), float(r)


def resolve_empty_emb_path(dataset_path, explicit_path=None):
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    dataset_path = Path(dataset_path)
    candidates.extend([
        dataset_path / "empty_emb.pt",
        dataset_path.parent / "empty_emb.pt",
    ])
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def teacher_cache_key(batch_idx, t_value, r_value, pair_id=None, sample_key=None):
    pair_key = pair_id or f"t{int(t_value)}_r{int(r_value)}"
    sample_key = sample_key or f"batch{batch_idx}"
    return f"{sample_key}/{pair_key}"


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_cli_pair_specs(pairs):
    specs = []
    for pair in pairs:
        t_value, r_value = parse_pair(pair)
        specs.append({
            "pair_id": f"t{int(t_value)}_r{int(r_value)}",
            "t": t_value,
            "r": r_value,
            "pair_seed": None,
        })
    return specs


def build_eval_records(trainer, eval_manifest_path=None, num_batches=1, split_name=None):
    if eval_manifest_path is None:
        batches = trainer._get_light_eval_batches()
        limit = max(1, int(num_batches))
        return [
            {
                "batch": batch,
                "sample_key": f"batch{batch_idx}",
                "global_index": None,
            }
            for batch_idx, batch in enumerate(batches[:limit])
        ]

    from torch.utils.data._utils.collate import default_collate

    manifest = load_json(eval_manifest_path)
    if split_name is not None and manifest.get("split") != split_name:
        raise ValueError(
            f"Eval manifest split is {manifest.get('split')!r}, expected {split_name!r}"
        )
    dataset = trainer.train_loader.dataset
    records = dataset_records_for_manifest(
        dataset,
        manifest,
        manifest_is_compact=bool(
            getattr(trainer.config, "offline_eval_manifest_is_compact", False)
        ),
    )
    if int(num_batches) > 0:
        records = records[:int(num_batches)]
    eval_records = []
    for record in records:
        global_index = int(record["global_index"])
        eval_records.append({
            "batch": default_collate([dataset[global_index]]),
            "sample_key": f"idx{int(global_index)}",
            "global_index": global_index,
            "task": record["task"],
        })
    return eval_records


def init_mask(trainer, input_dict):
    from modules.model import FlexAttnFunc

    ld = input_dict["latent_dict"]
    ad = input_dict["action_dict"]
    total_length = (
        ld["noisy_latents"].flatten(0, 1).shape[0] * 2
        + ad["noisy_latents"].flatten(0, 1).shape[0] * 2
    )
    padded_length = (128 - total_length % 128) % 128
    FlexAttnFunc.init_mask(
        ld["noisy_latents"].shape,
        ad["noisy_latents"].shape,
        padded_length,
        input_dict["chunk_size"],
        window_size=input_dict["window_size"],
        patch_size=trainer.patch_size,
        device=trainer.device,
    )


def masked_mse_l1(diff, mask):
    if mask is None:
        return (diff.float().pow(2).mean(), diff.float().abs().mean())
    mask = mask.float()
    denom = (mask.sum() * diff.shape[1]).clamp(min=1)
    return ((diff.float() * mask).pow(2).sum() / denom,
            (diff.float().abs() * mask).sum() / denom)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="distillation_flowmap.config_libero_fullfinetune_stage2_anyflow")
    parser.add_argument("--teacher-model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--empty-emb-path", default=os.environ.get("EMPTY_EMB_PATH"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from-path", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--pairs", nargs="+", default=["1000,0", "1000,500", "750,250"])
    parser.add_argument("--eval-manifest", type=Path, default=None)
    parser.add_argument("--eval-pairs-json", type=Path, default=None)
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--student-steps", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--teacher-steps", nargs="+", type=int, default=[4])
    parser.add_argument(
        "--load-target-student",
        action="store_true",
        help="Load EMA target student during offline eval. Disabled by default to reduce memory.",
    )
    parser.add_argument(
        "--disable-eval-gradient-checkpointing",
        action="store_true",
        help=(
            "Run offline student rollout under no_grad without forcing checkpoint "
            "recompute. This avoids retaining inference graphs on memory-constrained "
            "single-GPU eval jobs."
        ),
    )
    parser.add_argument(
        "--disable-eval-force-cfg",
        action="store_true",
        help=(
            "Do not force cond/uncond CFG when --cfg-scale <= 1 during offline "
            "student rollout. Training defaults are unchanged."
        ),
    )
    parser.add_argument(
        "--eval-rollout-grad-mode",
        choices=("endpoint", "last_step", "full"),
        default=None,
        help=(
            "Override opd_rollout_grad_mode for offline eval. Use endpoint for "
            "memory-only metric passes that do not need student rollout gradients."
        ),
    )
    parser.add_argument(
        "--eval-empty-cache",
        action="store_true",
        help="Release large temporary tensors and empty CUDA cache after each eval pair.",
    )
    parser.add_argument(
        "--use-nofsdp-teacher",
        action="store_true",
        help="Use the full non-FSDP teacher copy during offline eval. Disabled by default to reduce memory.",
    )
    parser.add_argument(
        "--teacher-cache-path",
        default=None,
        help="Path to a CPU teacher-target cache. If it exists, teacher loading is skipped.",
    )
    parser.add_argument(
        "--teacher-cache-only",
        action="store_true",
        help="Only compute and save teacher rollout targets, then exit before student eval.",
    )
    parser.add_argument(
        "--refresh-teacher-cache",
        action="store_true",
        help="Recompute teacher targets even when --teacher-cache-path already exists.",
    )
    args = parser.parse_args()
    teacher_steps = normalize_rollout_steps(args.teacher_steps)

    init_logger()
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)

    teacher_cache_path = Path(args.teacher_cache_path) if args.teacher_cache_path else None
    if args.teacher_cache_only and teacher_cache_path is None:
        raise ValueError("--teacher-cache-only requires --teacher-cache-path")
    if (
        args.teacher_cache_only
        and teacher_cache_path is not None
        and teacher_cache_path.exists()
        and not args.refresh_teacher_cache
    ):
        if rank == 0:
            logger.info("Teacher cache already exists at %s; not refreshing.", teacher_cache_path)
        if dist.is_initialized():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        return
    use_teacher_cache = (
        teacher_cache_path is not None
        and teacher_cache_path.exists()
        and not args.refresh_teacher_cache
        and not args.teacher_cache_only
    )
    teacher_cache = None
    if use_teacher_cache:
        teacher_cache = torch.load(teacher_cache_path, map_location="cpu")
        if rank == 0:
            logger.info("Loaded teacher cache from %s", teacher_cache_path)
    teacher_cache_to_write = {}

    cfg = importlib.import_module(args.config).cfg
    cfg.rank = rank
    cfg.local_rank = local_rank
    cfg.world_size = world_size
    cfg.teacher_model_path = args.teacher_model_path
    cfg.dataset_path = args.dataset_path
    cfg.empty_emb_path = resolve_empty_emb_path(args.dataset_path, args.empty_emb_path)
    cfg.output_dir = args.output_dir
    cfg.resume_from_path = args.resume_from_path
    if args.eval_manifest is not None:
        eval_manifest = load_json(args.eval_manifest)
        task_names = [str(entry["task"]) for entry in eval_manifest.get("tasks", [])]
        if not task_names:
            raise ValueError("Eval manifest must contain at least one task")
        cfg.dataset_sample_manifest = str(args.eval_manifest)
        cfg.dataset_task_filter = ",".join(task_names)
        cfg.offline_eval_manifest_is_compact = True
    cfg.reset_resume_step = False
    cfg.enable_wandb = False
    cfg.enable_light_eval = False
    cfg.enable_stage1_start_eval = False
    cfg.enable_stage1_start_eval_baseline = False
    cfg.offline_eval_skip_target_student = not args.load_target_student
    cfg.offline_eval_use_fsdp_teacher = not args.use_nofsdp_teacher
    cfg.offline_eval_skip_teacher = use_teacher_cache
    cfg.offline_eval_force_gradient_checkpointing = not args.disable_eval_gradient_checkpointing
    cfg.offline_eval_force_cfg = not args.disable_eval_force_cfg
    cfg.offline_eval_return_final_action = True
    if args.eval_rollout_grad_mode is not None:
        cfg.opd_rollout_grad_mode = args.eval_rollout_grad_mode
    cfg.skip_teacher_compile = True
    cfg.light_eval_num_batches = max(1, args.num_batches)
    cfg.light_eval_seed = args.seed
    cfg.light_eval_start_index = 0
    # Keep memory closer to training but avoid the full no-FSDP student copy for this offline pass.
    cfg.opd_aux_use_nofsdp_rollout = False

    trainer = FlowMapDistiller(cfg)
    trainer.student.eval()
    if getattr(trainer, "teacher", None) is not None:
        trainer.teacher.eval()
    if trainer.target_student is not None:
        trainer.target_student.eval()
    if getattr(trainer, "_student_nofsdp", None) is not None:
        trainer._student_nofsdp.eval()

    action_ds = getattr(trainer.config, "action_downsample_factor", 4)
    pair_specs = (
        load_eval_pairs(args.eval_pairs_json)
        if args.eval_pairs_json is not None
        else build_cli_pair_specs(args.pairs)
    )
    metrics = {}
    counts = {}
    task_metrics = {}
    task_counts = {}
    current_task = None

    def add(name, value):
        value = value.detach().float()
        metrics[name] = metrics.get(name, torch.zeros((), device=trainer.device)) + value
        counts[name] = counts.get(name, 0) + 1
        if current_task is not None:
            sums = task_metrics.setdefault(current_task, {})
            task_counts_for_name = task_counts.setdefault(current_task, {})
            sums[name] = sums.get(name, torch.zeros((), device=trainer.device)) + value
            task_counts_for_name[name] = task_counts_for_name.get(name, 0) + 1

    eval_records = build_eval_records(
        trainer,
        eval_manifest_path=args.eval_manifest,
        num_batches=args.num_batches,
        split_name=args.split_name,
    )
    for batch_idx, record in enumerate(eval_records):
        current_task = record.get("task")
        batch = record["batch"]
        batch = trainer._move_eval_batch_to_device(batch)
        base_input = trainer._prepare_base_dict(batch)
        B = batch["latents"].shape[0]
        ref_shape = batch["latents"].shape
        num_frames = ref_shape[2]
        video_frame_mask = video_frame_mask_from_batch(batch, num_frames=num_frames)
        empty_emb = trainer.empty_emb.expand(base_input["latent_dict"]["text_emb"].shape[0], -1, -1)

        for pair_idx, pair_spec in enumerate(pair_specs):
            t_value = float(pair_spec["t"])
            r_value = float(pair_spec["r"])
            gen = torch.Generator(device=trainer.device)
            if pair_spec.get("pair_seed") is None:
                gen.manual_seed(args.seed + batch_idx * 1009 + pair_idx)
            else:
                gen.manual_seed(eval_seed_for_pair(pair_spec, batch_idx))
            video_t = torch.full((B, num_frames), t_value, device=trainer.device, dtype=torch.float32)
            video_r = torch.full((B, num_frames), r_value, device=trainer.device, dtype=torch.float32)
            action_t = video_t
            action_r = video_r

            video_noise = torch.randn(batch["latents"].shape, device=trainer.device,
                                      dtype=batch["latents"].dtype, generator=gen)
            action_noise = torch.randn(batch["actions"].shape, device=trainer.device,
                                       dtype=batch["actions"].dtype, generator=gen)
            video_noisy_t = trainer.train_scheduler_latent.add_noise(
                batch["latents"], video_noise, video_t, t_dim=2)
            video_noisy_r = trainer.train_scheduler_latent.add_noise(
                batch["latents"], video_noise, video_r, t_dim=2)
            video_v_target_r = trainer.train_scheduler_latent.training_target(
                batch["latents"], video_noise, video_r)

            action_noisy_t = trainer.train_scheduler_action.add_noise(
                batch["actions"], action_noise, action_t, t_dim=2)
            action_noisy_r = trainer.train_scheduler_action.add_noise(
                batch["actions"], action_noise, action_r, t_dim=2)
            action_v_target_t = trainer.train_scheduler_action.training_target(
                batch["actions"], action_noise, action_t)

            input_dict = {
                "latent_dict": {
                    **base_input["latent_dict"],
                    "noisy_latents": video_noisy_t,
                    "timesteps": video_t,
                    "targets": trainer.train_scheduler_latent.training_target(
                        batch["latents"], video_noise, video_t),
                },
                "action_dict": {
                    **base_input["action_dict"],
                    "noisy_latents": action_noisy_t,
                    "timesteps": action_t,
                    "targets": action_v_target_t,
                },
                "chunk_size": base_input["chunk_size"],
                "window_size": base_input["window_size"],
            }
            student_input = {
                "latent_dict": input_dict["latent_dict"],
                "action_dict": {
                    "noisy_latents": action_noisy_t[:, :, ::action_ds],
                    "latent": batch["actions"][:, :, ::action_ds],
                    "timesteps": action_t[:, ::action_ds],
                    "cond_timesteps": input_dict["action_dict"]["cond_timesteps"][:, ::action_ds],
                    "text_emb": input_dict["action_dict"]["text_emb"],
                    "grid_id": _downsample_action_grid_id(
                        input_dict["action_dict"].get("grid_id"), batch["actions"], action_ds),
                    "actions_mask": (
                        input_dict["action_dict"]["actions_mask"][:, :, ::action_ds]
                        if input_dict["action_dict"].get("actions_mask") is not None else None
                    ),
                },
                "chunk_size": input_dict["chunk_size"],
                "window_size": input_dict["window_size"],
            }

            pair_name = str(pair_spec.get("pair_id") or f"t{int(t_value)}_r{int(r_value)}")
            cache_key = teacher_cache_key(
                batch_idx,
                t_value,
                r_value,
                pair_id=pair_name,
                sample_key=record.get("sample_key"),
            )

            def cache_teacher_target(teacher_step, teacher_x_r, teacher_v_r):
                if teacher_cache_path is None or rank != 0:
                    return
                entry = teacher_cache_to_write.setdefault(
                    cache_key, {"teacher_by_steps": {}}
                )
                entry["teacher_by_steps"][str(teacher_step)] = {
                    "teacher_x_r": teacher_x_r.detach().cpu(),
                    "teacher_v_r": teacher_v_r.detach().cpu(),
                }

            def load_teacher_target(teacher_step):
                targets = teacher_cache.get("targets", {})
                if cache_key not in targets:
                    raise KeyError(f"Teacher cache missing key: {cache_key}")
                cached = targets[cache_key]
                if "teacher_by_steps" in cached:
                    by_step = cached["teacher_by_steps"]
                    if str(teacher_step) not in by_step:
                        raise KeyError(
                            f"Teacher cache missing step {teacher_step} for key: {cache_key}"
                        )
                    cached = by_step[str(teacher_step)]
                elif len(teacher_steps) != 1:
                    raise ValueError(
                        "Legacy scalar teacher cache cannot satisfy multiple teacher steps; "
                        "rebuild it with --teacher-cache-only."
                    )
                return (
                    cached["teacher_x_r"].to(
                        device=trainer.device,
                        dtype=video_noisy_t.dtype,
                        non_blocking=True,
                    ),
                    cached["teacher_v_r"].to(
                        device=trainer.device,
                        dtype=video_noisy_t.dtype,
                        non_blocking=True,
                    ),
                )

            if args.teacher_cache_only:
                for teacher_step in teacher_steps:
                    teacher_x_r, teacher_v_r = trainer._teacher_integrate_to_r(
                        noisy_latents=video_noisy_t,
                        timesteps=video_t,
                        target_r=video_r,
                        input_dict=input_dict,
                        empty_emb=empty_emb,
                        cfg_scale=args.cfg_scale,
                        ref_shape=ref_shape,
                        B=B,
                        num_frames=num_frames,
                        num_steps=teacher_step,
                    )
                    cache_teacher_target(teacher_step, teacher_x_r, teacher_v_r)
                    del teacher_x_r, teacher_v_r
                continue

            # Compute each student trajectory once. All teacher budgets below
            # compare against identical x_t, r, noise, and student states.
            student_rollouts = {}
            for k_steps in args.student_steps:
                rollout_out = trainer._student_euler_integrate(
                    noisy_latents=video_noisy_t,
                    timesteps=video_t,
                    target_r=video_r,
                    base_input_dict=student_input,
                    empty_emb=empty_emb,
                    cfg_scale=args.cfg_scale,
                    ref_shape=ref_shape,
                    B=B,
                    num_frames=num_frames,
                    K_steps=k_steps,
                    action_target_r=action_r,
                    return_final_action=True,
                )
                if isinstance(rollout_out, tuple) and len(rollout_out) == 3:
                    student_x_r, student_v_r, student_action_seq = rollout_out
                else:
                    student_x_r, student_v_r = rollout_out
                    student_action_seq = None
                student_metrics = {}
                mse, l1 = masked_video_mse_l1(student_x_r - video_noisy_r, video_frame_mask)
                student_metrics["/video_gt_x_mse"] = mse
                student_metrics["/video_gt_x_l1"] = l1
                mse, _ = masked_video_mse_l1(student_v_r - video_v_target_r, video_frame_mask)
                student_metrics["/video_gt_v_mse"] = mse
                student_metrics["/video_latent_norm"] = masked_video_rms(
                    student_x_r, video_frame_mask
                )

                action_input = None
                if student_action_seq is None:
                    action_input = {
                        "latent_dict": {
                            **input_dict["latent_dict"],
                            "noisy_latents": student_x_r.detach(),
                            "timesteps": video_r,
                        },
                        "action_dict": student_input["action_dict"],
                        "chunk_size": input_dict["chunk_size"],
                        "window_size": input_dict["window_size"],
                    }
                    init_mask(trainer, action_input)
                    action_grad_context = (
                        torch.enable_grad()
                        if getattr(trainer.config, "offline_eval_force_gradient_checkpointing", False)
                        else torch.no_grad()
                    )
                    with action_grad_context:
                        _, student_action_seq = trainer.student(
                            action_input, train_mode=True,
                            r_timestep=video_r,
                            action_r_timestep=action_r[:, ::action_ds],
                        )
                action_frames = batch["actions"].shape[2] // action_ds
                student_action_v = trainer._extract_action_v(student_action_seq, action_frames)
                if getattr(trainer.config, "offline_eval_force_gradient_checkpointing", False):
                    student_action_v = student_action_v.detach()
                del student_action_seq
                action_noisy_t_ds = action_noisy_t[:, :, ::action_ds]
                action_noisy_r_ds = action_noisy_r[:, :, ::action_ds]
                action_v_target_ds = action_v_target_t[:, :, ::action_ds]
                action_sigma_t = action_t[:, None, ::action_ds, None, None] / trainer.config.num_train_timesteps
                action_sigma_r = action_r[:, None, ::action_ds, None, None] / trainer.config.num_train_timesteps
                action_pred_xr = action_noisy_t_ds + student_action_v * (
                    action_sigma_r.to(student_action_v) - action_sigma_t.to(student_action_v))
                mask = batch.get("actions_mask")
                mask_ds = mask[:, :, ::action_ds] if mask is not None else None
                mse, l1 = masked_mse_l1(action_pred_xr - action_noisy_r_ds, mask_ds)
                student_metrics["/action_gt_xr_mse"] = mse
                student_metrics["/action_gt_xr_l1"] = l1
                mse, l1 = masked_mse_l1(student_action_v - action_v_target_ds, mask_ds)
                student_metrics["/action_gt_v_mse"] = mse
                student_metrics["/action_gt_v_l1"] = l1
                if student_action_v.shape[2] > 1:
                    smooth = (student_action_v[:, :, 1:] - student_action_v[:, :, :-1]).float().abs().mean()
                    student_metrics["/action_v_smoothness_l1"] = smooth
                student_metrics["/action_v_norm"] = student_action_v.float().pow(2).mean().sqrt()
                student_rollouts[k_steps] = {
                    "x": student_x_r.detach(),
                    "v": student_v_r.detach(),
                    "metrics": student_metrics,
                }
                del (
                    action_input,
                    student_action_v,
                    action_noisy_t_ds,
                    action_noisy_r_ds,
                    action_v_target_ds,
                    action_sigma_t,
                    action_sigma_r,
                    action_pred_xr,
                )

            for teacher_step in teacher_steps:
                if use_teacher_cache:
                    teacher_x_r, teacher_v_r = load_teacher_target(teacher_step)
                else:
                    teacher_x_r, teacher_v_r = trainer._teacher_integrate_to_r(
                        noisy_latents=video_noisy_t,
                        timesteps=video_t,
                        target_r=video_r,
                        input_dict=input_dict,
                        empty_emb=empty_emb,
                        cfg_scale=args.cfg_scale,
                        ref_shape=ref_shape,
                        B=B,
                        num_frames=num_frames,
                        num_steps=teacher_step,
                    )
                    cache_teacher_target(teacher_step, teacher_x_r, teacher_v_r)

                for k_steps, student_result in student_rollouts.items():
                    prefix = f"rollout_eval/{pair_name}/s{k_steps}_t{teacher_step}"
                    mse, l1 = masked_video_mse_l1(
                        teacher_x_r - video_noisy_r, video_frame_mask
                    )
                    add(prefix + "/video_teacher_gt_x_mse", mse)
                    add(prefix + "/video_teacher_gt_x_l1", l1)
                    mse, _ = masked_video_mse_l1(
                        teacher_v_r - video_v_target_r, video_frame_mask
                    )
                    add(prefix + "/video_teacher_gt_v_mse", mse)
                    mse, l1 = masked_video_mse_l1(
                        student_result["x"] - teacher_x_r, video_frame_mask
                    )
                    add(prefix + "/video_teacher_x_mse", mse)
                    add(prefix + "/video_teacher_x_l1", l1)
                    mse, l1 = masked_video_mse_l1(
                        student_result["v"] - teacher_v_r, video_frame_mask
                    )
                    add(prefix + "/video_teacher_v_mse", mse)
                    add(prefix + "/video_teacher_v_l1", l1)
                    for name, value in student_result["metrics"].items():
                        add(prefix + name, value)
                del teacher_x_r, teacher_v_r

            if args.eval_empty_cache:
                teacher_x_r = teacher_v_r = None
                video_t = video_r = action_t = action_r = None
                video_noise = action_noise = None
                video_noisy_t = video_noisy_r = video_v_target_r = None
                action_noisy_t = action_noisy_r = action_v_target_t = None
                input_dict = student_input = student_rollouts = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if teacher_cache_path is not None and not use_teacher_cache and rank == 0:
        teacher_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": {
                    "config": args.config,
                    "teacher_model_path": args.teacher_model_path,
                    "dataset_path": args.dataset_path,
                    "num_batches": args.num_batches,
                    "seed": args.seed,
                    "cfg_scale": args.cfg_scale,
                    "pairs": pair_specs,
                    "eval_manifest": str(args.eval_manifest) if args.eval_manifest else None,
                    "eval_pairs_json": str(args.eval_pairs_json) if args.eval_pairs_json else None,
                    "split_name": args.split_name,
                    "teacher_steps": teacher_steps,
                },
                "targets": teacher_cache_to_write,
            },
            teacher_cache_path,
        )
        logger.info("Wrote teacher cache to %s", teacher_cache_path)
    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    if args.teacher_cache_only:
        return

    out = {}
    for name, total in metrics.items():
        avg = total / max(1, counts[name])
        if dist.is_initialized():
            avg = dist_mean(avg)
        out[name] = avg.item()
    if task_metrics:
        out["per_task"] = {}
        for task, task_sums in task_metrics.items():
            task_out = {}
            for name, total in task_sums.items():
                avg = total / max(1, task_counts[task][name])
                if dist.is_initialized():
                    avg = dist_mean(avg)
                task_out[name] = avg.item()
            out["per_task"][task] = task_out

    if rank == 0:
        Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_json, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        logger.info("Wrote rollout eval metrics to %s", args.result_json)
        print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
