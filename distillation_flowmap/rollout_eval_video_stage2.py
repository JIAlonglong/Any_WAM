import argparse
import gc
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video
from PIL import Image, ImageDraw


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
    crop_latent_video_to_valid_frames,
    masked_video_mse_l1,
    masked_video_rms,
    video_frame_mask_from_batch,
)
from distillation_flowmap.ablation.robotwin_mini_protocol import (
    dataset_records_for_manifest,
    eval_seed_for_pair,
    load_eval_pairs,
)
from distillation_flowmap.cosmos_future_video import (
    OFFICIAL_COSMOS_FUTURE_VIDEO_SOURCE,
    pad_frames_to_min_duration,
    predict_official_future_prediction,
)
from modules.utils import load_vae


def parse_pair(text):
    t, r = text.split(",")
    return float(t), float(r)


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
                "global_index": batch_idx,
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
    return [
        {
            "batch": default_collate([dataset[int(record["global_index"])]]),
            "global_index": int(record["global_index"]),
            "task": record["task"],
        }
        for record in records
    ]


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


def apply_first_frame_condition(noisy_latents, timesteps, target_timesteps, clean_latents):
    """Keep frame 0 clean and mark it as t=r=0, matching streaming inference."""
    if noisy_latents.shape[:3] != clean_latents.shape[:3]:
        raise ValueError(
            "noisy_latents and clean_latents must agree on batch/channel/frame shape: "
            f"{tuple(noisy_latents.shape)} vs {tuple(clean_latents.shape)}"
        )
    if timesteps.ndim != 2 or target_timesteps.ndim != 2:
        raise ValueError("timesteps and target_timesteps must be [B, F] tensors")
    if timesteps.shape[0] != noisy_latents.shape[0] or timesteps.shape[1] != noisy_latents.shape[2]:
        raise ValueError(
            "timesteps must match noisy_latents batch/frame dimensions: "
            f"{tuple(timesteps.shape)} vs {tuple(noisy_latents.shape)}"
        )
    if target_timesteps.shape != timesteps.shape:
        raise ValueError(
            "target_timesteps must have the same shape as timesteps: "
            f"{tuple(target_timesteps.shape)} vs {tuple(timesteps.shape)}"
        )
    noisy_latents[:, :, 0:1] = clean_latents[:, :, 0:1].to(
        device=noisy_latents.device,
        dtype=noisy_latents.dtype,
    )
    timesteps[:, 0] = 0.0
    target_timesteps[:, 0] = 0.0
    return noisy_latents, timesteps, target_timesteps


def should_save_student_latent_video(
    is_cosmos_policy_teacher,
    distill_video,
    allow_cosmos_latent_diagnostic=False,
):
    if is_cosmos_policy_teacher and not distill_video and not allow_cosmos_latent_diagnostic:
        return False
    return True


def set_eval_mode_for_optional_students(trainer):
    trainer.student.eval()
    if getattr(trainer, "teacher", None) is not None:
        trainer.teacher.eval()
    if getattr(trainer, "target_student", None) is not None:
        trainer.target_student.eval()
    if getattr(trainer, "_student_nofsdp", None) is not None:
        trainer._student_nofsdp.eval()


def decode_latents_to_np(vae, video_processor, latents):
    latents = latents.detach().to(next(vae.parameters()).device, dtype=vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std_inv = (
        1.0
        / torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents / latents_std_inv + latents_mean
    video = vae.decode(latents, return_dict=False)[0]
    return video_processor.postprocess_video(video, output_type="np")[0]


def frame_to_uint8(frame):
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.5 else arr, 0, 255).astype(np.uint8)
    return arr


def save_contact_sheet(named_videos, path):
    if not named_videos:
        return
    first_video = next(iter(named_videos.values()))
    frame_count = len(first_video)
    frame_ids = sorted(set([0, frame_count // 2, frame_count - 1]))
    thumb_w = 192
    thumbs = []
    for name, video in named_videos.items():
        row = []
        for frame_id in frame_ids:
            img = Image.fromarray(frame_to_uint8(video[frame_id])).convert("RGB")
            scale = thumb_w / img.width
            img = img.resize((thumb_w, max(1, int(img.height * scale))))
            row.append(img)
        thumbs.append((name, row))

    label_h = 24
    gap = 6
    row_h = label_h + max(img.height for _, row in thumbs for img in row)
    sheet_w = len(frame_ids) * thumb_w + (len(frame_ids) - 1) * gap
    sheet_h = len(thumbs) * row_h + (len(thumbs) - 1) * gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    y = 0
    for name, row in thumbs:
        draw.text((0, y + 4), name, fill=(0, 0, 0))
        x = 0
        for img in row:
            sheet.paste(img, (x, y + label_h))
            x += thumb_w + gap
        y += row_h + gap
    sheet.save(path)


def future_prediction_to_frame_np(prediction):
    prediction = prediction or {}
    images = []
    for key in ("future_wrist_image", "future_image"):
        image = prediction.get(key)
        if image is not None:
            images.append(frame_to_uint8(image))
    if not images:
        raise RuntimeError("Official Cosmos future prediction did not contain future image fields.")

    base_h, base_w = images[0].shape[:2]
    resized = []
    for image in images:
        if image.shape[:2] != (base_h, base_w):
            image = np.asarray(Image.fromarray(image).resize((base_w, base_h)))
        resized.append(image.astype(np.uint8))
    return np.hstack(resized).astype(np.uint8)


def future_predictions_to_video_np(predictions, num_frames=1):
    if isinstance(predictions, dict) or predictions is None:
        predictions = [predictions]
    frames = [future_prediction_to_frame_np(prediction) for prediction in predictions]
    num_frames = max(1, int(num_frames))
    if len(frames) == 1:
        return np.stack([frames[0].copy() for _ in range(num_frames)], axis=0)
    frame_ids = np.linspace(0, len(frames) - 1, num_frames)
    frame_ids = np.rint(frame_ids).astype(int)
    return np.stack([frames[idx].copy() for idx in frame_ids], axis=0)


def future_prediction_to_video_np(prediction, num_frames=1):
    return future_predictions_to_video_np([prediction], num_frames=num_frames)


def resolve_eval_dataset_sample(dataset, sample_index):
    sample_index = int(sample_index) % len(dataset)
    if hasattr(dataset, "_datasets") and hasattr(dataset, "item_id_to_dataset_id"):
        dataset_id = dataset.item_id_to_dataset_id[sample_index]
        local_index = sample_index - dataset.acc_dset_num[dataset_id]
        return dataset._datasets[dataset_id], local_index
    return dataset, sample_index


def select_future_prediction_frame_indices(cur_meta, num_predictions):
    start = int(cur_meta["start_frame"])
    end = int(cur_meta["end_frame"])
    if end <= start:
        return [start]
    count = max(1, min(int(num_predictions), end - start))
    frame_ids = np.rint(np.linspace(start, end - 1, count)).astype(int).tolist()
    out = []
    for frame_id in frame_ids:
        frame_id = int(frame_id)
        if not out or frame_id != out[-1]:
            out.append(frame_id)
    return out


def predict_official_future_prediction_sequence(teacher, dataset, sample_index, num_predictions):
    dataset, local_index = resolve_eval_dataset_sample(dataset, sample_index)
    if not hasattr(dataset, "new_metas") or not hasattr(dataset, "_get_raw_policy_observation"):
        raise RuntimeError(
            "Official Cosmos future video sequence export requires a LatentLeRobotDataset "
            "with raw observation access."
        )
    cur_meta = dataset.new_metas[int(local_index) % len(dataset.new_metas)]
    predictions = []
    for frame_id in select_future_prediction_frame_indices(cur_meta, num_predictions):
        raw_batch = dataset._get_raw_policy_observation(cur_meta, frame_id)
        predictions.append(predict_official_future_prediction(teacher, raw_batch))
    return predictions


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
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-max-pairs", type=int, default=1)
    parser.add_argument("--video-sample-index", type=int, default=0)
    parser.add_argument("--video-decode-device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device used for WanVAE video decoding. CPU avoids holding VAE weights during GPU rollout.",
    )
    parser.add_argument(
        "--cosmos-future-max-predictions",
        type=int,
        default=8,
        help="Maximum official Cosmos future_image_predictions calls per saved offline video.",
    )
    parser.add_argument(
        "--condition-first-frame",
        action="store_true",
        help="Keep frame 0 clean with t=r=0, matching WanVA/Cosmos streaming inference.",
    )
    parser.add_argument(
        "--save-cosmos-latent-diagnostic-videos",
        action="store_true",
        help=(
            "Also save student_s* WanVA latent diagnostic videos for Cosmos action-only "
            "configs. These are not true Cosmos policy environment rollouts."
        ),
    )
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
    args = parser.parse_args()
    teacher_steps = normalize_rollout_steps(args.teacher_steps)

    init_logger()
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)

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
    is_cosmos_policy_teacher_cfg = str(getattr(cfg, "teacher_backend", "wanva")).lower() in (
        "cosmos",
        "cosmos_policy",
        "cosmos-policy",
    )
    if is_cosmos_policy_teacher_cfg and args.video_dir is not None:
        cfg.cosmos_policy_use_raw_inference = True
        cfg.return_raw_observation = True
        cfg.cache_dataset_in_memory = False

    trainer = FlowMapDistiller(cfg)
    set_eval_mode_for_optional_students(trainer)

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
    video_dir = Path(args.video_dir) if args.video_dir else None
    vae = None
    video_processor = None
    vae_model_root = None
    saved_video_pairs = 0
    is_cosmos_policy_teacher = bool(getattr(trainer, "is_cosmos_policy_teacher", False))
    save_student_latent_video = should_save_student_latent_video(
        is_cosmos_policy_teacher=is_cosmos_policy_teacher,
        distill_video=bool(getattr(cfg, "distill_video", True)),
        allow_cosmos_latent_diagnostic=args.save_cosmos_latent_diagnostic_videos,
    )
    if rank == 0 and video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
        if not save_student_latent_video:
            logger.warning(
                "Skipping student_s* WanVA latent videos for Cosmos action-only config. "
                "Use evaluation/libero/run_eval_new.sh for true student policy env videos, "
                "or pass --save-cosmos-latent-diagnostic-videos for the latent diagnostic."
            )
        vae_model_root = (
            getattr(cfg, "student_base_model_path", args.teacher_model_path)
            if is_cosmos_policy_teacher
            else args.teacher_model_path
        )

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

            if args.condition_first_frame:
                video_noisy_t, video_t, video_r = apply_first_frame_condition(
                    video_noisy_t, video_t, video_r, batch["latents"]
                )
                video_noisy_r, _, _ = apply_first_frame_condition(
                    video_noisy_r, video_r, video_r, batch["latents"]
                )
                zero_action = torch.zeros_like(batch["actions"])
                action_noisy_t, action_t, action_r = apply_first_frame_condition(
                    action_noisy_t, action_t, action_r, zero_action
                )
                action_noisy_r, _, _ = apply_first_frame_condition(
                    action_noisy_r, action_r, action_r, zero_action
                )

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
            videos_to_save = {}
            save_official_future_video = False
            should_save_videos = (
                rank == 0
                and video_dir is not None
                and batch_idx == 0
                and saved_video_pairs < args.video_max_pairs
            )
            if should_save_videos:
                sample_idx = min(max(args.video_sample_index, 0), B - 1)
                videos_to_save["gt_r"] = crop_latent_video_to_valid_frames(
                    video_noisy_r[sample_idx:sample_idx + 1],
                    video_frame_mask,
                    sample_idx=sample_idx,
                ).detach().cpu()
                save_official_future_video = is_cosmos_policy_teacher

            # Roll out each student budget once. Every teacher budget below will
            # compare against these same states, noise realization, and condition.
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
                if should_save_videos:
                    sample_idx = min(max(args.video_sample_index, 0), B - 1)
                    if save_student_latent_video:
                        videos_to_save[f"student_s{k_steps}"] = crop_latent_video_to_valid_frames(
                            student_x_r[sample_idx:sample_idx + 1],
                            video_frame_mask,
                            sample_idx=sample_idx,
                        ).detach().cpu()
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

            comparison_teacher_steps = (
                teacher_steps if not is_cosmos_policy_teacher else [teacher_steps[-1]]
            )
            for teacher_step in comparison_teacher_steps:
                teacher_x_r = teacher_v_r = None
                if not is_cosmos_policy_teacher:
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
                    if should_save_videos:
                        sample_idx = min(max(args.video_sample_index, 0), B - 1)
                        videos_to_save[f"teacher_t{teacher_step}"] = (
                            crop_latent_video_to_valid_frames(
                                teacher_x_r[sample_idx:sample_idx + 1],
                                video_frame_mask,
                                sample_idx=sample_idx,
                            ).detach().cpu()
                        )

                for k_steps, student_result in student_rollouts.items():
                    prefix = f"rollout_eval/{pair_name}/s{k_steps}_t{teacher_step}"
                    if teacher_x_r is not None and teacher_v_r is not None:
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

            if rank == 0 and video_dir is not None and batch_idx == 0 and saved_video_pairs < args.video_max_pairs:
                decoded = {}
                for name, latent_cpu in videos_to_save.items():
                    if vae is None:
                        if vae_model_root is None:
                            raise RuntimeError("VAE model root is unavailable for video decoding.")
                        decode_device = torch.device("cuda", local_rank) if args.video_decode_device == "cuda" else torch.device("cpu")
                        if decode_device.type == "cuda":
                            torch.cuda.empty_cache()
                        vae = load_vae(
                            os.path.join(vae_model_root, "vae"),
                            torch_dtype=torch.float16 if decode_device.type == "cuda" else torch.float32,
                            torch_device=decode_device,
                        )
                        vae.eval()
                        video_processor = VideoProcessor(vae_scale_factor=1)
                    video_np = decode_latents_to_np(vae, video_processor, latent_cpu)
                    decoded[name] = video_np
                    out_path = video_dir / f"{pair_name}_{name}.mp4"
                    video_np = np.stack(
                        pad_frames_to_min_duration(list(video_np), fps=args.video_fps),
                        axis=0,
                    )
                    export_to_video(video_np, str(out_path), fps=args.video_fps)
                    logger.info("Saved rollout video: %s", out_path)
                if save_official_future_video:
                    reference_frame_count = max((len(video) for video in decoded.values()), default=args.video_fps)
                    dataset_index = int(record.get("global_index", batch_idx)) % len(trainer.train_loader.dataset)
                    prediction_count = min(
                        max(1, int(args.cosmos_future_max_predictions)),
                        max(1, int(reference_frame_count)),
                    )
                    predictions = predict_official_future_prediction_sequence(
                        trainer.teacher,
                        trainer.train_loader.dataset,
                        sample_index=dataset_index,
                        num_predictions=prediction_count,
                    )
                    video_np = future_predictions_to_video_np(
                        predictions,
                        num_frames=reference_frame_count,
                    )
                    video_np = np.stack(
                        pad_frames_to_min_duration(list(video_np), fps=args.video_fps),
                        axis=0,
                    )
                    name = "cosmos_teacher_official_future"
                    decoded[name] = video_np
                    out_path = video_dir / f"{pair_name}_{name}.mp4"
                    export_to_video(video_np, str(out_path), fps=args.video_fps)
                    logger.info(
                        "Saved rollout video from %s: %s",
                        OFFICIAL_COSMOS_FUTURE_VIDEO_SOURCE,
                        out_path,
                    )
                sheet_path = video_dir / f"{pair_name}_contact_sheet.png"
                save_contact_sheet(decoded, sheet_path)
                logger.info("Saved rollout contact sheet: %s", sheet_path)
                saved_video_pairs += 1

            if args.eval_empty_cache:
                teacher_x_r = teacher_v_r = None
                video_t = video_r = action_t = action_r = None
                video_noise = action_noise = None
                video_frame_mask = None
                video_noisy_t = video_noisy_r = video_v_target_r = None
                action_noisy_t = action_noisy_r = action_v_target_t = None
                input_dict = student_input = videos_to_save = student_rollouts = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
