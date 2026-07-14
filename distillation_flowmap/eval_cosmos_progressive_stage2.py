#!/usr/bin/env python3
"""Fixed-cache offline metrics for a progressive Cosmos Stage 2 checkpoint."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "wan_va"))

from distillation.patches import SafeMultiLatentLeRobotDataset, install_flash_attn_stub
from distillation_flowmap.cosmos_policy_adapter import (
    CosmosPolicyActionTeacher,
    resolve_cosmos_policy_assets,
)
from distillation_flowmap.cosmos_progressive_metrics import (
    build_paired_eval_timesteps,
    denoised_endpoint,
    macro_average_by_task,
)
from distillation_flowmap.cosmos_progressive_opd import build_uniform_timestep_path
from distillation_flowmap.cosmos_progressive_protocol import (
    aligned_teacher_path_indices,
    cache_path,
    manifest_digest,
)
from distillation_flowmap.eval_cosmos_policy_stage1_metrics import (
    load_stage1_model,
    move_batch,
    prepare_base_dict,
)
from distillation_flowmap.flowmap_step import FlowMapStepMixin, _downsample_action_grid_id
from utils import FlowMatchScheduler


install_flash_attn_stub()


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _mse_l1(left, right):
    diff = (left.float() - right.float())
    return diff.square().mean().item(), diff.abs().mean().item()


def configure_offline_eval_config(config, *, skip_same_state_velocity):
    """Set no-grad rollout semantics without violating the OPD mode contract."""
    config.return_raw_observation = not skip_same_state_velocity
    config.cosmos_policy_use_raw_inference = not skip_same_state_velocity
    config.dataset_sample_manifest = None
    config.gradient_checkpointing = False
    config.opd_rollout_grad_mode = "endpoint"
    # The integration helper validates suffix_steps for every mode, including
    # endpoint. Evaluation is already enclosed by torch.no_grad().
    config.opd_rollout_grad_steps = max(
        1, int(getattr(config, "opd_rollout_grad_steps", 1))
    )
    config.offline_eval_force_gradient_checkpointing = False
    config.offline_eval_force_cfg = True
    return config


class _StudentRolloutHarness(FlowMapStepMixin):
    """Provides the training rollout implementation without a trainer/FSDP shell."""

    def __init__(self, student, config, device):
        self.student = student
        self.config = config
        self.device = device
        self.patch_size = tuple(config.patch_size)
        self.distill_action = True
        self.action_aware = False
        self._student_nofsdp = None
        self._nofsdp_synced = False
        self._student_blocks_compiled = False


def _load_empty_embedding(path, device, batch_size):
    empty = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(empty):
        raise TypeError(f"Expected tensor empty embedding at {path}, got {type(empty)!r}")
    return empty.to(device=device, dtype=torch.bfloat16).expand(batch_size, -1, -1)


def _student_input(batch, base_input, video_x0, video_noise, video_t, action_noise, action_t, action_ds):
    video_noisy_t = (
        (1.0 - video_t[:, None, :, None, None].to(video_x0)) * video_x0
        + video_t[:, None, :, None, None].to(video_x0) * video_noise.to(video_x0)
    )
    action_scheduler = FlowMatchScheduler(
        shift=float(getattr(base_input["config"], "action_snr_shift", 1.0)),
        sigma_min=0.0,
        extra_one_step=True,
    )
    action_scheduler.set_timesteps(int(base_input["config"].num_train_timesteps), training=True)
    action_noisy_t = action_scheduler.add_noise(
        batch["actions"], action_noise, action_t, t_dim=2
    )
    action_base = base_input["input"]["action_dict"]
    student_action = {
        "noisy_latents": action_noisy_t[:, :, ::action_ds],
        "latent": batch["actions"][:, :, ::action_ds],
        "timesteps": action_t[:, ::action_ds],
        "cond_timesteps": action_base["cond_timesteps"][:, ::action_ds],
        "text_emb": action_base["text_emb"],
        "grid_id": _downsample_action_grid_id(
            action_base.get("grid_id"), batch["actions"], action_ds
        ) if action_base.get("grid_id") is not None else None,
        "actions_mask": action_base["actions_mask"][:, :, ::action_ds]
        if action_base.get("actions_mask") is not None else None,
    }
    student_input = {
        "latent_dict": {
            **base_input["input"]["latent_dict"],
            "latent": video_x0,
            "noisy_latents": video_noisy_t,
            "timesteps": video_t,
        },
        "action_dict": student_action,
        "chunk_size": base_input["input"]["chunk_size"],
        "window_size": base_input["input"]["window_size"],
    }
    return student_input, action_noisy_t


def _cache_payload(cache_dir, record, pair, *, expected_digest, teacher_steps):
    path = cache_path(cache_dir, sample_index=record["index"], pair_id=pair["pair_id"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing fixed teacher cache: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "cosmos_progressive_teacher_cache_v1":
        raise ValueError(f"Unexpected cache schema in {path}")
    if payload.get("manifest_digest") != expected_digest:
        raise ValueError(f"Cache manifest digest mismatch in {path}")
    if payload.get("pair") != pair:
        raise ValueError(f"Cache pair mismatch in {path}")
    if int(payload.get("teacher_steps", -1)) != int(teacher_steps):
        raise ValueError(f"Cache teacher-step mismatch in {path}")
    return payload


def install_cached_teacher_anchor(batch, cache, *, device, dtype):
    """Put both Cosmos modalities in cached teacher coordinates for a rollout."""
    dataset_actions = batch["actions"].detach().clone()
    batch["latents"] = cache["video_x0"].to(device=device, dtype=dtype)
    batch["actions"] = cache["teacher_action_x0"].to(device=device, dtype=dtype)
    return dataset_actions


def _task_means(task_sums, task_counts):
    out = {}
    for task in sorted(task_sums):
        out[task] = {
            name: task_sums[task][name] / max(1, task_counts[task][name])
            for name in sorted(task_sums[task])
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-transformer", required=True)
    parser.add_argument("--config", default="distillation_flowmap.config_libero_cosmos_policy_stage2_progressive")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--student-steps", required=True, type=int)
    parser.add_argument("--teacher-steps", required=True, type=int)
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-same-state-velocity", action="store_true")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    if args.student_steps <= 0 or args.teacher_steps <= 0:
        raise ValueError("student and teacher steps must be positive")
    aligned_indices = aligned_teacher_path_indices(
        teacher_steps=args.teacher_steps,
        student_steps=args.student_steps,
    )
    manifest = _load_json(args.manifest)
    records = list(manifest.get("records", []))
    if args.limit > 0:
        records = records[:args.limit]
    if not records:
        raise ValueError("Manifest did not provide any evaluation records")
    pairs = _load_json(args.pairs).get("pairs", [])
    if not pairs:
        raise ValueError("Pair file did not provide any evaluation pairs")
    digest = manifest_digest(manifest)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    cfg = importlib.import_module(args.config).cfg
    cfg.rank = 0
    cfg.local_rank = 0
    cfg.world_size = 1
    cfg.dataset_path = str(Path(args.dataset_path).resolve())
    cfg.empty_emb_path = os.path.join(cfg.dataset_path, "empty_emb.pt")
    cfg.cache_dataset_in_memory = False
    configure_offline_eval_config(
        cfg, skip_same_state_velocity=args.skip_same_state_velocity
    )
    cfg.teacher_model_path = resolve_cosmos_policy_assets(
        args.teacher_model_path or cfg.teacher_model_path
    )["root"]

    dataset = SafeMultiLatentLeRobotDataset(config=cfg)
    student = load_stage1_model(Path(args.checkpoint_transformer), cfg, device, torch.bfloat16)
    harness = _StudentRolloutHarness(student, cfg, device)
    teacher = None
    if not args.skip_same_state_velocity:
        teacher = CosmosPolicyActionTeacher(
            cfg.teacher_model_path,
            dtype=torch.bfloat16,
            device="cuda:0",
            config=cfg,
        )

    task_sums = defaultdict(lambda: defaultdict(float))
    task_counts = defaultdict(lambda: defaultdict(int))
    action_ds = int(getattr(cfg, "action_downsample_factor", 4))
    try:
        for record_index, record in enumerate(records):
            task = str(record["task"])
            batch = move_batch(
                default_collate([dataset[int(record["index"])]]) , device
            )
            batch_size = int(batch["actions"].shape[0])
            empty_emb = _load_empty_embedding(cfg.empty_emb_path, device, batch_size)
            source_actions = batch["actions"].detach().clone()
            for pair in pairs:
                cache = _cache_payload(
                    args.cache_dir,
                    record,
                    pair,
                    expected_digest=digest,
                    teacher_steps=args.teacher_steps,
                )
                # A progressive checkpoint is deployed on a joint Cosmos
                # video-action state. Retain the dataset action only for the
                # GT diagnostic, never as the model's rollout anchor.
                batch["actions"] = source_actions.clone()
                dataset_actions = install_cached_teacher_anchor(
                    batch, cache, device=device, dtype=torch.bfloat16
                )
                video_x0 = batch["latents"]
                video_noise = cache["video_noise"].to(device=device, dtype=torch.bfloat16)
                base = {"input": prepare_base_dict(batch, cfg, device), "config": cfg}
                video_t, video_r, action_t, action_r = build_paired_eval_timesteps(
                    batch_size=batch_size,
                    video_frames=video_x0.shape[2],
                    action_frames=batch["actions"].shape[2],
                    t=pair["t"],
                    r=pair["r"],
                    device=device,
                )
                generator = torch.Generator(device=device)
                generator.manual_seed(int(pair["pair_seed"]) + int(record["index"]) * 1000003 + 7919)
                action_noise = torch.randn(
                    batch["actions"].shape,
                    dtype=batch["actions"].dtype,
                    device=device,
                    generator=generator,
                )
                student_input, action_noisy_t = _student_input(
                    batch,
                    base,
                    video_x0,
                    video_noise,
                    video_t,
                    action_noise,
                    action_t,
                    action_ds,
                )
                with torch.no_grad():
                    student_x_r, student_v_r, action_seq, action_x_r, trajectory = harness._student_euler_integrate(
                        noisy_latents=student_input["latent_dict"]["noisy_latents"],
                        timesteps=video_t,
                        target_r=video_r,
                        base_input_dict=student_input,
                        empty_emb=empty_emb,
                        cfg_scale=float(args.cfg_scale),
                        ref_shape=tuple(video_x0.shape),
                        B=batch_size,
                        num_frames=video_x0.shape[2],
                        K_steps=args.student_steps,
                        action_target_r=action_r,
                        return_final_action=True,
                        return_final_action_state=True,
                        return_trajectory=True,
                    )
                sigma_r = video_r / float(cfg.num_train_timesteps)
                student_x0 = denoised_endpoint(student_x_r, student_v_r, sigma_r)
                teacher_x_r = cache["teacher_x_r"].to(device=device, dtype=student_x_r.dtype)
                teacher_v_r = cache["teacher_v_r"].to(device=device, dtype=student_v_r.dtype)
                teacher_x0 = denoised_endpoint(teacher_x_r, teacher_v_r, sigma_r)
                metric_values = {}
                metric_values["latent/endpoint_student_teacher_x0_mse"], metric_values["latent/endpoint_student_teacher_x0_l1"] = _mse_l1(student_x0, teacher_x0)
                metric_values["rollout/final_state_drift_mse"], _ = _mse_l1(student_x_r, teacher_x_r)
                teacher_path = cache["teacher_path"].to(device=device, dtype=student_x_r.dtype)
                drift_values = []
                for student_state, teacher_index in zip(trajectory[1:], aligned_indices[1:]):
                    drift_values.append((student_state.float() - teacher_path[teacher_index].float()).square().mean())
                metric_values["rollout/aligned_drift_mse"] = torch.stack(drift_values).mean().item()
                if teacher is not None:
                    same_state_velocity = teacher.predict_raw_latent_velocity(
                        batch,
                        query_latent=student_x_r.detach().float().cpu(),
                        t=sigma_r.detach().float().cpu(),
                    )["cosmos_latent_velocity"].to(device=device, dtype=student_v_r.dtype)
                    metric_values["velocity/same_state_mse"], metric_values["velocity/same_state_l1"] = _mse_l1(student_v_r, same_state_velocity)
                action_v = harness._extract_action_v(
                    action_seq,
                    action_x_r.shape[2],
                )
                action_sigma_r = action_r[:, None, ::action_ds, None, None] / float(cfg.num_train_timesteps)
                action_x0 = action_x_r - action_sigma_r.to(action_v) * action_v
                action_teacher_x0 = batch["actions"][:, :, ::action_ds]
                action_gt = dataset_actions[:, :, ::action_ds]
                metric_values["action/endpoint_student_teacher_mse"], metric_values["action/endpoint_student_teacher_l1"] = _mse_l1(action_x0, action_teacher_x0)
                metric_values["action/endpoint_student_gt_mse"], metric_values["action/endpoint_student_gt_l1"] = _mse_l1(action_x0, action_gt)
                for name, value in metric_values.items():
                    task_sums[task][name] += float(value)
                    task_counts[task][name] += 1
                print(
                    f"[eval] {record_index + 1}/{len(records)} index={record['index']} pair={pair['pair_id']}",
                    flush=True,
                )
    finally:
        if teacher is not None:
            teacher.close()

    per_task = _task_means(task_sums, task_counts)
    result = {
        "schema": "cosmos_progressive_stage2_metrics_v1",
        "checkpoint_transformer": str(Path(args.checkpoint_transformer).resolve()),
        "dataset_path": cfg.dataset_path,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_digest": digest,
        "pairs": pairs,
        "student_steps": int(args.student_steps),
        "teacher_steps": int(args.teacher_steps),
        "num_records": len(records),
        "metrics": macro_average_by_task(per_task),
        "per_task": per_task,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
