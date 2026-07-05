#!/usr/bin/env python3
"""Offline metrics for Cosmos Policy raw LIBERO Stage-1 checkpoints.

The evaluation mirrors the action-only Stage-1 training math:
  student_x0 = action_noisy_t - sigma_r * student_v

It compares:
  - student_x0 vs official Cosmos Policy teacher x0
  - student_x0 vs dataset GT action x0
  - official Cosmos Policy teacher x0 vs dataset GT action x0
  - noisy_t vs GT, as a sanity baseline
"""

import argparse
import importlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.data._utils.collate import default_collate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "wan_va"))
sys.path.append(str(PROJECT_ROOT))

from distillation.patches import install_flash_attn_stub

install_flash_attn_stub()

import torch._dynamo

torch._dynamo.config.suppress_errors = True

from distillation.patches import SafeMultiLatentLeRobotDataset
from distillation_flowmap.cosmos_policy_adapter import (
    CosmosPolicyActionTeacher,
    compute_masked_action_stats,
    cosmos_actions_to_flowmap_x0,
    resolve_cosmos_policy_assets,
)
from distillation_flowmap.cosmos_future_aux import normalize_future_images
from distillation_flowmap.flowmap_step import _downsample_action_grid_id
from distillation_flowmap.model_flowmap import patch_model_forward, setup_flowmap_model
from modules.model import FlexAttnFunc
from modules.utils import load_transformer
from utils import FlowMatchScheduler


def parse_pair(text):
    if ":" in text:
        left, right = text.split(":", 1)
    elif "," in text:
        left, right = text.split(",", 1)
    else:
        left = right = text
    return float(left), float(right)


def move_batch(batch, device):
    out = {}
    for key, value in batch.items():
        if key.startswith("raw_"):
            out[key] = value
        elif torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        elif isinstance(value, dict):
            out[key] = {
                inner_key: inner_value.to(device, non_blocking=True)
                if torch.is_tensor(inner_value) else inner_value
                for inner_key, inner_value in value.items()
            }
        else:
            out[key] = value
    return out


def build_grid_id(batch, config, device):
    from utils import get_mesh_id

    batch_size = batch["latents"].shape[0]
    patch_f, patch_h, patch_w = config.patch_size
    latent_grid_id = get_mesh_id(
        batch["latents"].shape[-3] // patch_f,
        batch["latents"].shape[-2] // patch_h,
        batch["latents"].shape[-1] // patch_w,
        t=0,
        f_w=1,
        f_shift=0,
        action=False,
    ).to(device)
    action_grid_id = get_mesh_id(
        batch["actions"].shape[-3],
        batch["actions"].shape[-2],
        batch["actions"].shape[-1],
        t=1,
        f_w=1,
        f_shift=0,
        action=True,
    ).to(device)
    return (
        latent_grid_id[None].repeat(batch_size, 1, 1),
        action_grid_id[None].repeat(batch_size, 1, 1),
    )


def prepare_base_dict(batch, config, device):
    latent_grid_id, action_grid_id = build_grid_id(batch, config, device)
    batch_size = batch["latents"].shape[0]
    cond_timesteps_video = torch.zeros(batch_size, batch["latents"].shape[2], device=device)
    cond_timesteps_action = torch.zeros(batch_size, batch["actions"].shape[2], device=device)
    return {
        "latent_dict": {
            "latent": batch["latents"],
            "cond_timesteps": cond_timesteps_video,
            "grid_id": latent_grid_id,
            "text_emb": batch["text_emb"],
        },
        "action_dict": {
            "latent": batch["actions"],
            "cond_timesteps": cond_timesteps_action,
            "grid_id": action_grid_id,
            "text_emb": batch["text_emb"],
            "actions_mask": batch.get("actions_mask"),
        },
        "chunk_size": config.frame_chunk_size,
        "window_size": config.attn_window,
    }


def extract_action_v(config, action_pred, num_frames):
    from einops import rearrange

    seq_len = action_pred.shape[1]
    if seq_len % num_frames != 0:
        action_per_frame = int(getattr(config, "action_per_frame", 0) or 0)
        expected_len = num_frames * action_per_frame
        if action_per_frame > 0 and seq_len >= expected_len:
            action_pred = action_pred[:, :expected_len]
        else:
            raise ValueError(
                f"Cannot reshape action tokens: seq_len={seq_len}, num_frames={num_frames}, "
                f"action_per_frame={action_per_frame}"
            )
    return rearrange(action_pred, "b (f n) c -> b c f n 1", f=num_frames)


def load_stage1_model(transformer_dir, config, device, dtype):
    model = load_transformer(str(transformer_dir), torch_dtype=dtype, torch_device="cpu")
    model = setup_flowmap_model(
        model,
        gate_value=getattr(config, "gate_value", 0.0),
        deltatime_type=getattr(config, "deltatime_type", "r"),
    )
    model = patch_model_forward(model)
    weights_path = Path(transformer_dir) / "diffusion_pytorch_model.safetensors"
    delta_state = {}
    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if ".delta_embedder." in key:
                delta_state[key] = f.get_tensor(key)
    if delta_state:
        model.load_state_dict(delta_state, strict=False)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    model.requires_grad_(False)
    model._flowmap_gradient_checkpointing = False
    return model


class Accumulator:
    def __init__(self):
        self.sum = defaultdict(float)
        self.count = defaultdict(float)
        self.channel_sum = defaultdict(lambda: None)
        self.channel_count = defaultdict(float)

    def add_scalar(self, name, value, count=1.0):
        self.sum[name] += float(value)
        self.count[name] += float(count)

    def add_diff(self, prefix, diff, mask):
        diff = diff.detach().float()
        mask = mask.detach().float()
        channels = diff.shape[1]
        denom = float((mask.sum() * channels).clamp(min=1).item())
        l1 = ((diff.abs() * mask).sum() / denom).item()
        mse = ((diff.square() * mask).sum() / denom).item()
        self.add_scalar(prefix + "/l1", l1)
        self.add_scalar(prefix + "/mse", mse)
        self.add_scalar(prefix + "/rmse", math.sqrt(max(mse, 0.0)))

        channel_denom = float(mask.sum().clamp(min=1).item())
        ch_l1 = ((diff.abs() * mask).sum(dim=(0, 2, 3, 4)) / channel_denom).cpu()
        ch_mse = ((diff.square() * mask).sum(dim=(0, 2, 3, 4)) / channel_denom).cpu()
        for suffix, tensor in (("channel_l1", ch_l1), ("channel_mse", ch_mse)):
            key = prefix + "/" + suffix
            if self.channel_sum[key] is None:
                self.channel_sum[key] = tensor.double()
            else:
                self.channel_sum[key] += tensor.double()
            self.channel_count[key] += 1.0

    def result(self):
        out = {}
        for key, value in sorted(self.sum.items()):
            out[key] = value / max(self.count[key], 1.0)
        for key, value in sorted(self.channel_sum.items()):
            out[key] = (value / max(self.channel_count[key], 1.0)).tolist()
        return out


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-transformer", required=True)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument("--config-file", default="distillation_flowmap.config_libero_cosmos_policy_stage1")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pairs", nargs="+", default=["1000:1000", "750:750", "500:500", "250:250"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--student-gpu", type=int, default=0)
    parser.add_argument("--teacher-gpu", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    device = torch.device(f"cuda:{args.student_gpu}")
    torch.cuda.set_device(device)

    cfg = importlib.import_module(args.config_file).cfg
    cfg.rank = 0
    cfg.local_rank = args.student_gpu
    cfg.world_size = 1
    cfg.batch_size = args.batch_size
    cfg.load_worker = 0
    cfg.enable_wandb = False
    cfg.enable_light_eval = False
    cfg.enable_rollout_eval = False
    cfg.gradient_checkpointing = False
    cfg.cosmos_policy_use_raw_inference = True
    cfg.return_raw_observation = True
    cfg.cosmos_policy_inference_mode = os.environ.get("COSMOS_POLICY_INFERENCE_MODE", "subprocess")
    cfg.teacher_model_path = resolve_cosmos_policy_assets(
        args.teacher_model_path or cfg.teacher_model_path
    )["root"]
    if args.dataset_path:
        cfg.dataset_path = args.dataset_path
        cfg.empty_emb_path = os.environ.get("EMPTY_EMB_PATH", os.path.join(args.dataset_path, "empty_emb.pt"))

    action_scheduler = FlowMatchScheduler(
        shift=cfg.action_snr_shift,
        sigma_min=0.0,
        extra_one_step=True,
    )
    action_scheduler.set_timesteps(cfg.num_train_timesteps, training=True)
    video_scheduler = FlowMatchScheduler(
        shift=cfg.snr_shift,
        sigma_min=0.0,
        extra_one_step=True,
    )
    video_scheduler.set_timesteps(cfg.num_train_timesteps, training=True)

    print(f"[eval] loading dataset: {cfg.dataset_path}", flush=True)
    dataset = SafeMultiLatentLeRobotDataset(config=cfg)
    total = len(dataset)
    indices = [
        (args.start_index + i * args.stride) % total
        for i in range(args.num_samples)
    ]
    print(f"[eval] dataset samples={total}, eval indices={indices[:5]}... n={len(indices)}", flush=True)

    ckpt_transformer = Path(args.checkpoint_transformer)
    print(f"[eval] loading student: {ckpt_transformer}", flush=True)
    student = load_stage1_model(ckpt_transformer, cfg, device, dtype)

    teacher_device = f"cuda:{args.teacher_gpu}" if torch.cuda.device_count() > args.teacher_gpu else str(device)
    print(f"[eval] loading official Cosmos teacher worker on {teacher_device}", flush=True)
    teacher = CosmosPolicyActionTeacher(
        cfg.teacher_model_path,
        dtype=dtype,
        device=teacher_device,
        config=cfg,
    )

    pairs = [parse_pair(pair) for pair in args.pairs]
    action_ds = int(getattr(cfg, "action_downsample_factor", 4))
    acc = Accumulator()
    teacher_once = Accumulator()
    noisy_acc = Accumulator()

    for sample_i, index in enumerate(indices):
        batch = default_collate([dataset[index]])
        batch = move_batch(batch, device)
        base_input = prepare_base_dict(batch, cfg, device)
        batch_size = batch["latents"].shape[0]
        num_video_frames = batch["latents"].shape[2]

        teacher_result = teacher.predict_raw_action_result(batch, include_future=True)
        teacher_action_x0 = cosmos_actions_to_flowmap_x0(
            teacher_result["actions"],
            target_shape=tuple(base_input["action_dict"]["latent"].shape),
            q01=cfg.norm_stat["q01"],
            q99=cfg.norm_stat["q99"],
            inverse_used_action_channel_ids=cfg.inverse_used_action_channel_ids,
            device=base_input["action_dict"]["latent"].device,
            dtype=base_input["action_dict"]["latent"].dtype,
        )
        future_predictions = teacher_result.get("future_image_predictions")
        if future_predictions is not None:
            future_items = future_predictions
            if not isinstance(future_items, list):
                future_items = [future_items]
            for future_item in future_items:
                try:
                    normalized_future = normalize_future_images(
                        future_item,
                        primary_key=getattr(cfg, "raw_primary_image_key", None),
                        wrist_key=getattr(cfg, "raw_wrist_image_key", None),
                    )
                except Exception as exc:
                    print(f"[eval] warning: could not parse Cosmos future images: {exc}", flush=True)
                    continue
                teacher_once.add_scalar(
                    "cosmos_future/primary_abs_mean",
                    normalized_future.primary.abs().mean().item(),
                )
                teacher_once.add_scalar(
                    "cosmos_future/primary_num_frames",
                    float(normalized_future.primary.shape[1]),
                )
                if normalized_future.wrist is not None:
                    teacher_once.add_scalar(
                        "cosmos_future/wrist_abs_mean",
                        normalized_future.wrist.abs().mean().item(),
                    )
                    teacher_once.add_scalar(
                        "cosmos_future/wrist_num_frames",
                        float(normalized_future.wrist.shape[1]),
                    )
        actions_gt_ds = batch["actions"][:, :, ::action_ds]
        teacher_action_ds = teacher_action_x0[:, :, ::action_ds]
        mask = batch.get("actions_mask")
        mask_ds = torch.ones_like(actions_gt_ds[:, :1]).float() if mask is None else mask[:, :, ::action_ds].float()
        teacher_once.add_diff("teacher_gt", teacher_action_ds - actions_gt_ds, mask_ds)
        raw_stats = compute_masked_action_stats(teacher_action_ds, actions_gt_ds, mask_ds)
        teacher_once.add_scalar("teacher_gt/teacher_abs_mean", raw_stats["teacher_abs_mean"].item())
        teacher_once.add_scalar("teacher_gt/gt_abs_mean", raw_stats["target_abs_mean"].item())

        for pair_i, (t_value, r_value) in enumerate(pairs):
            gen = torch.Generator(device=device)
            gen.manual_seed(args.seed + sample_i * 1009 + pair_i * 9173)
            video_t = torch.full((batch_size, num_video_frames), t_value, device=device, dtype=torch.float32)
            video_r = torch.full((batch_size, num_video_frames), r_value, device=device, dtype=torch.float32)
            action_t = video_t
            action_r = video_r

            video_noise = torch.randn(
                batch["latents"].shape,
                device=device,
                dtype=batch["latents"].dtype,
                generator=gen,
            )
            video_noisy = video_scheduler.add_noise(batch["latents"], video_noise, video_t, t_dim=2)
            video_target = video_scheduler.training_target(batch["latents"], video_noise, video_t)

            action_noise = torch.randn(
                batch["actions"].shape,
                device=device,
                dtype=batch["actions"].dtype,
                generator=gen,
            )
            action_noisy = action_scheduler.add_noise(batch["actions"], action_noise, action_t, t_dim=2)
            action_target_v = action_scheduler.training_target(batch["actions"], action_noise, action_t)
            action_noisy_ds = action_noisy[:, :, ::action_ds]
            action_r_ds = action_r[:, ::action_ds]

            student_input = {
                "latent_dict": {
                    **base_input["latent_dict"],
                    "noisy_latents": video_noisy,
                    "timesteps": video_t,
                    "targets": video_target,
                },
                "action_dict": {
                    "noisy_latents": action_noisy_ds,
                    "latent": actions_gt_ds,
                    "timesteps": action_t[:, ::action_ds],
                    "cond_timesteps": base_input["action_dict"]["cond_timesteps"][:, ::action_ds],
                    "text_emb": base_input["action_dict"]["text_emb"],
                },
                "chunk_size": base_input["chunk_size"],
                "window_size": base_input["window_size"],
            }
            if base_input["action_dict"].get("grid_id") is not None:
                student_input["action_dict"]["grid_id"] = _downsample_action_grid_id(
                    base_input["action_dict"]["grid_id"],
                    batch["actions"],
                    action_ds,
                )
            if base_input["action_dict"].get("actions_mask") is not None:
                student_input["action_dict"]["actions_mask"] = (
                    base_input["action_dict"]["actions_mask"][:, :, ::action_ds]
                )

            ld = student_input["latent_dict"]
            ad = student_input["action_dict"]
            total_length = (
                ld["noisy_latents"].flatten(0, 1).shape[0] * 2
                + ad["noisy_latents"].flatten(0, 1).shape[0] * 2
            )
            padded_length = (128 - total_length % 128) % 128
            FlexAttnFunc.init_mask(
                ld["noisy_latents"].shape,
                ad["noisy_latents"].shape,
                padded_length,
                student_input["chunk_size"],
                window_size=student_input["window_size"],
                patch_size=cfg.patch_size,
                device=device,
            )

            _, student_action_seq = student(
                student_input,
                train_mode=True,
                r_timestep=video_r,
                action_r_timestep=action_r_ds,
            )
            student_action_v = extract_action_v(
                cfg,
                student_action_seq,
                student_input["action_dict"]["noisy_latents"].shape[2],
            )
            sigma_r = (action_r[:, None, ::action_ds, None, None] / cfg.num_train_timesteps).to(student_action_v)
            student_action_x0 = action_noisy_ds - sigma_r * student_action_v
            pair_name = f"t{int(t_value)}_r{int(r_value)}"
            acc.add_diff(pair_name + "/student_teacher", student_action_x0 - teacher_action_ds, mask_ds)
            acc.add_diff(pair_name + "/student_gt", student_action_x0 - actions_gt_ds, mask_ds)
            acc.add_diff(pair_name + "/student_v_gtv", student_action_v - action_target_v[:, :, ::action_ds], mask_ds)
            noisy_acc.add_diff(pair_name + "/noisy_gt", action_noisy_ds - actions_gt_ds, mask_ds)

        print(f"[eval] {sample_i + 1}/{len(indices)} index={index}", flush=True)

    result = {
        "checkpoint_transformer": str(ckpt_transformer),
        "teacher_model_path": cfg.teacher_model_path,
        "dataset_path": cfg.dataset_path,
        "num_samples": len(indices),
        "indices": indices,
        "pairs": [{"t": t, "r": r} for t, r in pairs],
        "metrics": {
            **teacher_once.result(),
            **noisy_acc.result(),
            **acc.result(),
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n")
        print(f"[eval] wrote {args.output_json}", flush=True)
    teacher.close()


if __name__ == "__main__":
    main()
