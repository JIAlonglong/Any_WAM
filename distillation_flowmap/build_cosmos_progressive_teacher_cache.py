#!/usr/bin/env python3
"""Cache fixed Cosmos endpoint trajectories and action targets for Stage 2 eval."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "wan_va"))

from distillation.patches import SafeMultiLatentLeRobotDataset, install_flash_attn_stub
from distillation_flowmap.cosmos_policy_adapter import (
    CosmosPolicyActionTeacher,
    cosmos_actions_to_flowmap_x0,
    resolve_cosmos_policy_assets,
)
from distillation_flowmap.cosmos_progressive_protocol import (
    cache_path,
    cosmos_latent_shape,
    manifest_digest,
)
from distillation_flowmap.cosmos_progressive_opd import build_uniform_timestep_path


install_flash_attn_stub()


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _records(manifest):
    records = manifest.get("records", [])
    if not records:
        raise ValueError("Manifest must contain non-empty records for cache construction")
    return records


def _cache_is_valid(path, *, expected_digest, pair, teacher_steps):
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return (
        payload.get("schema") == "cosmos_progressive_teacher_cache_v1"
        and payload.get("manifest_digest") == expected_digest
        and payload.get("pair") == pair
        and int(payload.get("teacher_steps", -1)) == int(teacher_steps)
    )


@torch.no_grad()
def _teacher_rollout(teacher, batch, initial_state, timesteps, target_timesteps, teacher_steps):
    path = build_uniform_timestep_path(timesteps, target_timesteps, teacher_steps)
    current = initial_state.float().cpu()
    states = [current]
    for step_index in range(int(teacher_steps)):
        t_i = path[step_index].float().cpu()
        r_i = path[step_index + 1].float().cpu()
        velocity = teacher.predict_raw_latent_velocity(
            batch,
            query_latent=current,
            t=t_i,
        )["cosmos_latent_velocity"].float().cpu()
        current = current + velocity * (r_i - t_i)[:, None, :, None, None]
        states.append(current)
    terminal_velocity = teacher.predict_raw_latent_velocity(
        batch,
        query_latent=current,
        t=path[-1].float().cpu(),
    )["cosmos_latent_velocity"].float().cpu()
    return current, terminal_velocity, torch.stack(states, dim=0)


def _save_cache(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="distillation_flowmap.config_libero_cosmos_policy_stage2_progressive")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--teacher-steps", required=True, type=int)
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.teacher_steps <= 0:
        raise ValueError("--teacher-steps must be positive")
    manifest = _load_json(args.manifest)
    pairs = _load_json(args.pairs).get("pairs", [])
    if not pairs:
        raise ValueError("Pair file does not contain pairs")
    digest = manifest_digest(manifest)

    cfg = importlib.import_module(args.config).cfg
    cfg.rank = 0
    cfg.local_rank = 0
    cfg.world_size = 1
    cfg.dataset_path = str(Path(args.dataset_path).resolve())
    cfg.empty_emb_path = os.path.join(cfg.dataset_path, "empty_emb.pt")
    cfg.cache_dataset_in_memory = False
    cfg.return_raw_observation = True
    cfg.cosmos_policy_use_raw_inference = True
    cfg.dataset_sample_manifest = None
    cfg.teacher_model_path = resolve_cosmos_policy_assets(
        args.teacher_model_path or cfg.teacher_model_path
    )["root"]

    dataset = SafeMultiLatentLeRobotDataset(config=cfg)
    teacher = CosmosPolicyActionTeacher(
        cfg.teacher_model_path,
        dtype=torch.bfloat16,
        device="cuda:0",
        config=cfg,
    )
    cache_dir = Path(args.cache_dir)
    try:
        total = len(_records(manifest)) * len(pairs)
        completed = 0
        for record in _records(manifest):
            index = int(record["index"])
            batch = default_collate([dataset[index]])
            for pair in pairs:
                path = cache_path(cache_dir, sample_index=index, pair_id=pair["pair_id"])
                if not args.overwrite and _cache_is_valid(
                    path,
                    expected_digest=digest,
                    pair=pair,
                    teacher_steps=args.teacher_steps,
                ):
                    completed += 1
                    continue
                latent_shape = cosmos_latent_shape(cfg, batch_size=1)
                num_frames = latent_shape[2]
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(pair["pair_seed"]) + index * 1000003)
                noise = torch.randn(
                    latent_shape,
                    generator=generator,
                    dtype=torch.float32,
                )
                t = torch.full((1, num_frames), float(pair["t"]) / 1000.0)
                r = torch.full((1, num_frames), float(pair["r"]) / 1000.0)
                target = teacher.predict_raw_latent_target(
                    batch,
                    noise=noise,
                    t=t,
                    r=r,
                    epsilon=float(getattr(cfg, "cosmos_latent_epsilon", 0.001)),
                    include_cdiff=False,
                )
                x0 = target["cosmos_latent_x0"].float().cpu()
                noisy_t = (1.0 - t[:, None, :, None, None]) * x0 + t[:, None, :, None, None] * noise
                teacher_x_r, teacher_v_r, teacher_path = _teacher_rollout(
                    teacher,
                    batch,
                    noisy_t,
                    t,
                    r,
                    args.teacher_steps,
                )
                action_x0 = cosmos_actions_to_flowmap_x0(
                    target["actions"],
                    target_shape=tuple(batch["actions"].shape),
                    q01=cfg.norm_stat["q01"],
                    q99=cfg.norm_stat["q99"],
                    inverse_used_action_channel_ids=cfg.inverse_used_action_channel_ids,
                    device="cpu",
                    dtype=torch.float32,
                ).cpu()
                _save_cache(path, {
                    "schema": "cosmos_progressive_teacher_cache_v1",
                    "manifest_digest": digest,
                    "record": dict(record),
                    "pair": dict(pair),
                    "teacher_steps": int(args.teacher_steps),
                    "video_x0": x0,
                    "video_noise": noise,
                    "teacher_x_r": teacher_x_r,
                    "teacher_v_r": teacher_v_r,
                    "teacher_path": teacher_path,
                    "teacher_action_x0": action_x0,
                })
                completed += 1
                print(f"[cache] {completed}/{total} index={index} pair={pair['pair_id']}", flush=True)
    finally:
        teacher.close()


if __name__ == "__main__":
    main()
