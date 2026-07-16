#!/usr/bin/env python3
"""Paper-faithful offline S4 evaluation for progressive Cosmos checkpoints.

The paper protocol is deliberately stricter than the generic Stage-2 evaluator:
it only accepts the same-prior 1000 -> 0 cache, preserves the joint video/action
state at every deployed node, and queries the official Cosmos vector field again
for every off-path diagnostic.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from distillation_flowmap.cosmos_progressive_metrics import build_paired_eval_timesteps
from distillation_flowmap.cosmos_progressive_opd import rollout_velocity_field
from distillation_flowmap.cosmos_progressive_paper_metrics import (
    DEPLOYMENT_GRID,
    PAPER_METRIC_NAMES,
    composition_pairs,
    internal_rollout_nodes,
    mean_video_mse,
    merge_task_records,
    paper_metric_template,
)
from distillation_flowmap.cosmos_progressive_protocol import manifest_digest


S4_STUDENT_STEPS = 4
S4_TEACHER_STEPS = 8
PAPER_EVALUATOR_SCHEMA = "cosmos_progressive_s4_paper_metrics_v1"
CACHE_SMOKE_SCHEMA = "cosmos_progressive_s4_cache_smoke_v1"
_VIDEO_CHANNELS = 16


def parse_args(argv=None):
    """Parse the fixed S4 evaluator command line without touching Cosmos."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-transformer", required=True)
    parser.add_argument(
        "--config",
        default="distillation_flowmap.config_libero_cosmos_policy_stage2_progressive",
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-steps", type=int, default=S4_STUDENT_STEPS)
    parser.add_argument("--teacher-steps", type=int, default=S4_TEACHER_STEPS)
    parser.add_argument("--skip-same-state-velocity", action="store_true")
    parser.add_argument("--cache-only-smoke", action="store_true")
    return parser.parse_args(argv)


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_s4_contract(*, student_steps, teacher_steps, pairs):
    """Reject settings that cannot represent the paper's deployed S4 path."""
    if int(student_steps) != S4_STUDENT_STEPS:
        raise ValueError(
            f"Paper S4 requires student_steps={S4_STUDENT_STEPS}, got {student_steps}"
        )
    if int(teacher_steps) != S4_TEACHER_STEPS:
        raise ValueError(
            f"Paper S4 requires teacher_steps={S4_TEACHER_STEPS}, got {teacher_steps}"
        )
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Paper S4 requires at least one fixed same-prior pair")
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError("Each fixed pair must be a mapping")
        if not str(pair.get("pair_id", "")).strip():
            raise ValueError("Each fixed pair needs a non-empty pair_id")
        if "pair_seed" not in pair:
            raise ValueError("Each fixed pair needs pair_seed for deterministic action noise")
        try:
            t_value = float(pair["t"])
            r_value = float(pair["r"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Each fixed pair needs numeric t and r") from exc
        if t_value != 1000.0 or r_value != 0.0:
            raise ValueError(
                "Paper S4 requires a same-prior fixed pair with t=1000 and r=0"
            )


def grid_time(value, *, batch_size, frames, device):
    """Return normalized Cosmos times for an integer-scale student grid node."""
    value = float(value)
    if not 0.0 <= value <= 1000.0:
        raise ValueError(f"Student grid time must be in [0, 1000], got {value}")
    return torch.full(
        (int(batch_size), int(frames)),
        value / 1000.0,
        dtype=torch.float32,
        device=device,
    )


def _joint_state(joint_states, node):
    try:
        state = joint_states[node]
    except KeyError as exc:
        raise ValueError(f"Missing joint video/action state at deployment node {node}") from exc
    if not isinstance(state, tuple) or len(state) != 2:
        raise ValueError(f"Joint state at deployment node {node} must be (video, action)")
    video, action = state
    if not torch.is_tensor(video) or not torch.is_tensor(action):
        raise TypeError(f"Joint state at deployment node {node} must contain tensors")
    return video, action


def compute_paper_metrics(
    *,
    y0,
    joint_states,
    student_fields,
    finite_student_map: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    teacher_continuation: Callable[..., torch.Tensor],
    teacher_field: Callable[..., torch.Tensor],
):
    """Compute Table-3 metrics from cached joint nodes and dynamic fields.

    ``finite_student_map`` is called with ``(video, action, r=..., target=...)``.
    It deliberately receives action state on every edge so g_comp cannot silently
    degrade to a video-only semigroup diagnostic.
    """
    if not torch.is_tensor(y0):
        raise TypeError("y0 must be the cached terminal video tensor")
    metrics = paper_metric_template()
    direct_maps = {}

    for r in internal_rollout_nodes():
        video, action = _joint_state(joint_states, r)
        direct_video, direct_action = finite_student_map(video, action, r=r, target=0)
        if not torch.is_tensor(direct_video) or not torch.is_tensor(direct_action):
            raise TypeError("finite_student_map must return (video, action) tensors")
        direct_maps[r] = (direct_video, direct_action)
        metrics["video_ep"][f"node_{r}"] = mean_video_mse(direct_video, y0)

        teacher_terminal = teacher_continuation(
            video,
            r=r,
            target=0,
            steps=r // 125,
        )
        if not torch.is_tensor(teacher_terminal):
            raise TypeError("teacher_continuation must return a video tensor")
        metrics["g_anchor"][f"node_{r}"] = mean_video_mse(teacher_terminal, y0)

        try:
            student_field = student_fields[r]
        except KeyError as exc:
            raise ValueError(f"Missing student field at deployment node {r}") from exc
        teacher_velocity = teacher_field(video, r=r)
        metrics["field_match"][f"node_{r}"] = mean_video_mse(
            student_field,
            teacher_velocity,
        )

    for r, s in composition_pairs():
        video, action = _joint_state(joint_states, r)
        direct_video, _ = direct_maps[r]
        middle_video, middle_action = finite_student_map(video, action, r=r, target=s)
        composed_video, _ = finite_student_map(
            middle_video,
            middle_action,
            r=s,
            target=0,
        )
        metrics["g_comp"][f"pair_{r}_{s}"] = mean_video_mse(
            direct_video,
            composed_video,
        )

    return metrics


def write_paper_outputs(output_dir, records, *, metadata):
    """Write one paper record per line and an equal-task-weighted summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(records)
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    merged = merge_task_records(records)
    summary = {
        **dict(metadata),
        "schema": PAPER_EVALUATOR_SCHEMA,
        "is_paper_metric": True,
        "num_records": merged["num_records"],
        "metrics": merged["metrics"],
        "per_task": merged["per_task"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def cache_smoke_summary(*, checkpoint_transformer, validated_video_shapes):
    """Describe a student-only cache smoke without presenting paper metrics."""
    return {
        "schema": CACHE_SMOKE_SCHEMA,
        "is_paper_metric": False,
        "checkpoint_transformer": str(checkpoint_transformer),
        "student_steps": S4_STUDENT_STEPS,
        "validated_video_shapes": {
            str(name): list(shape)
            for name, shape in sorted(validated_video_shapes.items())
        },
    }


def _require_video_latent(name, tensor, *, reference_shape=None):
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a tensor")
    if tensor.ndim != 5 or int(tensor.shape[1]) != _VIDEO_CHANNELS:
        raise ValueError(
            f"{name} must be a 16-channel [B, C, F, H, W] video latent, got {tuple(tensor.shape)}"
        )
    if reference_shape is not None and tuple(tensor.shape) != tuple(reference_shape):
        raise ValueError(
            f"{name} shape mismatch: expected {tuple(reference_shape)}, got {tuple(tensor.shape)}"
        )


def _validate_cache_video_fields(cache):
    required = ("video_x0", "video_noise", "teacher_x_r", "teacher_v_r")
    tensors = {}
    for name in required:
        if name not in cache:
            raise ValueError(f"Teacher cache is missing {name}")
        tensors[name] = cache[name]
    reference_shape = tuple(tensors["video_noise"].shape)
    for name, tensor in tensors.items():
        _require_video_latent(name, tensor, reference_shape=reference_shape)
    if "teacher_action_x0" not in cache or not torch.is_tensor(cache["teacher_action_x0"]):
        raise ValueError("Teacher cache is missing tensor teacher_action_x0")
    return tensors


def _load_empty_embedding(path, device, batch_size):
    empty = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(empty):
        raise TypeError(f"Expected tensor empty embedding at {path}, got {type(empty)!r}")
    return empty.to(device=device, dtype=torch.bfloat16).expand(batch_size, -1, -1)


def _runtime_dependencies():
    """Import expensive runtime components only after cache-only branching."""
    project_root = Path(__file__).resolve().parents[1]
    for path in (project_root, project_root / "wan_va"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    from torch.utils.data._utils.collate import default_collate

    from distillation.patches import SafeMultiLatentLeRobotDataset, install_flash_attn_stub

    install_flash_attn_stub()

    from distillation_flowmap.cosmos_policy_adapter import (
        CosmosPolicyActionTeacher,
        resolve_cosmos_policy_assets,
    )
    from distillation_flowmap.eval_cosmos_policy_stage1_metrics import (
        load_stage1_model,
        move_batch,
        prepare_base_dict,
    )
    from distillation_flowmap.eval_cosmos_progressive_stage2 import (
        _StudentRolloutHarness,
        _cache_payload,
        _student_input,
        configure_offline_eval_config,
        install_cached_teacher_anchor,
    )

    return {
        "CosmosPolicyActionTeacher": CosmosPolicyActionTeacher,
        "SafeMultiLatentLeRobotDataset": SafeMultiLatentLeRobotDataset,
        "StudentRolloutHarness": _StudentRolloutHarness,
        "cache_payload": _cache_payload,
        "configure_offline_eval_config": configure_offline_eval_config,
        "default_collate": default_collate,
        "install_cached_teacher_anchor": install_cached_teacher_anchor,
        "load_stage1_model": load_stage1_model,
        "move_batch": move_batch,
        "prepare_base_dict": prepare_base_dict,
        "resolve_cosmos_policy_assets": resolve_cosmos_policy_assets,
        "student_input": _student_input,
    }


def _configure_runtime(args, runtime, *, cache_only_smoke):
    cfg = importlib.import_module(args.config).cfg
    cfg.rank = 0
    cfg.local_rank = 0
    cfg.world_size = 1
    cfg.dataset_path = str(Path(args.dataset_path).resolve())
    cfg.empty_emb_path = os.path.join(cfg.dataset_path, "empty_emb.pt")
    cfg.cache_dataset_in_memory = False
    runtime["configure_offline_eval_config"](
        cfg,
        skip_same_state_velocity=bool(cache_only_smoke),
    )
    # Environment overrides may have disabled the training default.  The paper
    # evaluator must receive and propagate action state on every finite map.
    cfg.opd_joint_action_rollout = True
    if not cache_only_smoke:
        cfg.teacher_model_path = runtime["resolve_cosmos_policy_assets"](
            args.teacher_model_path or cfg.teacher_model_path
        )["root"]
    return cfg


def _prepare_same_prior_student_input(
    *,
    batch,
    cache,
    record,
    pair,
    cfg,
    device,
    runtime,
):
    """Build exactly one deterministic joint prior for one record/pair cache."""
    _validate_cache_video_fields(cache)
    runtime["install_cached_teacher_anchor"](
        batch,
        cache,
        device=device,
        dtype=torch.bfloat16,
    )
    batch_size = int(batch["actions"].shape[0])
    action_ds = int(getattr(cfg, "action_downsample_factor", 4))
    if action_ds <= 0:
        raise ValueError("action_downsample_factor must be positive")
    video_x0 = batch["latents"]
    video_noise = cache["video_noise"].to(device=device, dtype=video_x0.dtype)
    _require_video_latent("video_noise", video_noise, reference_shape=video_x0.shape)

    video_t, _, action_t, _ = build_paired_eval_timesteps(
        batch_size=batch_size,
        video_frames=video_x0.shape[2],
        action_frames=batch["actions"].shape[2],
        t=1000,
        r=0,
        device=device,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(
        int(pair["pair_seed"]) + int(record["index"]) * 1000003 + 7919
    )
    action_noise = torch.randn(
        batch["actions"].shape,
        dtype=batch["actions"].dtype,
        device=device,
        generator=generator,
    )
    base = {"input": runtime["prepare_base_dict"](batch, cfg, device), "config": cfg}
    student_input, _ = runtime["student_input"](
        batch,
        base,
        video_x0,
        video_noise,
        video_t,
        action_noise,
        action_t,
        action_ds,
    )
    return student_input, video_noise


def run_joint_student_edge(
    harness,
    *,
    base_input,
    empty_emb,
    video_state,
    action_state,
    source,
    target,
    cfg_scale,
    ref_shape,
    batch_size,
    video_frames,
    action_frames,
    device,
):
    """Deploy one Euler map while explicitly carrying its action state forward."""
    cfg = harness.config
    action_ds = int(getattr(cfg, "action_downsample_factor", 4))
    video_t, video_r, action_t, action_r = build_paired_eval_timesteps(
        batch_size=batch_size,
        video_frames=video_frames,
        action_frames=action_frames,
        t=source,
        r=target,
        device=device,
    )
    edge_input = {
        "latent_dict": {
            **base_input["latent_dict"],
            "noisy_latents": video_state,
            "timesteps": video_t,
        },
        "action_dict": {
            **base_input["action_dict"],
            "noisy_latents": action_state,
            "timesteps": action_t[:, ::action_ds],
        },
        "chunk_size": base_input["chunk_size"],
        "window_size": base_input["window_size"],
    }
    output = harness._student_euler_integrate(
        noisy_latents=video_state,
        timesteps=video_t,
        target_r=video_r,
        base_input_dict=edge_input,
        empty_emb=empty_emb,
        cfg_scale=float(cfg_scale),
        ref_shape=tuple(ref_shape),
        B=int(batch_size),
        num_frames=int(video_frames),
        K_steps=1,
        action_target_r=action_r,
        return_final_action=True,
        return_final_action_state=True,
    )
    next_video, field_at_target, _action_seq, next_action = output
    _require_video_latent("student video state", next_video, reference_shape=ref_shape)
    _require_video_latent("student video field", field_at_target, reference_shape=ref_shape)
    if tuple(next_action.shape) != tuple(action_state.shape):
        raise ValueError(
            "Student joint edge changed action-state shape: "
            f"expected {tuple(action_state.shape)}, got {tuple(next_action.shape)}"
        )
    return next_video, next_action, field_at_target


def rollout_student_with_joint_action_trajectory(
    harness,
    *,
    base_input,
    initial_video,
    initial_action,
    empty_emb,
    cfg_scale,
    grid=DEPLOYMENT_GRID,
):
    """Chain deployment intervals and retain a `(video, action)` node per grid time."""
    grid = tuple(int(node) for node in grid)
    if grid != DEPLOYMENT_GRID:
        raise ValueError(f"Paper S4 grid must be {DEPLOYMENT_GRID}, got {grid}")
    batch_size = int(initial_video.shape[0])
    video_frames = int(initial_video.shape[2])
    action_frames = int(initial_action.shape[2]) * int(getattr(harness.config, "action_downsample_factor", 4))
    states = {grid[0]: (initial_video, initial_action)}
    fields = {}
    video_state, action_state = initial_video, initial_action
    for source, target in zip(grid[:-1], grid[1:]):
        video_state, action_state, field = run_joint_student_edge(
            harness,
            base_input=base_input,
            empty_emb=empty_emb,
            video_state=video_state,
            action_state=action_state,
            source=source,
            target=target,
            cfg_scale=cfg_scale,
            ref_shape=tuple(initial_video.shape),
            batch_size=batch_size,
            video_frames=video_frames,
            action_frames=action_frames,
            device=initial_video.device,
        )
        states[target] = (video_state, action_state)
        fields[target] = field
    return states, fields


def finite_student_map_joint(
    harness,
    *,
    base_input,
    empty_emb,
    video,
    action,
    r,
    target,
    cfg_scale,
):
    """Evaluate one finite joint student map for endpoint/composition metrics."""
    next_video, next_action, _ = run_joint_student_edge(
        harness,
        base_input=base_input,
        empty_emb=empty_emb,
        video_state=video,
        action_state=action,
        source=r,
        target=target,
        cfg_scale=cfg_scale,
        ref_shape=tuple(video.shape),
        batch_size=int(video.shape[0]),
        video_frames=int(video.shape[2]),
        action_frames=int(action.shape[2]) * int(getattr(harness.config, "action_downsample_factor", 4)),
        device=video.device,
    )
    return next_video, next_action


def _cosmos_velocity_field(teacher, raw_batch, *, device):
    def velocity_field(video, normalized_t):
        response = teacher.predict_raw_latent_velocity(
            raw_batch,
            query_latent=video.detach().float().cpu(),
            t=normalized_t.detach().float().cpu(),
        )
        velocity = response["cosmos_latent_velocity"].to(
            device=device,
            dtype=video.dtype,
        )
        _require_video_latent("Cosmos velocity", velocity, reference_shape=video.shape)
        return velocity

    return velocity_field


def integrate_cosmos_teacher(velocity_field, video, *, r, target=0, steps):
    """Dynamically integrate the Cosmos field from an off-path student state."""
    if int(steps) <= 0:
        raise ValueError("Cosmos continuation requires positive Euler steps")
    timesteps = grid_time(
        r,
        batch_size=video.shape[0],
        frames=video.shape[2],
        device=video.device,
    )
    target_timesteps = grid_time(
        target,
        batch_size=video.shape[0],
        frames=video.shape[2],
        device=video.device,
    )
    terminal, _terminal_velocity, _path = rollout_velocity_field(
        video,
        timesteps,
        target_timesteps,
        num_steps=int(steps),
        velocity_field=velocity_field,
    )
    _require_video_latent("Cosmos continuation", terminal, reference_shape=video.shape)
    return terminal


def _load_test_manifest(path):
    manifest = _load_json(path)
    if manifest.get("split") != "test":
        raise ValueError("Paper S4 evaluator requires the final test manifest")
    records = list(manifest.get("records", []))
    if not records:
        raise ValueError("Test manifest did not provide any records")
    for record in records:
        task = record.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Test manifest record needs a non-empty string task")
        int(record["index"])
    return manifest, records


def _load_pairs(path):
    payload = _load_json(path)
    pairs = list(payload.get("pairs", []))
    validate_s4_contract(
        student_steps=S4_STUDENT_STEPS,
        teacher_steps=S4_TEACHER_STEPS,
        pairs=pairs,
    )
    return pairs


def _run_cache_only_smoke(
    *,
    args,
    cfg,
    runtime,
    dataset,
    harness,
    records,
    pairs,
    digest,
    device,
):
    """Run exactly one K=4 student rollout and never construct/query Cosmos."""
    record = records[0]
    pair = pairs[0]
    batch = runtime["move_batch"](
        runtime["default_collate"]([dataset[int(record["index"])]]) ,
        device,
    )
    cache = runtime["cache_payload"](
        args.cache_dir,
        record,
        pair,
        expected_digest=digest,
        teacher_steps=args.teacher_steps,
    )
    cache_videos = _validate_cache_video_fields(cache)
    student_input, initial_video = _prepare_same_prior_student_input(
        batch=batch,
        cache=cache,
        record=record,
        pair=pair,
        cfg=cfg,
        device=device,
        runtime=runtime,
    )
    empty_emb = _load_empty_embedding(cfg.empty_emb_path, device, int(initial_video.shape[0]))
    video_t, video_r, _action_t, action_r = build_paired_eval_timesteps(
        batch_size=initial_video.shape[0],
        video_frames=initial_video.shape[2],
        action_frames=batch["actions"].shape[2],
        t=1000,
        r=0,
        device=device,
    )
    with torch.no_grad():
        student_video, student_field, _action_seq, student_action = (
            harness._student_euler_integrate(
                noisy_latents=initial_video,
                timesteps=video_t,
                target_r=video_r,
                base_input_dict=student_input,
                empty_emb=empty_emb,
                cfg_scale=float(args.cfg_scale),
                ref_shape=tuple(initial_video.shape),
                B=int(initial_video.shape[0]),
                num_frames=int(initial_video.shape[2]),
                K_steps=S4_STUDENT_STEPS,
                action_target_r=action_r,
                return_final_action=True,
                return_final_action_state=True,
            )
        )
    _require_video_latent("cache smoke student video", student_video, reference_shape=initial_video.shape)
    _require_video_latent("cache smoke student field", student_field, reference_shape=initial_video.shape)
    if tuple(student_action.shape) != tuple(student_input["action_dict"]["noisy_latents"].shape):
        raise ValueError("Cache smoke student action state has an unexpected shape")
    validated = {
        name: list(tensor.shape)
        for name, tensor in cache_videos.items()
    }
    validated["student_video"] = list(student_video.shape)
    validated["student_field"] = list(student_field.shape)
    summary = cache_smoke_summary(
        checkpoint_transformer=Path(args.checkpoint_transformer).resolve(),
        validated_video_shapes=validated,
    )
    summary.update({
        "dataset_path": cfg.dataset_path,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_digest": digest,
        "record_index": int(record["index"]),
        "pair_id": pair["pair_id"],
        "skip_same_state_velocity": True,
    })
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _run_paper_evaluation(
    *,
    args,
    cfg,
    runtime,
    dataset,
    harness,
    teacher,
    records,
    pairs,
    digest,
    device,
):
    output_records = []
    for record_position, record in enumerate(records):
        batch = runtime["move_batch"](
            runtime["default_collate"]([dataset[int(record["index"])]]) ,
            device,
        )
        source_actions = batch["actions"].detach().clone()
        for pair in pairs:
            # One exact cache payload supplies both the video prior and y0 for
            # this row.  No cached trajectory is used for off-path diagnostics.
            batch["actions"] = source_actions.clone()
            cache = runtime["cache_payload"](
                args.cache_dir,
                record,
                pair,
                expected_digest=digest,
                teacher_steps=args.teacher_steps,
            )
            cache_videos = _validate_cache_video_fields(cache)
            student_input, initial_video = _prepare_same_prior_student_input(
                batch=batch,
                cache=cache,
                record=record,
                pair=pair,
                cfg=cfg,
                device=device,
                runtime=runtime,
            )
            y0 = cache_videos["teacher_x_r"].to(device=device, dtype=initial_video.dtype)
            _require_video_latent("cached y0", y0, reference_shape=initial_video.shape)
            initial_action = student_input["action_dict"]["noisy_latents"]
            empty_emb = _load_empty_embedding(
                cfg.empty_emb_path,
                device,
                int(initial_video.shape[0]),
            )

            with torch.no_grad():
                joint_states, student_fields = rollout_student_with_joint_action_trajectory(
                    harness,
                    base_input=student_input,
                    initial_video=initial_video,
                    initial_action=initial_action,
                    empty_emb=empty_emb,
                    cfg_scale=args.cfg_scale,
                )
                velocity_field = _cosmos_velocity_field(teacher, batch, device=device)

                def finite_student_map(video, action, *, r, target):
                    return finite_student_map_joint(
                        harness,
                        base_input=student_input,
                        empty_emb=empty_emb,
                        video=video,
                        action=action,
                        r=r,
                        target=target,
                        cfg_scale=args.cfg_scale,
                    )

                def teacher_continuation(video, *, r, target, steps):
                    return integrate_cosmos_teacher(
                        velocity_field,
                        video,
                        r=r,
                        target=target,
                        steps=steps,
                    )

                def teacher_field(video, *, r):
                    return velocity_field(
                        video,
                        grid_time(
                            r,
                            batch_size=video.shape[0],
                            frames=video.shape[2],
                            device=video.device,
                        ),
                    )

                metrics = compute_paper_metrics(
                    y0=y0,
                    joint_states=joint_states,
                    student_fields=student_fields,
                    finite_student_map=finite_student_map,
                    teacher_continuation=teacher_continuation,
                    teacher_field=teacher_field,
                )
            output_records.append({
                "task": str(record["task"]),
                "record_index": int(record["index"]),
                "pair_id": str(pair["pair_id"]),
                "metrics": metrics,
            })
            print(
                f"[paper-s4] {record_position + 1}/{len(records)} "
                f"index={record['index']} pair={pair['pair_id']}",
                flush=True,
            )

    return write_paper_outputs(
        args.output_dir,
        output_records,
        metadata={
            "checkpoint_transformer": str(Path(args.checkpoint_transformer).resolve()),
            "dataset_path": cfg.dataset_path,
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_digest": digest,
            "pairs": pairs,
            "student_steps": S4_STUDENT_STEPS,
            "teacher_steps": S4_TEACHER_STEPS,
            "paper_metric_names": list(PAPER_METRIC_NAMES),
        },
    )


def main(argv=None):
    args = parse_args(argv)
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.skip_same_state_velocity and not args.cache_only_smoke:
        raise ValueError(
            "--skip-same-state-velocity is only valid with --cache-only-smoke; "
            "paper metrics require dynamic Cosmos field queries"
        )

    manifest, records = _load_test_manifest(args.manifest)
    pairs = _load_pairs(args.pairs)
    validate_s4_contract(
        student_steps=args.student_steps,
        teacher_steps=args.teacher_steps,
        pairs=pairs,
    )
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError("No test-manifest records remain after --limit")
    digest = manifest_digest(manifest)

    runtime = _runtime_dependencies()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    cfg = _configure_runtime(args, runtime, cache_only_smoke=args.cache_only_smoke)
    dataset = runtime["SafeMultiLatentLeRobotDataset"](config=cfg)
    student = runtime["load_stage1_model"](
        Path(args.checkpoint_transformer),
        cfg,
        device,
        torch.bfloat16,
    )
    harness = runtime["StudentRolloutHarness"](student, cfg, device)

    teacher = None
    try:
        if args.cache_only_smoke:
            summary = _run_cache_only_smoke(
                args=args,
                cfg=cfg,
                runtime=runtime,
                dataset=dataset,
                harness=harness,
                records=records,
                pairs=pairs,
                digest=digest,
                device=device,
            )
        else:
            teacher = runtime["CosmosPolicyActionTeacher"](
                cfg.teacher_model_path,
                dtype=torch.bfloat16,
                device=str(device),
                config=cfg,
            )
            summary = _run_paper_evaluation(
                args=args,
                cfg=cfg,
                runtime=runtime,
                dataset=dataset,
                harness=harness,
                teacher=teacher,
                records=records,
                pairs=pairs,
                digest=digest,
                device=device,
            )
    finally:
        if teacher is not None:
            teacher.close()

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
