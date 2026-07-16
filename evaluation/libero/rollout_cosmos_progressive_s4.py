#!/usr/bin/env python3
"""Run the dedicated 16-channel Cosmos Progressive S4 policy on LIBERO.

This entrypoint intentionally composes the existing validated FlowMap rollout
pieces (`load_stage1_model`, `prepare_base_dict`, `_StudentRolloutHarness`, and
`FlowMapStepMixin._student_euler_integrate`) instead of borrowing the generic
WanVA 48-channel websocket service.  Cosmos remains an anchor-only raw worker;
the LIBERO client keeps all environment ownership.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from evaluation.libero.cosmos_progressive_s4_client import CosmosProgressiveS4Client
from evaluation.libero.cosmos_progressive_s4_server import (
    COSMOS_VIDEO_SHAPE,
    PROMPT_EMBEDDING_SHAPE,
    ActionDecodingTemplate,
    CosmosProgressiveS4Engine,
    CosmosProgressiveS4Service,
    CosmosRuntimePrerequisiteError,
    PromptEmbeddingTable,
    S4_ACTION_DIM,
    S4_ACTION_STEPS,
)


S4_STEPS = 4


def require_live_s4_prerequisites(*, device: str, checkpoint_transformer: str | Path) -> None:
    """Reject unsupported live execution early, without spawning a Cosmos worker.

    The current project host can execute CPU pure-service tests, but it cannot
    substitute for the official Cosmos CUDA environment (which requires its own
    compatible driver/CUDA extra).  This function performs only local checks;
    it neither installs dependencies nor starts the raw worker.
    """
    if str(device).lower().startswith("cpu"):
        raise CosmosRuntimePrerequisiteError(
            "CPU-only execution cannot run the live Cosmos Progressive S4 rollout. "
            "Run the pure pytest suite on CPU, then use the official Cosmos CUDA environment "
            "with a compatible driver/CUDA extra for a real anchor worker."
        )
    checkpoint = Path(checkpoint_transformer)
    if not checkpoint.is_dir():
        raise CosmosRuntimePrerequisiteError(
            f"S4 checkpoint directory does not exist: {checkpoint}. "
            "Pass the progressive S4 transformer checkpoint, not a generic policy checkpoint."
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - training env always has torch
        raise CosmosRuntimePrerequisiteError("Live S4 rollout requires PyTorch with CUDA support.") from exc
    if not torch.cuda.is_available():
        raise CosmosRuntimePrerequisiteError(
            "CUDA is unavailable to this process. The official Cosmos raw-anchor worker requires "
            "a compatible CUDA driver and Cosmos CUDA extra; do not fall back to a generic server."
        )
    try:
        requested_index = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
        if requested_index >= torch.cuda.device_count():
            raise CosmosRuntimePrerequisiteError(
                f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
    except ValueError as exc:
        raise CosmosRuntimePrerequisiteError(f"Invalid CUDA device specifier: {device!r}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-transformer", required=True)
    parser.add_argument(
        "--config",
        default="distillation_flowmap.config_libero_cosmos_policy_stage2_progressive",
    )
    parser.add_argument(
        "--prompt-table",
        required=True,
        help="Task-keyed training prompt embedding table (.npz/.pt), each [1,512,4096].",
    )
    parser.add_argument("--empty-embedding", default=None)
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--anchor-record-dir", default="outputs/cosmos_progressive_s4/anchors")
    parser.add_argument("--output-dir", default="outputs/cosmos_progressive_s4")
    parser.add_argument("--libero-benchmark", default="libero_10")
    parser.add_argument("--task-range", type=int, nargs=2, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--max-env-steps", type=int, default=800)
    parser.add_argument("--env-seed", type=int, default=None)
    parser.add_argument("--initial-states-json", default=None)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--warmup-gripper", type=float, default=0.0)
    parser.add_argument("--skip-first-action", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--serve-stdio",
        action="store_true",
        help="Serve reset/infer JSON lines only; no LIBERO environment is constructed here.",
    )
    return parser.parse_args(argv)


def _runtime_dependencies() -> dict[str, Any]:
    """Lazy-load heavy components only after an explicit live-run request."""
    project_root = Path(__file__).resolve().parents[2]
    for path in (project_root, project_root / "wan_va"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    try:
        from distillation.patches import install_flash_attn_stub

        install_flash_attn_stub()
        from distillation_flowmap.cosmos_policy_adapter import (
            CosmosPolicyActionTeacher,
            cosmos_actions_to_flowmap_x0,
            resolve_cosmos_policy_assets,
        )
        from distillation_flowmap.cosmos_progressive_metrics import build_paired_eval_timesteps
        from distillation_flowmap.eval_cosmos_policy_stage1_metrics import (
            load_stage1_model,
            prepare_base_dict,
        )
        from distillation_flowmap.eval_cosmos_progressive_stage2 import (
            _StudentRolloutHarness,
            _student_input,
        )
    except Exception as exc:
        raise CosmosRuntimePrerequisiteError(
            "Could not import the existing 16-channel FlowMap/Cosmos runtime. "
            "Use the Flash-WAM environment for the student and a compatible official Cosmos worker env; "
            "do not install Cosmos extras into this CPU-only test process."
        ) from exc
    return {
        "CosmosPolicyActionTeacher": CosmosPolicyActionTeacher,
        "StudentRolloutHarness": _StudentRolloutHarness,
        "build_paired_eval_timesteps": build_paired_eval_timesteps,
        "cosmos_actions_to_flowmap_x0": cosmos_actions_to_flowmap_x0,
        "load_stage1_model": load_stage1_model,
        "prepare_base_dict": prepare_base_dict,
        "resolve_cosmos_policy_assets": resolve_cosmos_policy_assets,
        "student_input": _student_input,
    }


def _configure_live_config(args: argparse.Namespace, dependencies: Mapping[str, Any]) -> Any:
    config = importlib.import_module(args.config).cfg
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1
    config.return_raw_observation = True
    config.cosmos_policy_use_raw_inference = True
    config.opd_joint_action_rollout = True
    config.gradient_checkpointing = False
    config.offline_eval_force_gradient_checkpointing = False
    config.offline_eval_force_cfg = True
    config.opd_rollout_grad_mode = "endpoint"
    config.opd_rollout_grad_steps = max(1, int(getattr(config, "opd_rollout_grad_steps", 1)))
    if args.empty_embedding is not None:
        config.empty_emb_path = str(Path(args.empty_embedding).resolve())
    if not Path(config.empty_emb_path).is_file():
        raise CosmosRuntimePrerequisiteError(
            f"Missing empty embedding at {config.empty_emb_path}. "
            "Pass --empty-embedding from the progressive training data."
        )
    config.teacher_model_path = dependencies["resolve_cosmos_policy_assets"](
        args.teacher_model_path or config.teacher_model_path
    )["root"]
    return config


def _load_empty_embedding(path: str | Path, *, torch_module: Any, device: Any) -> Any:
    embedding = torch_module.load(path, map_location="cpu", weights_only=False)
    if not torch_module.is_tensor(embedding) or tuple(embedding.shape) != PROMPT_EMBEDDING_SHAPE:
        raise CosmosRuntimePrerequisiteError(
            f"Empty embedding must have shape {PROMPT_EMBEDDING_SHAPE}, got {getattr(embedding, 'shape', None)}"
        )
    return embedding.to(device=device, dtype=torch_module.bfloat16)


class FlowMapActionAnchorEncoder:
    """Map the raw Cosmos 16x7 action anchor into the joint FlowMap action x0."""

    def __init__(self, *, config: Any, device: Any, torch_module: Any, converter: Any) -> None:
        self.config = config
        self.device = device
        self.dtype = torch_module.bfloat16
        self.converter = converter
        self.action_channels = len(config.inverse_used_action_channel_ids)
        # The joint student downsamples this grid by action_downsample_factor.
        # With four actions per frame, 16 full-grid frames become four decoded
        # frames (4 * 4 = 16 actions), while the video anchor remains 9 frames.
        self.action_frames = S4_ACTION_STEPS
        self.actions_per_frame = int(getattr(config, "action_per_frame", 4))
        if self.actions_per_frame <= 0:
            raise CosmosRuntimePrerequisiteError("config.action_per_frame must be positive for joint S4 rollout")

    @property
    def target_shape(self) -> tuple[int, int, int, int, int]:
        return (1, self.action_channels, self.action_frames, self.actions_per_frame, 1)

    def __call__(self, raw_actions: Any) -> Any:
        return self.converter(
            raw_actions,
            target_shape=self.target_shape,
            q01=self.config.norm_stat["q01"],
            q99=self.config.norm_stat["q99"],
            inverse_used_action_channel_ids=self.config.inverse_used_action_channel_ids,
            device=self.device,
            dtype=self.dtype,
        )


class FlowMapJointS4Runner:
    """Validated joint 16-channel S4 deployment through ``_student_euler_integrate``."""

    def __init__(
        self,
        *,
        harness: Any,
        config: Any,
        device: Any,
        torch_module: Any,
        prepare_base_dict: Any,
        student_input: Any,
        build_paired_eval_timesteps: Any,
        empty_embedding: Any,
        cfg_scale: float,
    ) -> None:
        self.harness = harness
        self.config = config
        self.device = device
        self.torch = torch_module
        self.prepare_base_dict = prepare_base_dict
        self.student_input = student_input
        self.build_paired_eval_timesteps = build_paired_eval_timesteps
        self.empty_embedding = empty_embedding
        self.cfg_scale = float(cfg_scale)

    def __call__(
        self,
        video_x0: Any,
        action_x0: Any,
        text_emb: Any,
        *,
        noise: Any,
        t1000: Any,
        t0: Any,
        k_steps: int,
    ) -> Any:
        if int(k_steps) != S4_STEPS:
            raise ValueError(f"Progressive S4 deployment requires k_steps={S4_STEPS}, got {k_steps}")
        torch = self.torch
        video_x0 = torch.as_tensor(video_x0, device=self.device, dtype=torch.bfloat16)
        video_noise = torch.as_tensor(noise, device=self.device, dtype=video_x0.dtype)
        if tuple(video_x0.shape) != COSMOS_VIDEO_SHAPE or tuple(video_noise.shape) != COSMOS_VIDEO_SHAPE:
            raise ValueError(
                f"Joint S4 requires 16-channel video/action anchor shape {COSMOS_VIDEO_SHAPE}; "
                f"got video={tuple(video_x0.shape)}, noise={tuple(video_noise.shape)}"
            )
        action_x0 = torch.as_tensor(action_x0, device=self.device, dtype=video_x0.dtype)
        if action_x0.ndim != 5 or action_x0.shape[0] != 1 or action_x0.shape[-1] != 1:
            raise ValueError(f"FlowMap action x0 must be [1,C,F,N,1], got {tuple(action_x0.shape)}")
        text_emb = torch.as_tensor(text_emb, device=self.device, dtype=video_x0.dtype)
        if tuple(text_emb.shape) != PROMPT_EMBEDDING_SHAPE:
            raise ValueError(f"S4 prompt embedding must be {PROMPT_EMBEDDING_SHAPE}, got {tuple(text_emb.shape)}")
        batch = {
            "latents": video_x0,
            "actions": action_x0,
            "text_emb": text_emb,
            "actions_mask": torch.ones_like(action_x0[:, :1], dtype=torch.bool),
        }
        base = {"input": self.prepare_base_dict(batch, self.config, self.device), "config": self.config}
        action_downsample = int(getattr(self.config, "action_downsample_factor", 4))
        if action_downsample <= 0:
            raise ValueError("config.action_downsample_factor must be positive")
        video_t, video_r, action_t, action_r = self.build_paired_eval_timesteps(
            batch_size=1,
            video_frames=video_x0.shape[2],
            action_frames=action_x0.shape[2],
            t=1000,
            r=0,
            device=self.device,
        )
        action_noise = torch.randn_like(action_x0)
        rollout_input, _ = self.student_input(
            batch,
            base,
            video_x0,
            video_noise,
            video_t,
            action_noise,
            action_t,
            action_downsample,
        )
        with torch.no_grad():
            _video, _field, _action_sequence, final_action = self.harness._student_euler_integrate(
                noisy_latents=rollout_input["latent_dict"]["noisy_latents"],
                timesteps=video_t,
                target_r=video_r,
                base_input_dict=rollout_input,
                empty_emb=self.empty_embedding,
                cfg_scale=self.cfg_scale,
                ref_shape=tuple(video_x0.shape),
                B=1,
                num_frames=video_x0.shape[2],
                K_steps=S4_STEPS,
                action_target_r=action_r,
                return_final_action=True,
                return_final_action_state=True,
            )
        if tuple(final_action.shape) != tuple(rollout_input["action_dict"]["noisy_latents"].shape):
            raise RuntimeError(
                "S4 student returned an action state with an unexpected shape: "
                f"{tuple(final_action.shape)}"
            )
        return final_action


def run_joint_s4_student(
    runner: FlowMapJointS4Runner,
    video_x0: Any,
    action_x0: Any,
    text_emb: Any,
    *,
    noise: Any,
    t1000: Any,
    t0: Any,
    k_steps: int,
) -> Any:
    """Named deployment hook used by the service and audit logs."""
    return runner(
        video_x0,
        action_x0,
        text_emb,
        noise=noise,
        t1000=t1000,
        t0=t0,
        k_steps=k_steps,
    )


def build_live_service(args: argparse.Namespace) -> tuple[CosmosProgressiveS4Service, Any]:
    """Construct the live service without running any LIBERO environment."""
    require_live_s4_prerequisites(
        device=args.device,
        checkpoint_transformer=args.checkpoint_transformer,
    )
    import torch

    dependencies = _runtime_dependencies()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    config = _configure_live_config(args, dependencies)
    prompt_table = PromptEmbeddingTable.from_file(args.prompt_table)
    empty_embedding = _load_empty_embedding(config.empty_emb_path, torch_module=torch, device=device)
    student = dependencies["load_stage1_model"](
        Path(args.checkpoint_transformer), config, device, torch.bfloat16
    )
    harness = dependencies["StudentRolloutHarness"](student, config, device)
    teacher = dependencies["CosmosPolicyActionTeacher"](
        config.teacher_model_path,
        dtype=torch.bfloat16,
        device=str(device),
        config=config,
    )
    action_encoder = FlowMapActionAnchorEncoder(
        config=config,
        device=device,
        torch_module=torch,
        converter=dependencies["cosmos_actions_to_flowmap_x0"],
    )
    runner = FlowMapJointS4Runner(
        harness=harness,
        config=config,
        device=device,
        torch_module=torch,
        prepare_base_dict=dependencies["prepare_base_dict"],
        student_input=dependencies["student_input"],
        build_paired_eval_timesteps=dependencies["build_paired_eval_timesteps"],
        empty_embedding=empty_embedding,
        cfg_scale=args.cfg_scale,
    )
    engine = CosmosProgressiveS4Engine(
        cosmos_teacher=teacher,
        prompt_table=prompt_table,
        action_template=ActionDecodingTemplate.from_config(config),
        action_encoder=action_encoder,
        joint_s4_runner=lambda video_x0, action_x0, text_emb, **kwargs: run_joint_s4_student(
            runner, video_x0, action_x0, text_emb, **kwargs
        ),
        anchor_epsilon=float(getattr(config, "cosmos_latent_epsilon", 0.001)),
    )
    service = CosmosProgressiveS4Service(
        engine=engine,
        checkpoint_identifier=str(Path(args.checkpoint_transformer).resolve()),
        anchor_record_dir=args.anchor_record_dir,
    )
    return service, teacher


def serve_json_lines(service: CosmosProgressiveS4Service, *, input_stream: Any, output_stream: Any) -> None:
    """Serve only reset/infer messages; actions are serialized as JSON arrays."""
    for line in input_stream:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = dict(service.infer(request))
            if "action" in response:
                response["action"] = np.asarray(response["action"], dtype=np.float32).tolist()
            if "actions" in response:
                response["actions"] = np.asarray(response["actions"], dtype=np.float32).tolist()
            output_stream.write(json.dumps(response, sort_keys=True) + "\n")
        except Exception as exc:
            output_stream.write(
                json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}) + "\n"
            )
        output_stream.flush()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.preflight:
        require_live_s4_prerequisites(
            device=args.device,
            checkpoint_transformer=args.checkpoint_transformer,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "note": "Local preflight passed; it did not start a Cosmos worker or prove Cosmos CUDA-extra compatibility.",
                }
            )
        )
        return 0

    service, teacher = build_live_service(args)
    try:
        if args.serve_stdio:
            serve_json_lines(service, input_stream=sys.stdin, output_stream=sys.stdout)
            return 0
        client = CosmosProgressiveS4Client(
            service,
            output_dir=args.output_dir,
            warmup_steps=args.warmup_steps,
            warmup_gripper=args.warmup_gripper,
            skip_first_action=args.skip_first_action,
        )
        # Fetch benchmark cardinality lazily through the client route only when
        # a real rollout is requested.
        if args.task_range is None:
            from libero.libero import benchmark

            total = benchmark.get_benchmark_dict()[args.libero_benchmark]().get_num_tasks()
            task_indices = range(total)
        else:
            start, end = (int(value) for value in args.task_range)
            if start < 0 or end < start:
                raise ValueError("--task-range must be a non-negative [start, end) interval")
            task_indices = range(start, end)
        records = []
        for task_idx in task_indices:
            for episode_idx in range(args.episodes):
                records.append(
                    client.run_libero_task(
                        libero_benchmark=args.libero_benchmark,
                        task_idx=task_idx,
                        episode_idx=episode_idx,
                        camera_size=args.camera_size,
                        max_env_steps=args.max_env_steps,
                        env_seed=args.env_seed,
                        initial_states_json=args.initial_states_json,
                    )
                )
        print(json.dumps({"records": records}, indent=2, sort_keys=True))
        return 0
    finally:
        close = getattr(teacher, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())
