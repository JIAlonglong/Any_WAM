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
import hashlib
import importlib
import json
import os
import random
import re
import subprocess
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
    SUPPORTED_STUDENT_STEPS,
)
from distillation_flowmap.cosmos_training_contract import (
    normalize_cosmos_inference_request,
)
from distillation_flowmap.cosmos_stage2_lineage import (
    resolve_cosmos_inference_checkpoint,
)


_MIN_COSMOS_DRIVER = (570, 124, 6)
_MIN_COSMOS_CUDA = (12, 8)
def _version_tuple(value: str) -> tuple[int, ...]:
    """Extract a comparable numeric version without importing CUDA packages."""
    return tuple(int(part) for part in re.findall(r"\d+", str(value)))


def _nvidia_driver_version() -> str | None:
    """Read the installed NVIDIA driver without initializing a CUDA context."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def _cosmos_python_cuda_version(cosmos_python: str | Path) -> str:
    """Prove the external worker has an importable official Cosmos CUDA runtime."""
    env = os.environ.copy()
    repo = env.get("COSMOS_PREDICT2_REPO", "").strip()
    if repo:
        env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    try:
        result = subprocess.run(
            [
                str(cosmos_python),
                "-c",
                "import torch; import cosmos_predict2; print(torch.version.cuda or '')",
            ],
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CosmosRuntimePrerequisiteError(
            "Could not execute COSMOS_POLICY_PYTHON to validate the official Cosmos cu128 runtime."
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise CosmosRuntimePrerequisiteError(
            "COSMOS_POLICY_PYTHON cannot import the official Cosmos runtime "
            f"(cosmos_predict2): {detail}"
        )
    value = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if not value:
        raise CosmosRuntimePrerequisiteError(
            "COSMOS_POLICY_PYTHON reported no CUDA runtime; use the official Cosmos cu128 environment."
        )
    return value


def resolve_cosmos_policy_python() -> Path:
    """Require an explicitly supplied official Cosmos worker interpreter."""

    raw = os.environ.get("COSMOS_POLICY_PYTHON", "").strip()
    if not raw:
        raise CosmosRuntimePrerequisiteError(
            "COSMOS_POLICY_PYTHON must explicitly name the official Cosmos cu128 "
            "interpreter; no machine-local fallback is permitted."
        )
    path = Path(raw)
    if not path.is_file():
        raise CosmosRuntimePrerequisiteError(
            f"Official Cosmos Python is missing: {path}. Set COSMOS_POLICY_PYTHON "
            "to the cu128 Cosmos environment; do not substitute the Flash-WAM Python."
        )
    return path


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

    driver = _nvidia_driver_version()
    if driver is None:
        raise CosmosRuntimePrerequisiteError(
            "Could not determine the NVIDIA driver with nvidia-smi. Official Cosmos cu128 "
            "requires NVIDIA driver >=570.124.06; do not start the raw worker on this host."
        )
    if _version_tuple(driver) < _MIN_COSMOS_DRIVER:
        raise CosmosRuntimePrerequisiteError(
            f"NVIDIA driver {driver} is incompatible with official Cosmos cu128; "
            "requires >=570.124.06. Do not start the raw worker on this host."
        )

    cosmos_python = resolve_cosmos_policy_python()
    cosmos_cuda = _cosmos_python_cuda_version(cosmos_python)
    if _version_tuple(cosmos_cuda) < _MIN_COSMOS_CUDA:
        raise CosmosRuntimePrerequisiteError(
            f"Cosmos worker CUDA runtime {cosmos_cuda!r} is incompatible; requires CUDA >=12.8 "
            "from the official cu128 Cosmos environment."
        )


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
    parser.add_argument(
        "--model-role",
        choices=("stage1_target", "stage2_online", "stage2_target", "official_teacher"),
        default="stage2_target",
        help="Checkpoint role; official_teacher is rejected until a matched-K adapter exists.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument(
        "--student-steps",
        type=int,
        choices=SUPPORTED_STUDENT_STEPS,
        default=None,
        help="Deprecated compatibility alias; use --video-steps and --action-steps.",
    )
    parser.add_argument("--video-steps", type=int, choices=SUPPORTED_STUDENT_STEPS, default=None)
    parser.add_argument("--action-steps", type=int, choices=SUPPORTED_STUDENT_STEPS, default=None)
    parser.add_argument("--anchor-record-dir", default="outputs/cosmos_progressive_s4/anchors")
    parser.add_argument("--output-dir", default="outputs/cosmos_progressive_s4")
    parser.add_argument("--libero-benchmark", default="libero_10")
    parser.add_argument("--task-range", type=int, nargs=2, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--episode-index-offset",
        type=int,
        default=0,
        help="Add this offset to local episode indices for durable record identity.",
    )
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--max-env-steps", type=int, default=800)
    parser.add_argument("--env-seed", type=int, default=None)
    parser.add_argument("--initial-states-json", default=None)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--warmup-gripper", type=float, default=0.0)
    parser.add_argument("--skip-first-action", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--serve-stdio",
        action="store_true",
        help="Serve reset/infer JSON lines only; no LIBERO environment is constructed here.",
    )
    return parser.parse_args(argv)


def resolve_cli_inference_request(args: argparse.Namespace):
    """Resolve new explicit budgets while keeping an old bare CLI at K=4."""

    student_steps = args.student_steps
    if (
        args.video_steps is None
        and args.action_steps is None
        and student_steps is None
    ):
        student_steps = 4
    return normalize_cosmos_inference_request(
        model_role=args.model_role,
        video_steps=args.video_steps,
        action_steps=args.action_steps,
        student_steps=student_steps,
    )


def seed_live_rollout(seed: int) -> None:
    """Pair all host/student random streams for the same closed-loop seed."""
    import torch

    resolved_seed = int(seed)
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)


def derive_episode_seed(env_seed: int, task_idx: int, episode_idx: int) -> int:
    """Derive a stable K-independent RNG seed for one durable episode."""
    material = (
        f"cosmos-progressive-s4:{int(env_seed)}:{int(task_idx)}:{int(episode_idx)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


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
            packing_schema=self.config.action_packing_schema,
            downsample_factor=self.config.action_downsample_factor,
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
        model_role: str = "stage2_target",
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
        self.model_role = str(model_role)

    def __call__(
        self,
        video_x0: Any,
        action_x0: Any,
        text_emb: Any,
        *,
        noise: Any,
        t1000: Any,
        t0: Any,
        video_steps: int,
        action_steps: int,
        return_trajectory: bool = False,
    ) -> Any:
        request = normalize_cosmos_inference_request(
            model_role=self.model_role,
            video_steps=video_steps,
            action_steps=action_steps,
            student_steps=None,
        )
        k_steps = request.student_steps
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
        # ``_student_input`` mixes video states with a normalized sigma, while
        # FlowMap's deployment timestep remains in the raw 0..1000 domain.
        # At the S4 prior (t=1000), the state must be exactly the fresh noise;
        # passing the raw timestep into that mixing rule would produce an
        # invalid -999*x0 + 1000*noise state.
        rollout_input["latent_dict"]["noisy_latents"] = video_noise
        rollout_input["latent_dict"]["timesteps"] = video_t
        with torch.no_grad():
            integrated = self.harness._student_euler_integrate(
                noisy_latents=rollout_input["latent_dict"]["noisy_latents"],
                timesteps=video_t,
                target_r=video_r,
                base_input_dict=rollout_input,
                empty_emb=self.empty_embedding,
                cfg_scale=self.cfg_scale,
                ref_shape=tuple(video_x0.shape),
                B=1,
                num_frames=video_x0.shape[2],
                K_steps=k_steps,
                action_target_r=action_r,
                return_final_action=True,
                return_final_action_state=True,
                return_trajectory=return_trajectory,
            )
        if return_trajectory:
            _video, _field, _action_sequence, final_action, trajectory = integrated
        else:
            _video, _field, _action_sequence, final_action = integrated
        if tuple(final_action.shape) != tuple(rollout_input["action_dict"]["noisy_latents"].shape):
            raise RuntimeError(
                "S4 student returned an action state with an unexpected shape: "
                f"{tuple(final_action.shape)}"
            )
        return (final_action, trajectory) if return_trajectory else final_action


def run_joint_s4_student(
    runner: FlowMapJointS4Runner,
    video_x0: Any,
    action_x0: Any,
    text_emb: Any,
    *,
    noise: Any,
    t1000: Any,
    t0: Any,
    video_steps: int,
    action_steps: int,
) -> Any:
    """Named deployment hook used by the service and audit logs."""
    return runner(
        video_x0,
        action_x0,
        text_emb,
        noise=noise,
        t1000=t1000,
        t0=t0,
        video_steps=video_steps,
        action_steps=action_steps,
    )


def build_live_service(args: argparse.Namespace) -> tuple[CosmosProgressiveS4Service, Any]:
    """Construct the live service without running any LIBERO environment."""
    request = resolve_cli_inference_request(args)
    resolved_checkpoint = resolve_cosmos_inference_checkpoint(
        model_role=request.model_role,
        checkpoint_transformer=args.checkpoint_transformer,
    )
    require_live_s4_prerequisites(
        device=args.device,
        checkpoint_transformer=resolved_checkpoint.transformer_path,
    )
    import torch

    dependencies = _runtime_dependencies()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    config = _configure_live_config(args, dependencies)
    prompt_table = PromptEmbeddingTable.from_file(args.prompt_table)
    empty_embedding = _load_empty_embedding(config.empty_emb_path, torch_module=torch, device=device)
    student = dependencies["load_stage1_model"](
        Path(resolved_checkpoint.transformer_path), config, device, torch.bfloat16
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
        model_role=request.model_role,
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
        model_role=request.model_role,
        video_steps=request.video_steps,
        action_steps=request.action_steps,
        checkpoint_contract_identity=resolved_checkpoint.checkpoint_contract_identity,
    )
    service = CosmosProgressiveS4Service(
        engine=engine,
        checkpoint_identifier=resolved_checkpoint.transformer_path,
        checkpoint_contract_identity=resolved_checkpoint.checkpoint_contract_identity,
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
    if args.episode_index_offset < 0:
        raise ValueError("--episode-index-offset must be non-negative")
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

    # A durable closed-loop result is keyed by a shared rollout seed.  Require
    # it before constructing the student or raw Cosmos worker; stdio serving
    # is intentionally seedless because it owns no environment trials.
    if not args.serve_stdio and args.env_seed is None:
        raise ValueError(
            "--env-seed is required for live LIBERO rollout so every record has seed provenance"
        )

    if not args.serve_stdio:
        seed_live_rollout(args.env_seed)

    request = resolve_cli_inference_request(args)
    service, teacher = build_live_service(args)
    try:
        if args.serve_stdio:
            serve_json_lines(service, input_stream=sys.stdin, output_stream=sys.stdout)
            return 0
        client = CosmosProgressiveS4Client(
            service,
            output_dir=args.output_dir,
            student_steps=request.student_steps,
            expected_s4_checkpoint=getattr(
                service,
                "checkpoint_identifier",
                str(Path(args.checkpoint_transformer).resolve()),
            ),
            expected_checkpoint_contract_identity=getattr(
                service, "checkpoint_contract_identity", None
            ),
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
            for local_episode_idx in range(args.episodes):
                episode_idx = args.episode_index_offset + local_episode_idx
                seed_live_rollout(
                    derive_episode_seed(args.env_seed, task_idx, episode_idx)
                )
                records.append(
                    client.run_libero_task(
                        libero_benchmark=args.libero_benchmark,
                        task_idx=task_idx,
                        episode_idx=episode_idx,
                        camera_size=args.camera_size,
                        max_env_steps=args.max_env_steps,
                        env_seed=args.env_seed,
                        initial_states_json=args.initial_states_json,
                        save_video=args.save_video,
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
