"""Dedicated 16-channel Cosmos-to-Progressive-S4 policy service.

The service deliberately has no LIBERO environment dependency.  It receives one
LIBERO observation per request, asks the official Cosmos raw worker for a fresh
16-channel latent anchor, and deploys the joint FlowMap S4 student through an
injected runner.  Keeping the environment on the client side prevents the raw
Cosmos worker from silently becoming a generic rollout or 48-channel WanVA
server.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from distillation_flowmap.cosmos_training_contract import (
    normalize_cosmos_inference_request,
)


S4_ACTION_STEPS = 16
S4_ACTION_DIM = 7
COSMOS_VIDEO_SHAPE = (1, 16, 9, 28, 28)
PROMPT_EMBEDDING_SHAPE = (1, 512, 4096)
SUPPORTED_STUDENT_STEPS = (1, 2, 4)


def student_action_grid_contract(
    *,
    action_tensor_shape: tuple[int, ...] | list[int],
    action_downsample_factor: int,
) -> dict[str, Any]:
    """Describe how the compact FlowMap action state maps to 16 LIBERO actions."""
    shape = tuple(int(value) for value in action_tensor_shape)
    factor = int(action_downsample_factor)
    if len(shape) != 5 or shape[0] != 1 or shape[-1] != 1:
        raise ValueError(f"student action tensor must be [1,C,F,N,1], got {shape}")
    if factor <= 0:
        raise ValueError("action_downsample_factor must be positive")
    frames, actions_per_frame = shape[2], shape[3]
    horizon = frames * actions_per_frame
    if horizon != S4_ACTION_STEPS:
        raise ValueError(
            f"compact student action grid must decode {S4_ACTION_STEPS} actions, got {horizon}"
        )
    return {
        "schema": "cosmos_action_grid_v1",
        "layout": "flowmap_compact_temporal_grid",
        "action_downsample_factor": factor,
        "action_tensor_shape": list(shape),
        "updated_action_latent_indices": list(range(frames)),
        "updated_action_indices": list(range(horizon)),
        "action_horizon": horizon,
    }


def summarize_action_temporal_frames(action_tensor: Any) -> list[dict[str, float | int]]:
    """Return portable mean/std/absmax diagnostics for each compact temporal frame."""
    action = _as_numpy(action_tensor).astype(np.float32, copy=False)
    if action.ndim != 5 or action.shape[0] != 1 or action.shape[-1] != 1:
        raise ValueError(f"action tensor must be [1,C,F,N,1], got {action.shape}")
    summaries = []
    for frame in range(int(action.shape[2])):
        values = action[:, :, frame, :, :].reshape(-1)
        summaries.append(
            {
                "frame": frame,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "absmax": float(np.abs(values).max()),
            }
        )
    return summaries


def normalize_student_steps(value: int) -> int:
    value = int(value)
    if value not in SUPPORTED_STUDENT_STEPS:
        raise ValueError(
            f"student_steps must be one of {SUPPORTED_STUDENT_STEPS}, got {value}"
        )
    return value


class CosmosProgressiveS4Error(RuntimeError):
    """Base error for the dedicated progressive S4 service."""


class CosmosRuntimePrerequisiteError(CosmosProgressiveS4Error):
    """Raised when a caller asks a CPU-only process to run live Cosmos inference."""


def _as_numpy(value: Any) -> np.ndarray:
    """Convert NumPy or torch-like values without importing torch at module import."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    # PyTorch does not support ``Tensor.numpy()`` for bfloat16.  The public S4
    # student commonly emits bfloat16 actions, whereas this rollout boundary
    # deliberately returns a portable float32 NumPy array.
    to_float = getattr(value, "float", None)
    if callable(to_float):
        value = to_float()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    return np.asarray(value)


def _require_prompt(prompt: Any) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("LIBERO prompt must be a non-empty string")
    return prompt


def _image_from_observation(
    observation: Mapping[str, Any],
    *,
    raw_name: str,
    aliases: tuple[str, ...],
) -> np.ndarray:
    """Read a raw LIBERO RGB image and apply the official vertical flip once."""
    if raw_name in observation:
        image = _as_numpy(observation[raw_name])
        image = image[::-1]
    else:
        alias = next((name for name in aliases if name in observation), None)
        if alias is None:
            names = (raw_name,) + aliases
            raise KeyError(f"LIBERO observation is missing RGB image; expected one of {names}")
        # Alias values are already canonical observation images.  Do not flip a
        # second time when a caller passes the output of extract_video_obs().
        image = _as_numpy(observation[alias])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image with three channels, got {image.shape}")
    return np.ascontiguousarray(image.astype(np.uint8, copy=False))


def libero_state_to_official_proprio(state: Any) -> np.ndarray:
    """Convert the dataset's 8-D ``[xyz, rpy, gripper2]`` state into Cosmos 9-D.

    The official raw policy expects ``[gripper_qpos(2), eef_pos(3),
    eef_quat(4)]``.  This matches
    :func:`distillation_flowmap.cosmos_policy_adapter.libero_state_to_cosmos_proprio`
    while keeping pure service tests independent of heavyweight model imports.
    """
    state_arr = _as_numpy(state).astype(np.float32, copy=False)
    if state_arr.shape != (8,):
        raise ValueError(f"Expected 8-D LIBERO state, got shape {state_arr.shape}")
    xyz = state_arr[:3]
    roll, pitch, yaw = state_arr[3:6] * 0.5
    gripper = state_arr[6:8]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    quat = np.asarray(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float32,
    )
    proprio = np.concatenate([gripper, xyz, quat]).astype(np.float32, copy=False)
    if proprio.shape != (9,):
        raise AssertionError(f"internal proprio conversion produced {proprio.shape}, not (9,)")
    return proprio


def _proprio_from_observation(observation: Mapping[str, Any]) -> np.ndarray:
    raw_fields = (
        "robot0_gripper_qpos",
        "robot0_eef_pos",
        "robot0_eef_quat",
    )
    if all(name in observation for name in raw_fields):
        proprio = np.concatenate(
            [
                _as_numpy(observation["robot0_gripper_qpos"]).astype(np.float32, copy=False),
                _as_numpy(observation["robot0_eef_pos"]).astype(np.float32, copy=False),
                _as_numpy(observation["robot0_eef_quat"]).astype(np.float32, copy=False),
            ],
            axis=0,
        ).astype(np.float32, copy=False)
    elif "raw_proprio" in observation:
        proprio = _as_numpy(observation["raw_proprio"]).astype(np.float32, copy=False)
    elif "state" in observation:
        proprio = libero_state_to_official_proprio(observation["state"])
    elif "observation.state" in observation:
        proprio = libero_state_to_official_proprio(observation["observation.state"])
    else:
        raise KeyError(
            "LIBERO observation needs robot0_gripper_qpos/robot0_eef_pos/robot0_eef_quat "
            "or an 8-D state for Cosmos proprio conversion"
        )
    if proprio.shape != (9,):
        raise ValueError(
            "Expected official Cosmos LIBERO proprio [gripper_qpos(2), eef_pos(3), "
            f"eef_quat(4)] with shape (9,), got {proprio.shape}"
        )
    return np.ascontiguousarray(proprio)


def build_cosmos_raw_request(*, libero_obs: Mapping[str, Any], prompt: str) -> dict[str, Any]:
    """Build the raw-anchor request used by the official LIBERO Cosmos loop.

    The brief's legacy "8-D state" wording refers to the source state.  The
    request sent to Cosmos is always the official 9-D proprio vector.
    """
    if not isinstance(libero_obs, Mapping):
        raise TypeError("libero_obs must be a mapping")
    prompt = _require_prompt(prompt)
    return {
        "raw_primary_image": _image_from_observation(
            libero_obs,
            raw_name="agentview_image",
            aliases=("observation.images.agentview_rgb", "raw_primary_image", "primary_image"),
        ),
        "raw_wrist_image": _image_from_observation(
            libero_obs,
            raw_name="robot0_eye_in_hand_image",
            aliases=("observation.images.eye_in_hand_rgb", "raw_wrist_image", "wrist_image"),
        ),
        "raw_proprio": _proprio_from_observation(libero_obs),
        "raw_task": prompt,
    }


class PromptEmbeddingTable:
    """Strict lookup table containing only embeddings emitted by training data.

    There is intentionally no text-encoder fallback: a task absent from this
    table must fail instead of introducing a deployment-time prompt distribution.
    """

    def __init__(self, embeddings: Mapping[str, Any]):
        if not isinstance(embeddings, Mapping):
            raise TypeError("prompt embedding table must be a mapping of task text to embeddings")
        self._embeddings = dict(embeddings)

    @classmethod
    def from_file(cls, path: str | Path) -> "PromptEmbeddingTable":
        """Load a task-keyed ``.npz`` or torch ``.pt`` training embedding table."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Prompt embedding table does not exist: {source}")
        if source.suffix == ".npz":
            with np.load(source, allow_pickle=False) as payload:
                return cls({name: payload[name] for name in payload.files})
        if source.suffix in {".pt", ".pth"}:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - torch is present in the training runtime
                raise CosmosRuntimePrerequisiteError(
                    "Loading a .pt prompt table requires torch; use a task-keyed .npz table for CPU checks."
                ) from exc
            payload = torch.load(source, map_location="cpu", weights_only=False)
            if isinstance(payload, Mapping) and isinstance(payload.get("embeddings"), Mapping):
                payload = payload["embeddings"]
            return cls(payload)
        raise ValueError("Prompt table must be a task-keyed .npz, .pt, or .pth file")

    def get(self, prompt: str) -> Any:
        prompt = _require_prompt(prompt)
        try:
            embedding = self._embeddings[prompt]
        except KeyError as exc:
            raise KeyError(
                f"missing prompt embedding for LIBERO task {prompt!r}; "
                "supply the training-data [1, 512, 4096] prompt table"
            ) from exc
        shape = tuple(getattr(embedding, "shape", ()))
        if shape != PROMPT_EMBEDDING_SHAPE:
            raise ValueError(
                f"Prompt embedding for {prompt!r} must have shape {list(PROMPT_EMBEDDING_SHAPE)}, "
                f"got {shape}"
            )
        return embedding


@dataclass(frozen=True)
class ActionDecodingTemplate:
    """FlowMap action-normalization metadata needed to recover 7-D actions."""

    q01: Any
    q99: Any
    inverse_used_action_channel_ids: tuple[int, ...]
    action_dim: int = S4_ACTION_DIM

    @classmethod
    def from_config(cls, config: Any) -> "ActionDecodingTemplate":
        return cls(
            q01=getattr(config, "norm_stat")["q01"],
            q99=getattr(config, "norm_stat")["q99"],
            inverse_used_action_channel_ids=tuple(
                int(value) for value in getattr(config, "inverse_used_action_channel_ids")
            ),
            action_dim=S4_ACTION_DIM,
        )


def _coerce_action_template(template: ActionDecodingTemplate | Mapping[str, Any]) -> ActionDecodingTemplate:
    if isinstance(template, ActionDecodingTemplate):
        return template
    if isinstance(template, Mapping):
        return ActionDecodingTemplate(
            q01=template["q01"],
            q99=template["q99"],
            inverse_used_action_channel_ids=tuple(template["inverse_used_action_channel_ids"]),
            action_dim=int(template.get("action_dim", S4_ACTION_DIM)),
        )
    raise TypeError("template must be an ActionDecodingTemplate or mapping")


def decode_student_action(
    flowmap_action_tensor: Any,
    template: ActionDecodingTemplate | Mapping[str, Any],
) -> np.ndarray:
    """Decode one FlowMap action state into exactly ``float32 [16, 7]`` actions."""
    metadata = _coerce_action_template(template)
    action = _as_numpy(flowmap_action_tensor)
    if action.ndim != 5 or action.shape[0] != 1 or action.shape[-1] != 1:
        raise ValueError(
            "Expected FlowMap action tensor [1, C, F, N, 1], "
            f"got {tuple(action.shape)}"
        )
    channels = int(action.shape[1])
    q01 = _as_numpy(metadata.q01).astype(np.float32, copy=False).reshape(-1)
    q99 = _as_numpy(metadata.q99).astype(np.float32, copy=False).reshape(-1)
    inverse = np.asarray(metadata.inverse_used_action_channel_ids, dtype=np.int64).reshape(-1)
    if q01.size != channels or q99.size != channels or inverse.size != channels:
        raise ValueError(
            "Action decoding metadata must have one q01/q99/inverse entry per FlowMap channel; "
            f"channels={channels}, q01={q01.size}, q99={q99.size}, inverse={inverse.size}"
        )
    if int(metadata.action_dim) != S4_ACTION_DIM:
        raise ValueError(f"S4 LIBERO decoder requires action_dim=7, got {metadata.action_dim}")

    # [1, C, F, N, 1] -> [F*N, C], then invert the training normalization.
    normalized = action[0, :, :, :, 0].transpose(1, 2, 0).reshape(-1, channels)
    unnormalized = (normalized.astype(np.float32) + 1.0) * 0.5 * (q99 - q01) + q01
    decoded = np.zeros((unnormalized.shape[0], S4_ACTION_DIM), dtype=np.float32)
    for flowmap_channel, raw_channel in enumerate(inverse.tolist()):
        if 0 <= raw_channel < S4_ACTION_DIM:
            decoded[:, raw_channel] = unnormalized[:, flowmap_channel]
    if decoded.shape[0] < S4_ACTION_STEPS:
        raise ValueError(
            f"FlowMap action state supplies only {decoded.shape[0]} actions; need {S4_ACTION_STEPS}"
        )
    return np.ascontiguousarray(decoded[:S4_ACTION_STEPS], dtype=np.float32)


class SharedNpzCosmosAnchorWorker:
    """Anchor-only façade over the official sequential shared-NPZ raw worker.

    ``CosmosPolicyActionTeacher.predict_raw_latent_target`` already owns the
    request-NPZ/response-NPZ protocol.  This façade makes the boundary explicit:
    it never receives a LIBERO environment and cannot act as a generic policy
    server.
    """

    def __init__(self, cosmos_teacher: Any):
        if not hasattr(cosmos_teacher, "predict_raw_latent_target"):
            raise TypeError("Cosmos anchor worker needs predict_raw_latent_target(raw_batch, ...)")
        self._teacher = cosmos_teacher

    def predict_raw_latent_target(
        self,
        raw_batch: Mapping[str, Any],
        *,
        noise: Any,
        t: Any,
        r: Any,
        epsilon: float,
    ) -> Mapping[str, Any]:
        return self._teacher.predict_raw_latent_target(
            raw_batch,
            noise=noise,
            t=t,
            r=r,
            epsilon=float(epsilon),
            include_cdiff=False,
        )


@dataclass
class S4Decision:
    action: np.ndarray
    student_action_latent: Any
    raw_anchor: Any
    raw_actions: Any
    prompt_embedding: Any


class CosmosProgressiveS4Engine:
    """Fresh-anchor execution core with injectable CPU-testable boundaries."""

    def __init__(
        self,
        *,
        cosmos_teacher: Any,
        prompt_table: PromptEmbeddingTable,
        action_template: ActionDecodingTemplate | Mapping[str, Any],
        action_encoder: Callable[[Any], Any],
        joint_s4_runner: Callable[..., Any],
        anchor_noise_factory: Callable[[], Any] | None = None,
        anchor_epsilon: float = 0.001,
        model_role: str = "stage2_target",
        video_steps: int | None = None,
        action_steps: int | None = None,
        student_steps: int | None = None,
        checkpoint_contract_identity: str | None = None,
        action_downsample_factor: int = 4,
    ) -> None:
        self.anchor_worker = SharedNpzCosmosAnchorWorker(cosmos_teacher)
        self.prompt_table = prompt_table
        self.action_template = _coerce_action_template(action_template)
        self.action_encoder = action_encoder
        self.joint_s4_runner = joint_s4_runner
        self.anchor_noise_factory = anchor_noise_factory or self._default_anchor_noise
        self.anchor_epsilon = float(anchor_epsilon)
        if video_steps is None and action_steps is None and student_steps is None:
            student_steps = 4
        request = normalize_cosmos_inference_request(
            model_role=model_role,
            video_steps=video_steps,
            action_steps=action_steps,
            student_steps=student_steps,
        )
        self.model_role = request.model_role
        self.video_steps = request.video_steps
        self.action_steps = request.action_steps
        self.student_steps = request.student_steps
        self.checkpoint_contract_identity = checkpoint_contract_identity
        self.action_downsample_factor = int(action_downsample_factor)
        if self.anchor_epsilon <= 0:
            raise ValueError("anchor_epsilon must be positive")
        if self.action_downsample_factor <= 0:
            raise ValueError("action_downsample_factor must be positive")

    @staticmethod
    def _default_anchor_noise() -> np.ndarray:
        return np.random.standard_normal(COSMOS_VIDEO_SHAPE).astype(np.float32)

    @staticmethod
    def _validate_anchor(anchor: Mapping[str, Any]) -> tuple[Any, Any]:
        if not isinstance(anchor, Mapping):
            raise TypeError(f"Cosmos raw anchor must be a mapping, got {type(anchor)!r}")
        missing = [key for key in ("cosmos_latent_x0", "actions") if key not in anchor]
        if missing:
            raise KeyError(f"Cosmos raw anchor is missing {missing}")
        video_x0 = anchor["cosmos_latent_x0"]
        action_values = anchor["actions"]
        if tuple(_as_numpy(video_x0).shape) != COSMOS_VIDEO_SHAPE:
            raise ValueError(
                "Cosmos Progressive S4 requires a raw 16-channel latent anchor "
                f"with shape {COSMOS_VIDEO_SHAPE}, got {tuple(_as_numpy(video_x0).shape)}"
            )
        raw_actions = _as_numpy(action_values)
        if raw_actions.ndim == 2:
            raw_actions = raw_actions[None]
        if raw_actions.ndim != 3 or raw_actions.shape[0] != 1 or raw_actions.shape[1] < S4_ACTION_STEPS or raw_actions.shape[2] < S4_ACTION_DIM:
            raise ValueError(
                "Cosmos raw anchor actions must have shape [1, T>=16, D>=7], "
                f"got {raw_actions.shape}"
            )
        return video_x0, action_values

    def infer_raw(self, raw_batch: Mapping[str, Any], prompt: str) -> S4Decision:
        """Use a new raw anchor for this control cycle; no anchor cache exists."""
        text_emb = self.prompt_table.get(prompt)
        noise = self.anchor_noise_factory()
        if tuple(_as_numpy(noise).shape) != COSMOS_VIDEO_SHAPE:
            raise ValueError(
                f"Cosmos anchor noise must have shape {COSMOS_VIDEO_SHAPE}, got {tuple(_as_numpy(noise).shape)}"
            )
        # Cosmos's raw latent endpoint API consumes normalized [0, 1] times.
        t1000 = np.ones((1, COSMOS_VIDEO_SHAPE[2]), dtype=np.float32)
        t0 = np.zeros((1, COSMOS_VIDEO_SHAPE[2]), dtype=np.float32)
        anchor = self.anchor_worker.predict_raw_latent_target(
            raw_batch,
            noise=noise,
            t=t1000,
            r=t0,
            epsilon=self.anchor_epsilon,
        )
        video_x0, raw_actions = self._validate_anchor(anchor)
        action_x0 = self.action_encoder(raw_actions)
        student_action = self.joint_s4_runner(
            video_x0,
            action_x0,
            text_emb,
            noise=noise,
            t1000=t1000,
            t0=t0,
            video_steps=self.video_steps,
            action_steps=self.action_steps,
        )
        action = decode_student_action(student_action, self.action_template)
        return S4Decision(
            action=action,
            student_action_latent=student_action,
            raw_anchor=video_x0,
            raw_actions=raw_actions,
            prompt_embedding=text_emb,
        )


class RawAnchorLedger:
    """Persist raw 16-channel anchors for post-hoc rollout audits."""

    def __init__(self, root: str | Path | None) -> None:
        self.root = Path(root) if root is not None else None
        self._index = 0
        self.last_anchor: np.ndarray | None = None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        raw_anchor: Any,
        raw_actions: Any,
        student_action_latent: Any,
        prompt: str,
    ) -> str | None:
        anchor = np.ascontiguousarray(_as_numpy(raw_anchor).astype(np.float32, copy=False))
        if tuple(anchor.shape) != COSMOS_VIDEO_SHAPE:
            raise ValueError(f"Cannot record non-S4 anchor with shape {anchor.shape}")
        self.last_anchor = anchor.copy()
        if self.root is None:
            return None
        filename = f"anchor_{self._index:06d}.npz"
        self._index += 1
        path = self.root / filename
        np.savez_compressed(
            path,
            cosmos_latent_x0=self.last_anchor,
            actions=np.ascontiguousarray(_as_numpy(raw_actions).astype(np.float32, copy=False)),
            student_action_latent=np.ascontiguousarray(
                _as_numpy(student_action_latent).astype(np.float32, copy=False)
            ),
            prompt=np.asarray(prompt),
        )
        return filename


class CosmosProgressiveS4Service:
    """LIBERO-style reset/infer service returning only one 16x7 action chunk."""

    def __init__(
        self,
        *,
        engine: CosmosProgressiveS4Engine,
        checkpoint_identifier: str,
        checkpoint_contract_identity: str | None = None,
        anchor_record_dir: str | Path | None = None,
    ) -> None:
        self.engine = engine
        self.checkpoint_identifier = _require_prompt(str(checkpoint_identifier))
        self.checkpoint_contract_identity = checkpoint_contract_identity or engine.checkpoint_contract_identity
        self.anchor_ledger = RawAnchorLedger(anchor_record_dir)

    def _action_grid_contract(self) -> dict[str, Any]:
        channels = int(_as_numpy(self.engine.action_template.q01).reshape(-1).size)
        return student_action_grid_contract(
            action_tensor_shape=(1, channels, 4, 4, 1),
            action_downsample_factor=self.engine.action_downsample_factor,
        )

    def infer(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise TypeError("service request must be a mapping")
        if bool(request.get("reset", False)):
            return {
                "ok": True,
                "reset": True,
                "s4_checkpoint": self.checkpoint_identifier,
                "checkpoint_contract_identity": self.checkpoint_contract_identity,
                "model_role": self.engine.model_role,
                "video_steps": self.engine.video_steps,
                "action_steps": self.engine.action_steps,
                "student_steps": self.engine.student_steps,
                "action_grid_contract": self._action_grid_contract(),
            }
        prompt = _require_prompt(request.get("prompt", request.get("task")))
        libero_obs = request.get("obs", request.get("libero_obs"))
        if libero_obs is None:
            raise KeyError("infer request needs an 'obs' or 'libero_obs' mapping")
        raw_batch = build_cosmos_raw_request(libero_obs=libero_obs, prompt=prompt)
        started = time.perf_counter()
        decision = self.engine.infer_raw(raw_batch, prompt)
        duration = time.perf_counter() - started
        anchor_record = self.anchor_ledger.record(
            raw_anchor=decision.raw_anchor,
            raw_actions=decision.raw_actions,
            student_action_latent=decision.student_action_latent,
            prompt=prompt,
        )
        # ``decode_student_action`` is the final type/shape gate, but keep this
        # check local to protect callers replacing the engine in integration code.
        action = np.ascontiguousarray(decision.action, dtype=np.float32)
        if action.shape != (S4_ACTION_STEPS, S4_ACTION_DIM):
            raise AssertionError(f"service produced invalid S4 action shape {action.shape}")
        action_grid = student_action_grid_contract(
            action_tensor_shape=tuple(_as_numpy(decision.student_action_latent).shape),
            action_downsample_factor=self.engine.action_downsample_factor,
        )
        return {
            "ok": True,
            "action": action,
            "actions": action,
            "decision_duration_s": float(duration),
            "raw_anchor_record": anchor_record,
            "raw_anchor_shape": list(COSMOS_VIDEO_SHAPE),
            "s4_checkpoint": self.checkpoint_identifier,
            "checkpoint_contract_identity": self.checkpoint_contract_identity,
            "model_role": self.engine.model_role,
            "video_steps": self.engine.video_steps,
            "action_steps": self.engine.action_steps,
            "student_steps": self.engine.student_steps,
            "action_grid_contract": action_grid,
            "action_frame_stats": summarize_action_temporal_frames(
                decision.student_action_latent
            ),
        }


def serialize_service_response(response: Mapping[str, Any]) -> str:
    """Serialize metadata for line-oriented transports without embedding actions."""
    metadata = dict(response)
    action = metadata.pop("action", metadata.pop("actions", None))
    if action is not None:
        metadata["action_shape"] = list(_as_numpy(action).shape)
        metadata["action_dtype"] = str(_as_numpy(action).dtype)
    return json.dumps(metadata, sort_keys=True)
