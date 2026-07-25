from __future__ import annotations

from dataclasses import dataclass

import torch

CONTRACT_VERSION = 2
ACTION_PACKING_SCHEMA = "downsample_survivor_v2"
ACTION_DOWNSAMPLE_FACTOR = 4
RAW_STAGE1 = "raw_stage1"
PROGRESSIVE_STAGE2 = "progressive_stage2"
SUPPORTED_COSMOS_STUDENT_STEPS = (1, 2, 4)
SUPPORTED_COSMOS_MODEL_ROLES = (
    "stage1_target",
    "stage2_online",
    "stage2_target",
    "official_teacher",
)


@dataclass(frozen=True)
class CosmosInferenceRequest:
    """A fully resolved, matched-budget Cosmos inference request.

    ``student_steps`` remains an output-only compatibility field.  Callers
    should pass the explicit video/action budgets for every new request.
    """

    model_role: str
    video_steps: int
    action_steps: int
    student_steps: int | None


def normalize_cosmos_inference_request(
    *,
    model_role: str,
    video_steps: int | None,
    action_steps: int | None,
    student_steps: int | None,
) -> CosmosInferenceRequest:
    """Validate an explicit matched video/action inference budget.

    The historical scalar ``student_steps`` is accepted only as an
    unambiguous alias. Official-teacher requests use the same explicit
    matched-budget contract; their separate runtime adapter proves the
    effective budget returned by the official worker.
    """

    if model_role not in SUPPORTED_COSMOS_MODEL_ROLES:
        raise ValueError(
            "model_role must be one of "
            f"{SUPPORTED_COSMOS_MODEL_ROLES!r}, got {model_role!r}"
        )
    if student_steps is not None:
        if video_steps is not None or action_steps is not None:
            raise ValueError(
                "student_steps is a compatibility alias and cannot be combined "
                "with video_steps or action_steps"
            )
        video_steps = student_steps
        action_steps = student_steps
    if video_steps is None or action_steps is None:
        raise ValueError(
            "video_steps and action_steps must both be explicitly provided"
        )
    if type(video_steps) is not int or video_steps not in SUPPORTED_COSMOS_STUDENT_STEPS:
        raise ValueError(
            "video_steps must be one of "
            f"{SUPPORTED_COSMOS_STUDENT_STEPS!r}, got {video_steps!r}"
        )
    if type(action_steps) is not int or action_steps not in SUPPORTED_COSMOS_STUDENT_STEPS:
        raise ValueError(
            "action_steps must be one of "
            f"{SUPPORTED_COSMOS_STUDENT_STEPS!r}, got {action_steps!r}"
        )
    if video_steps != action_steps:
        raise ValueError(
            "video_steps and action_steps must be equal for matched-budget "
            f"evaluation, got {video_steps!r} and {action_steps!r}"
        )
    return CosmosInferenceRequest(
        model_role=model_role,
        video_steps=video_steps,
        action_steps=action_steps,
        student_steps=None if model_role == "official_teacher" else video_steps,
    )

_COMMON_CONTRACT = {
    "contract_version": CONTRACT_VERSION,
    "action_packing_schema": ACTION_PACKING_SCHEMA,
    "action_downsample_factor": ACTION_DOWNSAMPLE_FACTOR,
    "action_chunk_shape": [4, 4],
}
_PROGRESSIVE_STAGE2_CONTRACT = {
    "deployment_timestep_start": 1000,
    "deployment_timestep_end": 0,
    "joint_student_steps": [1, 2, 4],
    "deployment_joint_rollout_interval": 4,
    "deployment_action_weight": 1.0,
    "raw_teacher_window_is_auxiliary": True,
}


def _require_exact(field: str, actual: object, expected: object) -> None:
    if type(actual) is list and type(expected) is list:
        if len(actual) != len(expected):
            raise ValueError(
                f"{field} must be exactly {expected!r} (list), got {actual!r} (list)"
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _require_exact(f"{field}[{index}]", actual_item, expected_item)
        return
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(
            f"{field} must be exactly {expected!r} "
            f"({type(expected).__name__}), got {actual!r} "
            f"({type(actual).__name__})"
        )


def validate_contract_metadata(
    payload: dict[str, object], *, required_stage: str
) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"contract metadata must be a dict, got {type(payload).__name__}")
    if required_stage not in (RAW_STAGE1, PROGRESSIVE_STAGE2):
        raise ValueError(f"unsupported training contract stage {required_stage!r}")

    expected = {
        **_COMMON_CONTRACT,
        "training_contract_stage": required_stage,
    }
    if required_stage == PROGRESSIVE_STAGE2:
        expected.update(_PROGRESSIVE_STAGE2_CONTRACT)
    else:
        forbidden = sorted(set(payload).intersection(_PROGRESSIVE_STAGE2_CONTRACT))
        if forbidden:
            raise ValueError(
                "raw_stage1 contract metadata must not contain Stage-2 field "
                f"{forbidden[0]!r}"
            )

    for field, value in expected.items():
        if field not in payload:
            raise ValueError(f"contract metadata is missing required field {field!r}")
        _require_exact(field, payload[field], value)


def contract_metadata(config, *, stage: str) -> dict[str, object]:
    if stage not in (RAW_STAGE1, PROGRESSIVE_STAGE2):
        raise ValueError(f"unsupported training contract stage {stage!r}")

    payload = {
        "contract_version": getattr(config, "contract_version", None),
        "training_contract_stage": stage,
        "action_packing_schema": getattr(config, "action_packing_schema", None),
        "action_downsample_factor": getattr(config, "action_downsample_factor", None),
        "action_chunk_shape": getattr(config, "action_chunk_shape", None),
    }
    if stage == PROGRESSIVE_STAGE2:
        deployment_steps = getattr(config, "deployment_joint_steps", None)
        if type(deployment_steps) is not tuple:
            raise ValueError(
                "deployment_joint_steps must be exactly the tuple (1, 2, 4)"
            )
        payload.update(
            {
                "deployment_timestep_start": getattr(
                    config, "deployment_timestep_start", None
                ),
                "deployment_timestep_end": getattr(
                    config, "deployment_timestep_end", None
                ),
                "joint_student_steps": list(deployment_steps),
                "deployment_joint_rollout_interval": getattr(
                    config, "deployment_joint_rollout_interval", None
                ),
                "deployment_action_weight": getattr(
                    config, "deployment_action_weight", None
                ),
                "raw_teacher_window_is_auxiliary": getattr(
                    config, "raw_teacher_window_is_auxiliary", None
                ),
            }
        )
    validate_contract_metadata(payload, required_stage=stage)
    return payload


def pack_actions_for_downsample(
    aligned: torch.Tensor,
    target_shape: tuple[int, int, int, int, int],
    *,
    downsample_factor: int,
    schema: str,
) -> torch.Tensor:
    if schema != ACTION_PACKING_SCHEMA:
        raise ValueError(
            f"action packing schema must be {ACTION_PACKING_SCHEMA!r}, got {schema!r}"
        )
    batch, channels, frames, per_frame, width = target_shape
    if (
        downsample_factor != ACTION_DOWNSAMPLE_FACTOR
        or frames != downsample_factor * 4
        or per_frame != 4
        or width != 1
    ):
        raise ValueError(
            "production action carrier requires downsample_factor=4, "
            "compact_frames=4, per_frame=4, and width=1"
        )
    compact_frames = frames // downsample_factor
    capacity = compact_frames * per_frame
    if capacity != 16:
        raise ValueError(
            f"production action carrier requires capacity=16, got {capacity}"
        )
    if aligned.shape != (batch, capacity, channels):
        raise ValueError(
            f"expected aligned actions {(batch, capacity, channels)}, got {tuple(aligned.shape)}"
        )

    packed = torch.zeros(
        batch, channels, frames, per_frame, width,
        device=aligned.device, dtype=aligned.dtype,
    )
    compact = aligned.reshape(batch, compact_frames, per_frame, channels)
    packed[:, :, ::downsample_factor, :, 0] = compact.permute(0, 3, 1, 2)
    return packed
