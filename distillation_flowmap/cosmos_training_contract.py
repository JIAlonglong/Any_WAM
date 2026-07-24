from __future__ import annotations

import torch

CONTRACT_VERSION = 2
ACTION_PACKING_SCHEMA = "downsample_survivor_v2"
ACTION_DOWNSAMPLE_FACTOR = 4
RAW_STAGE1 = "raw_stage1"
PROGRESSIVE_STAGE2 = "progressive_stage2"

_COMMON_CONTRACT = {
    "contract_version": CONTRACT_VERSION,
    "action_packing_schema": ACTION_PACKING_SCHEMA,
    "action_downsample_factor": ACTION_DOWNSAMPLE_FACTOR,
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
