from __future__ import annotations

import torch


DEPLOYMENT_STUDENT_STEPS = (1, 2, 4)


def deployment_joint_step_for_update(update_index: int) -> int:
    if update_index < 0:
        raise ValueError("update_index must be non-negative")
    return DEPLOYMENT_STUDENT_STEPS[update_index % len(DEPLOYMENT_STUDENT_STEPS)]


def should_run_deployment_joint_rollout(step: int, interval: int = 4) -> bool:
    return step >= 0 and interval > 0 and step % interval == 0


def should_run_raw_auxiliary(
    step: int, *, warmup: int, interval: int = 8, phase: int = 2
) -> bool:
    if interval <= 0 or not 0 <= phase < interval:
        raise ValueError("raw auxiliary interval/phase are invalid")
    return step >= warmup and step % interval == phase


def deployment_endpoint_losses(
    video_final: torch.Tensor,
    video_x0: torch.Tensor,
    action_final: torch.Tensor,
    action_x0: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    action_weight: float,
) -> dict[str, torch.Tensor]:
    if action_weight < 0:
        raise ValueError("action_weight must be non-negative")
    video = (video_final.float() - video_x0.detach().float()).square().mean()
    action_diff = action_final.float() - action_x0.detach().float()
    try:
        mask = torch.broadcast_to(action_mask.detach().float(), action_diff.shape)
    except RuntimeError as exc:
        raise ValueError("action_mask must be broadcastable to action shape") from exc
    diff = action_diff * mask
    denom = mask.sum().clamp(min=1)
    action = diff.square().sum() / denom
    return {"video": video, "action": action, "total": video + action_weight * action}
