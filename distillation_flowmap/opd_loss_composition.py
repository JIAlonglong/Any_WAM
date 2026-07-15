"""Loss composition helpers for opt-in explicit OPD calibration."""

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class ExplicitOpdLossResult:
    loss: torch.Tensor
    contributions: Mapping[str, torch.Tensor]
    ratios: Mapping[str, torch.Tensor]


def compose_explicit_hybrid_opd(
    *,
    endpoint_video_loss: torch.Tensor,
    velocity_video_loss: torch.Tensor,
    endpoint_action_loss: torch.Tensor,
    beta_end_video: float,
    beta_vel_video: float,
    beta_end_action: float,
) -> ExplicitOpdLossResult:
    """Compose endpoint and velocity losses without hidden group scaling."""
    raw_losses = {
        "endpoint_video": endpoint_video_loss,
        "velocity_video": velocity_video_loss,
        "endpoint_action": endpoint_action_loss,
    }
    for name, value in raw_losses.items():
        if not bool(torch.isfinite(value.detach()).all()):
            raise ValueError(f"Non-finite explicit OPD component: {name}")

    contributions = {
        "endpoint_video": endpoint_video_loss * float(beta_end_video),
        "velocity_video": velocity_video_loss * float(beta_vel_video),
        "endpoint_action": endpoint_action_loss * float(beta_end_action),
    }
    loss = (
        contributions["endpoint_video"]
        + contributions["velocity_video"]
        + contributions["endpoint_action"]
    )
    denominator = sum(
        value.detach().abs() for value in contributions.values()
    ).clamp(min=1e-12)
    ratios = {
        name: value.detach().abs() / denominator
        for name, value in contributions.items()
    }
    return ExplicitOpdLossResult(
        loss=loss,
        contributions=contributions,
        ratios=ratios,
    )
