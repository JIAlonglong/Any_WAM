import pytest
import torch
from pathlib import Path

from distillation_flowmap.opd_loss_composition import (
    compose_explicit_hybrid_opd,
)


def test_explicit_endpoint_is_counted_once():
    endpoint = torch.tensor(2.0, requires_grad=True)
    velocity = torch.tensor(0.0, requires_grad=True)
    action = torch.tensor(0.0, requires_grad=True)

    result = compose_explicit_hybrid_opd(
        endpoint_video_loss=endpoint,
        velocity_video_loss=velocity,
        endpoint_action_loss=action,
        beta_end_video=0.5,
        beta_vel_video=0.0,
        beta_end_action=0.0,
    )

    assert torch.allclose(result.loss, torch.tensor(1.0))
    assert torch.allclose(
        result.contributions["endpoint_video"], torch.tensor(1.0)
    )


def test_explicit_velocity_only_keeps_nonzero_loss_and_gradient():
    endpoint = torch.tensor(0.0, requires_grad=True)
    velocity = torch.tensor(3.0, requires_grad=True)
    action = torch.tensor(0.0, requires_grad=True)

    result = compose_explicit_hybrid_opd(
        endpoint_video_loss=endpoint,
        velocity_video_loss=velocity,
        endpoint_action_loss=action,
        beta_end_video=0.0,
        beta_vel_video=0.25,
        beta_end_action=0.0,
    )
    result.loss.backward()

    assert torch.allclose(result.loss, torch.tensor(0.75))
    assert torch.allclose(velocity.grad, torch.tensor(0.25))
    assert torch.allclose(result.ratios["velocity_video"], torch.tensor(1.0))


def test_explicit_contribution_ratios_sum_to_one():
    result = compose_explicit_hybrid_opd(
        endpoint_video_loss=torch.tensor(2.0),
        velocity_video_loss=torch.tensor(4.0),
        endpoint_action_loss=torch.tensor(8.0),
        beta_end_video=1.0,
        beta_vel_video=0.5,
        beta_end_action=0.25,
    )

    assert torch.allclose(result.loss, torch.tensor(6.0))
    assert torch.allclose(sum(result.ratios.values()), torch.tensor(1.0))


def test_explicit_composer_identifies_nonfinite_component():
    with pytest.raises(ValueError, match="velocity_video"):
        compose_explicit_hybrid_opd(
            endpoint_video_loss=torch.tensor(0.0),
            velocity_video_loss=torch.tensor(float("nan")),
            endpoint_action_loss=torch.tensor(0.0),
            beta_end_video=1.0,
            beta_vel_video=1.0,
            beta_end_action=0.0,
        )


def test_flowmap_step_wires_explicit_loss_composer():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_step.py"
    ).read_text(encoding="utf-8")

    assert "compose_explicit_hybrid_opd(" in source
    assert "opd_loss_composition == 'explicit_hybrid'" in source
