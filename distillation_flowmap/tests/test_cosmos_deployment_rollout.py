import pytest
import torch

from distillation_flowmap.cosmos_deployment_rollout import (
    deployment_endpoint_losses,
    deployment_joint_step_for_update,
    should_run_deployment_joint_rollout,
    should_run_raw_auxiliary,
)


def test_deployment_k_cycle_is_balanced_and_deterministic():
    assert [deployment_joint_step_for_update(i) for i in range(9)] == [
        1, 2, 4, 1, 2, 4, 1, 2, 4
    ]


def test_deployment_and_raw_auxiliary_never_share_an_optimizer_step():
    deployment = {
        step for step in range(64)
        if should_run_deployment_joint_rollout(step, interval=4)
    }
    raw_aux = {
        step for step in range(64)
        if should_run_raw_auxiliary(step, warmup=8, interval=8, phase=2)
    }
    assert deployment
    assert raw_aux
    assert deployment.isdisjoint(raw_aux)


def test_deployment_endpoint_loss_uses_final_state_at_sigma_zero():
    video_x0 = torch.ones(1, 2, 1, 1, 1)
    action_x0 = torch.ones(1, 3, 4, 4, 1)
    losses = deployment_endpoint_losses(
        video_final=video_x0 + 2,
        video_x0=video_x0,
        action_final=action_x0 + 3,
        action_x0=action_x0,
        action_mask=torch.ones(1, 1, 4, 4, 1),
        action_weight=1.0,
    )
    assert losses["video"] == pytest.approx(torch.tensor(4.0))
    assert losses["action"] == pytest.approx(torch.tensor(9.0))
    assert losses["total"] == pytest.approx(torch.tensor(13.0))
