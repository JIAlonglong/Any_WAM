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


def test_deployment_steps_match_the_compositional_training_budgets():
    deployment_steps = tuple(
        deployment_joint_step_for_update(i) for i in range(3)
    )
    compositional_steps = (2, 4)

    assert deployment_steps == (1, 2, 4)
    assert tuple(step for step in deployment_steps if step > 1) == compositional_steps


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


def test_deployment_endpoint_loss_normalizes_a_full_shaped_mask():
    action_x0 = torch.zeros(1, 2, 1, 1, 2)
    action_final = torch.tensor([[[[[1.0, 2.0]]], [[[3.0, 4.0]]]]])
    losses = deployment_endpoint_losses(
        video_final=torch.zeros(1),
        video_x0=torch.zeros(1),
        action_final=action_final,
        action_x0=action_x0,
        action_mask=torch.ones_like(action_final),
        action_weight=1.0,
    )
    assert losses["action"] == pytest.approx(torch.tensor(7.5))


def test_deployment_endpoint_loss_excludes_masked_action_entries():
    action_x0 = torch.zeros(1, 2, 1, 1, 1)
    action_final = torch.tensor([[[[[2.0]]], [[[10.0]]]]])
    losses = deployment_endpoint_losses(
        video_final=torch.zeros(1),
        video_x0=torch.zeros(1),
        action_final=action_final,
        action_x0=action_x0,
        action_mask=torch.tensor([[[[[1.0]]], [[[0.0]]]]]),
        action_weight=1.0,
    )
    assert losses["action"] == pytest.approx(torch.tensor(4.0))


def test_deployment_endpoint_loss_empty_mask_is_finite_zero():
    action_x0 = torch.zeros(1, 2, 1, 1, 1)
    losses = deployment_endpoint_losses(
        video_final=torch.zeros(1),
        video_x0=torch.zeros(1),
        action_final=torch.ones_like(action_x0),
        action_x0=action_x0,
        action_mask=torch.zeros_like(action_x0),
        action_weight=1.0,
    )
    assert torch.isfinite(losses["action"])
    assert losses["action"] == pytest.approx(torch.tensor(0.0))


def test_deployment_endpoint_loss_rejects_incompatible_action_mask():
    action_x0 = torch.zeros(1, 2, 1, 1, 1)
    with pytest.raises(ValueError, match="broadcastable"):
        deployment_endpoint_losses(
            video_final=torch.zeros(1),
            video_x0=torch.zeros(1),
            action_final=torch.ones_like(action_x0),
            action_x0=action_x0,
            action_mask=torch.ones(3),
            action_weight=1.0,
        )


def test_deployment_endpoint_loss_only_backpropagates_into_final_states():
    video_final = torch.ones(1, requires_grad=True)
    video_x0 = torch.zeros(1, requires_grad=True)
    action_final = torch.ones(1, 2, 1, 1, 1, requires_grad=True)
    action_x0 = torch.zeros_like(action_final, requires_grad=True)
    action_mask = torch.ones_like(action_final, requires_grad=True)
    losses = deployment_endpoint_losses(
        video_final=video_final,
        video_x0=video_x0,
        action_final=action_final,
        action_x0=action_x0,
        action_mask=action_mask,
        action_weight=1.0,
    )
    losses["total"].backward()
    assert video_final.grad is not None
    assert action_final.grad is not None
    assert video_x0.grad is None
    assert action_x0.grad is None
    assert action_mask.grad is None


def test_deployment_schedule_and_loss_validation():
    with pytest.raises(ValueError, match="non-negative"):
        deployment_joint_step_for_update(-1)
    assert should_run_deployment_joint_rollout(0, interval=0) is False
    for interval, phase in ((0, 0), (8, -1), (8, 8)):
        with pytest.raises(ValueError, match="interval/phase"):
            should_run_raw_auxiliary(0, warmup=0, interval=interval, phase=phase)
    with pytest.raises(ValueError, match="non-negative"):
        deployment_endpoint_losses(
            video_final=torch.zeros(1),
            video_x0=torch.zeros(1),
            action_final=torch.zeros(1),
            action_x0=torch.zeros(1),
            action_mask=torch.ones(1),
            action_weight=-1.0,
        )
