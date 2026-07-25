import pytest
import torch

from distillation_flowmap.video_action_bridge import (
    bridge_probability,
    masked_action_x0_teacher_forcing_loss,
    masked_action_teacher_forcing_loss,
)


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (0, 0.25),
        (499, 0.25),
        (500, 0.50),
        (1499, 0.50),
        (1500, 0.75),
        (10000, 0.75),
    ],
)
def test_bridge_probability_curriculum(step, expected):
    assert bridge_probability(step) == expected


def test_masked_action_teacher_forcing_loss_uses_only_valid_actions():
    prediction = torch.tensor([[[[[1.0]], [[100.0]]]]])
    target = torch.tensor([[[[[1.0]], [[0.0]]]]])
    valid_mask = torch.tensor([[[[[1.0]], [[0.0]]]]])

    loss = masked_action_teacher_forcing_loss(
        prediction, target, valid_mask
    )

    assert loss.item() == pytest.approx(0.0)


def test_x0_bridge_loss_is_zero_for_exact_clean_action_reconstruction():
    clean_action = torch.tensor([[[[[2.0]], [[-1.0]]]]])
    noise = torch.tensor([[[[[6.0]], [[3.0]]]]])
    sigma = torch.tensor([0.25, 0.75]).view(1, 1, 2, 1, 1)
    noisy_action = (1.0 - sigma) * clean_action + sigma * noise
    exact_velocity = noise - clean_action

    loss = masked_action_x0_teacher_forcing_loss(
        exact_velocity,
        noisy_action,
        clean_action,
        sigma,
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_x0_bridge_loss_masks_invalid_action_tokens_after_reconstruction():
    clean_action = torch.zeros(1, 1, 2, 1, 1)
    noisy_action = torch.tensor([[[[[0.5]], [[100.0]]]]])
    sigma = torch.tensor([0.5, 1.0]).view(1, 1, 2, 1, 1)
    predicted_velocity = torch.tensor([[[[[1.0]], [[0.0]]]]])
    valid_mask = torch.tensor([[[[[1.0]], [[0.0]]]]])

    loss = masked_action_x0_teacher_forcing_loss(
        predicted_velocity,
        noisy_action,
        clean_action,
        sigma,
        valid_mask,
    )

    assert loss.item() == pytest.approx(0.0)


def test_bridge_configuration_rejects_invalid_curriculum():
    with pytest.raises(ValueError, match="curriculum"):
        bridge_probability(0, warmup_end=10, mid_end=5)
    with pytest.raises(ValueError, match="probabilities"):
        bridge_probability(0, start_probability=1.1)
