import pytest
import torch

from distillation_flowmap.video_action_bridge import (
    bridge_probability,
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


def test_bridge_configuration_rejects_invalid_curriculum():
    with pytest.raises(ValueError, match="curriculum"):
        bridge_probability(0, warmup_end=10, mid_end=5)
    with pytest.raises(ValueError, match="probabilities"):
        bridge_probability(0, start_probability=1.1)
