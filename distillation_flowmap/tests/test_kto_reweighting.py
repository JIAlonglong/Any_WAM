import torch

from distillation_flowmap.kto_reweighting import (
    compute_normalized_focal_weights,
    piecewise_linear_scale,
)


def test_normalized_focal_weights_keep_mean_one_and_emphasize_hard_tokens():
    token_error = torch.tensor([[0.05, 0.10, 0.50, 1.00]], dtype=torch.float32)
    token_ref = torch.ones_like(token_error)

    result = compute_normalized_focal_weights(
        token_error,
        token_ref,
        threshold=torch.tensor(0.25),
        alpha=1.0,
        temperature=0.10,
        min_weight=0.5,
        max_weight=1.8,
    )

    assert torch.isclose(result.weights.mean(), torch.tensor(1.0), atol=1e-6)
    assert result.weights[0, 3] > result.weights[0, 0]
    assert result.weights.min() >= 0.5
    assert result.weights.max() <= 1.8
    assert result.hard_ratio.item() == 0.5


def test_normalized_focal_weights_handle_flat_errors_without_nan():
    token_error = torch.zeros((2, 3), dtype=torch.float32)
    token_ref = torch.ones_like(token_error)

    result = compute_normalized_focal_weights(
        token_error,
        token_ref,
        threshold=torch.tensor(0.0),
        alpha=1.0,
        temperature=0.10,
        min_weight=0.5,
        max_weight=1.8,
    )

    assert torch.isfinite(result.weights).all()
    assert torch.isclose(result.weights.mean(), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(result.weight_std, torch.tensor(0.0), atol=1e-6)


def test_piecewise_linear_scale_holds_then_ramps_to_end_value():
    assert piecewise_linear_scale(0, start=0.65, end=1.0, hold_steps=25, ramp_steps=25) == 0.65
    assert piecewise_linear_scale(24, start=0.65, end=1.0, hold_steps=25, ramp_steps=25) == 0.65
    assert piecewise_linear_scale(49, start=0.65, end=1.0, hold_steps=25, ramp_steps=25) == 1.0
    assert piecewise_linear_scale(60, start=0.65, end=1.0, hold_steps=25, ramp_steps=25) == 1.0
