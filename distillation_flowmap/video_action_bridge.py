"""Utilities for deployment-aligned video-conditioned action supervision."""

import torch


def bridge_probability(
    step,
    *,
    warmup_end=500,
    mid_end=1500,
    start_probability=0.25,
    mid_probability=0.50,
    final_probability=0.75,
):
    """Return the piecewise-constant video-to-action bridge probability."""
    if warmup_end < 0 or mid_end < warmup_end:
        raise ValueError("bridge curriculum must satisfy 0 <= warmup_end <= mid_end")
    probabilities = (start_probability, mid_probability, final_probability)
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError("bridge probabilities must lie in [0, 1]")
    if step < warmup_end:
        return float(start_probability)
    if step < mid_end:
        return float(mid_probability)
    return float(final_probability)


def masked_action_teacher_forcing_loss(
    action_prediction,
    action_target,
    valid_mask=None,
):
    """Apply masked action teacher forcing under a generated-video condition."""
    squared_error = (
        action_prediction.float() - action_target.detach().float()
    ).pow(2)
    if valid_mask is None:
        return squared_error.mean()
    mask = valid_mask.to(device=squared_error.device, dtype=squared_error.dtype)
    while mask.ndim < squared_error.ndim:
        mask = mask.unsqueeze(-1)
    mask = torch.broadcast_to(mask, squared_error.shape)
    return (squared_error * mask).sum() / mask.sum().clamp(min=1.0)


def masked_action_x0_teacher_forcing_loss(
    predicted_velocity,
    noisy_action,
    clean_action,
    sigma,
    valid_mask=None,
):
    """Apply deployment-video action supervision in the main x0 parameterization."""
    predicted_x0 = (
        noisy_action.float()
        - sigma.to(device=noisy_action.device, dtype=torch.float32)
        * predicted_velocity.float()
    )
    return masked_action_teacher_forcing_loss(
        predicted_x0,
        clean_action,
        valid_mask,
    )
