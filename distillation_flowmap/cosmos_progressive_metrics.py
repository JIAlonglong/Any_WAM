"""Small metric primitives shared by cached Cosmos progressive evaluations."""

from __future__ import annotations

from collections import defaultdict

import torch


def build_paired_eval_timesteps(*, batch_size, video_frames, action_frames, t, r, device):
    """Build endpoint times without conflating Cosmos video and action horizons."""
    batch_size = int(batch_size)
    video_t = torch.full((batch_size, int(video_frames)), float(t), device=device)
    video_r = torch.full_like(video_t, float(r))
    action_t = torch.full((batch_size, int(action_frames)), float(t), device=device)
    action_r = torch.full_like(action_t, float(r))
    return video_t, video_r, action_t, action_r


def denoised_endpoint(state, velocity, sigma):
    """Return x0 = x_r - sigma_r * v(x_r, r) for [B, C, F, H, W] tensors."""
    if state.shape != velocity.shape:
        raise ValueError("state and velocity must have matching shapes")
    if sigma.ndim != 2 or sigma.shape != (state.shape[0], state.shape[2]):
        raise ValueError(
            "sigma must have shape [B, F] matching state; "
            f"got {tuple(sigma.shape)} for {tuple(state.shape)}"
        )
    return state - sigma[:, None, :, None, None].to(state) * velocity


def macro_average_by_task(per_task):
    """Macro-average scalar metrics so each task contributes equally."""
    if not per_task:
        return {}
    sums = defaultdict(float)
    counts = defaultdict(int)
    for metrics in per_task.values():
        for name, value in metrics.items():
            sums[str(name)] += float(value)
            counts[str(name)] += 1
    return {
        name: sums[name] / counts[name]
        for name in sorted(sums)
    }
