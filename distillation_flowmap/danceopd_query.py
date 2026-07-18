"""Small, model-agnostic utilities for DanceOPD-style field queries."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sample_low_noise_query_indices(
    *,
    n_states: int,
    batch_size: int,
    alpha: float,
    beta: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample one low-noise trajectory index per batch item.

    Trajectories are ordered from high to low noise.  A Beta(5, 2) draw
    therefore favors later, semantic-side states without always selecting the
    terminal state.
    """
    if n_states <= 0:
        raise ValueError("n_states must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")

    distribution = torch.distributions.Beta(
        torch.tensor(float(alpha), device=device),
        torch.tensor(float(beta), device=device),
    )
    normalized_indices = distribution.sample((batch_size,))
    return (normalized_indices * n_states).to(torch.long).clamp_(0, n_states - 1)


def sample_semantic_query_indices(
    *,
    rollout_steps: int,
    batch_size: int,
    alpha: float,
    beta: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample only post-update states from a terminal-to-clean rollout.

    Index zero is the initial pure-noise state.  Excluding it makes a one-step
    rollout query its denoised endpoint, rather than querying pure noise.
    """
    return sample_low_noise_query_indices(
        n_states=rollout_steps,
        batch_size=batch_size,
        alpha=alpha,
        beta=beta,
        device=device,
    ) + 1


def select_per_sample_trajectory_state(
    trajectory: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Return one state per sample from a ``[steps, batch, ...]`` trajectory."""
    if trajectory.ndim < 2:
        raise ValueError("trajectory must have shape [steps, batch, ...]")
    if indices.ndim != 1 or indices.shape[0] != trajectory.shape[1]:
        raise ValueError("indices must have shape [batch]")
    if indices.numel() and (
        int(indices.min()) < 0 or int(indices.max()) >= trajectory.shape[0]
    ):
        raise IndexError("query index is outside the trajectory")

    batch_indices = torch.arange(trajectory.shape[1], device=trajectory.device)
    return trajectory[indices.to(device=trajectory.device, dtype=torch.long), batch_indices]


def direct_velocity_mse(
    student_velocity: torch.Tensor,
    teacher_velocity: torch.Tensor,
) -> torch.Tensor:
    """DanceOPD's unweighted local velocity MSE with a detached teacher field."""
    if student_velocity.shape != teacher_velocity.shape:
        raise ValueError("student and teacher velocity shapes must match")
    return F.mse_loss(student_velocity.float(), teacher_velocity.detach().float())


def denoised_endpoint_mse(
    student_state: torch.Tensor,
    student_velocity: torch.Tensor,
    teacher_state: torch.Tensor,
    teacher_velocity: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Match denoised endpoints while keeping the teacher target detached."""
    student_x0 = student_state - sigma.to(student_velocity) * student_velocity
    teacher_x0 = teacher_state.detach() - sigma.to(teacher_velocity) * teacher_velocity.detach()
    return F.mse_loss(student_x0.float(), teacher_x0.float())
