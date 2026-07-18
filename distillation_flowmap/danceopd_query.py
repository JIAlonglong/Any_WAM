"""Small, model-agnostic utilities for DanceOPD-style field queries."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def legacy_danceopd_query_floor(
    *,
    num_train_timesteps: int | float,
    legacy_rollout_steps: int = 16,
) -> float:
    """Return the lowest pre-update query time from legacy DanceOPD.

    The established 16-step implementation stores one state before each
    Euler update, so its lowest query is one interval above zero.  Mixed
    short rollouts use this same positive floor for their post-update query
    candidate: raw Cosmos velocity extraction is singular at ``t=0``.
    """
    total = float(num_train_timesteps)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("num_train_timesteps must be finite and positive")
    if legacy_rollout_steps <= 0:
        raise ValueError("legacy_rollout_steps must be positive")
    return total / float(legacy_rollout_steps)


def require_query_timestep_floor(
    timestep: torch.Tensor,
    *,
    safe_floor: float,
    tolerance: float = 1e-4,
) -> float:
    """Reject an unsafe raw-Cosmos teacher query before it is evaluated."""
    if timestep.ndim != 2:
        raise ValueError("timestep must have shape [batch, frames]")
    floor = float(safe_floor)
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("safe_floor must be finite and positive")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    actual_min = float(timestep.detach().float().amin().item())
    if not math.isfinite(actual_min) or actual_min < floor - tolerance:
        raise RuntimeError(
            "Cosmos DanceOPD teacher query fell below the safe floor: "
            f"min_t={actual_min:.6g}, safe_floor={floor:.6g}"
        )
    return actual_min


def interpolate_raw_flow_segment(
    segment_start: torch.Tensor,
    velocity: torch.Tensor,
    start_timestep: torch.Tensor,
    target_timestep: torch.Tensor,
    *,
    num_train_timesteps: int | float,
    t_dim: int = 2,
) -> torch.Tensor:
    """Interpolate the final Euler segment at a raw positive flow time.

    DanceOPD evolves states using ``sigma=t/T`` directly.  Interpolating the
    final student Euler segment keeps a short mixed rollout on-policy while
    avoiding raw Cosmos's singular ``t=0`` velocity extraction.
    """
    if segment_start.shape != velocity.shape:
        raise ValueError("segment_start and velocity must have matching shapes")
    if segment_start.ndim <= t_dim:
        raise ValueError("t_dim must index a segment_start dimension")
    if start_timestep.ndim != 2 or target_timestep.ndim != 2:
        raise ValueError("timesteps must have shape [batch, frames]")
    if start_timestep.shape != target_timestep.shape:
        raise ValueError("start_timestep and target_timestep must have matching shapes")
    batch_size, frames = start_timestep.shape
    if segment_start.shape[0] != batch_size or segment_start.shape[t_dim] != frames:
        raise ValueError("timestep shape does not match segment_start")
    total = float(num_train_timesteps)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("num_train_timesteps must be finite and positive")

    sigma_shape = [1] * segment_start.ndim
    sigma_shape[0] = batch_size
    sigma_shape[t_dim] = frames
    start_sigma = (
        start_timestep.to(device=segment_start.device, dtype=segment_start.dtype)
        / total
    ).reshape(batch_size, frames).view(sigma_shape)
    target_sigma = (
        target_timestep.to(device=segment_start.device, dtype=segment_start.dtype)
        / total
    ).reshape(batch_size, frames).view(sigma_shape)
    return segment_start + velocity * (target_sigma - start_sigma)


def append_post_update_trajectory_state(
    states: list[torch.Tensor],
    timesteps: list[torch.Tensor],
    *,
    state: torch.Tensor,
    timestep: torch.Tensor,
) -> None:
    if len(states) != len(timesteps):
        raise ValueError("trajectory states and timesteps must have the same length")
    states.append(state.detach().clone())
    timesteps.append(timestep.detach().clone())


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
