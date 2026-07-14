"""Small, testable primitives for progressive Cosmos Stage 2 OPD."""

import math

import torch


def _time_view(timesteps):
    return timesteps[:, None, :, None, None]


def build_uniform_timestep_path(timesteps, target_timesteps, num_steps):
    """Return the inclusive linear path `[N + 1, B, F]` from t to r."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if timesteps.shape != target_timesteps.shape:
        raise ValueError(
            "timesteps and target_timesteps must have the same [B, F] shape; "
            f"got {tuple(timesteps.shape)} and {tuple(target_timesteps.shape)}"
        )
    alpha = torch.linspace(
        0.0,
        1.0,
        num_steps + 1,
        device=timesteps.device,
        dtype=timesteps.dtype,
    )
    return timesteps.unsqueeze(0) + (
        target_timesteps - timesteps
    ).unsqueeze(0) * alpha.view(-1, 1, 1)


@torch.no_grad()
def rollout_velocity_field(x_t, timesteps, target_timesteps, *, num_steps, velocity_field):
    """Euler-integrate a frozen normalized-time vector field from `t` to `r`.

    `velocity_field(state, timestep)` is deliberately injected so this helper
    stays independent of the raw Cosmos adapter.  The terminal field is queried
    after the last update because endpoint denoising uses `v(x_r, r)`.
    """
    path = build_uniform_timestep_path(timesteps, target_timesteps, num_steps)
    current = x_t.detach()
    for step_index in range(num_steps):
        t_i = path[step_index]
        r_i = path[step_index + 1]
        velocity = velocity_field(current, t_i)
        if velocity.shape != current.shape:
            raise ValueError(
                "velocity_field returned an incompatible shape: "
                f"expected {tuple(current.shape)}, got {tuple(velocity.shape)}"
            )
        current = (
            current + velocity.to(current) * _time_view(r_i - t_i).to(current)
        ).detach()
    terminal_velocity = velocity_field(current, path[-1])
    if terminal_velocity.shape != current.shape:
        raise ValueError(
            "velocity_field returned an incompatible terminal shape: "
            f"expected {tuple(current.shape)}, got {tuple(terminal_velocity.shape)}"
        )
    return current, terminal_velocity, path


def apply_full_endpoint_focus(
    timesteps,
    target_timesteps,
    *,
    probability,
    num_train_timesteps,
):
    """Replace a sample-level subset of pairs with the full deployment path."""
    if timesteps.shape != target_timesteps.shape:
        raise ValueError("timesteps and target_timesteps must have matching shapes")
    if not math.isfinite(float(probability)) or not 0.0 <= float(probability) <= 1.0:
        raise ValueError("probability must be finite and in [0, 1]")
    if int(num_train_timesteps) <= 0:
        raise ValueError("num_train_timesteps must be positive")

    if probability <= 0.0:
        focus_mask = torch.zeros(
            timesteps.shape[0], device=timesteps.device, dtype=torch.bool
        )
    elif probability >= 1.0:
        focus_mask = torch.ones(
            timesteps.shape[0], device=timesteps.device, dtype=torch.bool
        )
    else:
        focus_mask = torch.rand(timesteps.shape[0], device=timesteps.device) < probability

    expanded_mask = focus_mask[:, None].expand_as(timesteps)
    full_t = torch.full_like(timesteps, float(num_train_timesteps))
    full_r = torch.zeros_like(target_timesteps)
    return (
        torch.where(expanded_mask, full_t, timesteps),
        torch.where(expanded_mask, full_r, target_timesteps),
        focus_mask,
    )


def should_run_standalone_opd(*, step, warmup_steps, interval):
    """Return whether this optimizer update is reserved for a full OPD loss."""
    if int(warmup_steps) < 0:
        raise ValueError("warmup_steps must be non-negative")
    if int(interval) <= 0:
        raise ValueError("interval must be positive")
    return int(step) >= int(warmup_steps) and int(step) % int(interval) == 0
