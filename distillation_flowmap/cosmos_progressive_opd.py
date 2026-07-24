"""Small, testable primitives for progressive Cosmos Stage 2 OPD."""

import math

import torch

from distillation_flowmap.cosmos_deployment_rollout import (
    should_run_deployment_joint_rollout,
    should_run_raw_auxiliary,
)


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


def constrain_cosmos_teacher_timestep_pair(
    timesteps,
    target_timesteps,
    *,
    t_min,
    t_max,
    num_train_timesteps,
):
    """Keep a raw-Cosmos velocity rollout inside its supported time window.

    Raw Cosmos latent velocities are only calibrated on an interior noise
    interval.  Both endpoints must therefore be clamped before a rollout is
    constructed; clamping only the teacher's reported time would pair a
    low-noise state with the wrong noise label.
    """
    if timesteps.shape != target_timesteps.shape:
        raise ValueError("timesteps and target_timesteps must have matching shapes")
    if int(num_train_timesteps) <= 0:
        raise ValueError("num_train_timesteps must be positive")
    t_min = float(t_min)
    t_max = float(t_max)
    if not (
        math.isfinite(t_min)
        and math.isfinite(t_max)
        and 0.0 < t_min < t_max < 1.0
    ):
        raise ValueError(
            "Cosmos raw teacher window must satisfy 0 < t_min < t_max < 1"
        )

    lower = t_min * int(num_train_timesteps)
    upper = t_max * int(num_train_timesteps)
    timesteps = timesteps.clamp(min=lower, max=upper)
    target_timesteps = target_timesteps.clamp(min=lower, max=upper)
    return timesteps, torch.minimum(target_timesteps, timesteps)


def build_cosmos_teacher_window_path(
    *,
    batch_size,
    num_frames,
    num_steps,
    t_min,
    t_max,
    num_train_timesteps,
    device,
    dtype,
):
    """Build a full high-to-low raw-Cosmos rollout in the valid teacher band."""
    if int(batch_size) <= 0 or int(num_frames) <= 0:
        raise ValueError("batch_size and num_frames must be positive")
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")
    if int(num_train_timesteps) <= 0:
        raise ValueError("num_train_timesteps must be positive")

    probe = torch.empty(
        (int(batch_size), int(num_frames)), device=device, dtype=dtype
    )
    start, end = constrain_cosmos_teacher_timestep_pair(
        torch.full_like(probe, float(t_max) * int(num_train_timesteps)),
        torch.full_like(probe, float(t_min) * int(num_train_timesteps)),
        t_min=t_min,
        t_max=t_max,
        num_train_timesteps=num_train_timesteps,
    )
    return build_uniform_timestep_path(start, end, num_steps)


def broadcast_joint_action_timesteps(video_t, video_r, *, action_frames):
    """Map a scalar video endpoint pair to every action token in the joint state."""
    if video_t.ndim != 2 or video_r.ndim != 2 or video_t.shape != video_r.shape:
        raise ValueError("video_t and video_r must have matching [B, F] shapes")
    if int(action_frames) <= 0:
        raise ValueError("action_frames must be positive")
    if not torch.equal(video_t, video_t[:, :1].expand_as(video_t)):
        raise ValueError("joint endpoint pairs must be constant across video frames")
    if not torch.equal(video_r, video_r[:, :1].expand_as(video_r)):
        raise ValueError("joint endpoint pairs must be constant across video frames")
    shape = (video_t.shape[0], int(action_frames))
    return video_t[:, :1].expand(shape), video_r[:, :1].expand(shape)


def compose_cosmos_endpoint_loss(
    video_endpoint_loss,
    action_endpoint_loss,
    *,
    action_endpoint_weight,
):
    """Add the action endpoint term only when that objective is enabled.

    In particular, avoid ``0 * NaN`` turning a video-only loss non-finite and
    keep the disabled action branch out of the autograd graph.
    """
    action_endpoint_weight = float(action_endpoint_weight)
    if (
        not math.isfinite(action_endpoint_weight)
        or action_endpoint_weight < 0.0
    ):
        raise ValueError(
            "Cosmos OPD action endpoint weight must be finite and non-negative"
        )
    if action_endpoint_weight == 0.0:
        return video_endpoint_loss
    return video_endpoint_loss + action_endpoint_weight * action_endpoint_loss


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
    focus_timestep=None,
    focus_target_timestep=None,
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

    focus_timestep = (
        float(num_train_timesteps)
        if focus_timestep is None
        else float(focus_timestep)
    )
    focus_target_timestep = (
        0.0 if focus_target_timestep is None else float(focus_target_timestep)
    )
    if not (
        math.isfinite(focus_timestep)
        and math.isfinite(focus_target_timestep)
        and 0.0 <= focus_target_timestep <= focus_timestep <= int(num_train_timesteps)
    ):
        raise ValueError(
            "focused endpoint pair must satisfy "
            "0 <= focus_target_timestep <= focus_timestep <= num_train_timesteps"
        )

    expanded_mask = focus_mask[:, None].expand_as(timesteps)
    full_t = torch.full_like(timesteps, focus_timestep)
    full_r = torch.full_like(target_timesteps, focus_target_timestep)
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


def select_progressive_training_objective(
    *,
    step,
    deployment_enabled,
    deployment_interval,
    raw_auxiliary_enabled,
    raw_auxiliary_warmup,
    raw_auxiliary_interval,
    raw_auxiliary_phase,
):
    """Choose exactly one optimizer objective for a progressive training step."""
    if deployment_enabled and should_run_deployment_joint_rollout(
        int(step), int(deployment_interval)
    ):
        return "deployment"
    if raw_auxiliary_enabled and should_run_raw_auxiliary(
        int(step),
        warmup=int(raw_auxiliary_warmup),
        interval=int(raw_auxiliary_interval),
        phase=int(raw_auxiliary_phase),
    ):
        return "raw_auxiliary"
    return "main"


def should_stop_training_at_step(*, step, stop_after_step):
    """Return whether a chunked run has reached its requested absolute step."""
    if stop_after_step in (None, "", 0, "0"):
        return False
    stop_after_step = int(stop_after_step)
    if stop_after_step <= 0:
        return False
    return int(step) >= stop_after_step


def center_spatial_crop_slices(height, width, *, crop_size):
    """Return a centered square crop, or the full image when no crop is needed."""
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError("height and width must be positive")
    if int(crop_size) <= 0 or int(crop_size) >= min(int(height), int(width)):
        return slice(0, int(height)), slice(0, int(width))
    crop_size = int(crop_size)
    top = (int(height) - crop_size) // 2
    left = (int(width) - crop_size) // 2
    return slice(top, top + crop_size), slice(left, left + crop_size)
