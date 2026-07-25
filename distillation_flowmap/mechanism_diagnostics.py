"""Backend-neutral, finite-safe metrics for post-update mechanism probes.

The actual probe that supplies the tensors is intentionally outside this module.
Keeping the accounting here makes the same metric definition usable for every
policy backend and prevents rank-local means from leaking into experiment logs.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import torch


MECHANISM_RATIO_EPS = 1e-8
MECHANISM_NEAR_ZERO_THRESHOLD = 1e-10
MECHANISM_EXPLOSION_THRESHOLD = 1e6
MECHANISM_DOMINANCE_THRESHOLD = 100.0
MECHANISM_FATAL_EXIT_CODE = 86


def squared_l2_per_sample(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return (lhs.float() - rhs.float()).flatten(1).square().sum(1)


def mse_per_sample(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return (lhs.float() - rhs.float()).flatten(1).square().mean(1)


def masked_action_mse_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    squared_error = (prediction.float() - target.float()).square()
    if mask is None:
        return squared_error.flatten(1).mean(1)
    expanded_mask = torch.broadcast_to(mask.to(squared_error), squared_error.shape)
    numerator = (squared_error * expanded_mask).flatten(1).sum(1)
    denominator = expanded_mask.flatten(1).sum(1).clamp_min(1)
    return numerator / denominator


def _expanded_video_frame_mask(
    mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    expanded_mask = mask.to(device=reference.device, dtype=reference.dtype)
    if expanded_mask.ndim == 2 and reference.ndim >= 3:
        expanded_mask = expanded_mask.reshape(
            expanded_mask.shape[0],
            1,
            expanded_mask.shape[1],
            *([1] * (reference.ndim - 3)),
        )
    return torch.broadcast_to(expanded_mask, reference.shape)


def masked_video_squared_l2_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    squared_error = (prediction.float() - target.float()).square()
    if mask is None:
        return squared_error.flatten(1).sum(1)
    expanded_mask = _expanded_video_frame_mask(mask, squared_error)
    return (squared_error * expanded_mask).flatten(1).sum(1)


def masked_video_mse_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    squared_error = (prediction.float() - target.float()).square()
    if mask is None:
        return squared_error.flatten(1).mean(1)
    expanded_mask = _expanded_video_frame_mask(mask, squared_error)
    numerator = (squared_error * expanded_mask).flatten(1).sum(1)
    denominator = expanded_mask.flatten(1).sum(1).clamp_min(1)
    return numerator / denominator


def _finite_safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    eps: float = MECHANISM_RATIO_EPS,
) -> torch.Tensor:
    """Divide with a finite zero sentinel for contaminated diagnostic samples."""
    finite_inputs = torch.isfinite(numerator) & torch.isfinite(denominator)
    safe_numerator = torch.where(finite_inputs, numerator, torch.zeros_like(numerator))
    safe_denominator = torch.where(
        finite_inputs, denominator.clamp_min(eps), torch.ones_like(denominator)
    )
    return safe_numerator / safe_denominator


def compute_mechanism_metric_samples(
    *,
    teacher_continuation_video: torch.Tensor,
    same_prior_teacher_endpoint_video: torch.Tensor,
    direct_route_video: torch.Tensor,
    composed_route_video: torch.Tensor,
    teacher_endpoint_action: torch.Tensor,
    action_student_context: torch.Tensor,
    action_teacher_video_context: torch.Tensor,
    action_teacher_joint_context: torch.Tensor | None,
    action_mask: torch.Tensor | None,
    action_gt_context: torch.Tensor | None = None,
    student_field_video: torch.Tensor | None = None,
    teacher_field_video: torch.Tensor | None = None,
    video_frame_mask: torch.Tensor | None = None,
    teacher_joint_available: bool = True,
    shared_state_verified: bool = False,
    same_prior_verified: bool = False,
    effective_teacher_steps: int | None = None,
) -> dict[str, torch.Tensor]:
    """Return per-sample metrics; callers retain raw non-finite values for validity."""
    anchor = masked_video_squared_l2_per_sample(
        teacher_continuation_video,
        same_prior_teacher_endpoint_video,
        video_frame_mask,
    )
    anchor_mse = masked_video_mse_per_sample(
        teacher_continuation_video,
        same_prior_teacher_endpoint_video,
        video_frame_mask,
    )
    comp = masked_video_squared_l2_per_sample(
        direct_route_video, composed_route_video, video_frame_mask
    )
    comp_mse = masked_video_mse_per_sample(
        direct_route_video, composed_route_video, video_frame_mask
    )
    action_error_student = masked_action_mse_per_sample(
        action_student_context, teacher_endpoint_action, action_mask
    )
    action_error_teacher_video = masked_action_mse_per_sample(
        action_teacher_video_context, teacher_endpoint_action, action_mask
    )
    oracle_gain = action_error_student - action_error_teacher_video
    result = {
        "mechanism/g_anchor": anchor,
        "mechanism/g_anchor_mse": anchor_mse,
        "mechanism/g_comp": comp,
        "mechanism/g_comp_mse": comp_mse,
        # Direction is intentionally anchor / comp, not its reciprocal.
        "mechanism/g_anchor_to_comp_ratio": _finite_safe_divide(
            anchor_mse, comp_mse
        ),
        "mechanism/video_endpoint_error": masked_video_squared_l2_per_sample(
            direct_route_video,
            same_prior_teacher_endpoint_video,
            video_frame_mask,
        ),
        "mechanism/action_error_student_context": action_error_student,
        "mechanism/action_error_teacher_video_context": action_error_teacher_video,
        "mechanism/video_to_action_oracle_gain": oracle_gain,
        "mechanism/video_to_action_recoverable_fraction": _finite_safe_divide(
            oracle_gain.clamp_min(0), action_error_student
        ),
        "mechanism/g_anchor_near_zero": (
            anchor_mse <= MECHANISM_NEAR_ZERO_THRESHOLD
        ).to(anchor_mse),
        "mechanism/g_comp_near_zero": (
            comp_mse <= MECHANISM_NEAR_ZERO_THRESHOLD
        ).to(comp_mse),
        "mechanism/g_anchor_exploded": (
            anchor_mse >= MECHANISM_EXPLOSION_THRESHOLD
        ).to(anchor_mse),
        "mechanism/g_comp_exploded": (
            comp_mse >= MECHANISM_EXPLOSION_THRESHOLD
        ).to(comp_mse),
        "mechanism/branch_dominance": (
            (_finite_safe_divide(anchor_mse, comp_mse) >= MECHANISM_DOMINANCE_THRESHOLD)
            | (_finite_safe_divide(comp_mse, anchor_mse) >= MECHANISM_DOMINANCE_THRESHOLD)
        ).to(anchor_mse),
        "mechanism/teacher_joint_available": torch.full_like(
            action_error_student, float(bool(teacher_joint_available))
        ),
        "mechanism/shared_state_verified": torch.full_like(
            anchor, float(bool(shared_state_verified))
        ),
        "mechanism/same_prior_verified": torch.full_like(
            anchor, float(bool(same_prior_verified))
        ),
        "mechanism/effective_teacher_steps_verified": torch.full_like(
            anchor,
            float(
                type(effective_teacher_steps) is int
                and effective_teacher_steps == 8
            ),
        ),
    }
    if action_gt_context is not None:
        result["mechanism/action_error_gt_context"] = masked_action_mse_per_sample(
            action_gt_context, teacher_endpoint_action, action_mask
        )
    if student_field_video is not None and teacher_field_video is not None:
        result["mechanism/video_field_match_error"] = masked_video_mse_per_sample(
            student_field_video, teacher_field_video, video_frame_mask
        )

    if teacher_joint_available:
        if action_teacher_joint_context is None:
            raise ValueError(
                "teacher_joint_available=True requires action_teacher_joint_context"
            )
        action_error_teacher_joint = masked_action_mse_per_sample(
            action_teacher_joint_context, teacher_endpoint_action, action_mask
        )
        result.update(
            {
                "mechanism/action_error_teacher_joint_context": action_error_teacher_joint,
                "mechanism/video_to_action_full_joint_gain": action_error_student
                - action_error_teacher_joint,
                "mechanism/video_to_action_residual_action_gap": action_error_teacher_video
                - action_error_teacher_joint,
            }
        )
    else:
        unavailable = action_error_student.new_empty((0,))
        result.update(
            {
                "mechanism/action_error_teacher_joint_context": unavailable,
                "mechanism/video_to_action_full_joint_gain": unavailable,
                "mechanism/video_to_action_residual_action_gap": unavailable,
            }
        )
    return result


def pack_finite_metric_stats(
    samples: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Pack per-metric finite sums/counts plus whole-batch validity indicators."""
    if not samples:
        raise ValueError("mechanism diagnostic samples must not be empty")
    stats: dict[str, torch.Tensor] = {}
    finite_groups = []
    reference = next(iter(samples.values()))
    for name, values in samples.items():
        finite = torch.isfinite(values)
        finite_groups.append(finite.flatten())
        stats[f"{name}_sum"] = torch.where(
            finite, values, torch.zeros_like(values)
        ).sum()
        stats[f"{name}_count"] = finite.sum().to(dtype=values.dtype)

    all_finite = torch.cat(finite_groups).all()
    batch_count = reference.shape[0]
    stats["diagnostic_batch_count"] = torch.as_tensor(
        float(batch_count), device=reference.device, dtype=reference.dtype
    )
    stats["diagnostic_valid_count"] = torch.as_tensor(
        float(batch_count) if bool(all_finite) else 0.0,
        device=reference.device,
        dtype=reference.dtype,
    )
    return stats


def reduce_metric_stats(
    stats: Mapping[str, torch.Tensor],
    *,
    dist_module=None,
) -> dict[str, torch.Tensor]:
    """Perform exactly one packed SUM collective, preserving a stable key order."""
    if dist_module is None:
        import torch.distributed as dist_module
    keys = sorted(stats)
    packed = torch.stack([stats[key].detach().float() for key in keys])
    if dist_module.is_available() and dist_module.is_initialized():
        dist_module.all_reduce(packed, op=dist_module.ReduceOp.SUM)
    return {key: packed[index] for index, key in enumerate(keys)}


def means_from_reduced_stats(stats: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Reconstruct finite global means from global sum/counts."""
    means: dict[str, float] = {}
    for name in sorted(stats):
        if not name.endswith("_sum"):
            continue
        metric_name = name.removesuffix("_sum")
        count = float(stats[f"{metric_name}_count"].item())
        total = float(stats[name].item())
        means[metric_name] = total / count if count > 0 else 0.0
        means[f"{metric_name}_finite_count"] = count
        means[f"{metric_name}_available"] = float(count > 0)
    batch_count = float(stats.get("diagnostic_batch_count", torch.tensor(0.0)).item())
    valid_count = float(stats.get("diagnostic_valid_count", torch.tensor(0.0)).item())
    teacher_joint_available = means.get(
        "mechanism/teacher_joint_available", 1.0
    )
    capability_gated = (
        {
            "mechanism/action_error_teacher_joint_context",
            "mechanism/video_to_action_full_joint_gain",
            "mechanism/video_to_action_residual_action_gap",
        }
        if teacher_joint_available == 0.0
        else set()
    )
    every_metric_has_samples = all(
        (
            float(stats[f"{name.removesuffix('_sum')}_count"].item()) > 0
            or name.removesuffix("_sum") in capability_gated
        )
        for name in stats
        if name.endswith("_sum")
    )
    means["mechanism/diagnostic_valid"] = (
        valid_count / batch_count
        if batch_count > 0 and valid_count == batch_count and every_metric_has_samples
        else 0.0
    )
    return means


class MechanismDiagnosticScheduler:
    """Runs an interval diagnostic after, never before, a successful update."""

    def __init__(self, interval: int):
        if interval <= 0:
            raise ValueError("mechanism diagnostic interval must be positive")
        self.interval = int(interval)
        self.pending = False
        self._last_run_step: int | None = None

    def observe(self, *, completed_step: int, optimizer_succeeded: bool) -> bool:
        due = completed_step > 0 and completed_step % self.interval == 0
        if due and not optimizer_succeeded:
            self.pending = True
            return False
        if not optimizer_succeeded:
            return False
        if not (due or self.pending) or completed_step == self._last_run_step:
            return False
        self.pending = False
        self._last_run_step = completed_step
        return True

    def needs_snapshot(self, *, completed_step: int) -> bool:
        due = completed_step > 0 and completed_step % self.interval == 0
        return bool(
            (due or self.pending) and completed_step != self._last_run_step
        )


def _clone_diagnostic_value_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return type(value)(
            (key, _clone_diagnostic_value_to_cpu(item))
            for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(_clone_diagnostic_value_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_clone_diagnostic_value_to_cpu(item) for item in value]
    return value


def capture_diagnostic_snapshot_if_due(
    scheduler: MechanismDiagnosticScheduler,
    batch,
    *,
    completed_step: int,
):
    """Capture an immutable CPU snapshot only when this update may run a probe."""
    if not scheduler.needs_snapshot(completed_step=completed_step):
        return None
    return _clone_diagnostic_value_to_cpu(batch)


def materialize_diagnostic_snapshot(snapshot, *, device, _preserve_cpu=False):
    """Move model inputs to ``device`` while preserving raw-worker payloads."""
    if torch.is_tensor(snapshot):
        return snapshot if _preserve_cpu else snapshot.to(device=device)
    if isinstance(snapshot, Mapping):
        return type(snapshot)(
            (
                key,
                materialize_diagnostic_snapshot(
                    item,
                    device=device,
                    _preserve_cpu=(
                        _preserve_cpu
                        or (isinstance(key, str) and key.startswith("raw_"))
                    ),
                ),
            )
            for key, item in snapshot.items()
        )
    if isinstance(snapshot, tuple):
        return tuple(
            materialize_diagnostic_snapshot(
                item, device=device, _preserve_cpu=_preserve_cpu
            )
            for item in snapshot
        )
    if isinstance(snapshot, list):
        return [
            materialize_diagnostic_snapshot(
                item, device=device, _preserve_cpu=_preserve_cpu
            )
            for item in snapshot
        ]
    return snapshot


def fatal_mechanism_process_exit(error: BaseException, *, distributed: bool) -> None:
    """Make TorchElastic terminate every peer after an irrecoverable rank-local fault."""
    message = f"MECHANISM_FATAL_EXIT: {type(error).__name__}: {error}"
    print(message, file=sys.stderr, flush=True)
    if distributed:
        os._exit(MECHANISM_FATAL_EXIT_CODE)
    raise RuntimeError(message) from error


def diagnostic_seed(
    base_seed: int,
    diagnostic_index: int,
    batch_index: int,
    rank: int,
) -> int:
    return int(base_seed) + int(diagnostic_index) * 1009 + int(batch_index) * 97 + int(rank)


@contextmanager
def diagnostic_runtime(*, seed: int, models: tuple[torch.nn.Module, ...]) -> Iterator[None]:
    """Temporarily seed a no-grad probe and restore RNG plus model modes exactly."""
    torch_state = torch.random.get_rng_state()
    python_state = random.getstate()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    model_modes = [model.training for model in models]
    try:
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for model in models:
            model.eval()
        with torch.no_grad():
            yield
    finally:
        for model, was_training in zip(models, model_modes):
            model.train(was_training)
        torch.random.set_rng_state(torch_state)
        random.setstate(python_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def fan_out_mechanism_metrics(
    metrics: Mapping[str, float],
    *,
    step: int,
    rank: int,
    tb_writer,
    wandb_module,
    jsonl_path: str | Path,
) -> None:
    """Write one canonical finite mapping to W&B offline, TensorBoard, and JSONL."""
    if rank != 0:
        return
    canonical = {key: float(value) for key, value in metrics.items()}
    if any(not key.startswith("mechanism/") for key in canonical):
        raise ValueError("mechanism fan-out only accepts mechanism/* metrics")
    if any(not math.isfinite(value) for value in canonical.values()):
        raise ValueError("mechanism fan-out requires finite metrics")
    if wandb_module is not None and getattr(wandb_module, "run", None) is not None:
        mode = getattr(getattr(wandb_module.run, "settings", None), "mode", None)
        if mode != "offline":
            raise RuntimeError("mechanism diagnostics require W&B offline mode")
        wandb_module.log(canonical, step=step)
    if tb_writer is not None:
        for key, value in canonical.items():
            tb_writer.add_scalar(key, value, step)
        tb_writer.flush()
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": int(step), **canonical}, sort_keys=True) + "\n")
