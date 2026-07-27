"""Distributed agreement helpers for non-finite training control flow."""

from collections.abc import Mapping

import torch
import torch.distributed as dist


NONFINITE_ORIGINS = (
    "video",
    "action",
    "teacher_or_gt",
    "opd_endpoint",
    "opd_compositional",
    "opd_action",
    "gradient",
)


def _as_bool(value) -> bool:
    if torch.is_tensor(value):
        return bool(value.detach().bool().all().item())
    return bool(value)


def all_ranks_finite(local_finite, *, device, dist_module=dist) -> bool:
    """Return true only when every participating rank reports finite state."""
    finite = torch.tensor(
        int(_as_bool(local_finite)),
        device=device,
        dtype=torch.int32,
    )
    if dist_module.is_available() and dist_module.is_initialized():
        dist_module.all_reduce(finite, op=dist_module.ReduceOp.MIN)
    return bool(finite.item())


def reduce_nonfinite_origins(
    local_flags: Mapping[str, bool],
    *,
    device,
    dist_module=dist,
) -> dict[str, bool]:
    """Return the union of named non-finite origins reported by all ranks."""
    unknown = set(local_flags) - set(NONFINITE_ORIGINS)
    if unknown:
        raise ValueError(f"unknown non-finite origins: {sorted(unknown)}")
    reduced = {}
    for origin in NONFINITE_ORIGINS:
        present = torch.tensor(
            int(_as_bool(local_flags.get(origin, False))),
            device=device,
            dtype=torch.int32,
        )
        if dist_module.is_available() and dist_module.is_initialized():
            dist_module.all_reduce(present, op=dist_module.ReduceOp.MAX)
        reduced[origin] = bool(present.item())
    return reduced


def initialize_nonfinite_safety(owner) -> None:
    """Initialize trainer-owned sticky window state and cumulative counters."""
    owner.skip_accumulation_window = False
    owner.nonfinite_skipped_windows = 0
    owner.nonfinite_origin_counters = {
        origin: 0 for origin in NONFINITE_ORIGINS
    }
    owner._nonfinite_window_origins = {
        origin: False for origin in NONFINITE_ORIGINS
    }


def record_nonfinite_origins(
    owner,
    local_flags: Mapping[str, bool],
    *,
    device,
) -> dict[str, bool]:
    """Synchronize and retain non-finite origins for the current window."""
    synchronized = reduce_nonfinite_origins(local_flags, device=device)
    for origin, present in synchronized.items():
        owner._nonfinite_window_origins[origin] = (
            owner._nonfinite_window_origins[origin] or present
        )
    owner.skip_accumulation_window = (
        owner.skip_accumulation_window or any(synchronized.values())
    )
    return synchronized


def accumulation_backward_allowed(owner) -> bool:
    """Return whether optional backward work may run in this window."""
    return not owner.skip_accumulation_window


def finish_accumulation_window(
    owner,
    *,
    total_norm,
    grad_branch_norms,
    update_ema_fn,
):
    """Collectively decide and either advance or discard one optimizer window."""
    local_gradient_finite = bool(
        torch.isfinite(total_norm.detach()).all().item()
    )
    gradient_finite = all_ranks_finite(
        local_gradient_finite, device=owner.device
    )
    if not gradient_finite:
        record_nonfinite_origins(
            owner, {"gradient": True}, device=owner.device
        )

    skipped = owner.skip_accumulation_window
    if skipped:
        owner.nonfinite_skipped_windows += 1
        for origin, present in owner._nonfinite_window_origins.items():
            if present:
                owner.nonfinite_origin_counters[origin] += 1
        total_norm = torch.zeros((), device=owner.device)
        grad_branch_norms = {}
    else:
        owner.optimizer.step()
        owner.lr_scheduler.step()
        update_ema_fn()

    owner.optimizer.zero_grad(set_to_none=True)
    owner.skip_accumulation_window = False
    owner._nonfinite_window_origins = {
        origin: False for origin in NONFINITE_ORIGINS
    }
    return skipped, total_norm, grad_branch_norms
