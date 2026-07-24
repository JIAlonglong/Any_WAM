"""Pure numerical validation contracts shared by FlowMap training paths."""

from dataclasses import dataclass
import math

import torch


# FlowMatch scheduler arithmetic is evaluated in fp32 even when its source
# state is lower precision.  The observed terminal residue is 1.192e-6, so the
# absolute floor must be strictly larger while remaining small enough to catch
# material corruption.
FP32_SCHEDULER_ATOL = 2.0e-6


@dataclass(frozen=True)
class TerminalPriorCheck:
    """Metadata describing a terminal-prior numerical comparison."""

    max_error: float
    reference_scale: float
    atol: float
    rtol: float
    threshold: float
    severity: str


def compare_terminal_prior(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    source_dtype: torch.dtype,
    warn_factor: float = 0.5,
) -> TerminalPriorCheck:
    """Compare a terminal state using source-dtype-aware fp32 tolerances.

    The effective threshold is ``atol + rtol * max(abs(expected))``.  Values
    up to ``warn_factor * threshold`` are quiet, values up to the full
    threshold produce a warning classification, and larger or non-finite
    values are errors.
    """
    if not math.isfinite(warn_factor) or not 0.0 <= warn_factor <= 1.0:
        raise ValueError("warn_factor must be finite and in [0, 1]")

    eps = float(torch.finfo(source_dtype).eps)
    atol = max(FP32_SCHEDULER_ATOL, eps)
    rtol = eps

    actual_fp32 = actual.detach().float()
    expected_fp32 = expected.detach().float()
    finite_inputs = bool(
        torch.isfinite(actual_fp32).all().item()
        and torch.isfinite(expected_fp32).all().item()
    )
    if finite_inputs:
        max_error = float((actual_fp32 - expected_fp32).abs().amax().item())
        reference_scale = float(expected_fp32.abs().amax().item())
    else:
        max_error = math.inf
        reference_scale = (
            float(expected_fp32.abs().amax().item())
            if bool(torch.isfinite(expected_fp32).all().item())
            else math.inf
        )

    threshold = atol + rtol * reference_scale
    if not finite_inputs or not math.isfinite(threshold) or max_error > threshold:
        severity = "error"
    elif max_error > warn_factor * threshold:
        severity = "warning"
    else:
        severity = "quiet"

    return TerminalPriorCheck(
        max_error=max_error,
        reference_scale=reference_scale,
        atol=atol,
        rtol=rtol,
        threshold=threshold,
        severity=severity,
    )
