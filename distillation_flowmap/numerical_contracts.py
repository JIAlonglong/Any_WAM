"""Pure numerical validation contracts shared by FlowMap training paths."""

from dataclasses import dataclass
import math
from typing import Callable, Mapping

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


@dataclass(frozen=True)
class TerminalPriorValidation:
    """Checks and coherent diagnostics for all terminal-prior sources."""

    checks: Mapping[str, TerminalPriorCheck]
    worst_source: str

    def diagnostic_tensors(self, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return metadata from one coherent worst normalized comparison."""
        worst = self.checks[self.worst_source]
        values = {
            "max_error": worst.max_error,
            "reference_scale": worst.reference_scale,
            "atol": worst.atol,
            "rtol": worst.rtol,
            "threshold": worst.threshold,
            "severity": {"quiet": 0.0, "warning": 1.0, "error": 2.0}[
                worst.severity
            ],
        }
        return {
            name: torch.tensor(
                value, device=reference.device, dtype=torch.float32
            )
            for name, value in values.items()
        }


def compare_terminal_prior(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    source_dtype: torch.dtype,
    warn_factor: float = 0.5,
    atol: float = FP32_SCHEDULER_ATOL,
) -> TerminalPriorCheck:
    """Compare a terminal state using source-dtype-aware fp32 tolerances.

    The effective threshold is ``atol + rtol * max(abs(expected))``.  Values
    up to ``warn_factor * threshold`` are quiet, values up to the full
    threshold produce a warning classification, and larger or non-finite
    values are errors.
    """
    if not math.isfinite(warn_factor) or not 0.0 <= warn_factor <= 1.0:
        raise ValueError("warn_factor must be finite and in [0, 1]")
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and non-negative")

    eps = float(torch.finfo(source_dtype).eps)
    atol = float(atol)
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


def _check_score(check: TerminalPriorCheck) -> tuple[int, float]:
    ratio = (
        check.max_error / check.threshold
        if math.isfinite(check.max_error)
        and math.isfinite(check.threshold)
        and check.threshold > 0.0
        else math.inf
    )
    return {"quiet": 0, "warning": 1, "error": 2}[check.severity], ratio


def _check_details(checks: Mapping[str, TerminalPriorCheck]) -> str:
    return "; ".join(
        (
            f"{name}: max_error={check.max_error:.3e}, "
            f"reference_scale={check.reference_scale:.3e}, "
            f"atol={check.atol:.3e}, rtol={check.rtol:.3e}, "
            f"threshold={check.threshold:.3e}, severity={check.severity}"
        )
        for name, check in checks.items()
    )


def validate_terminal_prior_sources(
    sources: Mapping[
        str, tuple[torch.Tensor, torch.Tensor, torch.dtype]
    ],
    *,
    config,
    error_context: str,
    warning_callback: Callable[[str], None] | None = None,
) -> TerminalPriorValidation:
    """Validate video/action sources using shared config semantics.

    ``OPD_DANCEOPD_TERMINAL_PRIOR_TOLERANCE`` remains an explicit absolute
    tolerance override. Source dtype epsilon contributes only the relative
    term, while the warning factor controls classification independently.
    """
    atol = float(
        getattr(
            config,
            "opd_danceopd_terminal_prior_tolerance",
            FP32_SCHEDULER_ATOL,
        )
    )
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError(
            "terminal-prior tolerance "
            "(OPD_DANCEOPD_TERMINAL_PRIOR_TOLERANCE) must be finite and "
            "non-negative"
        )
    warn_factor = float(
        getattr(config, "opd_danceopd_terminal_prior_warn_factor", 0.5)
    )
    if not math.isfinite(warn_factor) or not 0.0 <= warn_factor <= 1.0:
        raise ValueError(
            "terminal-prior warn_factor "
            "(OPD_DANCEOPD_TERMINAL_PRIOR_WARN_FACTOR) must be finite and "
            "in [0, 1]"
        )

    checks = {
        name: compare_terminal_prior(
            actual,
            expected,
            source_dtype=source_dtype,
            warn_factor=warn_factor,
            atol=atol,
        )
        for name, (actual, expected, source_dtype) in sources.items()
    }
    if not checks:
        raise ValueError("terminal-prior sources must not be empty")
    worst_source = max(checks, key=lambda name: _check_score(checks[name]))
    result = TerminalPriorValidation(
        checks=checks,
        worst_source=worst_source,
    )

    if not bool(getattr(config, "opd_danceopd_verify_terminal_prior", True)):
        return result

    details = _check_details(checks)
    if any(check.severity == "error" for check in checks.values()):
        raise RuntimeError(f"{error_context}: {details}")
    if (
        warning_callback is not None
        and any(check.severity == "warning" for check in checks.values())
    ):
        warning_callback(
            f"{error_context} is within the numerical warning band: {details}"
        )
    return result
