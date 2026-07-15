from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NormalizedFocalWeights:
    weights: torch.Tensor
    threshold: torch.Tensor
    hard_ratio: torch.Tensor
    weight_std: torch.Tensor
    error_q50: torch.Tensor
    error_q70: torch.Tensor
    error_q90: torch.Tensor


def piecewise_linear_scale(step, *, start, end, hold_steps, ramp_steps):
    if hold_steps < 0:
        raise ValueError("hold_steps must be non-negative")
    if ramp_steps < 0:
        raise ValueError("ramp_steps must be non-negative")
    if step < hold_steps:
        return float(start)
    if ramp_steps == 0:
        return float(end)
    progress = min(1.0, float(step - hold_steps + 1) / float(ramp_steps))
    return float(start) + (float(end) - float(start)) * progress


def _as_scalar_tensor(value, *, device, dtype):
    return torch.as_tensor(value, device=device, dtype=dtype)


def compute_normalized_focal_weights(
    token_error,
    token_reference,
    *,
    threshold=None,
    threshold_quantile=0.70,
    eps=1e-5,
    alpha=1.0,
    temperature=0.10,
    min_weight=0.5,
    max_weight=1.8,
):
    """Build mean-one focal weights that emphasize high-error OPD tokens."""
    if token_error.shape != token_reference.shape:
        raise ValueError(
            "token_error and token_reference must have the same shape, got "
            f"{tuple(token_error.shape)} and {tuple(token_reference.shape)}"
        )
    if eps <= 0:
        raise ValueError("eps must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if min_weight <= 0 or max_weight <= 0:
        raise ValueError("min_weight and max_weight must be positive")
    if min_weight > max_weight:
        raise ValueError("min_weight must be <= max_weight")
    if not 0.0 <= threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile must be in [0, 1]")

    ratio = token_error.detach().float() / (token_reference.detach().float() + eps)
    flat = ratio.flatten()
    q50 = torch.quantile(flat, 0.50)
    q70 = torch.quantile(flat, 0.70)
    q90 = torch.quantile(flat, 0.90)
    if threshold is None:
        threshold_tensor = torch.quantile(flat, threshold_quantile)
    else:
        threshold_tensor = _as_scalar_tensor(
            threshold,
            device=ratio.device,
            dtype=ratio.dtype,
        )

    alpha_tensor = _as_scalar_tensor(alpha, device=ratio.device, dtype=ratio.dtype)
    min_tensor = _as_scalar_tensor(min_weight, device=ratio.device, dtype=ratio.dtype)
    max_tensor = _as_scalar_tensor(max_weight, device=ratio.device, dtype=ratio.dtype)
    hard_score = torch.sigmoid((ratio - threshold_tensor) / temperature)
    weights = 1.0 + alpha_tensor * (hard_score - hard_score.mean())
    weights = weights.clamp(min=min_tensor, max=max_tensor)
    weights = weights / weights.mean().clamp_min(1e-6)
    weights = weights.clamp(min=min_tensor, max=max_tensor)
    weights = weights / weights.mean().clamp_min(1e-6)

    return NormalizedFocalWeights(
        weights=weights.to(dtype=token_error.dtype),
        threshold=threshold_tensor.detach(),
        hard_ratio=(ratio > threshold_tensor).float().mean(),
        weight_std=weights.float().std(unbiased=False),
        error_q50=q50.detach(),
        error_q70=q70.detach(),
        error_q90=q90.detach(),
    )
