"""Reusable numerical diagnostics for offline rollout evaluation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch


def _as_unit_float_video(video: np.ndarray) -> np.ndarray:
    array = np.asarray(video)
    if array.ndim != 4:
        raise ValueError(
            "Decoded video must have shape [frames, height, width, channels], "
            f"got {array.shape}."
        )
    if array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Decoded video must have 1, 3, or 4 channels, got {array.shape}.")
    if np.issubdtype(array.dtype, np.integer):
        scale = float(np.iinfo(array.dtype).max)
        array = array.astype(np.float32) / scale
    else:
        array = array.astype(np.float32, copy=False)
        if array.size and float(np.nanmax(array)) > 1.5:
            array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def decoded_video_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    include_ssim: bool = True,
) -> dict[str, float]:
    """Measure decoded-frame fidelity and motion consistency in [0, 1] space."""
    prediction = _as_unit_float_video(prediction)
    target = _as_unit_float_video(target)
    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target videos must have the same shape, got "
            f"{prediction.shape} and {target.shape}."
        )

    difference = prediction - target
    pixel_mse = float(np.mean(np.square(difference), dtype=np.float64))
    metrics = {
        "pixel_mse": pixel_mse,
        "pixel_l1": float(np.mean(np.abs(difference), dtype=np.float64)),
        # A bounded representation keeps JSON valid for an exactly matching video.
        "psnr": float(-10.0 * math.log10(max(pixel_mse, 1e-12))),
    }
    if prediction.shape[0] > 1:
        temporal_difference = np.diff(prediction, axis=0) - np.diff(target, axis=0)
        metrics["temporal_difference_mse"] = float(
            np.mean(np.square(temporal_difference), dtype=np.float64)
        )

    if include_ssim:
        try:
            from skimage.metrics import structural_similarity
        except ImportError:
            structural_similarity = None
        if structural_similarity is not None:
            height, width = prediction.shape[1:3]
            win_size = min(7, height, width)
            if win_size % 2 == 0:
                win_size -= 1
            if win_size >= 3:
                values = [
                    structural_similarity(
                        prediction[index],
                        target[index],
                        channel_axis=-1,
                        data_range=1.0,
                        win_size=win_size,
                    )
                    for index in range(prediction.shape[0])
                ]
                metrics["ssim"] = float(np.mean(values, dtype=np.float64))
    return metrics


def _masked_video_mean(value: torch.Tensor, frame_mask: torch.Tensor | None) -> torch.Tensor:
    if frame_mask is None:
        return value.mean()
    if value.ndim != 5:
        raise ValueError(f"Expected latent video [B, C, F, H, W], got {value.shape}.")
    mask = frame_mask.to(device=value.device, dtype=value.dtype)
    if mask.ndim != 2 or mask.shape[0] != value.shape[0] or mask.shape[1] != value.shape[2]:
        raise ValueError(
            "frame_mask must have shape [B, F] matching the latent video, got "
            f"{mask.shape} for {value.shape}."
        )
    expanded_mask = mask[:, None, :, None, None]
    denominator = expanded_mask.sum() * value.shape[1] * value.shape[3] * value.shape[4]
    return (value * expanded_mask).sum() / denominator.clamp_min(1.0)


def rollout_trajectory_drift(
    student_states: Iterable[torch.Tensor],
    teacher_states: Iterable[torch.Tensor],
    *,
    frame_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | int]:
    """Average latent drift over aligned rollout states, excluding the shared start."""
    student_states = list(student_states)
    teacher_states = list(teacher_states)
    if len(student_states) != len(teacher_states):
        raise ValueError("Student and teacher trajectories must have the same number of states.")
    if len(student_states) < 2:
        raise ValueError("A rollout trajectory must include a shared start and one endpoint.")

    mse_values = []
    l1_values = []
    for student_state, teacher_state in zip(student_states[1:], teacher_states[1:]):
        if student_state.shape != teacher_state.shape:
            raise ValueError(
                "Student and teacher trajectory states must have matching shapes, got "
                f"{student_state.shape} and {teacher_state.shape}."
            )
        difference = student_state.float() - teacher_state.float()
        mse_values.append(_masked_video_mean(difference.square(), frame_mask))
        l1_values.append(_masked_video_mean(difference.abs(), frame_mask))
    return {
        "mse": torch.stack(mse_values).mean(),
        "l1": torch.stack(l1_values).mean(),
        "steps": len(mse_values),
    }
