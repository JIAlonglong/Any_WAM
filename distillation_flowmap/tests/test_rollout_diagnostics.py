import numpy as np
import pytest
import torch

from distillation_flowmap.rollout_diagnostics import (
    decoded_video_metrics,
    rollout_trajectory_drift,
)


def test_decoded_video_metrics_capture_motion_error():
    target = np.zeros((3, 8, 8, 3), dtype=np.float32)
    prediction = target.copy()
    prediction[1] = 0.25

    metrics = decoded_video_metrics(
        prediction,
        target,
        include_ssim=False,
    )

    assert metrics["pixel_mse"] == pytest.approx(0.25**2 / 3.0)
    assert metrics["pixel_l1"] == pytest.approx(0.25 / 3.0)
    assert metrics["temporal_difference_mse"] == pytest.approx(0.25**2)


def test_rollout_trajectory_drift_averages_noninitial_states():
    teacher = [
        torch.zeros((1, 1, 1, 2, 2)),
        torch.zeros((1, 1, 1, 2, 2)),
        torch.zeros((1, 1, 1, 2, 2)),
    ]
    student = [
        torch.zeros((1, 1, 1, 2, 2)),
        torch.ones((1, 1, 1, 2, 2)),
        torch.full((1, 1, 1, 2, 2), 2.0),
    ]

    metrics = rollout_trajectory_drift(student, teacher)

    assert metrics["mse"] == pytest.approx(2.5)
    assert metrics["l1"] == pytest.approx(1.5)
    assert metrics["steps"] == 2


def test_rollout_trajectory_drift_rejects_misaligned_paths():
    path = [torch.zeros((1, 1, 1, 2, 2))]
