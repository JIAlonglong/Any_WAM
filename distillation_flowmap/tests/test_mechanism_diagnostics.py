import math

import pytest
import torch

from distillation_flowmap.mechanism_diagnostics import (
    compute_mechanism_metric_samples,
    diagnostic_due,
    diagnostic_seed,
    means_from_reduced_stats,
    pack_finite_metric_stats,
)


def _metric_samples(**overrides):
    inputs = {
        "teacher_cont_video": torch.tensor([[3.0, 4.0], [2.0, 2.0]]),
        "teacher_endpoint_video": torch.zeros(2, 2),
        "student_direct_video": torch.tensor([[2.0, 0.0], [1.0, 1.0]]),
        "student_composed_video": torch.zeros(2, 2),
        "student_field_video": torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        "teacher_field_video": torch.zeros(2, 2),
        "teacher_endpoint_action": torch.zeros(2, 1),
        "action_student_context": torch.tensor([[3.0], [1.0]]),
        "action_teacher_video_context": torch.tensor([[1.0], [1.0]]),
        "action_teacher_joint_context": torch.zeros(2, 1),
        "action_mask": None,
    }
    inputs.update(overrides)
    return compute_mechanism_metric_samples(**inputs)


def test_video_metrics_keep_exact_squared_l2_separate_from_normalized_mse():
    samples = _metric_samples()

    assert torch.equal(samples["mechanism/g_anchor"], torch.tensor([25.0, 8.0]))
    assert torch.equal(
        samples["mechanism/g_anchor_mse"], torch.tensor([12.5, 4.0])
    )
    assert torch.equal(samples["mechanism/g_comp"], torch.tensor([4.0, 2.0]))
    assert torch.equal(samples["mechanism/g_comp_mse"], torch.tensor([2.0, 1.0]))
    assert torch.equal(
        samples["mechanism/video_endpoint_error"], torch.tensor([4.0, 2.0])
    )
    assert torch.equal(
        samples["mechanism/video_field_match_error"], torch.tensor([4.0, 4.0])
    )
    assert all(value.shape == (2,) for value in samples.values())


@pytest.mark.parametrize(
    ("student", "teacher_video", "expected_gain"),
    [
        (torch.tensor([[[3.0], [99.0]]]), torch.tensor([[[1.0], [88.0]]]), 8.0),
        (torch.tensor([[[2.0], [99.0]]]), torch.tensor([[[2.0], [88.0]]]), 0.0),
        (torch.tensor([[[1.0], [99.0]]]), torch.tensor([[[3.0], [88.0]]]), -8.0),
    ],
)
def test_action_intervention_metrics_are_masked_and_keep_signed_gains(
    student, teacher_video, expected_gain
):
    samples = _metric_samples(
        teacher_endpoint_action=torch.zeros(1, 2, 1),
        action_student_context=student,
        action_teacher_video_context=teacher_video,
        action_teacher_joint_context=torch.tensor([[[0.5], [77.0]]]),
        action_mask=torch.tensor([[[1.0], [0.0]]]),
        teacher_cont_video=torch.zeros(1, 2),
        teacher_endpoint_video=torch.zeros(1, 2),
        student_direct_video=torch.zeros(1, 2),
        student_composed_video=torch.zeros(1, 2),
        student_field_video=torch.zeros(1, 2),
        teacher_field_video=torch.zeros(1, 2),
    )

    student_error = samples["mechanism/action_error_student_context"]
    teacher_video_error = samples[
        "mechanism/action_error_teacher_video_context"
    ]
    teacher_joint_error = samples[
        "mechanism/action_error_teacher_joint_context"
    ]
    oracle_gain = samples["mechanism/video_to_action_oracle_gain"]

    assert oracle_gain.item() == expected_gain
    assert torch.equal(oracle_gain, student_error - teacher_video_error)
    assert torch.equal(
        samples["mechanism/video_to_action_recoverable_fraction"],
        oracle_gain.clamp_min(0) / student_error.clamp_min(1e-8),
    )
    assert torch.equal(
        samples["mechanism/video_to_action_full_joint_gain"],
        student_error - teacher_joint_error,
    )
    assert torch.equal(
        samples["mechanism/video_to_action_residual_action_gap"],
        teacher_video_error - teacher_joint_error,
    )


def test_pack_finite_stats_excludes_nonfinite_values_and_marks_batch_invalid():
    stats = pack_finite_metric_stats(
        {
            "mechanism/finite_and_nan": torch.tensor([3.5, float("nan")]),
            "mechanism/all_finite": torch.tensor([2.0, 4.0]),
        }
    )

    assert stats["mechanism/finite_and_nan_sum"].item() == 3.5
    assert stats["mechanism/finite_and_nan_count"].item() == 1.0
    assert stats["mechanism/all_finite_sum"].item() == 6.0
    assert stats["mechanism/all_finite_count"].item() == 2.0
    assert stats["diagnostic_valid"].item() == 0.0


def test_means_use_globally_reduced_sum_and_count_and_skip_empty_metrics():
    means = means_from_reduced_stats(
        {
            "mechanism/g_anchor_sum": torch.tensor(30.0),
            "mechanism/g_anchor_count": torch.tensor(6.0),
            "mechanism/empty_sum": torch.tensor(0.0),
            "mechanism/empty_count": torch.tensor(0.0),
            "diagnostic_valid": torch.tensor(0.0),
        }
    )

    assert means["mechanism/g_anchor"] == 5.0
    assert math.isnan(means["mechanism/empty"])
    assert "diagnostic_valid" not in means


def test_diagnostic_schedule_excludes_step_zero_and_seed_is_deterministic():
    assert not diagnostic_due(step=0, interval=100)
    assert not diagnostic_due(step=99, interval=100)
    assert diagnostic_due(step=100, interval=100)
    assert diagnostic_due(step=200, interval=100)

    assert diagnostic_seed(42, diagnostic_index=3, batch_index=5, rank=7) == (
        42 + 3 * 1009 + 5 * 97 + 7
    )
