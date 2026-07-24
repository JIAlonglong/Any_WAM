import math
from pathlib import Path

import pytest
import torch

from distillation_flowmap.numerical_contracts import compare_terminal_prior


def test_bfloat16_accepts_scheduler_residue_but_rejects_material_corruption():
    accepted = compare_terminal_prior(
        torch.tensor([1.192e-6]),
        torch.zeros(1),
        source_dtype=torch.bfloat16,
    )
    assert accepted.severity in {"quiet", "warning"}

    corrupt = compare_terminal_prior(
        torch.tensor([1e-2]),
        torch.zeros(1),
        source_dtype=torch.bfloat16,
    )
    assert corrupt.severity == "error"


def test_fp32_scheduler_floor_exceeds_observed_residue():
    check = compare_terminal_prior(
        torch.tensor([1.192e-6]),
        torch.zeros(1),
        source_dtype=torch.float32,
    )

    assert check.atol > 1.192e-6
    assert check.rtol == pytest.approx(torch.finfo(torch.float32).eps)
    assert check.threshold == pytest.approx(check.atol)
    assert check.severity in {"quiet", "warning"}


def test_nonzero_reference_scale_contributes_relative_tolerance():
    expected = torch.tensor([1000.0, -500.0])
    check = compare_terminal_prior(
        expected.clone(),
        expected,
        source_dtype=torch.float32,
    )

    assert check.reference_scale == pytest.approx(1000.0)
    assert check.threshold == pytest.approx(
        check.atol + check.rtol * check.reference_scale
    )
    assert check.severity == "quiet"


def test_warning_band_records_all_effective_metadata():
    baseline = compare_terminal_prior(
        torch.zeros(1),
        torch.zeros(1),
        source_dtype=torch.float32,
        warn_factor=0.5,
    )
    check = compare_terminal_prior(
        torch.tensor([baseline.threshold * 0.75]),
        torch.zeros(1),
        source_dtype=torch.float32,
        warn_factor=0.5,
    )

    assert check.severity == "warning"
    assert check.max_error == pytest.approx(baseline.threshold * 0.75)
    assert check.reference_scale == pytest.approx(0.0)
    assert check.atol == pytest.approx(baseline.atol)
    assert check.rtol == pytest.approx(baseline.rtol)
    assert check.threshold == pytest.approx(baseline.threshold)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (torch.tensor([float("nan")]), torch.zeros(1)),
        (torch.tensor([float("inf")]), torch.zeros(1)),
        (torch.zeros(1), torch.tensor([float("inf")])),
    ],
)
def test_nonfinite_inputs_are_errors(actual, expected):
    check = compare_terminal_prior(
        actual,
        expected,
        source_dtype=torch.float32,
    )

    assert check.severity == "error"
    assert math.isinf(check.max_error)


def test_both_cosmos_terminal_checks_use_dtype_aware_comparator_and_keep_formulas():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_step.py"
    ).read_text(encoding="utf-8")
    raw_window_block = source.split(
        "def _cosmos_danceopd_velocity_loss("
    )[1].split("def _cosmos_deployment_joint_rollout_step(")[0]
    generic_block = source.split(
        "def _danceopd_aux_transition_step("
    )[1].split("def _cosmos_latent_full_opd_aux_transition_step(")[0]

    assert raw_window_block.count("compare_terminal_prior(") == 2
    assert "expected_video_start = (" in raw_window_block
    assert "expected_action_start = (" in raw_window_block
    assert "terminal_prior_threshold" in raw_window_block
    assert generic_block.count("compare_terminal_prior(") == 2
    assert "current_video,\n                video_noise," in generic_block
    assert "current_action,\n                action_noise," in generic_block
    assert "danceopd_terminal_prior_threshold" in generic_block
    assert "terminal_prior_tolerance" not in source


def test_progressive_config_exposes_warning_factor_not_scalar_tolerance():
    source = (
        Path(__file__).resolve().parents[1]
        / "config_libero_cosmos_policy_stage2_progressive.py"
    ).read_text(encoding="utf-8")

    assert "opd_danceopd_terminal_prior_warn_factor" in source
    assert "OPD_DANCEOPD_TERMINAL_PRIOR_WARN_FACTOR" in source
    assert "opd_danceopd_terminal_prior_tolerance" not in source
