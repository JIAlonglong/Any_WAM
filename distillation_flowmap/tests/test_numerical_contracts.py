import importlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from distillation_flowmap.numerical_contracts import (
    compare_terminal_prior,
    validate_terminal_prior_sources,
)
from distillation_flowmap.cosmos_stage2_lineage import validate_stage1_parent


def _config(*, atol=2e-6, warn_factor=0.5, verify=True, rank=0):
    return SimpleNamespace(
        opd_danceopd_terminal_prior_tolerance=atol,
        opd_danceopd_terminal_prior_warn_factor=warn_factor,
        opd_danceopd_verify_terminal_prior=verify,
        rank=rank,
    )


def _sources(video_actual, video_expected, action_actual, action_expected):
    return {
        "video": (video_actual, video_expected, torch.bfloat16),
        "action": (action_actual, action_expected, torch.float32),
    }


def _flowmap_step_module():
    repo_root = Path(__file__).resolve().parents[2]
    for path in (repo_root / "wan_va", repo_root / "distillation_flowmap"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    import distillation_flowmap.flowmap_step as flowmap_step

    return flowmap_step


def _set_progressive_lineage_env(monkeypatch, tmp_path):
    stage1 = tmp_path / "stage1"
    wan_base = tmp_path / "wan-base"
    (wan_base / "transformer").mkdir(parents=True)
    (wan_base / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "WanTransformer3DModel"})
    )
    cosmos_teacher = tmp_path / "cosmos-teacher"
    cosmos_teacher.mkdir()
    (cosmos_teacher / "config.json").write_text(
        json.dumps({"model_type": "cosmos-policy"})
    )
    for name in (
        "Cosmos-Policy-LIBERO-Predict2-2B.pt",
        "libero_dataset_statistics.json",
        "libero_t5_embeddings.pkl",
    ):
        (cosmos_teacher / name).write_bytes(b"fixture")
    payload = {
        "contract_version": 2,
        "training_contract_stage": "raw_stage1",
        "action_packing_schema": "downsample_survivor_v2",
        "action_downsample_factor": 4,
        "action_chunk_shape": [4, 4],
        "checkpoint_step": 5000,
        "teacher_backend": "cosmos_policy",
        "student_backend": "wan_flowmap",
        "student_base_model_path": str(wan_base.resolve()),
        "teacher_model_path": str(cosmos_teacher.resolve()),
    }
    for variant in ("online_student", "target_student"):
        transformer = stage1 / variant / "transformer"
        transformer.mkdir(parents=True)
        (transformer / "config.json").write_text(json.dumps(payload))
        (transformer / "diffusion_pytorch_model.safetensors").write_bytes(
            b"weights"
        )
    parent = validate_stage1_parent(stage1)
    lineage = json.dumps(
        {
            "parent_stage1_contract_identity": parent.contract_identity,
            "parent_stage1_path": parent.canonical_path,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    for name, value in {
        "STUDENT_BASE_MODEL_PATH": str(stage1 / "target_student"),
        "WAN_STUDENT_BASE_MODEL_PATH": str(wan_base),
        "COSMOS_POLICY_PATH": str(cosmos_teacher),
        "RESUME_FROM_PATH": str(stage1),
        "PARENT_STAGE1_PATH": parent.canonical_path,
        "PARENT_STAGE1_CONTRACT_IDENTITY": parent.contract_identity,
        "STAGE2_LINEAGE_JSON": lineage,
        "RESUME_ONLINE_FROM_TARGET": "1",
        "RESET_RESUME_STEP": "1",
        "RESUME_OPTIMIZER_STATE": "0",
        "OUTPUT_DIR": str(tmp_path / "stage2-arm"),
    }.items():
        monkeypatch.setenv(name, value)


def test_zero_reference_scheduler_residue_is_accepted_for_bfloat16():
    check = compare_terminal_prior(
        torch.tensor([1.192e-6]),
        torch.zeros(1),
        source_dtype=torch.bfloat16,
    )

    assert check.atol == pytest.approx(2e-6)
    assert check.rtol == pytest.approx(torch.finfo(torch.bfloat16).eps)
    assert check.threshold == pytest.approx(2e-6)
    assert check.severity in {"quiet", "warning"}


def test_bfloat16_nonzero_reference_corruption_is_an_error():
    check = compare_terminal_prior(
        torch.tensor([1.01]),
        torch.tensor([1.0]),
        source_dtype=torch.bfloat16,
    )

    assert check.atol == pytest.approx(2e-6)
    assert check.threshold == pytest.approx(
        2e-6 + torch.finfo(torch.bfloat16).eps
    )
    assert check.severity == "error"


def test_source_dtypes_only_control_the_relative_term():
    video = compare_terminal_prior(
        torch.ones(1),
        torch.ones(1),
        source_dtype=torch.bfloat16,
    )
    action = compare_terminal_prior(
        torch.ones(1),
        torch.ones(1),
        source_dtype=torch.float32,
    )

    assert video.atol == pytest.approx(action.atol)
    assert video.rtol == pytest.approx(torch.finfo(torch.bfloat16).eps)
    assert action.rtol == pytest.approx(torch.finfo(torch.float32).eps)


def test_validation_boundary_accepts_residue_and_emits_rank_safe_warning_callback():
    warnings = []
    result = validate_terminal_prior_sources(
        _sources(
            torch.tensor([1.192e-6]),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        ),
        config=_config(),
        warning_callback=warnings.append,
        error_context="DanceOPD terminal state",
    )

    assert result.checks["video"].severity == "warning"
    assert result.checks["action"].severity == "quiet"
    assert len(warnings) == 1
    assert "video:" in warnings[0]
    assert "threshold=" in warnings[0]


def test_validation_boundary_raises_with_all_metadata_for_corruption():
    with pytest.raises(RuntimeError) as exc_info:
        validate_terminal_prior_sources(
            _sources(
                torch.tensor([1.01]),
                torch.tensor([1.0]),
                torch.zeros(1),
                torch.zeros(1),
            ),
            config=_config(),
            error_context="DanceOPD terminal state",
        )

    message = str(exc_info.value)
    assert "DanceOPD terminal state" in message
    assert "video:" in message
    assert "action:" in message
    for field in (
        "max_error=",
        "reference_scale=",
        "atol=",
        "rtol=",
        "threshold=",
        "severity=",
    ):
        assert field in message


def test_disabled_verification_records_error_without_raising_or_warning():
    warnings = []
    result = validate_terminal_prior_sources(
        _sources(
            torch.tensor([1.01]),
            torch.tensor([1.0]),
            torch.zeros(1),
            torch.zeros(1),
        ),
        config=_config(verify=False),
        warning_callback=warnings.append,
        error_context="DanceOPD terminal state",
    )

    assert result.checks["video"].severity == "error"
    assert warnings == []


def test_explicit_tolerance_override_is_honored_and_validated():
    result = validate_terminal_prior_sources(
        _sources(
            torch.tensor([3e-6]),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        ),
        config=_config(atol=4e-6),
        error_context="DanceOPD terminal state",
    )
    assert result.checks["video"].atol == pytest.approx(4e-6)
    assert result.checks["video"].severity == "warning"

    for invalid in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="tolerance"):
            validate_terminal_prior_sources(
                _sources(
                    torch.zeros(1),
                    torch.zeros(1),
                    torch.zeros(1),
                    torch.zeros(1),
                ),
                config=_config(atol=invalid),
                error_context="DanceOPD terminal state",
            )


def test_warning_factor_is_separate_from_tolerance_and_validated():
    quiet = validate_terminal_prior_sources(
        _sources(
            torch.tensor([3e-6]),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        ),
        config=_config(atol=4e-6, warn_factor=0.8),
        error_context="DanceOPD terminal state",
    )
    assert quiet.checks["video"].severity == "quiet"

    for invalid in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="warn_factor"):
            validate_terminal_prior_sources(
                _sources(
                    torch.zeros(1),
                    torch.zeros(1),
                    torch.zeros(1),
                    torch.zeros(1),
                ),
                config=_config(warn_factor=invalid),
                error_context="DanceOPD terminal state",
            )


def test_diagnostic_tensors_are_coherent_with_the_same_worst_source():
    result = validate_terminal_prior_sources(
        _sources(
            torch.tensor([1.005]),
            torch.tensor([1.0]),
            torch.tensor([1.5e-6]),
            torch.zeros(1),
        ),
        config=_config(),
        error_context="DanceOPD terminal state",
    )
    diagnostics = result.diagnostic_tensors(torch.zeros(1))
    worst = result.checks[result.worst_source]

    assert result.worst_source == "action"
    assert diagnostics["max_error"].item() == pytest.approx(worst.max_error)
    assert diagnostics["reference_scale"].item() == pytest.approx(
        worst.reference_scale
    )
    assert diagnostics["atol"].item() == pytest.approx(worst.atol)
    assert diagnostics["rtol"].item() == pytest.approx(worst.rtol)
    assert diagnostics["threshold"].item() == pytest.approx(worst.threshold)
    assert diagnostics["severity"].item() == pytest.approx(1.0)
    assert all(torch.is_tensor(value) for value in diagnostics.values())


def test_production_boundary_uses_video_and_action_source_dtypes(monkeypatch):
    flowmap_step = _flowmap_step_module()

    warnings = []
    monkeypatch.setattr(flowmap_step.logger, "warning", warnings.append)
    validation, diagnostics = flowmap_step._validate_terminal_prior_pair(
        config=_config(),
        video_actual=torch.tensor([1.192e-6]),
        video_expected=torch.zeros(1),
        video_source_dtype=torch.bfloat16,
        action_actual=torch.zeros(1),
        action_expected=torch.zeros(1),
        action_source_dtype=torch.float32,
        reference=torch.zeros(1),
        error_context="production terminal state",
    )

    assert validation.checks["video"].rtol == pytest.approx(
        torch.finfo(torch.bfloat16).eps
    )
    assert validation.checks["action"].rtol == pytest.approx(
        torch.finfo(torch.float32).eps
    )
    assert validation.checks["video"].severity == "warning"
    assert len(warnings) == 1
    assert diagnostics["severity"].item() == pytest.approx(1.0)


def test_production_boundary_raises_for_corruption_and_is_rank_safe(monkeypatch):
    flowmap_step = _flowmap_step_module()

    warnings = []
    monkeypatch.setattr(flowmap_step.logger, "warning", warnings.append)
    kwargs = dict(
        video_actual=torch.tensor([1.01]),
        video_expected=torch.tensor([1.0]),
        video_source_dtype=torch.bfloat16,
        action_actual=torch.zeros(1),
        action_expected=torch.zeros(1),
        action_source_dtype=torch.float32,
        reference=torch.zeros(1),
        error_context="production terminal state",
    )
    with pytest.raises(RuntimeError, match="production terminal state"):
        flowmap_step._validate_terminal_prior_pair(
            config=_config(rank=1),
            **kwargs,
        )
    assert warnings == []

    validation, _ = flowmap_step._validate_terminal_prior_pair(
        config=_config(verify=False, rank=0),
        **kwargs,
    )
    assert validation.checks["video"].severity == "error"
    assert warnings == []


@pytest.mark.parametrize(
    "module_name",
    [
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive",
        "distillation_flowmap.config_libero_fullfinetune_stage2_anyflow",
        "distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow",
    ],
)
def test_shared_configs_preserve_explicit_tolerance_override(
    monkeypatch, module_name, tmp_path
):
    _flowmap_step_module()
    if module_name.endswith("cosmos_policy_stage2_progressive"):
        _set_progressive_lineage_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPD_DANCEOPD_TERMINAL_PRIOR_TOLERANCE", "4e-6")
    monkeypatch.setenv("OPD_DANCEOPD_TERMINAL_PRIOR_WARN_FACTOR", "0.75")
    sys.modules.pop(module_name, None)
    config = importlib.import_module(module_name).cfg

    assert config.opd_danceopd_terminal_prior_tolerance == pytest.approx(4e-6)
    assert config.opd_danceopd_terminal_prior_warn_factor == pytest.approx(0.75)


@pytest.mark.parametrize(
    "module_name",
    [
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive",
        "distillation_flowmap.config_libero_fullfinetune_stage2_anyflow",
        "distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow",
    ],
)
def test_shared_configs_reject_invalid_tolerance(
    monkeypatch, module_name, tmp_path
):
    _flowmap_step_module()
    if module_name.endswith("cosmos_policy_stage2_progressive"):
        _set_progressive_lineage_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPD_DANCEOPD_TERMINAL_PRIOR_TOLERANCE", "nan")
    sys.modules.pop(module_name, None)
    with pytest.raises(ValueError, match="TERMINAL_PRIOR_TOLERANCE"):
        importlib.import_module(module_name)


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
