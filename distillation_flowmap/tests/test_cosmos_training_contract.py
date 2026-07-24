import json
from types import SimpleNamespace

import pytest

from distillation_flowmap.cosmos_training_contract import (
    ACTION_PACKING_SCHEMA,
    CONTRACT_VERSION,
    contract_metadata,
    validate_contract_metadata,
)
from distillation_flowmap.verify_cosmos_joint_training_contract import (
    main as verifier_main,
)


def _stage1_config(**overrides):
    values = {
        "contract_version": CONTRACT_VERSION,
        "action_packing_schema": ACTION_PACKING_SCHEMA,
        "action_downsample_factor": 4,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _stage2_config(**overrides):
    values = {
        **vars(_stage1_config()),
        "deployment_timestep_start": 1000,
        "deployment_timestep_end": 0,
        "deployment_joint_steps": (1, 2, 4),
        "deployment_joint_rollout_interval": 4,
        "raw_teacher_window_is_auxiliary": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_stage1_contract_contains_packing_but_not_stage2_attestation():
    payload = contract_metadata(_stage1_config(), stage="raw_stage1")

    assert payload == {
        "contract_version": 2,
        "training_contract_stage": "raw_stage1",
        "action_packing_schema": "downsample_survivor_v2",
        "action_downsample_factor": 4,
    }
    assert "joint_student_steps" not in payload


def test_stage2_contract_contains_full_deployment_attestation():
    payload = contract_metadata(_stage2_config(), stage="progressive_stage2")

    assert payload["deployment_timestep_start"] == 1000
    assert payload["deployment_timestep_end"] == 0
    assert payload["joint_student_steps"] == [1, 2, 4]
    assert payload["deployment_joint_rollout_interval"] == 4
    assert payload["raw_teacher_window_is_auxiliary"] is True
    validate_contract_metadata(payload, required_stage="progressive_stage2")


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"action_packing_schema": "legacy_dense_v1"}, "action_packing_schema"),
        ({"action_downsample_factor": True}, "action_downsample_factor"),
        ({"deployment_joint_steps": (1, 4, 2)}, "joint_student_steps"),
        ({"raw_teacher_window_is_auxiliary": 1}, "raw_teacher_window_is_auxiliary"),
    ],
)
def test_contract_metadata_rejects_wrong_values_and_types(overrides, match):
    with pytest.raises((TypeError, ValueError), match=match):
        contract_metadata(_stage2_config(**overrides), stage="progressive_stage2")


def test_validate_contract_metadata_rejects_missing_stage2_field():
    payload = contract_metadata(_stage2_config(), stage="progressive_stage2")
    del payload["deployment_timestep_end"]

    with pytest.raises(ValueError, match="deployment_timestep_end"):
        validate_contract_metadata(payload, required_stage="progressive_stage2")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract_version", True),
        ("deployment_timestep_start", 1000.0),
        ("joint_student_steps", (1, 2, 4)),
        ("deployment_joint_rollout_interval", True),
        ("raw_teacher_window_is_auxiliary", 1),
    ],
)
def test_validate_contract_metadata_is_exact_and_type_strict(field, value):
    payload = contract_metadata(_stage2_config(), stage="progressive_stage2")
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        validate_contract_metadata(payload, required_stage="progressive_stage2")


def test_contract_verifier_runs_production_action_round_trip(tmp_path):
    output = tmp_path / "attestation.json"

    assert verifier_main(["--output", str(output)]) == 0
    payload = json.loads(output.read_text())

    assert payload["action_round_trip"]["num_actions"] == 16
    assert payload["action_round_trip"]["max_abs_error"] < 1e-5
    assert payload["verifier_module"] == (
        "distillation_flowmap.verify_cosmos_joint_training_contract"
    )
    validate_contract_metadata(payload, required_stage="progressive_stage2")


def test_contract_verifier_refuses_to_overwrite_existing_attestation(tmp_path):
    output = tmp_path / "attestation.json"
    output.write_text('{"sentinel": true}\n')

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        verifier_main(["--output", str(output)])

    assert output.read_text() == '{"sentinel": true}\n'
    assert not (tmp_path / ".attestation.json.tmp").exists()
