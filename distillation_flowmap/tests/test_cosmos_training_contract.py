import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

from distillation_flowmap.cosmos_training_contract import (
    ACTION_PACKING_SCHEMA,
    CONTRACT_VERSION,
    contract_metadata,
    validate_contract_metadata,
)


def _verifier_module():
    # The production verifier imports the strict Stage-2 config.  Keep pure
    # contract tests runnable without a real Stage-1 checkpoint environment.
    import distillation_flowmap.verify_cosmos_joint_training_contract as verifier

    return verifier


class _IntSubclass(int):
    pass


def _stage1_config(**overrides):
    values = {
        "contract_version": CONTRACT_VERSION,
        "action_packing_schema": ACTION_PACKING_SCHEMA,
        "action_downsample_factor": 4,
        "action_chunk_shape": [4, 4],
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
        "deployment_action_weight": 1.0,
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
        "action_chunk_shape": [4, 4],
    }
    assert "joint_student_steps" not in payload


def test_stage2_contract_contains_full_deployment_attestation():
    payload = contract_metadata(_stage2_config(), stage="progressive_stage2")

    assert payload["deployment_timestep_start"] == 1000
    assert payload["deployment_timestep_end"] == 0
    assert payload["joint_student_steps"] == [1, 2, 4]
    assert payload["deployment_joint_rollout_interval"] == 4
    assert payload["deployment_action_weight"] == 1.0
    assert payload["raw_teacher_window_is_auxiliary"] is True
    validate_contract_metadata(payload, required_stage="progressive_stage2")


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"action_packing_schema": "legacy_dense_v1"}, "action_packing_schema"),
        ({"action_downsample_factor": True}, "action_downsample_factor"),
        ({"action_chunk_shape": (4, 4)}, "action_chunk_shape"),
        ({"action_chunk_shape": [True, 4]}, "action_chunk_shape"),
        ({"action_chunk_shape": [4, 3]}, "action_chunk_shape"),
        ({"action_chunk_shape": [3, 4]}, "action_chunk_shape"),
        ({"deployment_joint_steps": (1, 4, 2)}, "joint_student_steps"),
        ({"deployment_action_weight": 1}, "deployment_action_weight"),
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


def test_validate_contract_metadata_rejects_missing_action_chunk_shape():
    payload = contract_metadata(_stage1_config(), stage="raw_stage1")
    del payload["action_chunk_shape"]

    with pytest.raises(ValueError, match="action_chunk_shape"):
        validate_contract_metadata(payload, required_stage="raw_stage1")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract_version", True),
        ("action_chunk_shape", (4, 4)),
        ("action_chunk_shape", [True, 4]),
        ("action_chunk_shape", [3, 4]),
        ("action_chunk_shape", [4, 3]),
        ("deployment_timestep_start", 1000.0),
        ("joint_student_steps", (1, 2, 4)),
        ("joint_student_steps", [True, 2, 4]),
        ("joint_student_steps", [_IntSubclass(1), 2, 4]),
        ("deployment_joint_rollout_interval", True),
        ("deployment_action_weight", 1),
        ("raw_teacher_window_is_auxiliary", 1),
    ],
)
def test_validate_contract_metadata_is_exact_and_type_strict(field, value):
    payload = contract_metadata(_stage2_config(), stage="progressive_stage2")
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        validate_contract_metadata(payload, required_stage="progressive_stage2")


@pytest.mark.parametrize(
    "field",
    [
        "deployment_timestep_start",
        "deployment_timestep_end",
        "joint_student_steps",
        "deployment_joint_rollout_interval",
        "deployment_action_weight",
        "raw_teacher_window_is_auxiliary",
    ],
)
def test_stage1_validator_rejects_stage2_only_fields(field):
    payload = contract_metadata(_stage1_config(), stage="raw_stage1")
    payload[field] = {
        "deployment_timestep_start": 1000,
        "deployment_timestep_end": 0,
        "joint_student_steps": [1, 2, 4],
        "deployment_joint_rollout_interval": 4,
        "deployment_action_weight": 1.0,
        "raw_teacher_window_is_auxiliary": True,
    }[field]

    with pytest.raises(ValueError, match=field):
        validate_contract_metadata(payload, required_stage="raw_stage1")


def test_verifier_executes_exact_production_deployment_contract():
    evidence = _verifier_module()._verify_deployment_contract(_stage2_config(
        opd_aux_warmup_steps=8,
        opd_aux_interval=8,
        opd_aux_phase=2,
    ))

    assert evidence == {
        "joint_student_step_cycle": [1, 2, 4, 1, 2, 4],
        "deployment_schedule_steps": [8, 12, 16, 20, 24],
        "raw_auxiliary_schedule_steps": [10, 18],
        "schedules_disjoint": True,
        "endpoint_losses": {
            "video": 1.0,
            "action": 4.0,
            "total": 5.0,
        },
    }


def test_contract_verifier_runs_production_action_round_trip(tmp_path):
    output = tmp_path / "attestation.json"

    assert _verifier_module().main(["--output", str(output)]) == 0
    payload = json.loads(output.read_text())

    assert payload["action_round_trip"]["num_actions"] == 16
    assert payload["action_round_trip"]["max_abs_error"] < 1e-5
    assert payload["verifier_module"] == (
        "distillation_flowmap.verify_cosmos_joint_training_contract"
    )
    assert payload["deployment_action_weight"] == 1.0
    assert payload["deployment_execution"]["joint_student_step_cycle"] == [
        1, 2, 4, 1, 2, 4
    ]
    assert payload["deployment_execution"]["endpoint_losses"]["total"] == 5.0
    validate_contract_metadata(payload, required_stage="progressive_stage2")


def test_contract_verifier_refuses_to_overwrite_existing_attestation(tmp_path):
    output = tmp_path / "attestation.json"
    output.write_text('{"sentinel": true}\n')

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _verifier_module().main(["--output", str(output)])

    assert output.read_text() == '{"sentinel": true}\n'


def test_attestation_refuses_dangling_symlink_destination(tmp_path):
    output = tmp_path / "attestation.json"
    target = tmp_path / "missing-target.json"
    output.symlink_to(target.name)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _verifier_module()._write_new_attestation(output, {"sentinel": True})

    assert output.is_symlink()
    assert os.readlink(output) == target.name
    assert not target.exists()


def test_concurrent_attestation_publication_has_one_complete_winner(tmp_path):
    output = tmp_path / "attestation.json"
    workers = 8
    barrier = Barrier(workers)

    def publish(index):
        barrier.wait()
        try:
            _verifier_module()._write_new_attestation(
                output,
                {"writer": index, "body": "x" * 100_000},
            )
        except FileExistsError:
            return "exists"
        return "published"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(publish, range(workers)))

    assert results.count("published") == 1
    assert results.count("exists") == workers - 1
    payload = json.loads(output.read_text())
    assert payload["writer"] in range(workers)
    assert payload["body"] == "x" * 100_000
    assert not list(tmp_path.glob(".attestation.json.*.tmp"))


def test_verifier_resolves_git_commit_from_unrelated_cwd(tmp_path, monkeypatch):
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    output = tmp_path / "attestation.json"
    monkeypatch.chdir(unrelated)

    assert _verifier_module().main(["--output", str(output)]) == 0
    payload = json.loads(output.read_text())

    assert len(payload["git_commit"]) == 40
    assert set(payload["git_commit"]) <= set("0123456789abcdef")


def test_inference_request_requires_explicit_matched_video_and_action_budgets():
    import distillation_flowmap.cosmos_training_contract as contract

    request = contract.normalize_cosmos_inference_request(
        model_role="stage1_target",
        video_steps=2,
        action_steps=2,
        student_steps=None,
    )

    assert request.model_role == "stage1_target"
    assert request.video_steps == 2
    assert request.action_steps == 2
    assert request.student_steps == 2

    with pytest.raises(ValueError, match="video_steps.*action_steps"):
        contract.normalize_cosmos_inference_request(
            model_role="stage1_target",
            video_steps=1,
            action_steps=2,
            student_steps=None,
        )


def test_inference_request_supports_student_steps_only_as_unambiguous_alias():
    import distillation_flowmap.cosmos_training_contract as contract

    request = contract.normalize_cosmos_inference_request(
        model_role="stage2_target",
        video_steps=None,
        action_steps=None,
        student_steps=4,
    )
    assert (request.video_steps, request.action_steps, request.student_steps) == (4, 4, 4)

    with pytest.raises(ValueError, match="student_steps"):
        contract.normalize_cosmos_inference_request(
            model_role="stage2_target",
            video_steps=4,
            action_steps=4,
            student_steps=4,
        )


def test_official_teacher_matched_k_request_is_explicit_before_runtime_load():
    import distillation_flowmap.cosmos_training_contract as contract

    request = contract.normalize_cosmos_inference_request(
        model_role="official_teacher",
        video_steps=2,
        action_steps=2,
        student_steps=None,
    )

    assert request.model_role == "official_teacher"
    assert (request.video_steps, request.action_steps) == (2, 2)
