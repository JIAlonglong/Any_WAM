import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from distillation_flowmap.cosmos_official_teacher_eval import (
    OfficialTeacherEvaluationService,
    ResolvedCosmosOfficialTeacher,
)
from evaluation.libero.cosmos_progressive_eval_summary import merge_complete_matrix
from evaluation.libero.cosmos_progressive_s4_client import CosmosProgressiveS4Client


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh"


class _Adapter:
    video_steps = 2
    action_steps = 2

    def infer_raw(self, raw_batch):
        assert raw_batch == {"converted": "obs"}
        return {
            "actions": np.zeros((1, 16, 7), dtype=np.float32),
            "requested_video_steps": 2,
            "requested_action_steps": 2,
            "effective_video_steps": 2,
            "effective_action_steps": 2,
            "matched_budget_verified": True,
            "observed_joint_nfe": 2,
            "cosmos_repo_commit": "1eb8457072b4a1adfe1f83c3076e4aa5452cbab2",
            "cosmos_source_sha256": (
                "c8cf94e18f840dda55afa162d6f6b0a4cada36fbbd8bbb45bf27c131f940f980"
            ),
            "future_image_predictions": [{"primary": np.zeros((2, 2, 3), dtype=np.uint8)}],
        }


def _resolved(tmp_path):
    return ResolvedCosmosOfficialTeacher(
        model_role="official_teacher",
        backend="cosmos_policy",
        root_path=str(tmp_path / "teacher"),
        weight_path=str(tmp_path / "teacher/model.pt"),
        config_path=str(tmp_path / "teacher/config.json"),
        dataset_stats_path=str(tmp_path / "teacher/stats.json"),
        t5_embeddings_path=str(tmp_path / "teacher/t5.pkl"),
        contract_identity="teacher-contract",
    )


def test_official_service_returns_raw_action_and_effective_k_without_student_metadata(tmp_path):
    service = OfficialTeacherEvaluationService(
        adapter=_Adapter(),
        resolved_teacher=_resolved(tmp_path),
        raw_request_builder=lambda obs, prompt: {"converted": obs},
    )

    reset = service.infer({"reset": True})
    response = service.infer({"obs": "obs", "prompt": "pick"})

    assert reset["model_role"] == response["model_role"] == "official_teacher"
    assert reset["action_grid_contract"] == response["action_grid_contract"]
    assert "student_steps" not in reset
    assert "student_steps" not in response
    assert response["action"].shape == (16, 7)
    assert response["video_steps"] == response["effective_video_steps"] == 2
    assert response["action_steps"] == response["effective_action_steps"] == 2
    assert response["matched_budget_verified"] is True
    assert response["future_prediction_keys"] == ["primary"]
    assert response["action_grid_contract"] == {
        "schema": "cosmos_action_grid_v1",
        "layout": "official_teacher_native_horizon",
        "action_downsample_factor": 1,
        "action_tensor_shape": [1, 16, 7],
        "updated_action_latent_indices": list(range(16)),
        "updated_action_indices": list(range(16)),
        "action_horizon": 16,
    }
    assert len(response["action_frame_stats"]) == 16
    assert response["action_frame_stats"][0] == {
        "frame": 0,
        "mean": 0.0,
        "std": 0.0,
        "absmax": 0.0,
    }


def test_official_client_requires_role_and_effective_k_and_omits_student_steps(tmp_path):
    service = OfficialTeacherEvaluationService(
        adapter=_Adapter(),
        resolved_teacher=_resolved(tmp_path),
        raw_request_builder=lambda obs, prompt: {"converted": obs},
    )
    client = CosmosProgressiveS4Client(
        service,
        output_dir=tmp_path / "output",
        model_role="official_teacher",
        video_steps=2,
        action_steps=2,
        expected_s4_checkpoint=_resolved(tmp_path).root_path,
        expected_checkpoint_contract_identity="teacher-contract",
        warmup_steps=0,
    )

    assert client.student_steps is None
    assert client._inference_contract_record() == {
        "model_role": "official_teacher",
        "video_steps": 2,
        "action_steps": 2,
    }
    client._validate_service_student_steps(service.infer({"reset": True}))


def _matrix_env(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    teacher_lock = tmp_path / "teacher.lock.json"
    teacher_lock.write_text("{}", encoding="utf-8")
    empty = dataset / "empty.pt"
    empty.write_bytes(b"x")
    prompts = dataset / "prompts.pt"
    prompts.write_bytes(b"x")
    env = os.environ.copy()
    env.update(
        MATRIX_ROOT=str(tmp_path / "matrix"),
        S4_CKPT_ROOT=str(checkpoint),
        COSMOS_POLICY_PATH=str(teacher),
        COSMOS_POLICY_TEACHER_LOCK=str(teacher_lock),
        S4_DATASET_PATH=str(dataset),
        S4_EMPTY_EMBEDDING=str(empty),
        S4_PROMPT_TABLE=str(prompts),
        PYTHON_BIN=sys.executable,
        S4_ALIGNMENT_VERIFIED="1",
        S4_MATRIX_ROLES="stage2_target,official_teacher",
    )
    return env


def test_matrix_dry_run_plans_both_roles_four_suites_and_matched_124(tmp_path):
    env = _matrix_env(tmp_path)
    result = subprocess.run(
        ["bash", str(MATRIX), "dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    roles = [
        line.split("=", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("MATRIX_ROLE=")
    ]
    assert roles == ["stage2_target", "official_teacher"]
    assert result.stdout.count("MATRIX_SUITE=") == 24
    assert result.stdout.count("--model-role stage2_target") > 0
    assert result.stdout.count("--model-role official_teacher") > 0
    assert "--cosmos-policy-path" in result.stdout
    assert result.stdout.count("TEACHER_LOCK_VERIFICATION_COMMAND=") == 1
    assert "--teacher-contract-identity" in result.stdout
    assert "--teacher-provenance-lock" not in result.stdout
    assert not Path(env["MATRIX_ROOT"]).exists()


def _write_role_matrix(root, checkpoint, role):
    for step in (1, 2, 4):
        for suite in ("libero_10", "libero_spatial", "libero_object", "libero_goal"):
            child = root / role / f"k{step}" / suite
            child.mkdir(parents=True)
            payload = {
                "checkpoint": str(checkpoint.resolve()),
                "libero_benchmark": suite,
                "model_role": role,
                "video_steps": step,
                "action_steps": step,
                "checkpoint_contract_identity": f"{role}-contract",
                "evaluation_classification": "formal_verified",
                "is_formal": True,
                "num_tasks": 10,
                "num_records": 10,
                "seeds_per_task": 1,
                "per_task_success": {f"{suite}:{i}": 0.5 for i in range(10)},
                "macro_success": 0.5,
            }
            if role != "official_teacher":
                payload["student_steps"] = step
            else:
                payload["cosmos_repo_commit"] = (
                    "1eb8457072b4a1adfe1f83c3076e4aa5452cbab2"
                )
                payload["cosmos_source_sha256"] = (
                    "c8cf94e18f840dda55afa162d6f6b0a4cada36fbbd8bbb45bf27c131f940f980"
                )
            (child / "formal_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_complete_matrix_separates_roles_and_requires_all_240_task_budget_cells(tmp_path):
    student = tmp_path / "student"
    teacher = tmp_path / "teacher"
    student.mkdir()
    teacher.mkdir()
    root = tmp_path / "matrix"
    _write_role_matrix(root, student, "stage2_target")
    _write_role_matrix(root, teacher, "official_teacher")

    summary = merge_complete_matrix(
        root=root,
        checkpoints={
            "stage2_target": student,
            "official_teacher": teacher,
        },
        evaluation_classification="formal_verified",
        is_formal=True,
        episodes_per_task=1,
    )

    assert summary["roles"] == ["stage2_target", "official_teacher"]
    assert summary["role_task_budget_cells"] == 240
    assert len(summary["summaries"]) == 24
    assert summary["official_teacher_cosmos_repo_commit"] == (
        "1eb8457072b4a1adfe1f83c3076e4aa5452cbab2"
    )
    assert summary["official_teacher_cosmos_source_sha256"] == (
        "c8cf94e18f840dda55afa162d6f6b0a4cada36fbbd8bbb45bf27c131f940f980"
    )

    extra = root / "official_teacher/k2/libero_goal/duplicate/formal_summary.json"
    extra.parent.mkdir()
    extra.write_text(
        (root / "official_teacher/k2/libero_goal/formal_summary.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="unexpected.*summary"):
        merge_complete_matrix(
            root=root,
            checkpoints={"stage2_target": student, "official_teacher": teacher},
            evaluation_classification="formal_verified",
            is_formal=True,
            episodes_per_task=1,
        )
    extra.unlink()
    extra.parent.rmdir()

    duplicate = root / "official_teacher/k2/libero_goal/formal_summary.json"
    payload = json.loads(duplicate.read_text(encoding="utf-8"))
    payload["model_role"] = "stage2_target"
    duplicate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="model_role mismatch"):
        merge_complete_matrix(
            root=root,
            checkpoints={"stage2_target": student, "official_teacher": teacher},
            evaluation_classification="formal_verified",
            is_formal=True,
            episodes_per_task=1,
        )
