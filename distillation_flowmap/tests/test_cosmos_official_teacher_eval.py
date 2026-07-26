import json
from pathlib import Path
import numpy as np
import pytest

import distillation_flowmap.cosmos_official_teacher_eval as teacher_eval
from distillation_flowmap.cosmos_official_teacher_eval import (
    OfficialTeacherMatchedBudgetAdapter,
    resolve_cosmos_official_teacher_root,
)
from distillation_flowmap.cosmos_policy_raw_worker import (
    _call_get_action_with_observed_joint_nfe,
    _official_matched_budget_response_fields,
)
from distillation_flowmap.cosmos_training_contract import (
    normalize_cosmos_inference_request,
)
from distillation_flowmap.cosmos_libero_provenance import (
    build_artifact_lock,
    canonical_json,
)


@pytest.mark.parametrize("budget", (1, 2, 4))
def test_official_teacher_accepts_explicit_matched_budgets(budget):
    request = normalize_cosmos_inference_request(
        model_role="official_teacher",
        video_steps=budget,
        action_steps=budget,
        student_steps=None,
    )

    assert request.model_role == "official_teacher"
    assert request.video_steps == budget
    assert request.action_steps == budget
    assert request.student_steps is None


def test_official_teacher_missing_flowmap_metadata_uses_native_continuous_grid():
    contract = teacher_eval.official_teacher_action_grid_contract(
        action_shape=(1, 16, 7),
    )

    assert contract == {
        "schema": "cosmos_action_grid_v1",
        "layout": "official_teacher_native_horizon",
        "action_downsample_factor": 1,
        "action_tensor_shape": [1, 16, 7],
        "updated_action_latent_indices": list(range(16)),
        "updated_action_indices": list(range(16)),
        "action_horizon": 16,
    }


def _write_official_teacher_root(root: Path) -> Path:
    root.mkdir()
    (root / "Cosmos-Policy-LIBERO-Predict2-2B.pt").write_bytes(b"teacher")
    (root / "libero_dataset_statistics.json").write_text("{}", encoding="utf-8")
    (root / "libero_t5_embeddings.pkl").write_bytes(b"embeddings")
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "cosmos-policy",
                "architecture": "diffusion-transformer",
                "diffusion_config": {
                    "generation_mode": "parallel",
                    "denoising_steps": 5,
                },
                "output_spec": {"actions": {"dim": 7, "horizon": 16}},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_official_teacher_root_resolves_monolithic_policy_not_student(tmp_path):
    resolved = resolve_cosmos_official_teacher_root(
        _write_official_teacher_root(tmp_path / "teacher"),
        verified_contract_identity="a" * 64,
    )

    assert resolved.model_role == "official_teacher"
    assert resolved.backend == "cosmos_policy"
    assert resolved.root_path == str((tmp_path / "teacher").resolve())
    assert resolved.weight_path.endswith("Cosmos-Policy-LIBERO-Predict2-2B.pt")
    assert resolved.transformer_path is None
    assert len(resolved.contract_identity) == 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("model_type", "wan", "model_type"),
        ("architecture", "student-transformer", "architecture"),
    ),
)
def test_official_teacher_root_rejects_wrong_backend_contract(
    tmp_path, field, value, message
):
    root = _write_official_teacher_root(tmp_path / "teacher")
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[field] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        resolve_cosmos_official_teacher_root(
            root, verified_contract_identity="a" * 64
        )


def test_official_teacher_root_rejects_student_transformer_directory(tmp_path):
    transformer = tmp_path / "target_student" / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="monolithic"):
        resolve_cosmos_official_teacher_root(
            transformer, verified_contract_identity="a" * 64
        )


def test_official_teacher_root_requires_verified_provenance_identity(tmp_path):
    root = _write_official_teacher_root(tmp_path / "teacher")

    with pytest.raises(ValueError, match="provenance contract identity"):
        resolve_cosmos_official_teacher_root(root)


def test_official_teacher_root_reproduces_and_verifies_real_artifact_lock(tmp_path):
    root = _write_official_teacher_root(tmp_path / "teacher")
    lock = build_artifact_lock(
        root,
        compact_paths=(
            "config.json",
            "libero_dataset_statistics.json",
            "libero_t5_embeddings.pkl",
        ),
        large_paths=("Cosmos-Policy-LIBERO-Predict2-2B.pt",),
        immutable_store=False,
    )
    lock_path = tmp_path / "teacher.lock.json"
    lock_path.write_text(canonical_json(lock) + "\n", encoding="utf-8")

    resolved = resolve_cosmos_official_teacher_root(
        root, provenance_lock_path=lock_path
    )

    assert resolved.contract_identity
    (root / "Cosmos-Policy-LIBERO-Predict2-2B.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest|size"):
        resolve_cosmos_official_teacher_root(root, provenance_lock_path=lock_path)


@pytest.mark.parametrize("budget", (1, 2, 4))
def test_worker_budget_contract_reports_requested_and_effective_k(budget):
    fields = _official_matched_budget_response_fields(
        {
            "requested_video_steps": budget,
            "requested_action_steps": budget,
        },
        configured_action_steps=budget,
        configured_future_steps=budget,
        include_future=True,
        observed_joint_nfe=budget,
    )

    assert fields == {
        "requested_video_steps": np.int64(budget),
        "requested_action_steps": np.int64(budget),
        "effective_video_steps": np.int64(budget),
        "effective_action_steps": np.int64(budget),
        "matched_budget_verified": np.bool_(True),
        "observed_joint_nfe": np.int64(budget),
    }


@pytest.mark.parametrize(
    "budget_request",
    (
        {},
        {"requested_video_steps": 2},
        {"requested_action_steps": 2},
        {"requested_video_steps": 2, "requested_action_steps": 4},
        {"requested_video_steps": 5, "requested_action_steps": 5},
    ),
)
def test_worker_budget_contract_rejects_missing_mismatched_or_fixed_five(
    budget_request,
):
    with pytest.raises(ValueError, match="matched|requested|1, 2, 4"):
        _official_matched_budget_response_fields(
            budget_request,
            configured_action_steps=2,
            configured_future_steps=2,
            include_future=True,
            observed_joint_nfe=2,
        )


def test_worker_budget_contract_rejects_silent_configured_five_fallback():
    with pytest.raises(RuntimeError, match="configured action steps"):
        _official_matched_budget_response_fields(
            {"requested_video_steps": 2, "requested_action_steps": 2},
            configured_action_steps=5,
            configured_future_steps=2,
            include_future=True,
            observed_joint_nfe=2,
        )


def test_worker_budget_contract_rejects_observed_sampler_nfe_mismatch():
    with pytest.raises(RuntimeError, match="observed joint"):
        _official_matched_budget_response_fields(
            {"requested_video_steps": 2, "requested_action_steps": 2},
            configured_action_steps=2,
            configured_future_steps=2,
            include_future=True,
            observed_joint_nfe=5,
        )


def test_official_runtime_counts_actual_joint_denoiser_calls_and_restores_model():
    class Model:
        def get_x0_fn_from_batch(self, _batch):
            return lambda value: value

    model = Model()
    original = model.get_x0_fn_from_batch

    def get_action(*, model, requested_steps):
        denoiser = model.get_x0_fn_from_batch({})
        for step in range(requested_steps):
            denoiser(step)
        return {"actions": "sentinel"}

    result, observed = _call_get_action_with_observed_joint_nfe(
        get_action,
        model=model,
        get_action_kwargs={"requested_steps": 4},
    )

    assert result == {"actions": "sentinel"}
    assert observed == 4
    assert model.get_x0_fn_from_batch.__func__ is original.__func__


class _Teacher:
    def __init__(self, result, *, configured_steps=2):
        self.result = result
        self.num_denoising_steps_action = configured_steps
        self.calls = []

    def predict_raw_action_result(self, raw_batch, include_future=False, **kwargs):
        self.calls.append((raw_batch, include_future, kwargs))
        return dict(self.result)


def _verified_result(budget=2):
    return {
        "actions": np.zeros((1, 16, 7), dtype=np.float32),
        "future_image_predictions": [{"primary": np.zeros((4, 4, 3), dtype=np.uint8)}],
        "requested_video_steps": budget,
        "requested_action_steps": budget,
        "effective_video_steps": budget,
        "effective_action_steps": budget,
        "matched_budget_verified": True,
        "observed_joint_nfe": budget,
        "cosmos_repo_commit": "1eb8457072b4a1adfe1f83c3076e4aa5452cbab2",
        "cosmos_source_sha256": (
            "c8cf94e18f840dda55afa162d6f6b0a4cada36fbbd8bbb45bf27c131f940f980"
        ),
    }


@pytest.mark.parametrize("budget", (1, 2, 4))
def test_official_adapter_returns_raw_actions_and_verified_metadata(budget):
    teacher = _Teacher(_verified_result(budget), configured_steps=budget)
    adapter = OfficialTeacherMatchedBudgetAdapter(
        teacher=teacher,
        video_steps=budget,
        action_steps=budget,
        include_future=True,
    )

    result = adapter.infer_raw({"obs": "raw"})

    assert result["actions"].shape == (1, 16, 7)
    assert result["effective_action_steps"] == budget
    assert result["effective_video_steps"] == budget
    assert result["cosmos_repo_commit"] == "1eb8457072b4a1adfe1f83c3076e4aa5452cbab2"
    assert "future_image_predictions" in result
    assert teacher.calls == [
        (
            {"obs": "raw"},
            True,
            {"video_steps": budget, "action_steps": budget},
        )
    ]


@pytest.mark.parametrize("missing", ("cosmos_repo_commit", "cosmos_source_sha256"))
def test_official_adapter_requires_audited_cosmos_source_identity(missing):
    payload = _verified_result()
    payload.pop(missing)
    adapter = OfficialTeacherMatchedBudgetAdapter(
        teacher=_Teacher(payload),
        video_steps=2,
        action_steps=2,
        include_future=True,
    )

    with pytest.raises(RuntimeError, match="source|commit|missing"):
        adapter.infer_raw({"obs": "raw"})


def test_official_adapter_rejects_wrong_well_formed_cosmos_source_digest():
    payload = _verified_result()
    payload["cosmos_source_sha256"] = "c" * 64
    adapter = OfficialTeacherMatchedBudgetAdapter(
        teacher=_Teacher(payload),
        video_steps=2,
        action_steps=2,
        include_future=True,
    )

    with pytest.raises(RuntimeError, match="audited Cosmos source identity"):
        adapter.infer_raw({"obs": "raw"})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("effective_action_steps", 4),
        ("effective_video_steps", 4),
        ("matched_budget_verified", False),
    ),
)
def test_official_adapter_rejects_response_budget_mismatch(field, value):
    result = _verified_result(2)
    result[field] = value
    adapter = OfficialTeacherMatchedBudgetAdapter(
        teacher=_Teacher(result),
        video_steps=2,
        action_steps=2,
        include_future=True,
    )

    with pytest.raises(RuntimeError, match="verified matched budget"):
        adapter.infer_raw({})


def test_official_adapter_rejects_missing_effective_k_metadata():
    result = _verified_result(2)
    result.pop("effective_action_steps")
    adapter = OfficialTeacherMatchedBudgetAdapter(
        teacher=_Teacher(result),
        video_steps=2,
        action_steps=2,
        include_future=True,
    )

    with pytest.raises(RuntimeError, match="missing"):
        adapter.infer_raw({})


def test_official_adapter_rejects_config_echo_without_observed_joint_nfe():
    result = _verified_result(2)
    result.pop("observed_joint_nfe")
    adapter = OfficialTeacherMatchedBudgetAdapter(
        teacher=_Teacher(result),
        video_steps=2,
        action_steps=2,
        include_future=True,
    )

    with pytest.raises(RuntimeError, match="missing"):
        adapter.infer_raw({})


def test_official_adapter_rejects_student_metadata_leakage():
    result = _verified_result(2)
    result["student_steps"] = 2
    adapter = OfficialTeacherMatchedBudgetAdapter(
        teacher=_Teacher(result),
        video_steps=2,
        action_steps=2,
        include_future=True,
    )

    with pytest.raises(RuntimeError, match="student"):
        adapter.infer_raw({})


def test_official_adapter_refuses_teacher_configured_with_default_five():
    adapter = OfficialTeacherMatchedBudgetAdapter(
        teacher=_Teacher(_verified_result(2), configured_steps=5),
        video_steps=2,
        action_steps=2,
        include_future=True,
    )

    with pytest.raises(RuntimeError, match="configured"):
        adapter.infer_raw({})
