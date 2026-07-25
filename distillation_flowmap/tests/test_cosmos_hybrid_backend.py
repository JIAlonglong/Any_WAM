import json
from pathlib import Path

import pytest

from distillation_flowmap.cosmos_hybrid_backend import (
    validate_cosmos_teacher_model_path,
    validate_wan_student_base_model_path,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_accepts_realistic_wan_student_and_cosmos_teacher_layouts(tmp_path):
    wan_base = tmp_path / "lingbot-va" / "checkpoints" / "libero"
    _write_json(
        wan_base / "transformer" / "config.json",
        {"_class_name": "WanTransformer3DModel"},
    )
    cosmos_teacher = tmp_path / "Cosmos-Policy-LIBERO-Predict2-2B"
    _write_json(cosmos_teacher / "config.json", {"model_type": "cosmos-policy"})
    for name in (
        "Cosmos-Policy-LIBERO-Predict2-2B.pt",
        "libero_dataset_statistics.json",
        "libero_t5_embeddings.pkl",
    ):
        (cosmos_teacher / name).write_bytes(b"fixture")

    assert validate_wan_student_base_model_path(wan_base) == wan_base.resolve()
    assert validate_cosmos_teacher_model_path(cosmos_teacher) == cosmos_teacher.resolve()


def test_rejects_cosmos_policy_root_as_wan_student(tmp_path):
    cosmos_root = tmp_path / "Cosmos-Policy-LIBERO-Predict2-2B"
    _write_json(cosmos_root / "config.json", {"model_type": "cosmos-policy"})

    with pytest.raises(ValueError, match="wan_flowmap.*WanTransformer3DModel"):
        validate_wan_student_base_model_path(cosmos_root)


def test_rejects_wan_transformer_as_cosmos_teacher(tmp_path):
    wan_base = tmp_path / "lingbot-va"
    _write_json(
        wan_base / "transformer" / "config.json",
        {"_class_name": "WanTransformer3DModel"},
    )

    with pytest.raises(ValueError, match="Cosmos Policy teacher config"):
        validate_cosmos_teacher_model_path(wan_base)
