import json

import pytest

from evaluation.libero.write_lingbotva_eval_contract import build_eval_contract


def test_default_contract_keeps_stage1_distinct_from_missing_naive(tmp_path):
    contract = build_eval_contract(
        teacher=tmp_path / "teacher",
        stage1_only=tmp_path / "stage1",
        stage2=tmp_path / "stage2",
        naive=None,
        episodes_per_task=10,
    )

    assert contract["models"]["stage1_only"]["status"] == "evaluated"
    assert contract["models"]["stage1_only"]["distilled"] is True
    assert contract["models"]["naive_composition"] == {
        "status": "missing",
        "distilled": False,
        "reason": "no explicit no-distillation naive checkpoint supplied",
    }
    assert contract["excluded"]["default_teacher_20_50"] is True


def test_contract_rejects_stage1_checkpoint_as_naive(tmp_path):
    stage1 = tmp_path / "stage1"
    with pytest.raises(ValueError, match="Stage-I-only"):
        build_eval_contract(
            teacher=tmp_path / "teacher",
            stage1_only=stage1,
            stage2=tmp_path / "stage2",
            naive=stage1,
            episodes_per_task=10,
        )
