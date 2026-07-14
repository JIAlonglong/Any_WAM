from pathlib import Path

import pytest

from distillation_flowmap.run_cosmos_progressive_stage2 import build_stage_chunk_plan


def _plan(tmp_path, **overrides):
    defaults = {
        "stage": "s4",
        "root": tmp_path / "progressive",
        "dataset_path": Path("/tmp/libero"),
        "train_manifest": Path("/tmp/train.json"),
        "selection_manifest": Path("/tmp/selection.json"),
        "eval_pairs": Path("/tmp/pairs.json"),
        "selection_cache_dir": Path("/tmp/selection_cache"),
        "stage1_checkpoint": Path("/tmp/stage1_step5000"),
        "current_step": 0,
        "chunk_size": 250,
        "master_port": 29761,
        "train_seed": 20260714,
        "torchrun": Path("/tmp/torchrun"),
    }
    defaults.update(overrides)
    return build_stage_chunk_plan(**defaults)


def test_initial_s4_chunk_starts_from_fixed_stage1_and_preserves_full_schedule(tmp_path):
    plan = _plan(tmp_path)

    assert plan["spec"] == {"teacher_steps": 8, "student_steps": 4, "max_steps": 5000}
    assert plan["checkpoint_dir"] == tmp_path / "progressive" / "s4" / "checkpoints" / "step_250"
    assert plan["train_env"]["RESUME_FROM_PATH"] == "/tmp/stage1_step5000"
    assert plan["train_env"]["RESET_RESUME_STEP"] == "1"
    assert plan["train_env"]["RESUME_OPTIMIZER_STATE"] == "0"
    assert plan["train_env"]["MAX_TRAIN_STEPS"] == "5000"
    assert plan["train_env"]["STOP_AFTER_STEP"] == "250"
    assert plan["train_env"]["DATASET_SAMPLE_MANIFEST"] == "/tmp/train.json"
    assert plan["train_env"]["CUDA_VISIBLE_DEVICES"] == "6,7"
    assert plan["train_env"]["COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES"] == "6,7"
    assert "--nproc_per_node=2" in plan["train_argv"]
    gradient_arg = plan["train_argv"].index("--gradient-accumulation-steps")
    assert plan["train_argv"][gradient_arg + 1] == "1"
    assert plan["eval_env"]["CUDA_VISIBLE_DEVICES"] == "6"
    assert plan["eval_env"]["COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES"] == "7"
    assert "--student-steps" in plan["eval_argv"]
    assert "4" in plan["eval_argv"]


def test_resume_chunk_uses_the_previous_checkpoint_and_optimizer_state(tmp_path):
    plan = _plan(tmp_path, current_step=250)

    assert plan["checkpoint_dir"] == tmp_path / "progressive" / "s4" / "checkpoints" / "step_500"
    assert plan["train_env"]["RESUME_FROM_PATH"].endswith("s4/checkpoints/step_250")
    assert plan["train_env"]["RESET_RESUME_STEP"] == "0"
    assert plan["train_env"]["RESUME_OPTIMIZER_STATE"] == "1"


def test_later_stage_requires_an_explicit_selected_predecessor_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="selected predecessor"):
        _plan(tmp_path, stage="s2")

    plan = _plan(
        tmp_path,
        stage="s2",
        resume_from_path=Path("/tmp/s4_selected_step2500"),
    )
    assert plan["spec"] == {"teacher_steps": 4, "student_steps": 2, "max_steps": 3000}
    assert plan["train_env"]["RESUME_FROM_PATH"] == "/tmp/s4_selected_step2500"
    assert plan["train_env"]["RESET_RESUME_STEP"] == "1"


def test_standalone_opd_runner_rejects_gradient_accumulation_above_one(tmp_path):
    with pytest.raises(ValueError, match="gradient_accumulation_steps=1"):
        _plan(tmp_path, gradient_accumulation_steps=8)
