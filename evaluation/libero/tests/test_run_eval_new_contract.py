import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "evaluation" / "libero" / "run_eval_new.sh"


def _env(tmp_path, suite):
    output = tmp_path / "output"
    student = output / "checkpoints" / "step_1" / "target_student" / "transformer"
    student.mkdir(parents=True)
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "OUTPUT_ROOT": str(output),
            "TEACHER_CKPT": str(teacher),
            "LIBERO_BENCHMARK": suite,
            "CHECK_ONLY": "1",
            "EVAL_MODE": "success",
            "TEST_NUM": "2",
            "TASK_START": "0",
            "TASK_END": "5",
            "NUM_STEPS": "2",
            "ACTION_NUM_STEPS": "2",
            "CUDA_VISIBLE_DEVICES": "7",
            "SAVE_ROOT": str(tmp_path / "results"),
        }
    )
    return env


@pytest.mark.parametrize(
    "suite", ["libero_10", "libero_spatial", "libero_object", "libero_goal"]
)
def test_check_only_propagates_suite_joint_steps_and_gpu(tmp_path, suite):
    result = subprocess.run(
        ["bash", str(SCRIPT), "step_1", "target_student"],
        cwd=ROOT,
        env=_env(tmp_path, suite),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Benchmark:      {suite}" in result.stdout
    assert "Video steps:    2" in result.stdout
    assert "Action steps:   2" in result.stdout
    assert "Visible GPUs:   7" in result.stdout
    assert "Base model:" in result.stdout


def test_unknown_suite_is_rejected(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT), "step_1", "target_student"],
        cwd=ROOT,
        env=_env(tmp_path, "libero_unknown"),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Unsupported LIBERO_BENCHMARK" in result.stderr
