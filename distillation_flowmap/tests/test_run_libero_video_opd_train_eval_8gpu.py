import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "distillation_flowmap" / "run_libero_video_opd_train_eval_8gpu.sh"


def test_dry_run_serializes_train_and_three_matched_evaluations(tmp_path):
    teacher = tmp_path / "teacher"
    stage1 = tmp_path / "stage1"
    for path in (teacher / "transformer", stage1 / "target_student" / "transformer"):
        path.mkdir(parents=True)
    env = os.environ.copy()
    env.update(
        {
            "TEACHER_MODEL_PATH": str(teacher),
            "STAGE1_CKPT": str(stage1),
            "OUTPUT_DIR": str(tmp_path / "stage2"),
            "EVAL_OUTPUT_ROOT": str(tmp_path / "eval"),
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--phase",
            "all",
            "--steps",
            "10000",
            "--episodes",
            "10",
            "--master-port",
            "29659",
            "--eval-master-port-base",
            "29680",
            "--eval-ws-port-base",
            "29780",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = result.stdout.splitlines()
    assert sum(line.startswith("TRAIN ") for line in lines) == 1
    eval_lines = [line for line in lines if line.startswith("EVAL ")]
    assert len(eval_lines) == 3
    assert [line.split()[1] for line in eval_lines] == ["teacher", "stage1", "stage2"]
    assert all("matched_budgets=1/1,2/2,4/4" in line for line in eval_lines)
    assert str(teacher / "transformer") in eval_lines[0]
    assert str(stage1 / "target_student" / "transformer") in eval_lines[1]
    assert str(tmp_path / "stage2" / "checkpoints" / "step_10000" / "target_student" / "transformer") in eval_lines[2]
