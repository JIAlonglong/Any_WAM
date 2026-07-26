import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "evaluation"
    / "libero"
    / "run_lingbotva_native_teacher_4suite_124_eval_8gpu.sh"
)


def test_check_only_plans_native_teacher_40_tasks_at_matched_124(tmp_path):
    checkpoint = tmp_path / "base" / "transformer"
    checkpoint.mkdir(parents=True)
    env = os.environ.copy()
    env["CHECK_ONLY"] = "1"
    result = subprocess.run(
        [
            "bash", str(SCRIPT),
            "--checkpoint", str(checkpoint),
            "--output-root", str(tmp_path / "results"),
            "--episodes", "50",
            "--gpu-ids", "0,1,2,3,4,5,6,7",
            "--master-port-base", "30680",
            "--ws-port-base", "30780",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("WORKER ") == 24
    assert "model=teacher_native" in result.stdout
    assert "backend=native_teacher" in result.stdout
    assert "video_steps=1 action_steps=1" in result.stdout
    assert "video_steps=2 action_steps=2" in result.stdout
    assert "video_steps=4 action_steps=4" in result.stdout
    assert "tasks=0:5" in result.stdout
    assert "tasks=5:10" in result.stdout


def test_launcher_rejects_duplicate_gpus(tmp_path):
    env = os.environ.copy()
    env["CHECK_ONLY"] = "1"
    result = subprocess.run(
        [
            "bash", str(SCRIPT),
            "--checkpoint", str(tmp_path / "transformer"),
            "--output-root", str(tmp_path / "results"),
            "--gpu-ids", "0,1,2,3,4,5,6,6",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "8 unique GPUs" in result.stderr
