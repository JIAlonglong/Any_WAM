import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "evaluation" / "libero" / "run_lingbotva_4suite_124_eval_8gpu.sh"


def test_dry_run_covers_40_tasks_and_matched_124_budgets(tmp_path):
    ckpt = tmp_path / "transformer"
    ckpt.mkdir()
    env = os.environ.copy()
    env["CHECK_ONLY"] = "1"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model-name",
            "teacher",
            "--checkpoint",
            str(ckpt),
            "--episodes",
            "10",
            "--output-root",
            str(tmp_path / "eval"),
            "--master-port-base",
            "31000",
            "--ws-port-base",
            "32000",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    worker_lines = [line for line in result.stdout.splitlines() if line.startswith("WORKER ")]
    assert len(worker_lines) == 24
    for steps in (1, 2, 4):
        lines = [line for line in worker_lines if f"steps={steps} " in line]
        assert len(lines) == 8
        assert all(f"video_steps={steps} action_steps={steps}" in line for line in lines)
    for suite in ("libero_10", "libero_spatial", "libero_object", "libero_goal"):
        suite_lines = [line for line in worker_lines if f"suite={suite} " in line]
        assert len(suite_lines) == 6
        assert {line.split("tasks=")[1].split()[0] for line in suite_lines} == {"0:5", "5:10"}


def test_requires_exactly_eight_unique_gpus(tmp_path):
    ckpt = tmp_path / "transformer"
    ckpt.mkdir()
    env = os.environ.copy()
    env.update({"CHECK_ONLY": "1", "GPU_IDS": "0,1,2,3"})
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model-name",
            "teacher",
            "--checkpoint",
            str(ckpt),
            "--output-root",
            str(tmp_path / "eval"),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "exactly 8 unique GPUs" in result.stderr
