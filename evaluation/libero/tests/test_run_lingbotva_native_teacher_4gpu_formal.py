import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "evaluation" / "libero" / "run_lingbotva_native_teacher_4gpu_formal.sh"


def _worker_fields(line):
    return dict(field.split("=", 1) for field in line.split()[1:])


def test_four_gpu_formal_wrapper_preserves_the_fixed_formal_contract(tmp_path):
    """The short user-facing launcher must expand to the 4-GPU 1/2/4 plan."""
    checkpoint = tmp_path / "base" / "transformer"
    checkpoint.mkdir(parents=True)
    output_root = tmp_path / "fresh-results"
    env = {
        **os.environ,
        "CHECK_ONLY": "1",
        "CHECKPOINT": str(checkpoint),
        "OUTPUT_ROOT": str(output_root),
    }

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    workers = [
        _worker_fields(line)
        for line in result.stdout.splitlines()
        if line.startswith("WORKER ")
    ]
    assert len(workers) == 12
    assert "RESULT_ROOT path=" + str(output_root / "teacher_native") in result.stdout
    assert "CHECK_ONLY=1: verified 12 dynamic workers (40 tasks x budgets 1,2,4)." in result.stdout
    assert not output_root.exists()

    for steps in ("1", "2", "4"):
        budget_workers = [worker for worker in workers if worker["steps"] == steps]
        assert len(budget_workers) == 4
        assert {worker["gpu"] for worker in budget_workers} == {"0", "1", "2", "3"}
        assert {worker["episodes"] for worker in budget_workers} == {"50"}
        assert {worker["client_flag"] for worker in budget_workers} == {"--no-save-video"}
        assert {worker["server_flag"] for worker in budget_workers} == {
            "--no-save-debug-tensors"
        }
