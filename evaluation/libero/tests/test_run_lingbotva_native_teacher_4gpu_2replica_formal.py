import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "evaluation"
    / "libero"
    / "run_lingbotva_native_teacher_4gpu_2replica_formal.sh"
)


def _worker_fields(line):
    return dict(field.split("=", 1) for field in line.split()[1:])


def test_fast_four_gpu_wrapper_expands_to_two_independent_replicas(tmp_path):
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
    worker_lines = [
        line for line in result.stdout.splitlines() if line.startswith("WORKER ")
    ]
    assert len(worker_lines) == 24
    workers = [_worker_fields(line) for line in worker_lines]
    assert "CHECK_ONLY=1: verified 24 dynamic workers" in result.stdout
    for steps in ("1", "2", "4"):
        lanes = [worker for worker in workers if worker["steps"] == steps]
        assert len(lanes) == 8
        assert {worker["replica"] for worker in lanes} == {"0", "1"}
        assert {worker["gpu"] for worker in lanes} == {"0", "1", "2", "3"}
    assert not output_root.exists()
