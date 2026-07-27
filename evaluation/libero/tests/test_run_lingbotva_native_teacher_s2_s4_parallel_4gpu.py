import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "evaluation"
    / "libero"
    / "run_lingbotva_native_teacher_s2_s4_parallel_4gpu.sh"
)


def _worker_fields(line):
    return dict(field.split("=", 1) for field in line.split()[1:])


def test_check_only_launches_isolated_s2_and_s4_four_worker_plans(tmp_path):
    checkpoint = tmp_path / "base" / "transformer"
    checkpoint.mkdir(parents=True)
    s2_root = tmp_path / "fresh-s2-results"
    s4_root = tmp_path / "fresh-s4-results"
    env = {
        **os.environ,
        "CHECK_ONLY": "1",
        "CHECKPOINT": str(checkpoint),
        "S2_OUTPUT_ROOT": str(s2_root),
        "S4_OUTPUT_ROOT": str(s4_root),
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
    assert len(worker_lines) == 8
    workers = [_worker_fields(line) for line in worker_lines]
    assert {worker["steps"] for worker in workers} == {"2", "4"}
    s2 = [worker for worker in workers if worker["steps"] == "2"]
    s4 = [worker for worker in workers if worker["steps"] == "4"]
    assert {(w["gpu"], w["replica"]) for w in s2} == {
        ("0", "0"),
        ("0", "1"),
        ("1", "0"),
        ("1", "1"),
    }
    assert {(w["gpu"], w["replica"]) for w in s4} == {
        ("2", "0"),
        ("2", "1"),
        ("3", "0"),
        ("3", "1"),
    }
    assert {int(w["master_port"]) for w in s2} == set(range(34680, 34684))
    assert {int(w["ws_port"]) for w in s2} == set(range(34780, 34784))
    assert {int(w["master_port"]) for w in s4} == set(range(34880, 34884))
    assert {int(w["ws_port"]) for w in s4} == set(range(34980, 34984))
    for group, output_root in ((s2, s2_root), (s4, s4_root)):
        assert len({worker["save_root"] for worker in group}) == 4
        assert len({worker["results_root"] for worker in group}) == 4
        assert len({worker["latency_jsonl"] for worker in group}) == 4
        for worker in group:
            assert worker["task_queue"] == "40"
            assert worker["claim_mode"] == "mkdir"
            assert worker["client_flag"] == "--no-save-video"
            assert worker["server_flag"] == "--no-save-debug-tensors"
            assert Path(worker["save_root"]).is_relative_to(output_root)
            assert Path(worker["results_root"]).is_relative_to(output_root)
            assert Path(worker["latency_jsonl"]).is_relative_to(output_root)
    assert result.stdout.count("CHECK_ONLY=1: verified 4 dynamic workers") == 2
    assert not s2_root.exists()
    assert not s4_root.exists()
