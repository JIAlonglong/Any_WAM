import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "evaluation"
    / "libero"
    / "run_lingbotva_native_teacher_4suite_124_eval_8gpu.sh"
)


def run_check_only(tmp_path, *args):
    checkpoint = tmp_path / "base" / "transformer"
    checkpoint.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CHECK_ONLY"] = "1"
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--checkpoint",
            str(checkpoint),
            "--output-root",
            str(tmp_path / "results"),
            *args,
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def worker_fields(line):
    return dict(field.split("=", 1) for field in line.split()[1:])


def test_check_only_plans_native_teacher_40_tasks_at_matched_124(tmp_path):
    result = run_check_only(
        tmp_path,
        "--gpu-ids",
        "0,1,2,3,4,5,6,7",
        "--master-port-base",
        "30680",
        "--ws-port-base",
        "30780",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    worker_lines = [
        line for line in result.stdout.splitlines() if line.startswith("WORKER ")
    ]
    assert len(worker_lines) == 24
    assert all(worker_fields(line)["model"] == "teacher_native" for line in worker_lines)
    assert all(worker_fields(line)["backend"] == "native_teacher" for line in worker_lines)
    assert all(worker_fields(line)["episodes"] == "50" for line in worker_lines)

    for steps in ("1", "2", "4"):
        lines = [line for line in worker_lines if worker_fields(line)["steps"] == steps]
        assert len(lines) == 8
        assert all(
            worker_fields(line)["video_steps"] == steps
            and worker_fields(line)["action_steps"] == steps
            for line in lines
        )

    expected_step_one = [
        ("libero_10", "0:5", "0", "30680", "30780"),
        ("libero_10", "5:10", "1", "30681", "30781"),
        ("libero_spatial", "0:5", "2", "30682", "30782"),
        ("libero_spatial", "5:10", "3", "30683", "30783"),
        ("libero_object", "0:5", "4", "30684", "30784"),
        ("libero_object", "5:10", "5", "30685", "30785"),
        ("libero_goal", "0:5", "6", "30686", "30786"),
        ("libero_goal", "5:10", "7", "30687", "30787"),
    ]
    step_one = [
        worker_fields(line) for line in worker_lines if worker_fields(line)["steps"] == "1"
    ]
    assert [
        (
            fields["suite"],
            fields["tasks"],
            fields["gpu"],
            fields["master_port"],
            fields["ws_port"],
        )
        for fields in step_one
    ] == expected_step_one


def test_launcher_rejects_duplicate_gpus(tmp_path):
    result = run_check_only(tmp_path, "--gpu-ids", "0,1,2,3,4,5,6,6")

    assert result.returncode != 0
    assert "8 unique GPUs" in result.stderr


@pytest.mark.parametrize(
    ("master_port_base", "ws_port_base"),
    [("30680", "30680"), ("30680", "30685")],
)
def test_launcher_rejects_overlapping_port_ranges(
    tmp_path, master_port_base, ws_port_base
):
    result = run_check_only(
        tmp_path,
        "--master-port-base",
        master_port_base,
        "--ws-port-base",
        ws_port_base,
    )

    assert result.returncode != 0
    assert "must not overlap" in result.stderr
