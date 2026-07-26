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
STUDENT_SCRIPT = (
    ROOT / "evaluation" / "libero" / "run_lingbotva_4suite_124_eval_8gpu.sh"
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


def run_formal_with_stubs(tmp_path, output_root):
    base_model = tmp_path / "base"
    checkpoint = base_model / "transformer"
    for component in ("transformer", "vae", "tokenizer", "text_encoder"):
        (base_model / component).mkdir(parents=True, exist_ok=True)

    server_marker = tmp_path / "server-invoked"
    server_stub = tmp_path / "server-python-stub.sh"
    server_stub.write_text('#!/bin/sh\n: > "$SERVER_MARKER"\n')
    server_stub.chmod(0o755)
    client_stub = tmp_path / "client-python-stub.sh"
    client_stub.write_text("#!/bin/sh\nexit 0\n")
    client_stub.chmod(0o755)

    env = os.environ.copy()
    env.pop("CHECK_ONLY", None)
    env.update(
        {
            "CLIENT_PYTHON": str(client_stub),
            "SERVER_MARKER": str(server_marker),
            "SERVER_PYTHON": str(server_stub),
            "SERVER_WAIT_SECONDS": "0",
            "WAN22_PRETRAINED_PATH": str(base_model),
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--checkpoint",
            str(checkpoint),
            "--output-root",
            str(output_root),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, server_marker


def worker_fields(line):
    return dict(field.split("=", 1) for field in line.split()[1:])


def test_native_default_ports_are_distinct_from_student_defaults(tmp_path):
    native_result = run_check_only(tmp_path)
    checkpoint = tmp_path / "base" / "transformer"
    env = os.environ.copy()
    env["CHECK_ONLY"] = "1"
    student_result = subprocess.run(
        [
            "bash",
            str(STUDENT_SCRIPT),
            "--model-name",
            "student",
            "--checkpoint",
            str(checkpoint),
            "--output-root",
            str(tmp_path / "student_results"),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert native_result.returncode == 0, native_result.stdout + native_result.stderr
    assert student_result.returncode == 0, (
        student_result.stdout + student_result.stderr
    )
    native_worker = next(
        worker_fields(line)
        for line in native_result.stdout.splitlines()
        if line.startswith("WORKER ")
    )
    student_worker = next(
        worker_fields(line)
        for line in student_result.stdout.splitlines()
        if line.startswith("WORKER ")
    )
    native_ports = (native_worker["master_port"], native_worker["ws_port"])
    student_ports = (student_worker["master_port"], student_worker["ws_port"])

    assert native_ports == ("30680", "30780")
    assert student_ports == ("29680", "29780")
    assert native_ports != student_ports


def test_launcher_rejects_nonempty_model_root_before_server_invocation(tmp_path):
    output_root = tmp_path / "results"
    model_root = output_root / "teacher_native"
    model_root.mkdir(parents=True)
    sentinel = model_root / "existing-result.json"
    sentinel.write_text('{"historical": true}\n')

    result, server_marker = run_formal_with_stubs(tmp_path, output_root)

    assert result.returncode == 2, result.stdout + result.stderr
    assert (
        f"Refusing to reuse nonempty native result root: {model_root}"
        in result.stderr
    )
    assert "WORKER " not in result.stdout
    assert not server_marker.exists()
    assert sentinel.read_text() == '{"historical": true}\n'


@pytest.mark.parametrize(
    ("precreate_model_root", "expected_state"),
    [(False, "absent"), (True, "empty")],
)
def test_launcher_accepts_new_or_empty_model_root(
    tmp_path, precreate_model_root, expected_state
):
    output_root = tmp_path / "results"
    model_root = output_root / "teacher_native"
    if precreate_model_root:
        model_root.mkdir(parents=True)

    result, server_marker = run_formal_with_stubs(tmp_path, output_root)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        f"RESULT_ROOT path={model_root} state={expected_state} launch_allowed=yes"
        in result.stdout
    )
    assert server_marker.exists()


def test_check_only_reports_nonempty_model_root_without_modifying_it(tmp_path):
    model_root = tmp_path / "results" / "teacher_native"
    model_root.mkdir(parents=True)
    sentinel = model_root / "existing-result.json"
    sentinel.write_text('{"historical": true}\n')

    result = run_check_only(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        f"RESULT_ROOT path={model_root} state=nonempty launch_allowed=no"
        in result.stdout
    )
    assert {path.relative_to(model_root) for path in model_root.rglob("*")} == {
        Path("existing-result.json")
    }
    assert sentinel.read_text() == '{"historical": true}\n'


def test_check_only_plans_native_teacher_40_tasks_at_matched_124(tmp_path):
    result = run_check_only(
        tmp_path,
        "--gpu-ids",
        "0,1,2,3,4,5,6,7",
        "--master-port-base",
        "31680",
        "--ws-port-base",
        "31780",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "RESULT_ROOT "
        f"path={tmp_path / 'results' / 'teacher_native'} "
        "state=absent launch_allowed=yes"
    ) in result.stdout
    assert not (tmp_path / "results").exists()
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
        ("libero_10", "0:5", "0", "31680", "31780"),
        ("libero_10", "5:10", "1", "31681", "31781"),
        ("libero_spatial", "0:5", "2", "31682", "31782"),
        ("libero_spatial", "5:10", "3", "31683", "31783"),
        ("libero_object", "0:5", "4", "31684", "31784"),
        ("libero_object", "5:10", "5", "31685", "31785"),
        ("libero_goal", "0:5", "6", "31686", "31786"),
        ("libero_goal", "5:10", "7", "31687", "31787"),
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


@pytest.mark.parametrize(
    ("port_flag", "port_base"),
    [
        ("--master-port-base", "000"),
        ("--ws-port-base", "not-a-number"),
    ],
)
def test_launcher_rejects_non_positive_or_non_numeric_port_bases(
    tmp_path, port_flag, port_base
):
    result = run_check_only(tmp_path, port_flag, port_base)

    assert result.returncode != 0
    assert "must be a positive integer" in result.stderr
