import os
import subprocess
import textwrap
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "evaluation"
    / "libero"
    / "run_lingbotva_native_teacher_4suite_24_eval_8gpu_dynamic.sh"
)


def _worker_fields(line):
    return dict(field.split("=", 1) for field in line.split()[1:])


def _run_check_only(
    tmp_path, gpu_ids="0,1,2,3,4,5,6,7", budgets=None, replicas_per_gpu=None
):
    checkpoint = tmp_path / "base" / "transformer"
    checkpoint.mkdir(parents=True, exist_ok=True)
    return _run_launcher(
        tmp_path,
        checkpoint=checkpoint,
        output_root=tmp_path / "results",
        check_only=True,
        gpu_ids=gpu_ids,
        budgets=budgets,
        replicas_per_gpu=replicas_per_gpu,
    )


def _run_launcher(
    tmp_path,
    *,
    checkpoint,
    output_root,
    check_only,
    gpu_ids="0,1,2,3,4,5,6,7",
    replicas_per_gpu=None,
    budgets=None,
    master_port="32680",
    ws_port="32780",
    extra_env=None,
    timeout=30,
):
    env = {**os.environ, **(extra_env or {})}
    if check_only:
        env["CHECK_ONLY"] = "1"
    else:
        env.pop("CHECK_ONLY", None)
    args = [
        "bash",
        str(SCRIPT),
        "--checkpoint",
        str(checkpoint),
        "--output-root",
        str(output_root),
        "--gpu-ids",
        gpu_ids,
        "--master-port-base",
        master_port,
        "--ws-port-base",
        ws_port,
    ]
    if budgets is not None:
        args.extend(["--budgets", budgets])
    if replicas_per_gpu is not None:
        args.extend(["--replicas-per-gpu", replicas_per_gpu])
    return subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_check_only_defaults_to_a_full_three_budget_dynamic_rerun_without_artifacts(tmp_path):
    """Removing either opt-out flag or a planned GPU must fail this contract."""
    result = _run_check_only(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    worker_lines = [
        line for line in result.stdout.splitlines() if line.startswith("WORKER ")
    ]
    assert len(worker_lines) == 24
    assert not (tmp_path / "results").exists()

    workers = [_worker_fields(line) for line in worker_lines]
    assert {worker["steps"] for worker in workers} == {"1", "2", "4"}
    for steps in ("1", "2", "4"):
        budget_workers = [worker for worker in workers if worker["steps"] == steps]
        assert len(budget_workers) == 8
        assert {worker["task_queue"] for worker in budget_workers} == {"40"}
        assert {worker["gpu"] for worker in budget_workers} == {
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
        }
        assert {worker["claim_mode"] for worker in budget_workers} == {"mkdir"}
        assert {worker["client_flag"] for worker in budget_workers} == {
            "--no-save-video"
        }
        assert {worker["server_flag"] for worker in budget_workers} == {
            "--no-save-debug-tensors"
        }
        assert len({worker["master_port"] for worker in budget_workers}) == 8
        assert len({worker["ws_port"] for worker in budget_workers}) == 8
        assert len({worker["save_root"] for worker in budget_workers}) == 8
        assert all(worker["save_root"].endswith("/server") for worker in budget_workers)
        assert len({worker["results_root"] for worker in budget_workers}) == 8
        assert all(
            worker["results_root"].endswith("/results") for worker in budget_workers
        )
        assert len({worker["latency_jsonl"] for worker in budget_workers}) == 8


def test_check_only_accepts_a_four_gpu_dynamic_plan(tmp_path):
    """A four-GPU allocation must retain dynamic claims and all formal artifacts."""
    result = _run_check_only(tmp_path, gpu_ids="0,1,2,3")

    assert result.returncode == 0, result.stdout + result.stderr
    workers = [
        _worker_fields(line)
        for line in result.stdout.splitlines()
        if line.startswith("WORKER ")
    ]
    assert len(workers) == 12
    assert "CHECK_ONLY=1: verified 12 dynamic workers" in result.stdout
    for steps in ("1", "2", "4"):
        budget_workers = [worker for worker in workers if worker["steps"] == steps]
        assert len(budget_workers) == 4
        assert {worker["gpu"] for worker in budget_workers} == {"0", "1", "2", "3"}
        assert {worker["task_queue"] for worker in budget_workers} == {"40"}
        assert {worker["claim_mode"] for worker in budget_workers} == {"mkdir"}
        assert {worker["client_flag"] for worker in budget_workers} == {
            "--no-save-video"
        }
        assert {worker["server_flag"] for worker in budget_workers} == {
            "--no-save-debug-tensors"
        }
        assert len({worker["master_port"] for worker in budget_workers}) == 4
        assert len({worker["ws_port"] for worker in budget_workers}) == 4


def test_check_only_expands_four_gpus_to_two_independent_replicas(tmp_path):
    result = _run_check_only(tmp_path, gpu_ids="0,1,2,3", replicas_per_gpu="2")

    assert result.returncode == 0, result.stdout + result.stderr
    workers = [
        _worker_fields(line)
        for line in result.stdout.splitlines()
        if line.startswith("WORKER ")
    ]
    assert len(workers) == 24
    for steps in ("1", "2", "4"):
        lanes = [worker for worker in workers if worker["steps"] == steps]
        assert len(lanes) == 8
        assert {lane["gpu"] for lane in lanes} == {"0", "1", "2", "3"}
        assert {lane["replica"] for lane in lanes} == {"0", "1"}
        assert len({lane["master_port"] for lane in lanes}) == 8
        assert len({lane["ws_port"] for lane in lanes}) == 8
        assert len({lane["save_root"] for lane in lanes}) == 8


@pytest.mark.parametrize("replicas_per_gpu", ("0", "3", "two"))
def test_check_only_rejects_invalid_replica_counts(tmp_path, replicas_per_gpu):
    result = _run_check_only(tmp_path, replicas_per_gpu=replicas_per_gpu)

    assert result.returncode != 0
    assert "--replicas-per-gpu must be 1 or 2" in result.stderr


def test_check_only_rejects_invalid_gpu_lists_outside_one_to_eight_workers(tmp_path):
    for gpu_ids in ("", "0,1,2,3,4,5,6,7,8", "0,1,2,3,4,5,6,7,"):
        result = _run_check_only(tmp_path, gpu_ids=gpu_ids)

        assert result.returncode != 0
        assert "between 1 and 8 unique" in result.stderr
        assert "non-negative integers" in result.stderr


def test_check_only_rejects_negative_gpu_identifiers(tmp_path):
    result = _run_check_only(tmp_path, gpu_ids="-1,1,2,3,4,5,6,7")

    assert result.returncode != 0
    assert "non-negative integers" in result.stderr


def test_check_only_rejects_port_ranges_that_exceed_tcp_limit(tmp_path):
    result = _run_launcher(
        tmp_path,
        checkpoint=tmp_path / "base" / "transformer",
        output_root=tmp_path / "results",
        check_only=True,
        master_port="65530",
        ws_port="65400",
    )

    assert result.returncode != 0
    assert "must end at or below 65535" in result.stderr


def test_check_only_refuses_an_output_root_that_overlaps_checkpoint(tmp_path):
    checkpoint = tmp_path / "base" / "transformer"
    checkpoint.mkdir(parents=True)

    result = _run_launcher(
        tmp_path,
        checkpoint=checkpoint,
        output_root=checkpoint,
        check_only=True,
    )

    assert result.returncode != 0
    assert "must not overlap checkpoint" in result.stderr


def test_normal_run_refuses_an_existing_model_root_before_starting_workers(tmp_path):
    checkpoint = tmp_path / "base" / "transformer"
    checkpoint.mkdir(parents=True)
    output_root = tmp_path / "results"
    (output_root / "teacher_native").mkdir(parents=True)

    result = _run_launcher(
        tmp_path,
        checkpoint=checkpoint,
        output_root=output_root,
        check_only=False,
        extra_env={
            "SERVER_PYTHON": "/bin/false",
            "CLIENT_PYTHON": "/bin/false",
            "SERVER_WAIT_SECONDS": "0",
        },
    )

    assert result.returncode != 0
    assert "Refusing to reuse existing native result root" in result.stderr


def test_first_worker_failure_terminates_sibling_server_groups_promptly(tmp_path):
    checkpoint = tmp_path / "base" / "transformer"
    checkpoint.mkdir(parents=True)
    output_root = tmp_path / "results"
    pid_file = tmp_path / "server-pids.txt"
    fake_server = tmp_path / "fake_server.py"
    fake_client = tmp_path / "fake_client.py"
    master_port = 43000 + (os.getpid() % 1000)
    ws_port = master_port + 100

    fake_server.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import os
            import sys
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            if sys.argv[1:3] != ["-m", "torch.distributed.run"]:
                raise SystemExit(0)
            port = int(sys.argv[sys.argv.index("--port") + 1])
            with open(os.environ["FAKE_SERVER_PID_FILE"], "a") as handle:
                handle.write(f"{os.getpid()}\\n")

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200 if self.path == "/healthz" else 404)
                    self.end_headers()
                def log_message(self, *_args):
                    pass

            ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
            """
        )
    )
    fake_client.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import os
            import sys
            import time

            port = sys.argv[sys.argv.index("--port") + 1]
            if port == os.environ["FAKE_FAIL_PORT"]:
                raise SystemExit(1)
            time.sleep(10)
            """
        )
    )
    fake_server.chmod(0o755)
    fake_client.chmod(0o755)

    started_at = time.monotonic()
    result = _run_launcher(
        tmp_path,
        checkpoint=checkpoint,
        output_root=output_root,
        check_only=False,
        gpu_ids="0",
        replicas_per_gpu="2",
        master_port=str(master_port),
        ws_port=str(ws_port),
        extra_env={
            "SERVER_PYTHON": str(fake_server),
            "CLIENT_PYTHON": str(fake_client),
            "FAKE_SERVER_PID_FILE": str(pid_file),
            "FAKE_FAIL_PORT": str(ws_port),
            "SERVER_WAIT_SECONDS": "0",
            "SERVER_READY_TIMEOUT_SECONDS": "5",
            "WORKER_STOP_TIMEOUT_SECONDS": "1",
        },
        timeout=20,
    )
    elapsed = time.monotonic() - started_at

    assert result.returncode != 0
    assert elapsed < 6, result.stdout + result.stderr
    assert pid_file.exists()
    for pid in {int(line) for line in pid_file.read_text().splitlines()}:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise AssertionError(f"orphaned fake server process: {pid}")


def test_one_budget_claims_every_task_once_and_reaches_the_merger(tmp_path):
    checkpoint = tmp_path / "base" / "transformer"
    checkpoint.mkdir(parents=True)
    output_root = tmp_path / "results"
    call_file = tmp_path / "client-calls.txt"
    pid_file = tmp_path / "server-pids.txt"
    fake_server = tmp_path / "fake_server.py"
    fake_server_child = tmp_path / "fake_server_child.py"
    fake_client = tmp_path / "fake_client.py"
    master_port = 45000 + (os.getpid() % 1000)
    ws_port = master_port + 100

    fake_server_child.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import os
            import sys
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            with open(os.environ["FAKE_SERVER_PID_FILE"], "a") as handle:
                handle.write(f"{os.getpid()}\\n")

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200 if self.path == "/healthz" else 404)
                    self.end_headers()
                def log_message(self, *_args):
                    pass

            ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
            """
        )
    )
    fake_server.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import os
            import subprocess
            import sys
            import time
            from pathlib import Path

            if sys.argv[1:3] != ["-m", "torch.distributed.run"]:
                output = Path(sys.argv[sys.argv.index("--output") + 1])
                output.write_text("{}\\n")
                raise SystemExit(0)
            port = int(sys.argv[sys.argv.index("--port") + 1])
            with open(os.environ["FAKE_SERVER_PID_FILE"], "a") as handle:
                handle.write(f"{os.getpid()}\\n")
            child = subprocess.Popen([sys.executable, os.environ["FAKE_SERVER_CHILD"], str(port)])
            time.sleep(3)
            """
        )
    )
    fake_client.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import os
            import sys
            import time

            suite = sys.argv[sys.argv.index("--libero-benchmark") + 1]
            task_range = sys.argv.index("--task-range")
            with open(os.environ["FAKE_CLIENT_CALL_FILE"], "a") as handle:
                handle.write(f"{suite}:{sys.argv[task_range + 1]}:{sys.argv[task_range + 2]}\\n")
            time.sleep(0.8)
            """
        )
    )
    fake_server.chmod(0o755)
    fake_server_child.chmod(0o755)
    fake_client.chmod(0o755)

    result = _run_launcher(
        tmp_path,
        checkpoint=checkpoint,
        output_root=output_root,
        check_only=False,
        budgets="1",
        gpu_ids="0,1",
        replicas_per_gpu="2",
        master_port=str(master_port),
        ws_port=str(ws_port),
        extra_env={
            "SERVER_PYTHON": str(fake_server),
            "CLIENT_PYTHON": str(fake_client),
            "FAKE_CLIENT_CALL_FILE": str(call_file),
            "FAKE_SERVER_PID_FILE": str(pid_file),
            "FAKE_SERVER_CHILD": str(fake_server_child),
            "SERVER_READY_TIMEOUT_SECONDS": "5",
            "WORKER_STOP_TIMEOUT_SECONDS": "3",
        },
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    expected_calls = {
        f"{suite}:{task_idx}:{task_idx + 1}"
        for suite in ("libero_spatial", "libero_goal", "libero_object", "libero_10")
        for task_idx in range(10)
    }
    calls = call_file.read_text().splitlines()
    assert set(calls) == expected_calls
    assert len(calls) == 40
    assert len(set(calls)) == 40
    assert (output_root / "teacher_native" / "steps_1" / "summary.json").exists()
    for pid in {int(line) for line in pid_file.read_text().splitlines()}:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise AssertionError(f"orphaned fake server-tree process: {pid}")


def test_check_only_supports_a_clean_three_budget_rerun(tmp_path):
    result = _run_check_only(tmp_path, budgets="1,2,4")

    assert result.returncode == 0, result.stdout + result.stderr
    worker_lines = [
        line for line in result.stdout.splitlines() if line.startswith("WORKER ")
    ]
    assert len(worker_lines) == 24
    assert {_worker_fields(line)["steps"] for line in worker_lines} == {"1", "2", "4"}
