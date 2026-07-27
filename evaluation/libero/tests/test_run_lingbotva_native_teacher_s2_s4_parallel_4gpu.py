import os
import signal
import subprocess
import time
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
    checkpoint.mkdir(parents=True, exist_ok=True)
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


def _run_check_only_with_overrides(tmp_path, overrides):
    checkpoint = tmp_path / "base" / "transformer"
    checkpoint.mkdir(parents=True, exist_ok=True)
    s2_root = tmp_path / "fresh-s2-results"
    s4_root = tmp_path / "fresh-s4-results"
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env={
            **os.environ,
            "CHECK_ONLY": "1",
            "CHECKPOINT": str(checkpoint),
            "S2_OUTPUT_ROOT": str(s2_root),
            "S4_OUTPUT_ROOT": str(s4_root),
            **overrides,
        },
        text=True,
        capture_output=True,
        check=False,
    )


def test_check_only_rejects_overlapping_sibling_overrides_before_launching(tmp_path):
    shared_root = tmp_path / "shared-results"
    cases = [
        (
            {"S2_OUTPUT_ROOT": str(shared_root), "S4_OUTPUT_ROOT": str(shared_root)},
            "S2_OUTPUT_ROOT and S4_OUTPUT_ROOT must not overlap",
        ),
        (
            {"S2_GPU_IDS": "0,1", "S4_GPU_IDS": "1,2"},
            "S2_GPU_IDS and S4_GPU_IDS must not overlap",
        ),
        (
            {"S2_GPU_IDS": "01", "S4_GPU_IDS": "1"},
            "S2_GPU_IDS and S4_GPU_IDS must not overlap",
        ),
        (
            {
                "S2_MASTER_PORT_BASE": "36000",
                "S4_MASTER_PORT_BASE": "36003",
            },
            "S2 and S4 master port ranges must not overlap",
        ),
        (
            {"S2_WS_PORT_BASE": "37000", "S4_WS_PORT_BASE": "37003"},
            "S2 and S4 WebSocket port ranges must not overlap",
        ),
        (
            {
                "S2_MASTER_PORT_BASE": "036000",
                "S4_MASTER_PORT_BASE": "36003",
            },
            "S2 and S4 master port ranges must not overlap",
        ),
    ]

    for overrides, expected_error in cases:
        result = _run_check_only_with_overrides(tmp_path, overrides)

        assert result.returncode != 0
        assert expected_error in result.stderr
        assert "WORKER " not in result.stdout


def test_check_only_rejects_cross_type_and_same_workload_port_overlaps(tmp_path):
    cases = [
        {"S2_MASTER_PORT_BASE": "36000", "S4_WS_PORT_BASE": "36003"},
        {"S2_WS_PORT_BASE": "36000", "S4_MASTER_PORT_BASE": "36003"},
        {"S2_MASTER_PORT_BASE": "36000", "S2_WS_PORT_BASE": "36003"},
        {"S4_MASTER_PORT_BASE": "36000", "S4_WS_PORT_BASE": "36003"},
    ]

    for overrides in cases:
        result = _run_check_only_with_overrides(tmp_path, overrides)

        assert result.returncode != 0
        assert "port ranges must be mutually disjoint" in result.stderr
        assert "WORKER " not in result.stdout


def test_check_only_rejects_s1_result_root_overlap_before_launching(tmp_path):
    s1_root = Path(
        "/kpfs-intern/jialongliu/projects/Flash-WAM/evaluation/results/"
        "libero_teacher_native_dynamic_4gpu_formal_py38fix_20260727/teacher_native"
    )
    cases = []
    for output_variable in ("S2_OUTPUT_ROOT", "S4_OUTPUT_ROOT"):
        cases.extend(
            [
                {output_variable: str(s1_root)},
                {output_variable: str(s1_root / "descendant")},
                {output_variable: str(s1_root.parent)},
            ]
        )

    for overrides in cases:
        result = _run_check_only_with_overrides(tmp_path, overrides)

        assert result.returncode != 0
        assert "must not overlap the active S1 result root" in result.stderr
        assert "WORKER " not in result.stdout


def test_check_only_accepts_non_overlapping_sibling_overrides(tmp_path):
    result = _run_check_only_with_overrides(
        tmp_path,
        {
            "S2_GPU_IDS": "4",
            "S4_GPU_IDS": "5",
            "S2_REPLICAS_PER_GPU": "1",
            "S4_REPLICAS_PER_GPU": "1",
            "S2_MASTER_PORT_BASE": "38000",
            "S2_WS_PORT_BASE": "38100",
            "S4_MASTER_PORT_BASE": "38200",
            "S4_WS_PORT_BASE": "38300",
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    workers = [
        _worker_fields(line)
        for line in result.stdout.splitlines()
        if line.startswith("WORKER ")
    ]
    assert {(worker["steps"], worker["gpu"]) for worker in workers} == {
        ("2", "4"),
        ("4", "5"),
    }
    assert {worker["master_port"] for worker in workers} == {"38000", "38200"}
    assert {worker["ws_port"] for worker in workers} == {"38100", "38300"}


def test_check_only_rejects_overflowing_gpu_and_port_overrides(tmp_path):
    cases = [
        (
            {"S2_GPU_IDS": "18446744073709551615"},
            "S2_GPU_IDS must be a decimal GPU ID from 0 to 2147483647",
        ),
        (
            {"S2_MASTER_PORT_BASE": "18446744073709551615"},
            "S2_MASTER_PORT_BASE must be a decimal port from 1 to 65535",
        ),
    ]

    for overrides, expected_error in cases:
        result = _run_check_only_with_overrides(tmp_path, overrides)

        assert result.returncode != 0
        assert expected_error in result.stderr
        assert "WORKER " not in result.stdout


def test_sigint_terminates_only_the_live_s4_child_when_s2_has_exited(tmp_path):
    checkpoint = tmp_path / "base" / "transformer"
    checkpoint.mkdir(parents=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    signal_log = tmp_path / "signal.log"
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        """#!/bin/sh
if [ "$(basename "$1")" = "run_lingbotva_native_teacher_4gpu_2replica_formal.sh" ]; then
    if [ "$BUDGETS" = "2" ]; then
        printf 'exited budget=2 pid=%s\\n' "$$" >> "$SIGNAL_LOG"
        exit 0
    fi
    trap 'printf "terminated budget=%s pid=%s\\n" "$BUDGETS" "$$" >> "$SIGNAL_LOG"; exit 0' TERM
    printf 'ready budget=4 pid=%s\\n' "$$" >> "$SIGNAL_LOG"
    while :; do sleep 1; done
fi
exec /bin/bash "$@"
"""
    )
    fake_bash.chmod(0o755)
    env = {
        **os.environ,
        "CHECKPOINT": str(checkpoint),
        "S2_OUTPUT_ROOT": str(tmp_path / "s2-results"),
        "S4_OUTPUT_ROOT": str(tmp_path / "s4-results"),
        "SIGNAL_LOG": str(signal_log),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    process = subprocess.Popen(
        ["bash", str(SCRIPT)], cwd=ROOT, env=env, text=True
    )
    try:
        deadline = time.monotonic() + 5
        while (
            (
                not signal_log.exists()
                or "exited budget=2" not in signal_log.read_text()
                or "ready budget=4" not in signal_log.read_text()
            )
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert signal_log.exists()
        signal_text = signal_log.read_text()
        assert "exited budget=2" in signal_text
        assert "ready budget=4" in signal_text

        s2_exit_line = next(
            line
            for line in signal_text.splitlines()
            if line.startswith("exited budget=2 ")
        )
        s2_pid = int(_worker_fields(s2_exit_line)["pid"])
        s2_proc = Path("/proc") / str(s2_pid)
        deadline = time.monotonic() + 5
        while s2_proc.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not s2_proc.exists()

        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=5) != 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)

    log_lines = signal_log.read_text().splitlines()
    assert "terminated budget=4" in "\n".join(log_lines)
    assert not any("terminated budget=2" in line for line in log_lines)


def test_sigint_during_first_child_launch_cleans_up_both_captured_children(tmp_path):
    checkpoint = tmp_path / "base" / "transformer"
    checkpoint.mkdir(parents=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    signal_log = tmp_path / "signal.log"
    bash_env = tmp_path / "bash-env.sh"
    bash_env.write_text(
        """startup_interrupt_sent=0
trap 'if [ "$startup_interrupt_sent" -eq 0 ] && [ "$BASH_COMMAND" = "s2_pid=\\$!" ]; then startup_interrupt_sent=1; kill -INT "$$"; elif [ "$startup_interrupt_sent" -eq 1 ] && [ "$BASH_COMMAND" = "s4_pid=\\$!" ]; then for _ in {1..500}; do [ "$(grep -c "^ready budget=" "$SIGNAL_LOG" 2>/dev/null || true)" -eq 2 ] && break; sleep 0.01; done; trap - DEBUG; fi' DEBUG
"""
    )
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        """#!/bin/sh
if [ "$(basename "$1")" = "run_lingbotva_native_teacher_4gpu_2replica_formal.sh" ]; then
    trap 'printf "terminated budget=%s pid=%s\\n" "$BUDGETS" "$$" >> "$SIGNAL_LOG"; exit 0' TERM
    printf 'ready budget=%s pid=%s\\n' "$BUDGETS" "$$" >> "$SIGNAL_LOG"
    while :; do sleep 1; done
fi
exec /bin/bash "$@"
"""
    )
    fake_bash.chmod(0o755)
    process = subprocess.Popen(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env={
            **os.environ,
            "CHECKPOINT": str(checkpoint),
            "S2_OUTPUT_ROOT": str(tmp_path / "s2-results"),
            "S4_OUTPUT_ROOT": str(tmp_path / "s4-results"),
            "SIGNAL_LOG": str(signal_log),
            "BASH_ENV": str(bash_env),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
        text=True,
        start_new_session=True,
    )
    try:
        assert process.wait(timeout=5) != 0
        deadline = time.monotonic() + 2
        while (
            (
                not signal_log.exists()
                or signal_log.read_text().count("terminated budget=") != 2
            )
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)

        log_lines = signal_log.read_text().splitlines()
        assert sum(line.startswith("ready budget=") for line in log_lines) == 2
        assert sum(line.startswith("terminated budget=") for line in log_lines) == 2
        assert {line.split()[1] for line in log_lines if line.startswith("ready ")} == {
            "budget=2",
            "budget=4",
        }
        child_pids = [
            int(line.split()[2].split("=", 1)[1])
            for line in log_lines
            if line.startswith("ready ")
        ]
        assert all(not Path(f"/proc/{pid}").exists() for pid in child_pids)
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.wait(timeout=5)


def test_sigint_signals_s4_before_waiting_for_a_slow_s2_shutdown(tmp_path):
    checkpoint = tmp_path / "base" / "transformer"
    checkpoint.mkdir(parents=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    signal_log = tmp_path / "signal.log"
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        """#!/bin/sh
if [ "$(basename "$1")" = "run_lingbotva_native_teacher_4gpu_2replica_formal.sh" ]; then
    if [ "$BUDGETS" = "2" ]; then
        trap 'sleep 3; printf "terminated budget=2\\n" >> "$SIGNAL_LOG"; exit 0' TERM
    else
        trap 'printf "terminated budget=4\\n" >> "$SIGNAL_LOG"; exit 0' TERM
    fi
    printf 'ready budget=%s\\n' "$BUDGETS" >> "$SIGNAL_LOG"
    while :; do sleep 1; done
fi
exec /bin/bash "$@"
"""
    )
    fake_bash.chmod(0o755)
    process = subprocess.Popen(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env={
            **os.environ,
            "CHECKPOINT": str(checkpoint),
            "S2_OUTPUT_ROOT": str(tmp_path / "s2-results"),
            "S4_OUTPUT_ROOT": str(tmp_path / "s4-results"),
            "SIGNAL_LOG": str(signal_log),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while (
            (
                not signal_log.exists()
                or signal_log.read_text().count("ready budget=") != 2
            )
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert signal_log.exists()
        assert signal_log.read_text().count("ready budget=") == 2

        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=8) != 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=8)

    log_lines = signal_log.read_text().splitlines()
    assert "terminated budget=4" in log_lines
    assert "terminated budget=2" in log_lines
    assert log_lines.index("terminated budget=4") < log_lines.index("terminated budget=2")
