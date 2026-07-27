# LIBERO S2/S4 Parallel Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-command wrapper that concurrently launches isolated two-replica native-teacher LIBERO S2 and S4 evaluations on a new four-GPU allocation.

**Architecture:** The wrapper calls the existing two-replica formal launcher twice in parallel. One child owns S2 GPUs 0/1 and the other owns S4 GPUs 2/3; each receives a distinct result root, and all four effective master/WebSocket port ranges are mutually disjoint. A focused `CHECK_ONLY=1` subprocess test validates the command contract without creating results or acquiring GPUs.

**Tech Stack:** Bash, existing LIBERO dynamic launcher, Python `pytest`.

## Global Constraints

- Work only in `/kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/libero-two-replica-final-verify-20260727T030010` on branch `codex/libero-s2-s4-parallel-launch`.
- Never modify, stop, or reuse the active S1 evaluation root `/kpfs-intern/jialongliu/projects/Flash-WAM/evaluation/results/libero_teacher_native_dynamic_4gpu_formal_py38fix_20260727/teacher_native`; reject either output-root override that equals, contains, or is contained by this path before launching either child.
- S2 defaults: GPUs `0,1`, two replicas/GPU, budget `2`, master ports `34680-34683`, WebSocket ports `34780-34783`, result root `/kpfs-intern/jialongliu/projects/Flash-WAM/evaluation/results/libero_teacher_native_dynamic_s2_2gpu2replica_formal_20260727`.
- S4 defaults: GPUs `2,3`, two replicas/GPU, budget `4`, master ports `34880-34883`, WebSocket ports `34980-34983`, result root `/kpfs-intern/jialongliu/projects/Flash-WAM/evaluation/results/libero_teacher_native_dynamic_s4_2gpu2replica_formal_20260727`.
- Preserve pass-through of `CHECKPOINT`, `EPISODES`, `CHECK_ONLY`, `SERVER_PYTHON`, and `CLIENT_PYTHON`; expose the S2/S4 GPU, replica, root, and port values as environment overrides.
- Require the S2 master, S2 WebSocket, S4 master, and S4 WebSocket effective port ranges to be mutually disjoint.
- Record `INT` or `TERM` received during either child launch/PID assignment until both PIDs are captured. Normal cleanup deliberately converts both parent signals to `TERM` for only the active child wrapper PIDs because noninteractive background children may ignore `INT`. Do not use GPU-wide or process-name-wide termination.
- Tests run without GPUs under `CHECK_ONLY=1`, do not depend on worker-line ordering, and prove no output root is created.

---

### Task 1: Parallel wrapper and check-only contract test

**Files:**
- Create: `evaluation/libero/run_lingbotva_native_teacher_s2_s4_parallel_4gpu.sh`
- Create: `evaluation/libero/tests/test_run_lingbotva_native_teacher_s2_s4_parallel_4gpu.py`

**Interfaces:**
- Consumes: `evaluation/libero/run_lingbotva_native_teacher_4gpu_2replica_formal.sh` with `CHECKPOINT`, `OUTPUT_ROOT`, `GPU_IDS`, `REPLICAS_PER_GPU`, `EPISODES`, `MASTER_PORT_BASE`, `WS_PORT_BASE`, `BUDGETS`, and inherited `CHECK_ONLY`.
- Produces: executable Bash entry point `bash evaluation/libero/run_lingbotva_native_teacher_s2_s4_parallel_4gpu.sh`; override variables are `S2_OUTPUT_ROOT`, `S4_OUTPUT_ROOT`, `S2_GPU_IDS`, `S4_GPU_IDS`, `S2_REPLICAS_PER_GPU`, `S4_REPLICAS_PER_GPU`, `S2_MASTER_PORT_BASE`, `S2_WS_PORT_BASE`, `S4_MASTER_PORT_BASE`, and `S4_WS_PORT_BASE`.

- [ ] **Step 1: Write the failing test**

Create `test_run_lingbotva_native_teacher_s2_s4_parallel_4gpu.py`. It creates a temporary checkpoint directory, chooses two nonexistent temporary result roots, sets `CHECK_ONLY=1`, `CHECKPOINT`, `S2_OUTPUT_ROOT`, and `S4_OUTPUT_ROOT`, then runs the new script using `subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, env=env, text=True, capture_output=True, check=False)`.

Parse only `WORKER ` lines into dictionaries. The finished test asserts:

```python
assert result.returncode == 0, result.stdout + result.stderr
assert len(worker_lines) == 8
assert {worker["steps"] for worker in workers} == {"2", "4"}
assert {(w["gpu"], w["replica"]) for w in s2} == {
    ("0", "0"), ("0", "1"), ("1", "0"), ("1", "1"),
}
assert {(w["gpu"], w["replica"]) for w in s4} == {
    ("2", "0"), ("2", "1"), ("3", "0"), ("3", "1"),
}
assert {int(w["master_port"]) for w in s2} == set(range(34680, 34684))
assert {int(w["ws_port"]) for w in s2} == set(range(34780, 34784))
assert {int(w["master_port"]) for w in s4} == set(range(34880, 34884))
assert {int(w["ws_port"]) for w in s4} == set(range(34980, 34984))
assert not s2_root.exists()
assert not s4_root.exists()
```

It also asserts each worker reports `task_queue=40`, `claim_mode=mkdir`, `client_flag=--no-save-video`, and `server_flag=--no-save-debug-tensors`; each steps group has four unique `save_root`, `results_root`, and `latency_jsonl` paths below its matching temporary root; and `result.stdout.count("CHECK_ONLY=1: verified 4 dynamic workers") == 2`.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
SERVER_PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python CLIENT_PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q evaluation/libero/tests/test_run_lingbotva_native_teacher_s2_s4_parallel_4gpu.py
```

Expected: FAIL because `evaluation/libero/run_lingbotva_native_teacher_s2_s4_parallel_4gpu.sh` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create the Bash wrapper with `set -euo pipefail`, resolve `PROJECT_ROOT` and `FORMAL_LAUNCHER`, then declare the exact defaults from Global Constraints. Launch S2 and S4 as background children using this contract:

```bash
OUTPUT_ROOT="$S2_OUTPUT_ROOT" GPU_IDS="$S2_GPU_IDS" REPLICAS_PER_GPU="$S2_REPLICAS_PER_GPU" BUDGETS=2 MASTER_PORT_BASE="$S2_MASTER_PORT_BASE" WS_PORT_BASE="$S2_WS_PORT_BASE" bash "$FORMAL_LAUNCHER" &
s2_pid=$!
OUTPUT_ROOT="$S4_OUTPUT_ROOT" GPU_IDS="$S4_GPU_IDS" REPLICAS_PER_GPU="$S4_REPLICAS_PER_GPU" BUDGETS=4 MASTER_PORT_BASE="$S4_MASTER_PORT_BASE" WS_PORT_BASE="$S4_WS_PORT_BASE" bash "$FORMAL_LAUNCHER" &
s4_pid=$!
```

Install a startup `INT`/`TERM` trap that records the signal while both launch/PID-assignment steps finish. Once both PIDs are captured, install normal cleanup traps and process any recorded startup signal. Cleanup sends `TERM` only to tracked active `s2_pid` and `s4_pid`, signals all such children before waiting for them, and exits nonzero. Otherwise wait for each child, retain both exit codes, print `S2 rc=<value>; S4 rc=<value>`, and return zero only if both are zero. Do not create result directories; the child launcher owns safe acquisition.

- [ ] **Step 4: Run test to verify it passes**

Run the exact pytest command from Step 2 and then:

```bash
bash -n evaluation/libero/run_lingbotva_native_teacher_s2_s4_parallel_4gpu.sh
```

Expected: pytest reports one passing test and `bash -n` exits zero.

- [ ] **Step 5: Run the regression suite and commit**

Run:

```bash
SERVER_PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python CLIENT_PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q evaluation/libero/tests/test_native_teacher_artifact_flags.py evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_24_eval_8gpu_dynamic.py evaluation/libero/tests/test_run_lingbotva_native_teacher_4gpu_formal.py evaluation/libero/tests/test_run_lingbotva_native_teacher_4gpu_2replica_formal.py evaluation/libero/tests/test_run_lingbotva_native_teacher_s2_s4_parallel_4gpu.py evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py evaluation/libero/tests/test_merge_lingbotva_4suite_results.py
```

Commit only the wrapper and test:

```bash
git add evaluation/libero/run_lingbotva_native_teacher_s2_s4_parallel_4gpu.sh evaluation/libero/tests/test_run_lingbotva_native_teacher_s2_s4_parallel_4gpu.py
git commit -m "feat: add parallel LIBERO S2 S4 launcher"
```
