# Two-replica-per-GPU LIBERO Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe, opt-in two-independent-server-per-GPU native-teacher LIBERO evaluator that doubles concurrent rollouts without changing the formal evaluation protocol.

**Architecture:** Preserve the existing one-client-per-stateful-server contract. Expand each physical GPU into `REPLICAS_PER_GPU` independent lanes; each lane owns its own server process, KV cache, ports, worker directory, and client, while all lanes atomically claim from the same per-budget 40-task queue. Keep the current launcher default at one replica and expose two replicas only through a new fast formal wrapper.

**Tech Stack:** Bash 4+, Python 3.8 LIBERO client, Python 3.10 native-teacher server, pytest, WebSocket health checks, atomic `mkdir` task claims.

## Global Constraints

- Do not touch the currently running one-replica evaluation, its process tree, checkpoint, or output root.
- Preserve 40 tasks, 50 episodes, matched video/action budgets `1,2,4`, no MP4, no debug tensors, and the existing merger schema.
- A single native-teacher server remains single-client; never share a server or its KV cache between replicas.
- Default `--replicas-per-gpu` is exactly `1`; supported values are `1` and `2`.
- Each simultaneously active lane has a unique worker ID, replica ID, master port, WebSocket port, server root, results root, and latency JSONL path.  The three budgets run serially and may reuse the same non-overlapping lane port range.
- All two-replica formal runs use a fresh output root and a new port range.

---

## File Structure

- Modify: `evaluation/libero/run_lingbotva_native_teacher_4suite_24_eval_8gpu_dynamic.sh`
  - Parse/validate replica count, derive lanes, and launch independent server/client pairs.
- Modify: `evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_24_eval_8gpu_dynamic.py`
  - Cover two-replica planning, port uniqueness, shared claims, and sibling cleanup.
- Create: `evaluation/libero/run_lingbotva_native_teacher_4gpu_2replica_formal.sh`
  - Fixed, user-facing four-GPU/two-replica formal launcher with a fresh output root.
- Create: `evaluation/libero/tests/test_run_lingbotva_native_teacher_4gpu_2replica_formal.py`
  - Verify the fast wrapper expands to eight lanes per budget without writing artifacts in check-only mode.

## Task 1: Make the dynamic launcher replica-aware

**Files:**
- Modify: `evaluation/libero/run_lingbotva_native_teacher_4suite_24_eval_8gpu_dynamic.sh:19-130, 235-585`
- Test: `evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_24_eval_8gpu_dynamic.py`

**Interfaces:**
- Consumes: `--gpu-ids 0,1,2,3`, `--replicas-per-gpu 1|2`, existing budget and port arguments.
- Produces: `LANE_COUNT = GPU_COUNT * REPLICAS_PER_GPU` independent `WORKER` records, each with `worker`, `replica`, `gpu`, distinct ports, and distinct artifact roots.

- [ ] **Step 1: Write failing check-only tests for a two-replica four-GPU plan**

Add `replicas_per_gpu` to `_run_launcher` and append `--replicas-per-gpu` when set. Add this behavioral contract:

```python
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
```

Add a parameterized invalid-count test for `"0"`, `"3"`, and `"two"`, asserting a nonzero exit and `--replicas-per-gpu must be 1 or 2` in stderr.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
SERVER_PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
CLIENT_PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python \
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_24_eval_8gpu_dynamic.py
```

Expected: the two-replica test fails because the launcher rejects the unknown flag.

- [ ] **Step 3: Implement lane derivation and replica validation**

Add these launcher variables and parser branch:

```bash
REPLICAS_PER_GPU=1
# parser branch
--replicas-per-gpu) REPLICAS_PER_GPU="$2"; shift 2 ;;
```

After GPU validation, validate `REPLICAS_PER_GPU` with an explicit case and derive lanes:

```bash
case "$REPLICAS_PER_GPU" in
    1|2) ;;
    *) echo "--replicas-per-gpu must be 1 or 2" >&2; exit 2 ;;
esac
GPU_COUNT="${#GPUS[@]}"
LANE_COUNT=$((GPU_COUNT * REPLICAS_PER_GPU))
LAST_LANE_OFFSET=$((LANE_COUNT - 1))
```

Use `LANE_COUNT` and `LAST_LANE_OFFSET` for every port-range check and launch loop. For lane `lane`, derive:

```bash
gpu_index=$((lane % GPU_COUNT))
gpu="${GPUS[$gpu_index]}"
replica=$((lane / GPU_COUNT))
```

Pass `DYNAMIC_REPLICA` through the re-exec environment and pass
`--replicas-per-gpu "$REPLICAS_PER_GPU"` to the re-exec launcher command so
the child validates the same lane count. Update `run_worker` to accept
`(steps, worker, gpu, replica, budget_root, claims_root)`, put `replica` in
its startup/claim log lines, and add it to `print_worker_plan`. Keep the
existing lane-indexed `worker_${worker}` artifact path and atomic claims
unchanged.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command. Expected: all dynamic launcher tests pass, including default one-replica contracts.

- [ ] **Step 5: Extend the fake lifecycle coverage to two replicas on one GPU**

Modify `test_first_worker_failure_terminates_sibling_server_groups_promptly` to invoke the launcher with `gpu_ids="0"` and `replicas_per_gpu="2"`. Keep one fake client failure bound to one WebSocket port and assert every PID recorded by the fake server is gone. Modify `test_one_budget_claims_every_task_once_and_reaches_the_merger` to use `gpu_ids="0,1"`, `replicas_per_gpu="2"`, and assert exactly 40 unique task calls plus a generated summary.

- [ ] **Step 6: Run lifecycle and merger tests**

Run:

```bash
SERVER_PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
CLIENT_PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python \
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_24_eval_8gpu_dynamic.py \
  evaluation/libero/tests/test_merge_lingbotva_4suite_results.py
```

Expected: all fake-server cleanup, exact-once claims, and merger coverage tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add evaluation/libero/run_lingbotva_native_teacher_4suite_24_eval_8gpu_dynamic.sh \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_24_eval_8gpu_dynamic.py
git commit -m "feat: add independent LIBERO replicas per GPU"
```

## Task 2: Add the explicit fast four-GPU formal wrapper

**Files:**
- Create: `evaluation/libero/run_lingbotva_native_teacher_4gpu_2replica_formal.sh`
- Create: `evaluation/libero/tests/test_run_lingbotva_native_teacher_4gpu_2replica_formal.py`

**Interfaces:**
- Consumes: optional `CHECKPOINT`, `OUTPUT_ROOT`, `GPU_IDS`, `EPISODES`, and port-base environment overrides.
- Produces: a 4-GPU, 2-replica plan with eight lanes per budget and no video/debug artifacts.

- [ ] **Step 1: Write the failing wrapper check-only test**

Create a subprocess test that sets `CHECK_ONLY=1`, a temporary checkpoint, and a temporary output root, then runs the new wrapper. Assert:

```python
assert result.returncode == 0, result.stdout + result.stderr
assert len(worker_lines) == 24
assert "CHECK_ONLY=1: verified 24 dynamic workers" in result.stdout
for steps in ("1", "2", "4"):
    lanes = [worker for worker in workers if worker["steps"] == steps]
    assert len(lanes) == 8
    assert {worker["replica"] for worker in lanes} == {"0", "1"}
    assert {worker["gpu"] for worker in lanes} == {"0", "1", "2", "3"}
assert not output_root.exists()
```

- [ ] **Step 2: Run the wrapper test and verify RED**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4gpu_2replica_formal.py
```

Expected: FAIL because the fast wrapper does not yet exist.

- [ ] **Step 3: Implement the fast wrapper**

Create a Bash wrapper following `run_lingbotva_native_teacher_4gpu_formal.sh`. Set these defaults:

```bash
OUTPUT_ROOT="${OUTPUT_ROOT:-/kpfs-intern/jialongliu/projects/Flash-WAM/evaluation/results/libero_teacher_native_dynamic_4gpu_2replica_formal_20260727}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-33680}"
WS_PORT_BASE="${WS_PORT_BASE:-33780}"
```

Forward `--replicas-per-gpu "${REPLICAS_PER_GPU}"` alongside the existing checkpoint, output, GPU, episode, port, and budget arguments. Keep the absolute interpreter defaults and only forward user-supplied trailing launcher flags after the defaults.

- [ ] **Step 4: Run wrapper tests and verify GREEN**

Run the Step 2 command plus:

```bash
bash -n evaluation/libero/run_lingbotva_native_teacher_4gpu_2replica_formal.sh
```

Expected: the check-only plan contains 24 lanes and writes no artifacts.

- [ ] **Step 5: Commit Task 2**

```bash
git add evaluation/libero/run_lingbotva_native_teacher_4gpu_2replica_formal.sh \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4gpu_2replica_formal.py
git commit -m "feat: add fast two-replica LIBERO launcher"
```

## Task 3: Verify the integrated evaluator and conduct the capacity gate

**Files:**
- Modify only if failures reveal a concrete integration defect in Task 1 or 2.
- Test: all files listed below.

**Interfaces:**
- Consumes: the new `--replicas-per-gpu 2` dynamic contract and fast wrapper.
- Produces: evidence that formal task/result/latency coverage remains complete before any full restart.

- [ ] **Step 1: Run complete offline regression suite**

Run:

```bash
SERVER_PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
CLIENT_PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python \
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  evaluation/libero/tests/test_native_teacher_artifact_flags.py \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_24_eval_8gpu_dynamic.py \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4gpu_formal.py \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4gpu_2replica_formal.py \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py \
  evaluation/libero/tests/test_merge_lingbotva_4suite_results.py
```

Expected: all tests pass with no regression to one-replica operation.

- [ ] **Step 2: Run remote check-only preflight**

On RobotWin A800, run the fast wrapper with `CHECK_ONLY=1` and a nonexistent temporary output root. Confirm 24 planned lanes, eight lanes per budget, eight unique master/WebSocket pairs within each serial budget, and no output directory creation.

- [ ] **Step 3: Stop the old one-replica run only after user confirmation and capacity availability**

Identify the active job by its exact fast-formal output root and launcher command. Terminate only its verified process group/job. Do not touch checkpoints, training jobs, or the failed-output root.

- [ ] **Step 4: Run the one-GPU/two-replica live capacity smoke**

Start two independent native-teacher servers on the same allocated A800 GPU with unique ports and roots, then run two one-episode, different-task clients concurrently. The smoke passes only if both health checks, both result JSONs, and both latency files exist and neither server log contains CUDA OOM or connection supersession.

- [ ] **Step 5: Compare throughput and decide formal restart**

Record wall-clock time per completed episode for both lanes and compare it against the current one-replica baseline. Proceed to the four-GPU fast wrapper only if both lanes complete and aggregate throughput improves; otherwise retain one replica per GPU and report the capacity/throughput limit.

- [ ] **Step 6: Push verified implementation**

```bash
git push https://github.com/JIAlonglong/Any_WAM.git \
  codex/libero-fast-formal-eval:codex/libero-fast-formal-eval
```
