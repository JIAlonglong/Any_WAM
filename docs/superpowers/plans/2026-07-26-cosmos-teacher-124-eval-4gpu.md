# Cosmos Teacher K=1/2/4 Four-GPU Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe four-GPU entrypoint for the official Cosmos teacher K=1/2/4 LIBERO evaluation while preserving the existing eight-GPU behavior and saving only representative videos.

**Architecture:** Keep the existing eight-GPU wrapper as the shared teacher-only contract. Add one guarded internal shard selector that defaults to four, then create a thin four-GPU wrapper that fixes the selector to two and `S4_VIDEO_SEEDS` to `0` before delegating.

**Tech Stack:** Bash, Python 3.10, pytest, existing Cosmos LIBERO matrix launcher.

## Global Constraints

- The four-GPU entrypoint uses exactly two formal shards.
- The eight-GPU entrypoint continues to use four formal shards by default.
- The four-GPU entrypoint forces `S4_VIDEO_SEEDS=0`.
- K=1, K=2, and K=4 remain sequential and cover all 40 LIBERO tasks.
- The default remains 50 episodes per task.
- Dry-run must not create `MATRIX_ROOT`.
- No live evaluation is started during implementation.

---

### Task 1: Add the Four-GPU Wrapper Contract

**Files:**
- Create: `evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_4gpu.py`
- Modify: `evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_8gpu.py`
- Modify: `evaluation/libero/run_cosmos_official_teacher_124_eval_8gpu.sh`
- Create: `evaluation/libero/run_cosmos_official_teacher_124_eval_4gpu.sh`

**Interfaces:**
- Consumes: `COSMOS_TEACHER_MATRIX_LAUNCHER`, the existing injectable delegated-launcher path.
- Produces: `COSMOS_TEACHER_FORMAL_NUM_SHARDS`, an internal value accepted only as `2` or `4`; the new four-GPU entrypoint fixes it to `2`.

- [ ] **Step 1: Write the failing four-GPU contract test**

Create a test sentinel that captures `S4_FORMAL_NUM_SHARDS` and
`S4_VIDEO_SEEDS`. Invoke
`run_cosmos_official_teacher_124_eval_4gpu.sh dry-run` with hostile caller
values:

```python
env["S4_FORMAL_NUM_SHARDS"] = "4"
env["S4_VIDEO_SEEDS"] = "0,1,2"
result = subprocess.run(
    ["bash", str(SCRIPT_4GPU), "dry-run"],
    cwd=ROOT,
    env=env,
    text=True,
    capture_output=True,
    check=False,
)
assert result.returncode == 0, result.stdout + result.stderr
payload = json.loads(capture.read_text(encoding="utf-8"))
assert payload["S4_FORMAL_NUM_SHARDS"] == "2"
assert payload["S4_VIDEO_SEEDS"] == "0"
assert not matrix_root.exists()
```

Extend the existing eight-GPU test sentinel to capture `S4_VIDEO_SEEDS` and
set `COSMOS_TEACHER_FORMAL_NUM_SHARDS=2` in its hostile environment. Assert
that a normal eight-GPU invocation still resolves `S4_FORMAL_NUM_SHARDS=4`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_4gpu.py \
  evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_8gpu.py
```

Expected: the four-GPU tests fail because the new script does not exist.

- [ ] **Step 3: Implement the minimal shared selector and wrapper**

In the eight-GPU script, resolve the internal selector without accepting the
public `S4_FORMAL_NUM_SHARDS` value:

```bash
COSMOS_TEACHER_FORMAL_NUM_SHARDS="${COSMOS_TEACHER_FORMAL_NUM_SHARDS:-4}"
case "${COSMOS_TEACHER_FORMAL_NUM_SHARDS}" in
    2|4) ;;
    *) die "COSMOS_TEACHER_FORMAL_NUM_SHARDS must be 2 or 4" ;;
esac
export S4_FORMAL_NUM_SHARDS="${COSMOS_TEACHER_FORMAL_NUM_SHARDS}"
```

Create the four-GPU wrapper:

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COSMOS_TEACHER_FORMAL_NUM_SHARDS=2
export S4_VIDEO_SEEDS=0
exec bash "${SCRIPT_DIR}/run_cosmos_official_teacher_124_eval_8gpu.sh" "$@"
```

Make the new script executable.

- [ ] **Step 4: Run focused and launcher regression tests**

Run:

```bash
bash -n evaluation/libero/run_cosmos_official_teacher_124_eval_8gpu.sh
bash -n evaluation/libero/run_cosmos_official_teacher_124_eval_4gpu.sh
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_4gpu.py \
  evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_8gpu.py \
  evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Run a real dry-run contract check**

Use a fresh nonexistent `MATRIX_ROOT`, the audited teacher lock, and
`S4_EPISODES_PER_TASK=1`. Verify the emitted matrix contains 12 cells, every
cell has `FORMAL_NUM_SHARDS=2`, every cell has `VIDEO_SEEDS=0`, and the output
root remains absent.

- [ ] **Step 6: Commit**

```bash
git add \
  evaluation/libero/run_cosmos_official_teacher_124_eval_8gpu.sh \
  evaluation/libero/run_cosmos_official_teacher_124_eval_4gpu.sh \
  evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_8gpu.py \
  evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_4gpu.py
git commit -m "feat: add four-GPU Cosmos teacher evaluator"
```
