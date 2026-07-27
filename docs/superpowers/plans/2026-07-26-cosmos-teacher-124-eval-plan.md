# Cosmos Official Teacher K=1/2/4 Evaluation Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one teacher-only eight-GPU launcher that evaluates the official Cosmos policy over all 40 LIBERO tasks at matched K=1/2/4 budgets.

**Architecture:** A thin shell wrapper sources the validated Cosmos runtime defaults, forces the existing joint matrix launcher into `official_teacher` mode, and delegates all evaluation, sharding, provenance, action-grid, completeness, and merge behavior. No evaluator or scheduler logic is duplicated.

**Tech Stack:** Bash, pytest, existing Cosmos LIBERO matrix launchers.

## Global Constraints

- The official teacher must use its native `[1,16,7]` action horizon with `action_downsample_factor=1`.
- K=1/2/4 changes only the teacher solver NFE.
- The evaluation covers `libero_10`, `libero_spatial`, `libero_object`, and `libero_goal`.
- The default is 50 episodes per task.
- Formal execution uses four shards across eight GPUs.
- Dry-run must not create the matrix root.
- Existing output roots must never be overwritten.
- The wrapper must not start a live job during implementation verification.

---

### Task 1: Teacher-only eight-GPU matrix wrapper

**Files:**
- Create: `evaluation/libero/run_cosmos_official_teacher_124_eval_8gpu.sh`
- Create: `evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_8gpu.py`

**Interfaces:**
- Consumes: `evaluation/libero/cosmos_progressive_s4_env.sh` for runtime defaults.
- Consumes: `evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh run|dry-run`.
- Produces: `bash evaluation/libero/run_cosmos_official_teacher_124_eval_8gpu.sh run|dry-run`.

- [ ] **Step 1: Write the failing launcher contract tests**

Create a temporary executable matrix sentinel, inject it through
`COSMOS_TEACHER_MATRIX_LAUNCHER`, and run the missing wrapper with an isolated
environment. Assert that:

```python
assert result.returncode == 0
assert captured["argv"] == ["dry-run"]
assert captured["S4_MATRIX_ROLES"] == "official_teacher"
assert captured["S4_ALIGNMENT_VERIFIED"] == "1"
assert captured["S4_FORMAL_NUM_SHARDS"] == "4"
assert captured["S4_EPISODES_PER_TASK"] == "50"
assert captured["MATRIX_ROOT"] == str(matrix_root)
assert not matrix_root.exists()
```

Add separate assertions that the wrapper rejects an existing `MATRIX_ROOT` and
does not invoke the sentinel, and that a caller can set
`S4_EPISODES_PER_TASK=1` for a scheduling smoke without changing the K matrix.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
PYTHONPATH=/kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/cosmos-progressive-s4-eval \
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_8gpu.py
```

Expected: FAIL because
`evaluation/libero/run_cosmos_official_teacher_124_eval_8gpu.sh` does not exist.

- [ ] **Step 3: Implement the minimal wrapper**

Create an executable Bash script with:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cosmos_progressive_s4_env.sh"

: "${MATRIX_ROOT:?set a new teacher evaluation root}"
: "${COSMOS_POLICY_PATH:?set the official Cosmos teacher root}"
: "${COSMOS_POLICY_TEACHER_LOCK:?set the verified teacher provenance lock}"
: "${S4_DATASET_PATH:?set the LIBERO dataset root}"
: "${S4_EMPTY_EMBEDDING:?set the empty embedding}"
: "${S4_PROMPT_TABLE:?set the all-40-task prompt table}"

export S4_MATRIX_ROLES=official_teacher
export S4_ALIGNMENT_VERIFIED=1
export S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH=0
export S4_FORMAL_NUM_SHARDS=4
export S4_EPISODES_PER_TASK="${S4_EPISODES_PER_TASK:-50}"
export S4_CKPT_ROOT="${COSMOS_POLICY_PATH}"

MATRIX_LAUNCHER="${COSMOS_TEACHER_MATRIX_LAUNCHER:-${SCRIPT_DIR}/run_cosmos_progressive_joint_124_eval_8gpu.sh}"
exec bash "${MATRIX_LAUNCHER}" "$1"
```

Validate exactly one argument with value `run` or `dry-run` before delegation.
Keep caller overrides for paths, Python interpreters, prompt table, episode
count, output root, and the test-only matrix injection seam; do not permit role
or formal-alignment overrides. Set `S4_CKPT_ROOT` to the teacher root because
the delegated matrix requires the variable syntactically but does not evaluate
it as a student checkpoint in `official_teacher` mode.

- [ ] **Step 4: Run GREEN and existing matrix regression tests**

Run:

```bash
PYTHONPATH=/kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/cosmos-progressive-s4-eval \
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_8gpu.py \
evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py \
evaluation/libero/tests/test_cosmos_official_teacher_matrix.py
```

Expected: all tests pass.

- [ ] **Step 5: Verify the real wrapper in write-free dry-run mode**

Run the wrapper with a temporary sentinel-compatible teacher lock, all-40
prompt table path, fresh matrix root, `S4_EPISODES_PER_TASK=1`, and
`dry-run`. Confirm output contains:

```text
MATRIX_ROLE=official_teacher
MATRIX_STEP=1
MATRIX_STEP=2
MATRIX_STEP=4
FORMAL_NUM_SHARDS=4
```

Confirm `MATRIX_ROOT` does not exist afterwards. Run `bash -n` on both shell
launchers and `git diff --check`.

- [ ] **Step 6: Commit**

```bash
git add \
  evaluation/libero/run_cosmos_official_teacher_124_eval_8gpu.sh \
  evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_8gpu.py
git commit -m "feat: add Cosmos teacher K124 evaluation launcher"
```
