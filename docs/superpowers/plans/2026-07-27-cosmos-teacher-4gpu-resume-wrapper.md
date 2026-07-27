# Cosmos Teacher Four-GPU Resume Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a short four-GPU command that resumes an existing official-teacher K=1/2/4 full40 matrix while strictly validating and skipping completed cells.

**Architecture:** A dedicated Bash wrapper owns resume orchestration and delegates every unfinished cell to the existing formal evaluator. A small Python validation block verifies completed summaries before skip, and the existing strict matrix merger remains the only publisher of the final JSON/CSV matrix.

**Tech Stack:** Bash, Python 3.10, pytest, existing `cosmos_progressive_eval_summary` APIs.

## Global Constraints

- Do not change the no-overwrite behavior of the existing matrix runner.
- Do not delete or rewrite completed episode records.
- Evaluate `libero_10`, `libero_spatial`, `libero_object`, and `libero_goal`.
- Use matched video/action budgets K=1, K=2, and K=4.
- Require 10 tasks and exactly `S4_EPISODES_PER_TASK` records per task before skipping a cell.
- Stop on an existing incomplete or invalid cell.
- Retain only representative-video seed 0 by default.

---

### Task 1: Resumable official-teacher matrix launcher

**Files:**
- Create: `evaluation/libero/run_cosmos_teacher_124_eval_4gpu_resume.sh`
- Create: `evaluation/libero/tests/test_run_cosmos_teacher_124_eval_4gpu_resume.py`

**Interfaces:**
- Consumes: positional `MATRIX_ROOT`; existing environment defaults from `evaluation/libero/cosmos_progressive_s4_env.sh`; existing `run_cosmos_progressive_s4_eval.sh formal`; existing `merge_student_matrix(...)`.
- Produces: validated skip records on stdout, remaining formal cell invocations, and final `matrix_summary.json` plus `matrix_summary.csv`.

- [ ] **Step 1: Write failing launcher contract tests**

Create tests that execute the wished-for wrapper against temporary paths and a fake formal launcher. Cover:

```python
def test_resume_skips_a_complete_k1_libero10_cell_and_launches_remaining_eleven(tmp_path):
    # Seed a valid official_teacher K=1/libero_10 formal_summary.json with
    # num_tasks=10, num_records=500, seeds_per_task=50 and formal_verified.
    # The fake formal launcher records EVAL_ROOT, suite, and K.
    # Assert the completed cell is absent and the other eleven cells occur
    # in deterministic K-major, suite-minor order.

def test_resume_rejects_existing_cell_without_valid_complete_summary(tmp_path):
    # Create k1/libero_10 without a complete summary.
    # Assert nonzero exit and "incomplete existing cell".

def test_resume_rejects_summary_with_wrong_budget_or_record_count(tmp_path):
    # Write video_steps=2 under k1/libero_10 or num_records=499.
    # Assert nonzero exit and no fake evaluator invocation.

def test_resume_dry_run_does_not_create_matrix_outputs(tmp_path):
    # Set S4_DRY_RUN=1 and use a nonexistent matrix root.
    # Assert planned twelve cells are printed and the root remains absent.
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  evaluation/libero/tests/test_run_cosmos_teacher_124_eval_4gpu_resume.py
```

Expected: FAIL because `run_cosmos_teacher_124_eval_4gpu_resume.sh` does not exist.

- [ ] **Step 3: Implement the minimal wrapper**

Implement these exact behaviors:

```bash
[[ $# -eq 1 ]] || die "usage: ... MATRIX_ROOT"
MATRIX_ROOT="$1"
readonly STEPS=(1 2 4)
readonly SUITES=(libero_10 libero_spatial libero_object libero_goal)
export S4_MODEL_ROLE=official_teacher
export S4_ALIGNMENT_VERIFIED=1
export S4_EVAL_CLASSIFICATION=formal_verified
export S4_EVAL_IS_FORMAL=1
export S4_FORMAL_NUM_SHARDS=4
export S4_FORMAL_GPU_LAYOUT=colocated
export S4_VIDEO_SEEDS="${S4_VIDEO_SEEDS:-0}"
export S4_EPISODES_PER_TASK="${S4_EPISODES_PER_TASK:-50}"
```

For every cell, validate a present summary using Python assertions over:

```text
model_role
libero_benchmark
video_steps
action_steps
num_tasks
num_records
seeds_per_task
is_formal
evaluation_classification
checkpoint
```

Print `SKIP_COMPLETED=<suite>,K=<k>,records=<count>` only after validation.
If no cell exists, invoke the configurable
`COSMOS_TEACHER_FORMAL_LAUNCHER` defaulting to
`evaluation/libero/run_cosmos_progressive_s4_eval.sh`.
After all cells, call `merge_student_matrix` unless `S4_DRY_RUN=1`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
bash -n evaluation/libero/run_cosmos_teacher_124_eval_4gpu_resume.sh
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  evaluation/libero/tests/test_run_cosmos_teacher_124_eval_4gpu_resume.py
```

Expected: all tests pass.

- [ ] **Step 5: Run related regression tests**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_4gpu.py \
  evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py \
  evaluation/libero/tests/test_cosmos_progressive_eval_summary.py
```

Expected: all tests pass without starting a live evaluator.

- [ ] **Step 6: Verify the real completed cell in read-only mode**

Run the wrapper with `S4_DRY_RUN=1` against:

```text
/kpfs-intern/jialongliu/results/cosmos-teacher-k124-4gpu-liberoenv-20260726-v5
```

Expected stdout includes:

```text
SKIP_COMPLETED=libero_10,K=1,records=500
```

and does not include a launch plan for `k1/libero_10`.

- [ ] **Step 7: Commit**

```bash
git add \
  evaluation/libero/run_cosmos_teacher_124_eval_4gpu_resume.sh \
  evaluation/libero/tests/test_run_cosmos_teacher_124_eval_4gpu_resume.py
git commit -m "feat: resume completed Cosmos teacher eval cells"
```
