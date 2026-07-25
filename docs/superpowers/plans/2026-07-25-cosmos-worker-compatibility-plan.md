# Cosmos Worker Compatibility Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the formal Cosmos raw worker import and start reliably before launching Stage-1, without modifying shared environments or the existing dirty Cosmos checkout.

**Architecture:** Build a separate clean Cosmos compatibility worktree containing only the six already-proven tracked compatibility patches. Seal the Stage-1 launcher to the clean repository's `cosmos-cuda`/`cosmos-oss` sources and the worker environment's NVIDIA runtime libraries, then run a full `cosmos_utils` import preflight before `torchrun`.

**Tech Stack:** Bash, Python 3.10, pytest, Git worktrees, Cosmos Predict2.5, PyTorch/CUDA 12.8.

## Global Constraints

- Preserve `/kpfs-intern/jialongliu/projects/cosmos-predict2.5` exactly as-is.
- Do not install or upgrade packages in any shared Python environment.
- Do not delete the failed run root.
- Do not start training from Codex.
- Dry-run and check-only operations must remain write-free.
- Formal execution must reject a dirty Cosmos compatibility repository.

---

### Task 1: Create the isolated Cosmos compatibility source

**Files:**
- Source: `/kpfs-intern/jialongliu/projects/cosmos-predict2.5`
- Create worktree: `/kpfs-intern/jialongliu/projects/cosmos-predict2.5-formal-compat-441b897`
- Modify only in compatibility worktree:
  - `cosmos_predict2/_src/imaginaire/utils/checkpoint_db.py`
  - `cosmos_predict2/_src/imaginaire/utils/fused_adam.py`
  - `cosmos_predict2/_src/imaginaire/utils/object_store.py`
  - `cosmos_predict2/_src/predict2/utils/fused_adam_dtensor.py`
  - `cosmos_predict2/_src/reason1/networks/qwen2_5_vl.py`
  - `cosmos_predict2/_src/reason1/utils/fused_adam.py`

**Interfaces:**
- Consumes: fixed base commit `441b89740d91922737008a61e7f71407d47944e7` and the six tracked diffs already present in the dirty checkout.
- Produces: a clean Git worktree whose HEAD contains exactly those six compatibility changes.

- [ ] **Step 1: Capture and validate the exact tracked patch**

Run:

```bash
git -C /kpfs-intern/jialongliu/projects/cosmos-predict2.5 diff -- \
  cosmos_predict2/_src/imaginaire/utils/checkpoint_db.py \
  cosmos_predict2/_src/imaginaire/utils/fused_adam.py \
  cosmos_predict2/_src/imaginaire/utils/object_store.py \
  cosmos_predict2/_src/predict2/utils/fused_adam_dtensor.py \
  cosmos_predict2/_src/reason1/networks/qwen2_5_vl.py \
  cosmos_predict2/_src/reason1/utils/fused_adam.py \
  > /tmp/cosmos-runtime-compat.patch
git -C /kpfs-intern/jialongliu/projects/cosmos-predict2.5 apply --check \
  --reverse /tmp/cosmos-runtime-compat.patch
```

Expected: reverse check succeeds, proving the patch exactly describes the six current tracked modifications.

- [ ] **Step 2: Create the isolated worktree**

Run:

```bash
git -C /kpfs-intern/jialongliu/projects/cosmos-predict2.5 worktree add \
  -b codex/formal-runtime-compat-441b897 \
  /kpfs-intern/jialongliu/projects/cosmos-predict2.5-formal-compat-441b897 \
  441b89740d91922737008a61e7f71407d47944e7
```

Expected: new worktree is clean at the fixed base commit; the original checkout status is unchanged.

- [ ] **Step 3: Apply and commit only the tracked compatibility patch**

Run:

```bash
git -C /kpfs-intern/jialongliu/projects/cosmos-predict2.5-formal-compat-441b897 \
  apply /tmp/cosmos-runtime-compat.patch
git -C /kpfs-intern/jialongliu/projects/cosmos-predict2.5-formal-compat-441b897 \
  diff --check
git -C /kpfs-intern/jialongliu/projects/cosmos-predict2.5-formal-compat-441b897 \
  add cosmos_predict2
git -C /kpfs-intern/jialongliu/projects/cosmos-predict2.5-formal-compat-441b897 \
  commit -m "fix: support deployed Cosmos runtime dependencies"
```

Expected: one commit changes exactly six tracked files; worktree is clean.

- [ ] **Step 4: Verify the exact worker import contract**

Run the worker Python with:

```bash
PYTHONPATH=<compat-repo>:<compat-repo>/packages/cosmos-cuda:<compat-repo>/packages/cosmos-oss \
LD_LIBRARY_PATH=<all existing worker-site-packages/nvidia/*/lib roots> \
/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/bin/python -c \
  'import cosmos_cuda, cosmos_predict2; from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import get_action; print("COSMOS_IMPORT_OK=1")'
```

Expected: exit 0 and `COSMOS_IMPORT_OK=1`.

---

### Task 2: Seal Stage-1 worker runtime inputs

**Files:**
- Modify: `distillation_flowmap/run_cosmos_raw_stage1_8gpu.sh`
- Test: `distillation_flowmap/tests/test_run_cosmos_raw_stage1_8gpu.py`
- Test: `distillation_flowmap/tests/test_cosmos_stage1_stage2_eval_pipeline.py`

**Interfaces:**
- Consumes: `COSMOS_PREDICT2_REPO`, `COSMOS_WORKER_ENV_ROOT`, `COSMOS_POLICY_PYTHON`.
- Produces: exported `COSMOS_POLICY_EXTRA_PYTHONPATH`, `COSMOS_WORKER_SITE_PACKAGES`, and `COSMOS_WORKER_CUDA_LIBRARY_PATH`; a full worker import preflight executed before `torchrun`.

- [ ] **Step 1: Write failing launcher tests**

Add assertions that Stage-1 resolved output contains:

```python
assert values["COSMOS_POLICY_EXTRA_PYTHONPATH"] == (
    f"{repo}/packages/cosmos-cuda:{repo}/packages/cosmos-oss"
)
assert values["COSMOS_WORKER_SITE_PACKAGES"] == str(worker_site_packages)
assert "nvidia/cudnn/lib" in values["COSMOS_WORKER_CUDA_LIBRARY_PATH"]
assert calls.index("worker-import-preflight") < calls.index("torchrun")
```

Add rejection cases for missing `packages/cosmos-cuda`, missing `packages/cosmos-oss`, dirty Cosmos repository, and a failing full `cosmos_utils` import.

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_run_cosmos_raw_stage1_8gpu.py \
  distillation_flowmap/tests/test_cosmos_stage1_stage2_eval_pipeline.py
```

Expected: failures specifically show missing worker paths/preflight.

- [ ] **Step 3: Implement the minimal launcher contract**

In `run_cosmos_raw_stage1_8gpu.sh`:

```bash
COSMOS_WORKER_SITE_PACKAGES="${COSMOS_WORKER_SITE_PACKAGES:-${COSMOS_WORKER_ENV_ROOT}/lib/python3.10/site-packages}"
COSMOS_POLICY_EXTRA_PYTHONPATH="$COSMOS_PREDICT2_REPO/packages/cosmos-cuda:$COSMOS_PREDICT2_REPO/packages/cosmos-oss"
COSMOS_WORKER_CUDA_LIBRARY_PATH="${COSMOS_WORKER_CUDA_LIBRARY_PATH:-...}"
```

Validate both package directories and every declared runtime library directory. Check the repository with `GIT_OPTIONAL_LOCKS=0 git status --porcelain=v1 --untracked-files=all`. Before `torchrun`, execute the full import under the exact worker environment:

```bash
PYTHONPATH="$COSMOS_PREDICT2_REPO:$COSMOS_POLICY_EXTRA_PYTHONPATH" \
LD_LIBRARY_PATH="$COSMOS_WORKER_CUDA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$COSMOS_POLICY_PYTHON" -c \
  'import cosmos_cuda, cosmos_predict2; from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import get_action'
```

Export the three resolved variables into the training environment and print them in the resolved config.

- [ ] **Step 4: Run GREEN and regression tests**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_run_cosmos_raw_stage1_8gpu.py \
  distillation_flowmap/tests/test_cosmos_stage1_stage2_eval_pipeline.py \
  distillation_flowmap/tests/test_run_cosmos_libero_train_8gpu.py
bash -n distillation_flowmap/run_cosmos_raw_stage1_8gpu.sh
git diff --check
```

Expected: all tests pass and static checks exit 0.

- [ ] **Step 5: Commit**

```bash
git add \
  distillation_flowmap/run_cosmos_raw_stage1_8gpu.sh \
  distillation_flowmap/tests/test_run_cosmos_raw_stage1_8gpu.py \
  distillation_flowmap/tests/test_cosmos_stage1_stage2_eval_pipeline.py
git commit -m "fix: preflight Cosmos Stage1 worker runtime"
```

---

### Task 3: Validate the retry chain without launching training

**Files:**
- Verify: `distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh`
- Verify: failed output root remains untouched.

**Interfaces:**
- Consumes: the clean compatibility worktree and repaired Stage-1 launcher.
- Produces: one new, unused retry run tag and a verified Stage-1 → Stage-2 → evaluation dry-run plan.

- [ ] **Step 1: Confirm failed artifacts and process state**

Run:

```bash
test -d <failed-run-root>
ps -eo pid,args | grep -E '[d]istillation_flowmap/train.py|[c]osmos_policy_raw_worker.py' || true
```

Expected: failed directory remains; no worker/training process remains.

- [ ] **Step 2: Run exact Stage-1 worker preflight**

Run the Stage-1 launcher in its read-only mode with the new compatibility repo and exact worker environment.

Expected: full `cosmos_utils` import passes before the printed `torchrun` command; no output directory is created.

- [ ] **Step 3: Run the full serial dry-run with a new tag**

Use run tag `cosmos-retrain-s1s2-joint124-20260725-retry1`.

Expected plan:

```text
Stage-1 step_5000
Stage-2 universal-video-action step_10000
Evaluation role stage2_target
Matched K=1/2/4, 500 student episodes per K
```

The retry root must remain absent after dry-run.

- [ ] **Step 4: Final verification and report**

Run the complete focused regression command from Task 2 again, confirm both Git worktrees are clean, and report:

- Flash-WAM commit.
- Cosmos compatibility commit and path.
- preserved failed run root.
- new retry command.
- teacher matched-K evaluation still pending as a separate approved feature.

