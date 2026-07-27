# Cosmos Stage-2 Student Full40 Eight-GPU Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one standalone eight-GPU chain that validates Stage-1 step 3000, trains universal video-action Stage-2 for 5000 steps, and evaluates only the target student on all four LIBERO suites at matched K=1/2/4.

**Architecture:** Parameterize the existing Stage-2 parent-step contract while preserving the current 5000 default, then add a thin orchestration wrapper around the reviewed Stage-2 launcher, provenance/prompt builders, and student-only matrix evaluator. Training uses all eight GPUs; evaluation reuses the allocation as four paired student/worker shards.

**Tech Stack:** Bash, Python 3.10, pytest, torchrun/FSDP, existing Cosmos lineage/provenance helpers and LIBERO evaluators.

## Global Constraints

- Stage-1 parent step is exactly 3000 for the new chain.
- Stage-2 uses `universal-video-action` for exactly 5000 steps.
- Existing callers retain Stage-1 expected step 5000 unless explicitly overridden.
- Student evaluation covers `libero_10`, `libero_spatial`, `libero_object`, and `libero_goal`.
- Evaluation uses matched video/action budgets K=1/2/4 and 50 episodes per task by default.
- The chain must never launch `official_teacher`.
- Training and evaluation must each use exactly eight visible GPU ordinals.
- Fresh outputs are no-overwrite; read-only modes write nothing.
- No live training or rollout is launched during implementation verification.

---

### Task 1: Parameterize the Stage-1 parent-step contract

**Files:**
- Modify: `distillation_flowmap/run_cosmos_libero_train_8gpu.sh`
- Modify: `distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh`
- Modify: `distillation_flowmap/cosmos_progressive_env_schema.py`
- Modify: `distillation_flowmap/tests/test_run_cosmos_libero_train_8gpu.py`
- Modify: `distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py`
- Modify: `distillation_flowmap/tests/test_cosmos_progressive_env_schema.py`

**Interfaces:**
- Consumes: optional `COSMOS_STAGE1_EXPECTED_STEP`, a positive decimal integer.
- Produces: validated `COSMOS_STAGE1_EXPECTED_STEP`; lineage calls `validate_stage1_parent(stage1, expected_step=int(...))`.

- [ ] **Step 1: Write failing tests**

Add behavior tests that run each launcher against synthetic Stage-1 metadata:

```python
def test_stage2_launcher_accepts_explicit_stage1_step_3000(tmp_path):
    env = valid_launcher_env(tmp_path, parent_step=3000)
    env["COSMOS_STAGE1_EXPECTED_STEP"] = "3000"
    result = run_dry(env)
    assert result.returncode == 0
    assert "COSMOS_STAGE1_EXPECTED_STEP=3000" in result.stdout

def test_stage2_launcher_defaults_parent_step_to_5000(tmp_path):
    env = valid_launcher_env(tmp_path, parent_step=5000)
    env.pop("COSMOS_STAGE1_EXPECTED_STEP", None)
    result = run_dry(env)
    assert result.returncode == 0
    assert "COSMOS_STAGE1_EXPECTED_STEP=5000" in result.stdout

def test_stage2_launcher_rejects_parent_metadata_that_disagrees(tmp_path):
    env = valid_launcher_env(tmp_path, parent_step=5000)
    env["COSMOS_STAGE1_EXPECTED_STEP"] = "3000"
    result = run_dry(env)
    assert result.returncode != 0
    assert "step" in result.stderr.lower()
```

Add schema tests proving the variable is recognized, pinned, and rejects zero,
negative, non-decimal, and ambient drift.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_run_cosmos_libero_train_8gpu.py \
  distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py \
  distillation_flowmap/tests/test_cosmos_progressive_env_schema.py
```

Expected: the step-3000 cases fail because both launchers still pass literal
`expected_step=5000`.

- [ ] **Step 3: Implement the minimal parent-step parameterization**

In both Stage-2 launchers:

```bash
COSMOS_STAGE1_EXPECTED_STEP="${COSMOS_STAGE1_EXPECTED_STEP:-5000}"
[[ "$COSMOS_STAGE1_EXPECTED_STEP" =~ ^[1-9][0-9]*$ ]] || \
    die "COSMOS_STAGE1_EXPECTED_STEP must be a positive integer"
export COSMOS_STAGE1_EXPECTED_STEP
```

Replace literal lineage validation with:

```python
parent = validate_stage1_parent(
    stage1,
    expected_step=int(os.environ["COSMOS_STAGE1_EXPECTED_STEP"]),
)
```

Include the value in resolved launcher output and the formal environment
schema. Do not infer it from a directory basename.

- [ ] **Step 4: Run focused tests and commit**

Run the Step 2 command and require all tests to pass, then:

```bash
git add \
  distillation_flowmap/run_cosmos_libero_train_8gpu.sh \
  distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh \
  distillation_flowmap/cosmos_progressive_env_schema.py \
  distillation_flowmap/tests/test_run_cosmos_libero_train_8gpu.py \
  distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py \
  distillation_flowmap/tests/test_cosmos_progressive_env_schema.py
git commit -m "feat: validate explicit Cosmos Stage-1 parent step"
```

---

### Task 2: Add the standalone Stage-2 to student-evaluation wrapper

**Files:**
- Create: `distillation_flowmap/run_cosmos_stage2_student_eval_8gpu.sh`
- Create: `distillation_flowmap/tests/test_run_cosmos_stage2_student_eval_8gpu.py`

**Interfaces:**
- Consumes: `--phase all|stage2|eval|check`, `--stage1-root`,
  `--stage1-step`, `--stage2-steps`, `--save-interval`, `--episodes`,
  `--master-port`, `--output-root`, `--run-tag`, `--dry-run`, and
  `--check-only`.
- Produces: Stage-2 output under
  `<output-root>/<run-tag>/universal-video-action` and student-only matrix under
  `<output-root>/<run-tag>/eval/student_only`.

- [ ] **Step 1: Write failing wrapper tests**

Use executable test sentinels for the provenance preparer, Stage-2 launcher,
prompt builder, and matrix launcher. Assert real wrapper outputs and side
effects:

```python
def test_all_phase_runs_stage2_then_student_only_matrix(tmp_path):
    result, calls = run_wrapper_with_sentinels(tmp_path, phase="all")
    assert result.returncode == 0
    assert [call["kind"] for call in calls] == [
        "provenance", "stage2", "prompt", "eval"
    ]
    assert calls[1]["parent_expected_step"] == "3000"
    assert calls[1]["stage2_steps"] == "5000"
    assert calls[2]["output"].endswith("libero_wan_prompt_embeddings_all40.pt")
    assert calls[3]["roles"] == "stage2_target"

def test_eval_uses_four_paired_shards_and_all_eight_gpus(tmp_path):
    result, calls = run_wrapper_with_sentinels(tmp_path, phase="eval")
    evaluation = calls[-1]
    assert evaluation["formal_num_shards"] == "4"
    assert evaluation["formal_gpu_layout"] == "paired"
    assert evaluation["visible_devices"] == "0,1,2,3,4,5,6,7"
    assert evaluation["video_seeds"] == "0"

def test_wrapper_rejects_missing_stage1_step_3000(tmp_path):
    result = run_wrapper(tmp_path, create_parent=False)
    assert result.returncode != 0
    assert "Stage-1" in result.stderr

def test_wrapper_rejects_existing_fresh_stage2_or_eval_root(tmp_path):
    # Assert both roots fail closed and no sentinel is invoked.

def test_check_only_prints_full_plan_without_writes(tmp_path):
    # Assert Stage-2 step 5000, student-only K=1/2/4 full40 plan, and no output.
```

- [ ] **Step 2: Run wrapper tests and verify RED**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_run_cosmos_stage2_student_eval_8gpu.py
```

Expected: FAIL because the wrapper is absent.

- [ ] **Step 3: Implement argument parsing and fixed defaults**

Implement:

```bash
PHASE=all
STAGE1_STEP=3000
STAGE2_STEPS=5000
SAVE_INTERVAL=1000
EPISODES=50
MASTER_PORT=29672
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
```

Validate the run tag, positive integers, divisibility by save interval, port,
and exactly eight unique GPU ordinals. Resolve all paths canonically.

- [ ] **Step 4: Implement Stage-2 delegation**

Set:

```bash
export COSMOS_STAGE1_EXPECTED_STEP="$STAGE1_STEP"
```

Resolve:

```bash
STAGE1_CHECKPOINT="$STAGE1_ROOT/checkpoints/step_$STAGE1_STEP"
STAGE1_TARGET="$STAGE1_CHECKPOINT/target_student"
LOCK_ROOT="$RUN_ROOT/provenance-locks"
STAGE2_OUTPUT="$RUN_ROOT/universal-video-action"
STAGE2_TARGET_TRANSFORMER="$STAGE2_OUTPUT/checkpoints/step_$STAGE2_STEPS/target_student/transformer"
```

First invoke `prepare_cosmos_libero_provenance_locks.py` with the dataset,
teacher, local Cosmos model, and `STAGE1_TARGET`, writing only to `LOCK_ROOT`.
Then delegate training directly to:

```bash
COSMOS_STAGE1_ROOT="$STAGE1_CHECKPOINT" \
STUDENT_BASE_MODEL_PATH="$STAGE1_TARGET" \
COSMOS_PROVENANCE_LOCK_ROOT="$LOCK_ROOT" \
COSMOS_STAGE1_EXPECTED_STEP="$STAGE1_STEP" \
bash distillation_flowmap/run_cosmos_libero_train_8gpu.sh \
  universal-video-action \
  --steps "$STAGE2_STEPS" \
  --save-interval "$SAVE_INTERVAL" \
  --master-port "$MASTER_PORT" \
  --output-root "$OUTPUT_ROOT" \
  --run-tag "$RUN_TAG"
```

Pass the audited Cosmos/Wan environment explicitly. Require the exact
`STAGE2_TARGET_TRANSFORMER` after the child exits. This direct delegation is
required because the existing combined pipeline derives its Stage-1 parent
inside the same output root and cannot safely point at an independent Stage-1
run.

- [ ] **Step 5: Implement student-only evaluation delegation**

Build or validate:

```text
<run-root>/libero_wan_prompt_embeddings_all40.pt
```

Then call the existing matrix launcher with:

```bash
export MATRIX_ROOT="$RUN_ROOT/eval/student_only"
export S4_CKPT_ROOT="$STAGE2_TARGET_TRANSFORMER"
export S4_MATRIX_ROLES=stage2_target
export S4_FORMAL_NUM_SHARDS=4
export S4_FORMAL_GPU_LAYOUT=paired
export S4_VIDEO_SEEDS=0
export S4_EPISODES_PER_TASK="$EPISODES"
export S4_ALIGNMENT_VERIFIED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
bash evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh run
```

The wrapper must not set or print `official_teacher` as an evaluation role.
Forward INT/TERM to the active child and stop on the first nonzero child exit.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
bash -n distillation_flowmap/run_cosmos_stage2_student_eval_8gpu.sh
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_run_cosmos_stage2_student_eval_8gpu.py
```

Then:

```bash
git add \
  distillation_flowmap/run_cosmos_stage2_student_eval_8gpu.sh \
  distillation_flowmap/tests/test_run_cosmos_stage2_student_eval_8gpu.py
git commit -m "feat: chain Cosmos Stage-2 to student full40 eval"
```

---

### Task 3: Integrated contract verification and handoff

**Files:**
- Modify only if a failing regression exposes a contract defect in a file from
  Tasks 1 or 2.

**Interfaces:**
- Consumes: the Task 1 parent-step contract and Task 2 wrapper.
- Produces: a verified dry-run command ready for an eight-A800 allocation.

- [ ] **Step 1: Run related regression suites**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_run_cosmos_libero_train_8gpu.py \
  distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py \
  distillation_flowmap/tests/test_cosmos_progressive_env_schema.py \
  distillation_flowmap/tests/test_cosmos_stage1_stage2_eval_pipeline.py \
  distillation_flowmap/tests/test_cosmos_stage2_lineage.py \
  distillation_flowmap/tests/test_run_cosmos_stage2_student_eval_8gpu.py \
  evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py \
  evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py \
  evaluation/libero/tests/test_cosmos_progressive_eval_summary.py
```

Expected: all pass.

- [ ] **Step 2: Run static verification**

Run:

```bash
bash -n \
  distillation_flowmap/run_cosmos_libero_train_8gpu.sh \
  distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh \
  distillation_flowmap/run_cosmos_stage2_student_eval_8gpu.sh
git diff --check
```

- [ ] **Step 3: Run a real-path check-only when Stage-1 step 3000 exists**

Run the new wrapper with the production Stage-1 root, `--phase check`, and a
fresh output root. Require printed evidence for:

```text
COSMOS_STAGE1_EXPECTED_STEP=3000
MAX_TRAIN_STEPS=5000
S4_MATRIX_ROLES=stage2_target
S4_FORMAL_NUM_SHARDS=4
S4_FORMAL_GPU_LAYOUT=paired
S4_VIDEO_SEEDS=0
```

If Stage-1 step 3000 is not yet complete, report that external prerequisite
without weakening validation or creating a fake production checkpoint.

- [ ] **Step 4: Final review and handoff**

Confirm no training/evaluation processes were launched and provide the short
production command:

```bash
cd /kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/cosmos-stage2-progressive-base
bash distillation_flowmap/run_cosmos_stage2_student_eval_8gpu.sh \
  --phase all \
  --stage1-root /kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_cosmos_aligned_stage1_stage2_full40_8gpu_20260725/aligned-anchor-field-full40-20260725/stage1 \
  --output-root /kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_cosmos_stage2_student_full40_8gpu_20260727 \
  --run-tag s1step3000-s2step5000-student-full40
```
