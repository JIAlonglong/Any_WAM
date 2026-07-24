# LIBERO LingBotVA Video-Only OPD Stage-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested 8×A800 pipeline that trains LingBotVA LIBERO Stage-2 with video-only universal DanceOPD, records mechanism diagnostics, evaluates checkpoint video-to-action transfer, and runs 1/2/4-step closed-loop evaluation over all 40 standard LIBERO tasks.

**Architecture:** A dedicated immutable Stage-2 config enforces video-only OPD while preserving the original joint AnyFlow/action main objective. Every formal rollout predicts video and action from the same pre-update joint state and advances them together. Training, offline checkpoint evaluation, and closed-loop evaluation are separate commands coordinated by a small phase-based pipeline; closed-loop workers use two GPUs per suite and split each 10-task suite into non-overlapping halves.

**Tech Stack:** Python 3.10, PyTorch/FSDP, pytest, Bash, LingBotVA/WanVA, LIBERO, TensorBoard, W&B offline.

## Global Constraints

- Work only in `/kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/libero-lingbotva-video-opd`.
- Never overwrite existing output directories or checkpoints.
- Train from Stage-1 `step_2000/target_student` with `RESUME_ONLINE_FROM_TARGET=1`.
- Train for 10,000 optimizer steps on exactly eight visible GPUs and save every 1,000 steps.
- OPD supervises video endpoint and video same-state velocity only; every action OPD contribution is zero.
- Preserve the action main loss and feed it the detached student rollout video context.
- Closed-loop formal evaluation covers all 40 tasks in `libero_10`, `libero_spatial`, `libero_object`, and `libero_goal`.
- Evaluate matched joint 1/1, 2/2, and 4/4-step deployment contracts for the
  original teacher, Stage-1, and Stage-2. Do not add a post-video action
  forward to the formal path.
- Training and closed-loop evaluation run serially, never concurrently.

---

### Task 1: Port deterministic mechanism diagnostics

**Files:**
- Create: `distillation_flowmap/mechanism_diagnostics.py`
- Modify: `distillation_flowmap/flowmap_step.py`
- Modify: `distillation_flowmap/flowmap_trainer.py`
- Create: `distillation_flowmap/tests/test_mechanism_diagnostics.py`
- Create: `distillation_flowmap/tests/test_online_mechanism_diagnostics.py`

**Interfaces:**
- Consumes: existing student/teacher rollout helpers in `FlowMapDistiller`.
- Produces: `compute_mechanism_metric_samples(...)`, distributed finite-stat reduction, and `FlowMapDistiller._maybe_run_mechanism_diagnostics(...)`.

- [ ] **Step 1: Apply the already tested RobotWin diagnostic series**

Run:

```bash
git cherry-pick 2f4489c 31e8fda 735d8d5 562ec91 02fcffc 89e4652
```

Expected: six commits apply, or a conflict identifies a concrete difference from the current Cosmos-derived base.

- [ ] **Step 2: Resolve only backend-neutral conflicts**

Keep the generic metric contract:

```python
{
    "mechanism/g_anchor",
    "mechanism/g_anchor_mse",
    "mechanism/g_comp",
    "mechanism/g_comp_mse",
    "mechanism/video_endpoint_error",
    "mechanism/video_field_match_error",
    "mechanism/action_error_student_context",
    "mechanism/action_error_teacher_video_context",
    "mechanism/action_error_teacher_joint_context",
    "mechanism/video_to_action_oracle_gain",
}
```

Do not import RobotWin dataset constants into LIBERO configuration.

- [ ] **Step 3: Run the diagnostic tests**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_mechanism_diagnostics.py \
  distillation_flowmap/tests/test_online_mechanism_diagnostics.py
```

Expected: all tests pass with no distributed synchronization failures.

### Task 2: Add immutable LIBERO video-only Stage-2 config

**Files:**
- Create: `distillation_flowmap/config_libero_fullfinetune_stage2_video_only_opd.py`
- Create: `distillation_flowmap/tests/test_libero_video_only_opd_config.py`

**Interfaces:**
- Consumes: `cfg` from `config_libero_fullfinetune_stage2_anyflow`.
- Produces: a validated `cfg` with universal 1/2/4 video DanceOPD and zero action OPD.

- [ ] **Step 1: Write failing config tests**

Tests must assert:

```python
assert cfg.opd_query_mode == "danceopd"
assert cfg.opd_danceopd_rollout_step_choices == (2, 4)
assert cfg.opd_rollout_step_pairs == [[8, 1], [8, 2], [8, 4]]
assert cfg.opd_danceopd_endpoint_weight == 1.0
assert cfg.opd_danceopd_velocity_weight == 1.0
assert cfg.opd_aux_action is False
assert cfg.opd_danceopd_action_velocity_weight == 0.0
assert cfg.opd_joint_action_rollout is False
assert cfg.action_loss_weight > 0
assert cfg.gt_regression_weight > 0
assert cfg.mechanism_diagnostics is True
```

The test also verifies `MAX_TRAIN_STEPS=10000`, `SAVE_INTERVAL=1000`, diagnostic interval 50, and seed 42.

- [ ] **Step 2: Verify the test fails**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_libero_video_only_opd_config.py
```

Expected: FAIL because the config module does not exist.

- [ ] **Step 3: Implement the config**

The module deep-copies the existing LIBERO Stage-2 config, parses the universal rollout choices, enables mechanism diagnostics, and raises if any action OPD weight is non-zero.

- [ ] **Step 4: Verify config tests**

Run the new test plus `test_libero_stage2_config.py`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add distillation_flowmap/config_libero_fullfinetune_stage2_video_only_opd.py \
  distillation_flowmap/tests/test_libero_video_only_opd_config.py
git commit -m "feat: configure LIBERO video-only OPD stage2"
```

### Task 3: Make DanceOPD video-only rollout explicit

**Files:**
- Modify: `distillation_flowmap/flowmap_step.py`
- Modify: `distillation_flowmap/tests/test_online_mechanism_diagnostics.py`
- Create: `distillation_flowmap/tests/test_libero_video_only_opd_step.py`

**Interfaces:**
- Consumes: `cfg.opd_joint_action_rollout`.
- Produces: DanceOPD video rollout that does not update the action noisy state when the flag is false, while the normal action main loss still consumes `student_x_r.detach()`.

- [ ] **Step 1: Write failing behavioral tests**

One test invokes the DanceOPD rollout helper with `opd_joint_action_rollout=False` and asserts the action state passed at each rollout step is unchanged. A second source-contract test verifies:

```python
video_context_r = student_x_r.detach() if self.distill_video else video_noisy_latents
```

remains in the action main-loss path.

- [ ] **Step 2: Verify RED**

Run the focused test and expect the action-state test to fail because DanceOPD currently updates action unconditionally.

- [ ] **Step 3: Implement the minimal branch**

When `opd_joint_action_rollout=False`, keep the action context fixed and do not require or integrate an action velocity during the OPD rollout. Do not change `distill_action`, GT regression, or action-aware main-loss execution.

- [ ] **Step 4: Verify GREEN**

Run the focused test and both diagnostic test modules.

- [ ] **Step 5: Commit**

```bash
git add distillation_flowmap/flowmap_step.py \
  distillation_flowmap/tests/test_libero_video_only_opd_step.py \
  distillation_flowmap/tests/test_online_mechanism_diagnostics.py
git commit -m "fix: isolate LIBERO video-only DanceOPD rollout"
```

### Task 4: Add robust 8-GPU training launcher

**Files:**
- Create: `distillation_flowmap/run_libero_video_only_opd_stage2_8gpu.sh`
- Create: `distillation_flowmap/tests/test_run_libero_video_only_opd_stage2_8gpu.py`

**Interfaces:**
- Consumes: Stage-1 checkpoint, teacher, dataset, empty embedding, and direct Python path.
- Produces: one validated `torchrun --nproc_per_node=8` launch, `launch_env.txt`, TensorBoard, and W&B offline output.

- [ ] **Step 1: Write failing dry-run tests**

Tests create fake checkpoint/data layouts and assert:

```text
CONFIG_FILE=distillation_flowmap.config_libero_fullfinetune_stage2_video_only_opd
MAX_TRAIN_STEPS=10000
SAVE_INTERVAL=1000
OPD_DANCEOPD_ROLLOUT_STEPS=2,4
OPD_DANCEOPD_ACTION_VELOCITY_WEIGHT=0.0
OPD_AUX_ACTION=0
MECHANISM_DIAGNOSTIC_INTERVAL=50
--nproc_per_node=8
```

They also assert rejection of missing files, duplicate/non-eight GPU lists, invalid ports, and existing non-empty output roots.

- [ ] **Step 2: Verify RED**

Run the new test and expect failure because the launcher is absent.

- [ ] **Step 3: Implement the launcher**

Support:

```text
--dry-run
--steps N
--save-interval N
--master-port PORT
--output-dir PATH
--run-tag TAG
--resume-step N
```

Use `PYTHON` directly; do not depend on shell conda initialization.

- [ ] **Step 4: Verify GREEN**

Run the launcher tests and a real-path `--dry-run`.

- [ ] **Step 5: Commit**

Commit launcher and tests with message `feat: launch LIBERO video-only OPD on eight GPUs`.

### Task 5: Parameterize the single-worker LIBERO closed loop

**Files:**
- Modify: `evaluation/libero/run_eval_new.sh`
- Modify: `evaluation/libero/client.py`
- Create: `evaluation/libero/tests/test_run_eval_new_contract.py`

**Interfaces:**
- Consumes: `LIBERO_BENCHMARK`, task range, GPU ID, video/action step counts, ports, and checkpoint path.
- Produces: one suite/range worker with deterministic result paths.

- [ ] **Step 1: Write failing contract tests**

Dry-run tests assert all four suite names are accepted, requested suite reaches `client.py`, selected GPU reaches the server environment, and video/action steps are printed.

- [ ] **Step 2: Verify RED**

Expected: failure because `run_eval_new.sh` currently hard-codes `libero_10`.

- [ ] **Step 3: Implement suite/device parameterization and `CHECK_ONLY` output**

Keep existing default behavior compatible. Add no-overwrite output validation and preserve the same observation preprocessing, action normalization, four-actions-per-frame execution, and maximum 800 environment steps.

- [ ] **Step 4: Verify GREEN**

Run the new tests and existing LIBERO service/launcher tests.

- [ ] **Step 5: Commit**

Commit with message `feat: parameterize LIBERO closed-loop worker`.

### Task 6: Add 40-task, 8-GPU, 1/2/4-step closed-loop orchestrator

**Files:**
- Create: `evaluation/libero/run_lingbotva_4suite_124_eval_8gpu.sh`
- Create: `evaluation/libero/merge_lingbotva_4suite_results.py`
- Create: `evaluation/libero/tests/test_run_lingbotva_4suite_124_eval_8gpu.py`
- Create: `evaluation/libero/tests/test_merge_lingbotva_4suite_results.py`

**Interfaces:**
- Consumes: checkpoint transformer path, output root, episode count, eight GPU IDs, and base ports.
- Produces: eight workers per K, 40 complete task results, per-suite and overall macro success JSON, and a 1/2/4 matrix summary.

- [x] **Step 1: Write failing assignment and merger tests**

Assert the assignment is exactly:

```text
GPU0 libero_10 0:5
GPU1 libero_10 5:10
GPU2 libero_spatial 0:5
GPU3 libero_spatial 5:10
GPU4 libero_object 0:5
GPU5 libero_object 5:10
GPU6 libero_goal 0:5
GPU7 libero_goal 5:10
```

For each K in 1, 2, 4, assert unique WebSocket/master ports, matched video/action steps, and no output creation in dry-run mode. Merger tests reject missing tasks and wrong episode counts.

- [x] **Step 2: Verify RED**

Expected: scripts do not exist.

- [x] **Step 3: Implement orchestrator and atomic merger**

Workers run concurrently within one K; K values run serially. A failed worker terminates the K and prevents summary publication. The merger writes per-task, per-suite macro, and overall macro metrics via temporary file plus `os.replace`.

- [x] **Step 4: Verify GREEN**

Run both new test modules and dry-run against the real Stage-1 checkpoint.

- [ ] **Step 5: Commit**

Commit with message `feat: evaluate all standard LIBERO tasks on eight GPUs`.

### Task 7: Add checkpoint offline intervention evaluator

**Files:**
- Create: `distillation_flowmap/eval_libero_video_action_conditions.py`
- Create: `distillation_flowmap/run_libero_video_action_conditions_8gpu.sh`
- Create: `distillation_flowmap/tests/test_eval_libero_video_action_conditions.py`

**Interfaces:**
- Consumes: checkpoint, teacher, dataset, fixed seed, student budgets 1/2/4.
- Produces: JSON rows for student rollout video, teacher rollout video, GT-at-r video, and clean-GT video, with ground-truth and teacher-endpoint action errors.

- [ ] **Step 1: Write failing pure metric and condition-routing tests**

Tests use small tensors and fake rollout callbacks to prove every condition reaches the action forward and that 1/2/4 rows use separately rolled action states.

- [ ] **Step 2: Verify RED**

Expected: evaluator module is missing.

- [ ] **Step 3: Implement evaluator by reusing existing rollout helpers**

Use fixed seed 42 and emit:

```json
{
  "student_steps": 1,
  "video_condition": "student_rollout",
  "action_gt_mse": 0.0,
  "action_teacher_endpoint_mse": 0.0,
  "video_gt_mse": 0.0,
  "video_teacher_mse": 0.0
}
```

plus derived oracle gain and Stage-1 drift fields. Do not load or execute a LIBERO environment.

- [ ] **Step 4: Verify GREEN**

Run unit tests, then a one-batch Stage-1 target smoke when GPU memory is available.

- [ ] **Step 5: Commit**

Commit with message `feat: evaluate LIBERO video-to-action transfer`.

### Task 8: Add serial train/eval pipeline and operator guide

**Files:**
- Create: `distillation_flowmap/run_libero_video_opd_train_eval_8gpu.sh`
- Create: `distillation_flowmap/tests/test_run_libero_video_opd_train_eval_8gpu.py`
- Modify: `evaluation/libero/EVAL_PROMPT.md`

**Interfaces:**
- Consumes: `--phase train|eval|all`.
- Produces: a serial training command followed by teacher, Stage-1, and
  Stage-2 full 40-task closed-loop evaluation at identical joint budgets.

- [ ] **Step 1: Write failing phase-order tests**

Assert `all` runs train → offline → closed-loop serially, propagates the final checkpoint, and stops on the first failure.

- [ ] **Step 2: Verify RED**

Expected: pipeline absent.

- [ ] **Step 3: Implement pipeline and documentation**

The pipeline defaults to final `step_10000/target_student/transformer`. `closed-loop` supports 10 episodes per task by default and `EPISODES_PER_TASK=50` for formal evaluation.

- [ ] **Step 4: Run full verification**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_mechanism_diagnostics.py \
  distillation_flowmap/tests/test_online_mechanism_diagnostics.py \
  distillation_flowmap/tests/test_libero_video_only_opd_config.py \
  distillation_flowmap/tests/test_libero_video_only_opd_step.py \
  distillation_flowmap/tests/test_run_libero_video_only_opd_stage2_8gpu.py \
  distillation_flowmap/tests/test_eval_libero_video_action_conditions.py \
  distillation_flowmap/tests/test_run_libero_video_only_opd_pipeline_8gpu.py \
  evaluation/libero/tests/test_run_eval_new_contract.py \
  evaluation/libero/tests/test_run_lingbotva_4suite_124_eval_8gpu.py \
  evaluation/libero/tests/test_merge_lingbotva_4suite_results.py
```

Expected: PASS.

- [ ] **Step 5: Run real-path dry runs**

Run the training, offline, closed-loop, and all-phase launchers in dry-run mode. Expected: all paths resolve, exactly eight GPUs are planned, 40 tasks are covered for each K, and no output directories are created.

- [ ] **Step 6: Run one-step GPU smoke**

Set `MAX_TRAIN_STEPS=1`, a fresh smoke output, and save interval 1. Expected: finite forward/backward, mechanism metrics, and a complete checkpoint.

- [ ] **Step 7: Commit**

Commit pipeline/docs with message `docs: add LIBERO stage2 train and evaluation workflow`.
