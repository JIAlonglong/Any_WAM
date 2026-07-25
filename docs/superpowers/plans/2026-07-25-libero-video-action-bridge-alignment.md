# LIBERO Video-Action Bridge Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the LIBERO video-action bridge with x0 action training and deployment-time generated action history.

**Architecture:** Keep the existing DanceOPD training path and introduce one
small pure x0-loss helper. Replace only the bridge target construction and the
action-history inputs, then extend the existing mechanism diagnostics without
changing the paper metric definitions.

**Tech Stack:** Python 3.10, PyTorch, pytest, LingBot-VA FlowMap trainer.

## Global Constraints

- Do not modify the main AnyFlow objective.
- Do not enable action OPD.
- Keep rollout video detached from bridge action gradients.
- Preserve existing diagnostic JSON keys.
- Never overwrite existing checkpoints or run outputs.

---

### Task 1: Bridge x0 objective

**Files:**
- Modify: `distillation_flowmap/video_action_bridge.py`
- Modify: `distillation_flowmap/flowmap_step.py`
- Test: `distillation_flowmap/tests/test_video_action_bridge.py`

**Interfaces:**
- Produces: `masked_action_x0_teacher_forcing_loss(predicted_velocity, noisy_action, clean_action, sigma, valid_mask=None) -> Tensor`

- [ ] **Step 1: Write failing tests**

Test exact x0 reconstruction, mask handling, and sigma broadcasting.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest -q distillation_flowmap/tests/test_video_action_bridge.py
```

Expected: import failure for the missing x0 helper.

- [ ] **Step 3: Implement the minimal helper and bridge call**

Convert velocity to x0 before masked MSE. Pass
`query_action_t / num_train_timesteps` with shape `[B, 1, F, 1, 1]`.

- [ ] **Step 4: Run tests and verify GREEN**

Run the same focused pytest command and require all tests to pass.

### Task 2: Generated action history and diagnostic isolation

**Files:**
- Modify: `distillation_flowmap/flowmap_step.py`
- Modify: `distillation_flowmap/mechanism_diagnostics.py`
- Test: `distillation_flowmap/tests/test_libero_danceopd_semantic_rollout.py`
- Test: `distillation_flowmap/tests/test_online_mechanism_diagnostics.py`

**Interfaces:**
- DanceOPD rollout and bridge consume detached generated action history.
- Diagnostics emit independent clean-history and generated-history action errors.

- [ ] **Step 1: Write failing tests**

Assert that rollout and bridge use `query_action`/`current_action`, and verify
new diagnostic reductions for an action-history-only swap.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest -q \
  distillation_flowmap/tests/test_libero_danceopd_semantic_rollout.py \
  distillation_flowmap/tests/test_online_mechanism_diagnostics.py
```

Expected: source/metric assertions fail against clean-action conditioning.

- [ ] **Step 3: Implement minimal alignment**

Use `current_action.detach()` during rollout and `query_action.detach()` in the
bridge. Add diagnostic contexts without changing existing paper metrics.

- [ ] **Step 4: Run focused and regression tests**

Run all four Stage-2 diagnostic/bridge test modules, followed by `py_compile`
and `git diff --check`.

### Task 3: Single-GPU smoke validation

**Files:**
- No tracked-file changes.

**Interfaces:**
- Uses the existing Stage-1 checkpoint and one-step training stop.

- [ ] **Step 1: Run a single-GPU one-step smoke in a new output directory**

Use a unique `RUN_TAG` and `MAX_TRAIN_STEPS=1`; never reuse the formal run root.

- [ ] **Step 2: Validate logs**

Require finite main loss, finite bridge x0 loss when active, correct
`action_downsample_factor=1`, and no backward/attention errors.

- [ ] **Step 3: Record the new eight-GPU launch command**

Use a new output directory and preserve the existing formal checkpoints.
