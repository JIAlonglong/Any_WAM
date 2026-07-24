# LIBERO Stage-2 Deployment Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align LIBERO video-only DanceOPD with deployed shifted 1/2/4-step trajectories and train the action head on detached generated-video context without enabling action OPD.

**Architecture:** Add model-agnostic shifted-grid and semantic-query utilities, then use them in the existing DanceOPD rollout. Extend joint-input construction with explicit condition states so diagnostics and the Stage-2 bridge can reproduce deployment conditioning. The bridge reuses the existing GT action anchor at the selected DanceOPD state and cannot backpropagate into generated video.

**Tech Stack:** Python 3.10, PyTorch, pytest, LingBotVA FlowMap/DanceOPD.

## Global Constraints

- Work only in `/kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/libero-stage2-deployment-alignment`.
- Do not change or delete checkpoints, outputs, logs, W&B data, or active processes.
- Do not modify the main AnyFlow loss or Stage-1 config.
- Keep `opd_aux_action=False` and `opd_danceopd_action_velocity_weight=0`.
- Generated video supplied to the action bridge must be detached.
- The action condition stream remains GT in this patch.
- All production changes require a failing test first.

---

### Task 1: Deployment-shifted DanceOPD grids

**Files:**
- Modify: `distillation_flowmap/danceopd_query.py`
- Modify: `distillation_flowmap/flowmap_step.py`
- Modify: `distillation_flowmap/tests/test_danceopd_query.py`
- Modify: `distillation_flowmap/tests/test_danceopd_runtime_contract.py`

**Interfaces:**
- `build_shifted_terminal_path(*, scheduler, num_steps, batch_size, num_frames, num_train_timesteps, device, dtype) -> torch.Tensor`
- Result shape: `[num_steps + 1, batch_size, num_frames]`.

- [ ] Write tests asserting exact video shift-5 and action shift-0.05 paths for K=1/2/4.
- [ ] Run the focused tests and verify RED because the helper is absent.
- [ ] Implement the helper from a raw `torch.linspace(1, 0, K+1)`, apply `scheduler.apply_shift`, multiply by `num_train_timesteps`, and expand to `[K+1,B,F]`.
- [ ] Replace `_build_timestep_path(T, 0, K)` in `_danceopd_aux_transition_step` with the new helper for video and action schedulers.
- [ ] Run `test_danceopd_query.py` and `test_danceopd_runtime_contract.py`; expect PASS.
- [ ] Commit as `fix: align LIBERO DanceOPD scheduler grids`.

### Task 2: Post-update semantic endpoint queries

**Files:**
- Modify: `distillation_flowmap/flowmap_step.py`
- Modify: `distillation_flowmap/tests/test_danceopd_runtime_contract.py`
- Create: `distillation_flowmap/tests/test_libero_danceopd_semantic_rollout.py`

**Interfaces:**
- DanceOPD state index 0 is initial pure noise and is never queryable.
- Legal semantic query indices are `[1, rollout_steps]`.

- [ ] Add a failing source/runtime test asserting the rollout appends the terminal post-update state and calls `sample_semantic_query_indices`.
- [ ] Verify RED against the current pre-update-only implementation.
- [ ] Append `current_video/current_action` and final timesteps after integration.
- [ ] Replace `sample_low_noise_query_indices(n_states=K)` with `sample_semantic_query_indices(rollout_steps=K)`.
- [ ] Release the complete trajectory after selecting the per-sample state, following the RobotWin memory-safe implementation.
- [ ] Run the new semantic rollout test plus all DanceOPD tests; expect PASS.
- [ ] Commit as `fix: query post-update LIBERO DanceOPD states`.

### Task 3: Explicit condition-state contract and diagnostics

**Files:**
- Modify: `distillation_flowmap/flowmap_step.py`
- Modify: `distillation_flowmap/mechanism_diagnostics.py`
- Modify: `distillation_flowmap/tests/test_online_mechanism_diagnostics.py`
- Modify: `distillation_flowmap/tests/test_mechanism_diagnostics.py`

**Interfaces:**
- `_build_joint_input(..., condition_video=None, condition_action=None)` overrides only the corresponding `latent` condition stream.
- `_diagnostic_student_joint_map(..., condition_video=None, condition_action=None)` forwards those overrides.

- [ ] Update the diagnostic harness tests first so every captured call records noisy video/action and condition video/action separately.
- [ ] Add failing route tests:
  - student route uses `z_r_video` as both noisy and condition video;
  - teacher-video route uses `y_r_video` as both noisy and condition video while preserving student action state;
  - teacher-joint route uses `y_r_video/y_r_action` in both noisy and condition streams;
  - GT route preserves clean GT condition video.
- [ ] Verify RED because current helpers always take condition tensors from `_prepare_base_dict`.
- [ ] Add optional condition overrides to `_build_joint_input`, `_mechanism_joint_input`, and diagnostic map/rollout helpers.
- [ ] Route the four diagnostics explicitly and keep existing metric names; add condition-route metadata fields without changing reductions.
- [ ] Run both mechanism diagnostic suites; expect PASS and no gradients in diagnostic values.
- [ ] Commit as `fix: diagnose deployment-faithful LIBERO conditioning`.

### Task 4: Detached generated-video action bridge

**Files:**
- Modify: `distillation_flowmap/config_libero_fullfinetune_stage2_video_only_opd.py`
- Modify: `distillation_flowmap/flowmap_step.py`
- Create: `distillation_flowmap/video_action_bridge.py`
- Create: `distillation_flowmap/tests/test_video_action_bridge.py`
- Modify: `distillation_flowmap/tests/test_libero_video_only_opd_config.py`

**Interfaces:**
- `bridge_probability(step, warmup_end=500, mid_end=1500, start=0.25, mid=0.50, final=0.75) -> float`
- `masked_action_x0_loss(action_state, action_velocity, action_gt, sigma, valid_mask) -> torch.Tensor`
- Config:
  - `video_action_bridge=True`
  - `video_action_bridge_weight=1.0`
  - probability curriculum `0.25/0.50/0.75`

- [ ] Write failing unit tests for curriculum boundaries, masked x0 loss, and detached generated-video gradients.
- [ ] Add a failing config contract proving action OPD remains disabled.
- [ ] Verify RED because the helper/config does not exist.
- [ ] Implement the pure bridge utilities.
- [ ] In DanceOPD, use detached current/query video as `condition_video` for rollout and query forwards while keeping the GT action condition.
- [ ] Request the action output from the selected student query only when the bridge Bernoulli is active.
- [ ] Compute action x0 from the selected query action and velocity, compare against downsampled GT action with the existing valid mask, multiply by bridge weight, and add it to the DanceOPD auxiliary loss.
- [ ] Record raw loss, weighted contribution, probability, and active fraction. Do not populate the existing action OPD transition fields.
- [ ] Run bridge, config, DanceOPD, and mechanism tests; expect PASS.
- [ ] Commit as `feat: bridge generated LIBERO video to action anchor`.

### Task 5: Regression and single-GPU smoke

**Files:**
- Modify only if a test exposes an implementation defect.

- [ ] Run:
  `python -m pytest -q distillation_flowmap/tests/test_danceopd_query.py distillation_flowmap/tests/test_danceopd_runtime_contract.py distillation_flowmap/tests/test_libero_danceopd_semantic_rollout.py distillation_flowmap/tests/test_online_mechanism_diagnostics.py distillation_flowmap/tests/test_mechanism_diagnostics.py distillation_flowmap/tests/test_video_action_bridge.py distillation_flowmap/tests/test_libero_video_only_opd_config.py`
- [ ] Run the broader `distillation_flowmap/tests` CPU subset that does not require Cosmos or a simulator.
- [ ] Run a fresh one-GPU, one-step Stage-2 smoke with a new output directory and verify main loss, shifted DanceOPD, bridge loss, mechanism diagnostics, backward, and checkpoint save.
- [ ] Record exact smoke command and output path in the final handoff.
- [ ] Inspect `git status --short` and `git diff --check`; expect no unrelated changes or whitespace errors.
