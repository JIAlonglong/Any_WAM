# Equal-NFE DanceOPD Endpoint Implementation Plan

> For agentic workers: use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** Add a fair T1/T2/T4/T8 versus S1/S2/S4 offline evaluator and an opt-in Stage2 objective that combines an independent teacher endpoint target with DanceOPD local velocity matching.

**Architecture:** A pure step-list helper lets both offline evaluators calculate one fixed student rollout for every K and compare it to multiple frozen-teacher solvers without changing scalar CLI behavior. The DanceOPD path gains a video-only independent endpoint helper: it samples one data-noised x_t, rolls the student K steps and teacher N steps to the same r, aligns denoised endpoints, and adds it to the existing detached DanceOPD velocity loss before its one auxiliary backward call.

**Tech Stack:** Python 3.10, PyTorch, FlowMatch scheduler, unittest, RobotWin manifest evaluator.

## Global Constraints

- Keep LingBot-VA and RoboTwin only; do not modify Cosmos code paths.
- Preserve exact Dance-only behavior when OPD_DANCEOPD_ENDPOINT_WEIGHT is zero.
- Do not introduce raw GT regression as a Stage2 loss.
- The endpoint teacher target must be an independent frozen N-step rollout from the same data-noised x_t and r as the K-step student rollout.
- DanceOPD remains video-loss-only; action remains a joint conditioning state with no new action OPD target.
- All formal runs reuse the shared Stage1 checkpoint, fixed core2 manifests, and held-out cache.

---

### Task 1: Validate multi-teacher schedules

**Files:**
- Create: distillation_flowmap/rollout_eval_steps.py
- Create: distillation_flowmap/tests/test_rollout_eval_steps.py

**Interfaces:**
- normalize_rollout_steps(values: Sequence[int]) -> list[int]
- Returns sorted unique positive counts; rejects empty, duplicate, zero, and negative values.

- [ ] Step 1: Write failing tests for [8,1,4,2] -> [1,2,4,8], and ValueError cases [1,1], [0], and [].
- [ ] Step 2: Run python -m pytest distillation_flowmap/tests/test_rollout_eval_steps.py -q. Expected: import failure before the helper exists.
- [ ] Step 3: Implement conversion to int, nonempty/positive validation, duplicate validation, and sorted return.
- [ ] Step 4: Re-run the focused test. Expected: 2 passed.
- [ ] Step 5: Commit only the helper and test with message: feat: validate multi-step rollout evaluation schedules.

### Task 2: Add equal-NFE teacher evaluation

**Files:**
- Modify: distillation_flowmap/rollout_eval_stage2.py
- Modify: distillation_flowmap/rollout_eval_video_stage2.py
- Modify: distillation_flowmap/ablation/run_final_eval.sh
- Modify: distillation_flowmap/tests/test_rollout_eval_steps.py

**Interfaces:**
- --teacher-steps accepts one or more integers, for example 1 2 4 8.
- Existing one-step calls remain valid.
- Metrics retain rollout_eval/{pair}/s{K}_t{N}/... names.
- Video evaluator writes teacher_t1, teacher_t2, teacher_t4, and teacher_t8 from the same pair seed.

- [ ] Step 1: Write source-contract tests requiring nargs="+" and normalize_rollout_steps(args.teacher_steps) in both evaluator scripts.
- [ ] Step 2: Run the test. Expected: failure because teacher steps are currently scalar.
- [ ] Step 3: Normalize requested teacher counts after CLI parsing. Compute each K-step student rollout once, record GT/action metrics once, retain detached x/v tensors, then loop teacher counts and call _teacher_integrate_to_r with num_steps=teacher_step.
- [ ] Step 4: Record every student/teacher comparison under the existing s{K}_t{N} namespace. Save one teacher video per N and do not recompute student rollouts per N. Defer eval_empty_cache cleanup until all teacher comparisons for a pair complete.
- [ ] Step 5: Make run_final_eval.sh default to teacher list 1 2 4 8 while accepting a scalar override unchanged.
- [ ] Step 6: Run focused tests plus py_compile for both evaluators.
- [ ] Step 7: Commit with message: feat: evaluate teacher and student at equal rollout budgets.

### Task 3: Add an independent endpoint term to DanceOPD

**Files:**
- Modify: distillation_flowmap/danceopd_query.py
- Modify: distillation_flowmap/flowmap_step.py
- Modify: distillation_flowmap/tests/test_danceopd_query.py

**Interfaces:**
- denoised_endpoint_mse(student_x, student_v, teacher_x, teacher_v, sigma_r) -> Tensor
- _danceopd_independent_endpoint_loss(self, batch, cfg_scale) -> (Tensor, dict)
- OPD_DANCEOPD_ENDPOINT_WEIGHT defaults to 0.0; OPD_DANCEOPD_VELOCITY_WEIGHT defaults to 1.0.

- [ ] Step 1: Write a scalar failing test: student x=2,v=1, teacher x=3,v=.5, sigma=.5 has endpoint MSE .0625; backward gives student gradients and no teacher gradients.
- [ ] Step 2: Add failing source-contract assertions for endpoint weight, independent endpoint helper, and one loss.backward call in the DanceOPD method.
- [ ] Step 3: Implement denoised_endpoint_mse using x0_S=x_S-sigma*v_S and detached x0_T=x_T-sigma*v_T.
- [ ] Step 4: Implement video-only endpoint rollout: sample t/r and noise once, build the same joint input as legacy OPD, choose N/K from opd_rollout_step_pairs, run K-step student and N-step frozen teacher from the same x_t to r, then return endpoint MSE and N/K/t/r diagnostics.
- [ ] Step 5: Keep terminal-prior Dance rollout untouched. Combine raw = velocity_weight*velocity_loss + endpoint_weight*endpoint_loss, apply OPD_AUX_WEIGHT once, and call the existing single loss.backward once. Do not invoke the endpoint helper when the endpoint weight is zero.
- [ ] Step 6: Log raw losses, weighted contributions, and ratios through existing OPD metric keys.
- [ ] Step 7: Run focused DanceOPD tests and py_compile; commit with message: feat: combine DanceOPD with teacher endpoint targets.

### Task 4: Register the composite calibration variant

**Files:**
- Modify: distillation_flowmap/ablation/robotwin_stepwam_variants.json
- Modify: distillation_flowmap/tests/test_robotwin_ablation_launcher.py

**Interfaces:**
- New variant: calib_endpoint_danceopd_i1.
- It uses OPD_QUERY_MODE=danceopd, positive endpoint/velocity weights, i1 cadence, and no legacy same-state/action OPD terms.

- [ ] Step 1: Write a failing JSON contract test: variant exists, both new weights are positive, OPD_AUX_ACTION is 0, OPD_SAME_STATE_VELOCITY_WEIGHT is 0.0, and VIDEO_TRANSITION_WEIGHT is 0.0.
- [ ] Step 2: Run the focused launcher test. Expected: variant absent.
- [ ] Step 3: Add the variant by copying DanceOPD i1 terminal-prior, 16-step, Beta(5,2), and diagnostic settings. Add endpoint and velocity weights of 1.0; leave legacy endpoint/velocity weights at zero.
- [ ] Step 4: Run launcher, DanceOPD, and schedule tests. Commit with message: feat: add endpoint DanceOPD calibration variant.

### Task 5: Run controlled smoke and equal-NFE evaluation

**Files:**
- Runtime output only: distillation_flowmap/output_robotwin_stepwam_ablation/protocol_endpoint_danceopd_smoke_v1/
- Modify after result: docs/experiments/2026-07-12-danceopd-component-confirmation.md

**Interfaces:**
- Uses shared Stage1, core2 20 train/10 held-out samples per task, seed 0, and a 100-step single-GPU Stage2 smoke.
- Produces one checkpoint, finite-loss diagnostics, equal-NFE JSON metrics, and T1/T2/T4/T8 plus S1/S2/S4 videos.

- [ ] Step 1: Launch calib_endpoint_danceopd_i1 with task-preset core2, 20 train, 10 held-out, protocol seed 0, Stage2 100, single GPU, and shared Stage1 step 5000.
- [ ] Step 2: Require finite losses, terminal_prior_max_error <= 1e-6, nonzero endpoint and Dance velocity contributions, and a saved step_100 checkpoint.
- [ ] Step 3: Evaluate held-out fixed pair seeds with student steps 1 2 4 and teacher steps 1 2 4 8. Save all teacher/student videos and contact sheets.
- [ ] Step 4: Record run root, git hashes, diagnostics, and metrics. Do not commit checkpoints, videos, or generated JSON.

