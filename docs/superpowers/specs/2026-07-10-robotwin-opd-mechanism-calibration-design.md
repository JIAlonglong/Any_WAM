# RobotWin OPD Mechanism Calibration Design

## Context

The current RobotWin Stage2 OPD path does not cleanly isolate endpoint and
same-state velocity supervision:

- In endpoint mode, `video_transition_loss` and `opd_endpoint_aux_loss`
  compute the same denoised endpoint error.
- The legacy transition multiplier (`25.0`) makes endpoint supervision much
  stronger than the nominal velocity weight (`0.1`).
- `velocity_only_opd` sets the transition group to zero while retaining an
  anchor cap relative to that group, which scales the velocity objective to
  zero.
- The default `last_step` rollout gradient only credits the final student
  Euler step, while the paper-facing evaluation uses four student steps.
- Existing OPD training pairs are teacher 4-step to student 1/2-step; they do
  not directly train the paper setting of student `K=4`.

The next experiment is a mechanism calibration, not the final ablation. It
must use a small, held-out RobotWin subset and must not retrain Stage1.

## Goals

1. Add an opt-in explicit OPD loss composition in which endpoint and velocity
   appear exactly once with independently visible weights.
2. Ensure endpoint-only, velocity-only, and full variants produce nonzero,
   measurable gradients for their enabled branches.
3. Compare terminal-loss credit assignment through `last_step` and `full`
   four-step student rollouts without changing the OPD target definition.
4. Run a controlled two-task calibration from one existing shared Stage1
   checkpoint using fixed train and held-out manifests.
5. Preserve the existing Stage2 behavior unless the new explicit mode is
   selected.

## Non-Goals

- Do not run the 10-12 task, three-seed final ablation yet.
- Do not change Cosmos code or configs.
- Do not add perceptual, PFM, KTO, FVD, or new action-velocity objectives.
- Do not replace the default `last_step` behavior in existing LingBot runs.
- Do not introduce random-boundary supervision before measuring the existing
  `full` rollout-gradient mode.

## Loss Design

Add `OPD_LOSS_COMPOSITION=explicit_hybrid`. The default remains `legacy`.

For endpoint teacher targets, define the canonical video losses as:

```text
L_endpoint_video = distance(x0_S, x0_T)
L_velocity_video = distance(S(sg(x_r^S), r), T(sg(x_r^S), r))

x0_S = x_r^S - sigma_r * S(sg(x_r^S), r)
x0_T = x_r^T - sigma_r * T(x_r^T, r)
```

The explicit calibration objective is:

```text
L_OPD = beta_end_video * L_endpoint_video
      + beta_vel_video * L_velocity_video
      + beta_end_action * L_endpoint_action
```

For the first calibration, action OPD is disabled and
`beta_end_action=0`. `LOCAL_FM_WEIGHT` and `ACTION_LOCAL_FM_WEIGHT` are also
zero so that endpoint and velocity attribution is not confounded.

Explicit mode does not apply `OPD_TRANSITION_GROUP_WEIGHT` or
`OPD_ANCHOR_CAP_RATIO` to these three canonical terms. The existing global
`OPD_AUX_WEIGHT`, warmup, interval, probability, and loss clip still apply.
Legacy mode keeps its current grouping and cap behavior unchanged.

Diagnostics must log each raw loss, configured beta, effective contribution,
contribution ratio, and the total OPD loss. A velocity-only synthetic test
must prove that its loss and parameter gradient remain nonzero.

## Rollout Credit Assignment

Use the existing terminal OPD objective for both modes:

- `last_step`: early student rollout steps are detached; the final Euler step
  and terminal velocity query receive gradients.
- `full`: all four student Euler steps remain in the autograd graph and the
  same terminal endpoint/velocity objective credits the complete trajectory.

The mechanism comparison changes only `OPD_ROLLOUT_GRAD_MODE`; it does not add
intermediate teacher targets. This keeps the comparison aligned with the
paper OPD definition.

The default calibration pair is `teacher N=8 -> student K=4`. Before training,
run a held-out teacher-only check comparing N=4 and N=8. Use N=8 only when its
mean denoised teacher-to-GT endpoint MSE is at least 1% lower than N=4;
otherwise use N=4 and record the decision in the run manifest.

## Small-Data Protocol

- Teacher: LingBot-VA RobotWin checkpoint.
- Stage1: reuse the existing seed-0 shared Stage1 `step_5000` checkpoint.
- Tasks: `place_a2b_right` and `open_microwave`.
- Train samples: 20 per task (40 total).
- Held-out samples: 10 per task (20 total), disjoint manifest.
- Stage2 calibration: 750 optimizer steps.
- Seed: 0 for the first mechanism gate.
- OPD action: disabled for all mechanism variants.
- Student evaluation steps: fixed `K=4`.
- Variants:
  - `calib_w_o_opd`
  - `calib_endpoint_only`
  - `calib_velocity_only`
  - `calib_full_last_step`
  - `calib_full_full_grad`

Each variant runs on one H100. No DDP is used. The launcher must require an
explicit Stage1 checkpoint and must emit a dry-run manifest showing task,
sample, step, loss, rollout, and checkpoint settings before training.

## Evaluation And Gates

All variants use the same held-out manifest, timestep pairs, noise seeds, and
teacher targets. Report per-task and aggregate values for:

- denoised endpoint error;
- same-state velocity error;
- four-step rollout drift;
- video endpoint error;
- action endpoint error from the normal joint model output;
- OPD branch contributions and branch gradient norms;
- seconds per optimizer step and peak GPU memory.

The mechanism passes the first gate only when:

1. endpoint-only reduces held-out endpoint error versus `calib_w_o_opd`;
2. velocity-only reduces held-out same-state velocity error versus
   `calib_w_o_opd`;
3. full does not regress either targeted metric by more than 2% relative to
   its corresponding single-objective variant;
4. full-gradient improves rollout drift over last-step and takes no more than
   1.5 times the average optimizer-step time;
5. all enabled OPD branches have finite, nonzero effective contributions and
   gradients.

These results may be described only as a trend. Passing the gate authorizes a
larger multi-task, multi-seed ablation; it is not itself a paper conclusion.

## Compatibility And Failure Handling

- `OPD_LOSS_COMPOSITION` defaults to `legacy`.
- Existing variant names and final-ablation scripts retain their current
  behavior.
- The calibration launcher rejects missing Stage1 checkpoints, overlapping
  train/held-out indices, task counts other than two, sample limits above the
  calibration defaults unless explicitly overridden, and unsupported loss or
  rollout modes.
- Non-finite component losses identify the offending component in the error.
- Each logical change is committed separately so it can be reverted without
  disturbing existing experiment outputs.
