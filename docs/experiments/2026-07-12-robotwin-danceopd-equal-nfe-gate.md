# RobotWin DanceOPD Equal-NFE Held-Out Gate - 2026-07-12

## Scope and status

This is a preliminary mechanism gate, not a paper ablation result.

- Teacher: LingBot-VA.
- Dataset: RoboTwin representative `core2`: `place_a2b_right` and `open_microwave`.
- Train cache: 20 samples per task. Held-out cache: 10 samples per task, 20 held-out records total.
- Shared Stage-1 initialization:
  `distillation_flowmap/output_robotwin_stepwam_ablation/protocol_final12_5000/shared_stage1/representative/seed_0/stage1/checkpoints/step_5000`.
- Compared Stage-2 checkpoints: `step_250`, `calib_w_o_opd` versus `calib_danceopd_i1`.
- The baseline run manifest has git hash `aeff3a0`; the DanceOPD run manifest has git hash `13b950a`. They share Stage-1 and the core2 manifests, but the code hashes differ. This is another reason the result is only a gate.

DanceOPD settings in the compared checkpoint:

- terminal-prior joint video/action rollout of 16 no-grad steps;
- one low-noise Beta(5, 2) query per update;
- video same-state velocity loss only;
- no action OPD target and no legacy endpoint/fixed-state velocity term.

## Equal-NFE evaluation protocol

The current evaluator was run after the teacher `eval()` correction and uses one shared teacher cache:

- eval pairs: `1000->0`, `1000->500`, `750->250`;
- students: `S1`, `S2`, `S4`;
- frozen teacher solvers: `T1`, `T2`, `T4`, `T8`;
- all variants use the same held-out manifest, pair seeds, noised `x_t`, and cached teacher targets;
- teacher cache:
  `distillation_flowmap/output_robotwin_stepwam_ablation/protocol_equal_nfe_existing_v1/teacher_cache_t1_t2_t4_t8.pt`;
- metrics shown below average the three fixed eval pairs and all 20 held-out records.

Artifacts:

- baseline metrics:
  `distillation_flowmap/output_robotwin_stepwam_ablation/protocol_equal_nfe_existing_v1/numeric_baseline/metrics.json`;
- Dance primary metrics:
  `distillation_flowmap/output_robotwin_stepwam_ablation/protocol_equal_nfe_existing_v1/numeric_dance/metrics.json`;
- Dance isolated repeat metrics:
  `distillation_flowmap/output_robotwin_stepwam_ablation/protocol_equal_nfe_existing_v1/numeric_dance_isolated/metrics.json`;
- equal-NFE videos and contact sheets:
  `distillation_flowmap/output_robotwin_stepwam_ablation/protocol_equal_nfe_existing_v1/video_baseline/videos/` and
  `distillation_flowmap/output_robotwin_stepwam_ablation/protocol_equal_nfe_existing_v1/video_dance/videos/`.

## Aggregate held-out results

Lower is better. The reported Dance value is the isolated GPU1 repeat.

| Budget | Metric | Shortcut-only | DanceOPD i1 | Delta vs shortcut-only |
| --- | --- | ---: | ---: | ---: |
| S1/T1 | video GT endpoint MSE | 0.11191910 | 0.10104741 | -9.71% |
| S1/T1 | student vs equal-NFE teacher x MSE | 0.04325304 | 0.04271303 | -1.25% |
| S1/T1 | action GT endpoint MSE | 0.000344744 | 0.000330857 | -4.03% |
| S2/T2 | video GT endpoint MSE | 0.07949489 | 0.07237971 | -8.95% |
| S2/T2 | student vs equal-NFE teacher x MSE | 0.02041093 | 0.02038751 | -0.11% |
| S2/T2 | action GT endpoint MSE | 0.000343380 | 0.000328820 | -4.24% |
| S4/T4 | video GT endpoint MSE | 0.07208520 | 0.06472200 | -10.21% |
| S4/T4 | student vs equal-NFE teacher x MSE | 0.01697581 | 0.01661523 | -2.12% |
| S4/T4 | student vs equal-NFE teacher velocity MSE | 0.11674100 | 0.11456623 | -1.86% |
| S4/T4 | action GT endpoint MSE | 0.000343868 | 0.000329441 | -4.20% |
| S1/T8 | student vs T8 teacher x MSE | 0.06603308 | 0.06097053 | -7.67% |
| S2/T8 | student vs T8 teacher x MSE | 0.02811434 | 0.02668205 | -5.09% |
| S4/T8 | student vs T8 teacher x MSE | 0.01926550 | 0.01830953 | -4.96% |

The equal-NFE teacher GT metrics are exactly identical between variants because both use the same teacher cache. This confirms that the observed delta is student-side, rather than a different teacher target.

## Per-task diagnostic at S4/T4

| Task | Video GT endpoint MSE | Student vs teacher x MSE | Action GT endpoint MSE |
| --- | --- | --- | --- |
| `open_microwave` | 0.0374288 -> 0.0343916 (-8.11%) | 0.00816517 -> 0.00908432 (+11.26%) | 0.000550042 -> 0.000533033 (-3.09%) |
| `place_a2b_right` | 0.106742 -> 0.0950524 (-10.95%) | 0.0257865 -> 0.0241461 (-6.36%) | 0.000137695 -> 0.000125848 (-8.60%) |

This mixed `open_microwave` teacher-retention result is important: DanceOPD improves GT endpoint and action metrics there, but not every teacher-relative metric. The aggregate trend should therefore not be simplified to a universal improvement claim.

## Repeat and visual audit

- The GPU4 Dance evaluation and an isolated GPU1 repeat have nonzero numerical variation, consistent with nondeterministic CUDA/FlexAttention execution. The isolated repeat is used in the table.
- Across the aggregate video GT metric, the two Dance evaluations differ by 1.72% at S1, 0.95% at S2, and 0.37% at S4. The S4/T8 teacher-x metric differs by 2.76%.
- The baseline and Dance video contact sheets are visibly very similar at 250 Stage-2 steps. This is not a checkpoint-loading error: their `target_student` files differ, and sampled parameter relative differences are roughly `8e-5` to `6.5e-4` for both online and EMA students.
- The videos therefore support evaluator correctness and the expected NFE ordering, but are not sufficiently separated for a qualitative method claim at this short training budget.

## Decision

The result passes the narrow DanceOPD mechanism gate: on the controlled held-out cache, its video GT endpoint and action endpoint trends are consistently better than shortcut-only at S1/S2/S4, and the scale exceeds the same-checkpoint evaluation repeat variation.

It does not justify a final ablation conclusion. The required next experiment is a clean, same-commit, shared-Stage1 2x2 at a longer Stage-2 budget:

1. shortcut-only;
2. independent endpoint-only through the matched Dance path;
3. DanceOPD velocity-only;
4. independent endpoint plus DanceOPD velocity.

Use the same core2 train and held-out manifests, run seed 0 to 750 Stage-2 steps after the current smoke completes, then add seeds 1 and 2 if the seed-0 trend is not contradicted. Do not add a raw GT objective to Stage 2 for this comparison.
