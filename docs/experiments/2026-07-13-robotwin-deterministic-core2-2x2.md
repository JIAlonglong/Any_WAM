# RobotWin Deterministic Core2 2x2 Mini-Ablation

## Scope

- Teacher: LingBot-VA.
- Tasks: `open_microwave`, `place_a2b_right`.
- Shared Stage1 checkpoint: 5,000-step representative Stage1 seed 0.
- Stage2 train indices: `0-19` per task.
- Primary held-out indices: `40-49` per task. These are unseen by the shared
  Stage1 checkpoint, not merely unseen by Stage2.
- Stage2: 750 steps, seeds `0,1,2`.
- Fixed evaluation pairs: `t1000->0`, `t1000->500`, `t750->250`.
- Reported aggregate: macro-average over the three pairs, then mean +/- sample
  standard deviation over the three Stage2 seeds.

Variants:

1. `calib_w_o_opd`: Stage1 shortcut continuation only.
2. `calib_endpoint_only_danceopd_i1`: endpoint-only OPD.
3. `calib_danceopd_i1`: DanceOPD local velocity only.
4. `calib_endpoint_danceopd_i1`: endpoint plus DanceOPD local velocity.

## Deterministic Evaluation Correction

Commit `38224e7` fixes an offline-evaluation bug discovered during this run.
`_prepare_base_dict` applied `drop_text_ratio=0.1` during evaluation, so two
independently built teacher caches could use different text conditioning for
the same manifest index and pair. The correction:

- forces text dropout to zero in numeric and video evaluators;
- sets dataset-level CFG drop probability to zero for offline evaluation;
- records `offline_eval_conditioning_v1` in teacher-cache metadata; and
- rejects and rebuilds caches without that schema.

The initial seed-0 metrics were regenerated after this fix. A cross-GPU,
cross-Stage2-source cache smoke test produced byte-identical teacher endpoint
and velocity targets. The fixed teacher-to-GT video MSE is identical across
all variants and seeds (`0.0516814304` for the s4/t4 macro aggregate), which
is the required cache-consistency check.

## Held-Out Results

Lower is better.

| Metric | Baseline | Endpoint | DanceOPD | Full |
| --- | --- | --- | --- | --- |
| s1/t1 video GT x MSE | 0.097083 +/- 0.000832 | 0.093263 (-3.93%) | 0.089376 (-7.94%) | 0.088647 (-8.69%) |
| s2/t2 video GT x MSE | 0.067598 +/- 0.000759 | 0.064323 (-4.84%) | 0.061247 (-9.40%) | 0.060463 (-10.55%) |
| s4/t4 video GT x MSE | 0.060495 +/- 0.000807 | 0.056677 (-6.31%) | 0.054295 (-10.25%) | 0.053414 (-11.71%) |
| s4/t4 student-teacher endpoint x MSE | 0.013817 +/- 0.000204 | 0.012182 (-11.83%) | 0.011748 (-14.97%) | 0.011329 (-18.00%) |
| s4/t4 student-teacher velocity MSE | 0.113556 +/- 0.000468 | 0.106686 (-6.05%) | 0.106945 (-5.82%) | 0.105042 (-7.50%) |
| s4/t4 action endpoint GT MSE | 0.000336936 +/- 0.00000141 | 0.000329904 (-2.09%) | 0.000327911 (-2.68%) | 0.000325608 (-3.36%) |
| s4/t8 student-teacher endpoint x MSE | 0.015646 +/- 0.000369 | 0.013714 (-12.35%) | 0.013059 (-16.54%) | 0.012573 (-19.64%) |

Per-task s4/t4 video GT x MSE:

| Task | Baseline | Endpoint | DanceOPD | Full |
| --- | --- | --- | --- | --- |
| `open_microwave` | 0.036795 | 0.035723 (-2.91%) | 0.035356 (-3.91%) | 0.035089 (-4.64%) |
| `place_a2b_right` | 0.084195 | 0.077631 (-7.80%) | 0.073234 (-13.02%) | 0.071739 (-14.79%) |

The s4/t4 train-minus-held-out video-GT gap is also smallest for the full
variant: baseline `0.009226`, endpoint `0.007677`, DanceOPD `0.007610`, full
`0.007240`.

## Interpretation

This is a robust mini-ablation trend, not a final paper conclusion. Endpoint
and DanceOPD each improve the held-out video endpoint; the full objective is
best on the three equal-NFE video points, student-teacher endpoint/velocity
metrics, and action endpoint metrics. Full beats endpoint-only on all three
seeds for s4/t4 video GT MSE. Its margin over DanceOPD-only is smaller, so it
must be tested on more tasks before claiming a general interaction effect.

This experiment does not include closed-loop RobotWin success rate, a
10-12-task representative subset, or cache-backed qualitative video metrics.
Those remain required for the final ablation result.

## Artifacts

- Per-seed roots:
  `output_robotwin_stepwam_ablation/protocol_dance_endpoint_2x2_seed{0,1,2}_v1`.
- Combined summary:
  `output_robotwin_stepwam_ablation/protocol_dance_endpoint_2x2_three_seed_v1/summary_final`.
- Deterministic cache smoke:
  `output_robotwin_stepwam_ablation/deterministic_eval_cache_smoke`.
