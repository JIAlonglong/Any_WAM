# RobotWin DanceOPD + Endpoint Seed-0 Gate

## Scope

This is a controlled, single-seed trend check, not a final ablation claim.

- Teacher: LingBot-VA.
- Tasks: `place_a2b_right`, `open_microwave`.
- Shared initialization: the same Stage1 `step_5000` checkpoint for all four
  Stage2 variants.
- Stage2 train split: indices `0-19` per task.
- Primary held-out split: indices `40-49` per task. These are unseen by the
  shared Stage1 checkpoint, whose relevant representative-protocol train
  indices are `0-39`.
- The former `20-29` split is retained only as `stage2_heldout` diagnostic;
  it is not used for the main result.
- Fixed pairs: `t1000->0`, `t1000->500`, `t750->250`.
- Stage2: 750 steps, one H100 per variant, same Stage1 checkpoint, same train
  manifest and protocol seed.

The held-out numerical evaluator built one teacher cache with 60 targets
(20 samples x 3 pairs) and reused it for every variant.

## Variants

1. `calib_w_o_opd`: shortcut baseline with no OPD auxiliary.
2. `calib_endpoint_only_danceopd_i1`: independent teacher endpoint only.
3. `calib_danceopd_i1`: DanceOPD local velocity only.
4. `calib_endpoint_danceopd_i1`: endpoint plus DanceOPD local velocity.

## Held-Out Equal-NFE Trend

Values below average the three fixed pairs. All listed errors are lower-is-
better. The percentage is relative to `calib_w_o_opd`.

| Student/teacher NFE | Endpoint-only video-GT MSE | Dance-only video-GT MSE | Full video-GT MSE |
| --- | ---: | ---: | ---: |
| S1/T1 | -3.16% | -7.14% | -8.67% |
| S2/T2 | -4.40% | -7.78% | -10.94% |
| S4/T4 | -5.51% | -6.93% | -11.48% |

At S4/T4, the raw video-GT MSE values were baseline `0.0574712`, endpoint-only
`0.0543066`, Dance-only `0.0534907`, and full `0.0508736`.

- Both tasks improved under the full variant: `open_microwave` -16.00% and
  `place_a2b_right` -9.49% video-GT MSE at S4/T4.
- S4/T4 same-state velocity MSE improved by 0.51% (endpoint-only), 0.88%
  (Dance-only), and 1.14% (full).
- S4/T8 teacher-retention endpoint error improved by 0.57% (endpoint-only),
  0.47% (Dance-only), and 0.42% (full).
- The direct student-teacher endpoint MSE change was small (within 0.52% at
  S4/T4); the larger gain appears in the GT video endpoint metric.
- Action endpoint-GT MSE at S4/T4 favored Dance-only: baseline `0.000337600`,
  endpoint-only `0.000330904` (-1.98%), Dance-only `0.000322621` (-4.44%),
  full `0.000327724` (-2.93%). This supports different roles for endpoint and
  local-velocity supervision rather than a single uniformly best component.

## Training Scale Check

Across the 750 logged updates:

| Variant | Median OPD | Median endpoint | Median local velocity | Median total grad norm |
| --- | ---: | ---: | ---: | ---: |
| Endpoint-only | 0.00920 | 0.00919 | 0.02260 raw, weight 0 | 0.51 |
| Dance-only | 0.01700 | 0 | 0.01700 | 1.83 |
| Full | 0.02980 | 0.00982 | 0.01680 | 1.98 |

For the full variant, endpoint contribution to the effective OPD term had
median fraction 0.384 and mean fraction 0.415. Losses and total gradient norms
were finite throughout. Per-branch gradient norms were enabled in config but
not persisted because neither TensorBoard nor offline WandB history was
available; this must be added before reporting branch-gradient analysis.

## Qualitative Asset Caveat

The numerical rollout evaluator is protocol-correct because all variants use
the same cached teacher targets. The current video evaluator independently
reruns the teacher after student rollouts. Its teacher clips were not identical
between all variants, despite equal GT/noise inputs. Therefore seed-0 contact
sheets are diagnostic assets only, not evidence for a qualitative comparison.
A cache-backed video rendering path should be added and seed-0 videos should be
regenerated from the completed checkpoints.

## Next Gate

Seed1 and seed2 repeat the same matrix on eight H100s: only `TRAIN_SEED`
changes, while shared Stage1, train manifest, held-out manifest, and pair
seeds stay fixed. The final conclusion requires their mean and standard
deviation; this seed-0 result is a positive trend only.
