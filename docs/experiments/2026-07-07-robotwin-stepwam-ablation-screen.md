# RobotWin StepWAM Ablation Screen - 2026-07-07

Scope:
- Teacher: LingBot-VA only.
- Dataset/task: RoboTwin `place_a2b_right`, `DATASET_MAX_SAMPLES_PER_TASK=50`.
- Stage 1 checkpoint: `distillation_flowmap/output_robotwin_stepwam_ablation/screen_place_a2b/full_stepwam/seed_0/stage1/checkpoints/step_100`.
- Stage 2 checkpoints: `distillation_flowmap/output_robotwin_stepwam_ablation/screen_place_a2b_2gpu/{variant}/seed_0/stage2/checkpoints/step_50`.
- Student steps: K=4.
- Teacher rollout: 4 steps.
- Eval pairs: `1000->0`, `750->250`.
- Eval batches: 1.
- Teacher cache: `distillation_flowmap/output_robotwin_stepwam_ablation/screen_place_a2b_2gpu/shared_teacher_cache/place_a2b_1batch_2pairs_teacher4.pt`.

Eval fix notes:
- Direct rollout eval OOMed with teacher and student resident together.
- The eval path now supports CPU teacher target cache and cache replay without loading teacher.
- Student-only eval still needs 2-GPU FSDP1; single GPU falls back to NO_SHARD and OOMs.
- Offline eval forces activation checkpointing for student forward and detaches outputs immediately; this is an eval memory path only.

Quick metrics:

| Variant | t1000 video teacher x MSE | t1000 video teacher v MSE | t1000 action GT xr MSE | t750 video teacher x MSE | t750 video teacher v MSE | t750 action GT xr MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_stepwam | 0.0403161682 | 0.152493358 | 0.000196231631 | 0.00548730511 | 0.0583023205 | 0.0000533227612 |
| w_o_opd | 0.0405469574 | 0.153036013 | 0.00019719408 | 0.00548235327 | 0.0582626499 | 0.0000529370991 |
| endpoint_only_opd | 0.0404824242 | 0.152698621 | 0.000196916721 | 0.00548956078 | 0.05830466 | 0.0000529886129 |
| velocity_only_opd | 0.0405604951 | 0.152836114 | 0.000197286339 | 0.00548018143 | 0.0582484007 | 0.0000534313986 |

Metric JSONs:
- `distillation_flowmap/output_robotwin_stepwam_ablation/screen_place_a2b_2gpu/full_stepwam/seed_0/eval/rollout_metrics_quick_cache_2gpu.json`
- `distillation_flowmap/output_robotwin_stepwam_ablation/screen_place_a2b_2gpu/w_o_opd/seed_0/eval/rollout_metrics_quick_cache_2gpu.json`
- `distillation_flowmap/output_robotwin_stepwam_ablation/screen_place_a2b_2gpu/endpoint_only_opd/seed_0/eval/rollout_metrics_quick_cache_2gpu.json`
- `distillation_flowmap/output_robotwin_stepwam_ablation/screen_place_a2b_2gpu/velocity_only_opd/seed_0/eval/rollout_metrics_quick_cache_2gpu.json`

Interpretation:
- On this very short 50-step screen, full StepWAM is slightly better than w/o OPD on `1000->0` teacher endpoint/velocity metrics.
- The margin is small and `750->250` is mixed, so this only validates that the ablation/eval pipeline runs and produces comparable metrics.
- Do not treat this as final evidence for the paper; use it to decide the next longer 10-12 task RoboTwin ablation run.

## 500-step Stage-2 screen

Setup:
- Same Stage-1 step-100 checkpoint family and same 1-task / 50-sample `place_a2b_right` screen.
- Stage 2 was continued to `step_500` for `full_stepwam`, `w_o_opd`, `endpoint_only_opd`, and `velocity_only_opd`.
- Eval used the same shared teacher cache and 2-GPU cached rollout path.

| Variant | t1000 video teacher x MSE | t1000 video teacher v MSE | t1000 action GT xr MSE | t750 video teacher x MSE | t750 video teacher v MSE | t750 action GT xr MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_stepwam | 0.0337463692 | 0.142175451 | 0.000194018066 | 0.00530336658 | 0.0560740642 | 0.0000541474838 |
| w_o_opd | 0.0334075615 | 0.141652673 | 0.000200641822 | 0.00532054296 | 0.0558490604 | 0.0000516249784 |
| endpoint_only_opd | 0.0337613225 | 0.141307473 | 0.000195253058 | 0.00529895211 | 0.0554515384 | 0.0000531455953 |
| velocity_only_opd | 0.0329119377 | 0.140769482 | 0.000201461924 | 0.00527633401 | 0.0558953732 | 0.0000524408388 |

Relative to `w_o_opd` (negative is better):

| Variant | t1000 x | t1000 v | t1000 action | t750 x | t750 v | t750 action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_stepwam | +1.014% | +0.369% | -3.301% | -0.323% | +0.403% | +4.886% |
| endpoint_only_opd | +1.059% | -0.244% | -2.686% | -0.406% | -0.712% | +2.946% |
| velocity_only_opd | -1.484% | -0.623% | +0.409% | -0.831% | +0.083% | +1.580% |

Interpretation:
- The 500-step 1-task screen does not show a clean Full > baseline trend.
- Full improves the `t1000->0` action GT endpoint metric, but does not beat `w_o_opd` on video teacher x/v metrics.
- `velocity_only_opd` is best on the two `t1000->0` video teacher metrics and on `t750->250` video teacher x MSE in this screen.
- This should not be used as a final method conclusion because it is a single-task, train-like screen. It suggests the next useful run should be a held-out 4-6 task mini-ablation with a shared Stage-1 checkpoint before spending full 10-12 task / multi-seed budget.
