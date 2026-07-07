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
