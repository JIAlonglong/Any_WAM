# Robotwin Stage2 KTO vs Baseline Experiment

Date: 2026-07-03

## Goal

Test whether the opt-in `kto_paopd_norm_focal` Stage2 variant improves over the current Robotwin Stage2 baseline.

Decision: use the current Stage2 baseline. The tested KTO variants did not show a useful improvement.

## Setup

- Project: `/root/nas/junjie/jj/Any_WAM`
- Dataset: `/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500`
- Empty embedding: `/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/empty_emb.pt`
- Teacher: `/root/nas/junjie/jj/Any_WAM/checkpoints/lingbot-va-posttrain-robotwin`
- GPUs: 2 x H100
- Frame setting: Robotwin default, 128 frames at 256x320
- Dataset cache: `CACHE_DATASET_IN_MEMORY=0`
- Stage2 OPD interval: `OPD_AUX_INTERVAL=4`
- Light eval: enabled, interval 12, 1 batch
- Rollout eval: disabled

The repository symlink `training_data/lerobot_robotwin_eef_aug_500` points to a stale `/root/public/...` path, so the real dataset path above was passed explicitly.

## Code Changes Needed For The Experiment

- `9bcd56d feat: add robotwin kto stage2 config`
  - Added `distillation_flowmap.config_robotwin_fullfinetune_stage2_kto_paopd`.
- `8944d3f fix: allow shared empty embedding path`
  - Added `EMPTY_EMB_PATH` override because Robotwin stores `empty_emb.pt` in the dataset parent directory.
- `d0bf1dc fix: expose robotwin stage2 light eval settings`
  - Added Robotwin Stage2 `LIGHT_EVAL_*` config fields. Before this, `ENABLE_LIGHT_EVAL=1` did not actually trigger eval because `light_eval_interval` was missing.

## Runs

Common Stage1 warmup checkpoint:

- Output: `distillation_flowmap/output_robotwin_stage1_smoke_20260703_0900`
- Checkpoint: `checkpoints/step_8`
- Config: `distillation_flowmap.config_robotwin_fullfinetune_stage1_warmup`
- Steps: 8

Stage2 comparisons all resumed from:

`distillation_flowmap/output_robotwin_stage1_smoke_20260703_0900/checkpoints/step_8`

| Run | Config | Extra settings | Output |
| --- | --- | --- | --- |
| Baseline | `config_robotwin_fullfinetune_stage2_anyflow` | default OPD aux | `distillation_flowmap/output_robotwin_stage2_baseline_evalcmp_20260703_0920` |
| KTO | `config_robotwin_fullfinetune_stage2_kto_paopd` | default sampler | `distillation_flowmap/output_robotwin_stage2_kto_evalcmp_20260703_1036` |
| KTO + group-balanced | `config_robotwin_fullfinetune_stage2_kto_paopd` | `STAGE2_SAMPLER=group_balanced`, `STAGE2_GROUP_BY=task` | `distillation_flowmap/output_robotwin_stage2_kto_group_evalcmp_20260703_1041` |

## Light Eval Results

Lower is better.

| Run | Step | `t1000_r0/video_xr_mse` | `t750_r250/video_xr_mse` | `t1000_r0/action_xr_mse` |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 12 | 0.087411 | 0.023475 | 0.001362 |
| Baseline | 24 | 0.084690 | 0.023406 | 0.001358 |
| KTO | 12 | 0.087406 | 0.023477 | 0.001358 |
| KTO | 24 | 0.087399 | 0.023483 | 0.001359 |
| KTO + group-balanced | 12 | 0.087353 | 0.023477 | 0.001360 |
| KTO + group-balanced | 24 | 0.087434 | 0.023455 | 0.001359 |

## Interpretation

The baseline improved clearly from step 12 to step 24 on the main high-noise video metric:

- Baseline `t1000_r0/video_xr_mse`: `0.087411 -> 0.084690`

The KTO variants did not show the same improvement:

- KTO default sampler: `0.087406 -> 0.087399`
- KTO + group-balanced: `0.087353 -> 0.087434`

Action metrics were effectively tied across all runs, around `0.00136` for `t1000_r0/action_xr_mse`.

The group-balanced sampler slightly improved `t750_r250/video_xr_mse` versus KTO default at step 24, but it did not beat baseline on the main `t1000_r0/video_xr_mse` metric.

## Conclusion

Use the current Robotwin Stage2 baseline. This KTO-PAOPD normalized focal reweighting variant is not worth extending as-is.

For future Robotwin work, a better direction is likely sample- or episode-level priority based on teacher-student error or OPD correction magnitude, instead of token-level OPD reweighting inside this KTO variant.

## Verification

- Stage1, baseline, KTO, and KTO + group-balanced final checkpoints were all saved.
- Both H100 GPUs were idle after the runs.
- No training process was left running.
