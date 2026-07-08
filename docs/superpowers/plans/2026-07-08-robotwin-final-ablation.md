# RobotWin Final StepWAM Ablation Plan

## Goal

Run the final StepWAM ablation on LingBot-VA only, using the RobotWin representative subset and a protocol-controlled setup that can produce the paper ablation table and videos without mixing in Cosmos.

## Fixed Protocol

- Teacher: LingBot-VA only.
- Benchmark: RobotWin representative 12-task subset from `robotwin_stepwam_tasks.json`.
- Split: 6 easy tasks + 6 hard tasks through `TASK_PRESET=representative`.
- Student inference steps: `K=4`.
- Stage1: shared shortcut distillation checkpoint per seed, 5000 steps.
- Stage2: OPD fine-tuning per variant, 5000 steps.
- Seeds: `0,1,2`.
- Cache per task: 50 samples, split into 40 train and 10 held-out offline eval samples.
- Closed-loop SR/video eval: run after training with 50 episodes per task where the evaluator supports it.

## Variants

Primary final variants:

1. `full_stepwam`: arbitrary t-to-r shortcut + endpoint OPD + same-state velocity OPD + joint video/action.
2. `w_o_opd`: Stage1 shortcut distillation only.
3. `endpoint_only_opd`: endpoint OPD enabled, velocity OPD disabled.
4. `velocity_only_opd`: same-state velocity OPD enabled, endpoint OPD disabled.
5. `local_adjacent_only`: local/adjacent transition matching only, no arbitrary t-to-r shortcut.
6. `action_only`: action-only/decoupled video-action variant.

Optional `no_video_loss_optional` stays out of the first final batch unless the first six leave enough GPU time.

## Implementation Steps

1. Add a dedicated final launcher:
   - Keep `run_parallel_train.sh` mini/core4 defaults unchanged.
   - Add `run_final_ablation.sh` with final12 defaults.
   - Make the final launcher a thin wrapper around the existing ablation launcher.

2. Harden parallel training:
   - Track active PIDs in each batch.
   - Wait for all PIDs before returning.
   - Record failed jobs instead of exiting early and leaving status ambiguous.
   - Stop before Stage2 if any Stage1 seed fails.

3. Verify launcher behavior:
   - Run pytest coverage for protocol defaults.
   - Run `bash -n` on both shell launchers.
   - Run a dry-run final command to confirm command construction, task preset, variants, seeds, and output root.

4. Launch training:
   - Start Stage1 first in tmux for seeds 0,1,2.
   - Confirm each run writes `run_manifest.json`, `protocol/manifest.json`, logs, and checkpoints.
   - After Stage1 checkpoints exist, start Stage2 in waves of up to 8 one-GPU jobs.

5. Evaluate and summarize:
   - Reuse the held-out cache for all variants.
   - Record endpoint error, same-state velocity error, rollout drift, action endpoint error, video endpoint error, and train-vs-heldout gap.
   - Generate teacher/student videos and contact sheets for representative seeds/tasks.
   - Write a final ablation summary table with mean +/- std and deltas vs `w_o_opd`.

## First Run Root

`distillation_flowmap/output_robotwin_stepwam_ablation/protocol_final12_5000`

## Stop Conditions

- Any Stage1 seed fails: stop before Stage2 and inspect logs.
- More than two Stage2 jobs fail in the first wave: pause Stage2, inspect common config/model errors, then resume.
- Offline eval OOM: rerun eval with no-grad, disabled eval checkpointing, final-action reuse, and smaller eval batch.
