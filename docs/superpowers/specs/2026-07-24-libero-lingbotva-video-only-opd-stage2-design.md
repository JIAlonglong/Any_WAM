# LIBERO LingBotVA Video-Only OPD Stage-2 Design

## Goal

Train a full LIBERO LingBotVA Stage-2 model on one 8×A800 node from the existing
Stage-1 EMA checkpoint, improve 1/2/4-step video rollout quality with video-only
DanceOPD, preserve the Stage-1 action anchor, and evaluate both offline
video-to-action transfer and LIBERO closed-loop success on the same 8-GPU node.

## Starting Point

- Repository worktree:
  `/kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/libero-lingbotva-video-opd`
- Stage-1 checkpoint:
  `/kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_fullft_stage1_warmup/checkpoints/step_2000`
- Resume source: `target_student`, using `RESUME_ONLINE_FROM_TARGET=1`.
- Teacher:
  `/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero`
- Training dataset:
  `/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot`
- Training environment:
  `/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python`

The training dataset contains only the 10 LIBERO-Long tasks. Spatial, Object,
and Goal are therefore cross-suite generalization evaluations rather than
in-distribution training tasks.

The Stage-1 checkpoint is complete and compatible with the current LIBERO
inference path. Its transformer config uses flex attention, action dimension 30,
video SNR shift 5.0, action SNR shift 0.05, and checkpoint step 2000.

## Stage-2 Objective

Stage-2 retains the Stage-1 joint main objective:

- AnyFlow video loss.
- Action FlowMap/distillation loss.
- GT action regression.
- Action-aware/local FM regularization.

DanceOPD is video-only:

- Enable video endpoint loss.
- Enable video same-state velocity loss.
- Disable action endpoint OPD.
- Disable action velocity OPD.
- Disable joint action rollout in the OPD auxiliary path.

The action main-loss forward must consume `student_x_r.detach()` as its video
context whenever video distillation is active. This preserves the intended
mechanism: OPD directly improves video rollout, while the existing action
teacher-forcing objective learns to predict actions from the student-generated
video state.

The run targets all three deployment budgets with a universal curriculum:
1-step, 2-step, and 4-step student rollouts. Dense DanceOPD query scheduling is
explicitly configured so the one-step case does not silently use an invalid
terminal velocity target.

## Training Schedule

- Hardware: one node with 8×A800.
- Full Stage-2 length: 10,000 optimizer steps.
- Save interval: 1,000 steps.
- Resume optimizer: disabled.
- Reset Stage-2 global step to zero.
- Offline Hugging Face mode and expandable CUDA segments enabled.
- TensorBoard enabled.
- W&B offline mode enabled.
- The output directory must be outside the worktree:
  `/kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_lingbotva_stage2_video_only_opd_universal_from_stage1_step2000_steps10000`
- Refuse accidental reuse of a non-empty output directory unless an explicit
  resume flag is supplied.

## Training-Time Diagnostics

Training-time diagnostics must be low-overhead and distributed-safe. They run on
the already available training batch and may not instantiate another teacher or
student model.

Record at least:

- Main video and action losses.
- GT action regression and action local-FM losses.
- Video endpoint OPD loss.
- Video same-state velocity OPD loss.
- `G_anchor` mean/MSE.
- `G_comp` mean/MSE.
- Student video endpoint error.
- Student video field/velocity error.
- OPD sampling budget and query-state statistics.
- Non-finite skip counts by modality and rank.

Diagnostics are logged every 50 optimizer steps to TensorBoard and W&B offline.
Checkpoint saving remains every 1,000 steps.

Heavy teacher/student rollout evaluation does not run inside the training
process because it would compete with FSDP shards for memory and can
asymmetrically stall ranks.

## Checkpoint Offline Evaluation

After a checkpoint is fully written, a separate evaluation command can evaluate
it on the 8-GPU node. It must never overlap with the 8-GPU training process.

For fixed dataset examples and fixed seed 42, evaluate each requested student
budget (1, 2, and 4 steps) under four video conditions:

1. Student rollout video → action error.
2. Teacher rollout video → action error.
3. GT video at the matched noisy/intermediate state → action error.
4. Clean GT video → action error.

Report action error to both:

- Ground-truth actions.
- Teacher action endpoint.

Also report:

- Student video error to GT.
- Student video error to teacher.
- Teacher video error to GT.
- `G_anchor` and `G_comp` diagnostics.
- The teacher-video oracle gain over student video.
- Clean-GT action drift relative to the Stage-1 baseline.

The evaluator must jointly roll out the action state when measuring action
budgets. Reusing a fixed action prediction for all 1/2/4-step rows is invalid.
Results are written as JSON under the selected checkpoint's evaluation output
directory.

## Closed-Loop LIBERO Evaluation

Closed-loop evaluation runs on the same 8×A800 node after training, or after the
training job has released all GPUs.

- Benchmarks:
  - `libero_10` / LIBERO-Long: all 10 tasks.
  - `libero_spatial`: all 10 tasks.
  - `libero_object`: all 10 tasks.
  - `libero_goal`: all 10 tasks.
- Evaluate all 40 tasks. Task-subset evaluation is permitted only for launcher
  smoke tests and may not be reported as a model result.
- One inference server per GPU.
- Eight distinct WebSocket and torch distributed master ports.
- Suite/task pairs are deterministically split across workers; no suite or task
  is omitted or duplicated.
- Each worker runs its own LIBERO client and writes worker-local results.
- A merger validates episode counts and produces per-task and aggregate success
  rates.
- Evaluate 1-step, 2-step, and 4-step video inference budgets.
- Action inference uses the explicitly configured matched deployment contract;
  the launcher prints both video and action step counts before execution.
- Stage-1 step-2000 target is evaluated on all four suites as a baseline using
  the identical server, client, normalization, task initialization, and action
  execution path.

Initial evaluation uses 10 episodes for every one of the 40 tasks to select
checkpoints and budgets. The final selected configuration uses 50 episodes for
every one of the 40 tasks. Reports contain per-task, per-suite macro average,
and overall macro average success rates. Long results are explicitly marked
in-distribution; Spatial, Object, and Goal are marked cross-suite.

## Training/Inference Alignment Checks

Before any GPU process starts, the launcher validates:

- Student, teacher, dataset, empty embedding, and Python paths.
- Checkpoint transformer config and recorded step.
- LIBERO camera keys and 128×128 image size.
- Action dimension 30 with used channels 0–6.
- Quantile normalization values.
- Four actions per generated video frame.
- Video/action SNR shifts.
- CFG scale.
- Attention mode.
- Student rollout budgets.
- Unique ports for every closed-loop worker.

The dry-run mode prints the resolved environment and commands without launching
training or evaluation.

## Acceptance Criteria

The completed tooling is accepted when:

1. Unit tests prove video-only OPD disables every action OPD term while retaining
   the action main loss.
2. A config test proves the action main-loss context is the detached student
   video rollout state.
3. The 8-GPU launcher dry run resolves all paths and produces one torchrun
   command with eight processes.
4. The offline evaluator produces distinct action-rollout rows for 1/2/4 steps
   and all four video conditions.
5. The closed-loop launcher produces eight non-overlapping worker assignments
   covering all 40 suite/task pairs and rejects port collisions.
6. A one-step smoke run loads Stage-1 target, performs forward/backward, records
   finite metrics, and writes a checkpoint.
7. Stage-2 clean-GT action error does not exceed 1.5× the fixed Stage-1 baseline
   during checkpoint evaluation.
8. Video endpoint error improves without a monotonic increase in `G_anchor` or
   `G_comp`.

## Non-Goals

- No Cosmos backend changes.
- No RobotWin changes.
- No action OPD ablation in this run.
- No `libero_90` evaluation unless it is explicitly added as a separate
  90-task generalization study; it is not silently folded into the four-suite
  40-task protocol.
- No concurrent closed-loop evaluation while the training job owns all GPUs.
- No deletion or overwrite of existing checkpoints or experiment records.
