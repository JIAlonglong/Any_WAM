# Action-Protected LIBERO Stage-2 Full-Training Design

Date: 2026-07-26

## Objective

Maximize closed-loop LIBERO task success. Video-only metrics are diagnostic and
must not override closed-loop action performance when selecting checkpoints.

## Starting Point

- Resume from the clean Stage-1 checkpoint at step 2000.
- Do not resume from the broken Stage-2 `target_student` at step 10000.
- Preserve the existing aligned-anchor/field DanceOPD video path.
- Run one full eight-GPU Stage-2 job, with early monitoring rather than a
  separate short-training branch.

## Scope

This change isolates one hypothesis: the existing action supervision failed
because BF16 EMA silently froze the action side of `target_student`.

- Keep the existing action consistency, GT regression, and local action
  flow-matching losses unchanged.
- Do not add a DanceOPD action endpoint.
- Do not enable action OPD or joint action OPD rollout.
- Do not change the weights of the existing action losses.
- Keep the existing video DanceOPD path unchanged.

The run may enable the already implemented video-to-action bridge, but bridge
configuration is not part of the EMA fix and must be reported independently.

## EMA Repair

Maintain a persistent FP32 EMA master state for the action-specific target
parameters while retaining BF16 target parameters for model forward and
checkpoint compatibility.

For every selected target/source parameter pair:

1. initialize the FP32 master from the target parameter;
2. update `master = decay * master + (1 - decay) * source.float()`;
3. copy `master` back to the BF16 target parameter;
4. retain the FP32 master across optimizer steps.

Continue using the existing BF16 EMA path for non-action parameters in this
urgent change. Shared parameters already moved in the failed run; the observed
silent freeze was isolated to all 18 action-specific tensors.

The FP32 action EMA state must:

- be sharded/local in the same way as the FSDP parameters;
- validate target/source parameter name, shape, and dtype compatibility;
- be saved with each checkpoint;
- be restored when resuming a repaired checkpoint;
- fail explicitly if a repaired checkpoint is missing or incompatible rather
  than silently restarting the EMA history;
- expose the maximum online/EMA action difference and the number of action
  target tensors that moved from initialization.

## Configuration and Launch Isolation

Create a new action-protected config and launcher instead of changing the
meaning of the existing `video_only_opd` experiment.

Before the full run, execute a 50–100 optimizer-step eight-GPU gradient probe
from Stage-1 step 2000 with the exact full-run loss configuration. The probe is
not a model-selection experiment and does not enable any new action loss.

Initial full-run settings:

- resume: Stage-1 step 2000;
- reset optimizer and global step;
- 8 GPUs;
- maximum 10,000 steps;
- checkpoints every 500 steps;
- teacher endpoint: 8 steps;
- no DanceOPD action endpoint;
- action OPD disabled;
- joint action OPD rollout disabled;
- EMA decay: 0.99 with FP32 action accumulation;
- rollout and light diagnostics enabled.

The final checkpoint is selected by closed-loop success, not by training step.

## Monitoring and Continue/Stop Signals

Monitor existing metrics:

- `loss/action_consistency`;
- `loss/gt_regression`;
- `loss/action_local_fm`;
- `loss/video_action_bridge`;
- `mechanism/action_error_student_generated_history_context`;
- `mechanism/action_error_teacher_joint_context`;
- skipped steps, gradient norm, and learning rate.

Add:

- action-branch gradient norm;
- video-branch and shared-branch gradient norms;
- ratios `action/shared` and `action/video`;
- target/online action-parameter maximum difference;
- number of action tensors changed from Stage-1;
- normalized action out-of-range fraction and gripper saturation on a fixed
  diagnostic batch when available.

Recommend termination to the user, but never terminate automatically, if any
of the following holds:

- non-finite loss or repeated skipped optimizer steps;
- an enabled existing action loss remains identically zero;
- FP32 action EMA state is missing, incompatible, or non-finite;
- no action-specific tensor has moved by the first checkpoint;
- generated-history action error worsens by more than 20% from its initial
  baseline for three consecutive diagnostic windows;
- checkpoint smoke evaluation remains at zero success after sufficient early
  evaluation episodes while Stage-1 succeeds on the same tasks.

Recommend continuing when existing action losses and gradients are finite,
action EMA tensors move, generated-history action error is stable or improving,
and early closed-loop success is non-zero.

For the short gradient probe:

- log every optimizer step for steps 1–10 and every 10 steps afterwards;
- report median and p90 action, video, and shared gradient norms;
- report the fraction of optimizer steps with non-zero action gradients;
- report action/shared and action/video gradient ratios;
- report per-checkpoint online action parameter movement and FP32 EMA movement;
- treat an all-zero action gradient, non-finite branch norm, unchanged online
  action branch, or unchanged FP32 action EMA as a launch blocker;
- do not impose a universal numeric gradient-ratio threshold before observing
  the probe, because parameter counts and branch scales differ substantially.

## Verification

Before launch:

- reproduce that repeated BF16 EMA loses a representative small action update;
- verify persistent FP32 action EMA accumulates the same update;
- verify non-action parameters retain the existing EMA behavior;
- verify action EMA state checkpoint save and restore;
- verify incompatible or missing repaired EMA state fails explicitly on resume;
- verify the video-only DanceOPD action losses remain disabled;
- run launcher contract checks.

After launch:

- inspect the first logged auxiliary event;
- inspect steps 50, 100, 250, and every 500 thereafter;
- run small K=1 closed-loop evaluation on early checkpoints when resources
  permit;
- perform the final four-suite, 50-episode evaluation only on checkpoints chosen
  by closed-loop evidence.
