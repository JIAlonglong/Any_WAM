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

## Action-Protection Changes

### 1. Add the missing DanceOPD action endpoint

The current DanceOPD route jointly rolls the multi-step teacher's video and
action states to the endpoint, but discards the teacher action endpoint and
anchors only video.

Retain `teacher_current_action` as a detached multi-step endpoint. At the same
student-reached joint query state used by the video anchor:

1. request both student video and action velocity;
2. reconstruct the student's one-step action endpoint from the query action
   state and query action sigma;
3. compute masked action endpoint MSE against the multi-step teacher action
   endpoint;
4. add it to the DanceOPD auxiliary with an explicit configurable weight.

Use the existing `opd_danceopd_action_endpoint_weight` configuration name.
Report the loss and weighted contribution through the existing action
transition/endpoint metric channels so a silently inactive action objective is
visible immediately.

### 2. Enable the video-to-action deployment bridge

Enable the existing bridge so the action branch learns under detached
student-generated video and action history. Keep direct normalized-action x0
supervision and its validity mask. The bridge remains separate from the
multi-step-teacher endpoint target.

### 3. Keep the regular action anchors

Retain:

- main action consistency loss;
- GT action regression;
- local action flow-matching regularization;
- action downsample factor 1.

No action clipping or gripper thresholding is added during training or
evaluation; those would hide model failure rather than restore policy quality.

## EMA and Checkpoint Policy

The previous BF16 EMA silently froze all 18 action-specific tensors while the
shared transformer changed. For this urgent run:

- set EMA decay to zero for the whole run;
- target student becomes an exact BF16 copy of the updated online student;
- save and retain both online and target checkpoints;
- assert at checkpoint time that target and online action tensors match;
- report action-branch and shared-branch parameter movement from the Stage-1
  initialization.

FP32 EMA is deferred to a later non-urgent change because it adds material
memory and distributed-checkpoint risk. It is not required for this run.

## Configuration and Launch Isolation

Create a new action-protected config and launcher instead of changing the
meaning of the existing `video_only_opd` experiment.

Initial full-run settings:

- resume: Stage-1 step 2000;
- reset optimizer and global step;
- 8 GPUs;
- maximum 10,000 steps;
- checkpoints every 500 steps, plus early checkpoints at 100 and 250 if the
  existing save path safely supports them;
- teacher endpoint: 8 steps;
- student deployment emphasis: one-step action endpoint;
- action endpoint weight: 4.0;
- bridge weight: 1.0;
- bridge enabled from the beginning;
- EMA decay: 0.0;
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

Add or activate:

- DanceOPD action endpoint loss and weighted contribution;
- action endpoint share of auxiliary loss;
- action-branch gradient norm;
- target/online action-parameter maximum difference;
- number of action tensors changed from Stage-1;
- normalized action out-of-range fraction and gripper saturation on a fixed
  diagnostic batch when available.

Recommend termination to the user, but never terminate automatically, if any
of the following holds:

- non-finite loss or repeated skipped optimizer steps;
- enabled bridge or action endpoint remains identically zero after its first
  scheduled executions;
- target and online differ with EMA decay zero;
- no action-specific tensor has moved by the first checkpoint;
- generated-history action error worsens by more than 20% from its initial
  baseline for three consecutive diagnostic windows;
- checkpoint smoke evaluation remains at zero success after sufficient early
  evaluation episodes while Stage-1 succeeds on the same tasks.

Recommend continuing when the action endpoint and bridge are active and finite,
action tensors move, generated-history action error is stable or improving, and
early closed-loop success is non-zero.

## Verification

Before launch:

- unit-test masked action endpoint reconstruction and weighting;
- verify zero weight exactly preserves the old video-only DanceOPD behavior;
- verify enabled action endpoint produces gradients in the action branch;
- verify the new config cannot silently disable the bridge or action endpoint;
- verify EMA decay zero produces exact target/online action equality;
- run launcher contract checks.

After launch:

- inspect the first logged auxiliary event;
- inspect steps 50, 100, 250, and every 500 thereafter;
- run small K=1 closed-loop evaluation on early checkpoints when resources
  permit;
- perform the final four-suite, 50-episode evaluation only on checkpoints chosen
  by closed-loop evidence.

