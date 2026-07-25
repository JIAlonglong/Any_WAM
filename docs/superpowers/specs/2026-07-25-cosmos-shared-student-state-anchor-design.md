# Cosmos Shared Student-State Anchor/Field Alignment Design

## Goal

Align Cosmos LIBERO video OPD with the verified
`libero-stage2-deployment-alignment` definition:

- one deployment-faithful student rollout produces a detached, nonterminal
  joint state \(z_r=(z_r^v,z_r^a)\);
- video endpoint/anchor and same-state field losses both depart from exactly
  that \(z_r\);
- the frozen teacher endpoint uses the same terminal prior and condition;
- action remains part of the joint state and conditions every video forward,
  while video anchor/field losses supervise video only.

The existing explicit Cosmos action-OPD arm remains a separate contribution.
It must not be folded into, or reported as, the video endpoint/field loss.

## Reference Contract

The reference command resolves to:

```text
OPD_QUERY_MODE=danceopd
OPD_DANCEOPD_ROLLOUT_STEPS=2,4
OPD_DANCEOPD_ANCHOR_TEACHER_STEPS=8
OPD_DANCEOPD_ENDPOINT_WEIGHT=1.0
OPD_DANCEOPD_VELOCITY_WEIGHT=1.0
OPD_DANCEOPD_ACTION_VELOCITY_WEIGHT=0.0
OPD_AUX_ACTION=0
OPD_JOINT_ACTION_ROLLOUT=0
VIDEO_ACTION_BRIDGE=0
OPD_AUX_INTERVAL=4
```

The Cosmos implementation copies this video-loss geometry, not WanVA-specific
timestep formulas or its action-token layout.

## Student Rollout and Shared Query State

For every scheduled video-OPD update:

1. Draw one video terminal prior and one action terminal prior.
2. Start at each scheduler's exact terminal-noise state.
3. Uniformly select \(K\in\{2,4\}\), synchronized across distributed ranks.
4. Build the exact video/action shifted grids used by deployment inference,
   using the resolved `snr_shift` and `action_snr_shift`.
5. Jointly roll the student video/action states without gradient. Each forward
   receives the current generated video and current generated action as both
   noisy state and self-conditioning state. Dataset GT video/action must not
   replace either generated state.
6. Save post-update states. A legal query must:
   - exclude index zero (terminal prior);
   - exclude index \(K\) (the clean endpoint);
   - remain inside the official Cosmos teacher band
     \([4/5,80/81]\) in normalized video sigma.
7. Sample one legal state per sample with the existing Beta(5,2) semantic-side
   bias. If a configured grid has no legal state, configuration preflight
   fails before training.

The selected state is detached:

\[
z_r=\operatorname{sg}\left[
\operatorname{Compose}_S(x_1;1\rightarrow r,c)
\right],\quad 0<r<1.
\]

The official Cosmos joint query packs \(z_r^a\) into the action carrier frame.
That packing is a representation conversion, not a replacement state. The raw
worker must return a video-frame mask, and training must assert that the
teacher canonicalization does not alter any valid video frame of \(z_r^v\).
The canonical joint state is then used by both student and teacher queries.

## Same-State Field Loss

At the shared joint state, query student and teacher at equal source/target
times:

\[
\mathcal L_{\mathrm{field}}=
\operatorname{MSE}_{\mathrm{valid\ video}}
\left(
F_\theta^v(z_r,r,r,c),
\operatorname{sg}[F_T^v(z_r,r,r,c)]
\right).
\]

The teacher API remains restricted to the calibrated Cosmos band. No
WanVA/LingBotVA timestep equation may be substituted for the Cosmos scheduler.
The action field may be used to construct the rollout state, but it is not part
of this video field MSE.

## Aligned Endpoint Loss

Run one trainable student direct map from the same \(z_r\) to zero:

\[
\hat z_0^v =
z_r^v-\sigma_r F_\theta^v(z_r,r,0,c).
\]

Independently, run the frozen official Cosmos teacher for exactly 8 steps from
the exact same terminal video/action prior and condition:

\[
y_0=\Phi_T^{1,0}(x_1,c).
\]

The raw-worker request/response must carry:

- caller-supplied terminal video prior;
- caller-supplied terminal action prior in native Cosmos packing;
- requested teacher budget `8`;
- returned endpoint video;
- returned effective teacher step count;
- deterministic prior fingerprints echoed by the worker.

The worker must use the supplied prior as the sampler initial state. Matching
only a random seed is insufficient. If the backend cannot honor or prove the
same-prior contract, it raises before the loss is constructed; there is no
fallback to an independently sampled teacher endpoint.

The training loss is elementwise video-latent MSE:

\[
\mathcal L_{\mathrm{endpoint}}=
\operatorname{MSE}
\left(
\hat z_0^v,\operatorname{sg}[y_0^v]
\right).
\]

The selected \(z_r\) and teacher endpoint are detached. Gradients enter only
through the student direct anchor forward.

## Objective Scheduling

The progressive trainer selects exactly one objective for each optimizer
update:

- `step % 4 == 0`: shared-state video anchor + field update;
- the existing explicitly scheduled Cosmos action-OPD phase remains separate;
- all remaining updates use the main AnyFlow objective.

The old independent teacher-anchored video endpoint must not run in parallel
with the shared-state video endpoint. The old full-rollout deployment endpoint
loss is replaced by the aligned direct anchor on video-OPD updates so one update
does not contain two incompatible definitions of `endpoint`.

For the video component:

\[
\mathcal L_{\mathrm{video\ OPD}}=
\mathcal L_{\mathrm{endpoint}}+
\mathcal L_{\mathrm{field}},
\]

with endpoint and field weights both `1.0`. The separate action-OPD
contribution retains its own name, weight, schedule, and diagnostics.

## Training Loss Versus Paper Diagnostics

The implementation must keep these distinct:

- training endpoint loss: elementwise MSE between the student's direct
  \(z_r\rightarrow0\) estimate and the same-prior teacher endpoint;
- `G_anchor`: per-sample squared L2 between a teacher continuation from \(z_r\)
  and the same-prior teacher endpoint;
- `G_comp`: per-sample squared L2 between direct and composed routes;
- training field loss: valid-video elementwise same-state teacher/student field
  MSE.

TensorBoard and W&B offline logs record raw losses, weighted contributions,
ratios, query index/sigma, effective student/teacher steps, valid-frame count,
and same-prior/canonicalization checks.

## Main AnyFlow and Action Semantics

The main AnyFlow mixed objective and its stable teacher-forcing behavior remain
unchanged.

During the shared-state video OPD rollout:

- action is generated jointly and every action prediction reads current
  generated video;
- dataset GT video is never used as the action/video condition;
- no action endpoint or action field term is added to the video OPD loss;
- `VIDEO_ACTION_BRIDGE=0` remains valid and adds no bridge loss.

The `universal-video-action` experiment may still run its explicitly named
action-OPD objective on its separate scheduled update. Video-only variants keep
that objective disabled.

## Failure Handling

- Reject K values with no nonterminal query inside the Cosmos teacher band.
- Reject any teacher query outside \([4/5,80/81]\).
- Reject teacher responses whose effective budget is not exactly 8.
- Reject prior-fingerprint mismatches.
- Reject canonicalization that changes valid video frames.
- Synchronize any non-finite loss or contract failure across ranks before
  backward so all ranks either update or skip/fail together.
- Preserve all existing output directories and checkpoints.

## Verification

Tests are written and observed failing before production changes. They cover:

1. Cosmos K=2/K=4 shifted grids equal inference grids.
2. Query indices exclude prior and endpoint and remain in the teacher band.
3. Endpoint and field receive the same detached \(z_r^v,z_r^a,r\).
4. Changing the student rollout state changes both anchor and field inputs.
5. The action forward reads current generated video, never GT video.
6. Teacher endpoint uses the caller-supplied prior and exactly 8 steps.
7. Prior mismatch and unsupported teacher budgets fail closed.
8. Endpoint is elementwise MSE; `G_anchor` remains squared-L2 diagnostic.
9. Video OPD has no action endpoint/field loss.
10. The separate `universal-video-action` action-OPD schedule remains explicit.
11. Distributed non-finite decisions are synchronized.
12. Training/inference K=2/K=4 grids and joint update formulas agree.

After unit/config tests, run one-GPU one-step smoke evidence for finite field
loss, finite endpoint loss, backward, optimizer step, checkpoint save/restore,
and then an 8-GPU dry-run. No full training starts automatically.
