# LIBERO Stage-2 Deployment Alignment Design

## Goal

Make LingBotVA LIBERO Stage-2 video-only OPD train on the same 1/2/4-step
trajectory states used by closed-loop deployment, and make the action branch
consume detached generated-video context so video improvements can affect action
predictions without introducing action OPD.

## Scope

This change covers four behaviors:

1. Build DanceOPD rollout paths with each branch's scheduler shift, matching
   `flowmap_inference` for video `snr_shift=5.0` and action
   `action_snr_shift=0.05`.
2. Record post-update states, exclude the initial pure-noise state from query
   sampling, and permit the final endpoint to be queried.
3. Extend mechanism diagnostics with deployment-faithful context routes where
   both the noisy video stream and condition-video stream are replaced by the
   selected student, teacher, or GT video.
4. Add a Stage-2-only action bridge: run the existing action target/GT loss with
   detached generated video as its video condition. The bridge does not add an
   action OPD target and cannot backpropagate through the generated video.

The main AnyFlow mixed `(t,r)` objective, Stage-1 behavior, teacher definitions,
checkpoint format, and closed-loop action execution protocol remain unchanged.

## Architecture

### Shifted rollout grids

A model-agnostic helper converts a raw linear sigma grid into branch-specific
shifted timesteps using the existing `FlowMatchScheduler.apply_shift` contract.
DanceOPD uses this helper for video and action paths. Tests compare the exact
K=1/2/4 grids against `flowmap_inference`.

### Post-update semantic queries

The rollout stores the initial state for integration but exposes only the
`K` post-update states to the query sampler. For K=1 the only legal query is the
endpoint. For K=2/4, Beta sampling selects among post-update states and includes
the final endpoint.

### Deployment-faithful diagnostic routes

Joint-input construction accepts explicit condition-video and condition-action
states instead of implicitly reusing clean GT tensors. Diagnostics report:

- GT condition video to action;
- student rollout condition video to action;
- teacher rollout condition video to action;
- teacher video plus teacher action condition to action.

Existing metric names remain available where their semantics are unchanged.
New route metadata makes condition provenance explicit.

### Detached video-to-action bridge

The bridge is active only in the LIBERO Stage-2 video-only OPD config. It takes
the selected student rollout video, detaches it, places it in the condition-video
stream, and computes the existing action teacher/GT objective. The initial
schedule is configurable and defaults to a conservative probability curriculum:

- steps 0-499: 0.25;
- steps 500-1499: 0.50;
- step 1500 onward: 0.75.

The action condition stream remains the existing GT condition in this change so
the experiment isolates the effect of video context. Action self-conditioning
is a separate ablation.

## Error Handling

- Reject non-positive rollout counts and non-finite/non-positive scheduler
  shifts.
- Assert that every semantic query index is in `[1, K]`.
- Assert that generated video is detached before the action bridge forward.
- Preserve distributed all-rank finite checks already used by DanceOPD.

## Verification

Tests are written before production changes and must demonstrate:

1. shifted K=1/2/4 grids equal inference grids;
2. K=1 queries the endpoint and no query can select pure noise;
3. diagnostic routes replace both noisy and condition video consistently;
4. bridge action loss has action/model gradients but no generated-video
   gradient;
5. `OPD_AUX_ACTION=0` remains enforced;
6. existing DanceOPD and mechanism-diagnostic suites remain green.

A one-step single-GPU smoke test must complete main loss, DanceOPD, diagnostics,
backward, and checkpoint save before an 8-GPU launch is prepared.
