# Cosmos Progressive Stage 2 Plan

## Goal

Train a Cosmos-only LIBERO Stage 2 chain from the fixed Stage 1 checkpoint
`output_libero_cosmos_policy_stage1_cosmos_latent_cdiff_8gpu_20260706_cosmos_latent_s1s2_8gpu/checkpoints/step_5000`.
The chain is intentionally isolated from the LingBot-VA/RoboTwin ablation work
and runs only on physical GPUs 6 and 7.

| Stage | Initialization | Teacher / student endpoint rollout | Maximum steps |
| --- | --- | --- | --- |
| S4 | fixed Stage 1 checkpoint | N=8 / K=4 | 5,000 |
| S2 | selected S4 checkpoint | N=4 / K=2 | 3,000 |
| S1 | selected S2 checkpoint | N=4 / K=1 | 3,000 |

S2 and S1 start only after a fixed-cache checkpoint sweep selects their parent
checkpoint. A later direct Stage1->K1 run is a required control, not part of
the progressive training launch.

## Objective

1. Keep the existing joint Cosmos video-action Stage 1 shortcut loss as the
   main loss.
2. Add a Cosmos-specific independent endpoint loss. From a common `x_t`, the
   frozen Cosmos teacher independently integrates `N` uniform steps to `r` and
   the student integrates `K` steps to the same `r`. Compare denoised endpoint
   estimates `x0 = x_r - sigma_r v(x_r,r)`.
3. Add a Cosmos-specific DanceOPD local velocity loss. A detached 16-step
   joint student rollout from terminal noise supplies a low-noise video state;
   student and Cosmos teacher fields are queried on exactly that state/time.

The endpoint rollout uses the complete deployment trajectory: focused samples
are `1000 -> 0`, whose internal nodes are `1000,750,500,250,0` for S4 and
`1000,500,0` for S2. It does not use a nested K-step rollout inside a local
subinterval.

## Implementation Tasks

1. Add pure, CPU-testable helpers for uniform time paths, independent
   teacher Euler integration, and focused endpoint sampling.
2. Add a new `cosmos_latent_full` OPD dispatch that combines independent
   endpoint and Cosmos DanceOPD losses. Keep the legacy
   `cosmos_latent_student_state` implementation untouched.
3. Extend the target-student safety gate so the new pure Cosmos mode can skip
   EMA/target-student construction. Continue to force FSDP1 whenever OPD and
   activation checkpointing are enabled.
4. Add a progressive Cosmos config whose stage is selected by environment
   variables. Make all step counts, N/K pairs, endpoint focus ratios,
   endpoint/velocity weights, and rollout-gradient suffix explicit.
5. Add a two-GPU runner that pins parent ranks and raw Cosmos workers to
   `CUDA_VISIBLE_DEVICES=6,7`, validates the requested checkpoint layout, and
   writes a stage manifest containing config, git hash, source checkpoint,
   GPU mapping, and fixed-evaluation split definitions.
6. Add deterministic selection/test cache tooling. Selection metrics run every
   250 steps on fixed indices/noise/time pairs; the final test indices remain
   separate and are not used by the sweep. Initially report latent endpoint,
   velocity, rollout drift, action endpoint, and decoded-video diagnostics.
7. Smoke test S4 for two optimizer steps on GPUs 6/7. Verify no process is
   placed on GPUs 0-5, raw workers are rank-local, both endpoint and DanceOPD
   contributions are finite, and a checkpoint saves. Only then launch S4 5k.

## Tests and Acceptance Criteria

- Unit tests first verify independent Euler teacher integration queries each
  newly visited teacher state and ends at the requested `r`.
- Unit tests verify focused pairs are full `1000 -> 0` deployment paths and
  that `cosmos_latent_full` is accepted by the target-student guard.
- Existing Cosmos tests and the new tests must pass under the Any-WAM Python
  environment.
- The two-rank smoke must complete two optimizer updates with finite main,
  endpoint, and local-velocity losses; GPU 0-5 utilization must not change.
- Checkpoint selection is based on three consecutive fixed-cache evaluations
  with less than roughly 1-2% primary-metric improvement and no action/proxy
  regression. The final test cache is evaluated once after selection.

## Risk Controls

- Start S4 with a two-step gradient suffix. Earlier student rollout steps are
  no-grad, avoiding the known full-K FSDP activation-checkpoint OOM. This is
  explicitly recorded in each run manifest.
- Use FSDP1 for OPD + checkpointing to avoid FSDP2 DTensor recomputation
  mixing.
- Teacher endpoint rollout and DanceOPD local rollout are detached; only
  student query/terminal branches retain gradients.
- Do not launch later stages automatically until the parent stage selection
  artifact exists and names a valid checkpoint.
