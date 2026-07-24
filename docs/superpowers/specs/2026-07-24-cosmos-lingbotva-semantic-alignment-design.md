# Cosmos LIBERO Semantic Alignment Design

**Date:** 2026-07-24

**Status:** Approved by the user on 2026-07-24

**Reference:** LingBotVA commit `351ec98c95e1bc2012bdcf9a8248af228e7208ae`

**Target baseline:** Cosmos commit `205550852c2efd912710f1417637a445b38c8cd7`

## Goal

Complete the Cosmos LIBERO training and evaluation path without replacing its
native scheduler, latent, policy, action-packing, or checkpoint semantics.
Backend-neutral safety and experiment-orchestration behavior may be ported
from LingBotVA, while all model calls and transition mathematics remain
Cosmos-native.

## Existing validated foundation

The target already provides:

- explicit `s1`, `s2`, `s4`, and `universal` budgets;
- explicit mixed query grids;
- endpoint-only `s1`;
- separated endpoint and local-field OPD losses;
- raw, weighted, and clamped-ratio OPD metrics;
- joint video/action 1000-to-0 deployment rollouts;
- shared training/inference Euler integration for student K=1/2/4;
- corrected action packing contract `downsample_survivor_v2`;
- atomic checkpoint metadata and a hardened Stage-1/Stage-2 lineage validator;
- strict one-suite, 500-episode result completeness.

These behaviors are preserved rather than reimplemented.

## Immutable training semantics

### Main AnyFlow objective

For every sample, draw two times and order them:

```text
t = max(u1, u2)
r = min(u1, u2)
```

The main objective then selects exactly one branch with probabilities:

- 50%: `r = t`, ordinary diffusion/flow matching;
- 25%: `r = 0`, direct consistency/endpoint target;
- 25%: retain `r < t`, arbitrary-interval AnyFlow mapping.

OPD is auxiliary. It must not replace or narrow the main AnyFlow distribution.
Tests must verify the categorical thresholds, all three branches, and the
preservation of arbitrary nonzero `r < t`.

### Endpoint / anchor objective

The endpoint experiment grid is:

```text
OPD_ROLLOUT_STEP_PAIRS=8,1;8,2;8,4
```

Each pair is sampled uniformly. Teacher and student start from the same
`x_t`. The endpoint time is biased toward the clean region:

```text
sigma_r = (1 - Beta(5, 2)) * 0.25
```

Thus `sigma_r` lies in `[0, 0.25]` and is biased toward zero. At `x_r`, the
clean endpoint estimate is:

```text
x0_hat = x_r - sigma_r * v(x_r, sigma_r)
L_anchor = mse(x0_hat_student, stop_gradient(x0_hat_teacher))
```

The implementation must preserve the Cosmos scheduler conversion between raw
timesteps and sigma. A WanVA timestep formula must not be substituted.

### Compositional / local vector-field objective

The compositional objective supervises the local diffusion/flow vector field,
not robot action velocity and not adjacent-frame optical or pixel motion.

The rollout budget is:

```text
OPD_DANCEOPD_ROLLOUT_STEPS=2,4
```

K=1 is excluded because it has no meaningful intermediate trajectory. A
rollout begins at terminal pure scheduler noise and integrates both the video
and action state under `no_grad`:

```text
K=2: sigma = 1.00 -> 0.50 -> 0.00
K=4: sigma = 1.00 -> 0.75 -> 0.50 -> 0.25 -> 0.00
```

Pre-update states are query candidates:

```text
K=2: 1.00, 0.50
K=4: 1.00, 0.75, 0.50, 0.25
```

Every batch sample independently draws:

```text
q = floor(Beta(5, 2) * K)
```

The index is clamped only for numerical boundary safety. Student and teacher
are queried at the identical joint `(video_state, action_state, sigma)`:

```text
L_velocity = mse(
    v_student(video_q, action_q, sigma_q),
    stop_gradient(v_teacher(video_q, action_q, sigma_q)),
)
```

The default video-only experiments set action velocity OPD weight to exactly
zero. Action OPD is enabled only in a separately named experiment.

## Synchronization boundary

### Backend-independent behavior to synchronize

- all-rank finite decisions before backward and before optimizer update;
- sticky skip state across a gradient-accumulation window;
- synchronized skip counters and branch-origin diagnostics;
- dtype-aware numerical comparison policy;
- diagnostic scheduling, deterministic sample selection, finite sum/count
  reduction, JSONL, TensorBoard, and offline W&B fan-out;
- pure G-anchor/G-comp and video-to-action gain formulas;
- independent output directories, no-overwrite claims, resolved-config
  printing, phase orchestration, and completeness validation;
- four-suite sharding and normalized JSON/CSV aggregation.

### Cosmos-specific behavior

- scheduler raw-time/sigma conversion and terminal expected-state formulas;
- raw teacher latent target and velocity APIs;
- 16-channel latent validation and crop handling;
- joint video/action state construction and `downsample_survivor_v2`;
- action masks and action-head extraction;
- official Cosmos teacher calls and worker lifecycle;
- online/target checkpoint interpretation and lineage metadata;
- shared joint Euler updates used by deployment training and inference.

### LingBotVA/WanVA behavior that must not be copied

- LingBotVA model paths or `/root/nas/...` fallbacks;
- WanVA timestep, endpoint, or denoising algebra;
- LingBotVA token layout, latent geometry, attention masks, or decoder;
- WanVA LoRA inheritance and action-disable configuration;
- fixed 16-step query defaults;
- treating teacher and student transformers as interchangeable checkpoints.

## Safety fixes

### Terminal prior

The pure-noise check remains mandatory. A shared comparator evaluates in
fp32 and returns max error, effective absolute tolerance, relative tolerance,
and effective threshold. Defaults are derived from the state/source dtype
with an fp32 scheduler-arithmetic floor that accepts `1.192e-6`.

Three outcomes are distinguished:

1. within the quiet threshold: continue;
2. within a small warning band: continue and record a warning/diagnostic;
3. materially above threshold: raise.

Both the generic terminal check and the Cosmos raw-window check use the
comparator while retaining their own expected-state formulas.

### Distributed non-finite handling

Every rank first produces local finite flags for main video, main action,
ground-truth/teacher, endpoint OPD, compositional OPD, and optional action
OPD. A collective minimum produces one global decision before any rank can
take a different backward branch.

If any rank is non-finite:

- every rank clears gradients;
- the entire accumulation window is marked skipped;
- no later auxiliary backward is launched in that window;
- optimizer, scheduler, and EMA do not advance;
- synchronized per-origin and total counters are recorded.

Gradient finiteness is synchronized again at the optimizer boundary.

## Mechanism diagnostics

Diagnostics are observation-only and run under `no_grad` after a successful
optimizer update. They do not modify the main teacher-forced action loss.

For the same fixed sample, construct:

- student rollout video context with the common action state;
- teacher rollout video context with the same action state;
- teacher joint video/action context when the Cosmos API supports it;
- GT video context as the reference action-input condition.

All contexts must pass through the actual Cosmos joint input builder and
action head. A value-sensitive test intercepts the model input and proves
that replacing video latent changes the action input. The test also proves
that no stale GT latent is silently substituted. Because diagnostics are
intentionally `no_grad`, they are not required to create training gradients.

Required metrics include:

```text
mechanism/action_error_student_context
mechanism/action_error_teacher_video_context
mechanism/action_error_teacher_joint_context
mechanism/video_to_action_oracle_gain
mechanism/video_to_action_full_joint_gain
mechanism/video_to_action_recoverable_fraction
mechanism/video_to_action_residual_action_gap
mechanism/g_anchor
mechanism/g_anchor_mse
mechanism/g_comp
mechanism/g_comp_mse
mechanism/g_anchor_to_comp_ratio
mechanism/video_endpoint_error
mechanism/video_field_match_error
```

Every denominator is clamped. Finite count, near-zero persistence, explosion,
and branch-dominance health indicators are also logged to TensorBoard and
offline W&B.

## Checkpoint and path contracts

Cosmos-facing entry points require explicit base, teacher, Stage-1, dataset,
and output paths. They do not fall back to LingBotVA or inaccessible user
directories.

Before torchrun:

- `config.json` must exist;
- backend must be `cosmos_policy`;
- Stage-1 online and target weights must be independent and non-symlinked;
- Stage-1 metadata must match the base model and packing contract;
- fresh Stage-2 output must be canonically disjoint and atomically claimed;
- Stage-2 resume must include online/target, optimizer, LR scheduler,
  checkpoint step, and parent-lineage identity.

The existing atomic metadata and lineage implementation is retained and wired
into launchers rather than replaced.

## Experiment family

Independent output directories are required for:

- `s1`;
- `s2`;
- `s4`;
- `universal`;
- `universal-video-action`.

Video-only APM arms hold action OPD at zero:

| Arm | Endpoint | Compositional | Action OPD |
|---|---:|---:|---:|
| `stage1_only` | 0 | 0 | 0 |
| `anchor_only` | 1 | 0 | 0 |
| `field_only` | 0 | 1 | 0 |
| `apm` / `full` | 1 | 1 | 0 |

Each launcher uses eight ranks, supports steps/ports/output-root/run-tag,
prints the resolved configuration, enables TensorBoard and offline W&B,
forces HuggingFace/Transformers offline mode, defaults checkpoint saving to
1,000 steps, supports a smoke override, performs a write-free dry-run, and
refuses to overwrite an existing output directory.

## Training and inference parity

The same joint Euler integrator must be exercised for K=1,2,4 in deployment
training and inference. A direct parity test compares state-by-state video
and action trajectories.

Inference explicitly selects a model role:

- official Cosmos teacher;
- corrected Stage-1 target;
- Stage-2 online student;
- Stage-2 target student.

Checkpoint metadata supplies backend, base identity, packing schema, supported
budgets, scheduler endpoints, and parent lineage. No inference entry point
may depend on a path from another machine.

For the official teacher, “matched K” means the official Cosmos scheduler
actually executes K state transitions. Preflight fails if the backend cannot
honor K=1,2,4; a fixed 4- or 8-step result is never relabelled as matched K.

## Evaluation protocol

The formal matrix covers:

- `libero_10`;
- `libero_spatial`;
- `libero_object`;
- `libero_goal`.

Every model role is evaluated with matched video/action budgets 1/1, 2/2,
and 4/4. Formal closed-loop evaluation uses 50 episodes for each of 40 tasks,
or 2,000 unique `(suite, task, seed)` records per model/budget.

Completeness is fail-closed:

- every expected key appears exactly once;
- every task has the requested episode count;
- checkpoint identity and K match the request;
- missing, duplicate, unexpected, or partial results fail aggregation.

Artifacts include episode/task CSV files, offline metrics CSV, mechanism CSV,
and a versioned JSON summary with protocol and lineage attestations.

## Verification gates

Verification runs in this order:

1. unit tests;
2. config import;
3. single-GPU one-step training;
4. main backward evidence;
5. Cosmos video OPD backward evidence;
6. optimizer-step evidence;
7. checkpoint save and resume;
8. single-GPU inference-service load;
9. one LIBERO task, one episode closed-loop smoke;
10. eight-GPU `CHECK_ONLY`/dry-run;
11. formal launch commands.

A smoke run passes only when logs explicitly identify the Cosmos base and
Stage-1 paths, sample count, experiment configuration, finite main/OPD loss,
successful optimizer step, checkpoint save, inference restore, and requested
video/action K.

Resource-gated gates remain explicitly unverified until an actual GPU and
simulator allocation executes them.

