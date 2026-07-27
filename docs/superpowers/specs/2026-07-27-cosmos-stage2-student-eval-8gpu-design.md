# Cosmos Stage-2 to Student Full40 Eight-GPU Chain

Date: 2026-07-27
Status: Approved design

## Goal

Provide one independent, reproducible eight-A800 chain that starts from the
existing aligned Cosmos LIBERO Stage-1 `step_3000`, trains Stage-2 for 5000
steps, and evaluates only the Stage-2 target student on all four LIBERO suites
at matched video/action budgets K=1/2/4.

The official teacher matrix is already running separately and must not be
repeated by this chain.

## User interface

Add:

```text
distillation_flowmap/run_cosmos_stage2_student_eval_8gpu.sh
```

Supported arguments:

```text
--phase all|stage2|eval|check
--stage1-root PATH
--stage1-step N
--stage2-steps N
--save-interval N
--episodes N
--master-port PORT
--output-root PATH
--run-tag TAG
--dry-run
--check-only
```

Production defaults:

```text
stage1-step=3000
stage2-steps=5000
save-interval=1000
episodes=50
master-port=29672
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

The wrapper must fail immediately if the Stage-1 checkpoint is not fully
materialized. It must not hold an eight-GPU allocation while waiting for
Stage-1.

## Stage-1 contract and lineage

The parent is:

```text
<stage1-root>/checkpoints/step_3000
```

Both `online_student` and `target_student` checkpoint structure, transformer
configuration, backend identity, checkpoint metadata, and canonical lineage
must be validated before Stage-2 starts. Stage-2 initializes from the validated
Stage-1 target student.

Existing Stage-2 launchers currently hard-code `expected_step=5000`. Replace
that assumption with an explicit, positive
`COSMOS_STAGE1_EXPECTED_STEP`. Preserve `5000` as the default for all existing
callers; the new chain pins it to `3000`. The expected step must be included in
resolved configuration and lineage validation so an incorrectly labelled
checkpoint cannot pass.

## Stage-2 training

Use the reviewed `universal-video-action` arm and all eight visible GPUs:

```text
MAX_TRAIN_STEPS=5000
SAVE_INTERVAL=1000
TRAIN_SEED=42
ALIGNED_VIDEO_OPD_INTERVAL=4
OPD_AUX_INTERVAL=4
OPD_ROLLOUT_STEP_PAIRS=8,1;8,2;8,4
OPD_DANCEOPD_ROLLOUT_STEPS=2,4
OPD_DANCEOPD_ANCHOR_TEACHER_STEPS=8
OPD_DANCEOPD_ENDPOINT_WEIGHT=1.0
OPD_DANCEOPD_VELOCITY_WEIGHT=1.0
OPD_DANCEOPD_ACTION_ENDPOINT_WEIGHT=1.0
ACTION_DOWNSAMPLE_FACTOR=4
VIDEO_ACTION_BRIDGE=0
```

The training output is isolated beneath:

```text
<output-root>/<run-tag>/universal-video-action
```

Fresh runs must reject any existing Stage-2 output directory. Exact Stage-2
resume may be supported by the underlying launcher, but the initial independent
chain must never infer a resume checkpoint.

All existing non-finite synchronization, optimizer compatibility, FSDP/EMA
cache invalidation, terminal-prior tolerance, shared student-state endpoint,
same-state field query, and mechanism diagnostics remain enabled.

## Provenance assets

Before Stage-2, prepare or strictly validate:

- dataset lock;
- official Cosmos teacher lock;
- Cosmos video VAE lock;
- local Cosmos model lock;
- Stage-1 target lock and contract identity;
- canonical Stage-2 lineage JSON;
- the all-40 Wan prompt embedding table.

Dry-run and check-only modes must not create these files. A live run may create
them only under the new run root and must not overwrite an existing conflicting
asset.

## Student-only evaluation

After Stage-2, evaluate:

```text
checkpoint=<stage2 output>/checkpoints/step_5000/target_student/transformer
role=stage2_target
suites=libero_10,libero_spatial,libero_object,libero_goal
video/action budgets=1/1,2/2,4/4
episodes per task=50
```

Set `S4_MATRIX_ROLES=stage2_target`; the official teacher role must never be
launched.

Use all eight GPUs as four paired shards:

```text
shard 0: student GPU 0, Cosmos worker GPU 1
shard 1: student GPU 2, Cosmos worker GPU 3
shard 2: student GPU 4, Cosmos worker GPU 5
shard 3: student GPU 6, Cosmos worker GPU 7
```

This gives four concurrent task ranges while keeping the student and the
official Cosmos raw-action worker on separate GPUs. Default representative
video retention is seed 0 only. JSON records remain complete for all episodes.

The strict matrix merger must require 40 unique tasks and 120 task-budget cells,
reject missing or duplicate cells, and write student-only `matrix_summary.json`
and `matrix_summary.csv`.

## Phase and failure behavior

- `--phase stage2`: prepare/validate provenance and train only.
- `--phase eval`: require the exact completed Stage-2 target checkpoint and
  evaluate only the student.
- `--phase all`: run Stage-2, then student evaluation in the same allocation.
- `--phase check`, `--check-only`, or `--dry-run`: validate and print the
  complete resolved plan without creating output.

Any child failure stops the chain. Signals are forwarded to the active child.
Completed Stage-2 checkpoints remain recoverable; no training output or
evaluation result is deleted.

## Verification

Tests must cover:

- Stage-1 step 3000 accepted only when its metadata agrees;
- existing callers still default to Stage-1 step 5000;
- wrong or missing parent step rejected;
- Stage-2 resolved config pins 5000 steps and the aligned OPD values above;
- student-only evaluation never includes `official_teacher`;
- four suites and K=1/2/4 are all planned;
- four paired shards use all eight GPU ordinals exactly once;
- representative video seed is 0;
- no-overwrite and read-only modes;
- Stage-2 target checkpoint path passed consistently to evaluation;
- final strict student full40 merge.

No live training or LIBERO rollout is started during implementation
verification.
