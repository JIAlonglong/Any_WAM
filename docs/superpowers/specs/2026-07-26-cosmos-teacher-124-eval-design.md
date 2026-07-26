# Cosmos Official Teacher K=1/2/4 Evaluation Wrapper

## Goal

Provide one dedicated eight-GPU entrypoint for evaluating the native Cosmos
LIBERO teacher under matched coarse solver budgets K=1, K=2, and K=4. The
wrapper must reuse the validated formal evaluator and must not route the teacher
through the student FlowMap action layout.

## Interface

The wrapper accepts exactly one mode:

- `dry-run`: print the complete resolved matrix without creating output files.
- `run`: execute the full teacher-only matrix.

Configuration is supplied through explicit environment variables for the new
matrix root, official Cosmos checkpoint, teacher provenance lock, LIBERO
dataset, empty embedding, prompt table, Python interpreters, and episode count.
Defaults remain owned by the existing formal launcher where already validated.

## Evaluation Matrix

- Model role: `official_teacher` only.
- Budgets: matched video/action K=1, K=2, and K=4.
- Suites: `libero_10`, `libero_spatial`, `libero_object`, and `libero_goal`.
- Tasks: all 40 tasks.
- Episodes: 50 per task by default, configurable for smoke or scheduling.
- Parallelism: four shards, using the existing eight-GPU student/worker device
  pairing. The teacher role still uses the audited official Cosmos worker path.

K changes only the official teacher solver NFE. It does not alter the native
16-action horizon or introduce a FlowMap action downsample factor.

## Implementation

Add a thin shell wrapper that:

1. Resolves the repository and existing matrix launcher.
2. Forces `S4_MATRIX_ROLES=official_teacher`.
3. Requires a fresh output root and the official teacher inputs.
4. Delegates all provenance checks, action-grid preflight, sharding, per-suite
   completeness checks, JSON/CSV merging, and no-overwrite behavior to
   `run_cosmos_progressive_joint_124_eval_8gpu.sh`.

The wrapper will not duplicate evaluator logic or accept a student checkpoint
as the evaluated model.

## Failure Policy

Execution stops before environment actions when:

- the teacher provenance lock fails;
- the official CUDA worker is unavailable;
- the action layout is not native factor=1 with horizon 16;
- requested and observed K differ;
- the output root already exists;
- any suite, task, or episode is missing from the final matrix.

## Verification

Launcher tests will prove:

- teacher-only role is forced;
- K=1/2/4 and all four suites are delegated;
- dry-run is write-free;
- the output root is unique;
- the existing formal launcher remains the only evaluation implementation.

No live GPU job is started while implementing or testing the wrapper.
