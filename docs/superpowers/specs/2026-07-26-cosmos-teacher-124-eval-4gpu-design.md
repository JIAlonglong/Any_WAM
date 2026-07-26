# Cosmos Teacher K=1/2/4 Four-GPU Adjustment

## Goal

Add a four-GPU entrypoint for the existing official Cosmos teacher K=1/2/4
LIBERO matrix without removing or weakening the eight-GPU entrypoint.

## Execution Layout

- Four visible GPUs are split into two evaluator pairs: `(0,1)` and `(2,3)`.
- The existing formal evaluator receives `S4_FORMAL_NUM_SHARDS=2`.
- K=1, K=2, and K=4 remain sequential.
- Every K still covers all 40 tasks and 50 episodes per task by default.
- Output completeness and no-overwrite checks are unchanged.

## Video Policy

Set `S4_VIDEO_SEEDS=0` in the four-GPU wrapper. This saves one representative
episode per task and budget:

- 40 videos for K=1;
- 40 videos for K=2;
- 40 videos for K=4;
- 120 videos total out of 6,000 evaluation episodes.

All non-video episodes still produce durable JSON records and participate in
the JSON/CSV success-rate summaries.

## Implementation

Keep `run_cosmos_official_teacher_124_eval_8gpu.sh` as the shared teacher-only
launcher. Add a guarded internal shard override that defaults to four shards
and accepts only two or four. Add
`run_cosmos_official_teacher_124_eval_4gpu.sh`, which sets the internal value to
two, fixes representative video seeds to `0`, and delegates to the shared
launcher.

Tests must prove the four-GPU wrapper selects two shards and seed 0, while the
existing eight-GPU entrypoint continues to force four shards.

No live evaluation is started during implementation.
