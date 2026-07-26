# Cosmos Official-Teacher Four-GPU Co-Located Evaluation Design

## Goal

Reduce the wall-clock time of the formal Cosmos official-teacher LIBERO
evaluation on four A800 GPUs without changing the model, matched-compute
protocol, episode count, result schema, or completeness checks.

The current four-GPU launcher creates only two heavy inference workers:

- evaluator GPU 0, Cosmos worker GPU 1;
- evaluator GPU 2, Cosmos worker GPU 3.

It therefore evaluates 6,000 episodes with only two Cosmos model replicas.

## Selected design

Add an explicit GPU-layout setting to the shared formal shard launcher:

- `paired`: existing evaluator/worker pairs, retained as the default;
- `colocated`: evaluator and Cosmos worker share one physical GPU.

The four-GPU official-teacher wrapper selects:

- four formal shards;
- `colocated` layout;
- physical GPU assignments `0/0`, `1/1`, `2/2`, and `3/3`;
- task ranges `[0,3)`, `[3,6)`, `[6,8)`, and `[8,10)`.

The co-located layout is valid only for `official_teacher`. Student roles still
load a FlowMap student in the evaluator and therefore retain the paired layout.
The launcher must reject a co-located student configuration before starting any
process.

## Runtime semantics

Each shard starts one LIBERO evaluator process. The evaluator creates a
`CosmosPolicyActionTeacher`, whose official model is loaded once in its raw
worker subprocess. In co-located mode both processes see the same physical GPU,
but only the worker holds the full teacher model. This increases teacher
replicas from two to four without adding a second teacher replica per GPU.

All experiment semantics remain unchanged:

- official teacher only;
- matched video/action budgets `K=1,2,4`;
- all four LIBERO suites and 40 tasks;
- 50 episodes per task;
- continuous native teacher action horizon;
- seed 0 representative videos only;
- strict per-task record uniqueness and episode-count validation;
- JSON/CSV matrix summaries.

The new run must use a fresh `MATRIX_ROOT`. Existing V5 results are neither
deleted nor reused.

## Safety and failure handling

The resolved plan must print, for each shard:

- evaluator GPU;
- worker GPU;
- task range;
- selected layout.

The launcher fails before live execution when:

- layout is not `paired` or `colocated`;
- `colocated` is requested for a non-teacher role;
- four-shard co-location does not resolve to GPUs `0,1,2,3`;
- the destination already exists.

Any shard failure continues to make the formal matrix fail. Partial results
cannot be merged as a complete evaluation.

## Verification

Implementation follows test-driven development:

1. Add launcher contract tests that fail because co-location is unsupported.
2. Implement the minimal layout resolver and wrapper settings.
3. Verify dry-run emits four shards with mappings `0/0`, `1/1`, `2/2`, `3/3`.
4. Verify existing paired student and eight-GPU paths remain unchanged.
5. Run shell syntax checks and the related evaluation test suite.
6. In a new four-GPU allocation, run preflight and one-episode smoke before the
   full 50-episode matrix.

Expected throughput improvement is close to 2x when Cosmos inference is the
dominant cost. Exact speedup remains an empirical runtime measurement.
