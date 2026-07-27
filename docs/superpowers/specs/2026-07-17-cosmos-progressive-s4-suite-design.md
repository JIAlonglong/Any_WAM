# Cosmos Progressive S4 Full-Evaluation Suite Design

## Goal

Provide one reproducible command that executes the paper-faithful offline
metrics and the formal closed-loop LIBERO evaluation for the released Cosmos
Progressive S4 checkpoint, without mixing their artifacts or overwriting any
existing experiment directory.

## Chosen design

Create `evaluation/libero/run_cosmos_progressive_s4_suite.sh` with exactly two
modes: `run` and `dry-run`. `run` requires a fresh, non-existing `SUITE_ROOT`.
It creates only that parent root, then runs four serial phases:

1. `protocol`: deterministic 3-selection / 5-test-per-task manifests and the
   sole same-prior `1000,0` pair.
2. `teacher_cache`: an eight-step Cosmos cache for the final test manifest,
   using the 0 student / 1 official-Cosmos worker pair.
3. `offline_paper`: the paper evaluator on exactly that test manifest/cache,
   with four student steps and eight teacher steps, again using 0/1.
4. `closed_loop_formal`: the existing formal launcher, which owns the fixed
   concurrent 0/1 and 2/3 shard allocation and produces 500 seeded episodes.

The suite is deliberately serial. Four GPUs cannot simultaneously host the
single-pair offline teacher workload and the two-pair formal rollout without
resource contention. The formal launcher remains the sole owner of its
two-shard parallelism.

## Interface and result layout

Required environment variables are `SUITE_ROOT`, `S4_CKPT_ROOT`,
`S4_DATASET_PATH`, `S4_EMPTY_EMBEDDING`, `COSMOS_POLICY_PATH`,
`COSMOS_POLICY_PYTHON`, and `COSMOS_PREDICT2_REPO`. `S4_PROMPT_TABLE` and
`S4_INITIAL_STATES_JSON` remain optional and are passed through to the formal
launcher. `PYTHON_BIN` defaults to `python` and `S4_CONFIG` to the released
Cosmos Progressive config.

The generated paths are fixed below the new suite root:

```text
$SUITE_ROOT/protocol/
$SUITE_ROOT/protocol/teacher_cache/test/
$SUITE_ROOT/offline_paper/{records.jsonl,summary.json}
$SUITE_ROOT/closed_loop_formal/{shard_*,formal_summary.json}
$SUITE_ROOT/suite_status.jsonl
```

The status file has `started` and `completed` events for each phase. If a
child fails, `set -e` leaves the last event as `started`, retaining the partial
artifacts for diagnosis and never attempting a later phase.

`dry-run` prints all phase commands and GPU assignment but invokes no child
and creates no directory. Its formal command includes `S4_DRY_RUN=1` as the
command that a user could inspect or run separately; the suite itself merely
prints the fixed shard and prompt-table plan.

## Safety boundaries

- A pre-existing `SUITE_ROOT` is rejected before any child process, including
  a fresh formal root nested within it.
- The suite does not delete, resume, or overwrite artifacts.
- The cache and paper evaluator share the same generated test manifest and
  pair file; neither accepts user-provided replacements.
- The suite uses the existing dedicated Cosmos S4 formal launcher rather than
  reimplementing its seed, checkpoint, shard, or merge validation.
- It is for an official Cosmos cu128-compatible node only. Dry-run is safe on
  the current host; a live child will fail rather than silently substitute the
  Flash-WAM Python for the official Cosmos worker.

## Test strategy

Use subprocess tests with paths containing spaces and a Python sentinel.
Assert that dry-run emits the protocol, cache, paper, and formal commands;
plans the 0/1 offline pair and formal 0/1 + 2/3 layout; creates no root; and
never invokes the sentinel. Assert that a pre-existing suite root is rejected
before the sentinel executes. Assert that the script's static commands pin the
test manifest, cache, K=4 student, K=8 teacher, and formal child launcher.
