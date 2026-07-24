# Cosmos Progressive Joint 1/2/4-Step LIBERO Evaluation Design

Date: 2026-07-24

## Goal

Evaluate the locally trained Cosmos progressive universal video-action
checkpoint with joint student rollout depths `K=1`, `K=2`, and `K=4`.
Each setting must run the complete LIBERO-10 closed-loop protocol: ten tasks,
fifty shared environment seeds per task, and exactly 500 durable episode
records. The three settings run sequentially and use identical tasks, seeds,
initial-state resolution, checkpoint, prompts, and Cosmos anchor worker.

The selected `K` applies to both the video and action state in the same joint
FlowMap integration. A run must never use one video step count and a different
action step count.

## Non-goals

- Do not retrain or modify the checkpoint.
- Do not reinterpret the paper-only S4 offline `G_anchor` or `G_comp` protocol
  as a K=1 or K=2 metric. Those paper diagnostics remain fixed at K=4.
- Do not save videos for all 1,500 episodes.
- Do not change the existing default K=4 behavior for callers that omit the
  new step configuration.
- Do not use the generic 48-channel WanVA service; the evaluator continues to
  use the 16-channel Cosmos progressive joint state and official Cosmos raw
  anchor worker.

## Runtime interface

The live rollout module gains:

```text
--student-steps {1,2,4}
--save-video
```

`--student-steps` defaults to `4` for backward compatibility. The value is
passed through the service and engine into the joint runner, which supplies
the same value as `K_steps` to `_student_euler_integrate`. The returned action
therefore comes from the action component of the same K-step joint rollout as
the video state.

`--save-video` controls whether the client collects frames and writes an MP4.
When it is absent, the client does not append frames in memory and records
`video_path=null`.

The formal launcher gains:

```text
S4_STUDENT_STEPS=1|2|4
S4_FORMAL_NUM_SHARDS=2|4
S4_VIDEO_SEEDS=<comma-separated seeds>
```

Defaults remain `4`, `2`, and an empty video seed set so existing K=4
invocations retain their established two-shard behavior without unexpected
video output.

## Eight-GPU topology

The new matrix launcher fixes `S4_FORMAL_NUM_SHARDS=4` and partitions eight
visible GPUs into four student/worker pairs:

| Shard | Student GPU | Cosmos worker GPU | LIBERO task range |
|---|---:|---:|---|
| 0 | 0 | 1 | `[0,3)` |
| 1 | 2 | 3 | `[3,6)` |
| 2 | 4 | 5 | `[6,8)` |
| 3 | 6 | 7 | `[8,10)` |

Every shard evaluates the same fifty seeds for its disjoint tasks. The
slightly uneven 3/3/2/2 task partition is deterministic and covers each task
exactly once.

The existing two-shard plan remains:

| Shard | Student GPU | Cosmos worker GPU | LIBERO task range |
|---|---:|---:|---|
| 0 | 0 | 1 | `[0,5)` |
| 1 | 2 | 3 | `[5,10)` |

## Sequential matrix launcher

Add:

```text
evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh
```

The launcher accepts `run` and `dry-run`, requires a new absent
`MATRIX_ROOT`, and invokes the formal closed-loop launcher sequentially:

```text
MATRIX_ROOT/k1
MATRIX_ROOT/k2
MATRIX_ROOT/k4
```

The order is strictly K=1, then K=2, then K=4. A failed setting stops the
matrix before starting the next one. Completed and partial outputs are
preserved; the launcher never overwrites an existing matrix root.

All settings use seeds `0..49`. The matrix launcher sets
`S4_VIDEO_SEEDS=0,1`, so each setting saves two shared episodes per task:
twenty videos per K and sixty videos total. The other 1,440 episodes retain
their JSON records but do not collect frames or write MP4 files.

If the caller supplies `S4_PROMPT_TABLE`, all settings use it. Otherwise K=1
materializes the task-keyed prompt table under its result root, and K=2/K=4
reuse that exact table.

## Result provenance and validation

Every episode record must contain:

```text
student_steps
s4_checkpoint
task_idx
seed
success
video_path
```

The formal merger validates:

- `student_steps` equals the requested K for every record;
- the checkpoint path is identical across all records;
- every expected `(task, seed)` appears exactly once;
- no unexpected or duplicate `(task, seed)` exists;
- each task belongs to the shard recorded in its path;
- exactly 500 records are present.

Each `formal_summary.json` includes `student_steps`, per-task success,
equal-task macro success, and the existing deterministic bootstrap confidence
interval. The matrix launcher writes a top-level summary that references the
three formal summaries and permits direct K=1/K=2/K=4 comparison without
merging their episode records.

## Failure handling

- CUDA/Cosmos preflight runs for every configured student/worker pair before
  episodes start.
- The existing driver requirement remains `>=570.124.06`.
- The official cu128 worker environment and its explicit Cosmos package paths
  remain mandatory.
- Server and setup failures still produce durable episode records.
- Missing records, step mismatches, shard mismatches, or checkpoint
  mismatches fail the merge instead of publishing a partial summary.
- Output roots follow the existing no-overwrite policy.

## Testing

Tests must first fail against the current hard-coded K=4 implementation, then
cover:

1. CLI accepts only K=1, K=2, and K=4.
2. The joint runner passes the requested K to `_student_euler_integrate` and
   uses it for the returned action state.
3. The engine sends the configured K rather than a hard-coded four.
4. Non-video episodes do not collect frames and record `video_path=null`.
5. Video seeds 0 and 1 save videos while other seeds do not.
6. The four-shard plan covers tasks 0 through 9 exactly once on all eight
   GPUs.
7. The merger rejects missing, duplicate, wrong-shard, wrong-checkpoint, and
   wrong-step records.
8. Matrix dry-run emits three sequential 500-episode plans, creates no output
   directory, and launches no child process.
9. Existing K=4 and two-shard tests remain green.

No live GPU evaluation is launched during implementation or verification.
