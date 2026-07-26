# Native Teacher Coarse-Grid Evaluation Design

## Goal

Evaluate the unmodified LingBotVA teacher under matched coarse sampling budgets
`K ∈ {1, 2, 4}` while preserving the teacher's native inference semantics.
Teacher and distilled student evaluations run independently on separate
eight-GPU allocations and write to separate result roots.

The native teacher evaluation is a diagnostic and comparison baseline. It does
not imply that the teacher was trained as a few-step model. `K` only controls
the evaluation-time solver budget.

## Non-goals

- Do not change the teacher checkpoint or retrain it.
- Do not route native teacher inference through FlowMap joint inference.
- Do not merge native-teacher and student results under one model identity.
- Do not modify student inference semantics.
- Do not make native `K=1/2/4` the teacher's canonical full-quality result;
  the standard multi-step teacher remains a separate baseline.

## Selected Architecture

Add a dedicated native-teacher server inside the Flash-WAM worktree. The server
uses the existing model-loading, WebSocket, LIBERO observation/action
preprocessing, provenance, and latency infrastructure, but its sampling core
implements the two original LingBotVA loops:

1. Run the native video denoising loop for `K` scheduler steps.
2. Run the native action denoising loop for `K` scheduler steps.
3. Update each modality's KV cache using the native update rules.
4. Return the complete continuous action grid to the existing LIBERO client.

The native server must not import or invoke `flowmap_inference`, and it must not
use `action_downsample_factor`. Every action temporal position executed by the
client must be produced by the native action loop.

The implementation lives in a separate server entry point rather than adding
another mode to the student server. This keeps teacher and student semantics
auditable and prevents a model-name or fallback error from silently selecting
the wrong backend.

## Components

### Native teacher server

Create `wan_va/wan_va_native_teacher_server.py`.

Its command-line interface accepts the same deployment controls required by the
current evaluator:

- `--config-name libero`
- `--port`
- `--checkpoint-path`
- `--num-steps`
- `--action-num-steps`
- `--model-name`
- `--latency-jsonl`
- `--save-root`

For the formal coarse-grid evaluation, both step arguments must have the same
value in `{1, 2, 4}`. The server rejects non-teacher model identities and logs
the resolved runtime contract before accepting a client:

```text
backend=native_teacher
model=teacher_native
video_steps=K
action_steps=K
action_grid=full
video_action_bridge=disabled
```

The checkpoint path remains configurable for reproducibility but must resolve
to a valid LingBotVA transformer directory.

### Evaluation entry point

Extend `evaluation/libero/run_eval_new.sh` with an explicit server backend
selection, defaulting to the existing FlowMap backend:

```text
SERVER_BACKEND=flowmap
SERVER_BACKEND=native_teacher
```

Only `native_teacher` launches the dedicated native server. The check-only
header prints the selected backend so a dry run can verify routing without
loading a model.

### Eight-GPU teacher launcher

Create `evaluation/libero/run_lingbotva_native_teacher_4suite_124_eval_8gpu.sh`.
It preserves the existing formal contract:

- four LIBERO suites;
- all 40 tasks;
- task partitions `[0,5)` and `[5,10)`;
- eight unique GPUs;
- 50 episodes per task for formal evaluation;
- matched native budgets `1/1`, `2/2`, and `4/4`;
- one summary per budget;
- sampler-latency provenance.

The launcher always sets `SERVER_BACKEND=native_teacher` and uses model identity
`teacher_native`. Its default output root is distinct from the student output.
Ports are caller-configurable so teacher and student eight-GPU allocations can
run concurrently.

## Data Flow

For each worker and budget:

```text
LIBERO client observation/history
  → native teacher WebSocket server
  → native video loop (K steps)
  → native action loop (K steps)
  → full continuous action grid
  → existing LIBERO client execution
  → task JSON, videos, action tensors, latency JSONL
  → budget-level summary
```

The client, task ordering, seeds, task partitions, and episode counts remain
identical to the student evaluation. Only the server backend differs.

## Validation and Failure Handling

The server fails before model loading when:

- the backend/model identity is inconsistent;
- video and action budgets differ;
- the budget is not a positive integer;
- the checkpoint directory is missing.

The launcher fails when:

- it does not receive exactly eight unique GPU IDs;
- a worker exits nonzero;
- a task result does not contain the expected episode count;
- merge-time coverage or sampler provenance is incomplete.

No partial run is reported as a completed formal result.

## Testing

Automated tests must verify:

1. The native server does not reference `flowmap_inference` or
   `action_downsample_factor`.
2. `K` configures both native schedulers and both modality loops.
3. The native action loop updates the complete action tensor.
4. `run_eval_new.sh` routes `SERVER_BACKEND=native_teacher` only to the native
   server and keeps FlowMap as the default.
5. Check-only output exposes backend, model identity, matched budgets, suites,
   task ranges, GPUs, and ports.
6. The eight-GPU launcher plans 24 workers across three budgets and rejects
   invalid GPU or budget contracts.
7. Teacher-native and student result roots/model identities cannot collide.

Runtime validation proceeds in two gates:

1. One task, one episode, `K=1/2/4`: verify startup contracts, tensor shapes,
   finite actions, full-grid updates, and clean shutdown.
2. Formal eight-GPU run: 40 tasks × 50 episodes for each `K`.

## Reporting Contract

Paper tables and figures label these results as:

```text
Native multi-step teacher, coarse solver budget K
```

They must not describe the teacher as a distilled or intrinsically few-step
model. Student results are labeled separately as distilled few-step inference.

