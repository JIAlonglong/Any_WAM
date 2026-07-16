# Cosmos mixed-step 8-GPU launcher runbook

This runbook describes the prepared launcher only.  It does not authorize a
training, preflight, cache build, evaluator, or tmux run by itself.  Review the
paths and available GPUs before invoking any command below.

## What this launcher schedules

`distillation_flowmap/launch_cosmos_mixed_step_8gpu.sh` owns exactly three new,
independent target-free policies:

- `universe`: mixed S1/S2/S4 endpoint sampling (`0.50/0.30/0.20`)
- `s2`: S2-focused mixed endpoint sampling (`0.20/0.60/0.20`)
- `s1`: S1-focused mixed endpoint sampling (`0.70/0.20/0.10`)

Each fresh full run starts from the same Stage-1 **online-student** checkpoint,
runs for 5,000 steps, saves every 250 steps, and owns all eight explicitly
provided CUDA devices.  It never queues or retrains the already-completed S4
policy.

The serial gates are fixed:

```text
9-step Universe preflight ──PREFLIGHT_COMPLETE──> universe
universe ──TRAINING_COMPLETE──> s2
s2 ──TRAINING_COMPLETE──> s1
```

`TRAINING_COMPLETE` is written only after the runner exits successfully, the
step-5000 online-student transformer exists, no `target_student` directory is
present, and all S1/S2/S4 fixed-cache proxy JSONs plus the combined proxy are
present.  A failed worker writes a separate `*_FAILED` marker; the launcher
will not overwrite it or delete any output.

## Inputs to confirm before any future launch

The launcher deliberately requires explicit paths rather than selecting an
output/protocol/dataset automatically:

- `ROOT_BASE`: an existing dedicated parent directory for only these new
  policies and their logs.  It must not be the legacy progressive S4 root.
- `PROTOCOL_SOURCE_ROOT`: the read-only protocol JSON source containing
  `selection_manifest.json`, `eval_pairs.json`, and complete fixed caches under
  `teacher_cache/selection/t4` and `teacher_cache/selection/t8`.
- `DATASET_PATH` and `TEACHER_MODEL_PATH`: existing read-only inputs.
- `STAGE1_CHECKPOINT`: the agreed common Stage-1 online-student checkpoint
  with `online_student/transformer/config.json`.
- Eight unique GPU IDs via `--device-list`; the script rejects any other count.

For a full run, use a root base that has no old `PREFLIGHT_*` marker and no
pre-existing `universe`, `s2`, or `s1` output.  The launcher never removes old
markers, logs, worker scripts, checkpoints, or caches to make this true.

## Future preflight command

After the above paths are confirmed, the following is the intended form.  It
creates one named tmux session and a unique preflight root; it does not reuse a
prior one.

```bash
bash distillation_flowmap/launch_cosmos_mixed_step_8gpu.sh preflight \
  --root-base "$ROOT_BASE" \
  --protocol-source-root "$PROTOCOL_SOURCE_ROOT" \
  --dataset-path "$DATASET_PATH" \
  --teacher-model-path "$TEACHER_MODEL_PATH" \
  --stage1-checkpoint "$STAGE1_CHECKPOINT" \
  --python "$PYTHON_BIN" \
  --torchrun "$TORCHRUN_BIN" \
  --device-list 0,1,2,3,4,5,6,7 \
  --eval-device-list 0 \
  --eval-worker-device-list 0
```

Its immutable runtime contract is:

- root: `ROOT_BASE/preflight-universe-<UTC timestamp>-<pid>`
- tmux session: `cosmos-mixed-preflight-universe-<UTC timestamp>-<pid>`
- port: `29860`
- policy: `universe`
- training: exactly 9 steps, save at step 9
- forced selector sequence:
  `s1,s2,s4,s1,s2,s4,s1,s2,s4`
- completion checks: online student only, no target student, and all three
  `s1.json`, `s2.json`, `s4.json` fixed-cache proxy outputs.

The preflight marker is written at `ROOT_BASE/PREFLIGHT_COMPLETE`, not before
those checks.  A failure writes `ROOT_BASE/PREFLIGHT_FAILED`; inspect the log
and choose a new root base or resolve the issue manually rather than deleting
artifacts through the script.

## Future full-policy commands

Run only one after its prior marker exists.  Each command creates only its own
tmux session/log/worker and does not queue the next policy.

```bash
# Requires ROOT_BASE/PREFLIGHT_COMPLETE; port 29861.
bash distillation_flowmap/launch_cosmos_mixed_step_8gpu.sh start universe \
  --root-base "$ROOT_BASE" --protocol-source-root "$PROTOCOL_SOURCE_ROOT" \
  --dataset-path "$DATASET_PATH" --teacher-model-path "$TEACHER_MODEL_PATH" \
  --stage1-checkpoint "$STAGE1_CHECKPOINT" --python "$PYTHON_BIN" \
  --torchrun "$TORCHRUN_BIN" --device-list 0,1,2,3,4,5,6,7

# Requires ROOT_BASE/universe/TRAINING_COMPLETE; port 29862.
bash distillation_flowmap/launch_cosmos_mixed_step_8gpu.sh start s2 \
  --root-base "$ROOT_BASE" --protocol-source-root "$PROTOCOL_SOURCE_ROOT" \
  --dataset-path "$DATASET_PATH" --teacher-model-path "$TEACHER_MODEL_PATH" \
  --stage1-checkpoint "$STAGE1_CHECKPOINT" --python "$PYTHON_BIN" \
  --torchrun "$TORCHRUN_BIN" --device-list 0,1,2,3,4,5,6,7

# Requires ROOT_BASE/s2/TRAINING_COMPLETE; port 29863.
bash distillation_flowmap/launch_cosmos_mixed_step_8gpu.sh start s1 \
  --root-base "$ROOT_BASE" --protocol-source-root "$PROTOCOL_SOURCE_ROOT" \
  --dataset-path "$DATASET_PATH" --teacher-model-path "$TEACHER_MODEL_PATH" \
  --stage1-checkpoint "$STAGE1_CHECKPOINT" --python "$PYTHON_BIN" \
  --torchrun "$TORCHRUN_BIN" --device-list 0,1,2,3,4,5,6,7
```

The full-run tmux names are `cosmos-mixed-{universe|s2|s1}-<UTC timestamp>-<pid>`.
Their log and generated, shell-quoted worker script are stored below
`ROOT_BASE/logs/`.  The worker, rather than the interactive caller, writes the
success/failure marker so an interrupted shell cannot make a false completion
claim.

## Status and offline proxy evaluation

Status is read-only: it lists existing markers, available tmux sessions, and
the last 20 lines of each launcher log.  It creates no output.

```bash
bash distillation_flowmap/launch_cosmos_mixed_step_8gpu.sh status \
  --root-base "$ROOT_BASE"
```

To re-evaluate an owned checkpoint without retraining, use the explicit
eval-only runner path.  The checkpoint must be directly below the named
policy's `checkpoints/step_N` directory and its root manifest must match the
policy.

```bash
bash distillation_flowmap/launch_cosmos_mixed_step_8gpu.sh eval universe \
  "$ROOT_BASE/universe/checkpoints/step_5000" \
  --root-base "$ROOT_BASE" \
  --protocol-source-root "$PROTOCOL_SOURCE_ROOT" \
  --dataset-path "$DATASET_PATH" \
  --teacher-model-path "$TEACHER_MODEL_PATH" \
  --python "$PYTHON_BIN" \
  --eval-device-list 0 \
  --eval-worker-device-list 0
```

The eval worker calls the dedicated `--eval-checkpoint ... --run` branch in
`run_cosmos_mixed_step_policy.py`; there is no train argv in that branch.  It
uses the reviewed distinct t4/t8 fixed caches and writes a combined selection
proxy only after all three S1/S2/S4 evaluator JSONs exist.  These metrics are
an offline fixed-cache proxy, not a claim of real robot rollout success.
