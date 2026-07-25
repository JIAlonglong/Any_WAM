# Task 4 Report — Progressive Aligned Video OPD

## Outcome

Implemented progressive scheduling, resolved configuration, launcher dry-run
output, and distributed logging for `_cosmos_aligned_video_opd_step`.

The trainer now selects exactly one named objective per optimizer update:

- `aligned_video_opd` at the configured interval (default 4), with precedence;
- `action_opd` on the existing enabled phase-offset arm;
- `main_anyflow` on all remaining updates.

The aligned branch calls only `_cosmos_aligned_video_opd_step`; it does not call
the legacy deployment rollout, independent video endpoint, or action-OPD path.

## Resolved configuration

- `opd_aux_interval=4`
- `opd_danceopd_rollout_steps=(2, 4)`
- `opd_danceopd_anchor_teacher_steps=8`
- `opd_danceopd_endpoint_weight=1.0`
- `opd_danceopd_velocity_weight=1.0`

Configuration import rejects invalid rollout grids, Teacher budgets other than
8, nonpositive/nonfinite endpoint or velocity weights, nonpositive intervals,
and a shifted Student grid without any legal query in the calibrated Cosmos
Teacher band.

The universal launcher variant also resolves interval 4, so its canonical
variant JSON agrees with the imported config.

## Logging

Added stable ordered aggregation and distributed reduction for the raw,
weighted, ratio, query, budget, valid-frame, and provenance metrics. The
following keys are emitted through the existing shared `log_dict`, and
therefore reach both TensorBoard and W&B offline:

```text
loss/opd_endpoint
loss/opd_same_state_velocity
loss_weighted/opd_endpoint
loss_weighted/opd_same_state_velocity
loss_ratio/opd_endpoint
loss_ratio/opd_same_state_velocity
opd/query_sigma
opd/query_index
opd/student_steps
opd/teacher_steps
opd/valid_video_frames
opd/same_prior_verified
opd/canonical_state_verified
```

The two contribution ratios share an absolute-contribution denominator clamped
to `1e-12`; tests cover both the 1:3 reduction and the all-zero finite case.

## Launcher and no-write proof

The wrapper accepts `--phase check`, `--steps`, and `--episodes` in the
repository's existing read-only conventions. Check/dry-run output prints the
complete resolved OPD fields, `ACTION_DOWNSAMPLE_FACTOR=1`,
`VIDEO_ACTION_BRIDGE=0`, and exactly one `--nproc_per_node=8` training command.

Behavior tests snapshot the fixture filesystem before and after both requested
invocations, assert no child launcher was called, and assert no output root was
created.

Both requested commands were also run against the real paths with an existing
clean formal Cosmos worktree supplied explicitly:

```bash
bash distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh \
  --phase check --run-tag aligned-anchor-check

bash distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh \
  --phase all --steps 1 --save-interval 1 --episodes 1 \
  --run-tag aligned-anchor-dry --dry-run
```

Both exited 0. Post-run `test ! -e` checks confirmed neither run root existed.
No training or child process started.

## TDD evidence

### RED

The initial protocol RED was:

```text
4 failed, 7 passed
```

Failures identified the old `main` name, missing aligned route/branch, and
missing stable metric order. Expanded config RED produced 14 expected failures
covering exact resolved values and invalid controls. Launcher RED produced:

```text
3 failed
```

for the old interval 8 variant, unsupported `--phase check`, and unsupported
smoke aliases. Dedicated ratio, log-schema, aligned enable-field, and
Teacher-band grid tests were each observed failing before their implementation.

### GREEN

Final prescribed Task 4 suite:

```text
57 passed in 76.01s
```

Broader related regression suite:

```text
107 passed in 77.20s
```

The broader run included progressive config/protocol/runner, the prior aligned
step harness, existing stage1→stage2 pipeline tests, and variant tests.

Additional verification:

- Python compilation passed for trainer and config.
- Shell syntax validation passed for the launcher.
- `git diff --check` passed.

## Self-review

- Confirmed objective precedence over optimizer steps 1 through 16, including
  aligned updates at 4/8/12/16, action updates at 10/14 when enabled, and main
  AnyFlow elsewhere.
- Confirmed disabling the action arm returns its slots to `main_anyflow`.
- Confirmed the aligned branch contains no call to the legacy deployment,
  independent full-OPD endpoint, or action auxiliary route.
- Confirmed action OPD retains a separately named branch and is not merged into
  the aligned result.
- Confirmed the 13-value accumulation and distributed reduction use one
  canonical order, with a test for that order and a separate public log-schema
  mapping test.
- Confirmed read-only modes return before output existence checks or child
  execution and leave the filesystem unchanged.
- Confirmed actual training still requires an explicit output root; only
  read-only invocations receive a deterministic non-created default.

## Concerns

`ACTION_DOWNSAMPLE_FACTOR=1` and `VIDEO_ACTION_BRIDGE=0` are launcher/runtime
bridge values printed and exported as required. The audited Stage-1/Stage-2
checkpoint action packing contract remains `downsample_survivor_v2` with factor
4; this task does not mutate that prior contract.

## Review fix round

This section supersedes the launcher/downsample statements above.

### Scheduler and objective names

The trainer adapter now receives `completed_updates=self.step`, converts that
counter to the one-based optimizer update being executed, and passes the
one-based value to the shared selector. A loop-convention test covers completed
updates 0 through 15 and confirms aligned video OPD on updates 4/8/12/16,
action OPD on 10/14, and `main_anyflow` elsewhere.

Aligned video cadence is a separate fixed config contract:
`ALIGNED_VIDEO_OPD_INTERVAL` must equal 4. `OPD_AUX_INTERVAL` remains the
independent action-OPD cadence. The remaining runtime fallbacks and guards now
use `main_anyflow`; the stale unreachable `deployment` logging guard was
removed.

### Packing and bridge contract

The launcher now prints, exports, and preflights the audited
`ACTION_DOWNSAMPLE_FACTOR=4`. It no longer advertises factor 1. The config and
Stage-2 preflight expose `video_action_bridge=0`; the launcher and config reject
an enabled bridge because no bridge exists in the audited factor-4 packing
contract.

### Authentic read-only chain and evaluation count

Read-only output now prints the real Stage-1 wrapper, provenance lock preparer,
Stage-2 wrapper, and joint evaluator commands. Each command is shell escaped
and prefixed with its resolved environment inputs. The synthetic detached
`ALIGNED_TRAIN_COMMAND` was removed. Both real training wrappers still contain
exactly one eight-rank launch.

`--episodes` now means episodes per LIBERO task, defaults to 50, and is
propagated as `S4_EPISODES_PER_TASK`. Thus the default remains 500 episodes per
matched K, while `--episodes 1` produces a real 10-episode-per-K smoke plan.
The joint K=1/2/4 evaluator validates child record counts dynamically and
records both `episodes_per_task` and `episodes_per_k` in its matrix summary.

### Fix-round TDD and verification

The scheduler/config RED was:

```text
8 failed, 14 passed
```

The focused scheduler/config GREEN was:

```text
22 passed in 22.37s
```

Launcher RED was three expected failures: incomplete/misleading read-only
output and missing evaluator episode propagation. Launcher GREEN was:

```text
3 passed in 0.29s
```

The complete focused Task 4 fix suite passed:

```text
61 passed in 83.73s
```

Final post-edit regression runs passed:

```text
112 passed in 82.97s
56 passed in 10.52s
```

The second run covers the formal evaluator, suite launcher, and joint K=1/2/4
matrix, including a live sentinel test with three episodes per task.

Final static and operational checks also passed:

- Python compilation for the trainer and progressive config.
- Shell syntax validation for both training wrappers and both evaluator
  launchers.
- Exactly one `--nproc_per_node=8` occurrence in each training wrapper.
- `git diff --check`.
- Explicit-path `--phase check` and
  `--phase all --steps 1 --save-interval 1 --episodes 1 --dry-run`.
- Before/after assertions confirmed neither explicit-path read-only invocation
  created its run root or launched a child process.

### Remaining concerns

None identified within the Task 4 scope. Formal training and evaluation were
not launched; the requested verification is read-only plus automated tests.
