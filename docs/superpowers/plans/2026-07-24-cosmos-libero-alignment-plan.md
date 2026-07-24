# Cosmos LIBERO Semantic Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the Cosmos-native LIBERO Stage-2 training, diagnostics, experiment launch, checkpoint recovery, and matched-budget 40-task evaluation path while preserving the validated AnyFlow and Cosmos scheduler semantics.

**Architecture:** Keep the existing Cosmos joint rollout and checkpoint contract as the core. Add backend-neutral numerical/distributed safety helpers around it, Cosmos-native mechanism probes inside `FlowMapStepMixin`, declarative experiment and evaluation manifests, and fail-closed shell/Python entry points. Every mutation follows RED-GREEN-REFACTOR and is reviewed before the next task.

**Tech Stack:** Python 3.10, PyTorch distributed, pytest, Bash, torchrun, TensorBoard, W&B offline, Cosmos Predict2.5, LIBERO.

## Global Constraints

- Target worktree: `/kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/cosmos-progressive-s4-eval`.
- Reference is read-only at LingBotVA commit `351ec98c95e1bc2012bdcf9a8248af228e7208ae`.
- Preserve Cosmos raw scheduler, latent, policy, CFG worker, action packing, and joint Euler semantics.
- Main AnyFlow sampling remains 50% `r=t`, 25% `r=0`, and 25% arbitrary nonzero `r<t`.
- Endpoint pairs are exactly `8,1;8,2;8,4`, selected uniformly.
- Endpoint clean-region sampling is `sigma_r=(1-Beta(5,2))*0.25`.
- Compositional rollout choices are exactly `2,4`; K=1 is excluded.
- Compositional queries use pre-update joint video/action states and per-sample `floor(Beta(5,2)*K)`.
- Video-only experiments set action velocity OPD weight to exactly `0`.
- Do not copy WanVA timestep formulas, token layouts, LoRA inheritance, fixed 16-step defaults, or LingBotVA paths.
- No existing checkpoint, output directory, or experiment record may be deleted or overwritten.
- Fresh outputs are atomically claimed; dry-run and `CHECK_ONLY` do not write files.
- Every behavior change has a failing test before production code.
- TensorBoard and W&B offline receive the same mechanism metrics.
- Formal evaluation covers four suites, 40 tasks, matched video/action K=1/2/4, and exactly 50 episodes per task.
- GPU and simulator claims require fresh logs from the actual allocated resources.

---

### Task 1: Freeze AnyFlow, Anchor, and Compositional Sampling Contracts

**Files:**
- Modify: `distillation_flowmap/tests/test_cosmos_progressive_config.py`
- Modify: `distillation_flowmap/tests/test_danceopd_query.py`
- Modify: `distillation_flowmap/tests/test_cosmos_deployment_rollout.py`
- Modify only if a test exposes a gap: `distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py`
- Modify only if a test exposes a gap: `distillation_flowmap/danceopd_query.py`
- Modify only if a test exposes a gap: `distillation_flowmap/flowmap_step.py`

**Interfaces:**
- Consumes: current progressive config and `DanceOPDQuerySampler`.
- Produces: executable regression contracts for all immutable sampling semantics.

- [ ] **Step 1: Add failing distribution-boundary tests**

Add deterministic tests using patched uniform/Beta samples that assert:

```python
assert branch_for(0.00) == "diffusion"
assert branch_for(0.499999) == "diffusion"
assert branch_for(0.50) == "endpoint"
assert branch_for(0.749999) == "endpoint"
assert branch_for(0.75) == "arbitrary"
assert arbitrary_r > 0
assert arbitrary_r < t
```

Also assert `OPD_ROLLOUT_STEP_PAIRS == ((8, 1), (8, 2), (8, 4))`,
uniform pair selection, endpoint sigma in `[0, 0.25]`, and exact endpoint
formula with detached teacher target.

- [ ] **Step 2: Add failing compositional-grid tests**

For K=2 and K=4, assert the pre-update candidate sigmas are exactly:

```python
{2: (1.0, 0.5), 4: (1.0, 0.75, 0.5, 0.25)}
```

Patch distinct Beta samples per batch element and assert independent
`floor(beta*K)` indices. Assert teacher and student receive identical video,
action, and sigma tensors, teacher output is detached, student output remains
gradient-bearing, and action-velocity contribution is zero in video-only mode.

- [ ] **Step 3: Run RED**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_cosmos_progressive_config.py \
  distillation_flowmap/tests/test_danceopd_query.py \
  distillation_flowmap/tests/test_cosmos_deployment_rollout.py
```

Expected: any missing semantic assertion fails for the named contract rather
than import/setup errors.

- [ ] **Step 4: Implement only exposed gaps**

Do not rewrite passing Cosmos paths. Extract a pure helper only when needed,
for example:

```python
def sample_main_anyflow_branch(probability: torch.Tensor) -> torch.Tensor:
    return torch.where(
        probability < 0.5, 0,
        torch.where(probability < 0.75, 1, 2),
    )
```

Keep Cosmos timestep-to-sigma conversion in its existing scheduler helpers.

- [ ] **Step 5: Run GREEN and regressions**

Run the command from Step 3 and:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q distillation_flowmap/tests/test_cosmos_progressive_opd.py
```

- [ ] **Step 6: Commit**

```bash
git add distillation_flowmap/tests distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py distillation_flowmap/danceopd_query.py distillation_flowmap/flowmap_step.py
git commit -m "test: freeze Cosmos AnyFlow and OPD semantics"
```

### Task 2: Add Dtype-Aware Terminal-Prior Validation

**Files:**
- Create: `distillation_flowmap/numerical_contracts.py`
- Create: `distillation_flowmap/tests/test_numerical_contracts.py`
- Modify: `distillation_flowmap/flowmap_step.py`
- Modify: `distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py`

**Interfaces:**
- Produces: `compare_terminal_prior(actual, expected, *, source_dtype, warn_factor=0.5) -> TerminalPriorCheck`.
- `TerminalPriorCheck` contains `max_error`, `reference_scale`, `atol`, `rtol`, `threshold`, and `severity`.

- [ ] **Step 1: Write failing comparator tests**

Test these exact outcomes:

```python
accepted = compare_terminal_prior(
    torch.tensor([1.192e-6]), torch.zeros(1),
    source_dtype=torch.bfloat16,
)
assert accepted.severity in {"quiet", "warning"}

corrupt = compare_terminal_prior(
    torch.tensor([1e-2]), torch.zeros(1),
    source_dtype=torch.bfloat16,
)
assert corrupt.severity == "error"
```

Also cover fp32, nonzero reference scale, warning-band metadata, and non-finite
inputs.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q distillation_flowmap/tests/test_numerical_contracts.py
```

Expected: import failure because `numerical_contracts.py` does not exist.

- [ ] **Step 3: Implement the pure comparator**

Use fp32 differences, `torch.finfo(source_dtype).eps`, a documented fp32
scheduler floor that exceeds `1.192e-6`, and a relative tolerance against
`expected.abs().max()`. Do not delete or bypass either terminal expected-state
calculation.

- [ ] **Step 4: Integrate both Cosmos terminal checks**

Replace scalar `max_error > tolerance` branches with the helper. Warning
outcomes emit a rank-safe warning and return diagnostic tensors; error
outcomes raise with all effective thresholds.

- [ ] **Step 5: Run GREEN and targeted regressions**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_numerical_contracts.py \
  distillation_flowmap/tests/test_cosmos_progressive_opd.py \
  distillation_flowmap/tests/test_cosmos_deployment_rollout.py
```

- [ ] **Step 6: Commit**

```bash
git add distillation_flowmap/numerical_contracts.py distillation_flowmap/tests/test_numerical_contracts.py distillation_flowmap/flowmap_step.py distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py
git commit -m "fix: tolerate numerical Cosmos terminal prior error"
```

### Task 3: Synchronize Non-Finite Decisions Across Ranks

**Files:**
- Create: `distillation_flowmap/distributed_safety.py`
- Create: `distillation_flowmap/tests/test_distributed_nonfinite_guard.py`
- Modify: `distillation_flowmap/flowmap_step.py`
- Modify: `distillation_flowmap/flowmap_trainer.py`

**Interfaces:**
- Produces: `all_ranks_finite(local_finite: bool | Tensor, *, device) -> bool`.
- Produces: `reduce_nonfinite_origins(local_flags: Mapping[str, bool], *, device) -> dict[str, bool]`.
- Trainer owns a sticky `skip_accumulation_window` and cumulative origin counters.

- [ ] **Step 1: Write failing pure/fake-distributed tests**

Use a fake `all_reduce` to assert MIN semantics:

```python
assert all_ranks_finite(True) is True
assert simulated_two_rank_result([True, False]) is False
```

Test that an origin present on one rank is visible on all ranks and that
video, action, GT/teacher, endpoint OPD, compositional OPD, and action OPD
keys remain distinct.

- [ ] **Step 2: Write failing trainer control-flow tests**

With a two-microbatch accumulation window, assert that a non-finite value in
microbatch one causes both microbatches to skip all auxiliary backward and
causes zero calls to optimizer, LR scheduler, and EMA. Assert cumulative
total and per-origin counters increment once per skipped optimizer window.

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q distillation_flowmap/tests/test_distributed_nonfinite_guard.py
```

- [ ] **Step 4: Implement collective helpers and sticky control flow**

Synchronize before any divergent backward branch and again after gradient
norm calculation. On skip, every rank calls `zero_grad(set_to_none=True)`;
none advance optimizer/scheduler/EMA.

- [ ] **Step 5: Add logging**

Emit synchronized totals and origin counters to the existing common log dict:

```text
safety/nonfinite_skipped_windows
safety/nonfinite_video
safety/nonfinite_action
safety/nonfinite_teacher_or_gt
safety/nonfinite_opd_endpoint
safety/nonfinite_opd_compositional
safety/nonfinite_opd_action
safety/nonfinite_gradient
```

- [ ] **Step 6: Run GREEN and trainer regressions**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_distributed_nonfinite_guard.py \
  distillation_flowmap/tests/test_cosmos_progressive_opd.py \
  distillation_flowmap/tests/test_cosmos_training_contract.py
```

- [ ] **Step 7: Commit**

```bash
git add distillation_flowmap/distributed_safety.py distillation_flowmap/tests/test_distributed_nonfinite_guard.py distillation_flowmap/flowmap_step.py distillation_flowmap/flowmap_trainer.py
git commit -m "fix: synchronize nonfinite Cosmos training skips"
```

### Task 4: Enforce Explicit Cosmos Paths and Stage-2 Lineage

**Files:**
- Modify: `distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py`
- Modify: `distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh`
- Modify: `distillation_flowmap/cosmos_stage2_lineage.py`
- Modify: `distillation_flowmap/tests/test_cosmos_stage2_lineage.py`
- Modify: `distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py`
- Modify: `distillation_flowmap/tests/test_cosmos_progressive_config.py`

**Interfaces:**
- Consumes: `validate_stage1_parent` and `validate_stage2_resume`.
- Produces: a validated lineage JSON passed to training and embedded in the Stage-2 resolved config.

- [ ] **Step 1: Add failing config and launcher tests**

Assert config import without explicit `STUDENT_BASE_MODEL_PATH` and
`RESUME_FROM_PATH` fails with the variable name. Assert no source contains
`/root/nas` or LingBotVA fallback paths.

Launcher tests must reject:

- missing base `config.json`;
- backend other than `cosmos_policy`;
- contaminated or aliased Stage-1 online/target;
- base-model identity mismatch;
- output equal to, nested under, or containing Stage-1;
- output symlinks;
- resume missing target, optimizer, or `lr_scheduler.pt`;
- mismatched checkpoint step or parent identity.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_cosmos_stage2_lineage.py \
  distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py \
  distillation_flowmap/tests/test_cosmos_progressive_config.py
```

- [ ] **Step 3: Wire the existing validator into the launcher**

The shell launcher invokes the Python validator before torchrun and consumes
its machine-readable result. Remove inline weaker fallbacks. Fresh output is
claimed only after all preflight checks and never during dry-run.

- [ ] **Step 4: Make Python config fail closed**

Read explicit environment/arguments and validate `config.json`,
`backend=cosmos_policy`, stage contract, and parent identity. Preserve atomic
checkpoint config writes already in the trainer.

- [ ] **Step 5: Run GREEN, shell syntax, and no-write dry-run**

```bash
bash -n distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_cosmos_stage2_lineage.py \
  distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py \
  distillation_flowmap/tests/test_cosmos_progressive_config.py
```

- [ ] **Step 6: Commit**

```bash
git add distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh distillation_flowmap/cosmos_stage2_lineage.py distillation_flowmap/tests
git commit -m "fix: enforce Cosmos Stage2 lineage at launch"
```

### Task 5: Add Mechanism Metric Formulas and Logging Infrastructure

**Files:**
- Create: `distillation_flowmap/mechanism_diagnostics.py`
- Create: `distillation_flowmap/tests/test_mechanism_diagnostics.py`
- Modify: `distillation_flowmap/flowmap_trainer.py`

**Interfaces:**
- Produces: pure finite-safe G and action-gain summaries.
- Produces: deterministic post-update diagnostic scheduler and distributed sum/count reducer.

- [ ] **Step 1: Write failing formula tests**

Cover exact values, zero denominators, NaN inputs, and distributed sum/count
aggregation for:

```python
g_anchor
g_anchor_mse
g_comp
g_comp_mse
g_anchor_to_comp_ratio
video_to_action_oracle_gain
video_to_action_full_joint_gain
video_to_action_recoverable_fraction
video_to_action_residual_action_gap
```

Assert every ratio is finite due to a named clamp constant.

- [ ] **Step 2: Write failing schedule/fan-out tests**

Diagnostics run only after a successful optimizer step, at the configured
interval, with deterministic seed and RNG restoration. Assert identical keys
reach TensorBoard, offline W&B, and JSONL.

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q distillation_flowmap/tests/test_mechanism_diagnostics.py
```

- [ ] **Step 4: Implement formulas, reduction, scheduling, and fan-out**

Do not add model-specific calls. Keep the module pure except for explicit
distributed reduction helpers.

- [ ] **Step 5: Run GREEN**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_mechanism_diagnostics.py \
  distillation_flowmap/tests/test_cosmos_training_contract.py
```

- [ ] **Step 6: Commit**

```bash
git add distillation_flowmap/mechanism_diagnostics.py distillation_flowmap/tests/test_mechanism_diagnostics.py distillation_flowmap/flowmap_trainer.py
git commit -m "feat: add finite-safe mechanism diagnostics"
```

### Task 6: Implement Cosmos-Native Context and G Probes

**Files:**
- Create: `distillation_flowmap/tests/test_cosmos_mechanism_probes.py`
- Modify: `distillation_flowmap/flowmap_step.py`
- Modify: `distillation_flowmap/flowmap_trainer.py`
- Modify: `distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py`
- Modify: `distillation_flowmap/tests/test_cosmos_progressive_config.py`

**Interfaces:**
- Consumes: Task 5 formulas and scheduler.
- Produces: `_run_cosmos_mechanism_probe(batch, *, seed, teacher_steps, student_steps) -> dict[str, Tensor]`.

- [ ] **Step 1: Write failing value-sensitive routing test**

Intercept the real Cosmos joint-input builder. Feed distinct GT, student,
teacher-video, and teacher-joint video latents while holding the action state
fixed. Assert the captured video portion changes and the action head receives
each replacement rather than GT. Assert diagnostics are `no_grad` and do not
detach or substitute values before the forward call.

- [ ] **Step 2: Write failing metric-semantic tests**

Assert G-anchor is teacher continuation versus teacher endpoint, while G-comp
is direct student mapping versus composed student mapping. Do not permit OPD
weighted-loss ratios to be relabelled as G metrics.

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q distillation_flowmap/tests/test_cosmos_mechanism_probes.py
```

- [ ] **Step 4: Implement probes using Cosmos APIs**

Use corrected action packing, actual action masks, raw teacher worker calls,
and the shared joint integrator. Keep main teacher-forced action training
unchanged. Return all required action-error, gain, G, endpoint, field-match,
finite-count, near-zero, explosion, and dominance keys.

- [ ] **Step 5: Add config defaults**

Support:

```text
MECHANISM_DIAGNOSTICS=1
MECHANISM_DIAGNOSTIC_INTERVAL=100
MECHANISM_DIAGNOSTIC_SEED=42
MECHANISM_DIAGNOSTIC_R=500
MECHANISM_DIAGNOSTIC_S=250
MECHANISM_DIAGNOSTIC_TEACHER_STEPS=8
```

Validate ranges and print resolved values.

- [ ] **Step 6: Run GREEN and relevant regressions**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_cosmos_mechanism_probes.py \
  distillation_flowmap/tests/test_mechanism_diagnostics.py \
  distillation_flowmap/tests/test_cosmos_progressive_config.py
```

- [ ] **Step 7: Commit**

```bash
git add distillation_flowmap/flowmap_step.py distillation_flowmap/flowmap_trainer.py distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py distillation_flowmap/tests
git commit -m "feat: diagnose Cosmos video to action gains"
```

### Task 7: Add Independent Progressive and Video-Only APM Launchers

**Files:**
- Create: `distillation_flowmap/cosmos_libero_variants.py`
- Create: `distillation_flowmap/cosmos_libero_video_apm_variants.json`
- Create: `distillation_flowmap/run_cosmos_libero_train_8gpu.sh`
- Create: `distillation_flowmap/tests/test_cosmos_libero_variants.py`
- Create: `distillation_flowmap/tests/test_run_cosmos_libero_train_8gpu.py`

**Interfaces:**
- Produces: resolved immutable variant records for s1/s2/s4/universal/universal-video-action and four video-only APM arms.

- [ ] **Step 1: Write failing variant-isolation tests**

Load every arm and normalize run tag, output, port, and intended coefficients.
Assert s1/s2/s4/universal differ only in declared budget fields. Assert the
four APM arms differ only in endpoint/compositional coefficients and all have
action OPD zero.

- [ ] **Step 2: Write failing launcher-contract tests**

Assert support for `--steps`, `--save-interval`, `--master-port`,
`--output-root`, `--run-tag`, `--dry-run`, and `--check-only`; exactly one
`--nproc_per_node=8`; TensorBoard on; W&B offline; HF/Transformers offline;
default save interval 1000; resolved config complete; no writes on dry-run;
existing output rejected on formal run.

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_cosmos_libero_variants.py \
  distillation_flowmap/tests/test_run_cosmos_libero_train_8gpu.py
```

- [ ] **Step 4: Implement declarative resolver and launcher**

Reuse the command-array, shared-root, strict numeric/port, and synchronous
failure patterns from the reference wrapper. Call Task 4 lineage preflight.
Do not source or import LingBotVA-specific launchers.

- [ ] **Step 5: Run GREEN and real-path dry-runs**

Run tests, `bash -n`, and one write-free dry-run per variant. Verify no output
directory or training process is created.

- [ ] **Step 6: Commit**

```bash
git add distillation_flowmap/cosmos_libero_variants.py distillation_flowmap/cosmos_libero_video_apm_variants.json distillation_flowmap/run_cosmos_libero_train_8gpu.sh distillation_flowmap/tests
git commit -m "feat: launch independent Cosmos LIBERO variants"
```

### Task 8: Prove Training/Inference Parity and Model-Role Recovery

**Files:**
- Create: `evaluation/libero/tests/test_cosmos_train_inference_parity.py`
- Modify: `evaluation/libero/rollout_cosmos_progressive_s4.py`
- Modify: `evaluation/libero/cosmos_progressive_s4_server.py`
- Modify: `distillation_flowmap/cosmos_training_contract.py`
- Modify: `distillation_flowmap/tests/test_cosmos_training_contract.py`

**Interfaces:**
- Produces: explicit `model_role` and `video_steps`/`action_steps` request contract.
- Produces: state-by-state parity for K=1/2/4.

- [ ] **Step 1: Write failing trajectory-parity tests**

For deterministic fake velocity functions, compare every training deployment
state with service inference for K=1,2,4. Assert video and action K are both
explicit and equal for matched experiments. Assert different K produces
different state trajectories.

- [ ] **Step 2: Write failing checkpoint-role tests**

Cover official teacher, corrected Stage-1 target, Stage-2 online, and Stage-2
target. Reject missing/incompatible metadata and absolute fallback paths.
Require backend, base identity, packing schema, supported K, scheduler
endpoints, and lineage.

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  evaluation/libero/tests/test_cosmos_train_inference_parity.py \
  evaluation/libero/tests/test_cosmos_progressive_s4_service.py \
  distillation_flowmap/tests/test_cosmos_training_contract.py
```

- [ ] **Step 4: Implement explicit roles and parity fixes**

Keep the shared `_student_euler_integrate` as the only student transition
implementation. Add an official-teacher adapter that truly performs requested
K transitions or fails preflight.

- [ ] **Step 5: Run GREEN**

Run Step 3 and existing progressive runner tests.

- [ ] **Step 6: Commit**

```bash
git add evaluation/libero distillation_flowmap/cosmos_training_contract.py distillation_flowmap/tests
git commit -m "test: align Cosmos training and inference contracts"
```

### Task 9: Extend Formal Evaluation to Four Suites and Joined Products

**Files:**
- Create: `evaluation/libero/cosmos_4suite_eval_manifest.py`
- Create: `evaluation/libero/merge_cosmos_4suite_results.py`
- Create: `evaluation/libero/tests/test_cosmos_4suite_eval_manifest.py`
- Create: `evaluation/libero/tests/test_merge_cosmos_4suite_results.py`
- Modify: `evaluation/libero/run_cosmos_progressive_s4_eval.sh`
- Modify: `evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh`
- Modify: `evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py`
- Modify: `evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py`

**Interfaces:**
- Produces: exact expected keys `(suite, task, seed, model_role, K)`.
- Produces: `episodes.csv`, `tasks.csv`, `offline_metrics.csv`, `mechanism_metrics.csv`, and `summary.json`.

- [ ] **Step 1: Write failing manifest/completeness tests**

Assert the suites are exactly `libero_10`, `libero_spatial`, `libero_object`,
and `libero_goal`; each has tasks 0-9 and seeds/episodes 0-49; each
model/budget therefore expects exactly 2,000 records.

Test duplicate, missing, unexpected, wrong checkpoint, wrong suite, wrong K,
and partial-task failures.

- [ ] **Step 2: Write failing CSV/JSON tests**

Assert stable schemas, deterministic ordering, exact source artifact and
lineage identities, macro/micro success, exact episode counts, and joined
offline/mechanism metrics.

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  evaluation/libero/tests/test_cosmos_4suite_eval_manifest.py \
  evaluation/libero/tests/test_merge_cosmos_4suite_results.py \
  evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py \
  evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py
```

- [ ] **Step 4: Implement suite-aware sharding and aggregation**

Extend the current stronger episode-level Cosmos validator from `(task,seed)`
to `(suite,task,seed)`. Do not replace it with the weaker reference
task-aggregate merger.

- [ ] **Step 5: Add matched model/budget matrix**

Plan teacher, Stage-1 target, Stage-2 online, and Stage-2 target at 1/1, 2/2,
4/4. Every worker command contains explicit suite, task range, model role,
video steps, action steps, checkpoint identity, and ports.

- [ ] **Step 6: Run GREEN and shell dry-run**

Run Step 3, `bash -n` on both launchers, and verify dry-run plans the complete
matrix without creating result directories.

- [ ] **Step 7: Commit**

```bash
git add evaluation/libero
git commit -m "feat: evaluate Cosmos on four matched LIBERO suites"
```

### Task 10: Add Serial Stage-1 to Stage-2 to Evaluation Orchestration

**Files:**
- Create: `distillation_flowmap/cosmos_pipeline_state.py`
- Create: `distillation_flowmap/run_cosmos_libero_train_eval_8gpu.sh`
- Create: `distillation_flowmap/tests/test_cosmos_pipeline_state.py`
- Create: `distillation_flowmap/tests/test_run_cosmos_libero_train_eval_8gpu.py`

**Interfaces:**
- Consumes: corrected Stage-1 launcher, Task 7 Stage-2 launcher, Task 9 evaluation launcher.
- Produces: atomic resumable phase manifest with exact commands and artifacts.

- [ ] **Step 1: Write failing state-machine tests**

Cover `stage1`, `stage2`, `eval`, and `all`; serial failure propagation;
resume only from attested phase output; completed-stage skip; stale/running
manifest rejection; exact checkpoint selection; and no writes in dry-run.

- [ ] **Step 2: Write failing launcher-interface tests**

Match the proven reference UX:

```text
--phase stage1|stage2|eval|all
--steps
--save-interval
--episodes
--master-port
--eval-master-port-base
--eval-ws-port-base
--run-tag
--dry-run
--check-only
```

Assert Stage-1 completes before Stage-2, Stage-2 before evaluation, and every
phase consumes the exact preceding attestation rather than guessing a path.

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_cosmos_pipeline_state.py \
  distillation_flowmap/tests/test_run_cosmos_libero_train_eval_8gpu.py
```

- [ ] **Step 4: Implement atomic phase state and wrapper**

Use command arrays and synchronous execution. Preserve outputs on failure,
record the failed command and phase, and resume only after revalidating all
contracts.

- [ ] **Step 5: Run GREEN and full write-free dry-run**

Run tests, `bash -n`, and `--phase all --dry-run`. Assert one Stage-1,
one selected Stage-2 experiment, and the complete evaluation matrix are
printed in order with no filesystem mutations.

- [ ] **Step 6: Commit**

```bash
git add distillation_flowmap/cosmos_pipeline_state.py distillation_flowmap/run_cosmos_libero_train_eval_8gpu.sh distillation_flowmap/tests
git commit -m "feat: orchestrate Cosmos train and evaluation pipeline"
```

### Task 11: Execute Ordered Verification and Produce Handoff

**Files:**
- Create: `docs/superpowers/reports/2026-07-24-cosmos-libero-alignment-verification.md`
- Modify only for verified defects: files from Tasks 1-10 with a new failing test first.

**Interfaces:**
- Consumes: all completed tasks.
- Produces: evidence-backed handoff and exact formal commands.

- [ ] **Step 1: Run unit tests and config imports**

Run all new focused suites followed by the existing Cosmos/LIBERO suites.
Record commands, pass counts, warnings, environment, commit, and wall time.
Run config imports for every experiment arm with explicit real paths.

- [ ] **Step 2: Run one-GPU one-step training smoke**

Use a fresh smoke output and save interval 1. Require logs proving:

- correct Cosmos base and Stage-1;
- correct sample count and experiment mode;
- finite main and video OPD loss;
- main backward and Cosmos video OPD backward;
- optimizer step;
- checkpoint save.

- [ ] **Step 3: Verify checkpoint resume**

Resume the saved smoke checkpoint for one additional optimizer step. Require
online/target, optimizer, scheduler, step, and parent lineage restoration.

- [ ] **Step 4: Run one-GPU inference load and one-episode closed loop**

Load the smoke target checkpoint, request explicit matched K, and run one task
for one episode. Record service log, suite/task/seed, requested and actual
video/action K, result JSON, and any saved sample video.

- [ ] **Step 5: Run eight-GPU CHECK_ONLY/dry-run**

Run every formal training variant and the full matched evaluation wrapper.
Require exactly eight ranks per job, distinct ports/outputs, complete resolved
config, and no created output.

- [ ] **Step 6: Run whole-branch verification and review**

Run `git diff --check`, shell syntax checks, the related regression suite,
and an independent whole-branch review from plan base to HEAD. Fix Critical
or Important findings through a new RED-GREEN cycle.

- [ ] **Step 7: Write the verification report**

Include:

- LingBotVA-to-Cosmos audit table;
- changed files and commits;
- all test results;
- smoke logs and checkpoint path;
- s1/s2/s4/universal eight-GPU commands;
- video-only APM commands;
- 40-task matched 1/2/4 command;
- exact resource-gated or unverified items.

- [ ] **Step 8: Commit**

```bash
git add docs/superpowers/reports/2026-07-24-cosmos-libero-alignment-verification.md
git commit -m "docs: verify Cosmos LIBERO semantic alignment"
```

