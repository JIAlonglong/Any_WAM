# Task 5 Report — Aligned Anchor/Composition Mechanism Diagnostics

## Outcome

Aligned the observation-only Cosmos mechanism probe with the shared
Student-state and same-prior Teacher endpoint protocol. No training, torchrun,
checkpoint mutation, or evaluation launch was performed.

## Files changed

- `distillation_flowmap/mechanism_diagnostics.py`
  - Renamed the internal tensor inputs to encode their actual provenance:
    Teacher continuation, same-prior Teacher endpoint, and direct/composed
    routes.
  - Preserved the existing public TensorBoard/W&B/JSONL keys.
  - Preserved separate per-sample squared-L2 and elementwise-MSE metrics.
  - Added per-sample shared-state, same-prior, and effective-eight-step
    verification scalars.
  - Preserved finite-safe ratios, near-zero/exploded flags, and packed
    finite-count aggregation.
- `distillation_flowmap/flowmap_step.py`
  - Requires the same-prior Teacher endpoint capability at deterministic
    preflight.
  - Draws one float32 video prior and one native normalized float32 action
    prior, packs the action prior for the Student, and rolls the Student to the
    calibrated shared query state.
  - Uses that detached Student-reached joint state for the Teacher same-state
    query, Student direct route, Student composed route, and Teacher
    continuation.
  - Passes the original video/native action priors to
    `predict_raw_same_prior_endpoint(..., teacher_steps=8)`.
  - Requires canonical valid frames to equal the Student state and requires
    endpoint shape/mask/effective-step provenance before constructing metrics.
  - Feeds Teacher continuation versus same-prior endpoint into `G_anchor`, and
    Student direct versus composed routes into `G_comp`.
  - Retains `predict_raw_latent_target` only for the separately named action
    context sanity metrics; it no longer supplies the anchor endpoint or query
    state.
- `distillation_flowmap/tests/test_mechanism_diagnostics.py`
  - Added exact two-element squared-L2/MSE definitions and verified the
    training endpoint remains elementwise MSE rather than aliasing `G_anchor`.
- `distillation_flowmap/tests/test_cosmos_mechanism_probes.py`
  - Added fail-closed same-prior preflight coverage.
  - Added a live caller harness proving the same video prior object reaches
    Student rollout and Teacher endpoint, native action provenance survives
    packing, the Teacher field sees a detached generated state rather than GT,
    and the metric helper receives the same-prior endpoint and three verified
    provenance values.

## TDD evidence

### RED 1 — metric definitions

The prescribed two-file command failed exactly once:

```text
1 failed, 29 passed
TypeError: compute_mechanism_metric_samples() got an unexpected keyword
argument 'teacher_continuation_video'
```

The new test hand-derived:

- `G_anchor = 25.0`;
- `G_anchor_mse = 12.5`;
- `G_comp = 13.0`;
- `G_comp_mse = 6.5`;
- training `aligned_anchor_mse = 26.0`, distinct from `G_anchor`.

### RED 2 — same-prior preflight

The focused test failed because a legacy raw Teacher without
`predict_raw_same_prior_endpoint` was accepted:

```text
Failed: DID NOT RAISE RuntimeError
```

### RED 3 — live probe provenance

The live probe harness completed the old diagnostic path but observed:

```text
assert harness.teacher.same_prior_calls == 1
E assert 0 == 1
```

This proved the real caller, not only the metric helper, still used the old
independent endpoint path.

## GREEN and regression verification

Prescribed Task 5 command:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_mechanism_diagnostics.py \
  distillation_flowmap/tests/test_cosmos_mechanism_probes.py
```

Result: `32 passed in 11.94s`.

Task 3 shared-state/backward regression command:

```bash
PYTHONPATH=.:wan_va \
  /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_cosmos_progressive_opd.py \
  distillation_flowmap/tests/test_cosmos_teacher_roles.py \
  distillation_flowmap/tests/test_cosmos_progressive_runner.py
```

Result: `44 passed in 3.20s`.

Python compilation for all four scoped source/test files and
`git diff --check` both exited zero.

## Self-review

- Public `mechanism/*` log keys and the existing TensorBoard/W&B-offline fanout
  remain compatible.
- `G_anchor` and `G_comp` are per-sample squared L2; their `_mse` companions
  remain elementwise means.
- Ratio denominators remain clamped and zero-denominator tests remain finite.
- All Student rollout and diagnostic route tensors are detached by the
  observation-only `@torch.no_grad()` probe; no training loss or backward path
  was changed.
- The same original video prior object is supplied to Student and Teacher.
  The native action prior is packed once for Student use and supplied unchanged
  to the Teacher endpoint.
- The live probe fails closed on missing same-prior capability, canonical state
  mismatch, endpoint/mask mismatch, or an effective Teacher budget other than
  eight.

## Concerns

None identified within Task 5 scope.

## Review-fix addendum

The review round corrected the endpoint units and replaced the provisional
diagnostic construction with the exact Task 3 shifted-query path:

- The same-prior worker now converts caller-owned unit-epsilon joint priors to
  the official EDM state with `x_sigma_max = packed_epsilon * 80`; request
  fingerprints remain over the exact caller video/native-action epsilons.
- Training continues to use the original Task 3 same-prior Teacher endpoint.
  The new Teacher continuation API is diagnostic-only.
- `predict_raw_joint_continuation_endpoint` transports the exact canonical
  joint state and normalized time, validates fingerprints/masks, converts
  `sigma=r/(1-r)` and `x_sigma=z_r/(1-r)`, runs each sample at its scalar
  sigma, and reports the observed eight denoiser evaluations.
- Task 3 training and Task 5 diagnostics now reuse
  `_build_cosmos_shifted_shared_query`, including production action packing.
  Diagnostic scheduling cycles only K=2 and K=4.
- `G_anchor` is Teacher continuation clean versus same-epsilon Teacher
  endpoint clean. `G_comp` is the Student direct `r→0` route versus the
  Student composed `r→s→0` route with both video and action clocks advanced.
  The legacy target is isolated to action-context sanity metrics.
- The live probe uses `B=2`, production packing, exact K=4 shared states, and
  recorded direct/composed edges. It fails closed on empty or mismatched masks
  and mismatched observed step counts, and proves legacy target changes cannot
  affect any G input.

Review-fix verification:

```text
121 passed in 12.41s
```

This covers the raw worker, mechanism probes/metrics, progressive
runner/OPD, Teacher roles, and deployment rollout suites. The changed policy
endpoint selection separately passed `13 passed, 63 deselected`.

The broader combined run produced `184 passed, 13 failed`. All 13 failures
were unrelated host dependency imports: missing `libcudnn_adv.so.9` through
Transformer Engine or missing `libero`. No failing test entered a changed
worker, adapter, diagnostic, or rollout path. `git diff --check` and Python
compilation of every modified production module both exited zero.
