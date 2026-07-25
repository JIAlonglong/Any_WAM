# Cosmos Shared Student-State Anchor/Field Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Cosmos LIBERO video OPD use one detached deployment-reached Student joint state for both endpoint and field supervision, with an exact same-prior 8-step Teacher endpoint.

**Architecture:** Add scheduler-aware trajectory/query helpers first, then extend the Cosmos raw-worker protocol with a fail-closed same-prior endpoint request. Build a single aligned video-OPD step around those contracts and wire it into the progressive objective scheduler without changing main AnyFlow or the separately controlled action-OPD objective.

**Tech Stack:** Python 3.10, PyTorch, torch.distributed, NumPy NPZ subprocess protocol, pytest, Bash/torchrun.

## Global Constraints

- Preserve the Cosmos scheduler and latent parameterization; do not copy WanVA timestep formulas or action-token layout.
- Student rollout budgets are exactly `2,4`; Teacher endpoint budget is exactly `8`.
- A legal shared query excludes terminal prior and clean endpoint and lies inside normalized Cosmos Teacher sigma band `[4/5, 80/81]`.
- Endpoint and field use exactly the same detached generated joint state `(z_r^v, z_r^a, r)`; neither may substitute GT video/action.
- Teacher endpoint must consume the exact caller-supplied terminal video/action prior; matching only seed is forbidden.
- Video endpoint/field supervise video only. Existing action OPD remains separately named, weighted, scheduled, and disabled unless the experiment explicitly enables it.
- Main AnyFlow mixed training remains unchanged.
- Endpoint and field weights are both `1.0`; video OPD runs every `4` optimizer updates.
- Contract failures fail closed before backward; non-finite decisions remain synchronized across ranks.
- Do not delete or overwrite checkpoints, output directories, or experiment records. Do not launch full training automatically.

---

### Task 1: Shifted Deployment Path and Legal Shared-Query Helpers

**Files:**
- Modify: `distillation_flowmap/danceopd_query.py`
- Test: `distillation_flowmap/tests/test_danceopd_query.py`
- Test: `distillation_flowmap/tests/test_cosmos_progressive_protocol.py`

**Interfaces:**
- Produces: `build_shifted_terminal_path(*, steps: int, shift: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor`
- Produces: `legal_cosmos_query_indices(sigmas: torch.Tensor, *, sigma_min: float = 4/5, sigma_max: float = 80/81) -> torch.Tensor`
- Produces: `sample_nonterminal_semantic_query_indices(sigmas: torch.Tensor, batch_size: int) -> torch.Tensor`
- Produces: `aligned_anchor_mse(query_video: torch.Tensor, query_sigma: torch.Tensor, student_velocity: torch.Tensor, teacher_endpoint: torch.Tensor, valid_video_mask: torch.Tensor | None = None) -> torch.Tensor`

- [ ] **Step 1: Add failing shifted-grid and legal-query tests**

```python
def test_cosmos_shifted_paths_match_inference_for_k2_and_k4():
    for steps in (2, 4):
        got = build_shifted_terminal_path(
            steps=steps, shift=5.0, device=torch.device("cpu"), dtype=torch.float64
        )
        raw = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
        expected = 5.0 * raw / (1.0 + 4.0 * raw)
        torch.testing.assert_close(got, expected)

def test_cosmos_query_indices_exclude_prior_endpoint_and_teacher_invalid_band():
    sigmas = build_shifted_terminal_path(
        steps=4, shift=5.0, device=torch.device("cpu"), dtype=torch.float32
    )
    legal = legal_cosmos_query_indices(sigmas)
    assert legal.tolist() == [1, 2]
    sampled = sample_nonterminal_semantic_query_indices(sigmas, batch_size=128)
    assert set(sampled.tolist()) <= {1, 2}
```

- [ ] **Step 2: Run RED**

Run:
```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_danceopd_query.py \
  distillation_flowmap/tests/test_cosmos_progressive_protocol.py
```
Expected: new imports/tests fail while existing tests pass.

- [ ] **Step 3: Implement exact shifted path, band filtering, Beta(5,2) sampling, and elementwise anchor MSE**

```python
def build_shifted_terminal_path(*, steps, shift, device, dtype):
    if steps < 1 or not math.isfinite(shift) or shift <= 0:
        raise ValueError("steps must be positive and shift must be finite and > 0")
    raw = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    return shift * raw / (1.0 + (shift - 1.0) * raw)

def legal_cosmos_query_indices(sigmas, *, sigma_min=4 / 5, sigma_max=80 / 81):
    indices = torch.arange(sigmas.numel(), device=sigmas.device)
    mask = (indices > 0) & (indices < sigmas.numel() - 1)
    mask &= (sigmas >= sigma_min) & (sigmas <= sigma_max)
    legal = indices[mask]
    if legal.numel() == 0:
        raise ValueError("no nonterminal query lies inside the Cosmos teacher band")
    return legal

def sample_nonterminal_semantic_query_indices(sigmas, batch_size):
    legal = legal_cosmos_query_indices(sigmas)
    beta = torch.distributions.Beta(
        torch.tensor(5.0, device=sigmas.device),
        torch.tensor(2.0, device=sigmas.device),
    )
    draws = beta.sample((batch_size,))
    slots = torch.clamp((draws * legal.numel()).long(), max=legal.numel() - 1)
    return legal[slots]
```

`aligned_anchor_mse` must broadcast `query_sigma`, compute `query_video - sigma * student_velocity`, apply only the optional valid-video mask, and call `F.mse_loss(..., reduction="mean")`. Teacher endpoint is detached at the call site.

- [ ] **Step 4: Run GREEN and verify K=2 has one legal query and K=4 has two**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add distillation_flowmap/danceopd_query.py \
  distillation_flowmap/tests/test_danceopd_query.py \
  distillation_flowmap/tests/test_cosmos_progressive_protocol.py
git commit -m "feat: add Cosmos aligned query geometry"
```

### Task 2: Exact Same-Prior 8-Step Teacher Endpoint Protocol

**Files:**
- Modify: `distillation_flowmap/cosmos_policy_adapter.py`
- Modify: `distillation_flowmap/cosmos_policy_raw_worker.py`
- Test: `distillation_flowmap/tests/test_cosmos_policy_backend.py`
- Test: `distillation_flowmap/tests/test_cosmos_raw_worker_runtime.py`

**Interfaces:**
- Produces adapter method:
  `predict_raw_same_prior_endpoint(raw_batch, *, video_prior: torch.Tensor, action_prior: torch.Tensor, teacher_steps: int = 8) -> dict[str, Any]`
- Produces worker request mode: `same_prior_endpoint`
- Response keys: `endpoint_video`, `video_frame_mask`, `effective_teacher_steps`, `video_prior_sha256`, `action_prior_sha256`

- [ ] **Step 1: Add failing adapter protocol tests**

```python
def test_same_prior_endpoint_request_round_trips_exact_priors_and_eight_steps(monkeypatch):
    video_prior = torch.randn(1, 16, 8, 8)
    action_prior = torch.randn(1, 16, 7)
    captured = {}
    def fake_worker(payload, arrays):
        captured.update(payload)
        torch.testing.assert_close(torch.from_numpy(arrays["video_prior"]), video_prior)
        torch.testing.assert_close(torch.from_numpy(arrays["action_prior"]), action_prior)
        return {
            "endpoint_video": np.zeros_like(arrays["video_prior"]),
            "video_frame_mask": np.ones(video_prior.shape[1], dtype=bool),
            "effective_teacher_steps": 8,
            "video_prior_sha256": sha256_array(arrays["video_prior"]),
            "action_prior_sha256": sha256_array(arrays["action_prior"]),
        }
    result = adapter.predict_raw_same_prior_endpoint(
        batch, video_prior=video_prior, action_prior=action_prior, teacher_steps=8
    )
    assert captured["mode"] == "same_prior_endpoint"
    assert result["effective_teacher_steps"] == 8
```

Also test:
- `teacher_steps != 8` raises before spawning the worker;
- mismatched returned video/action SHA256 raises;
- effective step count other than 8 raises;
- worker backend without an explicit initial-state sampler raises and never falls back to a seed.

- [ ] **Step 2: Run RED**

Run:
```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_cosmos_policy_backend.py \
  distillation_flowmap/tests/test_cosmos_raw_worker_runtime.py
```
Expected: new adapter method and worker mode are absent.

- [ ] **Step 3: Implement deterministic request serialization and validation**

Add a shared contiguous-array fingerprint:
```python
def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()
```

The adapter writes both priors into the existing temporary NPZ request, sets `mode="same_prior_endpoint"` and `teacher_steps=8`, validates every response key, exact step count, exact fingerprints, finite endpoint, and boolean frame mask. Any mismatch raises `RuntimeError` before returning tensors.

- [ ] **Step 4: Implement worker sampler from the supplied canonical joint prior**

The worker must:
1. load and validate both arrays;
2. inject normalized `action_prior` into the Cosmos action carrier without changing valid video frames;
3. construct the canonical joint prior;
4. call the loaded Cosmos policy model's sampler directly with that tensor as the explicit initial state and `num_steps=8`;
5. reject model classes/signatures that cannot prove they consume an explicit initial state;
6. return decoded endpoint video, mask, effective count, and fingerprints.

Use a dedicated helper:
```python
def _generate_from_explicit_prior(model, data_batch, joint_prior, *, teacher_steps):
    if teacher_steps != 8:
        raise ValueError("same-prior endpoint requires exactly 8 teacher steps")
    generate = model.generate_samples_from_batch
    signature = inspect.signature(generate)
    if "x_sigma_max" not in signature.parameters:
        raise RuntimeError("Cosmos backend cannot honor explicit same-prior sampling")
    return generate(
        data_batch,
        x_sigma_max=joint_prior,
        num_steps=teacher_steps,
    )
```

Do not accept a generic `**kwargs` parameter as proof. Verify the actual runtime class has a named explicit-prior parameter or add an adapter for its documented direct sampler with an equivalent named initial-state argument.

- [ ] **Step 5: Run GREEN plus worker import preflight**

Run the Step 2 command, then:
```bash
COSMOS_PREDICT2_REPO=/kpfs-intern/jialongliu/projects/cosmos-predict2.5-formal-compat-441b897 \
  bash distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh \
  --phase check --run-tag aligned-anchor-worker-check
```
Expected: tests pass and preflight reports a backend capable of explicit-prior 8-step sampling. If the loaded backend lacks it, stop with the designed clear error; do not weaken the contract.

- [ ] **Step 6: Commit**

```bash
git add distillation_flowmap/cosmos_policy_adapter.py \
  distillation_flowmap/cosmos_policy_raw_worker.py \
  distillation_flowmap/tests/test_cosmos_policy_backend.py \
  distillation_flowmap/tests/test_cosmos_raw_worker_runtime.py
git commit -m "feat: add Cosmos same-prior teacher endpoint"
```

### Task 3: Shared Student-State Cosmos Video OPD Step

**Files:**
- Modify: `distillation_flowmap/flowmap_step.py`
- Test: `distillation_flowmap/tests/test_cosmos_progressive_opd.py`
- Test: `distillation_flowmap/tests/test_cosmos_teacher_roles.py`
- Test: `distillation_flowmap/tests/test_cosmos_progressive_runner.py`

**Interfaces:**
- Consumes Task 1 helpers and `teacher.predict_raw_same_prior_endpoint(...)` from Task 2.
- Produces: `_cosmos_aligned_video_opd_step(self, batch, batch_idx) -> dict[str, Any]`
- Result keys: `loss`, `opd_endpoint_loss`, `opd_same_state_velocity_loss`, `opd_endpoint_contrib`, `opd_same_state_velocity_contrib`, `opd_query_sigma`, `opd_query_index`, `opd_student_steps`, `opd_teacher_steps`, `opd_valid_video_frames`, `opd_same_prior_verified`, `opd_canonical_state_verified`

- [ ] **Step 1: Add RED harness proving endpoint and field share generated joint state**

Build a small fake Student that records every video/action input and a fake Teacher that returns a canonical query plus same-prior endpoint. Assert:
```python
assert torch.equal(field_student_call.video, anchor_student_call.video)
assert torch.equal(field_student_call.action, anchor_student_call.action)
assert field_student_call.video.data_ptr() == anchor_student_call.video.data_ptr()
assert not field_student_call.video.requires_grad
assert not field_student_call.action.requires_grad
assert not torch.equal(field_student_call.video, batch["gt_video"])
assert not torch.equal(field_student_call.action, batch["gt_action"])
```

Perturb one saved rollout state and assert both the field query input and direct anchor input change. Assert action prediction at every rollout step receives the current generated video. Assert no action endpoint/field loss key exists.

- [ ] **Step 2: Run RED**

Run:
```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_cosmos_progressive_opd.py \
  distillation_flowmap/tests/test_cosmos_teacher_roles.py \
  distillation_flowmap/tests/test_cosmos_progressive_runner.py
```
Expected: aligned method is absent.

- [ ] **Step 3: Implement no-grad deployment-faithful joint rollout**

Uniformly select synchronized `K` from `(2,4)`, build video/action paths with their resolved shifts, draw terminal video/action priors once, and run exactly K joint updates under `torch.no_grad()`. Store post-update joint states. Select one Task 1 legal video index per sample; map the same normalized progress onto the action path. Gather and detach `z_r^v`, `z_r^a`, and `r`.

Every Student video and action call receives the current generated video/action for noisy and self-conditioning inputs. Do not read GT future video/action after initial batch conditioning has been prepared.

- [ ] **Step 4: Implement canonical same-state field and direct anchor**

Call `predict_raw_joint_latent_velocity` with `(z_r^v,z_r^a,r,r)`. Verify returned valid-video frames are elementwise equal to the selected Student frames before using the returned canonical joint state. Use that exact canonical tensor object for:
- Student field forward at `r→r`;
- Teacher field target;
- Student trainable anchor forward at `r→0`.

Call `predict_raw_same_prior_endpoint` with the original video/action priors and require Teacher steps=8. Compute:
```python
field = masked_video_velocity_mse(student_field, teacher_field.detach(), mask)
endpoint = aligned_anchor_mse(
    canonical_video, r, student_anchor_velocity,
    teacher_endpoint.detach(), mask,
)
loss = endpoint + field
```
Only the two Student query forwards retain gradients.

- [ ] **Step 5: Add finite/provenance diagnostics and synchronized failure behavior**

Populate all result keys in the interface. Use the existing distributed finite-decision helper before backward. Contract mismatches raise on every rank using the existing synchronized fatal-error mechanism; finite loss skips use the existing synchronized skip path.

- [ ] **Step 6: Run GREEN and backward assertions**

Run the Step 2 command. Expected: PASS, and the harness verifies nonzero gradients only on Student query parameters.

- [ ] **Step 7: Commit**

```bash
git add distillation_flowmap/flowmap_step.py \
  distillation_flowmap/tests/test_cosmos_progressive_opd.py \
  distillation_flowmap/tests/test_cosmos_teacher_roles.py \
  distillation_flowmap/tests/test_cosmos_progressive_runner.py
git commit -m "feat: align Cosmos anchor and field state"
```

### Task 4: Progressive Scheduling, Configuration, Launcher, and Logging

**Files:**
- Modify: `distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py`
- Modify: `distillation_flowmap/flowmap_trainer.py`
- Modify: `distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh`
- Test: `distillation_flowmap/tests/test_cosmos_progressive_config.py`
- Test: `distillation_flowmap/tests/test_cosmos_progressive_protocol.py`
- Test: `distillation_flowmap/tests/test_cosmos_stage1_stage2_eval_runner.py`

**Interfaces:**
- Consumes `_cosmos_aligned_video_opd_step`.
- Resolved fields: `opd_aux_interval=4`, `opd_danceopd_rollout_steps=(2,4)`, `opd_danceopd_anchor_teacher_steps=8`, `opd_danceopd_endpoint_weight=1.0`, `opd_danceopd_velocity_weight=1.0`.

- [ ] **Step 1: Add failing schedule and resolved-config tests**

Test objective selection over at least 16 optimizer steps:
```python
assert objective_at(4) == "aligned_video_opd"
assert objective_at(8) == "aligned_video_opd"
assert objective_at(1) == "main_anyflow"
```
Add the existing action-OPD enabled variant and assert action updates retain their separate objective name and never silently add action loss to `aligned_video_opd`.

Dry-run assertions must include the exact five resolved fields above, `ACTION_DOWNSAMPLE_FACTOR=1`, `VIDEO_ACTION_BRIDGE=0`, and one `--nproc_per_node=8`.

- [ ] **Step 2: Run RED**

Run:
```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_cosmos_progressive_config.py \
  distillation_flowmap/tests/test_cosmos_progressive_protocol.py \
  distillation_flowmap/tests/test_cosmos_stage1_stage2_eval_runner.py
```

- [ ] **Step 3: Replace the old incompatible video endpoint route**

In the trainer objective selector, route scheduled video-OPD updates only to `_cosmos_aligned_video_opd_step`. Do not also call `_cosmos_deployment_joint_rollout_step` or the independent endpoint portion of `_cosmos_latent_full_opd_aux_transition_step`. Preserve main AnyFlow and the explicit action-OPD arm.

- [ ] **Step 4: Validate config and expose complete resolved values**

Reject rollout budgets outside `{2,4}`, Teacher budget other than `8`, nonpositive weights, interval other than a positive integer, or any grid without a legal Teacher-band query. Print these values in CHECK_ONLY/dry-run before torchrun. Dry-run must not create output directories.

- [ ] **Step 5: Aggregate and log raw/weighted/ratio metrics**

Add TensorBoard and W&B-offline keys:
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
Clamp ratio denominators with the existing epsilon helper.

- [ ] **Step 6: Run GREEN and two dry-runs**

Run Step 2, then:
```bash
bash distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh \
  --phase check --run-tag aligned-anchor-check
bash distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh \
  --phase all --steps 1 --save-interval 1 --episodes 1 \
  --run-tag aligned-anchor-dry --dry-run
```
Expected: tests pass; each dry-run prints one 8-rank command, creates no output, and does not start training.

- [ ] **Step 7: Commit**

```bash
git add distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py \
  distillation_flowmap/flowmap_trainer.py \
  distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh \
  distillation_flowmap/tests/test_cosmos_progressive_config.py \
  distillation_flowmap/tests/test_cosmos_progressive_protocol.py \
  distillation_flowmap/tests/test_cosmos_stage1_stage2_eval_runner.py
git commit -m "feat: schedule aligned Cosmos video OPD"
```

### Task 5: Align G_anchor/G_comp Mechanism Diagnostics

**Files:**
- Modify: `distillation_flowmap/mechanism_diagnostics.py`
- Modify: `distillation_flowmap/flowmap_step.py`
- Test: `distillation_flowmap/tests/test_mechanism_diagnostics.py`
- Test: `distillation_flowmap/tests/test_cosmos_mechanism_probes.py`

**Interfaces:**
- Consumes the shared query state, same-prior Teacher endpoint, Teacher continuation from the shared state, and direct/composed routes.
- Produces existing public keys `mechanism/g_anchor`, `mechanism/g_anchor_mse`, `mechanism/g_comp`, `mechanism/g_comp_mse`, `mechanism/g_anchor_to_comp_ratio` plus provenance flags.

- [ ] **Step 1: Add RED definition tests**

Use two-element samples so squared-L2 and MSE differ:
```python
teacher_continuation = torch.tensor([[1.0, 2.0]])
same_prior_endpoint = torch.tensor([[4.0, 6.0]])
samples = compute_mechanism_samples(...)
assert samples["mechanism/g_anchor"].item() == 25.0
assert samples["mechanism/g_anchor_mse"].item() == 12.5
```
Assert training `opd_endpoint_loss` remains elementwise MSE and is not reused as `g_anchor`. Add `g_comp` direct-versus-composed route assertions and denominator-clamp finite assertions.

- [ ] **Step 2: Run RED**

Run:
```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_mechanism_diagnostics.py \
  distillation_flowmap/tests/test_cosmos_mechanism_probes.py
```

- [ ] **Step 3: Feed aligned provenance into diagnostics**

Compute `G_anchor` from Teacher continuation of the shared `z_r` versus the same-prior Teacher endpoint. Compute `G_comp` from the defined direct and composed Teacher routes. Keep squared-L2 and elementwise-MSE values separate. Add finite-safe ratio, near-zero/exploded flags, and provenance scalars for shared-state, same-prior, and effective 8-step verification.

- [ ] **Step 4: Run GREEN**

Run Step 2. Expected: PASS with finite metrics for zero denominators.

- [ ] **Step 5: Commit**

```bash
git add distillation_flowmap/mechanism_diagnostics.py \
  distillation_flowmap/flowmap_step.py \
  distillation_flowmap/tests/test_mechanism_diagnostics.py \
  distillation_flowmap/tests/test_cosmos_mechanism_probes.py
git commit -m "feat: align Cosmos anchor composition diagnostics"
```

### Task 6: End-to-End Verification Without Full Training

**Files:**
- Modify only if a verification-discovered defect requires a test-first fix.
- Evidence: preserve smoke logs beneath a new, nonexisting run tag output only when an allocated GPU is available.

**Interfaces:**
- Consumes all prior tasks.
- Produces verified one-step checkpoint/reload evidence and final manual launch command.

- [ ] **Step 1: Run the complete focused unit/config suite**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_danceopd_query.py \
  distillation_flowmap/tests/test_cosmos_policy_backend.py \
  distillation_flowmap/tests/test_cosmos_raw_worker_runtime.py \
  distillation_flowmap/tests/test_cosmos_progressive_opd.py \
  distillation_flowmap/tests/test_cosmos_teacher_roles.py \
  distillation_flowmap/tests/test_cosmos_progressive_runner.py \
  distillation_flowmap/tests/test_cosmos_progressive_config.py \
  distillation_flowmap/tests/test_cosmos_progressive_protocol.py \
  distillation_flowmap/tests/test_cosmos_stage1_stage2_eval_runner.py \
  distillation_flowmap/tests/test_mechanism_diagnostics.py \
  distillation_flowmap/tests/test_cosmos_mechanism_probes.py
```
Expected: all pass.

- [ ] **Step 2: Run syntax, import, and diff checks**

```bash
bash -n distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -c \
  'import distillation_flowmap.config_libero_cosmos_policy_stage2_progressive'
git diff --check
git status --short --branch
```

- [ ] **Step 3: Run one-GPU, one-step Stage-2 smoke only when a GPU is allocated**

Use a new run tag and existing Stage-1 checkpoint; set steps/save interval to one. Require log evidence for:
```text
objective=aligned_video_opd
finite opd_endpoint_loss
finite opd_same_state_velocity_loss
same_prior_verified=1
canonical_state_verified=1
student_steps in {2,4}
teacher_steps=8
main backward complete
optimizer step complete
checkpoint saved
```
If no GPU is allocated, record this step as externally blocked and do not substitute a dry-run claim.

- [ ] **Step 4: Reload the one-step checkpoint and run single-GPU inference smoke**

Require successful metadata restoration and matched requested video/action budget. Run one LIBERO task/one episode only if simulator resources are available. Preserve the generated smoke video and JSON; do not claim 40-task evaluation.

- [ ] **Step 5: Run final 8-GPU CHECK_ONLY/dry-run**

```bash
COSMOS_PREDICT2_REPO=/kpfs-intern/jialongliu/projects/cosmos-predict2.5-formal-compat-441b897 \
  bash distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh \
  --phase all --steps 10000 --save-interval 1000 --episodes 500 \
  --run-tag cosmos-aligned-anchor-field-20260725 --dry-run
```
Expected: Stage-1 → Stage-2 → matched 1/2/4 Student and Teacher evaluation chain is printed, every training command uses one 8-rank torchrun, no process starts, and no output directory is created.

- [ ] **Step 6: Request final whole-branch review and report only verified evidence**

Report commits, files, exact test counts, smoke checkpoint/log paths, dry-run output, and any resource-blocked items. Do not call the work complete if same-prior backend preflight or one-step backward has not run successfully.
