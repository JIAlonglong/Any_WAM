# Cosmos Stage-1 → Stage-2 → Evaluation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one fail-closed, resumable 8×A800 command that verifies the corrected Cosmos training contract, trains raw Stage-1 and progressive `universal-video-action` Stage-2 for 5,000 steps each, then evaluates joint K=1/2/4 for 500 LIBERO episodes per K.

**Architecture:** Version the corrected action packing and deployment rollout contract in focused Python helpers, make checkpoints attest that contract, and keep raw-teacher-window supervision separate from the full 1000→0 deployment objective. A small Python state/manifest module supplies atomic validation to a Bash orchestrator that invokes dedicated Stage-1, Stage-2, and resumable evaluation launchers in one long-lived eight-GPU allocation.

**Tech Stack:** Python 3.10, PyTorch, pytest, Bash, torchrun/FSDP1, safetensors, Cosmos Predict2.5 worker subprocesses, LIBERO.

## Global Constraints

- Live implementation work must not start GPU training or LIBERO evaluation; use CPU tests, stubs, syntax checks, and dry-runs only.
- The clean Stage-1 base is `/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero`.
- The contaminated `raw_stage1_5000`, its descendants, optimizer state, and scheduler state must never be accepted by the corrected pipeline.
- `contract_version` is exactly `2`.
- `action_packing_schema` is exactly `downsample_survivor_v2`.
- Raw actions 0–15 must survive production converter → `F::4` → production decoder in order.
- The post-downsample action chunk shape is exactly `[F=4,N=4]`.
- Stage-2 full deployment rollout starts at raw timestep `1000`, ends at `0`, and uses joint K choices exactly `[1,2,4]`.
- Full deployment rollout runs where `step % 4 == 0`; K follows a balanced deterministic cycle.
- Raw Cosmos auxiliary rollout remains inside `[0.8,80/81]` and runs after warmup where `step % 8 == 2`.
- Deployment and raw-window rollout graphs must never execute on the same optimizer step.
- Stage-1 and Stage-2 both use all eight visible GPUs, seed `42`, and `5,000` optimizer steps.
- Default torchrun ports are Stage-1 `29671` and Stage-2 `29672`.
- Evaluation is sequential K=1→2→4, 500 records per K, four shards across all eight GPUs, video seeds `0,1`, and 60 saved videos total.
- Launcher structure follows the proven
  `run_libero_video_opd_train_eval_8gpu.sh` pattern: command arrays, shared
  checkout root resolution, strict numeric/port/GPU validation, `--phase
  train|eval|all`, `--run-tag`, resolved dry-run output, and synchronous
  train-before-eval failure propagation.
- `dry-run` creates no files/directories and starts no child process; `status` is read-only.
- A different immutable run identity, incomplete checkpoint, failed contract, child failure, or signal must prevent every later stage.
- State files, completion manifests, attestations, and summaries use sibling temporary files plus atomic `os.replace`.

---

## File Responsibility Map

- `distillation_flowmap/cosmos_training_contract.py`: contract constants, survivor-layout action packing, metadata validation.
- `distillation_flowmap/cosmos_policy_adapter.py`: public raw-action converter delegates packing to the contract helper.
- `distillation_flowmap/cosmos_deployment_rollout.py`: pure deployment schedule and endpoint-loss helpers.
- `distillation_flowmap/flowmap_step.py`: joint 1000→0 Stage-2 rollout implementation.
- `distillation_flowmap/flowmap_trainer.py`: mutually exclusive deployment/raw-aux scheduling and checkpoint metadata persistence.
- `distillation_flowmap/config_libero_cosmos_policy_stage1.py`: Stage-1 packing contract.
- `distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py`: Stage-2 deployment/raw-aux contract and weights.
- `distillation_flowmap/verify_cosmos_joint_training_contract.py`: executable CPU attestation producer.
- `distillation_flowmap/cosmos_pipeline_state.py`: atomic run identity, checkpoint, completion, and status validation.
- `distillation_flowmap/run_cosmos_raw_stage1_8gpu.sh`: corrected Stage-1 dry-run/run/resume launcher.
- `distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh`: corrected Stage-2 contract/resume enforcement.
- `evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh`: resumable K matrix.
- `distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh`: top-level state machine.
- Focused tests live under `distillation_flowmap/tests/` and `evaluation/libero/tests/`.

---

### Task 1: Version and Correct the 16-Action Packing

**Files:**
- Create: `distillation_flowmap/cosmos_training_contract.py`
- Modify: `distillation_flowmap/cosmos_policy_adapter.py:169-216`
- Modify: `distillation_flowmap/build_cosmos_progressive_teacher_cache.py`
- Modify: `distillation_flowmap/eval_cosmos_policy_stage1_metrics.py`
- Modify: `distillation_flowmap/flowmap_step.py`
- Modify: `distillation_flowmap/config_libero_cosmos_policy_stage1.py`
- Modify: `distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py`
- Modify: `evaluation/libero/rollout_cosmos_progressive_s4.py:322-352`
- Test: `distillation_flowmap/tests/test_cosmos_policy_backend.py`
- Test: `evaluation/libero/tests/test_cosmos_progressive_s4_service.py`

**Interfaces:**
- Produces: `CONTRACT_VERSION = 2`.
- Produces: `ACTION_PACKING_SCHEMA = "downsample_survivor_v2"`.
- Produces: `pack_actions_for_downsample(aligned: torch.Tensor, target_shape: tuple[int, int, int, int, int], *, downsample_factor: int, schema: str) -> torch.Tensor`.
- Consumes: raw aligned actions shaped `[B,T,C]`.
- Guarantees: after `[:, :, ::4]`, all 16 raw actions occupy `[F=4,N=4]` in order.

- [ ] **Step 1: Write the failing identity and validation tests**

Add a value-sensitive test, not a shape-only test:

```python
def test_cosmos_action_packing_v2_preserves_all_sixteen_actions_after_downsample():
    actions = torch.arange(1, 16 * 7 + 1, dtype=torch.float32).reshape(1, 16, 7)
    q01 = torch.zeros(30)
    q99 = torch.ones(30) * 200
    inverse = list(range(7)) + [7] * 23

    full = cosmos_actions_to_flowmap_x0(
        actions,
        target_shape=(1, 30, 16, 4, 1),
        q01=q01,
        q99=q99,
        inverse_used_action_channel_ids=inverse,
        device=torch.device("cpu"),
        dtype=torch.float32,
        packing_schema="downsample_survivor_v2",
        downsample_factor=4,
    )
    compact = full[:, :, ::4]
    expected = ((actions - q01[:7]) / (q99[:7] - q01[:7] + 1e-6) * 2 - 1)
    observed = compact[:, :7, :, :, 0].permute(0, 2, 3, 1).reshape(1, 16, 7)

    assert torch.allclose(observed, expected)
    assert full[:, :7, 1:4].abs().sum() == 0
```

Also test:

```python
@pytest.mark.parametrize("schema", ["", "legacy_dense_v1", "unknown"])
def test_corrected_cosmos_training_rejects_non_v2_packing(schema):
    with pytest.raises(ValueError, match="action packing schema"):
        pack_actions_for_downsample(
            torch.ones(1, 16, 7),
            (1, 30, 16, 4, 1),
            downsample_factor=4,
            schema=schema,
        )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  distillation_flowmap/tests/test_cosmos_policy_backend.py \
  evaluation/libero/tests/test_cosmos_progressive_s4_service.py \
  -k "packing_v2 or non_v2_packing"
```

Expected: failures because the schema helper and converter arguments do not exist and the old dense packing loses actions 4–15.

- [ ] **Step 3: Implement the survivor-layout helper**

Create:

```python
from __future__ import annotations

import torch

CONTRACT_VERSION = 2
ACTION_PACKING_SCHEMA = "downsample_survivor_v2"


def pack_actions_for_downsample(
    aligned: torch.Tensor,
    target_shape: tuple[int, int, int, int, int],
    *,
    downsample_factor: int,
    schema: str,
) -> torch.Tensor:
    if schema != ACTION_PACKING_SCHEMA:
        raise ValueError(
            f"action packing schema must be {ACTION_PACKING_SCHEMA!r}, got {schema!r}"
        )
    batch, channels, frames, per_frame, width = target_shape
    if width != 1 or downsample_factor <= 0:
        raise ValueError("action carrier requires width=1 and a positive downsample factor")
    if frames % downsample_factor != 0:
        raise ValueError("action carrier frames must be divisible by downsample_factor")
    compact_frames = frames // downsample_factor
    capacity = compact_frames * per_frame
    if aligned.shape != (batch, capacity, channels):
        raise ValueError(
            f"expected aligned actions {(batch, capacity, channels)}, got {tuple(aligned.shape)}"
        )

    packed = torch.zeros(
        batch, channels, frames, per_frame, width,
        device=aligned.device, dtype=aligned.dtype,
    )
    compact = aligned.reshape(batch, compact_frames, per_frame, channels)
    packed[:, :, ::downsample_factor, :, 0] = compact.permute(0, 3, 1, 2)
    return packed
```

Change `cosmos_actions_to_flowmap_x0` to accept keyword-only
`packing_schema` and `downsample_factor`, normalize/channel-align as before,
require exactly the compact capacity, and return `pack_actions_for_downsample`.
Do not retain the old `flat[:, :num_tokens]` implementation on corrected
training paths.

Set both Cosmos training configs:

```python
cfg.contract_version = 2
cfg.action_packing_schema = "downsample_survivor_v2"
cfg.action_downsample_factor = 4
```

Pass both fields at every production converter call in Stage-1, Stage-2,
offline cache/evaluation, and live evaluation. In
`FlowMapActionAnchorEncoder.__call__`, pass
`packing_schema=self.config.action_packing_schema` and
`downsample_factor=self.config.action_downsample_factor`; do not rely on
converter defaults.

- [ ] **Step 4: Run packing and service regressions**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  distillation_flowmap/tests/test_cosmos_policy_backend.py \
  distillation_flowmap/tests/test_cosmos_teacher_roles.py \
  evaluation/libero/tests/test_cosmos_progressive_s4_service.py
```

Expected: all tests pass; the value-sensitive 16-action identity test passes.

- [ ] **Step 5: Commit Task 1**

```bash
git add \
  distillation_flowmap/cosmos_training_contract.py \
  distillation_flowmap/cosmos_policy_adapter.py \
  distillation_flowmap/build_cosmos_progressive_teacher_cache.py \
  distillation_flowmap/eval_cosmos_policy_stage1_metrics.py \
  distillation_flowmap/flowmap_step.py \
  distillation_flowmap/config_libero_cosmos_policy_stage1.py \
  distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py \
  distillation_flowmap/tests/test_cosmos_policy_backend.py \
  distillation_flowmap/tests/test_cosmos_teacher_roles.py \
  evaluation/libero/rollout_cosmos_progressive_s4.py \
  evaluation/libero/tests/test_cosmos_progressive_s4_service.py
git commit -m "fix: preserve all Cosmos LIBERO actions"
```

---

### Task 2: Add Pure Deployment-Rollout Contracts and Losses

**Files:**
- Create: `distillation_flowmap/cosmos_deployment_rollout.py`
- Modify: `distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py`
- Test: `distillation_flowmap/tests/test_cosmos_deployment_rollout.py`
- Test: `distillation_flowmap/tests/test_cosmos_progressive_config.py`

**Interfaces:**
- Produces: `DEPLOYMENT_STUDENT_STEPS = (1, 2, 4)`.
- Produces: `deployment_joint_step_for_update(update_index: int) -> int`.
- Produces: `should_run_deployment_joint_rollout(step: int, interval: int = 4) -> bool`.
- Produces: `should_run_raw_auxiliary(step: int, *, warmup: int, interval: int = 8, phase: int = 2) -> bool`.
- Produces: `deployment_endpoint_losses(video_final, video_x0, action_final, action_x0, action_mask, *, action_weight) -> dict[str, torch.Tensor]`.

- [ ] **Step 1: Write failing schedule and loss tests**

Create tests:

```python
def test_deployment_k_cycle_is_balanced_and_deterministic():
    assert [deployment_joint_step_for_update(i) for i in range(9)] == [
        1, 2, 4, 1, 2, 4, 1, 2, 4
    ]


def test_deployment_and_raw_auxiliary_never_share_an_optimizer_step():
    deployment = {
        step for step in range(64)
        if should_run_deployment_joint_rollout(step, interval=4)
    }
    raw_aux = {
        step for step in range(64)
        if should_run_raw_auxiliary(step, warmup=8, interval=8, phase=2)
    }
    assert deployment
    assert raw_aux
    assert deployment.isdisjoint(raw_aux)


def test_deployment_endpoint_loss_uses_final_state_at_sigma_zero():
    video_x0 = torch.ones(1, 2, 1, 1, 1)
    action_x0 = torch.ones(1, 3, 4, 4, 1)
    losses = deployment_endpoint_losses(
        video_final=video_x0 + 2,
        video_x0=video_x0,
        action_final=action_x0 + 3,
        action_x0=action_x0,
        action_mask=torch.ones(1, 1, 4, 4, 1),
        action_weight=1.0,
    )
    assert losses["video"] == pytest.approx(torch.tensor(4.0))
    assert losses["action"] == pytest.approx(torch.tensor(9.0))
    assert losses["total"] == pytest.approx(torch.tensor(13.0))
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  distillation_flowmap/tests/test_cosmos_deployment_rollout.py \
  distillation_flowmap/tests/test_cosmos_progressive_config.py
```

Expected: import failures for the new deployment module and config assertions.

- [ ] **Step 3: Implement the pure module and config**

Implement:

```python
from __future__ import annotations

import torch

DEPLOYMENT_STUDENT_STEPS = (1, 2, 4)


def deployment_joint_step_for_update(update_index: int) -> int:
    if update_index < 0:
        raise ValueError("update_index must be non-negative")
    return DEPLOYMENT_STUDENT_STEPS[update_index % len(DEPLOYMENT_STUDENT_STEPS)]


def should_run_deployment_joint_rollout(step: int, interval: int = 4) -> bool:
    return step >= 0 and interval > 0 and step % interval == 0


def should_run_raw_auxiliary(
    step: int, *, warmup: int, interval: int = 8, phase: int = 2
) -> bool:
    if interval <= 0 or not 0 <= phase < interval:
        raise ValueError("raw auxiliary interval/phase are invalid")
    return step >= warmup and step % interval == phase


def deployment_endpoint_losses(
    video_final: torch.Tensor,
    video_x0: torch.Tensor,
    action_final: torch.Tensor,
    action_x0: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    action_weight: float,
) -> dict[str, torch.Tensor]:
    if action_weight < 0:
        raise ValueError("action_weight must be non-negative")
    video = (video_final.float() - video_x0.detach().float()).square().mean()
    mask = action_mask.detach().float()
    diff = (action_final.float() - action_x0.detach().float()) * mask
    denom = (mask.sum() * action_final.shape[1]).clamp(min=1)
    action = diff.square().sum() / denom
    return {"video": video, "action": action, "total": video + action_weight * action}
```

Set Stage-2 config:

```python
cfg.deployment_joint_rollout_enabled = True
cfg.deployment_joint_rollout_interval = 4
cfg.deployment_joint_steps = (1, 2, 4)
cfg.deployment_timestep_start = 1000
cfg.deployment_timestep_end = 0
cfg.deployment_action_weight = 1.0
cfg.raw_teacher_window_is_auxiliary = True
cfg.opd_aux_interval = 8
cfg.opd_aux_phase = 2
```

- [ ] **Step 4: Run pure/config tests**

Run the same command as Step 2.

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add \
  distillation_flowmap/cosmos_deployment_rollout.py \
  distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py \
  distillation_flowmap/tests/test_cosmos_deployment_rollout.py \
  distillation_flowmap/tests/test_cosmos_progressive_config.py
git commit -m "feat: define full Cosmos deployment rollout contract"
```

---

### Task 3: Integrate the Full 1000→0 Joint Deployment Objective

**Files:**
- Modify: `distillation_flowmap/flowmap_step.py:2999-3250,4232-4575,5468-5505`
- Modify: `distillation_flowmap/flowmap_trainer.py:2540-2650`
- Modify: `distillation_flowmap/cosmos_progressive_opd.py`
- Test: `distillation_flowmap/tests/test_cosmos_progressive_opd.py`
- Test: `distillation_flowmap/tests/test_cosmos_progressive_protocol.py`
- Test: `distillation_flowmap/tests/test_cosmos_progressive_runner.py`

**Interfaces:**
- Produces: `FlowMapStepMixin._cosmos_deployment_joint_rollout_step(batch, batch_idx, *, student_steps) -> dict[str, torch.Tensor]`.
- Consumes: corrected packed `batch["actions"]` and clean Cosmos `video_x0`
  returned in `["cosmos_latent_x0"]` by the exact endpoint-only production
  call shown in Step 3 with `include_cdiff=False`,
  one video noise, and one action noise.
- Produces metrics: `deployment/video_endpoint_loss`, `deployment/action_endpoint_loss`, `deployment/total_loss`, `deployment/student_steps`, `deployment/t_start`, `deployment/t_end`.

- [ ] **Step 1: Write failing full-path integration and scheduling tests**

Add a runner contract test whose recording harness asserts:

```python
result = harness._cosmos_deployment_joint_rollout_step(
    batch, 0, student_steps=2
)
call = harness.integrate_calls[0]
assert torch.equal(call["timesteps"], torch.full((1, 9), 1000.0))
assert torch.equal(call["target_r"], torch.zeros(1, 9))
assert torch.equal(call["action_target_r"], torch.zeros(1, 16))
assert call["K_steps"] == 2
assert call["return_final_action_state"] is True
assert result["deployment_t_start"].item() == 1000
assert result["deployment_t_end"].item() == 0
```

Add trainer schedule tests with sentinel methods:

```python
def test_trainer_runs_deployment_and_raw_aux_on_disjoint_steps():
    assert scheduled_kind(step=8) == "deployment"
    assert scheduled_kind(step=10) == "raw_auxiliary"
    assert scheduled_kind(step=12) == "deployment"
    assert scheduled_kind(step=11) == "main"
```

Add a source-level guard only for call placement:

```python
assert "constrain_cosmos_teacher_timestep_pair" not in deployment_method_source
assert "_cosmos_deployment_joint_rollout_step" in trainer_source
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  distillation_flowmap/tests/test_cosmos_progressive_opd.py \
  distillation_flowmap/tests/test_cosmos_progressive_protocol.py \
  distillation_flowmap/tests/test_cosmos_progressive_runner.py
```

Expected: failures because the deployment method, metrics, and disjoint trainer route do not exist.

- [ ] **Step 3: Implement the deployment method**

First obtain one endpoint-only teacher result. This call supplies the clean
video and action endpoints; it is not a velocity query and does not request
central differences:

```python
with torch.no_grad():
    endpoint_result = teacher.predict_raw_latent_target(
        batch,
        noise=anchor_noise,
        t=anchor_t_norm,
        r=anchor_r_norm,
        epsilon=float(self.config.cosmos_latent_epsilon),
        include_cdiff=False,
    )
video_x0 = endpoint_result["cosmos_latent_x0"].to(
    device=self.device, dtype=batch["latents"].dtype
)
action_x0_full = cosmos_actions_to_flowmap_x0(
    endpoint_result["actions"],
    target_shape=tuple(batch["actions"].shape),
    q01=self.config.norm_stat["q01"],
    q99=self.config.norm_stat["q99"],
    inverse_used_action_channel_ids=self.config.inverse_used_action_channel_ids,
    device=self.device,
    dtype=batch["actions"].dtype,
    packing_schema=self.config.action_packing_schema,
    downsample_factor=self.config.action_downsample_factor,
)
```

The endpoint query's `anchor_t_norm` and `anchor_r_norm` remain inside the raw
teacher window, but their noisy state is discarded. Only the returned clean
`cosmos_latent_x0` and raw actions are used as deployment endpoints. Then
perform these exact production operations:

```python
student_steps = int(student_steps)
if student_steps not in (1, 2, 4):
    raise ValueError("deployment student_steps must be one of 1, 2, 4")

video_t = torch.full((B, video_frames), 1000.0, device=self.device)
video_r = torch.zeros_like(video_t)
action_t, action_r = broadcast_joint_action_timesteps(
    video_t, video_r, action_frames=batch["actions"].shape[2]
)
video_noise = torch.randn_like(video_x0)
action_noise = torch.randn_like(action_x0_full)
```

Use `video_noise` as the video state at `t=1000`; use downsampled
`action_noise[:, :, ::4]` as the action state. Build the normal student input,
then call once:

```python
video_final, _, _, action_final = self._student_euler_integrate(
    noisy_latents=video_noise,
    timesteps=video_t,
    target_r=video_r,
    base_input_dict=student_input,
    empty_emb=empty_emb,
    cfg_scale=cfg_scale,
    ref_shape=video_x0.shape,
    B=B,
    num_frames=video_frames,
    K_steps=student_steps,
    action_target_r=action_r,
    return_final_action=True,
    return_final_action_state=True,
)
```

At `r=0`, compare `video_final` directly with `video_x0` and `action_final`
directly with `batch["actions"][:, :, ::4]` through
`deployment_endpoint_losses`, passing the real compact action mask
`batch["actions_mask"][:, :, ::4]`. Do not call the raw velocity teacher,
`constrain_cosmos_teacher_timestep_pair`, or `x - sigma*v`.

- [ ] **Step 4: Route mutually exclusive optimizer steps**

In `flowmap_trainer.py`, compute one `scheduled_kind` before choosing a step:

```python
if should_run_deployment_joint_rollout(
    self.step, config.deployment_joint_rollout_interval
):
    scheduled_kind = "deployment"
elif should_run_raw_auxiliary(
    self.step,
    warmup=config.opd_aux_warmup_steps,
    interval=config.opd_aux_interval,
    phase=config.opd_aux_phase,
):
    scheduled_kind = "raw_auxiliary"
else:
    scheduled_kind = "main"
```

For deployment, derive the balanced update index and K:

```python
update_index = self.step // config.deployment_joint_rollout_interval
student_steps = deployment_joint_step_for_update(update_index)
result = self._cosmos_deployment_joint_rollout_step(
    batch, step_in_acc, student_steps=student_steps
)
```

For raw auxiliary, call the existing `_opd_aux_transition_step`. For main,
call `_train_step`. Keep the standalone-step requirement
`gradient_accumulation_steps == 1`.

Accumulate/reduce/log all deployment metrics in the same positional ordering
on every rank. Add an explicit reduction-count test so metric insertion cannot
silently shift existing values.

- [ ] **Step 5: Run Stage-2 focused and regression tests**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  distillation_flowmap/tests/test_cosmos_deployment_rollout.py \
  distillation_flowmap/tests/test_cosmos_progressive_opd.py \
  distillation_flowmap/tests/test_cosmos_progressive_protocol.py \
  distillation_flowmap/tests/test_cosmos_progressive_runner.py \
  distillation_flowmap/tests/test_cosmos_progressive_metrics.py
```

Expected: all tests pass; deployment and raw auxiliary never share a step.

- [ ] **Step 6: Commit Task 3**

```bash
git add \
  distillation_flowmap/flowmap_step.py \
  distillation_flowmap/flowmap_trainer.py \
  distillation_flowmap/cosmos_progressive_opd.py \
  distillation_flowmap/tests/test_cosmos_progressive_opd.py \
  distillation_flowmap/tests/test_cosmos_progressive_protocol.py \
  distillation_flowmap/tests/test_cosmos_progressive_runner.py \
  distillation_flowmap/tests/test_cosmos_progressive_metrics.py
git commit -m "feat: train full joint Cosmos deployment rollouts"
```

---

### Task 4: Persist and Execute the Training Contract

**Files:**
- Modify: `distillation_flowmap/cosmos_training_contract.py`
- Modify: `distillation_flowmap/flowmap_trainer.py:1220-1290`
- Create: `distillation_flowmap/verify_cosmos_joint_training_contract.py`
- Create: `distillation_flowmap/tests/test_cosmos_training_contract.py`
- Modify: `distillation_flowmap/tests/test_cosmos_latent_checkpoint_config.py`

**Interfaces:**
- Produces: `contract_metadata(config, *, stage: str) -> dict[str, object]`.
- Produces: `validate_contract_metadata(payload, *, required_stage: str) -> None`.
- Produces CLI: `python -m distillation_flowmap.verify_cosmos_joint_training_contract --output PATH`.
- Produces atomic attestation JSON with production round-trip evidence.

- [ ] **Step 1: Write failing metadata and verifier tests**

Add:

```python
def test_stage1_contract_contains_packing_but_not_stage2_attestation():
    payload = contract_metadata(stage1_config, stage="raw_stage1")
    assert payload["contract_version"] == 2
    assert payload["action_packing_schema"] == "downsample_survivor_v2"
    assert "joint_student_steps" not in payload


def test_stage2_contract_contains_full_deployment_attestation():
    payload = contract_metadata(stage2_config, stage="progressive_stage2")
    assert payload["deployment_timestep_start"] == 1000
    assert payload["deployment_timestep_end"] == 0
    assert payload["joint_student_steps"] == [1, 2, 4]
    assert payload["deployment_joint_rollout_interval"] == 4
    assert payload["raw_teacher_window_is_auxiliary"] is True


def test_contract_verifier_runs_production_action_round_trip(tmp_path):
    output = tmp_path / "attestation.json"
    assert verifier_main(["--output", str(output)]) == 0
    payload = json.loads(output.read_text())
    assert payload["action_round_trip"]["num_actions"] == 16
    assert payload["action_round_trip"]["max_abs_error"] < 1e-5
```

Also test wrong schema, missing Stage-2 fields, wrong K list, and a pre-existing
attestation output are rejected.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  distillation_flowmap/tests/test_cosmos_training_contract.py \
  distillation_flowmap/tests/test_cosmos_latent_checkpoint_config.py
```

Expected: failures because metadata builders and verifier do not exist.

- [ ] **Step 3: Implement metadata and checkpoint persistence**

`contract_metadata` returns the common Stage-1 fields and adds the exact
Stage-2 deployment fields only for `progressive_stage2`. Validation must
compare exact values and types; booleans must not accept integers.

In `_save_checkpoint`, merge:

```python
stage_name = getattr(self.config, "training_contract_stage", None)
if stage_name is not None:
    config_dict.update(contract_metadata(self.config, stage=stage_name))
```

Set:

```python
# raw Stage-1 config
cfg.training_contract_stage = "raw_stage1"

# progressive Stage-2 config
cfg.training_contract_stage = "progressive_stage2"
```

Write checkpoint `config.json` through `.config.json.tmp` and `os.replace`
rather than direct overwrite.

- [ ] **Step 4: Implement the executable verifier**

The CLI must:

1. construct 16 distinguishable actions;
2. call `cosmos_actions_to_flowmap_x0` with the production schema;
3. downsample through the production `::4` route;
4. decode the compact tensor through the production
   `evaluation.libero.cosmos_progressive_s4_server.decode_student_action`
   function using `ActionDecodingTemplate.from_config(config)`;
5. compute `max_abs_error`;
6. validate the exact deployment and raw-window specs;
7. write one new attestation atomically and refuse overwrite.

Attestation fields:

```python
payload = {
    **contract_metadata(stage2_config, stage="progressive_stage2"),
    "git_commit": git_commit,
    "verifier_module": "distillation_flowmap.verify_cosmos_joint_training_contract",
    "action_round_trip": {
        "num_actions": 16,
        "max_abs_error": max_abs_error,
    },
    "verified_at_utc": datetime.now(timezone.utc).isoformat(),
}
```

- [ ] **Step 5: Run contract/checkpoint tests and CLI dry execution**

Run:

```bash
tmp_dir="$(mktemp -d)"
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  distillation_flowmap/tests/test_cosmos_training_contract.py \
  distillation_flowmap/tests/test_cosmos_latent_checkpoint_config.py
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m distillation_flowmap.verify_cosmos_joint_training_contract \
  --output "${tmp_dir}/attestation.json"
test -s "${tmp_dir}/attestation.json"
```

Expected: tests and verifier exit zero.

- [ ] **Step 6: Commit Task 4**

```bash
git add \
  distillation_flowmap/cosmos_training_contract.py \
  distillation_flowmap/flowmap_trainer.py \
  distillation_flowmap/verify_cosmos_joint_training_contract.py \
  distillation_flowmap/tests/test_cosmos_training_contract.py \
  distillation_flowmap/tests/test_cosmos_latent_checkpoint_config.py
git commit -m "feat: attest Cosmos training contracts"
```

---

### Task 5: Add a Corrected Eight-GPU Raw Stage-1 Launcher

**Files:**
- Create: `distillation_flowmap/run_cosmos_raw_stage1_8gpu.sh`
- Create: `distillation_flowmap/tests/test_run_cosmos_raw_stage1_8gpu.py`

**Interfaces:**
- Consumes env: `STAGE1_OUTPUT`, `CLEAN_STUDENT_BASE_MODEL_PATH`, `DATASET_PATH`, `COSMOS_POLICY_PATH`, `TRAIN_SEED`, `MASTER_PORT`.
- Produces modes: `run`, `dry-run`.
- Produces options: `--steps N`, `--save-interval N`, `--master-port PORT`,
  `--output-dir PATH`, `--run-tag TAG`, and `--resume-step N`.
- Produces corrected Stage-1 `step_5000`.

- [ ] **Step 1: Write failing launcher tests**

Tests create fake clean base/dataset/teacher/worker layouts and assert:

```python
result = run_stage1("dry-run", env=env)
assert result.returncode == 0
assert "CONFIG_FILE=distillation_flowmap.config_libero_cosmos_policy_stage1" in result.stdout
assert "MAX_TRAIN_STEPS=5000" in result.stdout
assert "SAVE_INTERVAL=1000" in result.stdout
assert "TRAIN_SEED=42" in result.stdout
assert "--nproc_per_node=8" in result.stdout
assert "--master_port=29671" in result.stdout
assert env["CLEAN_STUDENT_BASE_MODEL_PATH"] in result.stdout
assert "raw_stage1_5000" not in result.stdout
assert not Path(env["STAGE1_OUTPUT"]).exists()
```

Resume tests require both online/target transformer configs with contract v2,
optimizer, scheduler, and exact `checkpoint_step`. Test the highest-step
selection is performed by the orchestrator, not this launcher.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  distillation_flowmap/tests/test_run_cosmos_raw_stage1_8gpu.py
```

Expected: failure because the launcher does not exist.

- [ ] **Step 3: Implement the launcher**

Use `set -euo pipefail`, the same strict eight-device/port/path validation as
the progressive launcher, and absolute defaults:

```bash
PYTHON_BIN="${PYTHON_BIN:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/torchrun}"
CLEAN_STUDENT_BASE_MODEL_PATH="${CLEAN_STUDENT_BASE_MODEL_PATH:-/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero}"
MAX_TRAIN_STEPS=5000
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
TRAIN_SEED="${TRAIN_SEED:-42}"
MASTER_PORT="${MASTER_PORT:-29671}"
```

Fresh run sets no contaminated resume path and passes the clean base through
`STUDENT_BASE_MODEL_PATH`. Resume requires `--resume-step`, loads its own
online checkpoint, and sets:

```bash
RESUME_ONLINE_FROM_TARGET=0
RESET_RESUME_STEP=0
RESUME_OPTIMIZER_STATE=1
```

Fresh output with existing nonempty checkpoints is rejected. `dry-run` prints
all resolved assignments and command, then exits before `mkdir`.

- [ ] **Step 4: Run launcher tests and syntax**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  distillation_flowmap/tests/test_run_cosmos_raw_stage1_8gpu.py
bash -n distillation_flowmap/run_cosmos_raw_stage1_8gpu.sh
```

Expected: all pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add \
  distillation_flowmap/run_cosmos_raw_stage1_8gpu.sh \
  distillation_flowmap/tests/test_run_cosmos_raw_stage1_8gpu.py
git commit -m "feat: launch corrected Cosmos raw Stage-1"
```

---

### Task 6: Enforce Corrected Stage-1 Handoff in the Progressive Launcher

**Files:**
- Modify: `distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh`
- Modify: `distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py`

**Interfaces:**
- Consumes: corrected `COSMOS_STAGE1_ROOT`.
- Requires: Stage-1 online/target contract version 2 and packing schema v2.
- Produces: `universal-video-action` Stage-2 at 5,000 steps with deployment contract.

- [ ] **Step 1: Write failing corrected-lineage tests**

Update fixtures to carry real contract fields. Add rejection tests for:

```python
@pytest.mark.parametrize(
    "mutation",
    [
        {"contract_version": 1},
        {"action_packing_schema": "legacy_dense_v1"},
        {"checkpoint_step": 4999},
    ],
)
def test_stage2_rejects_incompatible_stage1_contract(tmp_path, mutation):
    checkpoint = make_stage1_checkpoint(
        tmp_path / "stage1/checkpoints/step_5000",
        contract={
            "contract_version": 2,
            "action_packing_schema": "downsample_survivor_v2",
            "checkpoint_step": 5000,
            "training_contract_stage": "raw_stage1",
        },
    )
    config_path = checkpoint / "online_student/transformer/config.json"
    config = json.loads(config_path.read_text())
    config.update(mutation)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    env = launcher_env(tmp_path, COSMOS_STAGE1_ROOT=str(checkpoint))
    result = run_launcher("universal-video-action", "--dry-run", env=env)
    assert result.returncode != 0
    assert "Stage-1 contract" in result.stderr
```

Positive dry-run assertions:

```python
assert "COSMOS_STAGE1_ROOT=" + corrected_stage1 in result.stdout
assert "DEPLOYMENT_JOINT_ROLLOUT_INTERVAL=4" in result.stdout
assert "OPD_AUX_INTERVAL=8" in result.stdout
assert "OPD_AUX_PHASE=2" in result.stdout
assert "MAX_TRAIN_STEPS=5000" in result.stdout
assert "TRAIN_SEED=42" in result.stdout
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py
```

Expected: failures because the launcher does not validate contract metadata or
export deployment scheduling.

- [ ] **Step 3: Implement fail-closed Stage-1 and resume validation**

Add a Python validation block that reads online and target `config.json` and
requires:

```python
{
    "contract_version": 2,
    "action_packing_schema": "downsample_survivor_v2",
    "action_chunk_shape": [4, 4],
    "checkpoint_step": 5000,
}
```

Export:

```bash
DEPLOYMENT_JOINT_ROLLOUT=1
DEPLOYMENT_JOINT_ROLLOUT_INTERVAL=4
DEPLOYMENT_ACTION_WEIGHT=1.0
OPD_AUX_INTERVAL=8
OPD_AUX_PHASE=2
TRAIN_SEED=42
```

Stage-2 resume validates its own full Stage-2 contract plus optimizer and
scheduler. Fresh mode still initializes online from corrected Stage-1 target
and resets step/optimizer.

- [ ] **Step 4: Run launcher/config regressions**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py \
  distillation_flowmap/tests/test_cosmos_progressive_config.py
bash -n distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh
```

Expected: all pass.

- [ ] **Step 5: Commit Task 6**

```bash
git add \
  distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh \
  distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py
git commit -m "fix: require corrected Cosmos Stage-1 lineage"
```

---

### Task 7: Add Atomic Pipeline State and Checkpoint Validation

**Files:**
- Create: `distillation_flowmap/cosmos_pipeline_state.py`
- Create: `distillation_flowmap/tests/test_cosmos_pipeline_state.py`

**Interfaces:**
- Produces: `RunIdentity` dataclass.
- Produces: `IncompleteCheckpointError(RuntimeError)`.
- Produces: `atomic_write_json(path: Path, payload: Mapping) -> None`.
- Produces: `initialize_or_validate_run(root: Path, identity: RunIdentity, *, mutate: bool) -> str`.
- Produces: `find_latest_complete_checkpoint(output: Path, *, stage: str, max_step: int) -> Path | None`.
- Produces: `validate_completion_manifest(path: Path, *, expected_stage: str, expected_identity: RunIdentity) -> dict`.
- Produces CLI subcommands used by Bash: `init`, `status`, `checkpoint`, `complete`.

- [ ] **Step 1: Write failing state/checkpoint tests**

Create helpers `identity(**overrides)`, `make_checkpoint(path, *, stage,
complete=True)`, and `write_completion(path, *, identity, stage,
checkpoint_identity)` in the test module. Then add:

```python
def test_atomic_write_replaces_target_and_removes_temporary_sibling(tmp_path):
    target = tmp_path / "state.json"
    target.write_text('{"old": true}', encoding="utf-8")
    atomic_write_json(target, {"new": True})
    assert json.loads(target.read_text()) == {"new": True}
    assert not (tmp_path / ".state.json.tmp").exists()


def test_dry_run_identity_validation_does_not_create_root(tmp_path):
    root = tmp_path / "absent"
    assert initialize_or_validate_run(root, identity(), mutate=False) == "new"
    assert not root.exists()


def test_same_identity_is_reusable(tmp_path):
    root = tmp_path / "run"
    expected = identity()
    assert initialize_or_validate_run(root, expected, mutate=True) == "created"
    assert initialize_or_validate_run(root, expected, mutate=True) == "existing"


@pytest.mark.parametrize(
    "override",
    [
        {"git_commit": "different"},
        {"dataset_path": "/different/dataset"},
        {"seed": 0},
        {"action_packing_schema": "legacy_dense_v1"},
    ],
)
def test_different_run_identity_is_rejected(tmp_path, override):
    root = tmp_path / "run"
    initialize_or_validate_run(root, identity(), mutate=True)
    with pytest.raises(ValueError, match="run identity mismatch"):
        initialize_or_validate_run(root, identity(**override), mutate=False)


def test_latest_complete_checkpoint_requires_all_six_artifacts(tmp_path):
    output = tmp_path / "stage"
    complete = make_checkpoint(output / "checkpoints/step_100", stage="raw_stage1")
    assert find_latest_complete_checkpoint(
        output, stage="raw_stage1", max_step=5000
    ) == complete
    (complete / "optimizer.pt").unlink()
    with pytest.raises(IncompleteCheckpointError, match="optimizer.pt"):
        find_latest_complete_checkpoint(output, stage="raw_stage1", max_step=5000)


def test_incomplete_higher_checkpoint_is_not_silently_skipped(tmp_path):
    output = tmp_path / "stage"
    make_checkpoint(output / "checkpoints/step_100", stage="raw_stage1")
    make_checkpoint(
        output / "checkpoints/step_200", stage="raw_stage1", complete=False
    )
    with pytest.raises(IncompleteCheckpointError, match="step_200"):
        find_latest_complete_checkpoint(output, stage="raw_stage1", max_step=5000)


def test_completion_manifest_requires_parent_and_checkpoint_identity(tmp_path):
    expected = identity()
    manifest = tmp_path / "stage.complete.json"
    write_completion(
        manifest,
        identity=expected,
        stage="raw_stage1",
        checkpoint_identity="sha256:abc",
    )
    payload = validate_completion_manifest(
        manifest, expected_stage="raw_stage1", expected_identity=expected
    )
    assert payload["checkpoint_identity"] == "sha256:abc"
    payload["parent_identity"] = "sha256:different"
    atomic_write_json(manifest, payload)
    with pytest.raises(ValueError, match="parent identity"):
        validate_completion_manifest(
            manifest, expected_stage="raw_stage1", expected_identity=expected
        )
```

Use fixture configs with exact Stage-1/Stage-2 contract fields and compute
weight identity as SHA-256 of each small sentinel safetensors file.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  distillation_flowmap/tests/test_cosmos_pipeline_state.py
```

Expected: import failure.

- [ ] **Step 3: Implement the focused state module**

`RunIdentity` fields:

```python
@dataclass(frozen=True)
class RunIdentity:
    git_commit: str
    clean_base_path: str
    clean_base_identity: str
    dataset_path: str
    teacher_path: str
    seed: int
    contract_version: int
    action_packing_schema: str
    stage1_steps: int
    stage2_steps: int
    evaluation_protocol: str
```

`atomic_write_json` writes `.{name}.tmp`, fsyncs the file, and uses
`os.replace`. It refuses symlink targets. `mutate=False` performs every
validation but never calls `mkdir`, `write_text`, or `os.replace`.

Checkpoint completeness requires:

```text
online_student/transformer/config.json
online_student/transformer/diffusion_pytorch_model.safetensors
target_student/transformer/config.json
target_student/transformer/diffusion_pytorch_model.safetensors
optimizer.pt
lr_scheduler.pt
```

If the highest discovered `step_*` is incomplete, return a structured hard
error; do not fall back to an older checkpoint without reporting it.

- [ ] **Step 4: Run tests and CLI smoke**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  distillation_flowmap/tests/test_cosmos_pipeline_state.py
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m distillation_flowmap.cosmos_pipeline_state --help
```

Expected: all pass.

- [ ] **Step 5: Commit Task 7**

```bash
git add \
  distillation_flowmap/cosmos_pipeline_state.py \
  distillation_flowmap/tests/test_cosmos_pipeline_state.py
git commit -m "feat: track resumable Cosmos pipeline state"
```

---

### Task 8: Make the K1/K2/K4 Matrix Resumable

**Files:**
- Modify: `evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh`
- Modify: `evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py`

**Interfaces:**
- Adds mode: `resume`.
- Keeps modes: `run`, `dry-run`.
- `resume` validates and skips complete K roots; runs the first incomplete K
  and every later K; never overwrites invalid or partial K output.

- [ ] **Step 1: Write failing resume tests**

Add sentinel child summaries and assert:

```python
def test_resume_skips_valid_k1_and_runs_k2_then_k4(tmp_path):
    make_complete_summary(tmp_path / "matrix/k1", k=1)
    result = run_matrix("resume", root=tmp_path / "matrix")
    assert parsed_started_steps(result.stdout) == [2, 4]


def test_resume_rejects_existing_k_with_missing_or_mismatched_summary(tmp_path):
    (tmp_path / "matrix/k1").mkdir(parents=True)
    result = run_matrix("resume", root=tmp_path / "matrix")
    assert result.returncode != 0
    assert "incomplete K=1" in result.stderr


def test_resume_with_three_valid_children_only_rebuilds_atomic_matrix_summary(tmp_path):
    root = tmp_path / "matrix"
    for k in (1, 2, 4):
        make_complete_summary(root / f"k{k}", k=k)
    result = run_matrix("resume", root=root)
    assert result.returncode == 0
    assert parsed_started_steps(result.stdout) == []
    assert json.loads((root / "matrix_summary.json").read_text())["steps"] == [1, 2, 4]


def test_dry_run_does_not_create_matrix_root(tmp_path):
    root = tmp_path / "matrix"
    result = run_matrix("dry-run", root=root)
    assert result.returncode == 0
    assert not root.exists()


def test_run_refuses_existing_matrix_root(tmp_path):
    root = tmp_path / "matrix"
    root.mkdir()
    result = run_matrix("run", root=root)
    assert result.returncode != 0
    assert "already exists" in result.stderr
```

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py
```

Expected: `resume` is rejected.

- [ ] **Step 3: Implement validated resume**

For each K:

```bash
if [[ -d "${MATRIX_ROOT}/k${k}" ]]; then
    validate_child_summary_or_die "${k}"
    emit_kv "MATRIX_SKIP_STEP" "${k}"
    continue
fi
```

Existing path without a valid complete summary is a hard error. A valid child
summary must have 500 records, matching K/checkpoint/classification/formal
status. Prompt-table reuse must work when K1 is skipped. After all K values,
validate and atomically publish the matrix summary.

- [ ] **Step 4: Run matrix/formal regressions**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py \
  evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py
bash -n evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh
```

Expected: all pass.

- [ ] **Step 5: Commit Task 8**

```bash
git add \
  evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh \
  evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py
git commit -m "feat: resume joint Cosmos evaluation matrix"
```

---

### Task 9: Add the Single-Allocation State-Machine Orchestrator

**Files:**
- Create: `distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh`
- Create: `distillation_flowmap/tests/test_run_cosmos_stage1_stage2_eval_8gpu.py`

**Interfaces:**
- Modes: positional `run`, `dry-run`, `status`; no positional mode defaults
  to `run` for compatibility with the reference launcher.
- Phase filter: `--phase train|eval|all`, default `all`. `train` executes
  contract→Stage-1→Stage-2; `eval` revalidates those completed artifacts then
  runs K1→K2→K4; `all` runs the full chain.
- Accepts reference-style options: `--steps N` (applies to both training
  stages), `--save-interval N`, `--episodes N`, `--stage1-master-port PORT`,
  `--stage2-master-port PORT`, `--eval-master-port-base PORT`,
  `--eval-ws-port-base PORT`, and `--run-tag TAG`.
- `RUN_ROOT` may be supplied explicitly. Otherwise a validated nonempty
  `--run-tag` derives one immutable root below the shared checkout.
- Produces sequence: contract → Stage-1 → Stage-2 → K1 → K2 → K4.
- Consumes all launchers and state APIs from Tasks 4–8.

- [ ] **Step 1: Write failing full-state-machine tests**

Use executable sentinel launchers and a temporary run root. Cover:

```python
def test_dry_run_prints_complete_order_and_creates_nothing(tmp_path):
    result = run_pipeline(
        "dry-run", "--phase", "all", "--run-tag", "contract-test",
        root=tmp_path / "run",
    )
    assert parsed_stage_order(result.stdout) == [
        "contract", "stage1", "stage2", "evaluation:k1",
        "evaluation:k2", "evaluation:k4",
    ]
    assert "--nproc_per_node=8" in result.stdout
    assert "29671" in result.stdout
    assert "29672" in result.stdout
    assert not (tmp_path / "run").exists()


def test_run_hands_corrected_stage1_target_to_stage2_and_stage2_online_to_eval(tmp_path):
    fixture = PipelineSentinels(tmp_path)
    result = fixture.run("run")
    assert result.returncode == 0
    stage1_step_5000 = fixture.run_root / "stage1/checkpoints/step_5000"
    stage2_step_5000 = (
        fixture.run_root
        / "stage2/universal-video-action/checkpoints/step_5000"
    )
    stage2_env = fixture.recorded_env("stage2")
    eval_env = fixture.recorded_env("evaluation")
    assert stage2_env["COSMOS_STAGE1_ROOT"] == str(stage1_step_5000)
    assert eval_env["S4_CKPT_ROOT"] == str(
        stage2_step_5000 / "online_student/transformer"
    )
    assert eval_env["S4_ALIGNMENT_VERIFIED"] == "1"


def test_reference_style_phase_and_port_options_are_forwarded(tmp_path):
    fixture = PipelineSentinels(tmp_path)
    result = fixture.run(
        "dry-run",
        "--phase", "all",
        "--steps", "5000",
        "--save-interval", "1000",
        "--episodes", "500",
        "--stage1-master-port", "29671",
        "--stage2-master-port", "29672",
        "--eval-master-port-base", "29680",
        "--eval-ws-port-base", "29780",
        "--run-tag", "reference-style",
    )
    assert result.returncode == 0
    assert fixture.resolved_value(result.stdout, "STAGE1_STEPS") == "5000"
    assert fixture.resolved_value(result.stdout, "STAGE2_STEPS") == "5000"
    assert fixture.resolved_value(result.stdout, "SAVE_INTERVAL") == "1000"
    assert fixture.resolved_value(result.stdout, "EPISODES") == "500"
    assert fixture.resolved_value(result.stdout, "STAGE1_MASTER_PORT") == "29671"
    assert fixture.resolved_value(result.stdout, "STAGE2_MASTER_PORT") == "29672"


def test_rerun_skips_completed_stage1_and_resumes_stage2(tmp_path):
    fixture = PipelineSentinels(tmp_path, stop_after="stage2:step_2000")
    assert fixture.run("run").returncode != 0
    fixture.stop_after = None
    result = fixture.run("run")
    assert result.returncode == 0
    assert fixture.invocation_count("stage1") == 1
    assert fixture.last_stage2_resume_step() == 2000


def test_failed_stage1_never_starts_stage2_or_evaluation(tmp_path):
    fixture = PipelineSentinels(tmp_path, fail_stage="stage1")
    assert fixture.run("run").returncode != 0
    assert fixture.invocation_count("stage2") == 0
    assert fixture.invocation_count("evaluation") == 0


def test_failed_stage2_never_starts_evaluation(tmp_path):
    fixture = PipelineSentinels(tmp_path, fail_stage="stage2")
    assert fixture.run("run").returncode != 0
    assert fixture.invocation_count("stage1") == 1
    assert fixture.invocation_count("evaluation") == 0


def test_contract_failure_starts_no_training(tmp_path):
    fixture = PipelineSentinels(tmp_path, fail_stage="contract")
    assert fixture.run("run").returncode != 0
    assert fixture.invocation_count("stage1") == 0
    assert fixture.invocation_count("stage2") == 0


def test_status_is_read_only(tmp_path):
    fixture = PipelineSentinels(tmp_path)
    before = snapshot_tree(fixture.run_root)
    result = fixture.run("status")
    assert result.returncode == 0
    assert snapshot_tree(fixture.run_root) == before
    assert fixture.total_child_invocations() == 0


def test_different_run_identity_is_rejected_before_children(tmp_path):
    fixture = PipelineSentinels(tmp_path)
    fixture.initialize_identity(seed=42)
    result = fixture.run("run", extra_env={"TRAIN_SEED": "0"})
    assert result.returncode != 0
    assert "run identity mismatch" in result.stderr
    assert fixture.total_child_invocations() == 0


def test_incomplete_checkpoint_stops_without_deletion(tmp_path):
    fixture = PipelineSentinels(tmp_path)
    partial = fixture.make_partial_checkpoint("stage1", step=200)
    result = fixture.run("run")
    assert result.returncode != 0
    assert "incomplete checkpoint" in result.stderr
    assert partial.exists()
    assert fixture.invocation_count("stage2") == 0
```

Define `PipelineSentinels` in the same test file: each injected executable
records argv/environment as JSON and materializes only the exact sentinel
artifacts needed by the state validator. Add a signal test that starts a
sentinel child process group, waits for its PID file, sends SIGTERM to the
orchestrator, and asserts the child exits and the interrupted state is written:

```python
def test_sigterm_reaps_active_child_group_and_marks_interrupted(tmp_path):
    fixture = PipelineSentinels(tmp_path, blocking_stage="stage1")
    process = fixture.popen("run")
    child_pid = wait_for_pid_file(fixture.run_root / "sentinel-child.pid")
    os.kill(process.pid, signal.SIGTERM)
    assert process.wait(timeout=10) == 143
    wait_until(lambda: not process_exists(child_pid), timeout=10)
    state = json.loads((fixture.run_root / "state/pipeline.json").read_text())
    assert state["stage"] == "stage1"
    assert state["status"] == "interrupted"
    assert state["signal"] == "TERM"
```

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  distillation_flowmap/tests/test_run_cosmos_stage1_stage2_eval_8gpu.py
```

Expected: failure because the orchestrator does not exist.

- [ ] **Step 3: Implement immutable defaults and preflight**

Use:

```bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"
CLEAN_STUDENT_BASE_MODEL_PATH="${CLEAN_STUDENT_BASE_MODEL_PATH:-/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero}"
DATASET_PATH="${DATASET_PATH:-/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot}"
TRAIN_SEED="${TRAIN_SEED:-42}"
STAGE1_STEPS=5000
STAGE2_STEPS=5000
SAVE_INTERVAL=1000
EPISODES=500
STAGE1_OUTPUT="${RUN_ROOT}/stage1"
STAGE2_OUTPUT_ROOT="${RUN_ROOT}/stage2"
MATRIX_ROOT="${RUN_ROOT}/evaluation"
```

Require exactly eight unique GPU ordinals and driver `>=570.124.06` in `run`
mode. Dry-run prints the check without invoking `nvidia-smi`. Run the contract
verifier before creating a training output or child process.

Create this exact layout only after immutable identity validation succeeds:

```text
$RUN_ROOT/
├── manifests/
│   ├── run.json
│   ├── contract-attestation.json
│   ├── stage1.complete.json
│   ├── stage2.complete.json
│   └── evaluation.complete.json
├── state/pipeline.json
├── logs/{preflight,stage1,stage2,evaluation}.log
├── stage1/checkpoints/step_5000/
├── stage2/universal-video-action/checkpoints/step_5000/
└── evaluation/
    ├── k1/formal_summary.json
    ├── k2/formal_summary.json
    ├── k4/formal_summary.json
    └── matrix_summary.json
```

- [ ] **Step 4: Implement state transitions and handoffs**

For each state:

1. ask `cosmos_pipeline_state` whether to run, resume, or skip;
2. emit `PIPELINE_STAGE` and `PIPELINE_DECISION`;
3. build one Bash array command;
4. in dry-run, print `%q` tokens and do not execute;
5. in run, execute synchronously in its own `setsid` process group, redirect
   stdout/stderr to the stage's exact log path, wait, and capture its exit code;
6. validate the resulting checkpoint/summary;
7. atomically write its completion manifest.

Stage-1 command uses `run_cosmos_raw_stage1_8gpu.sh`. Stage-2 command exports
the corrected Stage-1 root and uses only `universal-video-action`. Evaluation
exports:

```bash
S4_CKPT_ROOT="${stage2_step_5000}/online_student/transformer"
S4_ALIGNMENT_VERIFIED=1
S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH=0
```

and calls the matrix `run` or `resume` mode according to state.

- [ ] **Step 5: Implement signal/process-group handling**

Require Linux `setsid` during live preflight. Launch every child with
`setsid "${command[@]}" &` and retain its process-group leader PID in
`ACTIVE_CHILD_PID`. Trap:

```bash
on_signal() {
    local signal="$1"
    local exit_code="$2"
    if [[ -n "${ACTIVE_CHILD_PID:-}" ]]; then
        kill -s "${signal}" -- "-${ACTIVE_CHILD_PID}" 2>/dev/null || \
            kill -s "${signal}" "${ACTIVE_CHILD_PID}" 2>/dev/null || true
        wait "${ACTIVE_CHILD_PID}" || true
    fi
    mark_interrupted_atomically "${CURRENT_STAGE}" "${signal}"
    exit "${exit_code}"
}
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
```

Do not remove files in the trap.

- [ ] **Step 6: Run orchestrator and all launcher tests**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  distillation_flowmap/tests/test_run_cosmos_stage1_stage2_eval_8gpu.py \
  distillation_flowmap/tests/test_cosmos_pipeline_state.py \
  distillation_flowmap/tests/test_run_cosmos_raw_stage1_8gpu.py \
  distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py \
  evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py
bash -n distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh
```

Expected: all pass; no real GPU process or persistent output exists.

- [ ] **Step 7: Commit Task 9**

```bash
git add \
  distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh \
  distillation_flowmap/tests/test_run_cosmos_stage1_stage2_eval_8gpu.py
git commit -m "feat: orchestrate Cosmos retraining and evaluation"
```

---

### Task 10: Final Contract, Regression, and Exact Dry-Run Verification

**Files:**
- Modify only if verification reveals a defect in Tasks 1–9.

**Interfaces:**
- Produces: verified one-command handoff.
- Does not start live training/evaluation.

- [ ] **Step 1: Run the complete focused regression suite**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  distillation_flowmap/tests/test_cosmos_policy_backend.py \
  distillation_flowmap/tests/test_cosmos_teacher_roles.py \
  distillation_flowmap/tests/test_cosmos_deployment_rollout.py \
  distillation_flowmap/tests/test_cosmos_progressive_config.py \
  distillation_flowmap/tests/test_cosmos_progressive_opd.py \
  distillation_flowmap/tests/test_cosmos_progressive_protocol.py \
  distillation_flowmap/tests/test_cosmos_progressive_runner.py \
  distillation_flowmap/tests/test_cosmos_progressive_metrics.py \
  distillation_flowmap/tests/test_cosmos_training_contract.py \
  distillation_flowmap/tests/test_cosmos_latent_checkpoint_config.py \
  distillation_flowmap/tests/test_cosmos_pipeline_state.py \
  distillation_flowmap/tests/test_run_cosmos_raw_stage1_8gpu.py \
  distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py \
  distillation_flowmap/tests/test_run_cosmos_stage1_stage2_eval_8gpu.py \
  evaluation/libero/tests/test_cosmos_progressive_s4_service.py \
  evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py \
  evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py
```

Expected: zero failures.

- [ ] **Step 2: Run static checks**

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m py_compile \
  distillation_flowmap/cosmos_training_contract.py \
  distillation_flowmap/cosmos_deployment_rollout.py \
  distillation_flowmap/verify_cosmos_joint_training_contract.py \
  distillation_flowmap/cosmos_pipeline_state.py
bash -n distillation_flowmap/run_cosmos_raw_stage1_8gpu.sh
bash -n distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh
bash -n distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh
bash -n evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 3: Run the exact intended dry-run**

First require the root to be absent:

```bash
cd /kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/cosmos-progressive-s4-eval
export RUN_ROOT=/kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/runs/cosmos_joint_fixed_20260724
test ! -e "${RUN_ROOT}"
bash distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh dry-run
test ! -e "${RUN_ROOT}"
```

Verify output contains, in order:

```text
contract
stage1 run, 8 GPUs, seed 42, 5000 steps, port 29671, clean LingBotVA base
stage2 run, universal-video-action, 8 GPUs, seed 42, 5000 steps, port 29672
evaluation K1, K2, K4, 500 records each, four shards, video seeds 0,1
```

Verify no matching training/evaluation process and no GPU compute process owned
by this command.

- [ ] **Step 4: Review the full feature diff**

Review from plan base through HEAD for:

- action value identity, not shape-only correctness;
- exact deployment endpoint semantics;
- no raw velocity query outside its auxiliary window;
- disjoint deployment/raw-aux schedules;
- Stage-1/Stage-2 checkpoint metadata and lineage;
- clean-base enforcement;
- resume/no-overwrite/signal behavior;
- evaluation handoff and formal attestation;
- no dry-run mutation.

Expected: no unresolved Critical, Important, or Minor finding.

- [ ] **Step 5: Record final status**

Report:

- branch and HEAD;
- exact passing-test count;
- contract attestation fields;
- exact dry-run evidence;
- no live process/output;
- one-command `run`, `dry-run`, and `status` examples.
