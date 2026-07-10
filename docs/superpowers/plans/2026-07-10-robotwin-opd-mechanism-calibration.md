# RobotWin OPD Mechanism Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backward-compatible explicit endpoint/velocity OPD objective and a two-task, shared-Stage1 calibration that compares last-step and full-rollout credit assignment.

**Architecture:** Keep the legacy Stage2 path unchanged behind `OPD_LOSS_COMPOSITION=legacy`. Put explicit scalar loss composition in a small pure module, select it from `flowmap_step.py`, and add calibration-only variants plus a guarded launcher that reuses an explicit Stage1 checkpoint and fixed held-out manifests.

**Tech Stack:** Python 3.10, PyTorch, pytest/unittest, shell launchers, single-H100 RobotWin training.

## Global Constraints

- LingBot-VA and RobotWin only; do not modify Cosmos.
- Reuse one existing Stage1 checkpoint; do not retrain Stage1.
- Use exactly two tasks, 20 train samples per task, 10 held-out samples per task, seed 0, and 750 Stage2 steps for the first gate.
- Keep existing Stage2 and final-ablation defaults unchanged.
- Use test-first red-green cycles and make small Git commits.
- Do not launch the 10-12 task final ablation in this plan.

---

### Task 1: Explicit OPD Loss Composition

**Files:**
- Create: `distillation_flowmap/opd_loss_composition.py`
- Create: `distillation_flowmap/tests/test_opd_loss_composition.py`
- Modify: `distillation_flowmap/flowmap_step.py`
- Modify: `distillation_flowmap/config_robotwin_fullfinetune_stage2_anyflow.py`

**Interfaces:**
- Produces: `compose_explicit_hybrid_opd(endpoint_video_loss, velocity_video_loss, endpoint_action_loss, beta_end_video, beta_vel_video, beta_end_action) -> ExplicitOpdLossResult`.
- Produces: config field `cfg.opd_loss_composition` with values `legacy` or `explicit_hybrid`.
- Consumes: existing scalar endpoint, same-state velocity, and action endpoint losses from `_opd_aux_transition_step`.

- [ ] **Step 1: Write failing unit tests for single-count endpoint and nonzero velocity-only gradients**

```python
def test_endpoint_is_counted_once():
    endpoint = torch.tensor(2.0, requires_grad=True)
    zero = torch.tensor(0.0, requires_grad=True)
    result = compose_explicit_hybrid_opd(
        endpoint, zero, zero,
        beta_end_video=0.5,
        beta_vel_video=0.0,
        beta_end_action=0.0,
    )
    assert torch.allclose(result.loss, torch.tensor(1.0))


def test_velocity_only_has_nonzero_loss_and_gradient():
    endpoint = torch.tensor(0.0, requires_grad=True)
    velocity = torch.tensor(3.0, requires_grad=True)
    action = torch.tensor(0.0, requires_grad=True)
    result = compose_explicit_hybrid_opd(
        endpoint, velocity, action,
        beta_end_video=0.0,
        beta_vel_video=0.25,
        beta_end_action=0.0,
    )
    result.loss.backward()
    assert torch.allclose(result.loss, torch.tensor(0.75))
    assert torch.allclose(velocity.grad, torch.tensor(0.25))
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest -q distillation_flowmap/tests/test_opd_loss_composition.py
```

Expected: collection fails because `distillation_flowmap.opd_loss_composition` does not exist.

- [ ] **Step 3: Implement the pure explicit composer**

```python
@dataclass(frozen=True)
class ExplicitOpdLossResult:
    loss: torch.Tensor
    contributions: Mapping[str, torch.Tensor]
    ratios: Mapping[str, torch.Tensor]


def compose_explicit_hybrid_opd(...):
    contributions = {
        "endpoint_video": endpoint_video_loss * float(beta_end_video),
        "velocity_video": velocity_video_loss * float(beta_vel_video),
        "endpoint_action": endpoint_action_loss * float(beta_end_action),
    }
    loss = sum(contributions.values())
    denom = sum(value.detach().abs() for value in contributions.values()).clamp(min=1e-12)
    ratios = {name: value.detach().abs() / denom for name, value in contributions.items()}
    return ExplicitOpdLossResult(loss=loss, contributions=contributions, ratios=ratios)
```

- [ ] **Step 4: Add explicit config parsing and validation**

```python
cfg.opd_loss_composition = os.environ.get("OPD_LOSS_COMPOSITION", "legacy").lower()
if cfg.opd_loss_composition not in ("legacy", "explicit_hybrid"):
    raise ValueError("OPD_LOSS_COMPOSITION must be legacy or explicit_hybrid")
```

- [ ] **Step 5: Select the explicit composer in `_opd_aux_transition_step`**

In explicit mode, use `video_transition_loss` as the one canonical endpoint
loss, use `opd_same_state_velocity_loss` as velocity, use
`opd_action_transition_loss` as action endpoint, and bypass the legacy
transition multiplier and anchor cap. Set the duplicate endpoint auxiliary
contribution to zero. Retain `opd_aux_weight` and the existing global loss
clip after composition.

- [ ] **Step 6: Run focused and existing Stage2 tests**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_opd_loss_composition.py \
  distillation_flowmap/tests/test_robotwin_stage2_endpoint.py \
  distillation_flowmap/tests/test_robotwin_diagnostics.py
```

Expected: all tests pass and legacy config assertions remain unchanged.

- [ ] **Step 7: Commit**

```bash
git add distillation_flowmap/opd_loss_composition.py \
  distillation_flowmap/flowmap_step.py \
  distillation_flowmap/config_robotwin_fullfinetune_stage2_anyflow.py \
  distillation_flowmap/tests/test_opd_loss_composition.py \
  distillation_flowmap/tests/test_robotwin_stage2_endpoint.py
git commit -m "fix: isolate explicit OPD endpoint and velocity losses"
```

### Task 2: Calibration Variants And Explicit Stage1 Reuse

**Files:**
- Modify: `distillation_flowmap/ablation/robotwin_stepwam_tasks.json`
- Modify: `distillation_flowmap/ablation/robotwin_stepwam_variants.json`
- Modify: `distillation_flowmap/ablation/launch_robotwin_stepwam_ablation.py`
- Modify: `distillation_flowmap/tests/test_robotwin_ablation_launcher.py`

**Interfaces:**
- Produces: task preset `core2` containing `place_a2b_right` and `open_microwave`.
- Produces: launcher option `--stage1-ckpt PATH`, which overrides the derived Stage1 path for Stage2 only.
- Produces: five `calib_*` variant definitions with explicit loss composition and fixed N/K rollout settings.

- [ ] **Step 1: Write failing dry-run tests**

```python
def test_calibration_full_grad_uses_small_protocol_and_explicit_stage1(tmp_path):
    result = run_calibration_dry_run(tmp_path, "calib_full_full_grad")
    manifest = parse_dry_run_manifest(result.stdout)
    assert manifest["selected_task_filter"] == ["place_a2b_right", "open_microwave"]
    assert manifest["stage1_ckpt"] == "/tmp/shared-stage1"
    assert manifest["stage2_env"]["OPD_LOSS_COMPOSITION"] == "explicit_hybrid"
    assert manifest["stage2_env"]["OPD_ROLLOUT_GRAD_MODE"] == "full"
    assert manifest["stage2_env"]["OPD_ROLLOUT_STEP_PAIRS"] == "8,4"
    assert manifest["stage2_env"]["OPD_AUX_ACTION"] == "0"
```

Add a velocity-only assertion that endpoint weight is zero, velocity weight is
one, and `OPD_ANCHOR_CAP_RATIO=-1` cannot clear its loss.

- [ ] **Step 2: Run launcher tests and verify RED**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_robotwin_ablation_launcher.py
```

Expected: failures for missing `core2`, calibration variants, and
`--stage1-ckpt`.

- [ ] **Step 3: Add `core2` and five calibration variants**

Every explicit variant sets:

```json
{
  "OPD_LOSS_COMPOSITION": "explicit_hybrid",
  "OPD_AUX_ACTION": "0",
  "OPD_ENDPOINT_AUX_WEIGHT": "0.0",
  "LOCAL_FM_WEIGHT": "0.0",
  "ACTION_LOCAL_FM_WEIGHT": "0.0",
  "OPD_TRANSITION_GROUP_WEIGHT": "1.0",
  "OPD_ANCHOR_CAP_RATIO": "-1.0",
  "OPD_ROLLOUT_STEP_PAIRS": "8,4"
}
```

Set endpoint/velocity weights to `(1,0)`, `(0,1)`, or `(1,1)` according to the
variant. Set rollout mode to `last_step` or `full` for the two full variants.

- [ ] **Step 4: Implement explicit Stage1 checkpoint override**

Add `--stage1-ckpt PATH`; when supplied, use that path in `stage2` command and
manifest while leaving the Stage1 command and existing shared-path behavior
unchanged. Validate existence for non-dry-run Stage2 execution.

- [ ] **Step 5: Run launcher and protocol tests**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_robotwin_ablation_launcher.py \
  distillation_flowmap/tests/test_robotwin_parallel_train_protocol.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add distillation_flowmap/ablation/robotwin_stepwam_tasks.json \
  distillation_flowmap/ablation/robotwin_stepwam_variants.json \
  distillation_flowmap/ablation/launch_robotwin_stepwam_ablation.py \
  distillation_flowmap/tests/test_robotwin_ablation_launcher.py
git commit -m "feat: add controlled RobotWin OPD calibration variants"
```

### Task 3: Guarded Two-Task Calibration Runner

**Files:**
- Create: `distillation_flowmap/ablation/run_opd_mechanism_calibration.sh`
- Create: `distillation_flowmap/tests/test_robotwin_opd_calibration_protocol.py`
- Modify: `distillation_flowmap/ablation/README.md`

**Interfaces:**
- Produces: one-GPU-per-variant runner with defaults `core2`, seed 0, 20 train,
  10 held-out, and 750 Stage2 steps.
- Consumes: an existing Stage1 checkpoint passed through
  `SHARED_STAGE1_CKPT`.

- [ ] **Step 1: Write a failing source-level protocol test**

```python
def test_calibration_runner_is_small_and_stage2_only():
    source = RUNNER.read_text()
    assert 'TASK_PRESET="${TASK_PRESET:-core2}"' in source
    assert 'STAGE2_STEPS="${STAGE2_STEPS:-750}"' in source
    assert 'TRAIN_SAMPLES_PER_TASK="${TRAIN_SAMPLES_PER_TASK:-20}"' in source
    assert 'HELDOUT_SAMPLES_PER_TASK="${HELDOUT_SAMPLES_PER_TASK:-10}"' in source
    assert '--stage stage2' in source
    assert '--stage1-ckpt "${SHARED_STAGE1_CKPT}"' in source
```

- [ ] **Step 2: Run the protocol test and verify RED**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_robotwin_opd_calibration_protocol.py
```

Expected: failure because the runner does not exist.

- [ ] **Step 3: Implement the guarded runner**

The script must verify the Stage1 directory exists, generate dry-run manifests
first, reject more than two selected tasks in its default mode, then launch the
five variants on GPUs 0-4. It must write PIDs and per-variant logs under a new
calibration root and never invoke Stage1 training.

- [ ] **Step 4: Run shell syntax, dry-run, and protocol tests**

Run:

```bash
bash -n distillation_flowmap/ablation/run_opd_mechanism_calibration.sh
DRY_RUN=1 bash distillation_flowmap/ablation/run_opd_mechanism_calibration.sh
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_robotwin_opd_calibration_protocol.py \
  distillation_flowmap/tests/test_robotwin_ablation_launcher.py
```

Expected: shell syntax succeeds, five dry-run manifests contain only `core2`,
and all tests pass.

- [ ] **Step 5: Commit**

```bash
git add distillation_flowmap/ablation/run_opd_mechanism_calibration.sh \
  distillation_flowmap/ablation/README.md \
  distillation_flowmap/tests/test_robotwin_opd_calibration_protocol.py
git commit -m "feat: add small-data OPD mechanism calibration runner"
```

### Task 4: Smoke Verification And Calibration Launch

**Files:**
- Runtime outputs only under `distillation_flowmap/output_robotwin_stepwam_ablation/protocol_opd_mechanism_calibration_v1/`.

**Interfaces:**
- Consumes: the five calibration variants and the existing Stage1 checkpoint.
- Produces: manifests, logs, checkpoints, metrics, and timing data for the first mechanism gate.

- [ ] **Step 1: Run complete focused verification**

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m py_compile \
  distillation_flowmap/opd_loss_composition.py \
  distillation_flowmap/flowmap_step.py \
  distillation_flowmap/ablation/launch_robotwin_stepwam_ablation.py
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_opd_loss_composition.py \
  distillation_flowmap/tests/test_robotwin_stage2_endpoint.py \
  distillation_flowmap/tests/test_robotwin_diagnostics.py \
  distillation_flowmap/tests/test_robotwin_ablation_launcher.py \
  distillation_flowmap/tests/test_robotwin_parallel_train_protocol.py \
  distillation_flowmap/tests/test_robotwin_opd_calibration_protocol.py
```

Expected: compilation succeeds and all focused tests pass.

- [ ] **Step 2: Run a 5-step smoke on endpoint, velocity, and full-gradient**

Use one task, five train samples, two held-out samples, and the explicit Stage1
checkpoint. Confirm each enabled branch has finite nonzero effective loss and
gradient, and full-gradient stays within memory.

- [ ] **Step 3: Run teacher N=4 versus N=8 preflight**

Evaluate both teacher step counts on the fixed `core2` held-out manifest. Use
N=8 only when mean teacher-to-GT endpoint MSE is at least 1% lower; otherwise
change only `OPD_ROLLOUT_STEP_PAIRS` to `4,4` in calibration variant metadata
and record the measured decision in the calibration README.

- [ ] **Step 4: Launch the 750-step five-variant calibration**

```bash
SHARED_STAGE1_CKPT=/root/nas/junjie/jj/Any_WAM/distillation_flowmap/output_robotwin_stepwam_ablation/protocol_final12_5000/shared_stage1/representative/seed_0/stage1/checkpoints/step_5000 \
  bash distillation_flowmap/ablation/run_opd_mechanism_calibration.sh
```

Expected: five single-GPU jobs start, each manifest reports two tasks and 40
training samples total, and no Stage1 process is launched.
