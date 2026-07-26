# Action FP32 EMA and Gradient Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent BF16 EMA from freezing action-specific target parameters and run a 50–100-step probe that measures whether the existing action supervision produces healthy gradients.

**Architecture:** Add a generic selective FP32 EMA accumulator that owns persistent FP32 shadows only for selected parameters and delegates non-selected parameters to the existing EMA path. Integrate it into the FlowMap trainer for parameters classified as `action`, checkpoint one sharded state file per rank, and expose branch-gradient and EMA-health metrics without enabling any DanceOPD action loss.

**Tech Stack:** Python, PyTorch/FSDP1, safetensors checkpoints, pytest, TensorBoard/W&B metrics, Bash launchers.

## Global Constraints

- Keep `opd_aux_action=False` and `opd_joint_action_rollout=False`.
- Do not add a DanceOPD action endpoint or change existing action-loss weights.
- Keep BF16 model forward/checkpoint weights; only the selected EMA accumulator is FP32.
- A repaired non-reset resume must fail if its per-rank FP32 EMA state is missing or incompatible.
- Preserve all unrelated dirty worktree changes.
- The short probe logs gradients but never terminates a job automatically.

---

### Task 1: Selective FP32 EMA Core

**Files:**
- Modify: `distillation/ema.py`
- Create: `distillation_flowmap/tests/test_action_fp32_ema.py`

**Interfaces:**
- Produces: `SelectiveFp32EMA(target_named_parameters, source_named_parameters, selected_names)`
- Produces: `SelectiveFp32EMA.update(rate) -> dict[str, float]`
- Produces: `SelectiveFp32EMA.state_dict() -> dict`
- Produces: `SelectiveFp32EMA.load_state_dict(state: Mapping) -> None`

- [ ] **Step 1: Write a failing BF16 dead-zone regression test**

```python
def test_selective_fp32_ema_accumulates_updates_lost_by_bf16():
    target = torch.nn.Parameter(torch.tensor([0.03125], dtype=torch.bfloat16))
    source = torch.nn.Parameter(torch.tensor([0.03125], dtype=torch.bfloat16))
    ema = SelectiveFp32EMA(
        [("action_proj_out.weight", target)],
        [("action_proj_out.weight", source)],
        {"action_proj_out.weight"},
    )
    source.data.copy_(torch.tensor([0.03173828125], dtype=torch.bfloat16))
    for _ in range(200):
        ema.update(0.99)
    assert target.item() != torch.tensor(0.03125, dtype=torch.bfloat16).item()
```

- [ ] **Step 2: Run the regression test and verify it fails**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q distillation_flowmap/tests/test_action_fp32_ema.py \
  -k accumulates_updates_lost_by_bf16
```

Expected: FAIL because `SelectiveFp32EMA` does not exist.

- [ ] **Step 3: Add validation and FP32-shadow tests**

Test that constructor rejects mismatched names/shapes, `rate` outside `[0,1]`,
non-selected parameters use the existing BF16 update behavior, metrics are
finite, and `state_dict`/`load_state_dict` preserve the next update exactly.

- [ ] **Step 4: Implement the minimal selective accumulator**

Implement:

```python
class SelectiveFp32EMA:
    STATE_VERSION = 1

    def __init__(self, target_named_parameters, source_named_parameters, selected_names):
        self._pairs = _validated_named_parameter_pairs(
            target_named_parameters, source_named_parameters
        )
        self._selected_names = tuple(
            name for name, _, _ in self._pairs if name in set(selected_names)
        )
        if not self._selected_names:
            raise ValueError("SelectiveFp32EMA requires at least one selected parameter")
        self._masters = {
            name: target.detach().float().clone()
            for name, target, _ in self._pairs
            if name in self._selected_names
        }
        self._initial_masters = {
            name: value.clone() for name, value in self._masters.items()
        }

    @torch.no_grad()
    def update(self, rate):
        if not 0.0 <= float(rate) <= 1.0:
            raise ValueError("EMA rate must lie in [0, 1]")
        selected = set(self._selected_names)
        nonselected_target = []
        nonselected_source = []
        moved = 0
        master_delta_max = 0.0
        target_source_max = 0.0
        for name, target, source in self._pairs:
            if name not in selected:
                nonselected_target.append(target)
                nonselected_source.append(source)
                continue
            master = self._masters[name]
            before = master.clone()
            master.lerp_(source.detach().float(), 1.0 - float(rate))
            target.copy_(master.to(dtype=target.dtype))
            delta = float((master - before).abs().max().item())
            moved += int(delta > 0.0)
            master_delta_max = max(master_delta_max, delta)
            target_source_max = max(
                target_source_max,
                float((target.detach().float() - source.detach().float()).abs().max().item()),
            )
        if nonselected_target:
            update_ema(nonselected_target, nonselected_source, rate=float(rate))
        return {
            "selected_count": float(len(self._selected_names)),
            "selected_moved_count": float(moved),
            "master_delta_max": master_delta_max,
            "target_source_max": target_source_max,
        }
```

State serialization must store version, selected names, FP32 masters, and FP32
initial masters on CPU. Loading must validate version, exact names, shapes, and
finite tensors before copying to the local parameter device.

- [ ] **Step 5: Run core EMA tests**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q distillation_flowmap/tests/test_action_fp32_ema.py
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add distillation/ema.py distillation_flowmap/tests/test_action_fp32_ema.py
git commit -m "fix: accumulate action EMA in fp32"
```

---

### Task 2: Trainer Integration and Sharded Resume

**Files:**
- Modify: `distillation_flowmap/flowmap_trainer.py`
- Modify: `distillation_flowmap/tests/test_action_fp32_ema.py`

**Interfaces:**
- Consumes: `SelectiveFp32EMA`
- Produces: `FlowMapDistiller._save_action_ema_state(step_dir: Path) -> None`
- Produces: `FlowMapDistiller._restore_action_ema_state() -> None`

- [ ] **Step 1: Write failing trainer-integration tests**

Use small fake named-parameter modules and a trainer created with `__new__`.
Verify:

```python
def test_trainer_selects_only_action_branch_parameters():
    module = torch.nn.Module()
    module.action_embedder = torch.nn.Linear(2, 2)
    module.action_proj_out = torch.nn.Linear(2, 2)
    module.shared = torch.nn.Linear(2, 2)
    selected = {
        name for name, _ in module.named_parameters()
        if classify_parameter_branch(name) == "action"
    }
    assert selected
    assert all(
        "action_embedder" in name or "action_proj_out" in name
        for name in selected
    )
    assert all("shared" not in name for name in selected)

def test_repaired_resume_requires_rank_state(tmp_path):
    trainer = FlowMapDistiller.__new__(FlowMapDistiller)
    trainer.step = 50
    trainer._resume_ckpt_dir = str(tmp_path)
    trainer.config = SimpleNamespace(
        reset_resume_step=False,
        rank=0,
        world_size=1,
    )
    with pytest.raises(FileNotFoundError, match="action EMA"):
        trainer._restore_action_ema_state()

def test_stage_boundary_reset_initializes_new_state(tmp_path):
    trainer = FlowMapDistiller.__new__(FlowMapDistiller)
    trainer.step = 0
    trainer._resume_ckpt_dir = str(tmp_path)
    trainer.config = SimpleNamespace(
        reset_resume_step=True,
        rank=0,
        world_size=1,
    )
    assert trainer._restore_action_ema_state() is False
```

- [ ] **Step 2: Run integration tests and verify failure**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q distillation_flowmap/tests/test_action_fp32_ema.py -k trainer
```

Expected: FAIL because trainer hooks do not exist.

- [ ] **Step 3: Integrate after FSDP wrapping**

When `config.action_ema_fp32` is true and `target_student` exists:

```python
target_named = list(self.target_student.named_parameters())
source_named = list(self.student.named_parameters())
selected_names = {
    name for name, _ in target_named
    if classify_parameter_branch(name) == "action"
}
self._action_ema = SelectiveFp32EMA(
    target_named, source_named, selected_names
)
```

If this is a genuine resume (`self.step > 0` and
`reset_resume_step=False`), restore the local rank state. If it is a fresh
Stage-2 boundary (`reset_resume_step=True`), initialize from the loaded Stage-1
target.

- [ ] **Step 4: Replace only the configured EMA update**

At the optimizer boundary:

```python
if self._action_ema is not None:
    self._last_action_ema_stats = self._action_ema.update(ema_decay)
else:
    update_ema(
        self.target_student.parameters(),
        self.student.parameters(),
        rate=ema_decay,
    )
```

Do not change warmup or decay scheduling.

- [ ] **Step 5: Save and restore one state file per rank**

Save:

```text
checkpoints/step_<N>/action_ema_fp32/rank_<RANK:05d>.pt
```

Each file contains `world_size`, `rank`, and the selective EMA `state_dict`.
Validate world size and rank on load. All ranks write their own file; rank 0
also writes `manifest.json` with version, world size, selected names, and
checkpoint step.

- [ ] **Step 6: Log EMA health metrics**

Add:

```text
ema_action/selected_count
ema_action/selected_moved_count
ema_action/master_delta_max
ema_action/target_source_max
```

Non-finite values must raise before checkpoint save.

- [ ] **Step 7: Run trainer and EMA tests**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_action_fp32_ema.py \
  distillation_flowmap/tests/test_runtime_metadata.py
```

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add distillation_flowmap/flowmap_trainer.py \
  distillation_flowmap/tests/test_action_fp32_ema.py
git commit -m "feat: checkpoint action fp32 EMA state"
```

---

### Task 3: Configuration, Launcher, and Gradient Metrics

**Files:**
- Modify: `distillation_flowmap/config_libero_fullfinetune_stage2_video_only_opd.py`
- Modify: `distillation_flowmap/run_libero_video_only_opd_stage2_8gpu.sh`
- Modify: `distillation_flowmap/flowmap_trainer.py`
- Modify: `distillation_flowmap/tests/test_libero_video_only_opd_config.py`
- Modify: `distillation_flowmap/tests/test_run_libero_video_only_opd_stage2_8gpu.py`

**Interfaces:**
- Produces environment flag: `ACTION_EMA_FP32=1`
- Produces environment flag: `ENABLE_GRAD_BRANCH_DIAGNOSTICS=1`
- Produces metrics: `grad_norm/action_to_shared`,
  `grad_norm/action_to_video`

- [ ] **Step 1: Write failing configuration tests**

Assert:

```python
assert cfg.action_ema_fp32 is True
assert cfg.enable_grad_branch_diagnostics is True
assert cfg.opd_aux_action is False
assert cfg.opd_joint_action_rollout is False
```

The launcher dry-run contract must include:

```text
ACTION_EMA_FP32=1
ENABLE_GRAD_BRANCH_DIAGNOSTICS=1
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_libero_video_only_opd_config.py \
  distillation_flowmap/tests/test_run_libero_video_only_opd_stage2_8gpu.py
```

Expected: FAIL because the flags are not wired.

- [ ] **Step 3: Add explicit configuration**

In the LIBERO video-only Stage-2 config:

```python
cfg.action_ema_fp32 = _env_bool("ACTION_EMA_FP32", True)
cfg.enable_grad_branch_diagnostics = _env_bool(
    "ENABLE_GRAD_BRANCH_DIAGNOSTICS", True
)
```

Keep the hard contracts:

```python
cfg.opd_aux_action = False
cfg.opd_joint_action_rollout = False
```

- [ ] **Step 4: Add launcher propagation**

Export both flags in the launch environment record and child training process,
defaulting each to `1`. Include them in `launch_env.txt`.

- [ ] **Step 5: Add safe branch-gradient ratios**

When branch diagnostics exist:

```python
eps = 1e-12
log_dict["grad_norm/action_to_shared"] = (
    grad_branch_norms["action"] / max(grad_branch_norms["shared"], eps)
)
log_dict["grad_norm/action_to_video"] = (
    grad_branch_norms["action"] / max(grad_branch_norms["video"], eps)
)
```

Keep raw branch norms as the primary evidence.

- [ ] **Step 6: Run focused tests**

Run the two configuration/launcher tests plus
`test_action_fp32_ema.py`. Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add \
  distillation_flowmap/config_libero_fullfinetune_stage2_video_only_opd.py \
  distillation_flowmap/run_libero_video_only_opd_stage2_8gpu.sh \
  distillation_flowmap/flowmap_trainer.py \
  distillation_flowmap/tests/test_libero_video_only_opd_config.py \
  distillation_flowmap/tests/test_run_libero_video_only_opd_stage2_8gpu.py
git commit -m "feat: expose action EMA and gradient diagnostics"
```

---

### Task 4: Verification and 100-Step Probe Handoff

**Files:**
- Modify only if verification finds a defect in files from Tasks 1–3.

**Interfaces:**
- Consumes the repaired launcher and metrics.
- Produces a verified 100-step launch command and a continue/stop checklist.

- [ ] **Step 1: Run the complete focused test set**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_action_fp32_ema.py \
  distillation_flowmap/tests/test_libero_video_only_opd_config.py \
  distillation_flowmap/tests/test_run_libero_video_only_opd_stage2_8gpu.py \
  distillation_flowmap/tests/test_runtime_metadata.py \
  distillation_flowmap/tests/test_danceopd_runtime_contract.py \
  distillation_flowmap/tests/test_libero_danceopd_semantic_rollout.py
```

- [ ] **Step 2: Run launcher contract validation**

Run the Stage-2 launcher with its existing dry-run/check-only mechanism and
verify:

```text
OPD_AUX_ACTION=0
OPD_JOINT_ACTION_ROLLOUT=0
ACTION_EMA_FP32=1
ENABLE_GRAD_BRANCH_DIAGNOSTICS=1
MAX_TRAIN_STEPS=100
SAVE_INTERVAL=50
LOG_INTERVAL=1
```

- [ ] **Step 3: Provide the eight-GPU probe command**

Use a new output tag/root, resume from Stage-1 step 2000 with reset step and
optimizer state disabled, run 100 optimizer steps, save at steps 50 and 100,
and keep `VIDEO_ACTION_BRIDGE=0` so EMA precision is the only changed training
variable.

- [ ] **Step 4: Define the probe readout**

At steps 10, 50, and 100 report:

- median/p90 raw action, video, and shared branch gradient norms;
- action/shared and action/video ratios;
- fraction of logged steps with non-zero finite action gradients;
- action consistency and GT regression trends;
- FP32 EMA selected/moved counts and target/source maximum difference;
- online and target action-parameter changes from Stage-1.

Recommend stopping before full training if action gradients are zero/non-finite,
FP32 masters do not move, or action losses worsen persistently. Do not
terminate automatically.

- [ ] **Step 5: Final diff and status review**

Run:

```bash
git diff --check
git status --short
```

Confirm unrelated pre-existing modifications remain untouched.
