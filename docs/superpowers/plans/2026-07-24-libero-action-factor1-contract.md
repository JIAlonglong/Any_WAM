# LIBERO Action Factor-1 Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the new LingBot-VA LIBERO video-only-OPD Stage-2 family train, save, serve, and evaluate all 16 continuous action positions with `action_downsample_factor=1`.

**Architecture:** The Stage-2 wrapper owns the strict factor-1 experiment contract. A small runtime-metadata module centralizes persistence and checkpoint-first resolution so the trainer and server cannot drift, while legacy metadata-free checkpoints retain the LIBERO job-config fallback of 4.

**Tech Stack:** Python 3.10, PyTorch, EasyDict configuration, Bash launchers, pytest.

## Global Constraints

- The LIBERO video-only-OPD Stage-2 wrapper always resolves `action_downsample_factor=1`.
- Action OPD stays disabled; the detached generated-video/action teacher-forcing bridge is unchanged.
- New checkpoints persist the resolved factor in `transformer/config.json`.
- Historical checkpoints without the field continue to use the existing LIBERO fallback of 4.
- Matched 1/1, 2/2, and 4/4 evaluation remains the public evaluation interface.

---

### Task 1: Enforce Factor 1 in Training and Launcher

**Files:**
- Modify: `distillation_flowmap/config_libero_fullfinetune_stage2_video_only_opd.py`
- Modify: `distillation_flowmap/run_libero_video_only_opd_stage2_8gpu.sh`
- Test: `distillation_flowmap/tests/test_libero_video_only_opd_config.py`
- Test: `distillation_flowmap/tests/test_run_libero_video_only_opd_stage2_8gpu.py`

**Interfaces:**
- Consumes: shared base config field `cfg.action_downsample_factor`.
- Produces: strict wrapper value `cfg.action_downsample_factor == 1` and launcher environment `ACTION_DOWNSAMPLE_FACTOR=1`.

- [ ] **Step 1: Write failing config and launcher tests**

Add assertions that the wrapper returns factor 1 by default and when
`ACTION_DOWNSAMPLE_FACTOR=4` is present. Add dry-run assertions for
`ACTION_DOWNSAMPLE_FACTOR=1` and `assert cfg.action_downsample_factor == 1`
inside launcher preflight.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_libero_video_only_opd_config.py \
  distillation_flowmap/tests/test_run_libero_video_only_opd_stage2_8gpu.py
```

Expected: failures showing the wrapper and launcher still resolve factor 4.

- [ ] **Step 3: Implement the strict wrapper and launcher contract**

After copying the base config, set:

```python
cfg.action_downsample_factor = 1
```

In the launcher environment and preflight add:

```bash
"ACTION_DOWNSAMPLE_FACTOR=1"
```

```python
assert cfg.action_downsample_factor == 1
```

- [ ] **Step 4: Run tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add \
  distillation_flowmap/config_libero_fullfinetune_stage2_video_only_opd.py \
  distillation_flowmap/run_libero_video_only_opd_stage2_8gpu.sh \
  distillation_flowmap/tests/test_libero_video_only_opd_config.py \
  distillation_flowmap/tests/test_run_libero_video_only_opd_stage2_8gpu.py
git commit -m "fix: train LIBERO continuous actions without downsampling"
```

### Task 2: Persist and Resolve Runtime Metadata

**Files:**
- Create: `distillation_flowmap/runtime_metadata.py`
- Modify: `distillation_flowmap/flowmap_trainer.py`
- Modify: `wan_va/wan_va_server.py`
- Create: `distillation_flowmap/tests/test_runtime_metadata.py`

**Interfaces:**
- Produces: `flowmap_runtime_metadata(config) -> dict[str, object]`.
- Produces: `resolve_action_downsample_factor(checkpoint_config, fallback) -> int`.
- Consumes: trainer config and transformer checkpoint JSON dictionary.

- [ ] **Step 1: Write failing pure-function tests**

Test that metadata includes factor 1, checkpoint value 1 overrides fallback 4,
missing checkpoint metadata returns fallback 4, and non-positive values raise
`ValueError`.

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_runtime_metadata.py
```

Expected: import failure because `runtime_metadata.py` does not exist.

- [ ] **Step 3: Implement the pure metadata helpers**

Create:

```python
def flowmap_runtime_metadata(config):
    return {
        "num_train_timesteps": getattr(config, "num_train_timesteps", 1000),
        "snr_shift": getattr(config, "snr_shift", 1.0),
        "action_snr_shift": getattr(config, "action_snr_shift", 1.0),
        "action_downsample_factor": int(
            getattr(config, "action_downsample_factor", 1)
        ),
    }


def resolve_action_downsample_factor(checkpoint_config, fallback):
    factor = int(checkpoint_config.get("action_downsample_factor", fallback))
    if factor <= 0:
        raise ValueError("action_downsample_factor must be positive")
    return factor
```

- [ ] **Step 4: Integrate trainer and server**

Use `config_dict.update(flowmap_runtime_metadata(self.config))` in
`_save_checkpoint`. Use `resolve_action_downsample_factor` in
`wan_va_server.py` while retaining the job-config fallback.

- [ ] **Step 5: Run metadata and server import regressions**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_runtime_metadata.py \
  distillation_flowmap/tests/test_libero_video_only_opd_config.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add \
  distillation_flowmap/runtime_metadata.py \
  distillation_flowmap/flowmap_trainer.py \
  wan_va/wan_va_server.py \
  distillation_flowmap/tests/test_runtime_metadata.py
git commit -m "fix: persist LIBERO action runtime metadata"
```

### Task 3: Prove Full-Frame Inference and Run Regression

**Files:**
- Modify: `distillation_flowmap/tests/test_runtime_metadata.py`
- Test: `distillation_flowmap/tests/test_video_action_bridge.py`
- Test: `distillation_flowmap/tests/test_libero_danceopd_semantic_rollout.py`
- Test: `distillation_flowmap/tests/test_danceopd_runtime_contract.py`

**Interfaces:**
- Consumes: `flowmap_inference(..., action_downsample_factor=1)`.
- Produces: regression evidence that factor 1 preserves every action frame.

- [ ] **Step 1: Add an inference characterization test**

Use AST/source inspection to characterize the existing generic inference
implementation: factor 1 yields
`current_action[:, :, ::action_ds]` with `action_ds == 1`, and that the cache
receives the complete action tensor. This test must fail if the inference path
reintroduces a hard-coded factor 4.

- [ ] **Step 2: Run the focused test**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_runtime_metadata.py
```

Expected: pass with the current generic inference implementation; no
production inference rewrite is required.

- [ ] **Step 3: Run the full deployment-alignment regression**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_runtime_metadata.py \
  distillation_flowmap/tests/test_run_libero_video_only_opd_stage2_8gpu.py \
  distillation_flowmap/tests/test_video_action_bridge.py \
  distillation_flowmap/tests/test_libero_danceopd_semantic_rollout.py \
  distillation_flowmap/tests/test_libero_video_only_opd_config.py \
  distillation_flowmap/tests/test_danceopd_query.py \
  distillation_flowmap/tests/test_danceopd_runtime_contract.py \
  distillation_flowmap/tests/test_online_mechanism_diagnostics.py \
  distillation_flowmap/tests/test_mechanism_diagnostics.py
```

Expected: zero failures.

- [ ] **Step 4: Verify the launcher and worktree**

Run:

```bash
git diff --check
git status --short --branch
```

Expected: no uncommitted files and branch
`fix/libero-stage2-deployment-alignment`.
