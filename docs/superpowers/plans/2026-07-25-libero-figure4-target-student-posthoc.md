# LIBERO Figure 4 Target-Student Post-hoc Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible single-A800 sweep of real
`target_student` checkpoints that exports the raw Figure 4 mechanism metrics,
manifest, CSV/JSONL, and paper-ready PNG/PDF plots.

**Architecture:** Reuse the existing diagnostic math and joint rollout methods.
A small pure helper module handles checkpoint discovery, schema validation,
atomic result writing, and plotting. A single-GPU CLI initializes the frozen
teacher and dataset once, creates one explicit probe bank, then replaces the
active unwrapped student with each target checkpoint in step order.

**Tech Stack:** Python 3.10, PyTorch, safetensors, Matplotlib, pytest, existing
LingBot-VA/FlowMap loaders.

## Global Constraints

- Evaluate `target_student`, not `online_student`.
- Use seed 42, `r=500`, `s=250`, and 8 teacher integration steps.
- Use only checkpoints that exist and preserve their real successful optimizer
  step.
- Use one fixed probe bank for every checkpoint.
- Do not train, call backward, alter a checkpoint, or modify training loss.
- Do not smooth or interpolate the default plots.
- Abort and report non-finite metrics instead of silently excluding them.
- Keep video/action tensor shapes, masks, and factor-1 action grid unchanged.
- Write under the run root's `outputs/mechanism_diagnostics/`.

---

### Task 1: Pure checkpoint and result helpers

**Files:**
- Create: `distillation_flowmap/mechanism_checkpoint_sweep.py`
- Create: `distillation_flowmap/tests/test_mechanism_checkpoint_sweep.py`

**Interfaces:**
- Produces:
  - `CheckpointSpec(step: int, root: Path, transformer: Path)`
  - `discover_target_checkpoints(run_root: Path) -> list[CheckpointSpec]`
  - `paper_record(step, checkpoint, metrics, valid_sample_count) -> dict`
  - `write_records_atomic(records, output_dir) -> None`
  - `plot_figure4(records, output_dir) -> dict[str, Path]`

- [ ] **Step 1: Write failing discovery tests**

Create temporary `step_1000` and `step_2000` target transformer directories
with minimal `config.json` and dummy safetensors filenames. Assert numeric
sorting, metadata agreement, missing-weight rejection, and duplicate-step
rejection.

- [ ] **Step 2: Run discovery tests and verify RED**

Run:

```bash
PYTHONPATH=. pytest -q \
  distillation_flowmap/tests/test_mechanism_checkpoint_sweep.py \
  -k checkpoint
```

Expected: import failure because `mechanism_checkpoint_sweep.py` does not
exist.

- [ ] **Step 3: Implement checkpoint discovery**

Implement an immutable `CheckpointSpec`, parse `step_<integer>`, read
`target_student/transformer/config.json`, require
`diffusion_pytorch_model.safetensors`, compare `checkpoint_step` when present,
reject duplicates, and return ascending steps.

- [ ] **Step 4: Verify discovery GREEN**

Run the command from Step 2. Expected: all checkpoint tests pass.

- [ ] **Step 5: Write failing record/output tests**

Test that:

- `paper_record` maps existing metric keys to all required paper fields;
- signed `delta_video`, clamped `r_video`, `delta_joint`, and `g_residual` are
  correct;
- two writes with the same checkpoint do not duplicate JSONL rows;
- JSONL and CSV remain sorted by step;
- one record is emitted per checkpoint;
- a non-finite metric raises an error;
- plot generation creates a PNG, vector PDF, and both panel PNGs without
  changing raw input values.

- [ ] **Step 6: Run result tests and verify RED**

Run:

```bash
PYTHONPATH=. pytest -q \
  distillation_flowmap/tests/test_mechanism_checkpoint_sweep.py \
  -k 'record or output or plot'
```

Expected: failures for missing result and plotting functions.

- [ ] **Step 7: Implement records, atomic writes, and plots**

Use temporary sibling files plus `Path.replace()` for JSONL, CSV, and manifest
data. Use Matplotlib's `Agg` backend. Panel (a) plots raw `g_anchor_l2` and
`g_comp_l2`; select a shared log y-axis only when the positive finite range
spans at least two orders of magnitude. Panel (b) plots raw `e_student`,
`e_video`, and `e_joint` on one axis. Export the combined PNG/PDF and
individual panel PNGs.

- [ ] **Step 8: Verify Task 1 GREEN**

Run:

```bash
PYTHONPATH=. pytest -q \
  distillation_flowmap/tests/test_mechanism_checkpoint_sweep.py
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 1**

```bash
git add \
  distillation_flowmap/mechanism_checkpoint_sweep.py \
  distillation_flowmap/tests/test_mechanism_checkpoint_sweep.py
git commit -m "feat: add Figure 4 checkpoint sweep outputs"
```

---

### Task 2: Explicit fixed probe-bank support

**Files:**
- Modify: `distillation_flowmap/flowmap_step.py`
- Modify: `distillation_flowmap/mechanism_diagnostics.py`
- Modify: `distillation_flowmap/tests/test_online_mechanism_diagnostics.py`
- Modify: `distillation_flowmap/tests/test_mechanism_diagnostics.py`

**Interfaces:**
- Produces:
  - `build_diagnostic_probe_noise(video_clean, action_clean, *, seed,
    batch_index, rank) -> dict[str, Tensor]`
  - Optional batch keys `_mechanism_probe_video_noise` and
    `_mechanism_probe_action_noise` consumed by
    `_compute_mechanism_diagnostic_stats_impl`.

- [ ] **Step 1: Write failing fixed-probe tests**

Add tests asserting:

- the helper returns detached video/action noise with exact input shape and
  dtype;
- the same seed produces bitwise-identical tensors;
- changing a checkpoint step argument is impossible because the helper API has
  no checkpoint-step input;
- explicit probe tensors are used instead of regenerated tensors;
- diagnostic returned tensors remain detached with `requires_grad=False`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=. pytest -q \
  distillation_flowmap/tests/test_mechanism_diagnostics.py \
  distillation_flowmap/tests/test_online_mechanism_diagnostics.py \
  -k probe
```

Expected: import or assertion failures for missing fixed-probe support.

- [ ] **Step 3: Implement the probe helper and override**

Move deterministic noise construction into
`build_diagnostic_probe_noise`. Preserve the current online seed behavior by
calling it with `diagnostic_seed(...)`. In the diagnostic implementation,
prefer explicit batch probe tensors when both are present, validate their
shape/dtype/device compatibility, detach them, and otherwise retain the
existing online path.

- [ ] **Step 4: Verify Task 2 GREEN and regressions**

Run:

```bash
PYTHONPATH=. pytest -q \
  distillation_flowmap/tests/test_mechanism_diagnostics.py \
  distillation_flowmap/tests/test_online_mechanism_diagnostics.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add \
  distillation_flowmap/flowmap_step.py \
  distillation_flowmap/mechanism_diagnostics.py \
  distillation_flowmap/tests/test_mechanism_diagnostics.py \
  distillation_flowmap/tests/test_online_mechanism_diagnostics.py
git commit -m "feat: support fixed mechanism probe banks"
```

---

### Task 3: Single-GPU target-student sweep CLI

**Files:**
- Create: `distillation_flowmap/eval_libero_figure4_mechanism.py`
- Create: `distillation_flowmap/run_libero_figure4_mechanism_1gpu.sh`
- Create: `distillation_flowmap/tests/test_eval_libero_figure4_mechanism.py`

**Interfaces:**
- CLI consumes:
  - `--run-root`
  - `--teacher-model-path`
  - `--dataset-path`
  - `--output-dir`
  - `--seed 42`
  - `--r 500`
  - `--s 250`
  - `--teacher-steps 8`
  - optional `--steps` for one-checkpoint smoke testing
- Produces all artifacts specified in the design.

- [ ] **Step 1: Write failing CLI/config tests**

Test pure CLI helpers for:

- default target role is `target_student`;
- checkpoint filtering preserves sorted real steps;
- config disables training logging, target-copy loading, optimizer resume, and
  training-only compilation;
- probe metadata records dataset index, shapes, seed, `r`, `s`, teacher steps,
  and action mask rule;
- manifest records checkpoint role and counterfactual `E_video` semantics;
- `reproduce.sh` quotes paths and includes the exact resolved options.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=. pytest -q \
  distillation_flowmap/tests/test_eval_libero_figure4_mechanism.py
```

Expected: import failure because the CLI module does not exist.

- [ ] **Step 3: Implement lightweight evaluator initialization**

Import the existing factor-1 Stage-II config, copy it, and override:

- rank/local-rank/world-size to one process;
- W&B and TensorBoard disabled;
- no target-student copy;
- no optimizer/scheduler restore;
- gradient checkpointing and compile disabled for inference;
- mechanism seed, `r`, `s`, teacher steps, and one diagnostic batch;
- the first discovered checkpoint as the temporary resume source.

Initialize `FlowMapDistiller` once, delete optimizer/scheduler references, and
retain only the teacher, schedulers, dataset, diagnostic methods, and active
student.

- [ ] **Step 4: Implement target-student replacement**

Load each checkpoint with the existing Wan transformer loader, apply
`setup_flowmap_model` and `patch_model_forward`, restore FlowMap delta weights
from safetensors, set inference flags, move it to the single CUDA device, set
`eval()`, disable gradients, assign it as `trainer.student`, and clear any
cached diagnostic context that refers to the previous student.

Delete the previous student and empty CUDA cache before loading the next one.
Verify checkpoint config declares `action_downsample_factor=1` and
`continuous_action_v1`.

- [ ] **Step 5: Implement probe creation and metric loop**

Build one dataset batch, create fixed probe noise once, save CPU tensors and
metadata, and attach the exact same tensors to every checkpoint batch. Run
`_compute_mechanism_diagnostic_stats` under `torch.inference_mode()`, reduce
with existing helpers, reject invalid/non-finite aggregates, convert to a paper
record, and atomically update outputs after each checkpoint.

Track elapsed wall time and `torch.cuda.max_memory_allocated()`. Generate the
manifest, figures, CSV, JSONL, run log, and reproduce script after each
successful point so a partial run remains inspectable.

- [ ] **Step 6: Implement shell launcher**

The shell script validates all required paths, selects one GPU through
`CUDA_VISIBLE_DEVICES`, activates the supplied `PYTHON`, forces offline model
loading, and forwards smoke/full options to the Python CLI.

- [ ] **Step 7: Verify CLI GREEN**

Run:

```bash
PYTHONPATH=. pytest -q \
  distillation_flowmap/tests/test_eval_libero_figure4_mechanism.py \
  distillation_flowmap/tests/test_mechanism_checkpoint_sweep.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 3**

```bash
git add \
  distillation_flowmap/eval_libero_figure4_mechanism.py \
  distillation_flowmap/run_libero_figure4_mechanism_1gpu.sh \
  distillation_flowmap/tests/test_eval_libero_figure4_mechanism.py
git commit -m "feat: add single-GPU LIBERO Figure 4 sweep"
```

---

### Task 4: Static and single-checkpoint smoke verification

**Files:**
- Modify only if a failing test exposes an implementation defect.

- [ ] **Step 1: Run focused and existing diagnostics tests**

Run:

```bash
PYTHONPATH=. pytest -q \
  distillation_flowmap/tests/test_mechanism_diagnostics.py \
  distillation_flowmap/tests/test_online_mechanism_diagnostics.py \
  distillation_flowmap/tests/test_mechanism_checkpoint_sweep.py \
  distillation_flowmap/tests/test_eval_libero_figure4_mechanism.py
```

Expected: zero failures.

- [ ] **Step 2: Run syntax and shell checks**

Run:

```bash
python -m py_compile \
  distillation_flowmap/mechanism_checkpoint_sweep.py \
  distillation_flowmap/eval_libero_figure4_mechanism.py
bash -n distillation_flowmap/run_libero_figure4_mechanism_1gpu.sh
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 3: Run step-1000 smoke sweep**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
bash distillation_flowmap/run_libero_figure4_mechanism_1gpu.sh \
  --run-root /kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_lingbotva_stage2_video_only_opd_universal_from_stage1_step2000_steps10000_factor1-deployment-align-20260724 \
  --steps 1000 \
  --seed 42 \
  --r 500 \
  --s 250 \
  --teacher-steps 8
```

Expected: one JSONL row for step 1000, all required artifacts present, and no
checkpoint mutation.

- [ ] **Step 4: Verify smoke artifacts**

Check manifest schema, JSONL/CSV equality, detached finite metrics, figure PDF
page count, GPU model, peak memory, and reproduce command. Compare checkpoint
config and weight file hashes before and after the smoke run.

- [ ] **Step 5: Commit smoke fixes if required**

Commit only changes supported by a failing test and a verified green rerun.

---

### Task 5: Formal sweep and final audit

**Files:**
- Runtime artifacts only under the run root.

- [ ] **Step 1: Rediscover final checkpoints**

Require the real discovered list. If step 10000 does not yet exist, report that
the training run is still incomplete and either wait or run a clearly labeled
partial sweep without presenting it as final.

- [ ] **Step 2: Run the full single-GPU sweep**

Run the launcher from Task 4 without `--steps`. Preserve the complete `run.log`.

- [ ] **Step 3: Validate output completeness**

Require:

- one JSONL and CSV row per discovered checkpoint;
- steps strictly increasing and equal between manifest, JSONL, and CSV;
- no NaN/Inf;
- all required files exist and are non-empty;
- combined PDF is vector output and readable;
- no checkpoint hashes changed.

- [ ] **Step 4: Run the full focused regression suite again**

Repeat Task 4 Step 1 after the formal run. Expected: zero failures.

- [ ] **Step 5: Report evidence**

Report modified files and commits, exact checkpoint paths/steps, trend summary,
the four requested hypothesis outcomes, absolute artifact paths, raw-data
paths, reproduction command, GPU, peak memory, runtime, and unmet requirements.
Do not alter or hide negative findings.
