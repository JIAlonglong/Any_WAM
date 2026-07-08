# RobotWin StepWAM Mini-Ablation Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a protocol-controlled RobotWin mini-ablation that tests whether OPD endpoint and same-state velocity provide real signal, without confounding the result with Stage 1 randomness, train-cache memorization, Cosmos, PFM, or perceptual auxiliary losses.

**Architecture:** Train one shared LingBot-VA Stage 1 checkpoint per seed on the train split, then branch only the Stage 2 OPD fine-tuning variants from that checkpoint. Build deterministic train and held-out eval manifests, reuse identical held-out `t/r/noise/teacher-target` samples across variants, and report seed-aggregated trends against the `w_o_opd` baseline.

**Tech Stack:** `distillation_flowmap/ablation/launch_robotwin_stepwam_ablation.py`, `distillation_flowmap/rollout_eval_stage2.py`, `distillation_flowmap/rollout_eval_video_stage2.py`, `distillation_flowmap/flowmap_step.py`, `distillation_flowmap/flowmap_trainer.py`, `distillation_flowmap/ablation/summarize_robotwin_ablation.py`, `distillation_flowmap/tests`, PyTorch, torchrun, tmux, 8x H100.

## Global Constraints

- Teacher: LingBot-VA only.
- Benchmark: RobotWin representative mini subset only; do not run 10-12 task full ablation yet.
- Cosmos: excluded from component ablation; Cosmos remains for main result and cross-teacher generality only.
- PFM/perceptual auxiliary: excluded until OPD endpoint/velocity mechanism is understood.
- Variants: exactly `w_o_opd`, `endpoint_only_opd`, `velocity_only_opd`, `full_stepwam`.
- Stage 1: shared within each seed. All four Stage 2 variants for a seed must resume from the same Stage 1 checkpoint path.
- Stage 2: variants compare only OPD fine-tuning; no variant-specific Stage 1 retraining.
- Eval: held-out only. Eval must not use train indices, train cache, or variant-specific sampled `t/r` pairs.
- Randomness: run 3 seeds for the four core variants unless explicitly reduced for smoke testing.
- Reporting language: write only trend statements. Do not claim final ablation conclusions from the mini-ablation.
- Commit after every independently verified implementation phase.

---

## Mini Task Set

Use the four-task core set first. Expand to the six-task set only after smoke and core4 mini runs are stable.

| group | task | reason |
| --- | --- | --- |
| core4 | `place_a2b_right` | pick-and-place, current screen anchor |
| core4 | `stack_bowls_three` | stacking / ordering |
| core4 | `open_microwave` | articulated / contact-rich |
| core4 | `pick_dual_bottles` | bimanual / coordination |
| core6-extra | `blocks_ranking_size` | ranking / visual-semantic ordering |
| core6-extra | `place_burger_fries` | longer-horizon hard task |

The launcher must validate task existence in the selected dataset before training. If a task is absent, the run must fail before starting torchrun and print the missing task names.

## Train / Held-Out Split

Create deterministic manifests under the ablation root:

```text
<root>/protocol/manifests/core4_seed_<seed>/
  train_manifest.json
  heldout_eval_manifest.json
  eval_pairs.json
```

Manifest rules:

- Split by stable dataset sample identity when available; otherwise split by deterministic per-task index after dataset ordering is fixed.
- Default train samples per task: `80`.
- Default held-out eval samples per task: `20`.
- Smoke split: `5` train samples and `5` held-out eval samples for one task.
- Train and held-out ids must be disjoint per task.
- `heldout_eval_manifest.json` stores the exact sample ids used for offline metrics and video export.
- `eval_pairs.json` stores the exact `(t, r, pair_seed)` list used by all variants.
- Teacher endpoint cache generated from the held-out manifest must be shared by all variants and seeds that use the same teacher, tasks, split, pairs, teacher steps, and cfg scale.

## Core Variants

| variant | Stage 1 source | Stage 2 OPD | required env |
| --- | --- | --- | --- |
| `w_o_opd` | shared seed checkpoint | disabled | `USE_OPD_AUX=0` |
| `endpoint_only_opd` | shared seed checkpoint | endpoint only | `USE_OPD_AUX=1 OPD_ENDPOINT_AUX_WEIGHT=0.1 OPD_SAME_STATE_VELOCITY_WEIGHT=0.0` |
| `velocity_only_opd` | shared seed checkpoint | same-state velocity only | `USE_OPD_AUX=1 OPD_ENDPOINT_AUX_WEIGHT=0.0 OPD_SAME_STATE_VELOCITY_WEIGHT=0.1 VIDEO_TRANSITION_WEIGHT=0.0 OPD_AUX_ACTION=0` |
| `full_stepwam` | shared seed checkpoint | endpoint + same-state velocity | `USE_OPD_AUX=1 OPD_ENDPOINT_AUX_WEIGHT=0.1 OPD_SAME_STATE_VELOCITY_WEIGHT=0.1` |

`velocity_only_opd` keeps `OPD_AUX_ACTION=0` until a true same-state action velocity target exists. Otherwise action endpoint OPD would leak endpoint loss into the velocity-only ablation.

## OPD Definition To Preserve

For each held-out eval sample and sampled pair `(t, r)`:

- Student rollout starts from the same `x_t` and rolls out `K` student steps to `x_r^S`.
- Endpoint teacher target is an independent teacher rollout from the same `x_t` to `r`, producing `x_r^T`.
- Endpoint loss aligns denoised endpoints:

```text
x0_S = x_r^S - sigma_r * S(sg(x_r^S), r)
x0_T = x_r^T - sigma_r * T(x_r^T, r)
L_endpoint = ||x0_S - x0_T||
```

- Velocity regularizer queries teacher and student at the student-induced endpoint:

```text
v_S = S(sg(x_r^S), r)
v_T = T(sg(x_r^S), r)
L_velocity = ||v_S - v_T||
```

The shared teacher endpoint cache can store `x_r^T` and `T(x_r^T, r)`. The same-state velocity teacher query depends on each variant's `x_r^S`, so it must be computed in a separate low-memory teacher pass after student endpoints are written.

## Metrics

Offline metrics must be computed separately on train and held-out splits, with held-out as the primary table.

Required held-out metrics:

- denoised endpoint error: `||x0_S - x0_T||`
- same-state velocity error: `||S(sg(x_r^S), r) - T(sg(x_r^S), r)||`
- K-step rollout drift over the inference schedule
- action endpoint error
- video endpoint error
- existing `t1000_x`, `t1000_v`, and `t1000_action` screen metrics retained for continuity
- representative rollout videos and contact sheets from held-out samples
- optional closed-loop SR or success proxy if the offline pipeline is stable

Required training diagnostics:

- `L_endpoint_video`
- `L_endpoint_action`
- `L_velocity_video`
- `L_velocity_action`
- weighted endpoint contribution after `beta_end`
- weighted velocity contribution after `beta_vel`
- effective `beta_end / beta_vel`
- video/action modality weights
- branch-level gradient norm at a low-frequency interval

Gradient diagnostics should be logged every `50` or `100` steps, not every step.

## Report Output

Generate:

```text
<root>/reports/mini_ablation_summary.csv
<root>/reports/mini_ablation_summary.json
<root>/reports/mini_ablation_summary.md
<root>/reports/videos/
```

Report contents:

- per-task table
- mean +/- std across seeds
- delta vs `w_o_opd` for each metric
- seed variance
- train vs held-out gap
- representative rollout video/contact-sheet paths
- short trend-only notes

Every run must save:

- config snapshot
- git hash
- checkpoint path
- variant name
- seed
- task list
- train and held-out manifest paths
- teacher cache path
- metrics json/jsonl path
- video/contact-sheet path

## Implementation Tasks

### Task 1: Record Protocol And Metadata

**Files:**
- Create: `docs/superpowers/plans/2026-07-08-robotwin-stepwam-mini-ablation-protocol.md`
- Create or modify: `distillation_flowmap/ablation/robotwin_stepwam_mini_tasks.json`
- Create or modify: `distillation_flowmap/ablation/robotwin_stepwam_variants.json`
- Test: `distillation_flowmap/tests/test_robotwin_ablation_launcher.py`

**Interfaces:**
- Produces a four-task `core4` preset and six-task `core6` preset.
- Produces exactly four mini-ablation variants for the protocol-controlled run.

- [ ] Add the markdown protocol document.
- [ ] Add metadata for `core4` and `core6` task presets.
- [ ] Ensure the launcher can select `--task-preset core4` and `--task-preset core6`.
- [ ] Ensure dry-run output contains the chosen task preset and exact selected task list.
- [ ] Run launcher tests.
- [ ] Commit the document and metadata change.

### Task 2: Shared Stage 1 Run Planning

**Files:**
- Modify: `distillation_flowmap/ablation/launch_robotwin_stepwam_ablation.py`
- Test: `distillation_flowmap/tests/test_robotwin_ablation_launcher.py`

**Interfaces:**
- Adds `--shared-stage1-root`.
- Adds `--stage2-resume-from-shared-stage1`.
- Produces Stage 1 path: `<root>/shared_stage1/<task_preset>/seed_<seed>/stage1/checkpoints/step_<stage1_steps>`.
- Produces Stage 2 path: `<root>/<variant>/seed_<seed>/stage2/checkpoints/step_<stage2_steps>`.

- [ ] Add failing dry-run tests proving all four variants for one seed use the exact same Stage 1 checkpoint.
- [ ] Implement shared Stage 1 path construction.
- [ ] Make `--stage stage2` require an existing shared Stage 1 checkpoint unless `--dry-run` is set.
- [ ] Store `shared_stage1_ckpt` in `run_manifest.json`.
- [ ] Run launcher tests.
- [ ] Commit shared Stage 1 planning.

### Task 3: Deterministic Split And Eval Pair Manifests

**Files:**
- Create: `distillation_flowmap/ablation/robotwin_mini_protocol.py`
- Modify: `distillation_flowmap/ablation/launch_robotwin_stepwam_ablation.py`
- Test: create `distillation_flowmap/tests/test_robotwin_mini_protocol.py`

**Interfaces:**
- `build_index_split(task_names, samples_per_task, heldout_per_task, seed) -> dict`
- `build_eval_pairs(pairs, seed) -> list[dict]`
- `write_protocol_manifests(root, task_preset, seed, task_names, samples_per_task, heldout_per_task, pairs) -> dict`

- [ ] Write tests for deterministic split reproducibility.
- [ ] Write tests proving train and held-out ids are disjoint per task.
- [ ] Write tests proving eval pairs are identical across variants for the same preset and seed.
- [ ] Implement manifest helpers.
- [ ] Wire manifest paths into launcher dry-run and `run_manifest.json`.
- [ ] Run targeted tests.
- [ ] Commit protocol manifest helpers.

### Task 4: Held-Out Offline Eval Control

**Files:**
- Modify: `distillation_flowmap/rollout_eval_stage2.py`
- Test: create `distillation_flowmap/tests/test_robotwin_rollout_eval_protocol.py`

**Interfaces:**
- Adds `--eval-manifest`.
- Adds `--eval-pairs-json`.
- Adds `--split-name train|heldout`.
- Eval reads sample indices from the manifest and pair definitions from `eval_pairs.json`.

- [ ] Write tests for parsing eval manifest and eval pairs.
- [ ] Implement argument parsing and validation.
- [ ] Ensure eval errors if a held-out manifest is missing or empty.
- [ ] Ensure teacher cache metadata records manifest path and pair hash.
- [ ] Run targeted tests.
- [ ] Commit held-out eval controls.

### Task 5: Endpoint, Same-State Velocity, And Rollout Drift Metrics

**Files:**
- Modify: `distillation_flowmap/rollout_eval_stage2.py`
- Modify if needed: `distillation_flowmap/flowmap_step.py`
- Test: `distillation_flowmap/tests/test_robotwin_rollout_eval_protocol.py`

**Interfaces:**
- Adds denoised endpoint metric names under `rollout_eval/<pair>/s<K>_t<T>/`.
- Adds same-state velocity metric names under the same prefix.
- Adds optional student path return from `_student_euler_integrate(..., return_path=True)`.

- [ ] Write tests for denoised endpoint formula.
- [ ] Write tests that same-state velocity teacher target is marked variant-dependent.
- [ ] Implement endpoint metric computation.
- [ ] Implement optional rollout path return and K-step drift metrics.
- [ ] Implement a low-memory second pass for same-state teacher velocity if needed.
- [ ] Run targeted tests.
- [ ] Commit eval metric expansion.

### Task 6: OPD Loss And Gradient Diagnostics

**Files:**
- Modify: `distillation_flowmap/flowmap_step.py`
- Modify: `distillation_flowmap/flowmap_trainer.py`
- Test: `distillation_flowmap/tests/test_robotwin_stage2_endpoint.py`

**Interfaces:**
- Logs raw endpoint video/action losses.
- Logs raw velocity video/action losses.
- Logs weighted endpoint and velocity contributions.
- Logs effective beta and modality weights.
- Logs low-frequency branch gradient norms when enabled.

- [ ] Write tests for metric keys returned by OPD aux step.
- [ ] Implement raw and weighted loss metric keys without changing default loss math.
- [ ] Add `GRAD_DIAG_INTERVAL` config/env handling.
- [ ] Implement branch gradient norm logging after backward at the configured interval.
- [ ] Run targeted tests.
- [ ] Commit OPD diagnostics.

### Task 7: Mini-Ablation Summary Report

**Files:**
- Modify: `distillation_flowmap/ablation/summarize_robotwin_ablation.py`
- Test: create or extend `distillation_flowmap/tests/test_robotwin_ablation_summary.py`

**Interfaces:**
- Aggregates rows by `(task, variant, seed, split)`.
- Computes mean, std, and delta vs `w_o_opd`.
- Writes CSV, JSON, and Markdown.

- [ ] Write tests for mean/std aggregation.
- [ ] Write tests for delta vs baseline.
- [ ] Implement aggregation.
- [ ] Add train vs held-out gap fields.
- [ ] Ensure report marks notes as trend-only.
- [ ] Run targeted tests.
- [ ] Commit summary report.

### Task 8: Smoke And Mini Run

**Files:**
- Uses: launcher, eval script, video eval script, summary script.
- Produces: `<root>/reports/mini_ablation_summary.*`

**Interfaces:**
- Smoke command uses `1 task x 1 seed x tiny steps`.
- Mini command uses `core4 x 3 seeds x 4 variants`.

- [ ] Run smoke split generation.
- [ ] Run shared Stage 1 smoke.
- [ ] Run four Stage 2 smoke variants from the same shared Stage 1 checkpoint.
- [ ] Run held-out offline eval and video export for smoke.
- [ ] Check that train and held-out metrics are both written.
- [ ] Check that held-out eval samples are disjoint from train samples.
- [ ] Run core4 mini-ablation after smoke passes.
- [ ] Generate report and record trend-only notes.

## Verification Commands

Run these after implementation phases:

```bash
cd /root/nas/junjie/jj/Any_WAM
/root/nas/junjie/conda_envs/any_wam/bin/python -m py_compile \
  distillation_flowmap/ablation/launch_robotwin_stepwam_ablation.py \
  distillation_flowmap/ablation/robotwin_mini_protocol.py \
  distillation_flowmap/ablation/summarize_robotwin_ablation.py \
  distillation_flowmap/rollout_eval_stage2.py
```

```bash
cd /root/nas/junjie/jj/Any_WAM
/root/nas/junjie/conda_envs/any_wam/bin/python -m unittest discover distillation_flowmap/tests
```

## Risk Checks Before Running Mini-Ablation

- `run_manifest.json` for all four variants in one seed points to the same `shared_stage1_ckpt`.
- `heldout_eval_manifest.json` sample ids do not overlap with `train_manifest.json`.
- Teacher cache metadata matches the held-out manifest path and eval pair hash.
- `velocity_only_opd` does not include endpoint action OPD through `OPD_AUX_ACTION=1`.
- Report computes deltas against the same seed/preset/split `w_o_opd` row before aggregating.
- Videos/contact sheets use held-out samples only.
