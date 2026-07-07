# LingBot StepWAM Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a controlled StepWAM ablation only on LingBot-VA + RobotWin, with reproducible training, offline diagnostics, RobotWin SR evaluation, videos/contact sheets, and a summary table.

**Architecture:** Keep Cosmos out of component ablation. Train one shared arbitrary-shortcut Stage 1 checkpoint per seed, branch Stage 2 OPD variants from it, and train the local/adjacent and action-only variants as separate controls. Every run writes config, git hash, ckpt path, seed, task list, metrics json/csv, and video paths under a single ablation root.

**Tech Stack:** `distillation_flowmap/train.py`, `distillation_flowmap/config_robotwin_fullfinetune_stage1_warmup.py`, `distillation_flowmap/config_robotwin_fullfinetune_stage2_anyflow.py`, `distillation_flowmap/rollout_eval_stage2.py`, `distillation_flowmap/rollout_eval_video_stage2.py`, `evaluation/robotwin/launch_server_multigpus.sh`, `evaluation/robotwin/launch_client_multigpus.sh`, `tmux`, 8x H100.

## Global Constraints

- Teacher: LingBot-VA only.
- Benchmark: RobotWin representative subset.
- Main task subset: 12 tasks, covering pick-and-place, stacking/ranking, contact-rich/articulated, bimanual, long-horizon, and hard low-success.
- Split: Easy + Hard. If RobotWin has only `demo_clean` task_config available on this machine, use Easy/Hard as task groups under `demo_clean`; do not invent a non-existent config name.
- Episodes: 50 per task for final SR eval; smoke is 1 task x 5 episodes.
- Student step: fixed `K=4` for evaluation.
- Seeds: full / w/o OPD / local transition at least 3 seeds; endpoint-only / velocity-only / action-only at least 1 seed first, then backfill to 3 if time remains.
- OPD formula: endpoint teacher target is an independent teacher rollout from the same `x_t` to `r`; endpoint loss aligns `x0_S = x_r^S - sigma_r S(x_r^S, r)` with `x0_T = x_r^T - sigma_r T(x_r^T, r)`; velocity regularizer queries both models at `sg(x_r^S), r`.
- Do not modify Cosmos ablation code paths for this run; Cosmos remains main-result/cross-teacher only.
- Commit after each independently verified implementation phase.

---

## Variant Matrix

| id | name | seed count | Stage 1 | Stage 2 OPD | key env/config |
| --- | --- | ---: | --- | --- | --- |
| v1 | full_stepwam | 3 | arbitrary `t->r`, joint video/action | endpoint + same-state velocity + joint action/video | `DISTILL_MODE=flashwam USE_OPD_AUX=1 OPD_AUX_ACTION=1 VIDEO_TRANSITION_WEIGHT=1 OPD_ENDPOINT_AUX_WEIGHT=0.1 OPD_SAME_STATE_VELOCITY_WEIGHT=0.1` |
| v2 | w_o_opd | 3 | arbitrary `t->r`, joint video/action | disabled | `DISTILL_MODE=flashwam USE_OPD_AUX=0` |
| v3 | endpoint_only_opd | 1 then 3 | arbitrary `t->r`, joint video/action | endpoint only | `USE_OPD_AUX=1 OPD_AUX_ACTION=1 VIDEO_TRANSITION_WEIGHT=1 OPD_ENDPOINT_AUX_WEIGHT=0.1 OPD_SAME_STATE_VELOCITY_WEIGHT=0` |
| v4 | velocity_only_opd | 1 then 3 | arbitrary `t->r`, joint video/action | same-state velocity only | `USE_OPD_AUX=1 OPD_AUX_ACTION=0 VIDEO_TRANSITION_WEIGHT=0 OPD_ENDPOINT_AUX_WEIGHT=0 OPD_SAME_STATE_VELOCITY_WEIGHT=0.1` |
| v5 | local_adjacent_only | 3 | adjacent grid only | adjacent grid only | `FLOWMAP_PAIR_MODE=adjacent_grid OPD_PAIR_MODE=adjacent_grid FLOWMAP_ADJACENT_GRID=1000,750,500,250,0` |
| v6 | action_only | 1 then 3 | action only | action OPD only if smoke passes | `DISTILL_MODE=action DISTILL_VIDEO=0 DISTILL_ACTION=1 OPD_AUX_ACTION=1` |
| v7 | no_video_loss_optional | optional | joint forward, no video loss | same as full | only add if implementation is a small config flag; otherwise skip and record as not run |

For v4, set `OPD_AUX_ACTION=0` initially because current action endpoint OPD is endpoint-style; enabling action OPD would reintroduce endpoint loss into a velocity-only ablation. Backfill an action-velocity variant only if a true same-state action velocity target is added and smoke-tested.

## Task 1: Freeze Metadata, Task Subset, And Run Directory

**Files:**
- Create: `distillation_flowmap/ablation/robotwin_stepwam_tasks.json`
- Create: `distillation_flowmap/ablation/robotwin_stepwam_variants.json`
- Create: `distillation_flowmap/ablation/README.md`

**Interfaces:**
- Produces `robotwin_stepwam_tasks.json` consumed by training/eval launchers.
- Produces `robotwin_stepwam_variants.json` with env overrides consumed by launcher scripts.

- [ ] **Step 1: Create the task subset file**

Use this exact task list:

```json
{
  "easy": [
    "place_a2b_right",
    "put_object_cabinet",
    "stack_bowls_three",
    "lift_pot",
    "place_can_basket",
    "handover_block"
  ],
  "hard": [
    "open_microwave",
    "open_laptop",
    "pick_dual_bottles",
    "blocks_ranking_size",
    "place_burger_fries",
    "rotate_qrcode"
  ]
}
```

- [ ] **Step 2: Create the variant matrix file**

Encode the seven rows from the Variant Matrix as JSON. Include fields: `id`, `name`, `seeds`, `stage1_env`, `stage2_env`, `requires_code_flag`, `priority`.

- [ ] **Step 3: Verify metadata is parseable**

Run:

```bash
cd /root/nas/junjie/jj/Any_WAM
/root/nas/junjie/conda_envs/any_wam/bin/python -m json.tool distillation_flowmap/ablation/robotwin_stepwam_tasks.json >/tmp/tasks.json
/root/nas/junjie/conda_envs/any_wam/bin/python -m json.tool distillation_flowmap/ablation/robotwin_stepwam_variants.json >/tmp/variants.json
```

Expected: exit code `0`.

- [ ] **Step 4: Commit**

```bash
git add distillation_flowmap/ablation/robotwin_stepwam_tasks.json distillation_flowmap/ablation/robotwin_stepwam_variants.json distillation_flowmap/ablation/README.md
git commit -m "chore: add robotwin stepwam ablation metadata"
```

## Task 2: Add Missing Ablation Config Knobs

**Files:**
- Modify: `distillation_flowmap/config_robotwin_fullfinetune_stage1_warmup.py`
- Modify: `distillation_flowmap/config_robotwin_fullfinetune_stage2_anyflow.py`
- Modify: `distillation_flowmap/flowmap_step.py`
- Test: `distillation_flowmap/tests/test_robotwin_stage2_endpoint.py`
- Test: create `distillation_flowmap/tests/test_robotwin_ablation_pair_modes.py`

**Interfaces:**
- Produces `cfg.flowmap_pair_mode`, `cfg.opd_pair_mode`, `cfg.flowmap_adjacent_grid`.
- Produces adjacent-grid timestep sampler behavior used by v5.

- [ ] **Step 1: Write failing tests for adjacent-grid mode**

Test expectations:
- `FLOWMAP_PAIR_MODE=adjacent_grid` parses grid `[1000, 750, 500, 250, 0]`.
- sampled `(t, r)` pairs are only adjacent grid edges.
- `OPD_PAIR_MODE=adjacent_grid` applies the same restriction inside OPD aux.

Run:

```bash
cd /root/nas/junjie/jj/Any_WAM
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest -q distillation_flowmap/tests/test_robotwin_ablation_pair_modes.py
```

Expected before implementation: FAIL because flags do not exist.

- [ ] **Step 2: Implement the flags**

Add config defaults:

```python
cfg.flowmap_pair_mode = os.environ.get("FLOWMAP_PAIR_MODE", "arbitrary").lower()
cfg.opd_pair_mode = os.environ.get("OPD_PAIR_MODE", cfg.flowmap_pair_mode).lower()
cfg.flowmap_adjacent_grid = [
    int(v) for v in os.environ.get("FLOWMAP_ADJACENT_GRID", "1000,750,500,250,0").split(",")
]
```

Implement sampler behavior in `flowmap_step.py` without changing default arbitrary mode. Adjacent-grid mode samples one edge from `(1000,750)`, `(750,500)`, `(500,250)`, `(250,0)` per sample.

- [ ] **Step 3: Run targeted tests**

```bash
cd /root/nas/junjie/jj/Any_WAM
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest -q \
  distillation_flowmap/tests/test_robotwin_ablation_pair_modes.py \
  distillation_flowmap/tests/test_robotwin_stage2_endpoint.py \
  distillation_flowmap/tests/test_robotwin_light_eval_config.py
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add distillation_flowmap/config_robotwin_fullfinetune_stage1_warmup.py distillation_flowmap/config_robotwin_fullfinetune_stage2_anyflow.py distillation_flowmap/flowmap_step.py distillation_flowmap/tests/test_robotwin_ablation_pair_modes.py distillation_flowmap/tests/test_robotwin_stage2_endpoint.py
git commit -m "feat: add robotwin ablation pair modes"
```

## Task 3: Add Reproducible Launchers And Run Registry

**Files:**
- Create: `distillation_flowmap/ablation/launch_robotwin_stepwam_ablation.py`
- Create: `distillation_flowmap/ablation/summarize_robotwin_ablation.py`
- Create: `distillation_flowmap/ablation/run_smoke.sh`
- Create: `distillation_flowmap/ablation/run_parallel_train.sh`

**Interfaces:**
- Consumes `robotwin_stepwam_tasks.json` and `robotwin_stepwam_variants.json`.
- Produces one directory per run: `distillation_flowmap/output_robotwin_stepwam_ablation/<variant>/seed_<seed>/`.
- Produces `run_manifest.json` with `git_hash`, `variant`, `seed`, `teacher_model_path`, `dataset_path`, `stage1_ckpt`, `stage2_ckpt`, `task_list`, `config_env`.

- [ ] **Step 1: Write failing launcher tests**

Create tests that call the launcher in `--dry-run` mode and assert:
- full variant command includes `OPD_SAME_STATE_VELOCITY_WEIGHT=0.1`.
- w/o OPD command includes `USE_OPD_AUX=0`.
- local variant command includes `FLOWMAP_PAIR_MODE=adjacent_grid`.
- output directory includes variant and seed.

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest -q distillation_flowmap/tests/test_robotwin_ablation_launcher.py
```

Expected before implementation: FAIL because launcher does not exist.

- [ ] **Step 2: Implement dry-run and real-run commands**

Use this base command for Stage 1:

```bash
CONFIG_FILE=distillation_flowmap.config_robotwin_fullfinetune_stage1_warmup \
OUTPUT_DIR=<run_dir>/stage1 \
MAX_TRAIN_STEPS=<stage1_steps> \
WANDB_MODE=offline \
torchrun --nproc_per_node=1 --master_port=<port> distillation_flowmap/train.py \
  --teacher-model-path <teacher_path> \
  --dataset-path <dataset_path> \
  --gradient-accumulation-steps 1
```

Use this base command for Stage 2:

```bash
CONFIG_FILE=distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow \
RESUME_FROM_PATH=<run_dir>/stage1/checkpoints/step_<stage1_steps> \
OUTPUT_DIR=<run_dir>/stage2 \
MAX_TRAIN_STEPS=<stage2_steps> \
WANDB_MODE=offline \
torchrun --nproc_per_node=1 --master_port=<port> distillation_flowmap/train.py \
  --teacher-model-path <teacher_path> \
  --dataset-path <dataset_path> \
  --gradient-accumulation-steps 1
```

- [ ] **Step 3: Verify launcher dry-run**

```bash
cd /root/nas/junjie/jj/Any_WAM
/root/nas/junjie/conda_envs/any_wam/bin/python distillation_flowmap/ablation/launch_robotwin_stepwam_ablation.py \
  --variant full_stepwam --seed 0 --dry-run
```

Expected: printed commands only, no checkpoint directories created.

- [ ] **Step 4: Commit**

```bash
git add distillation_flowmap/ablation/launch_robotwin_stepwam_ablation.py distillation_flowmap/ablation/summarize_robotwin_ablation.py distillation_flowmap/ablation/run_smoke.sh distillation_flowmap/ablation/run_parallel_train.sh distillation_flowmap/tests/test_robotwin_ablation_launcher.py
git commit -m "feat: add robotwin stepwam ablation launcher"
```

## Task 4: Smoke Test End To End

**Files:**
- Uses: `distillation_flowmap/ablation/run_smoke.sh`
- Produces: `distillation_flowmap/output_robotwin_stepwam_ablation/smoke/`

**Interfaces:**
- Validates train, offline eval, video export, RobotWin env SR eval, and summary parsing before full runs.

- [ ] **Step 1: Run 1 task x 5 episodes smoke**

Use one easy task:

```bash
cd /root/nas/junjie/jj/Any_WAM
TEACHER_MODEL_PATH=/root/nas/junjie/jj/Any_WAM/checkpoints/base \
DATASET_PATH=/root/nas/junjie/jj/Any_WAM/training_data/lerobot_robotwin_eef_aug_500 \
SMOKE_TASK=place_a2b_right \
SMOKE_EPISODES=5 \
SMOKE_STAGE1_STEPS=20 \
SMOKE_STAGE2_STEPS=20 \
bash distillation_flowmap/ablation/run_smoke.sh
```

Expected:
- one Stage 1 checkpoint under `smoke/full_stepwam/seed_0/stage1/checkpoints/`.
- one Stage 2 checkpoint under `smoke/full_stepwam/seed_0/stage2/checkpoints/`.
- one offline metrics json.
- one comparison video/contact sheet.
- one RobotWin metrics file with 5 attempted episodes.

- [ ] **Step 2: Inspect smoke metrics**

Required gates:
- no NaN in training log.
- offline endpoint MSE and same-state velocity metrics are finite.
- generated video has non-zero duration.
- RobotWin eval writes per-episode success records.

- [ ] **Step 3: Commit smoke script fixes only if needed**

If smoke exposes launcher or parser bugs, fix and commit:

```bash
git add distillation_flowmap/ablation
git commit -m "fix: stabilize robotwin ablation smoke"
```

## Task 5: Full Training Schedule

**Files:**
- Uses: `distillation_flowmap/ablation/run_parallel_train.sh`
- Produces: `distillation_flowmap/output_robotwin_stepwam_ablation/<variant>/seed_<seed>/`

**Interfaces:**
- Produces checkpoints consumed by offline metrics, video export, and RobotWin SR eval.

- [ ] **Step 1: Run priority-1 variants**

GPU assignment:

```text
GPU0 full_stepwam seed_0
GPU1 w_o_opd seed_0
GPU2 endpoint_only_opd seed_0
GPU3 velocity_only_opd seed_0
GPU4 local_adjacent_only seed_0
GPU5 action_only seed_0
GPU6 offline eval/video queue
GPU7 diagnostics/retry/cache
```

Command:

```bash
cd /root/nas/junjie/jj/Any_WAM
tmux new -s stepwam_ablation
bash distillation_flowmap/ablation/run_parallel_train.sh --stage1-steps 5000 --stage2-steps 5000 --priority first_pass
```

Expected: six first-pass variants finish training or fail with explicit log errors.

- [ ] **Step 2: Backfill required 3-seed variants**

Run:

```bash
bash distillation_flowmap/ablation/run_parallel_train.sh --stage1-steps 5000 --stage2-steps 5000 --variants full_stepwam,w_o_opd,local_adjacent_only --seeds 1,2
```

Expected: full / w/o OPD / local transition have seeds 0, 1, 2.

- [ ] **Step 3: Backfill optional 3-seed variants if first-pass result is meaningful**

Run endpoint-only / velocity-only / action-only seeds 1,2 only after first-pass metrics show finite losses and no obvious launcher/eval artifact.

## Task 6: Offline Metrics And Video Logging

**Files:**
- Uses: `distillation_flowmap/rollout_eval_stage2.py`
- Uses: `distillation_flowmap/rollout_eval_video_stage2.py`
- Produces: `metrics/offline_rollout.json`, `videos/*.mp4`, `videos/contact_sheet.png`

**Interfaces:**
- Produces endpoint error, same-state velocity error, rollout drift, video/action MSE/L1, and qualitative videos.

- [ ] **Step 1: Run offline metrics per checkpoint**

```bash
torchrun --nproc_per_node=1 --master_port=<port> distillation_flowmap/rollout_eval_stage2.py \
  --config distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow \
  --teacher-model-path <teacher_path> \
  --dataset-path <dataset_path> \
  --output-dir <run_dir>/eval \
  --resume-from-path <run_dir>/stage2/checkpoints/step_5000 \
  --result-json <run_dir>/metrics/offline_rollout.json \
  --num-batches 8 \
  --student-steps 4 \
  --teacher-steps 4 \
  --pairs 1000,0 1000,500 750,250
```

- [ ] **Step 2: Generate teacher/student videos**

```bash
torchrun --nproc_per_node=1 --master_port=<port> distillation_flowmap/rollout_eval_video_stage2.py \
  --config distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow \
  --teacher-model-path <teacher_path> \
  --dataset-path <dataset_path> \
  --output-dir <run_dir>/videos \
  --resume-from-path <run_dir>/stage2/checkpoints/step_5000 \
  --result-json <run_dir>/metrics/video_mse.json \
  --num-batches 2 \
  --student-steps 4 \
  --teacher-steps 4 \
  --pairs 1000,0
```

Expected: student 4-step and teacher 4-step comparison videos are non-zero duration and frames change over time.

## Task 7: RobotWin Success-Rate Evaluation

**Files:**
- Uses: `evaluation/robotwin/launch_server_multigpus.sh`
- Uses: `evaluation/robotwin/launch_client_multigpus.sh`
- Produces: `eval_result` and `results/<variant>/seed_<seed>/metrics/`

**Interfaces:**
- Produces SR, teacher retention, per-task metrics, latency/chunk, Hz, NFE, and speedup.

- [ ] **Step 1: Start policy servers**

Run in tmux pane:

```bash
cd /root/nas/junjie/jj/Any_WAM
bash evaluation/robotwin/launch_server_multigpus.sh
```

Expected: ports `29556..29563` are listening and each server log reports model loaded.

- [ ] **Step 2: Run 50 episodes per task**

For each variant/seed, run both task groups:

```bash
cd /root/nas/junjie/jj/Any_WAM
bash evaluation/robotwin/launch_client_multigpus.sh <save_root>/<variant>/seed_<seed>/easy 0 0 50
bash evaluation/robotwin/launch_client_multigpus.sh <save_root>/<variant>/seed_<seed>/hard 3 0 50
```

Because existing `launch_client_multigpus.sh` task groups do not exactly match the 12-task subset, first patch it or add `launch_client_taskfile.sh` to consume `robotwin_stepwam_tasks.json`; do not evaluate extra tasks silently.

- [ ] **Step 3: Parse SR and latency**

Aggregate per-task success, mean SR, std across seeds, latency/chunk, Hz, NFE, and speedup into:

```text
distillation_flowmap/output_robotwin_stepwam_ablation/summary/robotwin_sr.csv
distillation_flowmap/output_robotwin_stepwam_ablation/summary/robotwin_sr.json
```

## Task 8: Summary Table And Figure Assets

**Files:**
- Uses: `distillation_flowmap/ablation/summarize_robotwin_ablation.py`
- Produces: `summary/ablation_table.md`
- Produces: `summary/ablation_table.csv`
- Produces: `summary/figure5_contact_sheets/`
- Produces: `summary/figure6_rollout_videos/`

**Interfaces:**
- Produces paper-ready tables and media for Figure 5/6.

- [ ] **Step 1: Generate summary**

```bash
cd /root/nas/junjie/jj/Any_WAM
/root/nas/junjie/conda_envs/any_wam/bin/python distillation_flowmap/ablation/summarize_robotwin_ablation.py \
  --root distillation_flowmap/output_robotwin_stepwam_ablation \
  --tasks distillation_flowmap/ablation/robotwin_stepwam_tasks.json \
  --out distillation_flowmap/output_robotwin_stepwam_ablation/summary
```

- [ ] **Step 2: Check table columns**

Required columns:

```text
variant, seed_count, task_count, episodes_per_task, SR_easy, SR_hard, SR_all,
teacher_retention, endpoint_error, same_state_velocity_error, rollout_drift,
FVD_or_proxy, LPIPS_or_proxy, temporal_error, latency_chunk_ms, Hz, NFE, speedup
```

- [ ] **Step 3: Make the decision gate**

Declare the method useful only if:
- `full_stepwam` beats `w_o_opd` on SR_all by at least 3 absolute points or wins most hard tasks with comparable SR_all.
- `full_stepwam` does not lose more than 5 absolute points of teacher retention.
- endpoint/same-state metrics improve in the expected direction.
- latency speedup remains consistent with fixed `K=4`.

If this gate fails, keep full results but write the negative finding clearly and do not expand Cosmos ablation.

## Execution Order

1. Task 1 metadata.
2. Task 2 missing local/adjacent config.
3. Task 3 launchers.
4. Task 4 smoke.
5. Task 5 first-pass training.
6. Task 6 offline metrics/videos for first-pass.
7. Task 7 first-pass RobotWin SR.
8. Backfill seeds only after first-pass is sane.
9. Task 8 summary and figure assets.

## Immediate First Command After Approval

```bash
cd /root/nas/junjie/jj/Any_WAM
git status --short
mkdir -p distillation_flowmap/ablation
```

Then start Task 1 with tests/JSON validation and commit before touching sampler code.
