# LIBERO APM Small-Sample Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one four-GPU command that trains the four RobotWin-style APM ablation arms on 40 LIBERO task-0 trajectories, runs held-out diagnostics, and evaluates only that trained task at matched 1/1, 2/2, and 4/4 video-action budgets.

**Architecture:** A LIBERO-specific config and deterministic protocol module define the frozen experiment. A Python planner produces reproducible per-arm manifests and four-process training commands. A Bash orchestrator runs arms serially, invokes held-out offline evaluation, schedules single-task closed-loop jobs across four GPUs, and calls a strict Python merger for the final comparison table.

**Tech Stack:** Python 3.10, PyTorch/FSDP1, PEFT LoRA, pytest, Bash, LingBotVA WebSocket inference, LIBERO benchmark client.

## Global Constraints

- Use the shared Stage-1 checkpoint at `distillation_flowmap/output_libero_fullft_stage1_warmup/checkpoints/step_2000`.
- Use dataset episodes 0–39 for training and 40–49 for held-out diagnostics.
- Use LIBERO benchmark `libero_10`, task index 0 only.
- Keep seed 42, LoRA rank 128, alpha 64, dropout 0, learning rate `5e-6`, and `ATTN_MODE=torch`.
- Hard-disable action OPD, joint action OPD rollout, and DanceOPD action velocity in every arm.
- Evaluate Stage-1 plus the four final arms with matched budgets 1/1, 2/2, and 4/4 for 20 episodes each.
- Never overwrite an existing run directory or silently accept incomplete evaluation output.

---

### Task 1: Deterministic LIBERO protocol and ablation config

**Files:**
- Create: `distillation_flowmap/ablation/libero_small_sample_protocol.py`
- Create: `distillation_flowmap/config_libero_apm_lora_ablation.py`
- Create: `distillation_flowmap/tests/test_libero_small_sample_protocol.py`
- Create: `distillation_flowmap/tests/test_libero_apm_ablation_config.py`

**Interfaces:**
- Produces: `build_libero_task0_protocol() -> dict`
- Produces: `write_libero_task0_manifests(root: Path) -> dict[str, Path]`
- Produces: config object `cfg` compatible with `distillation_flowmap/train.py`

- [ ] **Step 1: Write failing protocol tests**

```python
def test_task0_protocol_is_frozen_to_40_train_and_10_heldout():
    protocol = build_libero_task0_protocol()
    assert protocol["benchmark"] == "libero_10"
    assert protocol["task_index"] == 0
    assert protocol["task_language"] == (
        "put both the alphabet soup and the tomato sauce in the basket"
    )
    assert protocol["train_indices"] == list(range(40))
    assert protocol["heldout_indices"] == list(range(40, 50))


def test_manifests_target_the_single_libero_repository(tmp_path):
    paths = write_libero_task0_manifests(tmp_path)
    train = json.loads(paths["train"].read_text())
    heldout = json.loads(paths["heldout"].read_text())
    assert train["tasks"] == [{"task": "libero", "indices": list(range(40))}]
    assert heldout["tasks"] == [{"task": "libero", "indices": list(range(40, 50))}]
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest -q \
  distillation_flowmap/tests/test_libero_small_sample_protocol.py
```

Expected: collection fails because `libero_small_sample_protocol` does not exist.

- [ ] **Step 3: Implement the frozen protocol**

Implement constants for benchmark, task index, task language, train indices,
and held-out indices. Write manifests atomically using a temporary file and
`os.replace`; each manifest uses repository key `"libero"` because the
dataset loader applies indices at the single-repository metadata level.

- [ ] **Step 4: Write failing config tests**

```python
def test_ablation_config_enables_lora_and_manifest(monkeypatch):
    monkeypatch.setenv("DATASET_SAMPLE_MANIFEST", "/tmp/train.json")
    monkeypatch.setenv("OPD_DANCEOPD_ENDPOINT_WEIGHT", "0")
    monkeypatch.setenv("OPD_DANCEOPD_VELOCITY_WEIGHT", "1")
    module = fresh_import("distillation_flowmap.config_libero_apm_lora_ablation")
    cfg = module.cfg
    assert cfg.use_lora
    assert cfg.lora_rank == 128
    assert cfg.lora_alpha == 64
    assert cfg.learning_rate == 5e-6
    assert cfg.dataset_sample_manifest == "/tmp/train.json"
    assert not cfg.opd_aux_action
    assert not cfg.opd_joint_action_rollout
    assert cfg.opd_danceopd_action_velocity_weight == 0.0
```

- [ ] **Step 5: Run the config test and verify RED**

Run:

```bash
python -m pytest -q \
  distillation_flowmap/tests/test_libero_apm_ablation_config.py
```

Expected: import fails because the config module does not exist.

- [ ] **Step 6: Implement the config wrapper**

Deep-copy `config_libero_fullfinetune_stage2_video_only_opd.cfg`, then assign:

```python
cfg.use_lora = True
cfg.lora_rank = int(os.environ.get("LORA_RANK", 128))
cfg.lora_alpha = int(os.environ.get("LORA_ALPHA", 64))
cfg.lora_dropout = float(os.environ.get("LORA_DROPOUT", 0.0))
cfg.learning_rate = float(os.environ.get("LEARNING_RATE", 5e-6))
cfg.dataset_sample_manifest = os.environ.get("DATASET_SAMPLE_MANIFEST")
cfg.dataset_max_episodes_per_task = None
cfg.dataset_max_samples_per_task = None
```

Reject missing manifests, nonpositive LoRA dimensions, action OPD overrides,
and any attention backend other than the inherited validated backend.

- [ ] **Step 7: Verify GREEN**

Run:

```bash
python -m pytest -q \
  distillation_flowmap/tests/test_libero_small_sample_protocol.py \
  distillation_flowmap/tests/test_libero_apm_ablation_config.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add distillation_flowmap/ablation/libero_small_sample_protocol.py \
  distillation_flowmap/config_libero_apm_lora_ablation.py \
  distillation_flowmap/tests/test_libero_small_sample_protocol.py \
  distillation_flowmap/tests/test_libero_apm_ablation_config.py
git commit -m "feat: define libero small-sample APM protocol"
```

---

### Task 2: Four-arm metadata and reproducible run planner

**Files:**
- Create: `distillation_flowmap/ablation/libero_apm_lora_variants.json`
- Create: `distillation_flowmap/ablation/launch_libero_apm_ablation.py`
- Create: `distillation_flowmap/tests/test_libero_apm_ablation_planner.py`

**Interfaces:**
- Consumes: `write_libero_task0_manifests`
- Produces: `build_run_plan(args) -> dict`
- Produces: per-arm `run_manifest.json` and a four-process Stage-2 command

- [ ] **Step 1: Write failing arm-mapping tests**

```python
@pytest.mark.parametrize(
    ("arm", "endpoint", "velocity"),
    [
        ("stage1_only", "0.0", "0.0"),
        ("anchor_only", "1.0", "0.0"),
        ("field_only", "0.0", "1.0"),
        ("apm", "1.0", "1.0"),
    ],
)
def test_arm_changes_only_video_opd_terms(arm, endpoint, velocity, tmp_path):
    plan = build_plan(tmp_path, arm)
    env = plan["stage2"]["env"]
    assert env["OPD_DANCEOPD_ENDPOINT_WEIGHT"] == endpoint
    assert env["OPD_DANCEOPD_VELOCITY_WEIGHT"] == velocity
    assert env["OPD_AUX_ACTION"] == "0"
    assert env["OPD_JOINT_ACTION_ROLLOUT"] == "0"
    assert env["OPD_DANCEOPD_ACTION_VELOCITY_WEIGHT"] == "0.0"
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
python -m pytest -q \
  distillation_flowmap/tests/test_libero_apm_ablation_planner.py
```

Expected: missing planner module.

- [ ] **Step 3: Add frozen variant metadata**

The JSON contains the four IDs above, common config
`distillation_flowmap.config_libero_apm_lora_ablation`, default 500 steps,
save interval 100, seed 42, LoRA settings, `OPD_AUX_INTERVAL=4`, rollout
choices `2,4`, and rollout pairs `8,1;8,2;8,4`.

- [ ] **Step 4: Implement planner and manifest**

`build_run_plan` must:

```python
run_dir = output_root / arm / "seed_42"
stage2_dir = run_dir / "stage2"
checkpoint = stage2_dir / "checkpoints" / f"step_{steps}"
argv = [
    str(torchrun),
    "--nproc_per_node=4",
    f"--master_port={master_port}",
    "distillation_flowmap/train.py",
    "--teacher-model-path", str(teacher),
    "--dataset-path", str(dataset),
    "--gradient-accumulation-steps", "1",
]
```

It writes exact environment, command, git hash, protocol, and checkpoint
paths. Real execution refuses an existing `run_dir`; dry-run performs no
writes.

- [ ] **Step 5: Add planner validation tests**

Test unknown arms, duplicate output directories, missing Stage-1 checkpoint,
nonpositive steps, manifest path propagation, four-process torchrun, seed 42,
and shared Stage-1 initialization.

- [ ] **Step 6: Verify GREEN**

Run:

```bash
python -m pytest -q \
  distillation_flowmap/tests/test_libero_apm_ablation_planner.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add distillation_flowmap/ablation/libero_apm_lora_variants.json \
  distillation_flowmap/ablation/launch_libero_apm_ablation.py \
  distillation_flowmap/tests/test_libero_apm_ablation_planner.py
git commit -m "feat: plan four-arm libero APM runs"
```

---

### Task 3: Four-GPU serial training and held-out offline diagnostics

**Files:**
- Create: `distillation_flowmap/ablation/run_libero_apm_lora_4gpu_serial.sh`
- Create: `distillation_flowmap/tests/test_run_libero_apm_lora_4gpu_serial.py`

**Interfaces:**
- Consumes: planner from Task 2
- Produces: final checkpoints and `offline_eval/heldout.json` for every arm
- CLI: `--phase train|offline-eval|closed-loop|all`

- [ ] **Step 1: Write failing shell-contract tests**

Use `subprocess.run(..., env={"CHECK_ONLY": "1"})` and assert:

```python
assert output.count("--nproc_per_node=4") == 4
assert "stage1_only" in output
assert "anchor_only" in output
assert "field_only" in output
assert "apm" in output
assert "step_2000" in output
assert "train_manifest.json" in output
assert "heldout_manifest.json" in output
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
python -m pytest -q \
  distillation_flowmap/tests/test_run_libero_apm_lora_4gpu_serial.py
```

Expected: launcher missing.

- [ ] **Step 3: Implement argument parsing and preflight**

Defaults:

```bash
PHASE=all
STEPS=500
SAVE_INTERVAL=100
EPISODES=20
GPU_IDS=0,1,2,3
ARMS=stage1_only,anchor_only,field_only,apm
```

Validate four unique visible GPUs, executable Python/torchrun, Stage-1,
teacher, dataset, ports, and output nonexistence. `--dry-run` prints exact
commands and creates nothing.

- [ ] **Step 4: Implement synchronous training**

For each selected arm, call the planner with the next master port, then run
its command synchronously and log to:

```text
<output-root>/<arm>/seed_42/logs/stage2.log
```

Stop immediately on failure.

- [ ] **Step 5: Implement held-out offline evaluation**

For each completed final checkpoint call:

```bash
python distillation_flowmap/rollout_eval_video_stage2.py \
  --config distillation_flowmap.config_libero_apm_lora_ablation \
  --teacher-model-path "$TEACHER_MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --empty-emb-path "$EMPTY_EMB_PATH" \
  --output-dir "$ARM_STAGE2_DIR" \
  --resume-from-path "$ARM_CHECKPOINT" \
  --result-json "$ARM_ROOT/offline_eval/heldout.json" \
  --eval-manifest "$HELDOUT_MANIFEST" \
  --split-name heldout \
  --num-batches 10 \
  --student-steps 1 2 4 \
  --teacher-steps 1 2 4
```

Skip only when the result JSON exists and validates as complete.

- [ ] **Step 6: Verify GREEN and syntax**

Run:

```bash
bash -n distillation_flowmap/ablation/run_libero_apm_lora_4gpu_serial.sh
python -m pytest -q \
  distillation_flowmap/tests/test_run_libero_apm_lora_4gpu_serial.py
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add distillation_flowmap/ablation/run_libero_apm_lora_4gpu_serial.sh \
  distillation_flowmap/tests/test_run_libero_apm_lora_4gpu_serial.py
git commit -m "feat: run libero APM training and heldout eval"
```

---

### Task 4: Four-GPU single-task closed-loop scheduler and strict merger

**Files:**
- Create: `evaluation/libero/run_lingbotva_task0_124_eval_4gpu.sh`
- Create: `evaluation/libero/merge_lingbotva_task0_ablation.py`
- Create: `evaluation/libero/tests/test_run_lingbotva_task0_124_eval_4gpu.py`
- Create: `evaluation/libero/tests/test_merge_lingbotva_task0_ablation.py`

**Interfaces:**
- Scheduler consumes repeated `--model NAME=CHECKPOINT` arguments
- Scheduler produces one result JSON per model-budget job
- Merger consumes the evaluation root and produces `summary.json`

- [ ] **Step 1: Write failing scheduler tests**

In `CHECK_ONLY=1`, assert exactly 15 jobs for Stage-1 plus four arms, task
range `0:1`, benchmark `libero_10`, and matched step pairs:

```python
assert output.count("TASK_START=0") == 15
assert output.count("TASK_END=1") == 15
assert set(re.findall(r"video_steps=(\\d+) action_steps=(\\d+)", output)) == {
    ("1", "1"), ("2", "2"), ("4", "4")
}
assert set(re.findall(r"gpu=(\\d+)", output)) <= {"0", "1", "2", "3"}
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
python -m pytest -q \
  evaluation/libero/tests/test_run_lingbotva_task0_124_eval_4gpu.py
```

Expected: scheduler missing.

- [ ] **Step 3: Implement bounded four-lane scheduling**

Each job invokes `evaluation/libero/run_eval_new.sh` with:

```bash
LIBERO_BENCHMARK=libero_10
EVAL_MODE=success
TASK_START=0
TASK_END=1
TEST_NUM=20
NUM_STEPS="$steps"
ACTION_NUM_STEPS="$steps"
```

At most four jobs run concurrently. Assign a unique GPU, WebSocket port, and
torchrun master port to every active lane. Install EXIT/INT/TERM cleanup that
terminates all child process groups. A completed, validated task result may be
skipped on resume.

- [ ] **Step 4: Write failing merger tests**

Create fixture results for five models and three budgets, then assert:

```python
assert summary["task_index"] == 0
assert summary["episodes_per_job"] == 20
assert len(summary["rows"]) == 15
assert summary["rows"][0].keys() >= {
    "model", "steps", "successes", "episodes", "success_rate",
    "delta_vs_stage1",
}
```

Also test that missing, duplicate, or non-20-episode results fail.

- [ ] **Step 5: Implement strict merger**

Discover exactly one `libero_10_0.json` per model/budget, validate
`0 <= succ_num <= total_num == 20`, compute rates, attach Stage-1 deltas for
the same budget, and atomically write JSON plus a compact CSV.

- [ ] **Step 6: Verify GREEN**

Run:

```bash
bash -n evaluation/libero/run_lingbotva_task0_124_eval_4gpu.sh
python -m pytest -q \
  evaluation/libero/tests/test_run_lingbotva_task0_124_eval_4gpu.py \
  evaluation/libero/tests/test_merge_lingbotva_task0_ablation.py
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add evaluation/libero/run_lingbotva_task0_124_eval_4gpu.sh \
  evaluation/libero/merge_lingbotva_task0_ablation.py \
  evaluation/libero/tests/test_run_lingbotva_task0_124_eval_4gpu.py \
  evaluation/libero/tests/test_merge_lingbotva_task0_ablation.py
git commit -m "feat: evaluate libero task0 APM ablation"
```

---

### Task 5: End-to-end orchestration, smoke, and handoff

**Files:**
- Modify: `distillation_flowmap/ablation/run_libero_apm_lora_4gpu_serial.sh`
- Modify: `distillation_flowmap/tests/test_run_libero_apm_lora_4gpu_serial.py`
- Modify: `distillation_flowmap/ablation/README.md`

**Interfaces:**
- Produces the final user command and phase-resumable pipeline

- [ ] **Step 1: Add failing end-to-end dry-run test**

Assert `--phase all --dry-run` prints, in order:

```text
PROTOCOL
TRAIN stage1_only
TRAIN anchor_only
TRAIN field_only
TRAIN apm
OFFLINE_EVAL stage1_only
OFFLINE_EVAL anchor_only
OFFLINE_EVAL field_only
OFFLINE_EVAL apm
CLOSED_LOOP models=stage1,stage1_only,anchor_only,field_only,apm
```

Assert no output directories are created.

- [ ] **Step 2: Run and verify RED**

Run the single test and confirm it fails because closed-loop orchestration is
not yet connected.

- [ ] **Step 3: Wire the closed-loop phase**

Build repeated model arguments for Stage-1 and completed final arm
checkpoints, then call the Task 4 scheduler. Preserve `--episodes`,
`--gpu-ids`, output root, and port-base overrides.

- [ ] **Step 4: Document the final command**

Document:

```bash
cd /kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/libero-lingbotva-video-opd
PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
bash distillation_flowmap/ablation/run_libero_apm_lora_4gpu_serial.sh \
  --phase all \
  --gpu-ids 0,1,2,3 \
  --steps 500 \
  --episodes 20 \
  --output-root /kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_apm_lora_ablation_20260724
```

- [ ] **Step 5: Run full static verification**

```bash
python -m pytest -q \
  distillation_flowmap/tests/test_libero_small_sample_protocol.py \
  distillation_flowmap/tests/test_libero_apm_ablation_config.py \
  distillation_flowmap/tests/test_libero_apm_ablation_planner.py \
  distillation_flowmap/tests/test_run_libero_apm_lora_4gpu_serial.py \
  evaluation/libero/tests/test_run_lingbotva_task0_124_eval_4gpu.py \
  evaluation/libero/tests/test_merge_lingbotva_task0_ablation.py
bash -n distillation_flowmap/ablation/run_libero_apm_lora_4gpu_serial.sh
bash -n evaluation/libero/run_lingbotva_task0_124_eval_4gpu.sh
git diff --check
```

Expected: zero failures.

- [ ] **Step 6: Run one-GPU one-step smoke**

Use arm `anchor_only`, a fresh smoke output, `MAX_TRAIN_STEPS=1`,
`SAVE_INTERVAL=1`, the 40-sample training manifest, and the verified
`ATTN_MODE=torch` path. Success requires the log to show LoRA enabled, 40
selected samples, main backward, video OPD backward, optimizer step, and a
saved step-1 checkpoint.

- [ ] **Step 7: Run four-GPU check-only preflight**

```bash
CHECK_ONLY=1 \
bash distillation_flowmap/ablation/run_libero_apm_lora_4gpu_serial.sh \
  --phase all --gpu-ids 0,1,2,3 --dry-run
```

Expected: four training commands, four offline jobs, and fifteen single-task
closed-loop jobs with no writes.

- [ ] **Step 8: Commit**

```bash
git add distillation_flowmap/ablation/run_libero_apm_lora_4gpu_serial.sh \
  distillation_flowmap/tests/test_run_libero_apm_lora_4gpu_serial.py \
  distillation_flowmap/ablation/README.md
git commit -m "docs: finalize libero APM ablation workflow"
```
