# Cosmos Official-Teacher Four-GPU Co-Located Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the formal Cosmos official-teacher LIBERO matrix with four teacher workers on four GPUs by safely co-locating each lightweight evaluator with its raw worker.

**Architecture:** Extend the shared formal shard launcher with an explicit `paired|colocated` layout contract. Keep `paired` as the default for all student paths, permit `colocated` only for `official_teacher`, and make the four-GPU teacher wrapper select four co-located shards.

**Tech Stack:** Bash launchers, Python/pytest launcher-contract tests, LIBERO formal JSON/CSV merger.

## Global Constraints

- Preserve official-teacher matched video/action budgets `K=1,2,4`.
- Preserve all 40 LIBERO tasks and 50 episodes per task.
- Preserve strict record uniqueness, completeness validation, and JSON/CSV summaries.
- Preserve representative videos only for seed 0.
- Do not modify student GPU layouts.
- Do not overwrite or delete the existing V5 result tree.
- Live execution must use a fresh `MATRIX_ROOT`.

---

### Task 1: Add a guarded co-located formal shard layout

**Files:**
- Modify: `evaluation/libero/run_cosmos_progressive_s4_eval.sh:222-318`
- Modify: `evaluation/libero/run_cosmos_progressive_s4_eval.sh:418-425`
- Test: `evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py`

**Interfaces:**
- Consumes: `S4_FORMAL_GPU_LAYOUT` with values `paired` or `colocated`; existing `S4_FORMAL_NUM_SHARDS` and `S4_MODEL_ROLE`.
- Produces: resolved `SHARD_<n>_STUDENT_GPU`, `SHARD_<n>_COSMOS_WORKER_GPU`, `SHARD_<n>_TASK_RANGE`, and `S4_FORMAL_GPU_LAYOUT` diagnostics.

- [ ] **Step 1: Write failing launcher-contract tests**

Add tests equivalent to:

```python
def test_official_teacher_four_shards_can_colocate_on_four_gpus(tmp_path):
    env = _launcher_env(tmp_path)
    env.update(
        S4_MODEL_ROLE="official_teacher",
        COSMOS_POLICY_PATH=str(tmp_path / "teacher"),
        COSMOS_POLICY_TEACHER_LOCK=str(tmp_path / "teacher.lock.json"),
        S4_FORMAL_NUM_SHARDS="4",
        S4_FORMAL_GPU_LAYOUT="colocated",
    )
    Path(env["COSMOS_POLICY_PATH"]).mkdir()
    Path(env["COSMOS_POLICY_TEACHER_LOCK"]).write_text("{}")
    result = run_launcher("formal", env=env)
    _assert_success(result)
    assert _shard_records(result.stdout) == [
        (0, 0, 0, "0,3"),
        (1, 1, 1, "3,6"),
        (2, 2, 2, "6,8"),
        (3, 3, 3, "8,10"),
    ]


def test_colocated_layout_rejects_student_role(tmp_path):
    env = _launcher_env(tmp_path)
    env["S4_FORMAL_NUM_SHARDS"] = "4"
    env["S4_FORMAL_GPU_LAYOUT"] = "colocated"
    result = run_launcher("formal", env=env)
    assert result.returncode == 2
    assert "official_teacher" in result.stderr
```

Extend the strict-control parameterization with an invalid
`S4_FORMAL_GPU_LAYOUT` value.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py \
  -k 'colocate or gpu_layout'
```

Expected: the official-teacher mapping test fails because the launcher still
maps four shards to GPUs `0/1,2/3,4/5,6/7`; the rejection test fails because
student co-location is not rejected.

- [ ] **Step 3: Implement the minimal layout contract**

In the control-validation section:

```bash
S4_FORMAL_GPU_LAYOUT="${S4_FORMAL_GPU_LAYOUT:-paired}"
case "${S4_FORMAL_GPU_LAYOUT}" in
    paired|colocated) ;;
    *) die "S4_FORMAL_GPU_LAYOUT must be paired or colocated" ;;
esac
if [[ "${S4_FORMAL_GPU_LAYOUT}" == "colocated" && \
      "${S4_MODEL_ROLE}" != "official_teacher" ]]; then
    die "S4_FORMAL_GPU_LAYOUT=colocated is only valid for official_teacher"
fi
if [[ "${S4_FORMAL_GPU_LAYOUT}" == "colocated" && \
      "${S4_FORMAL_NUM_SHARDS}" != "4" ]]; then
    die "S4_FORMAL_GPU_LAYOUT=colocated requires S4_FORMAL_NUM_SHARDS=4"
fi
```

In `run_live_evaluation`, retain the existing two-shard mapping. For four
shards, resolve:

```bash
if [[ "${S4_FORMAL_GPU_LAYOUT}" == "colocated" ]]; then
    student_gpus=(0 1 2 3)
    worker_gpus=(0 1 2 3)
else
    student_gpus=(0 2 4 6)
    worker_gpus=(1 3 5 7)
fi
task_starts=(0 3 6 8)
task_ends=(3 6 8 10)
```

Emit `S4_FORMAL_GPU_LAYOUT` beside the existing resolved controls.

- [ ] **Step 4: Run focused and existing layout tests and verify GREEN**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py
```

Expected: all tests pass, including unchanged two-shard and paired eight-GPU
mapping tests.

- [ ] **Step 5: Commit Task 1**

```bash
git add \
  evaluation/libero/run_cosmos_progressive_s4_eval.sh \
  evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py
git commit -m "feat: support colocated Cosmos teacher shards"
```

### Task 2: Select four co-located shards in the four-GPU teacher wrapper

**Files:**
- Modify: `evaluation/libero/run_cosmos_official_teacher_124_eval_4gpu.sh`
- Modify: `evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_4gpu.py`

**Interfaces:**
- Consumes: the Task 1 `S4_FORMAL_GPU_LAYOUT=colocated` contract.
- Produces: a teacher-only wrapper environment with
  `S4_FORMAL_NUM_SHARDS=4`, `S4_FORMAL_GPU_LAYOUT=colocated`, and
  `S4_VIDEO_SEEDS=0`.

- [ ] **Step 1: Change the wrapper test expectation and verify RED**

Extend the sentinel payload with `S4_FORMAL_GPU_LAYOUT`, rename the test to
describe four co-located shards, and expect:

```python
{
    "argv": ["dry-run"],
    "S4_MATRIX_ROLES": "official_teacher",
    "S4_FORMAL_NUM_SHARDS": "4",
    "S4_FORMAL_GPU_LAYOUT": "colocated",
    "S4_VIDEO_SEEDS": "0",
    "S4_EPISODES_PER_TASK": "50",
    "S4_CKPT_ROOT": env["COSMOS_POLICY_PATH"],
    "MATRIX_ROOT": str(matrix_root),
}
```

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_4gpu.py
```

Expected: FAIL because the wrapper still forces two shards and does not export
the layout.

- [ ] **Step 2: Implement the wrapper settings**

Replace the paired-shard exports with:

```bash
export COSMOS_TEACHER_FORMAL_NUM_SHARDS=4
export S4_FORMAL_GPU_LAYOUT=colocated
export S4_VIDEO_SEEDS=0
```

Update the comment to state that each of four visible GPUs hosts one evaluator
and one official-teacher raw worker.

- [ ] **Step 3: Run wrapper and integration tests**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_4gpu.py \
  evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_8gpu.py \
  evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py \
  evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py
```

Expected: all tests pass.

- [ ] **Step 4: Verify shell syntax and a real dry-run**

Run:

```bash
bash -n evaluation/libero/run_cosmos_progressive_s4_eval.sh
bash -n evaluation/libero/run_cosmos_official_teacher_124_eval_4gpu.sh
MATRIX_ROOT=/tmp/cosmos-teacher-colocated-dry-run \
  COSMOS_POLICY_PATH=/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
  COSMOS_POLICY_TEACHER_LOCK=/kpfs-intern/jialongliu/results/cosmos_teacher_eval_assets/teacher.lock.json \
  S4_PROMPT_TABLE=/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot/empty_emb.pt \
  bash evaluation/libero/run_cosmos_official_teacher_124_eval_4gpu.sh dry-run \
  > /tmp/cosmos-teacher-colocated-dry-run.log
```

Verify the log contains all mappings:

```text
S4_FORMAL_GPU_LAYOUT=colocated
SHARD_0_STUDENT_GPU=0
SHARD_0_COSMOS_WORKER_GPU=0
SHARD_1_STUDENT_GPU=1
SHARD_1_COSMOS_WORKER_GPU=1
SHARD_2_STUDENT_GPU=2
SHARD_2_COSMOS_WORKER_GPU=2
SHARD_3_STUDENT_GPU=3
SHARD_3_COSMOS_WORKER_GPU=3
```

Verify the dry-run did not create its destination.

- [ ] **Step 5: Commit Task 2**

```bash
git add \
  evaluation/libero/run_cosmos_official_teacher_124_eval_4gpu.sh \
  evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_4gpu.py
git commit -m "feat: use all four GPUs for Cosmos teacher eval"
```

### Task 3: Final verification and launch handoff

**Files:**
- No production changes.

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: verified fresh-root launch command and a documented live-smoke gate.

- [ ] **Step 1: Run the complete related regression suite**

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py \
  evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py \
  evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_4gpu.py \
  evaluation/libero/tests/test_run_cosmos_official_teacher_124_eval_8gpu.py \
  evaluation/libero/tests/test_cosmos_progressive_eval_summary.py
bash -n evaluation/libero/run_cosmos_progressive_s4_eval.sh
bash -n evaluation/libero/run_cosmos_official_teacher_124_eval_4gpu.sh
git diff --check
```

Expected: zero failures and no whitespace or syntax errors.

- [ ] **Step 2: Confirm no live task was started**

Check that the dry-run destination does not exist and that no matching rollout
process was created by verification.

- [ ] **Step 3: Provide the live command with a fresh V6 root**

Use:

```bash
MATRIX_ROOT=/kpfs-intern/jialongliu/results/cosmos-teacher-k124-4gpu-liberoenv-20260727-v6
```

Keep all other official checkpoint, lock, environment, prompt-table, and
50-episode values identical to the validated V5 command.
