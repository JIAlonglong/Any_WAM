# Cosmos Progressive Joint 1/2/4-Step LIBERO Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible eight-GPU closed-loop LIBERO matrix that sequentially evaluates joint video/action rollout depths K=1, K=2, and K=4 for 500 episodes each while saving only a fixed small video subset.

**Architecture:** Make rollout depth an explicit value that flows from CLI to the Cosmos progressive engine and then to the single joint FlowMap integration used by both video and action. Generalize the formal launcher from its backward-compatible two-shard K=4 default to an optional four-shard eight-GPU plan, then add a no-overwrite outer matrix launcher that runs K=1, K=2, and K=4 sequentially and validates separate summaries.

**Tech Stack:** Bash, Python 3.10, PyTorch, existing Cosmos progressive LIBERO client/service, pytest.

## Global Constraints

- The evaluated checkpoint is the local 16-channel `universal-video-action/checkpoints/step_5000/online_student/transformer`.
- Accepted joint rollout depths are exactly `1`, `2`, and `4`.
- One selected K drives both video and action in the same `_student_euler_integrate` call.
- Each K evaluates LIBERO-10 with seeds `0..49`: exactly 500 durable records.
- K values run sequentially in the order `1`, `2`, `4`.
- Eight GPUs form four fixed student/worker pairs: `0/1`, `2/3`, `4/5`, `6/7`.
- Four-shard task ranges are `[0,3)`, `[3,6)`, `[6,8)`, `[8,10)`.
- Only seeds `0,1` save videos: twenty videos per K and sixty total.
- Non-video episodes do not retain frames in memory and record `video_path=null`.
- Paper-only offline S4 metrics remain fixed at K=4 and are not run as K=1/K=2 metrics.
- Existing callers default to K=4 and the existing two-shard four-GPU topology.
- Live GPU evaluation is not launched during implementation or verification.

---

### Task 1: Parameterize the joint video/action rollout depth

**Files:**
- Modify: `evaluation/libero/rollout_cosmos_progressive_s4.py`
- Modify: `evaluation/libero/cosmos_progressive_s4_server.py`
- Test: `evaluation/libero/tests/test_cosmos_progressive_s4_service.py`

**Interfaces:**
- Produces: `normalize_student_steps(value: int) -> int`, accepting only `1`, `2`, or `4`.
- Produces: CLI option `--student-steps`, default `4`.
- Produces: `CosmosProgressiveS4Engine(..., student_steps: int = 4)`.
- Produces: response metadata `student_steps: int`.
- Consumes: existing `FlowMapJointS4Runner.__call__(..., k_steps: int)`.

- [ ] **Step 1: Write failing CLI and joint-runner tests**

Add tests that prove the default is four, invalid values are rejected, and K=1/K=2/K=4 reach the joint integration unchanged:

```python
@pytest.mark.parametrize("steps", [1, 2, 4])
def test_joint_runner_uses_requested_steps_for_video_and_action(steps):
    harness = RecordingHarness()
    runner = make_joint_runner(harness)
    runner(video_x0(), action_x0(), text_emb(), noise=noise(),
           t1000=t1000(), t0=t0(), k_steps=steps)
    assert harness.calls == [{
        "K_steps": steps,
        "return_final_action": True,
        "return_final_action_state": True,
    }]


@pytest.mark.parametrize("steps", [0, 3, 5])
def test_cli_rejects_unsupported_joint_steps(steps):
    with pytest.raises(SystemExit):
        parse_args(base_args() + ["--student-steps", str(steps)])
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  evaluation/libero/tests/test_cosmos_progressive_s4_service.py \
  -k "requested_steps or unsupported_joint_steps"
```

Expected: failures showing K=1/K=2 are rejected by the current hard-coded
`S4_STEPS=4` contract and `--student-steps` is absent.

- [ ] **Step 3: Implement step validation and propagation**

Implement a single validator and remove the hard-coded K from the integration:

```python
SUPPORTED_STUDENT_STEPS = (1, 2, 4)


def normalize_student_steps(value: int) -> int:
    value = int(value)
    if value not in SUPPORTED_STUDENT_STEPS:
        raise ValueError(
            f"student_steps must be one of {SUPPORTED_STUDENT_STEPS}, got {value}"
        )
    return value
```

Add the CLI option:

```python
parser.add_argument(
    "--student-steps",
    type=int,
    choices=SUPPORTED_STUDENT_STEPS,
    default=4,
)
```

In `FlowMapJointS4Runner.__call__`, validate `k_steps` and pass it directly:

```python
k_steps = normalize_student_steps(k_steps)
...
K_steps=k_steps,
return_final_action=True,
return_final_action_state=True,
```

Store the selected value on `CosmosProgressiveS4Engine`, call the joint runner
with `k_steps=self.student_steps`, and include it in service metadata:

```python
self.student_steps = normalize_student_steps(student_steps)
...
student_action = self.joint_s4_runner(..., k_steps=self.student_steps)
...
"student_steps": self.student_steps,
```

Construct the engine with `args.student_steps`.

- [ ] **Step 4: Run the complete service test file**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  evaluation/libero/tests/test_cosmos_progressive_s4_service.py
```

Expected: all tests pass, including the legacy default K=4 assertions.

- [ ] **Step 5: Commit Task 1**

```bash
git add \
  evaluation/libero/rollout_cosmos_progressive_s4.py \
  evaluation/libero/cosmos_progressive_s4_server.py \
  evaluation/libero/tests/test_cosmos_progressive_s4_service.py
git commit -m "feat: parameterize Cosmos joint rollout steps"
```

---

### Task 2: Save only explicitly selected episode videos

**Files:**
- Modify: `evaluation/libero/cosmos_progressive_s4_client.py`
- Modify: `evaluation/libero/rollout_cosmos_progressive_s4.py`
- Test: `evaluation/libero/tests/test_cosmos_progressive_s4_service.py`

**Interfaces:**
- Produces: CLI flag `--save-video`, default false.
- Produces: `CosmosProgressiveS4Client.run_libero_task(..., save_video: bool = True)`.
- Consumes: existing `video_path` and `save_video_fn` optional parameters in `run_with_env`.

- [ ] **Step 1: Write failing selective-capture tests**

Add a test that uses an extracting sentinel and proves frames are not collected
when video output is disabled:

```python
def test_non_video_episode_does_not_extract_or_save_frames(tmp_path):
    extracted = []
    saved = []
    record = client.run_with_env(
        ...,
        extract_video_fn=lambda obs: extracted.append(obs),
        save_video_fn=lambda frames, path: saved.append((frames, path)),
        video_path=None,
    )
    assert extracted == []
    assert saved == []
    assert record["video_path"] is None
```

Add a `run_libero_task` forwarding test:

```python
@pytest.mark.parametrize(("save_video", "has_path"), [(False, False), (True, True)])
def test_run_libero_task_controls_video_capture(save_video, has_path, ...):
    record = client.run_libero_task(..., save_video=save_video)
    assert (record["video_path"] is not None) is has_path
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  evaluation/libero/tests/test_cosmos_progressive_s4_service.py \
  -k "does_not_extract or controls_video_capture"
```

Expected: current code extracts every frame and `run_libero_task` has no
`save_video` argument.

- [ ] **Step 3: Implement capture gating**

In `run_with_env`, compute capture once and gate frame extraction:

```python
capture_video = video_path is not None and save_video_fn is not None
...
if capture_video:
    frames.append(extract_video_fn(obs))
...
if capture_video and frames:
    save_video_fn(frames, video_path)
```

In `run_libero_task`, add `save_video: bool = True` and set:

```python
video_path = planned_video_path if save_video else None
```

Add `--save-video` to the rollout CLI and pass `save_video=args.save_video` to
every real task call.

- [ ] **Step 4: Run service tests**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  evaluation/libero/tests/test_cosmos_progressive_s4_service.py
```

Expected: all tests pass and existing explicit-video tests still write MP4
paths.

- [ ] **Step 5: Commit Task 2**

```bash
git add \
  evaluation/libero/cosmos_progressive_s4_client.py \
  evaluation/libero/rollout_cosmos_progressive_s4.py \
  evaluation/libero/tests/test_cosmos_progressive_s4_service.py
git commit -m "feat: select Cosmos LIBERO video episodes"
```

---

### Task 3: Generalize formal evaluation to K provenance and four shards

**Files:**
- Modify: `evaluation/libero/run_cosmos_progressive_s4_eval.sh`
- Modify: `evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py`

**Interfaces:**
- Consumes: rollout CLI `--student-steps` and `--save-video`.
- Produces: environment controls `S4_STUDENT_STEPS`, `S4_FORMAL_NUM_SHARDS`, and `S4_VIDEO_SEEDS`.
- Produces: summaries containing `student_steps`.

- [ ] **Step 1: Write failing launcher contract tests**

Add dry-run assertions for the four-shard plan:

```python
def test_formal_four_shards_use_all_eight_gpus_and_joint_k2(tmp_path):
    result = run_launcher(
        "formal",
        dry_run=True,
        env={
            "S4_STUDENT_STEPS": "2",
            "S4_FORMAL_NUM_SHARDS": "4",
            "S4_VIDEO_SEEDS": "0,1",
        },
    )
    assert shard_records(result.stdout) == [
        (0, 0, 1, "0,3"),
        (1, 2, 3, "3,6"),
        (2, 4, 5, "6,8"),
        (3, 6, 7, "8,10"),
    ]
    assert result.stdout.count("--student-steps 2") == 204
    assert result.stdout.count("--save-video") == 8
```

The exact command counts include four preflights plus four shards times fifty
seeds; video is enabled only for seeds zero and one on each shard.

Add merger fixtures where all records include `"student_steps": 2`, then prove
one mismatched record fails with `step mismatch`.

- [ ] **Step 2: Run launcher tests and verify RED**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py
```

Expected: failures because the launcher only supports two shards, hard-codes
four steps, and does not select videos.

- [ ] **Step 3: Parse and validate the formal controls**

Add strict Bash validation:

```bash
S4_STUDENT_STEPS="${S4_STUDENT_STEPS:-4}"
case "${S4_STUDENT_STEPS}" in 1|2|4) ;; *)
    die "S4_STUDENT_STEPS must be 1, 2, or 4"
esac

S4_FORMAL_NUM_SHARDS="${S4_FORMAL_NUM_SHARDS:-2}"
case "${S4_FORMAL_NUM_SHARDS}" in 2|4) ;; *)
    die "S4_FORMAL_NUM_SHARDS must be 2 or 4"
esac

S4_VIDEO_SEEDS="${S4_VIDEO_SEEDS:-}"
```

Parse `S4_VIDEO_SEEDS` into a set after rejecting non-integers and values
outside `0..49`.

- [ ] **Step 4: Add deterministic shard plans and command propagation**

Represent the two valid plans as parallel Bash arrays. For the four-shard
case:

```bash
shard_ids=(0 1 2 3)
student_gpus=(0 2 4 6)
worker_gpus=(1 3 5 7)
task_starts=(0 3 6 8)
task_ends=(3 6 8 10)
```

Every preflight and rollout command receives:

```bash
--student-steps "${S4_STUDENT_STEPS}"
```

For a seed in `S4_VIDEO_SEEDS`, append:

```bash
--save-video
```

Run all configured shards concurrently and wait for every PID before merging.

- [ ] **Step 5: Generalize the embedded merge verifier**

Pass requested K and a serialized shard plan to the verifier. Validate each
record:

```python
if int(record.get("student_steps", -1)) != requested_steps:
    raise SystemExit(
        f"step mismatch: expected={requested_steps} "
        f"got={record.get('student_steps')!r}"
    )
```

Derive allowed tasks from the serialized plan rather than hard-coded shard
zero/one ranges. Include `"student_steps": requested_steps` in
`formal_summary.json`.

- [ ] **Step 6: Run launcher and service regression tests**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py \
  evaluation/libero/tests/test_cosmos_progressive_s4_service.py
bash -n evaluation/libero/run_cosmos_progressive_s4_eval.sh
```

Expected: all tests pass; default tests still show K=4 and two shards.

- [ ] **Step 7: Commit Task 3**

```bash
git add \
  evaluation/libero/run_cosmos_progressive_s4_eval.sh \
  evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py
git commit -m "feat: shard joint Cosmos LIBERO evaluation across eight GPUs"
```

---

### Task 4: Add the sequential K=1/K=2/K=4 matrix launcher

**Files:**
- Create: `evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh`
- Create: `evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py`
- Modify: `evaluation/libero/cosmos_progressive_s4_env.sh`

**Interfaces:**
- Consumes: formal launcher controls from Task 3.
- Produces: `MATRIX_ROOT/k1`, `MATRIX_ROOT/k2`, `MATRIX_ROOT/k4`.
- Produces: top-level `matrix_summary.json`.

- [ ] **Step 1: Write failing matrix dry-run tests**

Create a test that invokes the absent launcher with sentinel child commands:

```python
def test_matrix_dry_run_plans_three_full_sequential_evaluations(tmp_path):
    root = tmp_path / "matrix"
    result = run_matrix("dry-run", root=root)
    assert result.returncode == 0
    assert parse_k_order(result.stdout) == [1, 2, 4]
    assert result.stdout.count("REQUESTED_RECORDS=500") == 3
    assert result.stdout.count("FORMAL_NUM_SHARDS=4") == 3
    assert result.stdout.count("VIDEO_SEEDS=0,1") == 3
    assert not root.exists()
```

Add rejection tests for an existing matrix root and a child failure test that
proves K=4 is not started after K=2 fails.

- [ ] **Step 2: Run the matrix tests and verify RED**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py
```

Expected: failure because the matrix launcher does not exist.

- [ ] **Step 3: Implement the no-overwrite sequential launcher**

The launcher accepts exactly `run` or `dry-run`, requires `MATRIX_ROOT`,
`S4_CKPT_ROOT`, `S4_DATASET_PATH`, and `S4_EMPTY_EMBEDDING`, and sets:

```bash
readonly STEPS=(1 2 4)
export S4_FORMAL_NUM_SHARDS=4
export S4_VIDEO_SEEDS="${S4_VIDEO_SEEDS:-0,1}"
```

For each K:

```bash
export S4_STUDENT_STEPS="${k}"
export EVAL_ROOT="${MATRIX_ROOT}/k${k}"
bash evaluation/libero/run_cosmos_progressive_s4_eval.sh formal
```

Use `S4_DRY_RUN=1` in dry-run mode. If no prompt table was supplied, set
`S4_PROMPT_TABLE="${MATRIX_ROOT}/k1/prompt_embeddings.pt"` after K=1
completes and reuse it for K=2/K=4.

After all three summaries exist, write `matrix_summary.json` containing:

```json
{
  "schema": "cosmos_progressive_joint_124_matrix_v1",
  "checkpoint": "<resolved checkpoint>",
  "steps": [1, 2, 4],
  "summaries": {
    "1": "<MATRIX_ROOT>/k1/formal_summary.json",
    "2": "<MATRIX_ROOT>/k2/formal_summary.json",
    "4": "<MATRIX_ROOT>/k4/formal_summary.json"
  }
}
```

The summary writer must verify each child summary reports 500 records, the
matching K, and the same checkpoint before publishing the matrix summary.

- [ ] **Step 4: Add matrix defaults to the environment helper**

Add only defaults, preserving caller overrides:

```bash
export S4_VIDEO_SEEDS="${S4_VIDEO_SEEDS:-0,1}"
export S4_FORMAL_NUM_SHARDS="${S4_FORMAL_NUM_SHARDS:-4}"
export MATRIX_ROOT="${MATRIX_ROOT:-${S4_RESULTS_ROOT}/cosmos_joint_124_${S4_RUN_ID}}"
```

- [ ] **Step 5: Run matrix and full evaluation launcher tests**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py \
  evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py \
  evaluation/libero/tests/test_run_cosmos_progressive_s4_suite.py \
  evaluation/libero/tests/test_cosmos_progressive_s4_env.py
bash -n evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh
bash -n evaluation/libero/run_cosmos_progressive_s4_eval.sh
```

Expected: all tests and syntax checks pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add \
  evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh \
  evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py \
  evaluation/libero/cosmos_progressive_s4_env.sh
git commit -m "feat: add sequential joint 1-2-4 LIBERO matrix"
```

---

### Task 5: Final verification and launch handoff

**Files:**
- Modify only if verification exposes a defect in files owned by Tasks 1-4.

**Interfaces:**
- Consumes: all preceding task interfaces.
- Produces: a verified dry-run launch command; no live process.

- [ ] **Step 1: Run the complete focused regression set**

Run:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q -p no:cacheprovider \
  evaluation/libero/tests/test_cosmos_progressive_s4_service.py \
  evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py \
  evaluation/libero/tests/test_run_cosmos_progressive_s4_suite.py \
  evaluation/libero/tests/test_cosmos_progressive_s4_env.py \
  evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py
```

Expected: all tests pass.

- [ ] **Step 2: Run static checks**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m py_compile \
  evaluation/libero/rollout_cosmos_progressive_s4.py \
  evaluation/libero/cosmos_progressive_s4_server.py \
  evaluation/libero/cosmos_progressive_s4_client.py
bash -n evaluation/libero/run_cosmos_progressive_s4_eval.sh
bash -n evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh
git diff --check
```

Expected: exit zero for every command.

- [ ] **Step 3: Dry-run the exact local checkpoint matrix**

Use a verified-absent output root:

```bash
source /kpfs-intern/jialongliu/miniforge3/bin/activate
conda activate flashwam
cd /kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/cosmos-progressive-s4-eval
source evaluation/libero/cosmos_progressive_s4_env.sh
export PYTHONPATH="${COSMOS_POLICY_EXTRA_PYTHONPATH}:${COSMOS_PREDICT2_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export S4_CKPT_ROOT=/kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_cosmos_dance_windowfix_8gpu_20260723/universal-video-action/checkpoints/step_5000/online_student/transformer
export MATRIX_ROOT=/kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/eval_libero_cosmos_joint_124_step5000_20260724
bash evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh dry-run
```

Verify:

```text
K order: 1,2,4
500 records per K
four shards per K
video seeds 0,1
no MATRIX_ROOT created
no Python/GPU process started
```

- [ ] **Step 4: Review the full feature diff**

Review from the design commit through HEAD for:

- joint video/action K fidelity;
- unchanged legacy K=4 behavior;
- exact 500-record and shard coverage;
- record and summary step provenance;
- video capture suppression;
- no-overwrite and fail-stop behavior;
- dry-run mutation freedom.

Expected: no unresolved Critical, Important, or Minor findings.

- [ ] **Step 5: Record final status**

Report the final branch, HEAD, test counts, dry-run contract, driver
requirement `>=570.124.06`, exact launch command, and the three summary paths.
Do not launch the live matrix.
