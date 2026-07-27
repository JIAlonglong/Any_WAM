# Cosmos Progressive S4 Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paper-faithful offline diagnostic evaluator and a dedicated 16-channel Cosmos Progressive S4 LIBERO rollout path, then provide one-GPU smoke and two-replica/four-GPU launch modes.

**Architecture:** Keep the public S4 transformer isolated behind Cosmos-specific helpers. The offline evaluator reuses fixed same-prior anchors but dynamically queries the official teacher at student-visited states. The live server uses Cosmos for every replan anchor and the student only for the four-step correction, while the existing LIBERO client protocol remains the environment boundary.

**Tech Stack:** Python 3.10, PyTorch, safetensors, existing FlowMap/Cosmos adapter, WebSocket evaluation transport, pytest, bash.

## Global Constraints

- Source branch: `origin/research/cosmos-stage2-progressive`; never change the user's dirty `main` checkout.
- Public checkpoint: exactly `progressive_stage2_full/s4/step_5000/online_student/transformer`; expect `WanTransformer3DModel` with 16 input and 16 output channels.
- Student deployment grid: exactly `(1000, 750, 500, 250, 0)`; teacher continuation grid uses 125-point N=8 increments.
- Paper Table-3 metrics use video latent only: `G_anchor`, `G_comp`, `video_ep`, `field_match`.
- A live replan must obtain a Cosmos-generated `[1,16,9,28,28]` latent/action anchor and use the dataset's `[1,512,4096]` prompt embedding; never substitute a Cosmos T5 embedding or the 48-channel WanVA server.
- Formal evaluation: `libero_10`, 50 shared episode seeds per task, two disjoint two-GPU shards (`student,worker`), and a validated merge before aggregate statistics.
- Every new output path is passed explicitly; dry-run creates no output or cache directories.

---

### Task 1: Paper metric primitives and protocol validation

**Files:**
- Create: `distillation_flowmap/cosmos_progressive_paper_metrics.py`
- Create: `distillation_flowmap/tests/test_cosmos_progressive_paper_metrics.py`

**Interfaces:**
- Produces `DEPLOYMENT_GRID`, `internal_rollout_nodes()`, `composition_pairs()`, `mean_video_mse()`, `paper_metric_template()`, and `merge_task_records(records)`.
- Consumed by the paper evaluator and final result merger.

- [ ] **Step 1: Write the failing tests**

```python
from distillation_flowmap.cosmos_progressive_paper_metrics import (
    DEPLOYMENT_GRID, composition_pairs, internal_rollout_nodes, mean_video_mse,
)

def test_s4_paper_grid_uses_only_deployed_internal_states():
    assert DEPLOYMENT_GRID == (1000, 750, 500, 250, 0)
    assert internal_rollout_nodes() == (750, 500, 250)
    assert composition_pairs() == ((750, 500), (500, 250))

def test_mean_video_mse_ignores_action_channels():
    video_left = torch.zeros(1, 16, 1, 1, 1)
    video_right = torch.ones_like(video_left)
    action_left = torch.zeros(1, 30, 1, 1, 1)
    action_right = torch.full_like(action_left, 999.0)
    assert mean_video_mse(video_left, video_right) == 1.0
```

- [ ] **Step 2: Run the tests to verify RED**

Run:
`PYTHONDONTWRITEBYTECODE=1 /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest distillation_flowmap/tests/test_cosmos_progressive_paper_metrics.py -q`

Expected: import failure because `cosmos_progressive_paper_metrics` does not exist.

- [ ] **Step 3: Implement the minimal pure helpers**

```python
DEPLOYMENT_GRID = (1000, 750, 500, 250, 0)

def internal_rollout_nodes():
    return DEPLOYMENT_GRID[1:-1]

def composition_pairs():
    return tuple(zip(DEPLOYMENT_GRID[1:-2], DEPLOYMENT_GRID[2:-1]))

def mean_video_mse(left, right):
    if left.shape != right.shape:
        raise ValueError("video tensors must have equal shapes")
    return (left.float() - right.float()).square().mean().item()
```

Add record validation that rejects missing or duplicate `(task, record_index, pair_id)` keys and macro-averages per task rather than by raw sample count.

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run the command in Step 2.

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add distillation_flowmap/cosmos_progressive_paper_metrics.py \
        distillation_flowmap/tests/test_cosmos_progressive_paper_metrics.py
git commit -m "feat: add progressive S4 paper metric primitives"
```

### Task 2: Paper-faithful offline S4 evaluator

**Files:**
- Create: `distillation_flowmap/eval_cosmos_progressive_s4_paper.py`
- Create: `distillation_flowmap/tests/test_cosmos_progressive_s4_paper_eval.py`
- Modify: `distillation_flowmap/build_cosmos_progressive_teacher_cache.py` only if it must persist a missing deterministic anchor field.

**Interfaces:**
- Consumes the existing test manifest, fixed `eval_pairs.json`, cache payload schema `cosmos_progressive_teacher_cache_v1`, public S4 transformer directory, and the Cosmos policy adapter.
- Produces `records.jsonl` and `summary.json` with per-node metrics plus task-macro aggregates.

- [ ] **Step 1: Write failing evaluator-contract tests**

```python
def test_paper_evaluator_requires_full_s4_grid_and_test_manifest(tmp_path):
    args = parse_args([
        "--checkpoint-transformer", "/ckpt",
        "--dataset-path", "/data",
        "--manifest", "/proto/test_manifest.json",
        "--pairs", "/proto/eval_pairs.json",
        "--cache-dir", "/cache",
        "--output-dir", str(tmp_path),
    ])
    assert args.student_steps == 4
    assert args.teacher_steps == 8

def test_paper_metric_names_are_the_table_three_names():
    assert PAPER_METRIC_NAMES == ("g_anchor", "g_comp", "video_ep", "field_match")
```

- [ ] **Step 2: Run the test to verify RED**

Run:
`PYTHONDONTWRITEBYTECODE=1 /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest distillation_flowmap/tests/test_cosmos_progressive_s4_paper_eval.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement the evaluator**

Implement these exact semantics:

```python
# One same-prior cache payload per record and fixed pair.
y0 = cache["teacher_x_r"]
joint_states = rollout_student_with_joint_action_trajectory(
    x1=cache["video_noise"], action_anchor=cache["teacher_action_x0"],
    grid=(1000, 750, 500, 250, 0), k_steps=4,
)

for r in (750, 500, 250):
    z_video, z_action = joint_states[r]
    direct_video = finite_student_map(z_video, z_action, r=r, target=0)
    video_ep = mean_video_mse(direct_video, y0)
    teacher_continuation = integrate_cosmos_teacher(z_video, r=r, target=0, steps=r // 125)
    g_anchor = mean_video_mse(teacher_continuation, y0)
    field_match = mean_video_mse(
        equal_time_student_field(z_video, z_action, r=r),
        equal_time_cosmos_field(z_video, r=r),
    )
for r, s in ((750, 500), (500, 250)):
    g_comp = mean_video_mse(
        finite_student_map(*joint_states[r], r=r, target=0),
        finite_student_map(*finite_student_map_joint(*joint_states[r], r=r, target=s), r=s, target=0),
    )
```

The implementation may expose reusable helpers instead of using these names verbatim, but it must retain joint action state at every deployment node. Cache `y0` is reusable; all off-path teacher continuations are dynamic and must never be substituted by `teacher_path`.

Add `--cache-only-smoke` that performs one student K=4 forward with `--skip-same-state-velocity`, writes no paper metric claim, and validates all 16-channel shapes before live Cosmos queries.

- [ ] **Step 4: Run focused tests and the existing progressive evaluator tests**

Run:
`PYTHONDONTWRITEBYTECODE=1 /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest distillation_flowmap/tests/test_cosmos_progressive_paper_metrics.py distillation_flowmap/tests/test_cosmos_progressive_s4_paper_eval.py distillation_flowmap/tests/test_cosmos_progressive_metrics.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add distillation_flowmap/eval_cosmos_progressive_s4_paper.py \
        distillation_flowmap/tests/test_cosmos_progressive_s4_paper_eval.py \
        distillation_flowmap/build_cosmos_progressive_teacher_cache.py
git commit -m "feat: add paper-faithful progressive S4 offline evaluation"
```

### Task 3: Dedicated 16-channel Cosmos S4 rollout service

**Files:**
- Create: `evaluation/libero/cosmos_progressive_s4_server.py`
- Create: `evaluation/libero/cosmos_progressive_s4_client.py`
- Create: `evaluation/libero/rollout_cosmos_progressive_s4.py`
- Create: `evaluation/libero/tests/test_cosmos_progressive_s4_service.py`

**Interfaces:**
- Server accepts an existing LIBERO-style reset/infer request with task text, two RGB observations, and 8-D LIBERO state.
- Server returns a valid `float32 [16,7]` action chunk and records the raw 16-channel anchor, decision duration, and S4 checkpoint identifier.
- Client owns the LIBERO environment, shared initial states, video collection, and success records.

- [ ] **Step 1: Write failing pure service tests**

```python
def test_live_s4_request_uses_cosmos_raw_observation_contract():
    payload = build_cosmos_raw_request(libero_obs=OBS, prompt="open the drawer")
    assert payload["raw_primary_image"].shape[-1] == 3
    assert payload["raw_wrist_image"].shape[-1] == 3
    assert payload["raw_proprio"].shape == (9,)

def test_live_s4_action_decoder_returns_only_16_valid_seven_dof_actions():
    decoded = decode_student_action(flowmap_action_tensor, template)
    assert decoded.shape == (16, 7)
    assert decoded.dtype == np.float32

def test_prompt_table_rejects_a_missing_libero_task_embedding():
    with pytest.raises(KeyError, match="missing prompt embedding"):
        PromptEmbeddingTable({}).get("unseen task")
```

- [ ] **Step 2: Run the test to verify RED**

Run:
`PYTHONDONTWRITEBYTECODE=1 /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest evaluation/libero/tests/test_cosmos_progressive_s4_service.py -q`

Expected: import failure because the service module does not exist.

- [ ] **Step 3: Implement the service and client**

Implement these exact constraints:

```python
# At every control cycle, do not reuse a previous teacher anchor.
anchor = cosmos_teacher.predict_raw_latent_target(raw_batch, noise=noise, t=t1000, r=t0)
video_x0 = anchor["cosmos_latent_x0"]       # [1, 16, 9, 28, 28]
action_x0 = cosmos_actions_to_flowmap_x0(anchor["actions"], target_shape=template.shape, ...)
text_emb = prompt_table.get(prompt)          # [1, 512, 4096]
actions = run_joint_s4_student(video_x0, action_x0, text_emb, k_steps=4)
return actions[:16, :7].astype(np.float32)
```

Use the existing `rollout_cosmos_policy.py` reset, warm-up, frame flip, and 9-D proprio construction. Use `load_stage1_model`, `prepare_base_dict`, and `FlowMapStepMixin._student_euler_integrate` from the progressive evaluator rather than the generic 48-channel server. The client writes one JSON record per requested `(task_idx, episode_idx)` and never treats a failed server connection as a successful trial.

- [ ] **Step 4: Run focused tests**

Run:
`PYTHONDONTWRITEBYTECODE=1 /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest evaluation/libero/tests/test_cosmos_progressive_s4_service.py distillation_flowmap/tests/test_cosmos_progressive_paper_metrics.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add evaluation/libero/cosmos_progressive_s4_server.py \
        evaluation/libero/cosmos_progressive_s4_client.py \
        evaluation/libero/rollout_cosmos_progressive_s4.py \
        evaluation/libero/tests/test_cosmos_progressive_s4_service.py
git commit -m "feat: add Cosmos progressive S4 LIBERO rollout service"
```

### Task 4: Reproducible smoke and four-GPU orchestration

**Files:**
- Create: `evaluation/libero/run_cosmos_progressive_s4_eval.sh`
- Create: `evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py`
- Create: `docs/experiments/2026-07-16-cosmos-progressive-s4-evaluation.md`

**Interfaces:**
- `bash evaluation/libero/run_cosmos_progressive_s4_eval.sh smoke|gate|formal|dry-run`.
- Emits machine-readable `KEY=value` records and `COMMAND=` lines before launches.
- `formal` launches exactly two non-overlapping task shards: `0 5` and `5 10`.

- [ ] **Step 1: Write failing launcher tests**

```python
def test_formal_mode_assigns_two_student_worker_pairs():
    output = run_launcher("formal", dry_run=True)
    assert "SHARD_0_STUDENT_GPU=0" in output
    assert "SHARD_0_COSMOS_WORKER_GPU=1" in output
    assert "SHARD_1_STUDENT_GPU=2" in output
    assert "SHARD_1_COSMOS_WORKER_GPU=3" in output
    assert "SHARD_0_TASK_RANGE=0,5" in output
    assert "SHARD_1_TASK_RANGE=5,10" in output

def test_dry_run_has_no_output_side_effects(tmp_path):
    output_dir = tmp_path / "never-created"
    run_launcher("dry-run", output_dir=output_dir)
    assert not output_dir.exists()
```

- [ ] **Step 2: Run the launcher test to verify RED**

Run:
`PYTHONDONTWRITEBYTECODE=1 /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py -q`

Expected: failure because the launcher does not exist.

- [ ] **Step 3: Implement smoke, gate, formal, and merge modes**

Use explicit, non-overwriting defaults:

```bash
S4_CKPT_ROOT="${S4_CKPT_ROOT:?set the downloaded public S4 transformer root}"
EVAL_ROOT="${EVAL_ROOT:?set a new empty result root}"
COSMOS_POLICY_PATH="${COSMOS_POLICY_PATH:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"

# gate: CUDA_VISIBLE_DEVICES=0 and worker GPU 1
# formal: shard 0 uses 0,1 task range [0,5); shard 1 uses 2,3 task range [5,10)
```

`smoke` runs one cache-backed S4 forward on one GPU and checks its JSON result. `gate` runs 5 shared seeds per task. `formal` runs 50 shared seeds per task and invokes a merge command that rejects missing, duplicate, mismatched-checkpoint, or mismatched-seed records before writing macro success and bootstrap CI summaries. The documented command must include the exact 9.515-GiB ModelScope transformer subtree, cache/protocol generation order, and no-overwrite policy.

- [ ] **Step 4: Run syntax, unit, and dry-run verification**

Run:
`bash -n evaluation/libero/run_cosmos_progressive_s4_eval.sh`

Run:
`PYTHONDONTWRITEBYTECODE=1 /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py evaluation/libero/tests/test_cosmos_progressive_s4_service.py distillation_flowmap/tests/test_cosmos_progressive_paper_metrics.py -q`

Run:
`bash evaluation/libero/run_cosmos_progressive_s4_eval.sh dry-run`

Expected: all tests pass, bash syntax succeeds, and dry-run prints commands without creating result directories.

- [ ] **Step 5: Commit**

```bash
git add evaluation/libero/run_cosmos_progressive_s4_eval.sh \
        evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py \
        docs/experiments/2026-07-16-cosmos-progressive-s4-evaluation.md
git commit -m "feat: add progressive S4 evaluation launcher"
```

## Verification matrix

- Run all existing progressive tests plus all new tests with the flashwam Python.
- Run `bash -n` on the launcher.
- Download only the published S4 transformer subtree into a new cache path and verify its config reports 16 in/out channels before loading.
- Run `smoke` on the currently visible GPU. It must load the public checkpoint and complete one `K=4` cache-backed forward; it may not claim closed-loop success.
- After the user allocates four A800s, run `gate` before `formal`; only merge a formal evaluation after exactly 500 requested episode records exist.
