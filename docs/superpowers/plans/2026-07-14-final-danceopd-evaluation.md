# Final DanceOPD Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, protocol-controlled final 12-task RobotWin ablation package for the four LingBot-VA DanceOPD variants and two structural controls.

**Architecture:** Training remains in the frozen single-GPU jobs. A dedicated final evaluation runner builds one held-out teacher cache with the exact fixed manifest and evaluation pairs, then replays every compatible checkpoint against it. A dedicated report writer aggregates task means and paired task bootstrap intervals without treating twelve tasks as training-seed replicates. Native WanVAE video assets use a one-pair, one-held-out-record-per-task manifest and are kept separate from numerical cache replay.

**Tech Stack:** Bash, Python 3.10, PyTorch, LingBot-VA WanVAE, lpips 0.1.4, pytest.

## Global Constraints

- Teacher is LingBot-VA only; do not route LingBot through Cosmos APIs.
- Main variants are final_w_o_opd, final_endpoint_only_danceopd, final_danceopd_velocity_only, and final_stepwam_danceopd.
- Every main variant starts from the shared frozen Stage1 step_5000 checkpoint and uses seed 0.
- Held-out data is exactly twelve tasks by ten records; training data is exactly forty records per task.
- Primary student budget is K=4. Curves use K=1/2/4 and equal-NFE teacher T=1/2/4; T=8 is reference only.
- Final numerical video metrics require CUDA WanVAE decode and pretrained AlexNet LPIPS. CPU decode is not accepted.
- Do not put local-adjacent-only or action-only in the strict four-way OPD table.
- Preserve active training jobs and do not overwrite any checkpoint.
- All output records must include config, git hash, stage checkpoints, manifests, and teacher-cache SHA256.

---

### Task 1: Make multi-task native video asset export selectable

**Files:**
- Modify: `distillation_flowmap/rollout_eval_video_stage2.py`
- Modify: `distillation_flowmap/tests/test_robotwin_video_eval_memory.py`

**Interfaces:**
- Consumes: `--video-max-pairs`, `--eval-manifest`, and a one-pair JSON file.
- Produces: one native WanVAE video/contact sheet for each selected manifest record until `video-max-pairs` is reached.

- [x] **Step 1: Write the failing regression test**

```python
def test_video_eval_does_not_limit_assets_to_first_eval_record():
    source = _video_eval_source()
    assert "and batch_idx == 0" not in source
```

- [x] **Step 2: Run the focused test and verify it fails**

Run: `python -m pytest -q distillation_flowmap/tests/test_robotwin_video_eval_memory.py::test_video_eval_does_not_limit_assets_to_first_eval_record`

Expected: FAIL because both native asset-save guards contain `batch_idx == 0`.

- [x] **Step 3: Remove only the first-record guards**

Use `rank == 0`, non-null `video_dir`, and `saved_video_pairs < args.video_max_pairs` in both save decisions. Leave the counter increment after each contact sheet unchanged.

- [x] **Step 4: Run focused and related tests**

Run: `python -m pytest -q distillation_flowmap/tests/test_robotwin_video_eval_memory.py distillation_flowmap/tests/test_robotwin_smoke_video_logging.py`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add distillation_flowmap/rollout_eval_video_stage2.py distillation_flowmap/tests/test_robotwin_video_eval_memory.py
git commit -m "fix: export RobotWin assets across manifest records"
```

### Task 2: Add the strict final cache-replay runner

**Files:**
- Create: `distillation_flowmap/ablation/run_final_danceopd_eval.sh`
- Create: `distillation_flowmap/tests/test_robotwin_final_danceopd_eval.py`
- Modify: `docs/experiments/2026-07-14-robotwin-final-stepwam-ablation-protocol.md`

**Interfaces:**
- Consumes: the frozen training root, 12-task manifests, stage2 step_5000 checkpoints, GPUs 6 and 7.
- Produces: one trajectory teacher cache, cache SHA256 provenance, held-out metric JSON per run, optional ten-per-task train metrics, and selected baseline/full assets.

- [x] **Step 1: Write a failing dry-run protocol test**

```python
result = subprocess.run(["bash", script, "--dry-run"], text=True, capture_output=True, check=True)
assert "--cache-teacher-trajectories" in result.stdout
assert "--trajectory-teacher-steps 1 2 4" in result.stdout
assert "--rollout-drift" in result.stdout
assert "--same-state-velocity" in result.stdout
assert "--decoded-video-device cuda" in result.stdout
assert "--decoded-video-lpips" in result.stdout
```

- [x] **Step 2: Run the focused test and verify it fails**

Run: `python -m pytest -q distillation_flowmap/tests/test_robotwin_final_danceopd_eval.py`

Expected: FAIL because the dedicated runner does not exist.

- [x] **Step 3: Implement only the frozen protocol**

The runner must:
- build `teacher_cache_heldout_equal_nfe_trajectories.pt` from one named shared Stage1 checkpoint;
- cache T=1/2/4 trajectory states and T=8 endpoints;
- launch cache-replay jobs on GPU6 and GPU7 for the four main variants;
- pass `--rollout-drift --same-state-velocity --decoded-video-metrics --decoded-video-device cuda --decoded-video-lpips`;
- generate a one-pair `1000->0`, twelve-record asset manifest and export teacher/baseline/full videos with `--condition-first-frame`;
- keep structural controls in a separate execution group and mark action-only video metrics unavailable rather than fabricating them;
- record cache SHA256 and `git rev-parse HEAD` in a JSON provenance file.

- [x] **Step 4: Run dry-run and shell syntax verification**

Run: `bash -n distillation_flowmap/ablation/run_final_danceopd_eval.sh && python -m pytest -q distillation_flowmap/tests/test_robotwin_final_danceopd_eval.py`

Expected: PASS and dry-run prints no training command.

- [ ] **Step 5: Run one-record GPU smoke after a final checkpoint exists**

Run the runner with its smoke option and verify an output JSON contains finite K=1/2/4 equal-NFE metrics including decoded LPIPS, same-state velocity, and rollout drift.

- [x] **Step 6: Commit**

```bash
git add distillation_flowmap/ablation/run_final_danceopd_eval.sh distillation_flowmap/tests/test_robotwin_final_danceopd_eval.py docs/experiments/2026-07-14-robotwin-final-stepwam-ablation-protocol.md
git commit -m "feat: add final DanceOPD evaluation runner"
```

### Task 3: Add task-level final report and paired bootstrap

**Files:**
- Create: `distillation_flowmap/ablation/summarize_final_danceopd_ablation.py`
- Create: `distillation_flowmap/tests/test_summarize_final_danceopd_ablation.py`

**Interfaces:**
- Consumes: per-run `offline_rollout.json` with `per_task`, optional train replay JSON, run manifests, and runner provenance JSON.
- Produces: CSV/JSON/Markdown main table, per-task deltas, K=1/2/4 curves, structural-control table, and bootstrap intervals.

- [ ] **Step 1: Write a failing synthetic-root test**

```python
summary = build_final_summary(tmp_path, baseline_variant="final_w_o_opd", bootstrap_samples=1000)
assert summary["main"]["final_stepwam_danceopd"]["metric"]["macro_delta"] == pytest.approx(expected)
assert summary["main"]["final_stepwam_danceopd"]["metric"]["bootstrap_unit"] == "tasks"
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `python -m pytest -q distillation_flowmap/tests/test_summarize_final_danceopd_ablation.py`

Expected: FAIL because the final report module does not exist.

- [ ] **Step 3: Implement the minimal report contract**

Read only the equal-NFE metric keys for K=1/2/4. Compute every macro mean from one value per task. Compute paired deltas against final_w_o_opd task by task and bootstrap resample those twelve deltas with a fixed RNG seed. Write a statement that the CI covers task coverage, not training-seed significance. Write controls to a separate table.

- [ ] **Step 4: Run focused and existing summary tests**

Run: `python -m pytest -q distillation_flowmap/tests/test_summarize_final_danceopd_ablation.py distillation_flowmap/tests/test_robotwin_ablation_summary.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add distillation_flowmap/ablation/summarize_final_danceopd_ablation.py distillation_flowmap/tests/test_summarize_final_danceopd_ablation.py
git commit -m "feat: summarize final DanceOPD ablation"
```

### Task 4: Execute and validate the completed package

**Files:**
- Read: `distillation_flowmap/output_robotwin_stepwam_ablation/protocol_final12_danceopd_i1_single_seed_v1/`
- Write: final result files under that run root only.

- [ ] **Step 1: Wait for every requested stage2 step_5000 checkpoint**

Verify each checkpoint path exists and each training log reached step 5000 with no traceback.

- [ ] **Step 2: Build the shared teacher cache once**

Run the dedicated runner cache stage and record its SHA256 before starting variant replay.

- [ ] **Step 3: Run numerical held-out replay**

Run four main variants first, then structural controls where valid. Inspect every exit code and non-finite metric check before writing the report.

- [ ] **Step 4: Export qualitative assets**

Use the one-record-per-task asset manifest for teacher, final_w_o_opd, and final_stepwam_danceopd. Verify every selected task has a contact sheet and videos for K=1/2/4 and T=1/2/4/8.

- [ ] **Step 5: Run existing RobotWin closed-loop evaluator**

Use 50 episodes per task for the four main variants and write SR, teacher retention, latency/chunk, Hz, NFE, and speedup to each run's metrics directory. Do not replace this with an offline proxy.

- [ ] **Step 6: Build final report and audit provenance**

Verify exactly twelve task rows per comparable metric, source cache hash equality across main variants, task-macro aggregation, finite values, and separate treatment of structural controls.
