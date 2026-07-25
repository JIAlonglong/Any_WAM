# LIBERO Paper Evaluation Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the LingBot-VA/LIBERO Figure 4 export and formal evaluation artifacts match the approved paper-facing contract without changing Teacher sampling or shared RNG behavior.

**Architecture:** Keep training diagnostics backward-compatible and change only the paper export mapping. Add a small latency-recording module used by the inference server and a strict merger that attaches p50 latency to each closed-loop summary. Extend the formal orchestrator with a manifest that distinguishes Stage-I-only from an optional, explicitly supplied naive baseline.

**Tech Stack:** Python, PyTorch, Bash, JSON/JSONL, pytest.

## Global Constraints

- Do not change the matched-step Teacher sampler.
- Do not add shared model-prior seed enforcement.
- Do not evaluate the default 20/50-step Teacher baseline.
- Do not relabel Stage-I-only as naive composition.
- Do not modify training losses or checkpoint state.

---

### Task 1: Deployment-aligned Figure 4 mapping

**Files:**
- Modify: `distillation_flowmap/mechanism_checkpoint_sweep.py`
- Modify: `distillation_flowmap/tests/test_mechanism_checkpoint_sweep.py`

**Interfaces:**
- Consumes: reduced mechanism metric names written by online/post-hoc diagnostics.
- Produces: `METRIC_SOURCES` mapping used to write `e_student`, `e_video`, and `e_joint`.

- [ ] Add a failing test asserting `e_student` and `e_video` select the generated-history metric names and derived metrics use those values.
- [ ] Run the targeted test and confirm it fails on the legacy mapping.
- [ ] Change only the paper-facing metric mapping; keep legacy raw metrics available.
- [ ] Add manifest metadata that identifies the Figure 4 action context as deployment generated-history.
- [ ] Run the targeted test and existing mechanism diagnostics tests.
- [ ] Commit the task.

### Task 2: Auditable sampler latency

**Files:**
- Create: `evaluation/libero/sampler_latency.py`
- Modify: `wan_va/wan_va_server.py`
- Modify: `evaluation/libero/client.py`
- Modify: `evaluation/libero/run_eval_new.sh`
- Create: `evaluation/libero/tests/test_sampler_latency.py`

**Interfaces:**
- Consumes: `eval_metadata` sent on episode reset plus one complete `_infer` call.
- Produces: append-only `sampler_latency.jsonl` records containing suite, task, episode, model, video/action budgets, call index, and `elapsed_ms`.

- [ ] Add failing unit tests for record validation, append behavior, and p50 calculation.
- [ ] Add a failing source/contract test requiring client reset metadata and server timing around `_infer` only.
- [ ] Run the tests and confirm the missing interfaces fail.
- [ ] Implement atomic-line JSONL append and validated percentile helpers.
- [ ] Pass suite/task/episode/model metadata from client reset to server.
- [ ] Time `_infer` with `time.perf_counter()` and append one record after successful calls.
- [ ] Wire model name and latency output path through the evaluation launcher.
- [ ] Run the latency and client/server contract tests.
- [ ] Commit the task.

### Task 3: Merge latency with closed-loop success

**Files:**
- Modify: `evaluation/libero/merge_lingbotva_4suite_results.py`
- Modify: `evaluation/libero/run_lingbotva_4suite_124_eval_8gpu.sh`
- Modify: `evaluation/libero/tests/test_merge_lingbotva_4suite_results.py`
- Modify: `evaluation/libero/tests/test_run_lingbotva_4suite_124_eval_8gpu.py`

**Interfaces:**
- Consumes: worker success JSON files and worker `sampler_latency.jsonl` files.
- Produces: `summary.json` with validated latency record count and p50 sampler latency.

- [ ] Add failing tests for valid p50 aggregation and rejection of malformed/mismatched latency provenance.
- [ ] Add a failing launcher test requiring worker-specific latency paths and model/budget metadata.
- [ ] Run the tests and confirm expected failures.
- [ ] Extend the merger and launcher minimally.
- [ ] Run merger and launcher tests.
- [ ] Commit the task.

### Task 4: Honest naive-baseline contract and formal manifest

**Files:**
- Modify: `distillation_flowmap/run_libero_video_opd_train_eval_8gpu.sh`
- Create: `evaluation/libero/write_lingbotva_eval_contract.py`
- Create: `evaluation/libero/tests/test_lingbotva_eval_contract.py`
- Modify: `distillation_flowmap/tests/test_run_libero_video_opd_train_eval_8gpu.py`

**Interfaces:**
- Consumes: evaluated model paths plus optional `NAIVE_CKPT` or executable `NAIVE_RUNNER`.
- Produces: `evaluation_contract.json` with `stage1_only` and `naive_composition` as distinct entries; naive status is `missing` unless explicitly supplied.

- [ ] Add failing tests that reject Stage-I-only being labelled naive and require missing naive status by default.
- [ ] Add failing dry-run tests for optional explicit naive evaluation.
- [ ] Run tests and confirm expected failures.
- [ ] Implement contract writer and optional naive orchestration without changing Teacher behavior.
- [ ] Run formal orchestration contract tests.
- [ ] Commit the task.

### Task 5: Verification

**Files:**
- Verify all changed files.

- [ ] Run all targeted pytest modules.
- [ ] Run Python compilation on changed Python files.
- [ ] Run `bash -n` on changed shell scripts.
- [ ] Run `git diff --check` and confirm a clean worktree.
- [ ] Perform `CHECK_ONLY=1` formal evaluation dry-run and inspect model, suite, budget, latency, and naive status output.
