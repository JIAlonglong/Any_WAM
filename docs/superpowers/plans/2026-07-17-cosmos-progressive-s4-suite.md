# Cosmos Progressive S4 Full-Evaluation Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-command, no-overwrite serial suite that produces both paper offline metrics and formal closed-loop Cosmos Progressive S4 results.

**Architecture:** A small Bash orchestrator owns a fresh `SUITE_ROOT`, writes durable per-phase status events, and invokes the established protocol/cache/paper/formal entry points with fixed child paths. It owns the sequential 0/1 offline allocation while delegating the concurrent 0/1 and 2/3 formal rollout and strict result merge to the already tested formal launcher.

**Tech Stack:** Bash (`set -euo pipefail`), existing Python CLI modules, pytest subprocess tests.

## Global Constraints

- Keep all implementation in `/kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/cosmos-progressive-s4-eval`; do not modify the user's main checkout.
- `SUITE_ROOT` must not exist before a live run; never delete, resume, or overwrite a prior suite.
- Use fixed 3 selection and 5 test records per LIBERO task, pair `1000,0`, student K=4, and teacher K=8.
- Cache and offline metrics use student GPU 0 plus official Cosmos worker GPU 1; formal delegates the existing fixed 0/1 + 2/3 allocation.
- `dry-run` creates no files/directories and executes no child command.
- The launcher must use only existing public S4 / Cosmos paths supplied through environment variables.

---

### Task 1: Define suite launcher contract through failing subprocess tests

**Files:**
- Create: `evaluation/libero/tests/test_run_cosmos_progressive_s4_suite.py`
- Test: `evaluation/libero/tests/test_run_cosmos_progressive_s4_suite.py`

**Interfaces:**
- Consumes: `bash evaluation/libero/run_cosmos_progressive_s4_suite.sh run|dry-run`.
- Produces: tests that describe the expected environment validation, command plan, and no-overwrite behavior.

- [ ] **Step 1: Write the failing test**

```python
def test_suite_dry_run_plans_offline_and_formal_without_child_execution(tmp_path):
    env = _suite_env(tmp_path)
    result = run_suite("dry-run", env=env)
    _assert_success(result)
    assert "PHASE_PLAN=protocol" in result.stdout
    assert "PHASE_PLAN=teacher_cache" in result.stdout
    assert "PHASE_PLAN=offline_paper" in result.stdout
    assert "PHASE_PLAN=closed_loop_formal" in result.stdout
    assert "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES=1" in result.stdout
    assert "--student-steps 4" in result.stdout
    assert "--teacher-steps 8" in result.stdout
    assert not Path(env["SUITE_ROOT"]).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m pytest evaluation/libero/tests/test_run_cosmos_progressive_s4_suite.py -q`

Expected: FAIL because `run_cosmos_progressive_s4_suite.sh` does not exist.

- [ ] **Step 3: Add no-overwrite failure test**

```python
def test_suite_run_rejects_existing_root_before_child_execution(tmp_path):
    env = _suite_env(tmp_path)
    Path(env["SUITE_ROOT"]).mkdir()
    result = run_suite("run", env=env)
    assert result.returncode == 2
    assert "SUITE_ROOT already exists" in result.stderr
    assert not Path(env["SENTINEL_MARKER"]).exists()
```

- [ ] **Step 4: Run tests to verify the intended failures**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m pytest evaluation/libero/tests/test_run_cosmos_progressive_s4_suite.py -q`

Expected: FAIL only because the launcher is absent.

- [ ] **Step 5: Commit**

```bash
git add evaluation/libero/tests/test_run_cosmos_progressive_s4_suite.py
git commit -m "test: define Cosmos progressive S4 suite launcher"
```

### Task 2: Implement the serial no-overwrite suite launcher

**Files:**
- Create: `evaluation/libero/run_cosmos_progressive_s4_suite.sh`
- Modify: `evaluation/libero/tests/test_run_cosmos_progressive_s4_suite.py`

**Interfaces:**
- Consumes: required `SUITE_ROOT`, `S4_CKPT_ROOT`, `S4_DATASET_PATH`, `S4_EMPTY_EMBEDDING`, `COSMOS_POLICY_PATH`, `COSMOS_POLICY_PYTHON`, and `COSMOS_PREDICT2_REPO`; optional `S4_PROMPT_TABLE`, `S4_INITIAL_STATES_JSON`, `PYTHON_BIN`, and `S4_CONFIG`.
- Produces: protocol/cache/offline/formal child outputs beneath `SUITE_ROOT` and `suite_status.jsonl` for live runs.

- [ ] **Step 1: Write the minimal launcher after the failing tests are observed**

```bash
ensure_new_suite_root
run_phase protocol run_local "$PYTHON_BIN" -m distillation_flowmap.prepare_cosmos_progressive_protocol ...
run_phase teacher_cache run_pair 0 1 "$PYTHON_BIN" -m distillation_flowmap.build_cosmos_progressive_teacher_cache ...
run_phase offline_paper run_pair 0 1 "$PYTHON_BIN" -m distillation_flowmap.eval_cosmos_progressive_s4_paper ...
run_phase closed_loop_formal run_formal
```

Each phase must print a shell-quoted `COMMAND=`, write `started` then
`completed` status events only in live mode, and stop on the first failing
child. `dry-run` must print every command without invoking it.

- [ ] **Step 2: Run the new test file**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m pytest evaluation/libero/tests/test_run_cosmos_progressive_s4_suite.py -q`

Expected: PASS.

- [ ] **Step 3: Run Bash syntax validation**

Run: `bash -n evaluation/libero/run_cosmos_progressive_s4_suite.sh`

Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add evaluation/libero/run_cosmos_progressive_s4_suite.sh evaluation/libero/tests/test_run_cosmos_progressive_s4_suite.py
git commit -m "feat: add Cosmos progressive S4 full evaluation suite"
```

### Task 3: Document the runnable one-command entry point

**Files:**
- Create: `docs/experiments/2026-07-17-cosmos-progressive-s4-full-suite.md`
- Test: `evaluation/libero/tests/test_run_cosmos_progressive_s4_suite.py`

**Interfaces:**
- Consumes: the Task 2 launcher and fixed result layout.
- Produces: a compatible-node command that runs both result families under one new suite root.

- [ ] **Step 1: Add the exact invocation**

```bash
SUITE_ROOT="<new result root>" \
S4_CKPT_ROOT="<released S4 transformer>" \
S4_DATASET_PATH="$DATASET" \
S4_EMPTY_EMBEDDING="$DATASET/empty_emb.pt" \
COSMOS_POLICY_PATH="<Cosmos Policy root>" \
COSMOS_POLICY_PYTHON="<official Cosmos cu128 Python>" \
COSMOS_PREDICT2_REPO="<official Cosmos repository>" \
bash evaluation/libero/run_cosmos_progressive_s4_suite.sh run
```

Document the serial GPU ownership and the four output locations. Include a
`dry-run` command using the same inputs and an absent suite root.

- [ ] **Step 2: Run full targeted regression suite**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m pytest evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py evaluation/libero/tests/test_run_cosmos_progressive_s4_suite.py evaluation/libero/tests/test_cosmos_progressive_s4_service.py distillation_flowmap/tests/test_cosmos_progressive_paper_metrics.py distillation_flowmap/tests/test_cosmos_progressive_s4_paper_eval.py -q`

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add docs/experiments/2026-07-17-cosmos-progressive-s4-full-suite.md
git commit -m "docs: add Cosmos progressive S4 full evaluation command"
```

## Self-Review

- Spec coverage: Tasks 1-2 cover the fresh-root safety, fixed offline protocol/cache metrics, serial GPU allocations, formal delegation, dry-run, and status events. Task 3 covers the user-facing command and output layout.
- Placeholder scan: no implementation or verification step contains a deferred behavior; all commands, fixed paths, and metric-step settings are stated.
- Type consistency: the Bash interface and child paths use the same `SUITE_ROOT`, `protocol`, `teacher_cache/test`, `offline_paper`, and `closed_loop_formal` names across all tasks.
