# Cosmos Universe Mixed-Step Policies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train three new, independent, target-free Cosmos full-distillation policies on eight GPUs: a mixed-budget Universe policy and S2/S1-specialized policies. Each must start from the common compatible Stage-1 `online_student`, run for 5,000 optimizer steps, be reproducible, and never modify the existing completed S4 run.

**Architecture:** Keep the legacy progressive S4 runner unchanged. Add a small policy-spec module that owns the fixed rollout-pair distributions, make the scheduled standalone `cosmos_latent_full` OPD update draw one rank-synchronized weighted pair and record a histogram, and add a dedicated eight-GPU mixed-policy runner. The primary Cdiff update has no rollout-pair input and remains unchanged. The runner creates an isolated result root, resets optimizer/global step from the Stage-1 online student, evaluates every checkpoint at S1/S2/S4 inference budgets, and is driven through a serial tmux launcher after an eight-GPU smoke gate.

**Tech Stack:** Python 3.10, PyTorch distributed/FSDP, existing `distillation_flowmap` trainer/config/evaluator, `pytest`, `torchrun`, `tmux`.

## Global Constraints

- Common initialization checkpoint (online student only):
  `/root/nas/junjie/jj/Any_WAM/distillation_flowmap/output_libero_cosmos_policy_stage1_cosmos_latent_cdiff_8gpu_20260706_cosmos_latent_s1s2_8gpu/checkpoints/step_5000`.
- Do not use raw Stage-1 `target_student`; its action-only 48-channel architecture is incompatible with full Cosmos latent distillation. New policies remain target-free and deploy `online_student`.
- New policies and fixed distributions:
  - `universe`: `(teacher, student) = (4,1),(4,2),(8,4)` with weights `0.50,0.30,0.20`.
  - `s2`: the same pairs with weights `0.20,0.60,0.20`.
  - `s1`: the same pairs with weights `0.70,0.20,0.10`.
  - Existing `s4` remains untouched and is not retrained.
- Every new policy runs independently for 5,000 optimizer steps. No policy is a predecessor of another; no curriculum or stage hand-off is introduced.
- Use all eight H100s for the single active policy. Policies run serially: Universe, then S2, then S1.
- Preserve the existing uncommitted FSDP optimizer-state and progressive-runner edits; do not reset, clean, overwrite, or write output under the legacy S4 root.
- A 5–10 step real eight-GPU Universe preflight must pass checkpoint save/reload and basic eval planning before any 5,000-step tmux job starts.
- The full Cosmos objective remains enabled: joint video/action endpoint, teacher action anchor, DanceOPD same-state velocity, and OPD auxiliary. The weighted mix controls only scheduled standalone `cosmos_latent_full` OPD updates: within such an update it jointly controls the endpoint student integration length and endpoint teacher rollout length. It must not change the primary Cdiff update or DanceOPD's fixed 16-step same-state velocity rollout, and it must log sampled-mode counts.
- Do not claim environment rollout success without a separately runnable simulator/robot evaluation; evaluator output is a proxy until then.

---

### Task 1: Add a testable weighted, distributed mixed-step policy selector

**Files:**
- Create: `distillation_flowmap/cosmos_mixed_step_policy.py`
- Modify: `distillation_flowmap/flowmap_step.py`
- Create or modify: `distillation_flowmap/tests/test_cosmos_mixed_step_policy.py`
- Modify: `distillation_flowmap/tests/test_cosmos_latent_checkpoint_config.py`

- [ ] **Step 1: Write failing unit tests for immutable policy specifications.**

  Cover all three policy names, exact pair ordering and normalized probabilities, rejection of unknown policy names, and an explicit legacy S4 specification only if needed for shared evaluator code. Keep the policy module free of heavyweight model imports.

  Run:
  ```bash
  /root/nas/junjie/conda_envs/any_wam/bin/python -m pytest -q distillation_flowmap/tests/test_cosmos_mixed_step_policy.py
  ```
  Expected: FAIL because the policy module and selector do not exist yet.

- [ ] **Step 2: Implement the smallest pure policy-spec/selector API.**

  Define a frozen policy spec with `name`, `rollout_step_pairs`, and `weights`. Validate matching lengths, positive finite weights, and normalized probabilities. The sampling helper must accept an injected generator or deterministic seed for unit tests.

- [ ] **Step 3: Write failing tests for rank-synchronized weighted selection and accounting.**

  Mock a distributed broadcast boundary so nonzero ranks consume rank-zero’s selected index. Assert that one selected pair is carried through both endpoint branches of the same scheduled standalone full-OPD update rather than independently re-sampled. Assert histogram updates by pair label and that DanceOPD retains its fixed rollout length.

  Run the focused test again and confirm red before modifying trainer code.

- [ ] **Step 4: Wire the selector into the Cosmos full-loss path.**

  Extend configuration with `cosmos_mixed_step_policy`, `opd_rollout_step_pair_weights`, and a per-update selected-pair context. In `flowmap_step.py`, replace only the existing uniform pair draw in `_cosmos_latent_full_opd_aux_transition_step`: rank zero selects, broadcasts the index, and the endpoint's student Euler integration and teacher velocity rollout reuse it. Preserve legacy single-pair behavior when no mixed policy is requested. Do not route this pair into `_cosmos_policy_train_step` or `_cosmos_danceopd_velocity_loss`. Log a rank-zero JSONL/progress metric and aggregate histogram with policy name, pairs, weights, seed, and current global step.

- [ ] **Step 5: Run focused tests.**

  ```bash
  /root/nas/junjie/conda_envs/any_wam/bin/python -m pytest -q \
    distillation_flowmap/tests/test_cosmos_mixed_step_policy.py \
    distillation_flowmap/tests/test_cosmos_latent_checkpoint_config.py
  ```
  Expected: PASS.

### Task 2: Add an isolated eight-GPU independent-policy runner

**Files:**
- Create: `distillation_flowmap/run_cosmos_mixed_step_policy.py`
- Create or modify: `distillation_flowmap/tests/test_cosmos_mixed_step_runner.py`
- Modify only as needed: `distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py`

- [ ] **Step 1: Write failing runner-plan tests.**

  Test that each `universe`, `s2`, and `s1` plan:
  - starts at global step zero from the common Stage-1 `online_student` checkpoint;
  - has `max_train_steps=5000`, `reset_optimizer_state=1`, and no target resume;
  - launches `torchrun --nproc_per_node=8` with `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` and a matching worker-device list;
  - writes under an isolated policy-specific root;
  - rejects the legacy S4 root and rejects a nonempty checkpoint directory without explicit resume;
  - builds S1/S2/S4 evaluation plans for every selected checkpoint.

  Run:
  ```bash
  /root/nas/junjie/conda_envs/any_wam/bin/python -m pytest -q distillation_flowmap/tests/test_cosmos_mixed_step_runner.py
  ```
  Expected: FAIL because the dedicated runner does not exist.

- [ ] **Step 2: Implement pure plan-building and collision-guard functions.**

  Keep these functions independently testable: `build_policy_train_plan`, `validate_policy_root`, `build_multibudget_eval_plans`, and `build_torchrun_command`. Use explicit CLI flags for root, source checkpoint, device list, world size, save interval, max steps, and master port. Never default to the legacy S4 output root.

- [ ] **Step 3: Implement execution mode.**

  Generate a `policy_manifest.json` before launch that captures git HEAD, dirty diff hash, source checkpoint, policy spec, seed, all objective settings, visible devices, torchrun command, and timestamps. Copy/link only immutable protocol metadata into the new root; use existing teacher/cache artifacts read-only. Pass target-free checkpoint settings and exact mix settings through the environment/config. Save every 250 optimizer steps.

- [ ] **Step 4: Add all-budget evaluator orchestration.**

  Reuse the existing evaluator only through explicit child plans/commands. For each selected checkpoint build three isolated output destinations, one each for inference budgets `S1`, `S2`, and `S4`; record a combined `selection_proxy.json`. Build/read teacher caches separately for `t4` (S1/S2) and `t8` (S4), since cache payloads are teacher-step-specific. This is proxy evaluation, not a rollout-success claim.

- [ ] **Step 5: Run runner tests and regression tests.**

  ```bash
  /root/nas/junjie/conda_envs/any_wam/bin/python -m pytest -q \
    distillation_flowmap/tests/test_cosmos_mixed_step_runner.py \
    distillation_flowmap/tests/test_cosmos_progressive_runner.py \
    distillation_flowmap/tests/test_cosmos_latent_checkpoint_config.py
  git diff --check
  ```
  Expected: PASS and no whitespace errors.

### Task 3: Add tmux-safe preflight and serial training launcher

**Files:**
- Create: `distillation_flowmap/launch_cosmos_mixed_step_8gpu.sh`
- Create or modify: `distillation_flowmap/tests/test_cosmos_mixed_step_launcher.py`
- Create: `docs/cosmos_mixed_step_8gpu_runbook.md`

- [ ] **Step 1: Write failing launcher tests.**

  Verify generated commands use a new tmux session, execute Universe preflight before full training, and serially order `universe`, `s2`, `s1`. The script must refuse to run a later policy while an earlier policy has no `TRAINING_COMPLETE` marker. It must never schedule existing S4.

- [ ] **Step 2: Implement the launcher and runbook.**

  Supported commands:
  - `preflight`: run Universe for 9 steps in a unique preflight root with 8 GPUs and force the `S1,S2,S4` selector sequence three times so every path is exercised;
  - `start universe|s2|s1`: start one policy in a named tmux session only after preflight/previous-marker gates;
  - `status`: show sessions and last log lines;
  - `eval <policy> <checkpoint>`: invoke the three-budget proxy evaluation plan.

  Use `set -euo pipefail`, unique session names and logs, and write explicit success/failure markers. Do not remove old data automatically.

- [ ] **Step 3: Run all local/static tests.**

  ```bash
  /root/nas/junjie/conda_envs/any_wam/bin/python -m pytest -q \
    distillation_flowmap/tests/test_cosmos_mixed_step_policy.py \
    distillation_flowmap/tests/test_cosmos_mixed_step_runner.py \
    distillation_flowmap/tests/test_cosmos_mixed_step_launcher.py \
    distillation_flowmap/tests/test_cosmos_progressive_runner.py \
    distillation_flowmap/tests/test_cosmos_latent_checkpoint_config.py
  git diff --check
  ```
  Expected: PASS.

### Task 4: Verify the real eight-GPU preflight, then launch serially

**Files/outputs:**
- New output root only, e.g. `/root/nas/junjie/jj/Any_WAM_cosmos_universe_mixed_8gpu_20260716/`
- New tmux sessions/logs only.

- [ ] **Step 1: Record the baseline and check capacity.**

  Capture `git status --short`, `git rev-parse HEAD`, `nvidia-smi`, free disk, and existing tmux sessions. Confirm the target eight GPUs are idle without killing unrelated jobs.

- [ ] **Step 2: Run the 8-step Universe preflight in tmux.**

  Require all eight FSDP ranks and corresponding Cosmos workers to initialize, all three forced mixed-step selections to be logged, a target-free online checkpoint save, successful reload/continuation planning, and all three evaluator child plans to be generated. Inspect log tail and `nvidia-smi` during execution.

- [ ] **Step 3: Gate full training on evidence.**

  If preflight fails, stop at the failure marker, preserve logs, and diagnose before full training. If it passes, start `universe` (5,000 steps) in its own tmux session. Do not start S2/S1 until the prior policy has `TRAINING_COMPLETE` and the selected checkpoint’s proxy eval is available.

- [ ] **Step 4: Monitor and hand off concrete state.**

  Report active tmux session, output root, latest checkpoint, sampled histogram, GPU utilization, and which serial gate is next. State clearly that proxy scores are not physical rollout success.

## Completion Criteria

- Tests above pass, `git diff --check` is clean, and the new code never selects/overwrites the legacy S4 root.
- A real 8-GPU preflight has evidence of target-free save/reload, weighted mixed-pair logging, and three-budget eval planning.
- Universe’s 5,000-step tmux training has been started only after the preflight gate; S2/S1 are queued behind explicit completion gates.
- Existing checkpoints, `ckpt` directories, experiments, and unrelated dirty changes are preserved.
