# Final review fix report — Cosmos Stage-2 student evaluation chain

Date: 2026-07-27

Worktree:
`/kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/cosmos-stage2-progressive-base`

Starting HEAD: `8a8010273d56edbfb648e967d953faf731fd6825`

Implementation commit:
`dd60eee` — `fix: close Cosmos Stage-2 chain runtime contracts`

## Finding-by-finding resolution

### Critical 1 — clean-shell production command

Resolved.

- The wrapper runs the real Stage-1 validator first and derives
  `WAN_STUDENT_BASE_MODEL_PATH` from the validated Stage-1 hybrid metadata.
  A conflicting ambient Wan path is rejected.
- `DATASET_PATH` now defaults to the reviewed shared dataset:
  `/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot`.
- The default Cosmos repo is the clean reviewed compatibility worktree
  `/kpfs-intern/jialongliu/projects/cosmos-predict2.5-formal-compat-441b897`,
  sealed at commit `1eb8457072b4a1adfe1f83c3076e4aa5452cbab2`.
- The production command was run under `env -i`; it needed no hidden
  Wan/dataset/worker variables and wrote no run output.

Regression:
`test_clean_shell_derives_wan_and_uses_shared_dataset_default`.

### Critical 2 — complete audited Cosmos worker runtime

Resolved.

Before any output write, the wrapper resolves and validates:

- clean Cosmos Git repo and exact audited commit;
- worker interpreter and its containing environment root;
- worker site-packages;
- complete Cosmos Python path;
- every CUDA library path component;
- worker `LD_LIBRARY_PATH`.

The exact resolved values, including the repo commit and constructed
`LD_LIBRARY_PATH`, are explicitly present in both the Stage-2 and evaluation
child environments. No child-to-parent environment propagation is assumed.

Regression:
`test_complete_worker_runtime_is_resolved_once_and_passed_to_both_children`.

### Important 1 — real preflights before writes and in read-only mode

Resolved.

- Real `validate_stage1_parent(..., expected_step=3000)` plus validated hybrid
  roots runs before any output action.
- Provenance inputs are validated through the real lock preparer's
  `--dry-run` before a live write and during check-only.
- The canonical all-40 prompt contract is validated through `--dry-run`, or
  through `--validate-only` when the table already exists, before live output
  and during check-only.
- Eval-only runs call `resolve_cosmos_inference_checkpoint` and
  `bind_stage2_inference_runtime` before prompt or matrix output.
- All/stage2 runs call the same Stage-2 inference preflight after training
  returns and before prompt/evaluation output.

Regressions:

- `test_real_stage1_lineage_is_validated_before_any_output_or_child`
- `test_check_only_validates_lock_and_prompt_inputs_without_writes`
- `test_eval_preflights_real_stage2_inference_before_prompt_or_matrix_write`

### Important 2 — process groups and four-shard fail-fast

Resolved.

- Wrapper children run in independent sessions via `setsid --wait`; INT/TERM
  target the whole active process group.
- Each formal shard runs in its own session.
- The formal launcher uses `wait -n -p`, preserves the first failing shard's
  exact status, sends TERM to all remaining shard process groups, waits for
  cleanup, and skips merge.
- INT/TERM handlers also clean active shard process groups.

Regressions:

- `test_wrapper_forwards_signals_to_child_process_group_with_standard_exit_code`
- `test_formal_fails_fast_and_kills_other_shard_process_groups`

The fail-fast test observes the original shard code `17`, proves other shards
do not reach seed 49, and proves a spawned shard grandchild cannot leak.

### Important 3 — GPU ordinal contract

Resolved with the minimum-risk option: the wrapper accepts exactly
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`. This closes the contract with lower
evaluation code that assigns logical ordinals 0 through 7.

Regression:
`test_wrapper_restricts_gpu_contract_to_exact_ordinals_zero_through_seven`.

### Important 4 — symlinked run-root components

Resolved.

Before canonicalizing the output path or performing any write, the wrapper
passes the raw Stage-2 output path through
`validate_stage2_path_isolation`, whose component-wise `lstat` validation
rejects symlinked `RUN_ROOT` and any existing symlink component.

Regression:
`test_wrapper_rejects_symlinked_run_root_component_before_any_write`.

### Minor — signal-specific exit codes

Resolved. SIGINT exits `130`; SIGTERM exits `143`. Both are covered by the
process-group regression above.

## TDD evidence

### Main final-review repair wave

RED:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_run_cosmos_stage2_student_eval_8gpu.py \
  evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py
```

Observed:

```text
11 failed, 39 passed in 22.92s
```

The failures were the missing Wan derivation/runtime closure, absent real
Stage-1/lock/prompt/Stage-2 preflights, permissive GPU list, symlink
canonicalization, PID-only signal forwarding, wrong TERM code, and
wait-for-all shard behavior.

GREEN, same command:

```text
50 passed in 112.31s
```

### Exact student-role plan evidence

The real production check exposed that `S4_MATRIX_ROLES=stage2_target` was
only embedded inside the quoted evaluation command. A final exact output
contract was repaired with a separate TDD cycle.

RED:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_run_cosmos_stage2_student_eval_8gpu.py::test_check_only_prints_full_plan_without_writes
```

Observed:

```text
1 failed in 6.30s
```

GREEN, same command:

```text
1 passed in 5.91s
```

## Final verification

### Complete prescribed integration set

Fresh command after the last product/test change:

```bash
PYTHONPATH=. /kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_run_cosmos_libero_train_8gpu.py \
  distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py \
  distillation_flowmap/tests/test_cosmos_progressive_env_schema.py \
  distillation_flowmap/tests/test_cosmos_stage1_stage2_eval_pipeline.py \
  distillation_flowmap/tests/test_cosmos_stage2_lineage.py \
  distillation_flowmap/tests/test_run_cosmos_stage2_student_eval_8gpu.py \
  evaluation/libero/tests/test_run_cosmos_progressive_joint_124_eval_8gpu.py \
  evaluation/libero/tests/test_run_cosmos_progressive_s4_eval.py \
  evaluation/libero/tests/test_cosmos_progressive_eval_summary.py
```

Result:

```text
290 passed in 683.36s (0:11:23)
```

This is the existing 281-test integration set plus nine net-new behavior
tests.

### Static checks

```bash
bash -n \
  distillation_flowmap/run_cosmos_libero_train_8gpu.sh \
  distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh \
  distillation_flowmap/run_cosmos_stage2_student_eval_8gpu.sh \
  evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh \
  evaluation/libero/run_cosmos_progressive_s4_eval.sh
git diff --check
```

Result:

```text
BASH_N_EXIT=0
GIT_DIFF_CHECK_EXIT=0
```

### Real production check-only, clean shell, zero write

The wrapper was run with only a minimal process environment:

```bash
env -i \
  PATH=/usr/bin:/bin \
  HOME=/nonexistent \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONNOUSERSITE=1 \
  bash distillation_flowmap/run_cosmos_stage2_student_eval_8gpu.sh \
    --phase check \
    --stage1-root /kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_cosmos_aligned_stage1_stage2_full40_8gpu_20260725/aligned-anchor-field-full40-20260725/stage1 \
    --output-root /kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/final-fix-check-only-output-20260727 \
    --run-tag stage1step3000-final-fix-check
```

Fresh result after the last change:

```text
CHECK_ONLY_EXIT=0
CHECK_ONLY_OUTPUT_EXISTS=no
COSMOS_STAGE1_EXPECTED_STEP=3000
WAN_STUDENT_BASE_MODEL_PATH=/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero
DATASET_PATH=/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot
COSMOS_PREDICT2_REPO=/kpfs-intern/jialongliu/projects/cosmos-predict2.5-formal-compat-441b897
COSMOS_PREDICT2_REPO_COMMIT=1eb8457072b4a1adfe1f83c3076e4aa5452cbab2
COSMOS_POLICY_PYTHON=/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/bin/python3.10
MAX_TRAIN_STEPS=5000
S4_MATRIX_ROLES=stage2_target
PIPELINE_MODE=check-only
S4_FORMAL_NUM_SHARDS=4
S4_FORMAL_GPU_LAYOUT=paired
S4_VIDEO_SEEDS=0
MATRIX_STEP_COUNT=3
MATRIX_SUITE_COUNT=12
```

The candidate output root was absent both before and after. The check ran
real Stage-1 lineage, provenance-input, all-40 prompt, runtime, and full
matrix planning preflights without launching training or rollout.

### Process audit

After verification, a read-only process scan found no matching `torchrun`,
`distillation_flowmap/train.py`, or
`evaluation.libero.rollout_cosmos_progressive_s4` process.

## Final short production command

No Wan, dataset, or worker environment variables are required:

```bash
cd /kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/cosmos-stage2-progressive-base
bash distillation_flowmap/run_cosmos_stage2_student_eval_8gpu.sh \
  --phase all \
  --stage1-root /kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_cosmos_aligned_stage1_stage2_full40_8gpu_20260725/aligned-anchor-field-full40-20260725/stage1 \
  --output-root /kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_cosmos_stage2_student_full40_8gpu_20260727 \
  --run-tag s1step3000-s2step5000-student-full40
```

## Intentionally unverified

- No real Stage-2 training or LIBERO rollout was launched.
- The production Stage-2 target inference preflight cannot run until that
  future checkpoint exists. It is covered with complete synthetic
  Stage-1/Stage-2 checkpoints and is enforced immediately after training
  and before eval-only output.
- Signal/fail-fast behavior was verified with real OS process groups and
  sentinel descendants, not CUDA workloads.
- No branch push, merge, worktree cleanup, or user output deletion was
  performed.
