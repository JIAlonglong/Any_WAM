# Native Teacher Coarse-Grid Final Fix Report

Date: 2026-07-26 (Asia/Shanghai)

Base commit: `f9b320e1b5f426a12bea2fa5e7e561f100a5f8c3`

Implementation commit: `d42d8ef2bc29aa392abfafb39d63ffb8cc436703`

## Status

Complete. The final review wave prevents native-teacher identity ambiguity,
separates native launcher defaults from the student launcher, and protects
formal native results from stale latency/result contamination. No existing
result was deleted, truncated, or overwritten.

## Files

- `evaluation/libero/run_eval_new.sh`
- `evaluation/libero/tests/test_run_eval_new_contract.py`
- `evaluation/libero/run_lingbotva_native_teacher_4suite_124_eval_8gpu.sh`
- `evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py`
- `.superpowers/sdd/2026-07-26-native-teacher-coarse-grid/final-fix-report.md`

All unrelated dirty stage-2 files were left unstaged and uncommitted.

## Finding 1: Reverse Backend/Identity Routing

The FlowMap branch now rejects exactly `MODEL_NAME=teacher_native` with exit
status 2. The existing `target_student`, `stage2`, and other student identities
remain on the unchanged FlowMap path.

### RED

```text
python -m pytest \
  evaluation/libero/tests/test_run_eval_new_contract.py::test_flowmap_backend_rejects_reserved_native_teacher_identity \
  -q

1 failed
Expected return code 2; observed 0 with a valid FlowMap CHECK_ONLY plan.
```

### GREEN

```text
python -m pytest \
  evaluation/libero/tests/test_run_eval_new_contract.py::test_flowmap_backend_rejects_reserved_native_teacher_identity \
  -q

1 passed in 0.04s

python -m pytest evaluation/libero/tests/test_run_eval_new_contract.py -q

14 passed in 0.09s
```

## Finding 2: Native Default Port Separation

Native defaults now resolve to master/WebSocket ports `30680/30780`. The
student launcher remains at `29680/29780`. Explicit native overrides continue
to resolve worker-by-worker, and the pre-existing overlap rejection remains
covered.

### RED

```text
python -m pytest \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py::test_native_default_ports_are_distinct_from_student_defaults \
  -q

1 failed
Observed native ports ('29680', '29780'); expected ('30680', '30780').
```

### GREEN

```text
python -m pytest \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py::test_native_default_ports_are_distinct_from_student_defaults \
  -q

1 passed in 0.06s

python -m pytest \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py \
  evaluation/libero/tests/test_run_lingbotva_4suite_124_eval_8gpu.py \
  -q

9 passed in 0.08s
```

## Finding 3: Formal Result-Root Reuse Protection

The launcher classifies `<output-root>/teacher_native` as `absent`, `empty`,
`nonempty`, or `not_directory` before scheduling workers. A real launch accepts
only `absent` and `empty`; `nonempty` is rejected before any worker plan or
server invocation. CHECK_ONLY remains successful for inspection, prints the
root state and `launch_allowed` decision, and does not create directories or
modify a sentinel.

### RED

```text
python -m pytest \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py::test_launcher_rejects_nonempty_model_root_before_server_invocation \
  -q

1 failed
Observed return code 0, all 24 WORKER lines, and server-stub invocation.

python -m pytest \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py::test_check_only_reports_nonempty_model_root_without_modifying_it \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py::test_check_only_plans_native_teacher_40_tasks_at_matched_124 \
  -q

2 failed
No RESULT_ROOT contract was reported, and CHECK_ONLY created budget directories.
```

### GREEN

```text
python -m pytest \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py::test_launcher_rejects_nonempty_model_root_before_server_invocation \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py::test_launcher_accepts_new_or_empty_model_root \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py::test_check_only_reports_nonempty_model_root_without_modifying_it \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py::test_check_only_plans_native_teacher_40_tasks_at_matched_124 \
  -q

5 passed in 0.12s
```

The rejection test uses a sentinel historical result and an external server
stub. It confirms exit status 2, no `WORKER` output, no server marker, and
unchanged sentinel content. Separate behavior cases confirm both a new model
root and an existing empty model root proceed.

## Final Verification

```text
python -m pytest \
  distillation_flowmap/tests/test_native_teacher_contract.py \
  evaluation/libero/tests/test_run_eval_new_contract.py \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py \
  evaluation/libero/tests/test_run_lingbotva_4suite_124_eval_8gpu.py \
  distillation_flowmap/tests/test_runtime_metadata.py \
  -q

44 passed in 27.91s
```

Both of these commands exited 0:

```text
bash -n evaluation/libero/run_eval_new.sh
bash -n evaluation/libero/run_lingbotva_native_teacher_4suite_124_eval_8gpu.sh
```

Full-worktree `git diff --check` also exited 0.

## Self-Review

- Backend/model identity is now guarded in both directions.
- Only the reserved `teacher_native` name is rejected by FlowMap.
- Native defaults differ from student defaults exactly as demonstrated.
- Explicit `31680/31780` native overrides and overlapping-range failures remain
  covered by behavior tests.
- Result-root validation occurs before `run_budget`, worker scheduling, and
  model/server invocation.
- Empty and absent roots are accepted without weakening nonempty-root refusal.
- CHECK_ONLY is read-only with respect to the output root and makes the formal
  launch decision visible.
- No cleanup path was added; historical results remain user-owned.

## Concerns and Deferred Work

- No live GPU smoke or formal eight-GPU evaluation was rerun in this fix wave;
  the final change is covered by shell-level behavior tests with only external
  server/client executables stubbed.
- Persisting combined smoke stdout/stderr remains a deferred
  evidence-retention improvement. The existing Task 5 report records the
  successful smoke evidence, but this final wave intentionally makes no
  production logging change.
- CHECK_ONLY deliberately returns 0 even when it reports
  `launch_allowed=no`, so it remains usable to inspect all 24 planned workers.
  A real launch enforces the refusal with exit status 2.
