# Cosmos Teacher Four-GPU Evaluation Resume Wrapper

Date: 2026-07-27
Status: Approved design

## Goal

Provide a short, reproducible command that resumes the official Cosmos teacher
LIBERO matched-budget evaluation from an existing matrix root without rerunning
completed role/K/suite cells.

## Interface

Add:

```text
evaluation/libero/run_cosmos_teacher_124_eval_4gpu_resume.sh MATRIX_ROOT
```

The wrapper uses the audited project defaults for the official Cosmos teacher,
worker environment, all-40 prompt table, four colocated shards, 50 episodes per
task, and representative-video seed 0. Callers may override those paths through
the same environment variables used by the existing launchers.

## Resume semantics

The wrapper iterates K=1/2/4 and:

1. `libero_10`
2. `libero_spatial`
3. `libero_object`
4. `libero_goal`

For each cell:

- If no output exists, run the existing formal evaluator.
- If `formal_summary.json` exists, validate role, suite, matched video/action
  budget, checkpoint identity, formal classification, 10 unique tasks, and
  exactly the requested number of episodes before skipping it.
- If an output directory exists without a valid complete summary, stop without
  overwriting it.

After all twelve cells are complete, invoke the existing strict matrix merger.
The merger must still reject missing or duplicate tasks and must produce the
JSON and CSV full40 summaries.

## Safety and testing

- Do not modify the existing no-overwrite matrix runner.
- Do not delete or rewrite completed episode results.
- Dry-run and tests must not create evaluation directories.
- Add launcher tests that first fail because the wrapper is absent, then verify
  completed-cell skipping, incomplete-cell rejection, remaining-cell launch
  order, and final strict merge planning.
