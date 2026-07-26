# Cosmos Stage-1 → Stage-2 → Eval Recovery Wrapper

Date: 2026-07-26

## Goal

Provide one command that resumes the interrupted aligned Cosmos LIBERO run from
the existing Stage-1 step-1000 checkpoint and then continues through Stage-2
and the full student/teacher evaluation.

## Design

Add `distillation_flowmap/resume_cosmos_aligned_full40_8gpu.sh` as a thin
wrapper around `run_cosmos_stage1_stage2_eval_8gpu.sh`. The wrapper owns no
training or evaluation logic. It supplies the validated paths and experiment
arguments, then invokes the existing launcher three times:

1. `--phase stage1 --resume-stage stage1 --resume-step 1000`
2. `--phase stage2`
3. `--phase eval`

The commands are connected with fail-fast shell semantics, so Stage-2 cannot
start unless Stage-1 completes successfully, and evaluation cannot start
unless Stage-2 completes successfully.

## Interface

The default invocation is:

```bash
bash distillation_flowmap/resume_cosmos_aligned_full40_8gpu.sh
```

The wrapper accepts `--dry-run` to print the three resolved child commands
without writing files or starting workers. Environment variables may override
the Python executable, Cosmos repository, model paths, dataset paths, ports,
and GPU list. The output root, run tag, training steps, save interval, episode
count, and resume step have defaults matching the interrupted run.

## Safety

- Require exactly eight visible GPUs.
- Require the Stage-1 resume checkpoint before starting.
- Preserve the existing run tag and output tree.
- Let the existing launchers reject conflicting Stage-2/evaluation outputs.
- Do not delete, rename, or overwrite checkpoints.
- Do not duplicate resolved-config, lineage, provenance, or evaluation logic.

## Tests

A launcher contract test will run the wrapper against a recording stub and
verify:

- one invocation emits exactly three ordered child phases;
- Stage-1 receives the exact resume stage and step;
- Stage-2 and eval receive the same run identity and experiment sizes;
- `--dry-run` performs no production launch;
- missing Stage-1 checkpoint fails before any child launch.
