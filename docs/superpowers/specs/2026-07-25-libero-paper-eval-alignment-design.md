# LIBERO Paper Evaluation Alignment Design

## Goal

Align the LingBot-VA/LIBERO evaluation and Figure 4 export with the paper
definitions without changing the matched-step Teacher implementation, shared
prior/seed behavior, or the default 20/50-step Teacher baseline.

## Scope

### Figure 4 action diagnostics

The paper-facing checkpoint sweep must use deployment-aligned action history:

- `E_student` maps to
  `mechanism/action_error_student_generated_history_context`.
- `E_video` maps to
  `mechanism/action_error_teacher_video_generated_history_context`.
- `E_joint` remains
  `mechanism/action_error_teacher_joint_context`.

The derived `Delta_video`, `R_video`, `Delta_joint`, and `G_residual` values
must be recomputed from those three mapped values. Existing legacy
teacher-forced diagnostics remain available for debugging but are not exported
as the paper's Figure 4 curves.

Figure 4 is generated only from the fixed-probe post-hoc checkpoint sweep.
Online training-batch diagnostics are monitoring signals and must be labelled
as such; they are not a substitute for the fixed probe bank.

### Latency

Measure one complete joint video/action sampler call in the inference server
with `time.perf_counter()`. Exclude model loading, environment stepping, video
encoding, and client/server startup. Persist one JSONL record per sampler call
with model, suite, task, episode, video/action budgets, call index, and elapsed
milliseconds.

The evaluation merger validates the latency records and reports p50 sampler
latency alongside success rate. Raw records are retained so the percentile is
auditable.

### Naive composition

Do not relabel Stage-I-only as naive composition. Stage-I-only is distilled;
the paper's naive composition baseline is explicitly no-distillation.

The formal orchestrator may accept an optional `NAIVE_CKPT` or
`NAIVE_RUNNER`. It evaluates and reports the baseline only when a real naive
implementation is supplied. Otherwise the manifest records
`naive_composition.status = "missing"` instead of inventing a result.

## Explicit non-goals

- Do not change or validate the matched-step Teacher sampler.
- Do not add shared model-prior seed enforcement.
- Do not evaluate the default 20/50-step Teacher baseline.
- Do not modify training losses, checkpoints, or optimizer state.

## Outputs and compatibility

- Existing success summaries remain readable.
- New summaries add latency statistics and an evaluation contract manifest.
- Existing legacy mechanism metrics remain logged.
- Paper plots consume only deployment-aligned generated-history metrics.

## Tests

- The Figure 4 mapper must select generated-history metrics and recompute all
  derived quantities.
- Online monitoring records must not be accepted as post-hoc Figure 4 input.
- Latency p50 must be computed from raw valid records and reject missing or
  malformed provenance.
- Stage-I-only must never be labelled `naive_composition`.
- The orchestrator must report a missing naive baseline unless an explicit
  valid runner/checkpoint is supplied.
- The 40-task 1/1, 2/2, 4/4 success protocol remains unchanged.
