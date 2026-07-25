# LIBERO Figure 4 Target-Student Post-hoc Diagnostics Design

## Goal

Build a reproducible, single-GPU checkpoint sweep for Figure 4 on the
LingBot-VA/LIBERO factor-1 APM run. The sweep evaluates the deployed EMA
`target_student` at every checkpoint that actually exists, reuses one frozen
probe bank across all checkpoints, and exports raw metrics plus the two-panel
paper figure without changing training, checkpoints, optimizer state, or model
outputs.

## Scope

The target run is:

```text
/kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/
output_libero_lingbotva_stage2_video_only_opd_universal_from_stage1_step2000_
steps10000_factor1-deployment-align-20260724
```

The sweep uses:

- Student role: `target_student` (EMA deployment weights).
- Diagnostic seed: `42`.
- Diffusion locations: `r=500`, `s=250`.
- Teacher reference integration steps: `8`.
- One fixed LIBERO task subset and one fixed probe bank.
- Batch size 1 unless the existing loader requires a smaller supported unit.
- Actual checkpoint steps only. The run saves every 1000 successful optimizer
  steps, so the evaluator must not manufacture 100-step points.

Closed-loop environment stepping and retraining are out of scope.

## Existing Implementation to Reuse

The implementation must reuse the current mechanism diagnostic path:

- `distillation_flowmap/flowmap_trainer.py`
  - `FlowMapDistiller`
  - `_get_mechanism_diagnostic_batches`
  - `_run_mechanism_diagnostics`
- `distillation_flowmap/flowmap_step.py`
  - `_prepare_mechanism_diagnostic_context`
  - `_diagnostic_student_joint_map`
  - `_diagnostic_teacher_joint_rollout`
  - `_compute_mechanism_diagnostic_stats`
- `distillation_flowmap/mechanism_diagnostics.py`
  - exact squared-L2 and MSE reductions
  - masked action MSE
  - finite-value aggregation

The existing online JSONL is not a substitute for the checkpoint sweep. Its
diagnostic seed includes `diagnostic_index`, so different training steps use
different noise. The post-hoc evaluator must use the same probe tensors for
every checkpoint.

## Architecture

### Checkpoint discovery

Discover directories matching:

```text
checkpoints/step_*/target_student/transformer
```

Read the step from both the directory name and `config.json`. Reject a
mismatch, duplicate step, missing config, missing weights, or unsorted result.
Sort by the real checkpoint step.

Checkpoint discovery runs immediately before evaluation so checkpoints created
while the training job is still active can be included in a later invocation.

### Probe bank

Build the diagnostic sample once from the configured LIBERO dataset and save:

- task and dataset index;
- instruction/conditioning identity available from the batch;
- seed, `r`, `s`, and teacher step count;
- tensor keys, shapes, and dtypes;
- fixed video and action prior noise or sufficient RNG state to reproduce it;
- action mask metadata.

Every checkpoint must consume the exact same saved probe tensors. Checkpoint
step must not enter the probe seed.

The serialized probe bank is diagnostic input only. Loading it must not mutate
the source dataset or a checkpoint.

### Model lifecycle

Use one single-GPU process:

1. Initialize the frozen teacher, schedulers, dataset, and diagnostic helpers.
2. Disable W&B, TensorBoard training logging, optimizer resume, and training.
3. Skip the EMA target copy normally created by the training constructor.
4. Load one target-student checkpoint as the active student.
5. Set teacher and student to `eval()` and disable gradients.
6. Evaluate inside `torch.inference_mode()` or `torch.no_grad()`.
7. Release the old student weights and CUDA cache before loading the next
   checkpoint if in-place loading is not memory-safe.

The teacher may remain resident. No resolution, latent shape, attention mask,
or action downsampling setting may be changed to avoid OOM.

### Metrics

The evaluator calls the existing diagnostic implementation and maps its
aggregates to the paper schema:

- `g_anchor_l2` from `mechanism/g_anchor`;
- `g_anchor_mse` from `mechanism/g_anchor_mse`;
- `g_comp_l2` from `mechanism/g_comp`;
- `g_comp_mse` from `mechanism/g_comp_mse`;
- `e_student` from `mechanism/action_error_student_context`;
- `e_video` from `mechanism/action_error_teacher_video_context`;
- `e_joint` from `mechanism/action_error_teacher_joint_context`.

Derived values:

```text
delta_video = e_student - e_video
r_video = max(0, delta_video) / max(e_student, 1e-8)
delta_joint = e_student - e_joint
g_residual = e_video - e_joint
```

`E_video` is a counterfactual teacher-video swap:
`(y_r.video, z_r.action)`. Because it is not a trajectory-synchronized joint
state, the manifest and report must not describe it as a causal oracle.

Every non-finite sample records checkpoint, metric, and sample identity. It is
an error for a checkpoint aggregate to silently omit a non-finite sample.

### Outputs

Write under:

```text
<run-root>/outputs/mechanism_diagnostics/
```

Required artifacts:

- `manifest.json`
- `probe_bank.pt`
- `probe_bank_metadata.json`
- `mechanism_metrics.jsonl`
- `mechanism_metrics.csv`
- `figure4_mechanism_diagnostics.png`
- `figure4_mechanism_diagnostics.pdf`
- `figure4_panel_a.png`
- `figure4_panel_b.png`
- `run.log`
- `reproduce.sh`

Write results atomically. A completed JSONL contains exactly one aggregate
record for every evaluated checkpoint, ordered by real step. Re-running may
resume missing checkpoints, but it must not append duplicates.

The manifest records git commit, teacher path, student role, all checkpoint
paths and steps, dataset/task subset, seed, probe metadata, `r`, `s`, teacher
steps, dtype, GPU model, peak allocated memory, total runtime, reductions, and
action-mask rules.

### Figure

Generate a horizontal two-panel Figure 4.

Panel (a):

- x: successful optimizer steps;
- y: squared latent distance;
- raw `g_anchor_l2` and `g_comp_l2`;
- one shared y-axis, optionally logarithmic when the range requires it.

Panel (b):

- x: successful optimizer steps;
- y: masked action MSE;
- raw `e_student`, `e_video`, and `e_joint`;
- one shared scale.

The default plot does not smooth, interpolate, or fabricate uncertainty.
Single-seed results have no confidence interval. Export a raster PNG and a
vector PDF with paper-readable fonts, line widths, markers, and a legend that
does not cover data.

## Correctness and Tests

Tests must be written before new production code and must verify:

1. `G_comp=0` for identical direct and composed video tensors.
2. `G_anchor=0` for identical teacher continuation and teacher endpoint.
3. `E_student=E_video` for identical student and video-swap contexts.
4. Masked padding does not affect action MSE.
5. Diagnostic outputs are detached and have `requires_grad=False`.
6. JSONL writes exactly one aggregate record per checkpoint.
7. Checkpoints are sorted by real step and metadata mismatches are rejected.
8. Probe seed and serialized probe tensors do not depend on checkpoint step.
9. Plot and CSV generation preserve the raw checkpoint values.

The implementation must run the focused test file and the existing mechanism
diagnostic regression tests before the sweep.

## Execution Strategy

The training run may still be creating checkpoints. Implementation and unit
tests can proceed without interfering with training. Before the formal sweep,
rediscover checkpoints:

- If step 10000 exists, evaluate all available steps 1000 through 10000.
- If training is still active, a partial sweep may be run for smoke validation,
  but the final Figure 4 must be regenerated after the final checkpoint exists.

If the A800 cannot hold one teacher and one student at the unchanged model
shape, stop the sweep and report the peak memory and failing phase. Do not
silently switch protocol.

## Reporting

The final report lists modified files, evaluated checkpoints and real steps,
metric trends, requested hypothesis checks, artifact paths, test evidence,
reproduction command, GPU, peak memory, runtime, and every unmet requirement.
Negative or flat trends are reported as observed without editing or hiding
data.
