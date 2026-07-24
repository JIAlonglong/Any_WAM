# LIBERO APM Small-Sample Ablation Design

## Objective

Migrate the RobotWin APM LoRA ablation protocol to the LingBotVA LIBERO
Stage-2 branch while preserving the LIBERO video-only hypothesis. The
experiment must isolate whether endpoint anchoring and intermediate field
correction improve video generation, action prediction, and closed-loop
success without adding action OPD.

## Fixed Experimental Contract

- Teacher: `/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero`.
- Shared initialization:
  `distillation_flowmap/output_libero_fullft_stage1_warmup/checkpoints/step_2000`.
- Dataset:
  `training_data/libero-long-lerobot`.
- Training task:
  `put both the alphabet soup and the tomato sauce in the basket`.
- Training split: episodes 0–39; held-out diagnostic split: episodes 40–49.
- Seed: 42 for model training; evaluation uses the benchmark's episode seeds.
- Four GPUs train one arm at a time; arms run serially to avoid memory
  contention and make failures attributable to one run.
- Stage-2 uses LoRA rank 128, alpha 64, dropout 0, batch size 1, gradient
  accumulation 1, and learning rate `5e-6`.
- Default training budget: 500 optimizer steps with checkpoints at 100, 200,
  300, 400, and 500.
- Attention backend: `torch`, matching the verified LIBERO smoke path.
- Action main loss and GT regression remain enabled.
- Action OPD, joint action OPD rollout, and DanceOPD action-velocity loss are
  hard-disabled in every arm.

## Ablation Arms

| Arm | Endpoint weight | Velocity weight | Interpretation |
| --- | ---: | ---: | --- |
| `stage1_only` | 0 | 0 | Small-sample Stage-2 control with no video OPD |
| `anchor_only` | 1 | 0 | Endpoint anchor \(G_\text{anchor}\) only |
| `field_only` | 0 | 1 | Intermediate field correction \(G_\text{comp}\) only |
| `apm` | 1 | 1 | Endpoint anchor plus field correction |

All other settings, data, initialization, seed, and training duration are
identical. The four-arm comparison therefore attributes differences to the
two video OPD terms.

## Training Architecture

Create a LIBERO-specific ablation config derived from
`config_libero_fullfinetune_stage2_video_only_opd.py`. It exposes LoRA and
dataset-manifest settings but retains the parent config's video-only safety
checks. A frozen JSON variants file defines the four arms.

A protocol builder writes:

- `train_manifest.json` selecting dataset indices 0–39;
- `heldout_manifest.json` selecting indices 40–49;
- `run_manifest.json` for every arm, containing the git hash, exact
  environment, command, data indices, checkpoint paths, and evaluation
  contract.

The four-GPU launcher validates paths, GPU IDs, ports, manifests, and output
nonexistence before starting. It then executes each arm synchronously with
`torchrun --nproc_per_node=4`. A failure stops the pipeline without starting
the next arm.

## Diagnostics

Training retains the RobotWin diagnostics adapted to video-only OPD:

- raw and weighted endpoint/anchor loss;
- raw and weighted velocity/field loss;
- `G_anchor` and `G_comp` contribution ratios and scaling factors;
- video, action, and shared-branch gradient norms;
- DanceOPD query index and sigma;
- fixed-sample mechanism diagnostics tracking video improvement and the
  associated action-error trend.

The held-out split is recorded separately and is never used by the optimizer.
After each arm finishes, an offline held-out diagnostic evaluates its final
checkpoint on the same ten episodes, producing comparable video and action
error summaries.

## Closed-Loop Evaluation

Evaluation uses the existing LingBotVA LIBERO server/client path so training
and inference share the same joint AnyFlow conditioning semantics.

Each final arm checkpoint and the shared Stage-1 baseline are evaluated with
matched video/action budgets:

- 1 video step / 1 action step;
- 2 video steps / 2 action steps;
- 4 video steps / 4 action steps.

The default formal evaluation runs 10 episodes per task on all 40 standard
tasks:

- LIBERO-Long (`libero_10`);
- LIBERO-Spatial;
- LIBERO-Object;
- LIBERO-Goal.

With four GPUs, one worker owns one suite and evaluates its ten tasks
sequentially. Models and sampling budgets run serially so each GPU hosts only
one inference server. The selected training task is also reported explicitly
as the in-domain result.

The evaluator writes worker logs, raw episode records, per-suite summaries,
per-budget summaries, and a final matrix indexed by model and sampling budget.
The matrix includes successes, attempts, success rate, and selected-task
success rate.

## User Interface

The main entry point is one short command:

```bash
bash distillation_flowmap/ablation/run_libero_apm_lora_4gpu_serial.sh \
  --phase all \
  --gpu-ids 0,1,2,3
```

Supported phases are:

- `train`: build manifests and train the four arms;
- `offline-eval`: run held-out diagnostics;
- `closed-loop`: evaluate Stage-1 and all completed arms at 1/2/4 steps;
- `all`: run the three phases serially.

The launcher also accepts `--steps`, `--save-interval`, `--episodes`,
`--output-root`, port-base overrides, arm selection, and `--dry-run`.

## Failure Handling and Resumption

- Existing run directories are never overwritten.
- A completed checkpoint is detected before a phase is skipped or resumed.
- Evaluation skips already complete model/budget summaries and reruns only
  missing workers.
- Every spawned server/client is terminated on success, failure, or signal.
- Port and GPU collisions fail during preflight.
- Missing worker results make the aggregate phase fail rather than silently
  lowering the denominator.
- Training and evaluation logs live under the experiment root and contain the
  resolved command and environment.

## Verification

Implementation uses test-first coverage for:

- exact four-arm endpoint/velocity mapping;
- the hard action-OPD-zero contract;
- deterministic 40/10 manifest indices;
- four unique GPUs and noncolliding ports;
- dry-run commands using four processes and the shared Stage-1 checkpoint;
- matched 1/1, 2/2, and 4/4 inference budgets;
- all four LIBERO suites and all 40 tasks;
- resumable worker detection and strict result aggregation.

A one-GPU, one-step smoke of one arm verifies config loading, dataset
filtering, LoRA setup, main loss, video OPD, optimizer, and checkpoint save.
The four-GPU script then runs in dry-run/check-only mode before handoff.
