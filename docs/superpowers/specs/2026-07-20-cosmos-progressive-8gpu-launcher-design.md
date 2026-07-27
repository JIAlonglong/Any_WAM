# Cosmos progressive Stage-2 eight-GPU launcher

## Goal

Provide one reproducible, guarded remote entry point for the aligned Cosmos
progressive Stage-2 chain on eight A800 GPUs:

`S4 (5000 steps) -> S2 (3000 steps) -> S1 (3000 steps)`.

The launcher must start a new S4 from the released raw Cosmos Stage-1 EMA,
then require the final checkpoint of each preceding newly trained stage.  It
must not initialize a new S4 from either the released raw Stage-2 checkpoint
or the previously released progressive S4 checkpoint.

## Immutable inputs

The one-time ModelScope source is pinned to the Stage-1 commit
`dbec074e378aafdf025d1eae9941871353f60c29`:

- model: `JIAlonglong/any-wam-cosmos-checkpoints`
- source artifact: `raw_stage1_5000/target_student/transformer`
- model file: `diffusion_pytorch_model.safetensors`
- expected SHA-256:
  `13091097309a6579c938ea926fa4752f904f38f57c21f050e59b8b3cd70cc03a`

After one-time local download and transfer, the remote source root is:

`/kpfs-intern/jialongliu/models/modelscope/JIAlonglong/any-wam-cosmos-checkpoints/raw_stage1_5000`

This artifact intentionally contains only `target_student/transformer`.
The preparation action creates the relative, zero-copy link
`online_student -> target_student` in that root.  The resulting root is
compatible with the existing trainer's preliminary online-checkpoint check,
while S4 explicitly uses `RESUME_ONLINE_FROM_TARGET=1` and therefore loads
the Stage-1 EMA as its online initialization.

The common remote inputs are:

- project worktree:
  `/kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/cosmos-progressive-s4-eval`
- LIBERO dataset:
  `/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot`
- official Cosmos Policy model:
  `/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B`
- official Cosmos repository:
  `/kpfs-intern/jialongliu/projects/cosmos-predict2.5`
- official Cosmos cu128 Python:
  `/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/bin/python`
- local Predict2 base model:
  `/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World`

All paths are validated before any GPU process is launched.  They may be
overridden only through named environment variables for a different cluster.

## Launcher interface

Create one file:

`distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh`

The interface is:

```bash
bash distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh <s4|s2|s1> \
  [--master-port PORT] [--dry-run] [--resume-step STEP]
```

It uses the fixed default output root:

`/kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_cosmos_progressive_dance_aligned_8gpu_20260720`

with stage-specific directories `s4`, `s2`, and `s1`.  `OUTPUT_ROOT` is the
sole output-location override.  A fresh run refuses an output stage that
already contains checkpoints.  A continuation requires an explicit
`--resume-step`; it resumes only from
`$OUTPUT_ROOT/<stage>/checkpoints/step_<STEP>`, retains optimizer state, and
does not accept an arbitrary external checkpoint.

## Stage and checkpoint policy

| Stage | Teacher -> student | Dance rollout / velocity | Steps | Fresh initialization |
| --- | --- | --- | ---: | --- |
| `s4` | 8 -> 4 | 4 / 1.0 | 5000 | released Stage-1 root, target EMA |
| `s2` | 4 -> 2 | 2 / 1.0 | 3000 | `$OUTPUT_ROOT/s4/checkpoints/step_5000` online student |
| `s1` | 4 -> 1 | 1 / 0.0 | 3000 | `$OUTPUT_ROOT/s2/checkpoints/step_3000` online student |

The launcher sets `COSMOS_PROGRESSIVE_STAGE`; the aligned progressive config
remains the sole definition of the corresponding rollout pair, endpoint
weights, gradient schedule, Dance rollout length, and velocity weight.  This
prevents a shell default from drifting from the config.

For a fresh stage, `RESET_RESUME_STEP=1` and optimizer recovery is disabled.
S4 uses `RESUME_ONLINE_FROM_TARGET=1`; S2 and S1 use `0`.  For an explicit
same-stage continuation, `RESET_RESUME_STEP=0`, optimizer recovery is enabled,
and online resume is used.

## Runtime and storage policy

The launcher starts exactly eight FSDP1 ranks:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8
```

The adapter maps worker rank `r` to physical GPU `r`; no separate four-GPU
worker reservation is made.  This matches the established rank-aware worker
implementation and gives every worker the same GPU ordinal as its FSDP rank.

The launcher exports the official worker Python, repository, model root,
Predict2 local base-model root, CUDA/PyTorch allocator setting, and offline HF
flags.  It uses FSDP1, student/OPD activation checkpointing, serialized CFG,
and standalone OPD scheduling from the current progressive configuration.

`SAVE_INTERVAL=1000` is explicit.  The current configuration's 250-step
default would retain many full-size checkpoint directories because the trainer
does not implement the comment-labelled recent-checkpoint cleanup.  A 1000
step cadence retains useful recovery points without unnecessarily consuming
hundreds of GB.  The final checkpoint is always written by the trainer.

W&B remains disabled by the progressive config; existing TensorBoard logging
is left enabled wherever the trainer configuration creates its writer.

## Preflight and dry-run behavior

Before launching, the script verifies:

1. stage name and options are valid;
2. exactly eight unique numeric GPU ordinals are configured;
3. all immutable input paths and the needed checkpoint transformer files
   exist;
4. S4 has both compatibility paths and S2/S1 have their required predecessor;
5. the output safety rule is satisfied; and
6. the official Cosmos Python is executable.

`--dry-run` performs every validation and prints the fully resolved exported
environment and `torchrun` command, but creates no output directory and
starts no Python/GPU process.  Normal execution creates the stage output
directory only after successful preflight, then uses `exec` for `torchrun` so
its exit code is the script's exit code.

## Verification

Implementation verification must include:

1. `bash -n` for the new launcher;
2. dry-run tests for S4, S2, and S1 with temporary fake but structurally valid
   checkpoints;
3. negative tests for an absent Stage-1 compatibility link, missing
   predecessor, non-eight-GPU list, unsafe existing output, and invalid resume
   checkpoint; and
4. an import/config smoke proving `s4`, `s2`, and `s1` resolve to
   `(4, 1.0)`, `(2, 1.0)`, and `(1, 0.0)` Dance settings respectively.

No GPU training is started during verification.  The user manually requests
the eight A800 allocation and runs the final commands after the script's
dry-run output is accepted.
