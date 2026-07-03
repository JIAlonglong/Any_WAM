# Cosmos Policy Raw LIBERO Distillation

This records the runnable LIBERO path that uses official Cosmos Policy raw
observation inference as the action teacher. The normal WanVA stage1/stage2
configs remain unchanged; this is a parallel Cosmos backend.

## Readiness Items

1. Two-GPU raw smoke: use the smoke command below with
   `COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION=1`.
2. Speed pilot: use the pilot command below and compare per-step time against
   the single-GPU 1-step smoke baseline, which was about 85 seconds per step
   before checkpoint saving.
3. Teacher denoise setting: use `COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION=1`
   for smoke/pilot speed checks and `=5` for formal teacher quality. The formal
   launcher defaults to 5.
4. Teacher quality monitoring: training now logs
   `cosmos_raw/teacher_gt_mse`, `cosmos_raw/teacher_gt_l1`,
   `cosmos_raw/teacher_abs_mean`, `cosmos_raw/gt_abs_mean`, and
   `cosmos_raw/enabled_microbatches` when the raw teacher branch is active.
5. Environment/assets: run `cosmos_policy_preflight.py` before a formal run.
   It checks Cosmos weights, local model/tokenizer, h5py availability in the
   Cosmos env, official `cosmos_utils` import, student base, and dataset path.
6. Standardized run dirs and wandb: use
   `run_libero_cosmos_policy_raw_stage1_to_stage2.sh`. It defaults to
   `ENABLE_WANDB=0`; set `ENABLE_WANDB=1` plus the usual wandb env vars for
   tracked formal runs.

## Environment Defaults

```bash
export COSMOS_POLICY_PATH=/root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B
export STUDENT_BASE_MODEL_PATH=/root/nas/junjie/jj/Any_WAM/checkpoints/lingbot-va-posttrain-libero
export DATASET_PATH=/root/nas/junjie/jj/Any_WAM/training_data/libero-long-lerobot
export COSMOS_POLICY_USE_RAW_INFERENCE=1
export COSMOS_POLICY_INFERENCE_MODE=subprocess
export COSMOS_POLICY_PYTHON=/root/nas/junjie/cosmos_predict2_5/envs/predict2_py310/bin/python
export COSMOS_PREDICT2_REPO=/root/nas/junjie/cosmos_predict2_5/repos/cosmos-predict2.5
export COSMOS_POLICY_EXTRA_PYTHONPATH=/root/nas/junjie/conda_envs/any_wam/lib/python3.10/site-packages
export COSMOS_PREDICT25_LOCAL_MODEL_DIR=/root/nas/junjie/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World
```

The extra Python path is needed because the Cosmos env currently borrows `h5py`
from the Any_WAM env. The local model dir must contain
`tokenizer/tokenizer.pth`; preflight can create the symlink with
`--fix-local-tokenizer` when the source tokenizer exists.

## Preflight

Fast path and dataset raw sample:

```bash
cd /root/nas/junjie/jj/Any_WAM
/root/nas/junjie/conda_envs/any_wam/bin/python \
  distillation_flowmap/cosmos_policy_preflight.py \
  --fix-local-tokenizer \
  --check-dataset
```

Official worker inference check, more expensive:

```bash
COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION=1 \
/root/nas/junjie/conda_envs/any_wam/bin/python \
  distillation_flowmap/cosmos_policy_preflight.py \
  --fix-local-tokenizer \
  --check-dataset \
  --check-worker
```

## Two-GPU Smoke

```bash
cd /root/nas/junjie/jj/Any_WAM
COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION=1 \
COSMOS_POLICY_USE_RAW_INFERENCE=1 \
ENABLE_WANDB=0 \
MAX_TRAIN_STEPS=1 \
SAVE_INTERVAL=1 \
OUTPUT_DIR=distillation_flowmap/output_smoke_cosmos_policy_raw_stage1_2gpu \
CUDA_VISIBLE_DEVICES=0,1 \
/root/nas/junjie/conda_envs/any_wam/bin/torchrun \
  --standalone \
  --nproc_per_node=2 \
  distillation_flowmap/train.py \
  --load-worker 0 \
  --batch-size 1 \
  --gradient-accumulation-steps 1
```

## Short Pilot

```bash
cd /root/nas/junjie/jj/Any_WAM
COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION=1 \
COSMOS_POLICY_USE_RAW_INFERENCE=1 \
ENABLE_WANDB=0 \
MAX_TRAIN_STEPS=5 \
SAVE_INTERVAL=999 \
OUTPUT_DIR=distillation_flowmap/output_pilot_cosmos_policy_raw_stage1_2gpu_5step \
CUDA_VISIBLE_DEVICES=0,1 \
/root/nas/junjie/conda_envs/any_wam/bin/torchrun \
  --standalone \
  --nproc_per_node=2 \
  distillation_flowmap/train.py \
  --load-worker 0 \
  --batch-size 1 \
  --gradient-accumulation-steps 1
```

## Formal Stage1 -> Stage2

```bash
cd /root/nas/junjie/jj/Any_WAM
NGPU=2 \
STAGE1_STEPS=500 \
STAGE2_STEPS=5000 \
ENABLE_WANDB=1 \
distillation_flowmap/run_libero_cosmos_policy_raw_stage1_to_stage2.sh
```

For a quicker pilot through the same script:

```bash
cd /root/nas/junjie/jj/Any_WAM
NGPU=2 \
COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION=1 \
STAGE1_STEPS=5 \
STAGE2_STEPS=5 \
SAVE_INTERVAL=999 \
ENABLE_WANDB=0 \
distillation_flowmap/run_libero_cosmos_policy_raw_stage1_to_stage2.sh
```

## Observed Runs

- Single-GPU raw stage1 1-step with denoise steps 1: passed, loss about
  `0.0078`, step time about `85s`, full run with checkpoint save about `2min`.
- Single-GPU raw stage2 1-step with denoise steps 1: passed, loss about
  `0.0075`.
- Preflight with `--check-dataset`: passed. It found the LIBERO raw sample with
  `primary=(128, 128, 3)`, `wrist=(128, 128, 3)`, `proprio=(9,)`, and
  `actions=(30, 16, 4, 1)`.
- Preflight with `--check-worker` and denoise steps 1: passed. Official Cosmos
  inference returned raw teacher x0 shape `(1, 30, 16, 4, 1)`.
- Two-GPU raw stage1 1-step smoke with denoise steps 1: passed. Training step
  took about `38.6s`; full run including checkpoint save took about `2m13s`.
  Logged `ctgt=0.007/0.010`, `loss=0.0063`.
- Two-GPU raw stage1 5-step pilot with denoise steps 1: passed. First step
  took about `38s` including worker load; steady-state steps were about
  `3-4s` each. Final logged `ctgt=0.008/0.011`, `loss=0.0062`.
- Two-GPU raw stage2 1-step smoke resumed from the stage1 smoke checkpoint:
  passed. Training step took about `39.1s`; full run including checkpoint save
  took about `2m27s`. Logged `ctgt=0.007/0.010`, `loss=0.0064`.
