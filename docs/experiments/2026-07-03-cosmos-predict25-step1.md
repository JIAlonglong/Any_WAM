# Cosmos-Predict2.5-2B Step-1 Asset Check

Date: 2026-07-03

## Goal

Prepare the first integration step for using Cosmos-Predict2.5-2B as a baseline/distillation target:

- find an existing local copy if another user already downloaded it;
- otherwise prepare the official repo and identify the minimal robot/action-cond assets needed;
- verify whether the current environment can run the official inference path.

## Paths Prepared

- Official repo clone:
  `/root/nas/junjie/cosmos_predict2_5/repos/cosmos-predict2.5`
- Repo commit:
  `a2c298b Add Cosmos 3 README redirect (#158)`
- Checkpoint target:
  `/root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Predict2.5-2B`

## Local Asset Search

Checked common local Hugging Face/cache/checkpoint locations:

- `/root/.cache/huggingface/hub`
- `/root/nas/hf_home/hub`
- `/root/nas/shuaizhou/hf_home/hub`
- `/root/nas/xicheng/cosmos/checkpoints`
- `/root/nas/xicheng/cosmos_mixed_vla_vlmqa/checkpoints`

Result: no existing local `Cosmos-Predict2.5-2B` cache/checkpoint was found to copy.

Existing related assets:

- `/root/nas/xicheng/cosmos/checkpoints/Cosmos3-Nano`
- `/root/nas/xicheng/cosmos_mixed_vla_vlmqa`, a separate Cosmos3/D0/VLA/action adaptation workspace.

These are useful references but are not a drop-in copy of Predict2.5-2B.

## Hugging Face Model Layout

The official Predict2.5-2B Hugging Face repo is:

- `nvidia/Cosmos-Predict2.5-2B`

Robot/action-cond is a subdirectory inside that repo, not a separate model repo.
The minimal first-pass files identified from HF metadata are:

- `robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt`
- `robot/action-cond/cr1_empty_string_text_embeddings.pt`
- `tokenizer.pth`
- `README.md`

The target directory now contains the minimal robot/action-cond assets:

- `README.md`
- `tokenizer.pth` (508 MB)
- `robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt` (4.3 GB)
- `robot/action-cond/cr1_empty_string_text_embeddings.pt` (103 MB)

The empty-string embedding loads as a `torch.bfloat16` tensor with shape `[1, 512, 100352]`.

The official action-conditioned path also instantiates Reason1/Qwen text encoders even when the sample prompt is `null` and `guidance=0`. A local copy already exists at:

- `/root/nas/junjie/weights/Cosmos-Reason1-7B` (16 GB)

No extra download was needed for Reason1.

## Environment Check

Machine:

- GPU: 2 x NVIDIA H100 80GB HBM3
- Driver: 580.126.20
- CUDA toolkit: 12.8
- Free NAS space at check time: about 5.9T

The existing Any_WAM env has useful packages, but Predict2.5 needs the official CUDA extra. A separate Python 3.10 uv environment was prepared at:

- `/root/nas/junjie/cosmos_predict2_5/envs/predict2_py310`

Installed special CUDA 12.8 wheels from local wheelhouse:

- `flash_attn-2.7.3+cu128.torch27-cp310-cp310-linux_x86_64.whl`
- `natten-0.21.0+cu128.torch27-cp310-cp310-linux_x86_64.whl`
- `transformer_engine-2.2+cu128.torch27-cp310-cp310-linux_x86_64.whl`

Verified imports:

```text
torch 2.7.0+cu128
torchvision 0.22.0+cu128
flash_attn 2.7.3
natten 0.21.0
transformer_engine 2.2+cu128.torch27
cosmos_predict2 1.5.0
cuda available True, devices 2
```

`mediapy` could not find `ffmpeg` by default. The env now has:

- `/root/nas/junjie/cosmos_predict2_5/envs/predict2_py310/bin/ffmpeg`

as a symlink to the `imageio_ffmpeg` bundled binary.

## Local Runtime Patch

The official `checkpoint_db.py` always calls `uvx hf download` for registered HF checkpoints. On this machine that caused slow/stuck downloads for files already present locally.

A small env-gated patch was added in the Cosmos clone:

- `COSMOS_PREDICT25_LOCAL_MODEL_DIR`: return `${DIR}/${filename}` for single-file HF checkpoints when the file exists.
- `COSMOS_PREDICT25_LOCAL_HF_REPOS`: map `repo_id=/local/path` for directory HF checkpoints.

This leaves default behavior unchanged when the env vars are not set.

## Smoke Test

Official sample assets in the git clone were LFS pointers because `git-lfs` is not installed. Only sample `0` was materialized, and a clean one-sample input root was created:

- `/root/nas/junjie/cosmos_predict2_5/smoke_assets/basic/bridge`

Passing command:

```bash
cd /root/nas/junjie/cosmos_predict2_5/repos/cosmos-predict2.5
PATH=/root/nas/junjie/cosmos_predict2_5/envs/predict2_py310/bin:/root/.local/bin:$PATH \
HF_ENDPOINT=https://hf-mirror.com \
HF_HOME=/root/nas/junjie/cosmos_predict2_5/.hf_cache \
COSMOS_PREDICT25_LOCAL_MODEL_DIR=/root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Predict2.5-2B \
COSMOS_PREDICT25_LOCAL_HF_REPOS=nvidia/Cosmos-Reason1-7B=/root/nas/junjie/weights/Cosmos-Reason1-7B \
CUDA_VISIBLE_DEVICES=0 \
TORCHINDUCTOR_COMPILE_THREADS=4 \
/root/nas/junjie/cosmos_predict2_5/envs/predict2_py310/bin/python examples/action_conditioned.py \
  -i assets/action_conditioned/basic/inference_params.json \
  -o /root/nas/junjie/cosmos_predict2_5/outputs/action_conditioned_smoke_single0_ffmpeg \
  --checkpoint-path /root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Predict2.5-2B/robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt \
  --experiment ac_reason_embeddings_rectified_flow_2b_256_320 \
  --input-root /root/nas/junjie/cosmos_predict2_5/smoke_assets/basic/bridge \
  --save-root /root/nas/junjie/cosmos_predict2_5/outputs/action_conditioned_smoke_single0_ffmpeg/generated \
  --num-steps 2 \
  --end 1 \
  --single-chunk True \
  --disable-guardrails
```

Result:

- generated video:
  `/root/nas/junjie/cosmos_predict2_5/outputs/action_conditioned_smoke_single0_ffmpeg/generated/0_single_chunk.mp4`
- size: `54802` bytes
- readable via `mediapy.read_video`
- video shape: `(13, 256, 320, 3)`

No Cosmos smoke process was left running after the test.

## Follow-Up

For Any_WAM integration, the next useful step is not to train yet. First add a thin adapter around the official action-conditioned model that can:

- load the Cosmos Predict2.5 action-cond model from the prepared paths;
- accept Any_WAM/LeRobot observations and actions;
- return predicted frames or latent/video tensors in the format needed by the existing distillation code;
- keep all Cosmos-specific env vars and paths in one config block.
```
