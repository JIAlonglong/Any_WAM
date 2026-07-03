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

The target directory currently contains only:

- `README.md`

## Current Blocker

Downloading the robot/action-cond checkpoint failed with:

```text
403 Cannot access gated repo
Access to model nvidia/Cosmos-Predict2.5-2B is restricted and the current token is not authorized.
```

The root Hugging Face token is present and `whoami` reports the account as `JIAlonglong`, but that account does not currently have access to `nvidia/Cosmos-Predict2.5-2B`.

Required action before downloading:

1. Accept/request access for `nvidia/Cosmos-Predict2.5-2B` on Hugging Face for the token account, or provide/use a token that already has access.
2. Re-run the selective snapshot download with `HF_ENDPOINT=https://hf-mirror.com`.

## Environment Check

Machine:

- GPU: 2 x NVIDIA H100 80GB HBM3
- Driver: 580.126.20
- CUDA toolkit: 12.8
- Free NAS space at check time: about 5.9T

The existing Any_WAM env has `torch`, `transformers`, `diffusers`, and `huggingface_hub`, but it is not enough for Cosmos-Predict2.5:

```text
RuntimeError CUDA extra not installed. Please run 'uv sync --extra=<cuda_name>'
```

Predict2.5 repo supports CUDA 12.8 via:

```bash
uv sync --extra=cu128
```

The older `/root/nas/xicheng/cosmos_mixed_vla_vlmqa` workspace has local CUDA 12.8 wheel references and can be used as a setup reference, but it targets Cosmos3/cosmos-framework rather than Predict2.5 directly.

## Next Command After HF Access Is Fixed

```bash
cd /root/nas/junjie/cosmos_predict2_5/repos/cosmos-predict2.5
HF_ENDPOINT=https://hf-mirror.com /root/nas/junjie/conda_envs/any_wam/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="nvidia/Cosmos-Predict2.5-2B",
    local_dir="/root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Predict2.5-2B",
    allow_patterns=[
        "robot/action-cond/*",
        "tokenizer.pth",
        "README.md",
    ],
    endpoint="https://hf-mirror.com",
    max_workers=4,
)
PY
```

After weights are present, set up the Predict2.5 CUDA 12.8 environment and run the official single-GPU action-cond example:

```bash
python examples/action_conditioned.py \
  -i assets/action_conditioned/basic/inference_params.json \
  -o outputs/action_conditioned/basic \
  --config-file cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  --checkpoint-path /root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Predict2.5-2B/robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt \
  --experiment ac_reason_embeddings_rectified_flow_2b_256_320
```
