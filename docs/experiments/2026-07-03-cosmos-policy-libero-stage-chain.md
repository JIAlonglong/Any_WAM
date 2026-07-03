# Cosmos Policy LIBERO Stage Chain Check

Date: 2026-07-03

## Goal

Check whether the downloaded official `nvidia/Cosmos-Policy-LIBERO-Predict2-2B` checkpoint can be used directly as the teacher in the existing Any_WAM `distillation_flowmap` LIBERO Stage 1 and Stage 2 training chain.

## Assets

Cosmos Policy LIBERO checkpoint:

- `/root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B`
- weight file: `Cosmos-Policy-LIBERO-Predict2-2B.pt`
- config/statistics/t5 embeddings are present.

Existing FlowMap LIBERO dataset:

- `/root/nas/junjie/jj/Any_WAM/training_data/libero-long-lerobot`
- symlink to `/root/nas/junjie/data/libero-long-lerobot`
- resolved size: about 4.4 GB

Existing FlowMap LIBERO Stage 1 checkpoint for Stage 2 resume test:

- `/root/nas/junjie/jj/Any_WAM/distillation_flowmap/output_libero_fullft_stage1_liteval_20260627_012949/checkpoints/step_5000`

## Format Comparison

The current FlowMap chain expects a Diffusers/WanVA teacher root with subdirectories such as:

- `transformer/config.json`
- `transformer/diffusion_pytorch_model*.safetensors`
- `vae/config.json`
- `text_encoder/*`

The downloaded Cosmos Policy repo contains only:

- `.gitattributes`
- `Cosmos-Policy-LIBERO-Predict2-2B.pt`
- `README.md`
- `config.json`
- `libero_dataset_statistics.json`
- `libero_t5_embeddings.pkl`

The state-dict namespaces are also different:

- FlowMap/LingBot teacher sample keys: `blocks.0.attn1.to_q.weight`, `action_embedder.weight`, `action_proj_out.weight`
- Cosmos Policy sample keys: `net.blocks.0.self_attn.q_proj.weight`, `net.x_embedder.proj.1.weight`

The Cosmos Policy config is policy-oriented:

- input images: 224x224, views `agentview` and `eye_in_hand`
- proprioception dim: 9
- action output dim: 7, horizon: 16

Current FlowMap LIBERO config is video/action-distillation oriented around the WanVA transformer/vae teacher.

## Stage 1 Smoke

Command summary:

```bash
CONFIG_FILE=distillation_flowmap.config_libero_fullfinetune_stage1_warmup \
MAX_TRAIN_STEPS=1 SAVE_INTERVAL=1 ENABLE_LIGHT_EVAL=0 WANDB_MODE=disabled \
torchrun --nproc_per_node=1 --master_port=29631 distillation_flowmap/train.py \
  --teacher-model-path /root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
  --dataset-path /root/nas/junjie/jj/Any_WAM/training_data/libero-long-lerobot \
  --output-dir /root/nas/junjie/jj/Any_WAM/distillation_flowmap/output_smoke_cosmos_policy_libero_stage1 \
  --batch-size 1 --load-worker 0 --gradient-accumulation-steps 1
```

Result:

```text
FileNotFoundError: Invalid teacher_model_path: expected /root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B/transformer/config.json. For LIBERO use /kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero.
```

## Stage 2 Smoke

Command summary:

```bash
CONFIG_FILE=distillation_flowmap.config_libero_fullfinetune_stage2_anyflow \
MAX_TRAIN_STEPS=1 SAVE_INTERVAL=1 ENABLE_LIGHT_EVAL=0 ENABLE_STAGE1_START_EVAL=0 USE_OPD_AUX=0 WANDB_MODE=disabled \
torchrun --nproc_per_node=1 --master_port=29632 distillation_flowmap/train.py \
  --teacher-model-path /root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
  --dataset-path /root/nas/junjie/jj/Any_WAM/training_data/libero-long-lerobot \
  --output-dir /root/nas/junjie/jj/Any_WAM/distillation_flowmap/output_smoke_cosmos_policy_libero_stage2 \
  --resume-from-path /root/nas/junjie/jj/Any_WAM/distillation_flowmap/output_libero_fullft_stage1_liteval_20260627_012949/checkpoints/step_5000 \
  --batch-size 1 --load-worker 0 --gradient-accumulation-steps 1
```

Result:

```text
FileNotFoundError: Invalid teacher_model_path: expected /root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B/transformer/config.json. For LIBERO use /kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero.
```

## Conclusion

The official Cosmos Policy LIBERO checkpoint cannot be dropped directly into the current Any_WAM FlowMap Stage 1/Stage 2 teacher path.

This is not a data or GPU issue. It is a model-interface mismatch:

- FlowMap Stage 1/Stage 2 expects a WanVA/Diffusers teacher with transformer/vae/text-encoder folders.
- Cosmos Policy is a policy checkpoint state dict with a different config, tensor key namespace, image/proprio input spec, and action-horizon output spec.

To use Cosmos Policy in the Stage 1/Stage 2 workflow, add a separate Cosmos policy adapter path instead of changing the existing teacher loader in place. The adapter needs to:

- load `Cosmos-Policy-LIBERO-Predict2-2B.pt` with the official Cosmos policy architecture;
- convert Any_WAM/LeRobot LIBERO batches into Cosmos Policy inputs (`agentview`, `eye_in_hand`, proprio, text/t5 embeddings);
- expose a teacher API that returns action chunks and any auxiliary future image/proprio predictions needed by the chosen distillation objective;
- keep this path separate from the current WanVA teacher path so existing Stage 1/Stage 2 behavior remains unchanged.
