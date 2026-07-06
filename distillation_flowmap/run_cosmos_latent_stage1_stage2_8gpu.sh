#!/usr/bin/env bash
set -euo pipefail

cd /root/nas/junjie/jj/Any_WAM

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
NGPU="${NGPU:-8}"
ACCUM="${ACCUM:-2}"
STAGE1_STEPS="${STAGE1_STEPS:-5000}"
STAGE2_STEPS="${STAGE2_STEPS:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
MASTER_PORT_STAGE1="${MASTER_PORT_STAGE1:-29641}"
MASTER_PORT_STAGE2="${MASTER_PORT_STAGE2:-29642}"

SCRIPT_DIR="/root/nas/junjie/jj/Any_WAM/distillation_flowmap"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

STAGE1_OUTPUT="${STAGE1_OUTPUT:-${SCRIPT_DIR}/output_libero_cosmos_policy_stage1_cosmos_latent_cdiff_8gpu_${RUN_ID}}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-${SCRIPT_DIR}/output_libero_cosmos_policy_stage2_cosmos_latent_cdiff_8gpu_${RUN_ID}}"
STAGE1_CKPT="${STAGE1_OUTPUT}/checkpoints/step_${STAGE1_STEPS}"
STAGE1_LOG="${LOG_DIR}/cosmos_latent_stage1_${STAGE1_STEPS}_8gpu_${RUN_ID}.log"
STAGE2_LOG="${LOG_DIR}/cosmos_latent_stage2_${STAGE2_STEPS}_8gpu_${RUN_ID}.log"

export PYTHONPATH="/root/nas/junjie/jj/Any_WAM:/root/nas/junjie/jj/Any_WAM/distillation_flowmap:${PYTHONPATH:-}"
export COSMOS_POLICY_PATH="${COSMOS_POLICY_PATH:-/root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
export STUDENT_BASE_MODEL_PATH="${STUDENT_BASE_MODEL_PATH:-/root/nas/junjie/jj/Any_WAM/checkpoints/lingbot-va-posttrain-libero}"
export DATASET_PATH="${DATASET_PATH:-/root/nas/junjie/jj/Any_WAM/training_data/libero-long-lerobot}"
export COSMOS_POLICY_USE_RAW_INFERENCE="${COSMOS_POLICY_USE_RAW_INFERENCE:-1}"
export COSMOS_POLICY_INFERENCE_MODE="${COSMOS_POLICY_INFERENCE_MODE:-subprocess}"
export COSMOS_POLICY_PYTHON="${COSMOS_POLICY_PYTHON:-/root/nas/junjie/cosmos_predict2_5/envs/predict2_py310/bin/python}"
export COSMOS_PREDICT2_REPO="${COSMOS_PREDICT2_REPO:-/root/nas/junjie/cosmos_predict2_5/repos/cosmos-predict2.5}"
export COSMOS_POLICY_EXTRA_PYTHONPATH="${COSMOS_POLICY_EXTRA_PYTHONPATH:-/root/nas/junjie/conda_envs/any_wam/lib/python3.10/site-packages}"
export COSMOS_PREDICT25_LOCAL_MODEL_DIR="${COSMOS_PREDICT25_LOCAL_MODEL_DIR:-/root/nas/junjie/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World}"
export COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION="${COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION:-5}"
export COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES="${COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export COSMOS_LATENT_TARGET_MODE="${COSMOS_LATENT_TARGET_MODE:-hybrid_cdiff}"
export COSMOS_LATENT_CDIFF_INTERVAL="${COSMOS_LATENT_CDIFF_INTERVAL:-4}"
export COSMOS_LATENT_CENTER_VELOCITY_MODE="${COSMOS_LATENT_CENTER_VELOCITY_MODE:-symmetric_average}"
export SKIP_TARGET_STUDENT_FOR_COSMOS_LATENT="${SKIP_TARGET_STUDENT_FOR_COSMOS_LATENT:-1}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
export ENABLE_WANDB="${ENABLE_WANDB:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

echo "=========================================="
echo "Cosmos latent Stage1 -> Stage2"
echo "Run ID:       ${RUN_ID}"
echo "GPUs:         ${NGPU}"
echo "Accum:        ${ACCUM}"
echo "Stage1 steps: ${STAGE1_STEPS}"
echo "Stage2 steps: ${STAGE2_STEPS}"
echo "Stage1 out:   ${STAGE1_OUTPUT}"
echo "Stage2 out:   ${STAGE2_OUTPUT}"
echo "Stage1 log:   ${STAGE1_LOG}"
echo "Stage2 log:   ${STAGE2_LOG}"
echo "=========================================="

echo "[Stage 1] start $(date)"
(
  set -o pipefail
  CONFIG_FILE=distillation_flowmap.config_libero_cosmos_policy_stage1_cosmos_latent_cdiff \
  OUTPUT_DIR="${STAGE1_OUTPUT}" \
  MAX_TRAIN_STEPS="${STAGE1_STEPS}" \
  SAVE_INTERVAL="${SAVE_INTERVAL}" \
  /root/nas/junjie/conda_envs/any_wam/bin/torchrun \
    --nproc_per_node="${NGPU}" \
    --master_port="${MASTER_PORT_STAGE1}" \
    "${SCRIPT_DIR}/train.py" \
    --teacher-model-path "${COSMOS_POLICY_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --output-dir "${STAGE1_OUTPUT}" \
    --gradient-accumulation-steps "${ACCUM}" \
    2>&1 | tee "${STAGE1_LOG}"
)
echo "[Stage 1] done $(date)"

if [ ! -d "${STAGE1_CKPT}/online_student/transformer" ]; then
  echo "ERROR: missing Stage 1 checkpoint: ${STAGE1_CKPT}/online_student/transformer" >&2
  exit 1
fi

echo "[Stage 2] start $(date)"
(
  set -o pipefail
  CONFIG_FILE=distillation_flowmap.config_libero_cosmos_policy_stage2_cosmos_latent_cdiff \
  RESUME_FROM_PATH="${STAGE1_CKPT}" \
  OUTPUT_DIR="${STAGE2_OUTPUT}" \
  MAX_TRAIN_STEPS="${STAGE2_STEPS}" \
  SAVE_INTERVAL="${SAVE_INTERVAL}" \
  USE_OPD_AUX=1 \
  OPD_TEACHER_TARGET_MODE=cosmos_latent_student_state \
  OPD_AUX_ACTION=0 \
  /root/nas/junjie/conda_envs/any_wam/bin/torchrun \
    --nproc_per_node="${NGPU}" \
    --master_port="${MASTER_PORT_STAGE2}" \
    "${SCRIPT_DIR}/train.py" \
    --teacher-model-path "${COSMOS_POLICY_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --output-dir "${STAGE2_OUTPUT}" \
    --resume-from-path "${STAGE1_CKPT}" \
    --gradient-accumulation-steps "${ACCUM}" \
    2>&1 | tee "${STAGE2_LOG}"
)
echo "[Stage 2] done $(date)"
echo "All done."
