#!/bin/bash
# Clean LIBERO full-parameter FlowMap run:
#   1. short Stage 1 AnyFlow warmup
#   2. Stage 2 AnyFlow-style continuation with focused rollout step pairs
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f /root/nas/junjie/miniconda3/etc/profile.d/conda.sh ]; then
    source /root/nas/junjie/miniconda3/etc/profile.d/conda.sh
else
    echo "conda not found; set PATH or install conda before running this script" >&2
    exit 1
fi
conda activate "${CONDA_ENV:-/root/nas/junjie/conda_envs/any_wam}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export TEACHER_PATH="${TEACHER_PATH:-${PROJECT_ROOT}/checkpoints/lingbot-va-posttrain-libero}"
export DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/training_data/libero-long-lerobot}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

NGPU="${NGPU:-8}"
MASTER_PORT_STAGE1="${MASTER_PORT_STAGE1:-29501}"
MASTER_PORT_STAGE2="${MASTER_PORT_STAGE2:-29502}"
ORIG_ACCUM="${ORIG_ACCUM:-16}"
ACCUM=$((ORIG_ACCUM / NGPU))
[ "$ACCUM" -lt 1 ] && ACCUM=1

STAGE1_OUTPUT="${STAGE1_OUTPUT:-${SCRIPT_DIR}/output_libero_fullft_stage1_warmup}"
STAGE1_STEPS="${STAGE1_STEPS:-500}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-${SCRIPT_DIR}/output_libero_fullft_stage2_anyflow}"
STAGE2_STEPS="${STAGE2_STEPS:-5000}"
STAGE1_CKPT="${STAGE1_CKPT:-${STAGE1_OUTPUT}/checkpoints/step_${STAGE1_STEPS}}"

echo "=========================================="
echo "LIBERO FullFT Stage1 Warmup -> Stage2 AnyFlow"
echo "=========================================="
echo "Teacher:       ${TEACHER_PATH}"
echo "Dataset:       ${DATASET_PATH}"
echo "GPUs:          ${NGPU}"
echo "Grad accum:    ${ACCUM} (effective batch: $((ACCUM * NGPU)))"
echo "Stage1 output: ${STAGE1_OUTPUT}"
echo "Stage1 steps:  ${STAGE1_STEPS}"
echo "Stage2 output: ${STAGE2_OUTPUT}"
echo "Stage2 steps:  ${STAGE2_STEPS}"
echo "=========================================="

echo "[Stage 1] Full-parameter AnyFlow warmup"
CONFIG_FILE=distillation_flowmap.config_libero_fullfinetune_stage1_warmup \
OUTPUT_DIR="${STAGE1_OUTPUT}" \
MAX_TRAIN_STEPS="${STAGE1_STEPS}" \
torchrun \
    --nproc_per_node="${NGPU}" \
    --master_port="${MASTER_PORT_STAGE1}" \
    "${SCRIPT_DIR}/train.py" \
    --teacher-model-path "${TEACHER_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --output-dir "${STAGE1_OUTPUT}" \
    --gradient-accumulation-steps "${ACCUM}"

if [ ! -d "${STAGE1_CKPT}" ]; then
    echo "ERROR: missing Stage 1 checkpoint: ${STAGE1_CKPT}" >&2
    exit 1
fi

echo "[Stage 2] AnyFlow-style continuation with rollout_step_pairs=[[1,1],[2,1]]"
CONFIG_FILE=distillation_flowmap.config_libero_fullfinetune_stage2_anyflow \
RESUME_FROM_PATH="${STAGE1_CKPT}" \
OUTPUT_DIR="${STAGE2_OUTPUT}" \
MAX_TRAIN_STEPS="${STAGE2_STEPS}" \
torchrun \
    --nproc_per_node="${NGPU}" \
    --master_port="${MASTER_PORT_STAGE2}" \
    "${SCRIPT_DIR}/train.py" \
    --teacher-model-path "${TEACHER_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --output-dir "${STAGE2_OUTPUT}" \
    --resume-from-path "${STAGE1_CKPT}" \
    --gradient-accumulation-steps "${ACCUM}"

echo "Done."
