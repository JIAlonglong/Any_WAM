#!/bin/bash
# LIBERO FlowMap 蒸馏启动脚本 —— Stage 2（On-Policy Distillation）
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

# ============================================================
# 路径配置
# ============================================================
export TEACHER_PATH="${TEACHER_PATH:-${PROJECT_ROOT}/checkpoints/lingbot-va-posttrain-libero}"
export DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/training_data/libero-long-lerobot}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output_libero_stage2_opd}"
export CONFIG_FILE="${CONFIG_FILE:-distillation_flowmap.config_libero_optimized_stage2}"
export DISTILL_MODE="${DISTILL_MODE:-flashwam}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ============================================================
# 训练参数
# ============================================================
NGPU="${NGPU:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-${SCRIPT_DIR}/output_libero_stage1_retrain_20260623/checkpoints/step_4000}"

ORIG_ACCUM=${ORIG_ACCUM:-16}
ACCUM=$((ORIG_ACCUM / NGPU))
[ "$ACCUM" -lt 1 ] && ACCUM=1

ARGS=""
[ -n "$RESUME_FROM_PATH" ] && ARGS="$ARGS --resume-from-path $RESUME_FROM_PATH"

echo "=========================================="
echo "LIBERO FlowMap Distillation - Stage 2 (OPD)"
echo "=========================================="
echo "Teacher:     ${TEACHER_PATH}"
echo "Dataset:     ${DATASET_PATH}"
echo "Output:      ${OUTPUT_DIR}"
echo "Config:      ${CONFIG_FILE}"
echo "Resume:      ${RESUME_FROM_PATH}"
echo "GPUs:        ${NGPU}"
echo "Grad accum:  ${ACCUM}"
echo "=========================================="

echo "Starting Stage 2 training..."
torchrun \
    --nproc_per_node="${NGPU}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/train.py" \
    --teacher-model-path "$TEACHER_PATH" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --gradient-accumulation-steps "${ACCUM}" \
    $ARGS

echo "Stage 2 training completed!"
