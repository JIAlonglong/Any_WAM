#!/bin/bash
# LIBERO FlowMap 蒸馏启动脚本（默认 4 卡）
#
# 用法：
#   # 4 卡训练（默认）
#   bash distillation_flowmap/run_libero.sh
#
#   # 单卡训练
#   NGPU=1 bash distillation_flowmap/run_libero.sh
#
#   # 从检查点恢复
#   RESUME_FROM_STEP=1000 bash distillation_flowmap/run_libero.sh
#
#   # 自定义输出目录
#   OUTPUT_DIR=/path/to/output bash distillation_flowmap/run_libero.sh
set -euo pipefail

# 激活 flashwam conda 环境
eval "$(conda shell.bash hook)"
conda activate flashwam

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================
# 路径配置
# ============================================================
export TEACHER_PATH="${TEACHER_PATH:-${PROJECT_ROOT}/checkpoints/libero}"
export DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/training_data/libero-long-lerobot}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output_libero_new}"
export CONFIG_FILE="${CONFIG_FILE:-distillation_flowmap.config_libero}"
export DISTILL_MODE="${DISTILL_MODE:-flashwam}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 离线模式：禁止联网下载数据集
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 如果指定了优化配置，使用优化版（单卡训练优化）
if [ "${OPTIMIZED:-0}" = "1" ]; then
    export CONFIG_FILE="distillation_flowmap.config_libero_optimized"
    echo "Using optimized config for single-GPU training"
fi

# ============================================================
# 训练参数
# ============================================================
NGPU="${NGPU:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:-}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

# 多卡时自动换算 gradient_accumulation_steps，保持等效 batch size 不变
# 原始配置：batch_size=1, gradient_accumulation_steps=4, 等效 batch=16
# num_frames=64, lora_rank=128, lora_alpha=64
# 多卡等效 batch = batch_size × ACCUM × NGPU
# NGPU=4 时: 1 × 4 × 4 = 16
ORIG_ACCUM=${ORIG_ACCUM:-4}
ACCUM=$((ORIG_ACCUM / NGPU))
[ "$ACCUM" -lt 1 ] && ACCUM=1

# ============================================================
# 构建命令行参数
# ============================================================
ARGS=""
[ -n "$RESUME_FROM_STEP" ] && ARGS="$ARGS --resume-from-step $RESUME_FROM_STEP"
[ -n "$RESUME_FROM_PATH" ] && ARGS="$ARGS --resume-from-path $RESUME_FROM_PATH"

# ============================================================
# 打印配置信息
# ============================================================
echo "=========================================="
echo "LIBERO FlowMap Distillation"
echo "=========================================="
echo "Teacher:     ${TEACHER_PATH}"
echo "Dataset:     ${DATASET_PATH}"
echo "Output:      ${OUTPUT_DIR}"
echo "Config:      ${CONFIG_FILE}"
echo "Distill mode: ${DISTILL_MODE}"
echo "GPUs:        ${NGPU}"
echo "Grad accum:  ${ACCUM} (effective batch: $((4 * ACCUM * NGPU)))"
echo "Master port: ${MASTER_PORT}"
if [ -n "$RESUME_FROM_STEP" ]; then
    echo "Resume step: ${RESUME_FROM_STEP}"
fi
if [ -n "$RESUME_FROM_PATH" ]; then
    echo "Resume path: ${RESUME_FROM_PATH}"
fi
echo "=========================================="

# ============================================================
# 检查前置条件
# ============================================================
# 检查教师模型
if [ ! -d "$TEACHER_PATH" ]; then
    echo "ERROR: Teacher model not found: $TEACHER_PATH"
    exit 1
fi

# 检查数据集
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found: $DATASET_PATH"
    exit 1
fi

# 检查 empty_emb.pt
if [ ! -f "$DATASET_PATH/empty_emb.pt" ]; then
    echo "WARNING: empty_emb.pt not found in dataset directory"
    echo "Copying from robotwin_hf..."
    cp "${PROJECT_ROOT}/training_data/robotwin_hf/empty_emb.pt" "$DATASET_PATH/empty_emb.pt"
    echo "Done."
fi

# ============================================================
# 启动训练
# ============================================================
echo "Starting training..."
torchrun \
    --nproc_per_node="${NGPU}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/train.py" \
    --teacher-model-path "$TEACHER_PATH" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --gradient-accumulation-steps "${ACCUM}" \
    $ARGS

echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR/checkpoints"
