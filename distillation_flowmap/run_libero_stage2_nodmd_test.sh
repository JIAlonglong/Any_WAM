#!/bin/bash
# 测试无 DMD 版本的训练速度
# 用于对比 DMD 的开销
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
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output_libero_optimized_stage2_nodmd}"
export CONFIG_FILE="distillation_flowmap.config_libero_optimized_stage2_nodmd"
export DISTILL_MODE="flashwam"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 离线模式
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Stage 1 checkpoint
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${SCRIPT_DIR}/output_libero_new}"
STAGE1_STEP="${STAGE1_STEP:-8500}"
STAGE1_PATH="${STAGE1_PATH:-${STAGE1_OUTPUT_DIR}/checkpoints/step_${STAGE1_STEP}}"

# 训练参数
NGPU="${NGPU:-1}"
MASTER_PORT="${MASTER_PORT:-29503}"  # 使用不同端口

# 多卡时自动换算 gradient_accumulation_steps
ORIG_ACCUM=16
ACCUM=$((ORIG_ACCUM / NGPU))
[ "$ACCUM" -lt 1 ] && ACCUM=1

ARGS=""
ARGS="$ARGS --resume-from-path $STAGE1_PATH"

echo "=========================================="
echo "LIBERO FlowMap Distillation - Stage 2 (No DMD) - Speed Test"
echo "=========================================="
echo "Teacher:     ${TEACHER_PATH}"
echo "Dataset:     ${DATASET_PATH}"
echo "Output:      ${OUTPUT_DIR}"
echo "Config:      ${CONFIG_FILE}"
echo "GPUs:        ${NGPU}"
echo "Grad accum:  ${ACCUM}"
echo "Resume from: ${STAGE1_PATH}"
echo "DMD:         DISABLED"
echo "=========================================="

# 检查前置条件
if [ ! -d "$TEACHER_PATH" ]; then
    echo "ERROR: Teacher model not found: $TEACHER_PATH"
    exit 1
fi

if [ ! -d "$STAGE1_PATH" ]; then
    echo "ERROR: Stage 1 checkpoint not found: $STAGE1_PATH"
    exit 1
fi

echo "Starting speed test (No DMD)..."
time torchrun \
    --nproc_per_node="${NGPU}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/train.py" \
    --teacher-model-path "$TEACHER_PATH" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --gradient-accumulation-steps "${ACCUM}" \
    $ARGS

echo "Speed test completed!"
