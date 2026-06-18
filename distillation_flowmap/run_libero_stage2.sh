#!/bin/bash
# LIBERO FlowMap 蒸馏启动脚本 —— Stage 2（DMD 对抗训练）
#
# Stage 2 从 Stage 1 checkpoint 恢复，启用 DMD 进行对抗训练。
#
# 用法：
#   # 从 Stage 1 最新 checkpoint 恢复（默认 step_8500）
#   bash distillation_flowmap/run_libero_stage2.sh
#
#   # 指定 Stage 1 checkpoint 步数
#   STAGE1_STEP=8000 bash distillation_flowmap/run_libero_stage2.sh
#
#   # 指定 Stage 1 checkpoint 路径
#   STAGE1_PATH=/path/to/checkpoint bash distillation_flowmap/run_libero_stage2.sh
#
#   # 单卡训练
#   NGPU=1 bash distillation_flowmap/run_libero_stage2.sh
#
#   # 自定义输出目录
#   OUTPUT_DIR=/path/to/output bash distillation_flowmap/run_libero_stage2.sh
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
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output_libero_optimized_stage2}"
export CONFIG_FILE="${CONFIG_FILE:-distillation_flowmap.config_libero_optimized_stage2}"
export DISTILL_MODE="${DISTILL_MODE:-flashwam}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 离线模式：禁止联网下载数据集
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ============================================================
# Stage 1 Checkpoint 配置
# ============================================================
# Stage 1 输出目录（默认使用 output_libero_new）
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${SCRIPT_DIR}/output_libero_new}"

# Stage 1 checkpoint 步数（默认使用最新的 step_8500）
STAGE1_STEP="${STAGE1_STEP:-8500}"

# Stage 1 checkpoint 路径（优先级高于 STAGE1_STEP）
STAGE1_PATH="${STAGE1_PATH:-}"

# 如果没有指定 STAGE1_PATH，使用 STAGE1_STEP 构建路径
if [ -z "$STAGE1_PATH" ]; then
    STAGE1_PATH="${STAGE1_OUTPUT_DIR}/checkpoints/step_${STAGE1_STEP}"
fi

# ============================================================
# 训练参数
# ============================================================
NGPU="${NGPU:-1}"                   # Stage 2 默认单卡
MASTER_PORT="${MASTER_PORT:-29502}" # 使用不同端口避免冲突

# 多卡时自动换算 gradient_accumulation_steps
ORIG_ACCUM=16
ACCUM=$((ORIG_ACCUM / NGPU))
[ "$ACCUM" -lt 1 ] && ACCUM=1

# ============================================================
# 构建命令行参数
# ============================================================
ARGS=""
ARGS="$ARGS --resume-from-path $STAGE1_PATH"

# ============================================================
# 打印配置信息
# ============================================================
echo "=========================================="
echo "LIBERO FlowMap Distillation - Stage 2 (DMD)"
echo "=========================================="
echo "Teacher:     ${TEACHER_PATH}"
echo "Dataset:     ${DATASET_PATH}"
echo "Output:      ${OUTPUT_DIR}"
echo "Config:      ${CONFIG_FILE}"
echo "Distill mode: ${DISTILL_MODE}"
echo "GPUs:        ${NGPU}"
echo "Grad accum:  ${ACCUM} (effective batch: $((1 * ACCUM * NGPU)))"
echo "Master port: ${MASTER_PORT}"
echo "--- Stage 1 Checkpoint ---"
echo "Stage 1 dir: ${STAGE1_OUTPUT_DIR}"
echo "Resume from: ${STAGE1_PATH}"
echo "--- DMD Config ---"
echo "DMD enabled: True"
echo "DMD weight:  0.1"
echo "DMD warmup:  0 steps"
echo "DMD disc warmup: 200 steps"
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

# 检查 Stage 1 checkpoint
if [ ! -d "$STAGE1_PATH" ]; then
    echo "ERROR: Stage 1 checkpoint not found: $STAGE1_PATH"
    echo "Available checkpoints:"
    ls -d "${STAGE1_OUTPUT_DIR}"/checkpoints/step_* 2>/dev/null | head -5
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
echo "Starting Stage 2 training with DMD..."
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
echo "Checkpoints saved to: $OUTPUT_DIR/checkpoints"
