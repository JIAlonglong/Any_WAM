#!/bin/bash
# ============================================================
# FlowMap 蒸馏模型视频生成脚本
#
# 用法：
#   bash evaluation/libero/run_generate_videos.sh
#   bash evaluation/libero/run_generate_videos.sh step_5000
#   bash evaluation/libero/run_generate_videos.sh step_8500 online_student 2 3
# ============================================================

set -e

STEP="${1:-step_8500}"
VARIANT="${2:-online_student}"
NUM_STEPS="${3:-2}"
NUM_TASKS="${4:-3}"
NUM_EPISODES="${5:-2}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHECKPOINT_PATH="${PROJECT_ROOT}/distillation_flowmap/output_libero_new/checkpoints/${STEP}/${VARIANT}/transformer"
OUTPUT_DIR="${PROJECT_ROOT}/evaluation/outputs/flowmap_${STEP}_${VARIANT}"

echo "============================================"
echo "  FlowMap Video Generation"
echo "============================================"
echo "  Step:           ${STEP} / ${VARIANT}"
echo "  Num steps:      ${NUM_STEPS}"
echo "  Num tasks:      ${NUM_TASKS}"
echo "  Num episodes:   ${NUM_EPISODES}"
echo "  Checkpoint:     ${CHECKPOINT_PATH}"
echo "  Output:         ${OUTPUT_DIR}"
echo "============================================"

# 检查 checkpoint
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    echo "Available checkpoints:"
    ls "${PROJECT_ROOT}/distillation_flowmap/output_libero_new/checkpoints/"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 设置 Python 路径
export PYTHONPATH="${PROJECT_ROOT}/wan_va:${PROJECT_ROOT}/distillation_flowmap:${PYTHONPATH}"

# 运行视频生成
echo "Starting video generation..."
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python "${PROJECT_ROOT}/evaluation/libero/generate_videos_flowmap.py" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --num-steps "$NUM_STEPS" \
    --output-dir "$OUTPUT_DIR" \
    --num-tasks "$NUM_TASKS" \
    --num-episodes "$NUM_EPISODES"

echo ""
echo "============================================"
echo "  Video generation complete!"
echo "  Output: $OUTPUT_DIR"
echo "============================================"
