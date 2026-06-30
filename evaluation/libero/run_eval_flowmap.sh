#!/bin/bash
# ============================================================
# FlowMap 蒸馏模型视频生成脚本
#
# 用法：
#   bash evaluation/libero/run_eval_flowmap.sh                    # 评估 step_8500 online_student
#   bash evaluation/libero/run_eval_flowmap.sh step_5000          # 评估指定 step
#   bash evaluation/libero/run_eval_flowmap.sh step_8500 target   # 评估 target_student
#   NUM_STEPS=4 bash evaluation/libero/run_eval_flowmap.sh        # 使用 4 步推理
#   TEST_NUM=3 bash evaluation/libero/run_eval_flowmap.sh         # 每任务测试 3 个 episode
#
# 环境变量：
#   OUTPUT_ROOT: checkpoint 输出目录
#   NUM_STEPS:   推理步数 (默认 2)
#   TEST_NUM:    每任务测试 episode 数 (默认 5)
#   PORT:        WebSocket 端口 (默认 29056)
# ============================================================

set -e

CONDA_ENV="${CONDA_ENV:-/root/nas/junjie/conda_envs/any_wam}"
if [ -d "$CONDA_ENV" ]; then
    PYTHON_CMD="conda run -p $CONDA_ENV python"
else
    PYTHON_CMD="conda run -n $CONDA_ENV python"
fi

STEP="${1:-step_8500}"
VARIANT="${2:-online_student}"
NUM_STEPS="${NUM_STEPS:-2}"
TEST_NUM="${TEST_NUM:-5}"
PORT="${PORT:-29056}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/distillation_flowmap/output_libero_stage1_retrain_20260623}"
STUDENT_CKPT="${OUTPUT_ROOT}/checkpoints/${STEP}/${VARIANT}/transformer"
TEACHER_CKPT="${TEACHER_CKPT:-${PROJECT_ROOT}/checkpoints/lingbot-va-posttrain-libero/transformer}"
SAVE_ROOT="${PROJECT_ROOT}/evaluation/outputs/flowmap_${STEP}_${VARIANT}"
VIDEO_DIR="${SAVE_ROOT}/videos"

echo "============================================"
echo "  FlowMap Distillation Video Generation"
echo "============================================"
echo "  Step:       ${STEP} / ${VARIANT}"
echo "  Num steps:  ${NUM_STEPS}"
echo "  Test num:   ${TEST_NUM}"
echo "  Port:       ${PORT}"
echo "  Student:    ${STUDENT_CKPT}"
echo "  Teacher:    ${TEACHER_CKPT}"
echo "  Output:     ${VIDEO_DIR}"
echo "============================================"

# 检查 checkpoint
if [ ! -d "$STUDENT_CKPT" ]; then
    echo "ERROR: Student checkpoint not found: $STUDENT_CKPT"
    echo "Available checkpoints:"
    ls "${OUTPUT_ROOT}/checkpoints/"
    exit 1
fi

# ============================================================
# 函数：启动推理服务器
# ============================================================
start_server() {
    local ckpt_path="$1"
    local save_root="$2"
    local num_steps="$3"

    echo "[Server] Starting with checkpoint: $ckpt_path"
    echo "[Server] Save root: $save_root"
    echo "[Server] Num steps: $num_steps"

    $PYTHON_CMD -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port 29061 \
        wan_va/wan_va_server.py \
        --config-name libero \
        --port "$PORT" \
        --checkpoint-path "$ckpt_path" \
        --num-steps "$num_steps" \
        --save_root "$save_root" &

    SERVER_PID=$!
    echo "[Server] PID: $SERVER_PID"
    echo "[Server] Waiting 60s for model loading..."
    sleep 60
}

# ============================================================
# 函数：停止服务器
# ============================================================
stop_server() {
    if [ -n "$SERVER_PID" ]; then
        echo "[Server] Stopping PID $SERVER_PID"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
}

trap stop_server EXIT

# ============================================================
# 主流程：生成视频
# ============================================================
echo ""
echo "=== Starting Video Generation ==="

# 创建输出目录
mkdir -p "$VIDEO_DIR"

# 启动服务器
start_server "$STUDENT_CKPT" "$VIDEO_DIR" "$NUM_STEPS"

# 运行客户端生成视频
echo "[Client] Starting video generation..."
$PYTHON_CMD evaluation/libero/client.py \
    --libero-benchmark libero_10 \
    --port "$PORT" \
    --test-num "$TEST_NUM" \
    --task-range 0 3 \
    --out-dir "$VIDEO_DIR"

# 停止服务器
stop_server

echo ""
echo "=== Video Generation Complete ==="
echo "Videos saved to: $VIDEO_DIR"
find "$VIDEO_DIR" -name "*.mp4" | head -20
echo "Total videos: $(find "$VIDEO_DIR" -name "*.mp4" | wc -l)"
echo ""
echo "============================================"
echo "  Video generation complete!"
echo "  Output: $VIDEO_DIR"
echo "============================================"
