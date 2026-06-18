#!/bin/bash
# ============================================================
# LIBERO 阶段一评估编排脚本
#
# 用法：
#   bash evaluation/libero/run_eval.sh                    # 评估 step_1000 online_student
#   bash evaluation/libero/run_eval.sh step_2000          # 评估指定 step
#   bash evaluation/libero/run_eval.sh step_1000 target   # 评估 target_student
#   EVAL_MODE=compare bash evaluation/libero/run_eval.sh  # 对比 teacher vs student
#   EVAL_MODE=success bash evaluation/libero/run_eval.sh  # 跑任务成功率
#
# 环境变量：
#   EVAL_MODE:   visualize | compare | success | all (默认 visualize)
#   NUM_STEPS:   推理步数 (默认 2)
#   TEST_NUM:    每任务测试 episode 数 (默认 10，快速验证用)
#   PORT:        WebSocket 端口 (默认 29056)
# ============================================================

set -e

# 使用 OSMesa 软件渲染（无 GPU 显示）
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# Server 用 flashwam（有 diffusers/peft），Client 用 libero（有 robosuite）
SERVER_PYTHON="conda run -n flashwam python"
CLIENT_PYTHON="conda run -n libero python"

STEP="${1:-step_8500}"
VARIANT="${2:-online_student}"
EVAL_MODE="${EVAL_MODE:-visualize}"
NUM_STEPS="${NUM_STEPS:-20}"
ACTION_NUM_STEPS="${ACTION_NUM_STEPS:-20}"
TEST_NUM="${TEST_NUM:-3}"
PORT="${PORT:-29057}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/wan_va:${PROJECT_ROOT}/distillation_flowmap:${PYTHONPATH}"

STUDENT_CKPT="${PROJECT_ROOT}/distillation_flowmap/output_libero_new/checkpoints/${STEP}/${VARIANT}/transformer"
TEACHER_CKPT="${PROJECT_ROOT}/checkpoints/libero/transformer"
SAVE_ROOT="${PROJECT_ROOT}/evaluation/outputs/${STEP}_${VARIANT}"
VIDEO_DIR="${SAVE_ROOT}/videos"
ACTION_DIR="${SAVE_ROOT}/actions"
RESULT_DIR="${SAVE_ROOT}/results"

echo "============================================"
echo "  LIBERO Phase 1 Evaluation"
echo "============================================"
echo "  Step:           ${STEP} / ${VARIANT}"
echo "  Eval mode:      ${EVAL_MODE}"
echo "  Video steps:    ${NUM_STEPS}"
echo "  Action steps:   ${ACTION_NUM_STEPS}"
echo "  Test num:       ${TEST_NUM}"
echo "  Port:           ${PORT}"
echo "  Student:        ${STUDENT_CKPT}"
echo "  Teacher:        ${TEACHER_CKPT}"
echo "============================================"

# 检查 checkpoint
if [ ! -d "$STUDENT_CKPT" ]; then
    echo "ERROR: Student checkpoint not found: $STUDENT_CKPT"
    echo "Available checkpoints:"
    ls "${PROJECT_ROOT}/distillation_flowmap/output_libero_new/checkpoints/"
    exit 1
fi

# ============================================================
# 函数：启动推理服务器
# ============================================================
start_server() {
    local ckpt_path="$1"
    local save_root="$2"
    local num_steps="$3"
    local action_num_steps="$4"

    echo "[Server] Starting with checkpoint: $ckpt_path"
    echo "[Server] Save root: $save_root"
    echo "[Server] Video steps: $num_steps"
    echo "[Server] Action steps: $action_num_steps"

    $SERVER_PYTHON -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port 29062 \
        wan_va/wan_va_server.py \
        --config-name libero \
        --port "$PORT" \
        --checkpoint-path "$ckpt_path" \
        --num-steps "$num_steps" \
        --action-num-steps "$action_num_steps" \
        --save-root "$save_root" &

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
# 模式 1: 视频可视化
# ============================================================
run_visualize() {
    echo ""
    echo "=== Mode: Video Visualization ==="

    # 用 student checkpoint 启动服务器
    start_server "$STUDENT_CKPT" "$VIDEO_DIR" "$NUM_STEPS" "$ACTION_NUM_STEPS"

    # 跑少量 episode 生成视频
    $CLIENT_PYTHON evaluation/libero/client.py \
        --libero-benchmark libero_10 \
        --port "$PORT" \
        --test-num "$TEST_NUM" \
        --task-range 0 3 \
        --out-dir "$VIDEO_DIR"

    stop_server

    echo ""
    echo "=== Videos saved to: $VIDEO_DIR ==="
    find "$VIDEO_DIR" -name "*.mp4" | head -10
    echo "Total videos: $(find "$VIDEO_DIR" -name "*.mp4" | wc -l)"
}

# ============================================================
# 模式 2: 量化精度对比 (student vs teacher)
# ============================================================
run_compare() {
    echo ""
    echo "=== Mode: Quantitative Comparison ==="

    local TEACHER_SAVE="${ACTION_DIR}/teacher"
    local STUDENT_SAVE="${ACTION_DIR}/student"

    # Teacher inference
    echo "[Compare] Running teacher inference..."
    start_server "$TEACHER_CKPT" "$TEACHER_SAVE" "$NUM_STEPS" "$ACTION_NUM_STEPS"

    $CLIENT_PYTHON evaluation/libero/client.py \
        --libero-benchmark libero_10 \
        --port "$PORT" \
        --test-num "$TEST_NUM" \
        --task-range 0 3 \
        --out-dir "${VIDEO_DIR}/teacher"

    stop_server
    sleep 5

    # Student inference
    echo "[Compare] Running student inference..."
    start_server "$STUDENT_CKPT" "$STUDENT_SAVE" "$NUM_STEPS" "$ACTION_NUM_STEPS"

    $CLIENT_PYTHON evaluation/libero/client.py \
        --libero-benchmark libero_10 \
        --port "$PORT" \
        --test-num "$TEST_NUM" \
        --task-range 0 3 \
        --out-dir "${VIDEO_DIR}/student"

    stop_server
    sleep 5

    # Compare actions
    echo "[Compare] Computing metrics..."
    $CLIENT_PYTHON evaluation/libero/compare_actions.py \
        --teacher-dir "$TEACHER_SAVE" \
        --student-dir "$STUDENT_SAVE" \
        --output-file "${RESULT_DIR}/action_comparison.json"
}

# ============================================================
# 模式 3: LIBERO 任务成功率
# ============================================================
run_success() {
    echo ""
    echo "=== Mode: LIBERO Task Success Rate ==="

    start_server "$STUDENT_CKPT" "$ACTION_DIR" "$NUM_STEPS" "$ACTION_NUM_STEPS"

    $CLIENT_PYTHON evaluation/libero/client.py \
        --libero-benchmark libero_10 \
        --port "$PORT" \
        --test-num "$TEST_NUM" \
        --task-range 0 10 \
        --out-dir "${RESULT_DIR}/libero_eval"

    stop_server

    echo ""
    echo "=== Results ==="
    for f in "${RESULT_DIR}"/libero_eval/*.json; do
        if [ -f "$f" ]; then
            echo "$(basename $f): $(cat $f)"
        fi
    done
}

# ============================================================
# 执行
# ============================================================
mkdir -p "$VIDEO_DIR" "$ACTION_DIR" "$RESULT_DIR"

case "$EVAL_MODE" in
    visualize)
        run_visualize
        ;;
    compare)
        run_compare
        ;;
    success)
        run_success
        ;;
    all)
        run_visualize
        run_compare
        run_success
        ;;
    *)
        echo "Unknown EVAL_MODE: $EVAL_MODE"
        echo "Options: visualize | compare | success | all"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  Evaluation complete!"
echo "  Results: $RESULT_DIR"
echo "============================================"
