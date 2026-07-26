#!/bin/bash
# ============================================================
# LIBERO 真实环境评估编排脚本
#
# 用法：
#   bash evaluation/libero/run_eval_new.sh
#   bash evaluation/libero/run_eval_new.sh step_5000
#   bash evaluation/libero/run_eval_new.sh step_5000 target_student
#   EVAL_MODE=compare bash evaluation/libero/run_eval_new.sh
#   EVAL_MODE=success TEST_NUM=50 bash evaluation/libero/run_eval_new.sh
#
# 环境变量：
#   OUTPUT_ROOT:      distillation 输出目录，默认 stage2 anyflow 输出
#   STUDENT_CKPT:     任意待测 transformer 路径；设置后不再按 OUTPUT_ROOT/STEP/VARIANT 推导
#   TEACHER_CKPT:     teacher transformer 路径，默认 /kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero/transformer
#   WAN22_PRETRAINED_PATH: VAE/tokenizer/text-encoder/base-transformer 根目录
#   LIBERO_BENCHMARK: libero_10 | libero_spatial | libero_object | libero_goal
#   EVAL_MODE:        visualize | compare | success | all，默认 visualize
#   NUM_STEPS:        视频推理步数，默认 20
#   ACTION_NUM_STEPS: action 推理步数，默认 50
#   TEST_NUM:         每任务 episode 数，默认 3，正式成功率建议 50
#   TASK_START/END:   任务范围 [start, end)，默认 visualize/compare 跑 0..3，success 跑 0..10
#   PORT:             WebSocket 端口，默认 29057
#   CHECK_ONLY=1:     只检查路径和参数，不启动模型
#
# 说明：
#   distillation_flowmap/rollout_eval_stage2.py 是离线 teacher/student rollout
#   指标，不会也不应该生成真正的 LIBERO 环境视频。真正的视频和成功率
#   评估走本脚本：wan_va_server.py + evaluation/libero/client.py。
# ============================================================

set -e

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# Server 用 flashwam（有 diffusers/peft），Client 用 libero（有 robosuite）
SERVER_PYTHON="${SERVER_PYTHON:-conda run -n flashwam python}"
CLIENT_PYTHON="${CLIENT_PYTHON:-conda run -n libero python}"

STEP="${1:-}"
VARIANT="${2:-online_student}"
EVAL_MODE="${EVAL_MODE:-visualize}"
NUM_STEPS="${NUM_STEPS:-20}"
ACTION_NUM_STEPS="${ACTION_NUM_STEPS:-50}"
TEST_NUM="${TEST_NUM:-3}"
PORT="${PORT:-29057}"
TASK_START="${TASK_START:-0}"
LIBERO_BENCHMARK="${LIBERO_BENCHMARK:-libero_10}"

case "$LIBERO_BENCHMARK" in
    libero_10|libero_spatial|libero_object|libero_goal)
        ;;
    *)
        echo "Unsupported LIBERO_BENCHMARK: ${LIBERO_BENCHMARK}" >&2
        exit 1
        ;;
esac

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEFAULT_OUTPUT_ROOT="${PROJECT_ROOT}/distillation_flowmap/output_libero_fullft_stage2_anyflow"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/wan_va:${PROJECT_ROOT}/distillation_flowmap:${PYTHONPATH}"

latest_step() {
    local best=""
    local best_num=-1
    local d base num
    for d in "${OUTPUT_ROOT}"/checkpoints/step_*; do
        [ -d "$d" ] || continue
        base="${d##*/}"
        num="${base#step_}"
        case "$num" in
            ''|*[!0-9]*) continue ;;
        esac
        if [ "$num" -gt "$best_num" ]; then
            best_num="$num"
            best="$base"
        fi
    done
    printf '%s\n' "$best"
}

if [ -z "$STEP" ] && [ -z "${STUDENT_CKPT:-}" ]; then
    STEP="$(latest_step)"
fi
if { [ -z "$STEP" ] || [ "$STEP" = "step_" ]; } && [ -z "${STUDENT_CKPT:-}" ]; then
    echo "ERROR: Could not infer STEP because no checkpoints were found under ${OUTPUT_ROOT}/checkpoints"
    echo "Run training first, or set OUTPUT_ROOT to an existing distillation output dir, or pass a step explicitly."
    exit 1
fi

STUDENT_CKPT="${STUDENT_CKPT:-${OUTPUT_ROOT}/checkpoints/${STEP}/${VARIANT}/transformer}"
TEACHER_CKPT="${TEACHER_CKPT:-/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero/transformer}"
WAN22_PRETRAINED_PATH="${WAN22_PRETRAINED_PATH:-/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero}"
export WAN22_PRETRAINED_PATH
SAVE_ROOT="${SAVE_ROOT:-${PROJECT_ROOT}/evaluation/outputs/libero_env_${STEP}_${VARIANT}}"
MODEL_NAME="${MODEL_NAME:-${VARIANT}}"
SERVER_BACKEND="${SERVER_BACKEND:-flowmap}"
case "$SERVER_BACKEND" in
    flowmap)
        SERVER_ENTRYPOINT="wan_va/wan_va_server.py"
        ;;
    native_teacher)
        SERVER_ENTRYPOINT="wan_va/wan_va_native_teacher_server.py"
        [ "$MODEL_NAME" = "teacher_native" ] || {
            echo "native_teacher requires MODEL_NAME=teacher_native" >&2
            exit 2
        }
        ;;
    *)
        echo "Unsupported SERVER_BACKEND: $SERVER_BACKEND" >&2
        exit 2
        ;;
esac
LATENCY_JSONL="${LATENCY_JSONL:-${SAVE_ROOT}/sampler_latency.jsonl}"
VIDEO_DIR="${SAVE_ROOT}/videos"
ACTION_DIR="${SAVE_ROOT}/actions"
RESULT_DIR="${SAVE_ROOT}/results"

if [ -z "${TASK_END:-}" ]; then
    if [ "$EVAL_MODE" = "success" ] || [ "$EVAL_MODE" = "all" ]; then
        TASK_END=10
    else
        TASK_END=3
    fi
fi

MASTER_PORT="${MASTER_PORT:-29062}"

log_header() {
    echo "============================================"
    echo "  LIBERO Environment Evaluation"
    echo "============================================"
    echo "  Step:           ${STEP} / ${VARIANT}"
    echo "  Output root:    ${OUTPUT_ROOT}"
    echo "  Eval mode:      ${EVAL_MODE}"
    echo "  Benchmark:      ${LIBERO_BENCHMARK}"
    echo "  Video steps:    ${NUM_STEPS}"
    echo "  Action steps:   ${ACTION_NUM_STEPS}"
    echo "  Visible GPUs:   ${CUDA_VISIBLE_DEVICES:-all}"
    echo "  Test num:       ${TEST_NUM}"
    echo "  Task range:     ${TASK_START}..${TASK_END}"
    echo "  Port:           ${PORT}"
    echo "  Master port:    ${MASTER_PORT}"
    echo "  Student:        ${STUDENT_CKPT}"
    echo "  Teacher:        ${TEACHER_CKPT}"
    echo "  Base model:     ${WAN22_PRETRAINED_PATH}"
    echo "  Save root:      ${SAVE_ROOT}"
    echo "  Model name:     ${MODEL_NAME}"
    echo "  Server backend: ${SERVER_BACKEND}"
    echo "  Server entry:   ${SERVER_ENTRYPOINT}"
    echo "  Latency JSONL:  ${LATENCY_JSONL}"
    echo "============================================"
}

check_inputs() {
    if [ ! -d "$STUDENT_CKPT" ]; then
        echo "ERROR: Student checkpoint not found: $STUDENT_CKPT"
        echo "Available checkpoints:"
        ls "${OUTPUT_ROOT}/checkpoints/" 2>/dev/null || true
        exit 1
    fi
    if [ ! -d "$TEACHER_CKPT" ]; then
        echo "ERROR: Teacher checkpoint not found: $TEACHER_CKPT"
        exit 1
    fi
    for component in transformer vae tokenizer text_encoder; do
        if [ ! -d "${WAN22_PRETRAINED_PATH}/${component}" ]; then
            echo "ERROR: Base model component not found: ${WAN22_PRETRAINED_PATH}/${component}"
            exit 1
        fi
    done
}

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
        --master_port "$MASTER_PORT" \
        "$SERVER_ENTRYPOINT" \
        --config-name libero \
        --port "$PORT" \
        --checkpoint-path "$ckpt_path" \
        --num-steps "$num_steps" \
        --action-num-steps "$action_num_steps" \
        --model-name "$MODEL_NAME" \
        --latency-jsonl "$LATENCY_JSONL" \
        --save-root "$save_root" &

    SERVER_PID=$!
    echo "[Server] PID: $SERVER_PID"
    echo "[Server] Waiting ${SERVER_WAIT_SECONDS:-60}s for model loading..."
    sleep "${SERVER_WAIT_SECONDS:-60}"
}

stop_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo "[Server] Stopping PID $SERVER_PID"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}

trap stop_server EXIT

run_client() {
    local out_dir="$1"
    $CLIENT_PYTHON evaluation/libero/client.py \
        --libero-benchmark "$LIBERO_BENCHMARK" \
        --port "$PORT" \
        --test-num "$TEST_NUM" \
        --task-range "$TASK_START" "$TASK_END" \
        --out-dir "$out_dir"
}

run_visualize() {
    echo ""
    echo "=== Mode: Video Visualization ==="
    start_server "$STUDENT_CKPT" "$VIDEO_DIR" "$NUM_STEPS" "$ACTION_NUM_STEPS"
    run_client "$VIDEO_DIR"
    stop_server

    echo ""
    echo "=== Videos saved to: $VIDEO_DIR ==="
    find "$VIDEO_DIR" -name "*.mp4" | head -10
    echo "Total videos: $(find "$VIDEO_DIR" -name "*.mp4" | wc -l)"
}

run_compare() {
    echo ""
    echo "=== Mode: Quantitative Action Comparison ==="

    local teacher_save="${ACTION_DIR}/teacher"
    local student_save="${ACTION_DIR}/student"

    echo "[Compare] Running teacher inference..."
    start_server "$TEACHER_CKPT" "$teacher_save" "$NUM_STEPS" "$ACTION_NUM_STEPS"
    run_client "${VIDEO_DIR}/teacher"
    stop_server
    sleep 5

    echo "[Compare] Running student inference..."
    start_server "$STUDENT_CKPT" "$student_save" "$NUM_STEPS" "$ACTION_NUM_STEPS"
    run_client "${VIDEO_DIR}/student"
    stop_server
    sleep 5

    echo "[Compare] Computing metrics..."
    $SERVER_PYTHON evaluation/libero/compare_actions.py \
        --teacher-dir "$teacher_save" \
        --student-dir "$student_save" \
        --output-file "${RESULT_DIR}/action_comparison.json"
}

run_success() {
    echo ""
    echo "=== Mode: LIBERO Task Success Rate ==="
    start_server "$STUDENT_CKPT" "$ACTION_DIR" "$NUM_STEPS" "$ACTION_NUM_STEPS"
    run_client "${RESULT_DIR}/libero_eval"
    stop_server

    echo ""
    echo "=== Results ==="
    for f in "${RESULT_DIR}"/libero_eval/*.json; do
        if [ -f "$f" ]; then
            echo "$(basename "$f"): $(cat "$f")"
        fi
    done
}

mkdir -p "$VIDEO_DIR" "$ACTION_DIR" "$RESULT_DIR"
log_header
check_inputs

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo "CHECK_ONLY=1: inputs look valid; not starting server."
    exit 0
fi

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
echo "  Evaluation complete"
echo "  Results: $RESULT_DIR"
echo "============================================"
