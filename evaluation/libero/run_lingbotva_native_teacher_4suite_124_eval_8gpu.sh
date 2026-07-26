#!/bin/bash
# Evaluate the native teacher on all 40 standard LIBERO tasks.
# Each matched video/action budget in {1, 2, 4} uses eight workers: one per GPU.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EVAL_SCRIPT="${PROJECT_ROOT}/evaluation/libero/run_eval_new.sh"
MERGE_SCRIPT="${PROJECT_ROOT}/evaluation/libero/merge_lingbotva_4suite_results.py"
SERVER_PYTHON="${SERVER_PYTHON:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"
CLIENT_PYTHON="${CLIENT_PYTHON:-/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MASTER_PORT_BASE=29680
WS_PORT_BASE=29780
EPISODES=50
MODEL_NAME="teacher_native"
CHECKPOINT=""
OUTPUT_ROOT=""
BUDGETS=(1 2 4)
SUITES=(libero_10 libero_10 libero_spatial libero_spatial libero_object libero_object libero_goal libero_goal)
STARTS=(0 5 0 5 0 5 0 5)
ENDS=(5 10 5 10 5 10 5 10)

usage() {
    echo "Usage: $0 --checkpoint TRANSFORMER --output-root DIR [options]"
    echo "Options: --episodes N --gpu-ids 0,...,7 --master-port-base N --ws-port-base N"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        --gpu-ids) GPU_IDS="$2"; shift 2 ;;
        --master-port-base) MASTER_PORT_BASE="$2"; shift 2 ;;
        --ws-port-base) WS_PORT_BASE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "$CHECKPOINT" ] || [ -z "$OUTPUT_ROOT" ]; then
    usage >&2
    exit 2
fi
case "$EPISODES" in ''|*[!0-9]*|0) echo "--episodes must be a positive integer" >&2; exit 2 ;; esac
case "$MASTER_PORT_BASE" in ''|*[!0-9]*|0) echo "--master-port-base must be a positive integer" >&2; exit 2 ;; esac
case "$WS_PORT_BASE" in ''|*[!0-9]*|0) echo "--ws-port-base must be a positive integer" >&2; exit 2 ;; esac
MASTER_PORT_BASE=$((10#$MASTER_PORT_BASE))
WS_PORT_BASE=$((10#$WS_PORT_BASE))
if [ "$MASTER_PORT_BASE" -eq 0 ]; then
    echo "--master-port-base must be a positive integer" >&2
    exit 2
fi
if [ "$WS_PORT_BASE" -eq 0 ]; then
    echo "--ws-port-base must be a positive integer" >&2
    exit 2
fi
if (( MASTER_PORT_BASE <= WS_PORT_BASE + 7 && WS_PORT_BASE <= MASTER_PORT_BASE + 7 )); then
    echo "Master and WebSocket port ranges must not overlap: [${MASTER_PORT_BASE}, $((MASTER_PORT_BASE + 7))] and [${WS_PORT_BASE}, $((WS_PORT_BASE + 7))]" >&2
    exit 2
fi

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
    if [ -z "$gpu" ] || [ -n "${SEEN_GPUS[$gpu]:-}" ]; then
        echo "GPU_IDS must contain exactly 8 unique GPUs; got: $GPU_IDS" >&2
        exit 2
    fi
    SEEN_GPUS[$gpu]=1
done
if [ "${#GPUS[@]}" -ne 8 ]; then
    echo "GPU_IDS must contain exactly 8 unique GPUs; got: $GPU_IDS" >&2
    exit 2
fi

if [ ! -d "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

run_budget() {
    local steps="$1"
    local budget_root="${OUTPUT_ROOT}/${MODEL_NAME}/steps_${steps}"
    local -a pids=()
    local worker suite start end gpu save_root master_port ws_port

    mkdir -p "${budget_root}/workers"
    for worker in 0 1 2 3 4 5 6 7; do
        suite="${SUITES[$worker]}"
        start="${STARTS[$worker]}"
        end="${ENDS[$worker]}"
        gpu="${GPUS[$worker]}"
        save_root="${budget_root}/workers/${suite}_${start}_${end}"
        master_port=$((MASTER_PORT_BASE + worker))
        ws_port=$((WS_PORT_BASE + worker))
        echo "WORKER model=${MODEL_NAME} backend=native_teacher steps=${steps} episodes=${EPISODES} suite=${suite} tasks=${start}:${end} gpu=${gpu} video_steps=${steps} action_steps=${steps} master_port=${master_port} ws_port=${ws_port}"
        if [ "${CHECK_ONLY:-0}" = "1" ]; then
            continue
        fi
        (
            CUDA_VISIBLE_DEVICES="$gpu" \
            SERVER_PYTHON="$SERVER_PYTHON" \
            CLIENT_PYTHON="$CLIENT_PYTHON" \
            SERVER_BACKEND=native_teacher \
            MODEL_NAME=teacher_native \
            STUDENT_CKPT="$CHECKPOINT" \
            TEACHER_CKPT="$CHECKPOINT" \
            LIBERO_BENCHMARK="$suite" \
            EVAL_MODE=success \
            NUM_STEPS="$steps" \
            ACTION_NUM_STEPS="$steps" \
            TEST_NUM="$EPISODES" \
            TASK_START="$start" \
            TASK_END="$end" \
            PORT="$ws_port" \
            MASTER_PORT="$master_port" \
            SAVE_ROOT="$save_root" \
            LATENCY_JSONL="${save_root}/sampler_latency.jsonl" \
            bash "$EVAL_SCRIPT" external teacher_native
        ) >"${save_root}.log" 2>&1 &
        pids+=("$!")
    done

    if [ "${CHECK_ONLY:-0}" = "1" ]; then
        return
    fi
    local failed=0
    for worker in 0 1 2 3 4 5 6 7; do
        if ! wait "${pids[$worker]}"; then
            echo "Worker ${worker} failed; see ${budget_root}/workers/${SUITES[$worker]}_${STARTS[$worker]}_${ENDS[$worker]}.log" >&2
            failed=1
        fi
    done
    if [ "$failed" -ne 0 ]; then
        return 1
    fi
    "$SERVER_PYTHON" "$MERGE_SCRIPT" \
        --input-root "$budget_root" \
        --expected-episodes "$EPISODES" \
        --expected-model "$MODEL_NAME" \
        --expected-video-steps "$steps" \
        --expected-action-steps "$steps" \
        --output "${budget_root}/summary.json"
}

for steps in "${BUDGETS[@]}"; do
    run_budget "$steps"
done

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo "CHECK_ONLY=1: verified 24 workers (40 tasks x 3 matched budgets)."
else
    echo "Completed ${MODEL_NAME}: all 40 tasks at matched 1/2/4 video-action steps."
fi
