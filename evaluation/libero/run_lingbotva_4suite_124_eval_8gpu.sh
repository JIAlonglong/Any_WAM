#!/bin/bash
# Evaluate one LingBotVA-compatible transformer on all 40 standard LIBERO tasks.
# For each sampling budget K in {1,2,4}, both video and action use exactly K steps.
# Eight workers split four suites into [0,5) and [5,10), one worker per GPU.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EVAL_SCRIPT="${PROJECT_ROOT}/evaluation/libero/run_eval_new.sh"
MERGE_SCRIPT="${PROJECT_ROOT}/evaluation/libero/merge_lingbotva_4suite_results.py"
SERVER_PYTHON="${SERVER_PYTHON:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"
CLIENT_PYTHON="${CLIENT_PYTHON:-/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MASTER_PORT_BASE=29680
WS_PORT_BASE=29780
EPISODES=10
MODEL_NAME=""
CHECKPOINT=""
OUTPUT_ROOT=""

usage() {
    echo "Usage: $0 --model-name NAME --checkpoint TRANSFORMER --output-root DIR [options]"
    echo "Options: --episodes N --gpu-ids 0,...,7 --master-port-base N --ws-port-base N"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model-name) MODEL_NAME="$2"; shift 2 ;;
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

if [ -z "$MODEL_NAME" ] || [ -z "$CHECKPOINT" ] || [ -z "$OUTPUT_ROOT" ]; then
    usage >&2
    exit 2
fi
if [ ! -d "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi
case "$EPISODES" in ''|*[!0-9]*|0) echo "--episodes must be a positive integer" >&2; exit 2 ;; esac

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

SUITES=(libero_10 libero_10 libero_spatial libero_spatial libero_object libero_object libero_goal libero_goal)
STARTS=(0 5 0 5 0 5 0 5)
ENDS=(5 10 5 10 5 10 5 10)

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
        echo "WORKER steps=${steps} suite=${suite} tasks=${start}:${end} gpu=${gpu} video_steps=${steps} action_steps=${steps} master_port=${master_port} ws_port=${ws_port}"
        if [ "${CHECK_ONLY:-0}" = "1" ]; then
            continue
        fi
        (
            CUDA_VISIBLE_DEVICES="$gpu" \
            SERVER_PYTHON="$SERVER_PYTHON" \
            CLIENT_PYTHON="$CLIENT_PYTHON" \
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
            bash "$EVAL_SCRIPT" "external" "$MODEL_NAME"
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
        --output "${budget_root}/summary.json"
}

for steps in 1 2 4; do
    run_budget "$steps"
done

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo "CHECK_ONLY=1: verified 24 workers (40 tasks x 3 matched budgets)."
else
    echo "Completed ${MODEL_NAME}: all 40 tasks at matched 1/2/4 video-action steps."
fi
