#!/bin/bash
# Dynamically balance all 40 standard LIBERO tasks across one to eight
# native-teacher servers. Each server stays alive for one matched video/action
# budget and claims one task at a time, so slow suites cannot strand idle GPUs.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_PATH="${PROJECT_ROOT}/evaluation/libero/$(basename "$0")"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/wan_va:${PROJECT_ROOT}/distillation_flowmap:${PYTHONPATH:-}"
SERVER_ENTRYPOINT="${PROJECT_ROOT}/wan_va/wan_va_native_teacher_server.py"
CLIENT_SCRIPT="${PROJECT_ROOT}/evaluation/libero/client.py"
MERGE_SCRIPT="${PROJECT_ROOT}/evaluation/libero/merge_lingbotva_4suite_results.py"
SERVER_PYTHON="${SERVER_PYTHON:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"
CLIENT_PYTHON="${CLIENT_PYTHON:-/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python}"
READY_PYTHON="${READY_PYTHON:-python3}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
REPLICAS_PER_GPU=1
MASTER_PORT_BASE=32680
WS_PORT_BASE=32780
EPISODES=50
MODEL_NAME="teacher_native"
CHECKPOINT=""
OUTPUT_ROOT=""
BUDGETS_CSV="1,2,4"
SERVER_READY_TIMEOUT_SECONDS="${SERVER_READY_TIMEOUT_SECONDS:-600}"
WORKER_STOP_TIMEOUT_SECONDS="${WORKER_STOP_TIMEOUT_SECONDS:-20}"

# Spatial tasks are observed to be slow, so they receive the earliest claims.
SUITE_ORDER=(libero_spatial libero_goal libero_object libero_10)
TASK_SPECS=()
ACTIVE_WORKER_PIDS=()
TERMINATING=0

usage() {
    echo "Usage: $0 --checkpoint TRANSFORMER --output-root DIR [options]"
    echo "Options: --episodes N --gpu-ids 0,...,N (1-8 GPUs) --replicas-per-gpu 1|2 --master-port-base N --ws-port-base N --budgets 1,2,4"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        --gpu-ids) GPU_IDS="$2"; shift 2 ;;
        --replicas-per-gpu) REPLICAS_PER_GPU="$2"; shift 2 ;;
        --master-port-base) MASTER_PORT_BASE="$2"; shift 2 ;;
        --ws-port-base) WS_PORT_BASE="$2"; shift 2 ;;
        --budgets) BUDGETS_CSV="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

require_positive_integer() {
    local flag="$1"
    local value="$2"
    case "$value" in
        ''|*[!0-9]*|0)
            echo "${flag} must be a positive integer" >&2
            return 1
            ;;
    esac
}

if [ -z "$CHECKPOINT" ] || [ -z "$OUTPUT_ROOT" ]; then
    usage >&2
    exit 2
fi
require_positive_integer "--episodes" "$EPISODES" || exit 2
require_positive_integer "--master-port-base" "$MASTER_PORT_BASE" || exit 2
require_positive_integer "--ws-port-base" "$WS_PORT_BASE" || exit 2
require_positive_integer "SERVER_READY_TIMEOUT_SECONDS" "$SERVER_READY_TIMEOUT_SECONDS" || exit 2
require_positive_integer "WORKER_STOP_TIMEOUT_SECONDS" "$WORKER_STOP_TIMEOUT_SECONDS" || exit 2

MASTER_PORT_BASE=$((10#$MASTER_PORT_BASE))
WS_PORT_BASE=$((10#$WS_PORT_BASE))
if [ -z "$GPU_IDS" ]; then
    echo "GPU_IDS must contain between 1 and 8 unique non-negative integers; got: $GPU_IDS" >&2
    exit 2
fi
case "$GPU_IDS" in
    ,*|*,|*,,*)
        echo "GPU_IDS must contain between 1 and 8 unique non-negative integers; got: $GPU_IDS" >&2
        exit 2
        ;;
esac
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
    case "$gpu" in
        ''|*[!0-9]*)
            echo "GPU_IDS must contain between 1 and 8 unique non-negative integers; got: $GPU_IDS" >&2
            exit 2
            ;;
    esac
    duplicate_gpu=0
    for seen_gpu in "${SEEN_GPUS[@]-}"; do
        if [ "$seen_gpu" = "$gpu" ]; then
            duplicate_gpu=1
            break
        fi
    done
    if [ "$duplicate_gpu" -ne 0 ]; then
        echo "GPU_IDS must contain between 1 and 8 unique non-negative integers; got: $GPU_IDS" >&2
        exit 2
    fi
    SEEN_GPUS+=("$gpu")
done
GPU_COUNT="${#GPUS[@]}"
if [ "$GPU_COUNT" -lt 1 ] || [ "$GPU_COUNT" -gt 8 ]; then
    echo "GPU_IDS must contain between 1 and 8 unique non-negative integers; got: $GPU_IDS" >&2
    exit 2
fi
case "$REPLICAS_PER_GPU" in
    1|2) ;;
    *) echo "--replicas-per-gpu must be 1 or 2" >&2; exit 2 ;;
esac
LANE_COUNT=$((GPU_COUNT * REPLICAS_PER_GPU))
LAST_LANE_OFFSET=$((LANE_COUNT - 1))
if (( MASTER_PORT_BASE + LAST_LANE_OFFSET > 65535 )); then
    echo "--master-port-base range must end at or below 65535" >&2
    exit 2
fi
if (( WS_PORT_BASE + LAST_LANE_OFFSET > 65535 )); then
    echo "--ws-port-base range must end at or below 65535" >&2
    exit 2
fi
if (( MASTER_PORT_BASE <= WS_PORT_BASE + LAST_LANE_OFFSET && WS_PORT_BASE <= MASTER_PORT_BASE + LAST_LANE_OFFSET )); then
    echo "Master and WebSocket port ranges must not overlap: [${MASTER_PORT_BASE}, $((MASTER_PORT_BASE + LAST_LANE_OFFSET))] and [${WS_PORT_BASE}, $((WS_PORT_BASE + LAST_LANE_OFFSET))]" >&2
    exit 2
fi

if [ -z "$BUDGETS_CSV" ]; then
    echo "--budgets must include at least one positive integer" >&2
    exit 2
fi
case "$BUDGETS_CSV" in
    ,*|*,|*,,*)
        echo "--budgets must be a comma-separated list of positive integers; got: $BUDGETS_CSV" >&2
        exit 2
        ;;
esac
IFS=',' read -r -a BUDGETS <<< "$BUDGETS_CSV"
SEEN_BUDGETS=()
for steps in "${BUDGETS[@]}"; do
    case "$steps" in
        ''|*[!0-9]*|0)
            echo "--budgets must be a comma-separated list of positive integers; got: $BUDGETS_CSV" >&2
            exit 2
            ;;
    esac
    duplicate_budget=0
    for seen_budget in "${SEEN_BUDGETS[@]-}"; do
        if [ "$seen_budget" = "$steps" ]; then
            duplicate_budget=1
            break
        fi
    done
    if [ "$duplicate_budget" -ne 0 ]; then
        echo "--budgets must not contain duplicates; got: $BUDGETS_CSV" >&2
        exit 2
    fi
    SEEN_BUDGETS+=("$steps")
done

if [ ! -d "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

canonical_path() {
    "$READY_PYTHON" - "$1" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}

path_is_within() {
    local parent="$1"
    local child="$2"
    [ "$child" = "$parent" ] || [[ "$child" == "$parent/"* ]]
}

if ! CHECKPOINT_CANON="$(canonical_path "$CHECKPOINT")"; then
    echo "Unable to canonicalize checkpoint path: $CHECKPOINT" >&2
    exit 2
fi
if ! OUTPUT_ROOT_CANON="$(canonical_path "$OUTPUT_ROOT")"; then
    echo "Unable to canonicalize output path: $OUTPUT_ROOT" >&2
    exit 2
fi
if path_is_within "$CHECKPOINT_CANON" "$OUTPUT_ROOT_CANON" || path_is_within "$OUTPUT_ROOT_CANON" "$CHECKPOINT_CANON"; then
    echo "--output-root must not overlap checkpoint or its parent training tree" >&2
    exit 2
fi
CHECKPOINT="$CHECKPOINT_CANON"
OUTPUT_ROOT="$OUTPUT_ROOT_CANON"

for suite in "${SUITE_ORDER[@]}"; do
    for task_idx in {0..9}; do
        TASK_SPECS+=("${suite}:${task_idx}")
    done
done

MODEL_RESULT_ROOT="${OUTPUT_ROOT}/${MODEL_NAME}"

port_is_available() {
    "$READY_PYTHON" - "$1" <<'PY'
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

preflight_ports() {
    local lane master_port ws_port
    for ((lane = 0; lane < LANE_COUNT; lane++)); do
        master_port=$((MASTER_PORT_BASE + lane))
        ws_port=$((WS_PORT_BASE + lane))
        if ! port_is_available "$master_port"; then
            echo "Master port is already occupied: ${master_port}" >&2
            return 1
        fi
        if ! port_is_available "$ws_port"; then
            echo "WebSocket port is already occupied: ${ws_port}" >&2
            return 1
        fi
    done
}

result_root_state() {
    if [ ! -e "$MODEL_RESULT_ROOT" ]; then
        echo "absent"
    elif [ ! -d "$MODEL_RESULT_ROOT" ]; then
        echo "not_directory"
    elif [ -n "$(find "$MODEL_RESULT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "nonempty"
    else
        echo "empty"
    fi
}

prepare_result_root() {
    local state
    state="$(result_root_state)"
    if [ "${CHECK_ONLY:-0}" = "1" ]; then
        echo "RESULT_ROOT path=${MODEL_RESULT_ROOT} state=${state} launch_allowed=$([ "$state" = "absent" ] && echo yes || echo no)"
        return 0
    fi

    mkdir -p "$OUTPUT_ROOT"
    if ! mkdir "$MODEL_RESULT_ROOT"; then
        echo "Refusing to reuse existing native result root: ${MODEL_RESULT_ROOT}" >&2
        return 1
    fi
    echo "RESULT_ROOT path=${MODEL_RESULT_ROOT} state=acquired launch_allowed=yes"
}

print_worker_plan() {
    local steps="$1"
    local worker="$2"
    local gpu="$3"
    local replica="$4"
    local budget_root="${MODEL_RESULT_ROOT}/steps_${steps}"
    local worker_root="${budget_root}/workers/worker_${worker}"
    local master_port=$((MASTER_PORT_BASE + worker))
    local ws_port=$((WS_PORT_BASE + worker))

    echo "WORKER model=${MODEL_NAME} backend=native_teacher steps=${steps} worker=${worker} replica=${replica} gpu=${gpu} episodes=${EPISODES} task_queue=${#TASK_SPECS[@]} master_port=${master_port} ws_port=${ws_port} save_root=${worker_root}/server results_root=${worker_root}/results latency_jsonl=${worker_root}/sampler_latency.jsonl claim_mode=mkdir client_flag=--no-save-video server_flag=--no-save-debug-tensors"
}

server_group_is_alive() {
    [ -n "${SERVER_PGID:-}" ] && kill -0 -- "-${SERVER_PGID}" 2>/dev/null
}

stop_server() {
    local deadline
    if [ -z "${SERVER_PID:-}" ] && [ -z "${SERVER_PGID:-}" ]; then
        return 0
    fi
    if server_group_is_alive; then
        kill -TERM -- "-${SERVER_PGID}" 2>/dev/null || true
        deadline=$((SECONDS + WORKER_STOP_TIMEOUT_SECONDS))
        while server_group_is_alive && (( SECONDS < deadline )); do
            sleep 1
        done
        if server_group_is_alive; then
            kill -KILL -- "-${SERVER_PGID}" 2>/dev/null || true
        fi
    fi
    if [ -n "${SERVER_PID:-}" ]; then
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
    SERVER_PGID=""
}

wait_for_server() {
    local ws_port="$1"
    local worker="$2"
    local deadline=$((SECONDS + SERVER_READY_TIMEOUT_SECONDS))

    while true; do
        if "$READY_PYTHON" - "$ws_port" <<'PY'
from urllib.request import urlopen
import sys

try:
    with urlopen(f"http://127.0.0.1:{int(sys.argv[1])}/healthz", timeout=1) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected health status: {response.status}")
except Exception:
    raise SystemExit(1)
PY
        then
            return 0
        fi
        if ! server_group_is_alive; then
            echo "Native teacher server exited before worker ${worker} became ready" >&2
            if [ -n "${SERVER_PID:-}" ]; then
                wait "$SERVER_PID" 2>/dev/null || true
            fi
            return 1
        fi
        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for native teacher server on WebSocket port ${ws_port}; see worker log" >&2
            return 1
        fi
        sleep 2
    done
}

run_worker() {
    local steps="$1"
    local worker="$2"
    local gpu="$3"
    local replica="$4"
    local budget_root="$5"
    local claims_root="$6"
    local worker_root="${budget_root}/workers/worker_${worker}"
    local server_root="${worker_root}/server"
    local results_root="${worker_root}/results"
    local latency_jsonl="${worker_root}/sampler_latency.jsonl"
    local master_port=$((MASTER_PORT_BASE + worker))
    local ws_port=$((WS_PORT_BASE + worker))
    local task_spec suite task_idx claim_dir

    SERVER_PID=""
    SERVER_PGID=""
    trap 'stop_server; exit 130' INT TERM HUP
    trap stop_server EXIT
    mkdir -p "$server_root" "$results_root"
    echo "SERVER worker=${worker} replica=${replica} steps=${steps} gpu=${gpu} master_port=${master_port} ws_port=${ws_port}"
    CUDA_VISIBLE_DEVICES="$gpu" setsid "$SERVER_PYTHON" -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port "$master_port" \
        "$SERVER_ENTRYPOINT" \
        --config-name libero \
        --port "$ws_port" \
        --checkpoint-path "$CHECKPOINT" \
        --num-steps "$steps" \
        --action-num-steps "$steps" \
        --model-name "$MODEL_NAME" \
        --latency-jsonl "$latency_jsonl" \
        --save-root "$server_root" \
        --no-save-debug-tensors &
    SERVER_PID=$!
    SERVER_PGID="$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d '[:space:]')"
    if [ -z "$SERVER_PGID" ] || [ "$SERVER_PGID" != "$SERVER_PID" ]; then
        echo "Failed to isolate native teacher server process group for worker ${worker}" >&2
        SERVER_PGID=""
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        return 1
    fi

    wait_for_server "$ws_port" "$worker"

    for task_spec in "${TASK_SPECS[@]}"; do
        suite="${task_spec%%:*}"
        task_idx="${task_spec##*:}"
        claim_dir="${claims_root}/${suite}_${task_idx}"
        if mkdir "$claim_dir" 2>/dev/null; then
            echo "CLAIM worker=${worker} replica=${replica} steps=${steps} suite=${suite} task=${task_idx}"
        elif [ -d "$claim_dir" ]; then
            continue
        else
            echo "Failed to atomically claim ${suite} task ${task_idx}" >&2
            return 1
        fi

        if ! CUDA_VISIBLE_DEVICES="$gpu" "$CLIENT_PYTHON" "$CLIENT_SCRIPT" \
            --libero-benchmark "$suite" \
            --port "$ws_port" \
            --test-num "$EPISODES" \
            --task-range "$task_idx" "$((task_idx + 1))" \
            --out-dir "$results_root" \
            --no-save-video; then
            echo "Client failed for ${suite} task ${task_idx} on worker ${worker}" >&2
            return 1
        fi
    done
}

all_tasks_claimed() {
    local claims_root="$1"
    local task_spec suite task_idx
    for task_spec in "${TASK_SPECS[@]}"; do
        suite="${task_spec%%:*}"
        task_idx="${task_spec##*:}"
        if [ ! -d "${claims_root}/${suite}_${task_idx}" ]; then
            echo "Task was never claimed: ${suite} task ${task_idx}" >&2
            return 1
        fi
    done
}

remove_active_worker() {
    local completed_pid="$1"
    local pid
    local updated=()
    for pid in "${ACTIVE_WORKER_PIDS[@]}"; do
        if [ "$pid" != "$completed_pid" ]; then
            updated+=("$pid")
        fi
    done
    ACTIVE_WORKER_PIDS=("${updated[@]}")
}

terminate_active_workers() {
    local pid deadline
    if [ "${#ACTIVE_WORKER_PIDS[@]}" -eq 0 ]; then
        return 0
    fi
    for pid in "${ACTIVE_WORKER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    deadline=$((SECONDS + WORKER_STOP_TIMEOUT_SECONDS + 5))
    while (( SECONDS < deadline )); do
        local alive=0
        for pid in "${ACTIVE_WORKER_PIDS[@]}"; do
            if ! worker_has_exited "$pid"; then
                alive=1
                break
            fi
        done
        if [ "$alive" -eq 0 ]; then
            break
        fi
        sleep 1
    done
    for pid in "${ACTIVE_WORKER_PIDS[@]}"; do
        if ! worker_has_exited "$pid"; then
            kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${ACTIVE_WORKER_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    ACTIVE_WORKER_PIDS=()
}

abort_run() {
    local signal="$1"
    if [ "$TERMINATING" -ne 0 ]; then
        exit 1
    fi
    TERMINATING=1
    echo "Received ${signal}; terminating active dynamic workers" >&2
    terminate_active_workers
    exit 128
}

trap 'abort_run INT' INT
trap 'abort_run TERM' TERM
trap 'abort_run HUP' HUP

worker_has_exited() {
    local pid="$1"
    local state
    state="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
    [ -z "$state" ] || [[ "$state" == Z* ]]
}

wait_for_active_workers() {
    local completed_pid status pid observed_completion
    while [ "${#ACTIVE_WORKER_PIDS[@]}" -gt 0 ]; do
        observed_completion=0
        for pid in "${ACTIVE_WORKER_PIDS[@]}"; do
            if ! worker_has_exited "$pid"; then
                continue
            fi
            observed_completion=1
            completed_pid="$pid"
            if wait "$completed_pid"; then
                status=0
            else
                status=$?
            fi
            remove_active_worker "$completed_pid"
            if [ "$status" -ne 0 ]; then
                echo "Dynamic worker PID ${completed_pid} failed; terminating remaining workers" >&2
                terminate_active_workers
                return "$status"
            fi
            break
        done
        if [ "$observed_completion" -eq 0 ]; then
            sleep 1
        fi
    done
}

run_budget() {
    local steps="$1"
    local budget_root="${MODEL_RESULT_ROOT}/steps_${steps}"
    local claims_root="${budget_root}/claims"
    local lane gpu_index gpu replica worker_root

    mkdir -p "${budget_root}/workers" "$claims_root"
    ACTIVE_WORKER_PIDS=()
    for ((lane = 0; lane < LANE_COUNT; lane++)); do
        worker="$lane"
        gpu_index=$((lane % GPU_COUNT))
        gpu="${GPUS[$gpu_index]}"
        replica=$((lane / GPU_COUNT))
        worker_root="${budget_root}/workers/worker_${worker}"
        mkdir -p "$worker_root"
        setsid env \
            DYNAMIC_WORKER_MODE=1 \
            DYNAMIC_STEPS="$steps" \
            DYNAMIC_WORKER="$worker" \
            DYNAMIC_GPU="$gpu" \
            DYNAMIC_REPLICA="$replica" \
            DYNAMIC_BUDGET_ROOT="$budget_root" \
            DYNAMIC_CLAIMS_ROOT="$claims_root" \
            SERVER_PYTHON="$SERVER_PYTHON" \
            CLIENT_PYTHON="$CLIENT_PYTHON" \
            READY_PYTHON="$READY_PYTHON" \
            SERVER_READY_TIMEOUT_SECONDS="$SERVER_READY_TIMEOUT_SECONDS" \
            WORKER_STOP_TIMEOUT_SECONDS="$WORKER_STOP_TIMEOUT_SECONDS" \
            bash "$SCRIPT_PATH" \
            --checkpoint "$CHECKPOINT" \
            --output-root "$OUTPUT_ROOT" \
            --episodes "$EPISODES" \
            --gpu-ids "$GPU_IDS" \
            --replicas-per-gpu "$REPLICAS_PER_GPU" \
            --master-port-base "$MASTER_PORT_BASE" \
            --ws-port-base "$WS_PORT_BASE" \
            --budgets "$steps" >"${worker_root}/worker.log" 2>&1 &
        ACTIVE_WORKER_PIDS+=("$!")
    done

    wait_for_active_workers
    all_tasks_claimed "$claims_root"
    "$SERVER_PYTHON" "$MERGE_SCRIPT" \
        --input-root "$budget_root" \
        --expected-episodes "$EPISODES" \
        --expected-model "$MODEL_NAME" \
        --expected-video-steps "$steps" \
        --expected-action-steps "$steps" \
        --output "${budget_root}/summary.json"
}

if [ "${DYNAMIC_WORKER_MODE:-0}" = "1" ]; then
    for required in DYNAMIC_STEPS DYNAMIC_WORKER DYNAMIC_GPU DYNAMIC_REPLICA DYNAMIC_BUDGET_ROOT DYNAMIC_CLAIMS_ROOT; do
        if [ -z "${!required:-}" ]; then
            echo "Missing dynamic worker setting: ${required}" >&2
            exit 2
        fi
    done
    run_worker "$DYNAMIC_STEPS" "$DYNAMIC_WORKER" "$DYNAMIC_GPU" "$DYNAMIC_REPLICA" "$DYNAMIC_BUDGET_ROOT" "$DYNAMIC_CLAIMS_ROOT"
    exit 0
fi

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    prepare_result_root
    for steps in "${BUDGETS[@]}"; do
        for ((worker = 0; worker < LANE_COUNT; worker++)); do
            gpu_index=$((worker % GPU_COUNT))
            gpu="${GPUS[$gpu_index]}"
            replica=$((worker / GPU_COUNT))
            print_worker_plan "$steps" "$worker" "$gpu" "$replica"
        done
    done
    echo "CHECK_ONLY=1: verified $((${#BUDGETS[@]} * LANE_COUNT)) dynamic workers (40 tasks x budgets ${BUDGETS_CSV})."
    exit 0
fi

if ! command -v setsid >/dev/null 2>&1; then
    echo "setsid is required so every worker can be terminated as one process group" >&2
    exit 1
fi
preflight_ports || exit 1
prepare_result_root || exit 2

for steps in "${BUDGETS[@]}"; do
    run_budget "$steps"
done

echo "Completed ${MODEL_NAME}: all 40 tasks at matched budgets ${BUDGETS_CSV}."
