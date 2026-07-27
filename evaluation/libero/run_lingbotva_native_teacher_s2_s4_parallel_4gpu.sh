#!/usr/bin/env bash
# Launch isolated native-teacher LIBERO S2 and S4 evaluations concurrently.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FORMAL_LAUNCHER="${PROJECT_ROOT}/evaluation/libero/run_lingbotva_native_teacher_4gpu_2replica_formal.sh"

S2_OUTPUT_ROOT="${S2_OUTPUT_ROOT:-/kpfs-intern/jialongliu/projects/Flash-WAM/evaluation/results/libero_teacher_native_dynamic_s2_2gpu2replica_formal_20260727}"
S4_OUTPUT_ROOT="${S4_OUTPUT_ROOT:-/kpfs-intern/jialongliu/projects/Flash-WAM/evaluation/results/libero_teacher_native_dynamic_s4_2gpu2replica_formal_20260727}"
S2_GPU_IDS="${S2_GPU_IDS:-0,1}"
S4_GPU_IDS="${S4_GPU_IDS:-2,3}"
S2_REPLICAS_PER_GPU="${S2_REPLICAS_PER_GPU:-2}"
S4_REPLICAS_PER_GPU="${S4_REPLICAS_PER_GPU:-2}"
S2_MASTER_PORT_BASE="${S2_MASTER_PORT_BASE:-34680}"
S2_WS_PORT_BASE="${S2_WS_PORT_BASE:-34780}"
S4_MASTER_PORT_BASE="${S4_MASTER_PORT_BASE:-34880}"
S4_WS_PORT_BASE="${S4_WS_PORT_BASE:-34980}"

path_is_within() {
    local parent="$1"
    local child="$2"
    [ "$child" = "$parent" ] || [[ "$child" == "$parent/"* ]]
}

parse_gpu_ids() {
    local label="$1"
    local gpu_ids="$2"
    local gpu seen
    local -a parsed=()

    case "$gpu_ids" in
        ''|,*|*,|*,,*)
            echo "${label} must contain comma-separated non-negative GPU IDs" >&2
            exit 2
            ;;
    esac
    IFS=',' read -r -a parsed <<< "$gpu_ids"
    for gpu in "${parsed[@]}"; do
        case "$gpu" in
            *[!0-9]*)
                echo "${label} must contain comma-separated non-negative GPU IDs" >&2
                exit 2
                ;;
        esac
        gpu=$((10#$gpu))
        for seen in "${PARSED_GPU_IDS[@]-}"; do
            if [ "$gpu" = "$seen" ]; then
                echo "${label} must not contain duplicate GPU IDs" >&2
                exit 2
            fi
        done
        PARSED_GPU_IDS+=("$gpu")
    done
}

validate_replicas() {
    local label="$1"
    local replicas="$2"
    case "$replicas" in
        1|2) ;;
        *)
            echo "${label} must be 1 or 2" >&2
            exit 2
            ;;
    esac
}

validate_port_base() {
    local label="$1"
    local port="$2"
    local lane_count="$3"
    local normalized_port
    case "$port" in
        ''|*[!0-9]*)
            echo "${label} must be a positive integer" >&2
            exit 2
            ;;
    esac
    normalized_port=$((10#$port))
    if (( normalized_port == 0 )); then
        echo "${label} must be a positive integer" >&2
        exit 2
    fi
    if (( normalized_port + lane_count - 1 > 65535 )); then
        echo "${label} range must end at or below 65535" >&2
        exit 2
    fi
    NORMALIZED_PORT="$normalized_port"
}

ranges_overlap() {
    local start_a="$1"
    local end_a="$2"
    local start_b="$3"
    local end_b="$4"
    (( start_a <= end_b && start_b <= end_a ))
}

S2_OUTPUT_ROOT_CANON="$(realpath -m "$S2_OUTPUT_ROOT")"
S4_OUTPUT_ROOT_CANON="$(realpath -m "$S4_OUTPUT_ROOT")"
if path_is_within "$S2_OUTPUT_ROOT_CANON" "$S4_OUTPUT_ROOT_CANON" || path_is_within "$S4_OUTPUT_ROOT_CANON" "$S2_OUTPUT_ROOT_CANON"; then
    echo "S2_OUTPUT_ROOT and S4_OUTPUT_ROOT must not overlap: ${S2_OUTPUT_ROOT_CANON} vs ${S4_OUTPUT_ROOT_CANON}" >&2
    exit 2
fi

PARSED_GPU_IDS=()
parse_gpu_ids "S2_GPU_IDS" "$S2_GPU_IDS"
S2_GPUS=("${PARSED_GPU_IDS[@]}")
S2_GPU_IDS="$(IFS=,; printf '%s' "${S2_GPUS[*]}")"
PARSED_GPU_IDS=()
parse_gpu_ids "S4_GPU_IDS" "$S4_GPU_IDS"
S4_GPUS=("${PARSED_GPU_IDS[@]}")
S4_GPU_IDS="$(IFS=,; printf '%s' "${S4_GPUS[*]}")"
for s2_gpu in "${S2_GPUS[@]}"; do
    for s4_gpu in "${S4_GPUS[@]}"; do
        if [ "$s2_gpu" = "$s4_gpu" ]; then
            echo "S2_GPU_IDS and S4_GPU_IDS must not overlap (shared GPU ${s2_gpu})" >&2
            exit 2
        fi
    done
done

validate_replicas "S2_REPLICAS_PER_GPU" "$S2_REPLICAS_PER_GPU"
validate_replicas "S4_REPLICAS_PER_GPU" "$S4_REPLICAS_PER_GPU"
S2_LANE_COUNT=$((${#S2_GPUS[@]} * S2_REPLICAS_PER_GPU))
S4_LANE_COUNT=$((${#S4_GPUS[@]} * S4_REPLICAS_PER_GPU))
validate_port_base "S2_MASTER_PORT_BASE" "$S2_MASTER_PORT_BASE" "$S2_LANE_COUNT"
S2_MASTER_PORT_BASE="$NORMALIZED_PORT"
validate_port_base "S2_WS_PORT_BASE" "$S2_WS_PORT_BASE" "$S2_LANE_COUNT"
S2_WS_PORT_BASE="$NORMALIZED_PORT"
validate_port_base "S4_MASTER_PORT_BASE" "$S4_MASTER_PORT_BASE" "$S4_LANE_COUNT"
S4_MASTER_PORT_BASE="$NORMALIZED_PORT"
validate_port_base "S4_WS_PORT_BASE" "$S4_WS_PORT_BASE" "$S4_LANE_COUNT"
S4_WS_PORT_BASE="$NORMALIZED_PORT"
if ranges_overlap "$S2_MASTER_PORT_BASE" "$((S2_MASTER_PORT_BASE + S2_LANE_COUNT - 1))" "$S4_MASTER_PORT_BASE" "$((S4_MASTER_PORT_BASE + S4_LANE_COUNT - 1))"; then
    echo "S2 and S4 master port ranges must not overlap" >&2
    exit 2
fi
if ranges_overlap "$S2_WS_PORT_BASE" "$((S2_WS_PORT_BASE + S2_LANE_COUNT - 1))" "$S4_WS_PORT_BASE" "$((S4_WS_PORT_BASE + S4_LANE_COUNT - 1))"; then
    echo "S2 and S4 WebSocket port ranges must not overlap" >&2
    exit 2
fi

s2_pid=""
s4_pid=""

is_active_child() {
    local pid="$1"
    jobs -pr | grep -Fxq "$pid"
}

terminate_children() {
    local pid
    local -a active_pids=()
    for pid in "$s2_pid" "$s4_pid"; do
        if [ -n "$pid" ] && is_active_child "$pid"; then
            active_pids+=("$pid")
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${active_pids[@]}"; do
        if is_active_child "$pid"; then
            wait "$pid" 2>/dev/null || true
        fi
    done
    s2_pid=""
    s4_pid=""
    exit 1
}

trap 'terminate_children INT' INT
trap 'terminate_children TERM' TERM

OUTPUT_ROOT="$S2_OUTPUT_ROOT" GPU_IDS="$S2_GPU_IDS" REPLICAS_PER_GPU="$S2_REPLICAS_PER_GPU" BUDGETS=2 MASTER_PORT_BASE="$S2_MASTER_PORT_BASE" WS_PORT_BASE="$S2_WS_PORT_BASE" bash "$FORMAL_LAUNCHER" &
s2_pid=$!
OUTPUT_ROOT="$S4_OUTPUT_ROOT" GPU_IDS="$S4_GPU_IDS" REPLICAS_PER_GPU="$S4_REPLICAS_PER_GPU" BUDGETS=4 MASTER_PORT_BASE="$S4_MASTER_PORT_BASE" WS_PORT_BASE="$S4_WS_PORT_BASE" bash "$FORMAL_LAUNCHER" &
s4_pid=$!

s2_rc=0
s4_rc=0
wait "$s2_pid" || s2_rc=$?
s2_pid=""
wait "$s4_pid" || s4_rc=$?
s4_pid=""
printf 'S2 rc=%s; S4 rc=%s\n' "$s2_rc" "$s4_rc"

if [ "$s2_rc" -eq 0 ] && [ "$s4_rc" -eq 0 ]; then
    exit 0
fi
exit 1
