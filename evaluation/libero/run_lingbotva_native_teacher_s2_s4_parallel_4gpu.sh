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

s2_pid=""
s4_pid=""

terminate_children() {
    local signal="$1"
    local pid
    for pid in "$s2_pid" "$s4_pid"; do
        if [ -n "$pid" ]; then
            kill -"$signal" "$pid" 2>/dev/null || true
        fi
    done
    for pid in "$s2_pid" "$s4_pid"; do
        if [ -n "$pid" ]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
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
wait "$s4_pid" || s4_rc=$?
printf 'S2 rc=%s; S4 rc=%s\n' "$s2_rc" "$s4_rc"

if [ "$s2_rc" -eq 0 ] && [ "$s4_rc" -eq 0 ]; then
    exit 0
fi
exit 1
