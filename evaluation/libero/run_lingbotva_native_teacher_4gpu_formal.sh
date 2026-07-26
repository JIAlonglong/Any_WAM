#!/usr/bin/env bash
# Short, safe launcher for the formal four-suite native-teacher LIBERO rerun.
# It writes only compact JSON/latency evidence under OUTPUT_ROOT; it never
# writes into the checkpoint tree and does not retain rollout videos or debug
# tensors.  Override any default below through the corresponding environment
# variable, or forward a dynamic-launcher flag after the script name.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DYNAMIC_LAUNCHER="${PROJECT_ROOT}/evaluation/libero/run_lingbotva_native_teacher_4suite_24_eval_8gpu_dynamic.sh"

export SERVER_PYTHON="${SERVER_PYTHON:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"
export CLIENT_PYTHON="${CLIENT_PYTHON:-/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python}"

CHECKPOINT="${CHECKPOINT:-/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero/transformer}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kpfs-intern/jialongliu/projects/Flash-WAM/evaluation/results/libero_teacher_native_dynamic_4gpu_formal_20260726}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
EPISODES="${EPISODES:-50}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-32680}"
WS_PORT_BASE="${WS_PORT_BASE:-32780}"
BUDGETS="${BUDGETS:-1,2,4}"

exec bash "${DYNAMIC_LAUNCHER}" \
    --checkpoint "${CHECKPOINT}" \
    --output-root "${OUTPUT_ROOT}" \
    --gpu-ids "${GPU_IDS}" \
    --episodes "${EPISODES}" \
    --master-port-base "${MASTER_PORT_BASE}" \
    --ws-port-base "${WS_PORT_BASE}" \
    --budgets "${BUDGETS}" \
    "$@"
