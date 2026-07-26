#!/usr/bin/env bash
# Fast, safe formal launcher: four GPUs with two independent native-teacher
# server/client replicas each. It writes only compact evaluation evidence under
# a fresh OUTPUT_ROOT; rollout videos and debug tensors remain disabled by the
# dynamic launcher.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DYNAMIC_LAUNCHER="${PROJECT_ROOT}/evaluation/libero/run_lingbotva_native_teacher_4suite_24_eval_8gpu_dynamic.sh"

export SERVER_PYTHON="${SERVER_PYTHON:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"
export CLIENT_PYTHON="${CLIENT_PYTHON:-/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python}"

CHECKPOINT="${CHECKPOINT:-/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero/transformer}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kpfs-intern/jialongliu/projects/Flash-WAM/evaluation/results/libero_teacher_native_dynamic_4gpu_2replica_formal_20260727}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-2}"
EPISODES="${EPISODES:-50}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-33680}"
WS_PORT_BASE="${WS_PORT_BASE:-33780}"
BUDGETS="${BUDGETS:-1,2,4}"

exec bash "${DYNAMIC_LAUNCHER}" \
    --checkpoint "${CHECKPOINT}" \
    --output-root "${OUTPUT_ROOT}" \
    --gpu-ids "${GPU_IDS}" \
    --replicas-per-gpu "${REPLICAS_PER_GPU}" \
    --episodes "${EPISODES}" \
    --master-port-base "${MASTER_PORT_BASE}" \
    --ws-port-base "${WS_PORT_BASE}" \
    --budgets "${BUDGETS}" \
    "$@"
