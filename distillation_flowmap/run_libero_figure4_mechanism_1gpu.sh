#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: Python is not executable: ${PYTHON}" >&2
    exit 2
fi
if [[ "$#" -eq 0 ]]; then
    echo "ERROR: pass --run-root, --teacher-model-path, and --dataset-path" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/wan_va:${PROJECT_ROOT}/distillation_flowmap:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export SKIP_TEACHER_COMPILE="${SKIP_TEACHER_COMPILE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${PROJECT_ROOT}"
exec "${PYTHON}" -m distillation_flowmap.eval_libero_figure4_mechanism "$@"
