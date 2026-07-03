#!/bin/bash
# LIBERO Cosmos Policy raw-observation distillation:
#   1. action-only Stage 1 warmup from WanVA student base
#   2. action-only Stage 2 continuation from the Stage 1 EMA target
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f /root/nas/junjie/miniconda3/etc/profile.d/conda.sh ]; then
    source /root/nas/junjie/miniconda3/etc/profile.d/conda.sh
else
    echo "conda not found; set PATH or install conda before running this script" >&2
    exit 1
fi
conda activate "${CONDA_ENV:-/root/nas/junjie/conda_envs/any_wam}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export COSMOS_POLICY_PATH="${COSMOS_POLICY_PATH:-/root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
export STUDENT_BASE_MODEL_PATH="${STUDENT_BASE_MODEL_PATH:-/root/nas/junjie/jj/Any_WAM/checkpoints/lingbot-va-posttrain-libero}"
export DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/training_data/libero-long-lerobot}"
export COSMOS_POLICY_USE_RAW_INFERENCE="${COSMOS_POLICY_USE_RAW_INFERENCE:-1}"
export COSMOS_POLICY_INFERENCE_MODE="${COSMOS_POLICY_INFERENCE_MODE:-subprocess}"
export COSMOS_POLICY_PYTHON="${COSMOS_POLICY_PYTHON:-/root/nas/junjie/cosmos_predict2_5/envs/predict2_py310/bin/python}"
export COSMOS_PREDICT2_REPO="${COSMOS_PREDICT2_REPO:-/root/nas/junjie/cosmos_predict2_5/repos/cosmos-predict2.5}"
export COSMOS_POLICY_EXTRA_PYTHONPATH="${COSMOS_POLICY_EXTRA_PYTHONPATH:-/root/nas/junjie/conda_envs/any_wam/lib/python3.10/site-packages}"
export COSMOS_PREDICT25_LOCAL_MODEL_DIR="${COSMOS_PREDICT25_LOCAL_MODEL_DIR:-/root/nas/junjie/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World}"
export COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION="${COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION:-5}"
export ENABLE_WANDB="${ENABLE_WANDB:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

NGPU="${NGPU:-2}"
MASTER_PORT_STAGE1="${MASTER_PORT_STAGE1:-29531}"
MASTER_PORT_STAGE2="${MASTER_PORT_STAGE2:-29532}"
ORIG_ACCUM="${ORIG_ACCUM:-16}"
ACCUM=$((ORIG_ACCUM / NGPU))
[ "$ACCUM" -lt 1 ] && ACCUM=1

RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
PREFLIGHT_CHECK_DATASET="${PREFLIGHT_CHECK_DATASET:-1}"
PREFLIGHT_CHECK_WORKER="${PREFLIGHT_CHECK_WORKER:-0}"
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"

STAGE1_OUTPUT="${STAGE1_OUTPUT:-${SCRIPT_DIR}/output_libero_cosmos_policy_raw_stage1}"
STAGE1_STEPS="${STAGE1_STEPS:-500}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-${SCRIPT_DIR}/output_libero_cosmos_policy_raw_stage2}"
STAGE2_STEPS="${STAGE2_STEPS:-5000}"
STAGE1_CKPT="${STAGE1_CKPT:-${STAGE1_OUTPUT}/checkpoints/step_${STAGE1_STEPS}}"

echo "=========================================="
echo "LIBERO Cosmos Policy Raw Stage1 -> Stage2"
echo "=========================================="
echo "Teacher:          ${COSMOS_POLICY_PATH}"
echo "Student base:     ${STUDENT_BASE_MODEL_PATH}"
echo "Dataset:          ${DATASET_PATH}"
echo "Cosmos repo:      ${COSMOS_PREDICT2_REPO}"
echo "Cosmos python:    ${COSMOS_POLICY_PYTHON}"
echo "Raw inference:    ${COSMOS_POLICY_USE_RAW_INFERENCE}"
echo "Denoise steps:    ${COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION}"
echo "GPUs:             ${NGPU}"
echo "Grad accum:       ${ACCUM} (effective batch: $((ACCUM * NGPU)))"
echo "Stage1 output:    ${STAGE1_OUTPUT}"
echo "Stage1 steps:     ${STAGE1_STEPS}"
echo "Stage2 output:    ${STAGE2_OUTPUT}"
echo "Stage2 steps:     ${STAGE2_STEPS}"
echo "=========================================="

if [ "${RUN_PREFLIGHT}" = "1" ]; then
    PREFLIGHT_ARGS=("--fix-local-tokenizer")
    if [ "${PREFLIGHT_CHECK_DATASET}" = "1" ]; then
        PREFLIGHT_ARGS+=("--check-dataset")
    fi
    if [ "${PREFLIGHT_CHECK_WORKER}" = "1" ]; then
        PREFLIGHT_ARGS+=("--check-worker")
    fi
    python "${SCRIPT_DIR}/cosmos_policy_preflight.py" "${PREFLIGHT_ARGS[@]}"
fi

if [ "${RUN_STAGE1}" = "1" ]; then
    echo "[Stage 1] Cosmos raw action warmup"
    CONFIG_FILE=distillation_flowmap.config_libero_cosmos_policy_stage1 \
    OUTPUT_DIR="${STAGE1_OUTPUT}" \
    MAX_TRAIN_STEPS="${STAGE1_STEPS}" \
    torchrun \
        --nproc_per_node="${NGPU}" \
        --master_port="${MASTER_PORT_STAGE1}" \
        "${SCRIPT_DIR}/train.py" \
        --teacher-model-path "${COSMOS_POLICY_PATH}" \
        --dataset-path "${DATASET_PATH}" \
        --output-dir "${STAGE1_OUTPUT}" \
        --gradient-accumulation-steps "${ACCUM}"
fi

if [ ! -d "${STAGE1_CKPT}/online_student/transformer" ] || [ ! -d "${STAGE1_CKPT}/target_student/transformer" ]; then
    echo "ERROR: missing complete Stage 1 checkpoint: ${STAGE1_CKPT}" >&2
    echo "Expected online_student/transformer and target_student/transformer" >&2
    exit 1
fi

python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); expected=int(sys.argv[2]); assert p.exists(), f"missing Stage 1 target config: {p}"; actual=int(json.loads(p.read_text()).get("checkpoint_step", -1)); assert actual == expected, f"Stage 1 target checkpoint_step={actual}, expected {expected}"; print(f"Stage 1 EMA target checkpoint verified at step {actual}")' "${STAGE1_CKPT}/target_student/transformer/config.json" "${STAGE1_STEPS}"

if [ "${RUN_STAGE2}" = "1" ]; then
    export RESUME_ONLINE_FROM_TARGET="${RESUME_ONLINE_FROM_TARGET:-1}"
    export RESET_RESUME_STEP="${RESET_RESUME_STEP:-1}"
    echo "[Stage 2] Cosmos raw action continuation from Stage 1 EMA target"
    CONFIG_FILE=distillation_flowmap.config_libero_cosmos_policy_stage2 \
    RESUME_FROM_PATH="${STAGE1_CKPT}" \
    OUTPUT_DIR="${STAGE2_OUTPUT}" \
    MAX_TRAIN_STEPS="${STAGE2_STEPS}" \
    torchrun \
        --nproc_per_node="${NGPU}" \
        --master_port="${MASTER_PORT_STAGE2}" \
        "${SCRIPT_DIR}/train.py" \
        --teacher-model-path "${COSMOS_POLICY_PATH}" \
        --dataset-path "${DATASET_PATH}" \
        --output-dir "${STAGE2_OUTPUT}" \
        --resume-from-path "${STAGE1_CKPT}" \
        --gradient-accumulation-steps "${ACCUM}"
fi

echo "Done."
