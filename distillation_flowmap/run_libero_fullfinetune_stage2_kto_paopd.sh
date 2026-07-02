#!/bin/bash
# LIBERO Stage 2 KTO-PAOPD normalized focal reweighting variant.
# This script is Stage2-only and does not modify the default Stage2 run path.
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

export TEACHER_PATH="${TEACHER_PATH:-${PROJECT_ROOT}/checkpoints/lingbot-va-posttrain-libero}"
export DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/training_data/libero-long-lerobot}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export RESUME_ONLINE_FROM_TARGET="${RESUME_ONLINE_FROM_TARGET:-1}"

NGPU="${NGPU:-8}"
MASTER_PORT="${MASTER_PORT:-29513}"
ORIG_ACCUM="${ORIG_ACCUM:-16}"
ACCUM=$((ORIG_ACCUM / NGPU))
[ "$ACCUM" -lt 1 ] && ACCUM=1

STAGE1_CKPT="${STAGE1_CKPT:-${SCRIPT_DIR}/output_libero_fullft_stage1_liteval_20260627_012949/checkpoints/step_5000}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-${SCRIPT_DIR}/output_libero_fullft_stage2_kto_paopd_norm_focal}"
STAGE2_STEPS="${STAGE2_STEPS:-5000}"

if [ ! -d "${STAGE1_CKPT}/online_student/transformer" ] || [ ! -d "${STAGE1_CKPT}/target_student/transformer" ]; then
    echo "ERROR: missing complete Stage 1 checkpoint: ${STAGE1_CKPT}" >&2
    echo "Expected online_student/transformer and target_student/transformer" >&2
    exit 1
fi

echo "=========================================="
echo "LIBERO FullFT Stage2 KTO-PAOPD Normalized Focal"
echo "=========================================="
echo "Teacher:       ${TEACHER_PATH}"
echo "Dataset:       ${DATASET_PATH}"
echo "Resume:        ${STAGE1_CKPT}"
echo "Output:        ${STAGE2_OUTPUT}"
echo "GPUs:          ${NGPU}"
echo "Grad accum:    ${ACCUM} (effective batch: $((ACCUM * NGPU)))"
echo "Stage2 steps:  ${STAGE2_STEPS}"
echo "KTO mode:      ${KTO_REWEIGHT_MODE:-normalized_focal} (variant=${OPD_AUX_VARIANT:-kto_paopd_norm_focal})"
echo "KTO focal:     alpha=${KTO_ALPHA:-0.5}, temp=${KTO_TEMPERATURE:-0.10}, clip=[${KTO_MIN_WEIGHT:-0.5},${KTO_MAX_WEIGHT:-1.8}], q=${KTO_THRESHOLD_QUANTILE:-0.70}, ema=${KTO_THRESHOLD_EMA_DECAY:-0.90}"
echo "KTO alpha-decay: hold=${KTO_ALPHA_DECAY_HOLD_STEPS:-20}, ramp=${KTO_ALPHA_DECAY_RAMP_STEPS:-20}"
echo "KTO v-scale:   ${KTO_VIDEO_SCALE_START:-0.85} hold ${KTO_VIDEO_SCALE_HOLD_STEPS:-20} -> ${KTO_VIDEO_SCALE_END:-1.0} over ${KTO_VIDEO_SCALE_RAMP_STEPS:-20} steps"
echo "KTO main:      enabled=${KTO_MAIN_VIDEO_REWEIGHT:-0}, alpha=${KTO_MAIN_ALPHA:-1.0}, temp=${KTO_MAIN_TEMPERATURE:-${KTO_TEMPERATURE:-0.10}}, clip=[${KTO_MAIN_MIN_WEIGHT:-${KTO_MIN_WEIGHT:-0.5}},${KTO_MAIN_MAX_WEIGHT:-${KTO_MAX_WEIGHT:-1.8}}], q=${KTO_MAIN_THRESHOLD_QUANTILE:-0.70}, ema=${KTO_MAIN_THRESHOLD_EMA_DECAY:-0.90}"
echo "KTO hard mode: good=${KTO_GOOD_WEIGHT:-0.3}, bad=${KTO_BAD_WEIGHT:-1.0}, threshold=${KTO_THRESHOLD:-auto}"
echo "=========================================="

CONFIG_FILE=distillation_flowmap.config_libero_fullfinetune_stage2_kto_paopd \
RESUME_FROM_PATH="${STAGE1_CKPT}" \
OUTPUT_DIR="${STAGE2_OUTPUT}" \
MAX_TRAIN_STEPS="${STAGE2_STEPS}" \
torchrun \
    --nproc_per_node="${NGPU}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/train.py" \
    --teacher-model-path "${TEACHER_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --output-dir "${STAGE2_OUTPUT}" \
    --resume-from-path "${STAGE1_CKPT}" \
    --gradient-accumulation-steps "${ACCUM}"
