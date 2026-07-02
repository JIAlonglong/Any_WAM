#!/bin/bash
# Run RobotWin full-parameter Stage 1 warmup followed by Stage 2 OPD continuation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

NGPU="${NGPU:-8}"
MASTER_PORT_STAGE1="${MASTER_PORT_STAGE1:-29610}"
MASTER_PORT_STAGE2="${MASTER_PORT_STAGE2:-29611}"

TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${TEACHER_PATH:-${PROJECT_ROOT}/checkpoints/base}}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/training_data/lerobot_robotwin_eef_aug_500}"

STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-${SCRIPT_DIR}/output_robotwin_fullft_stage1_warmup}"
ROBOTWIN_STAGE1_MAX_TRAIN_STEPS="${ROBOTWIN_STAGE1_MAX_TRAIN_STEPS:-2000}"
ROBOTWIN_STAGE1_RESUME_STEP="${ROBOTWIN_STAGE1_RESUME_STEP:-${ROBOTWIN_STAGE1_MAX_TRAIN_STEPS}}"

STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-${SCRIPT_DIR}/output_robotwin_fullft_stage2_anyflow_from_stage1_step${ROBOTWIN_STAGE1_RESUME_STEP}}"
STAGE2_RESUME_FROM_PATH="${STAGE2_RESUME_FROM_PATH:-${STAGE1_OUTPUT_DIR}/checkpoints/step_${ROBOTWIN_STAGE1_RESUME_STEP}}"
ROBOTWIN_STAGE2_MAX_TRAIN_STEPS="${ROBOTWIN_STAGE2_MAX_TRAIN_STEPS:-5000}"

ORIG_ACCUM="${ORIG_ACCUM:-8}"
ACCUM="${GRADIENT_ACCUMULATION_STEPS:-$((ORIG_ACCUM / NGPU))}"
[ "$ACCUM" -lt 1 ] && ACCUM=1

RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"

echo "RobotWin teacher: ${TEACHER_MODEL_PATH}"
echo "RobotWin dataset: ${DATASET_PATH}"
echo "GPUs: ${NGPU}, gradient_accumulation_steps: ${ACCUM}"

if [ "$RUN_STAGE1" = "1" ]; then
    echo "===== RobotWin Stage 1 warmup ====="
    CONFIG_FILE=distillation_flowmap.config_robotwin_fullfinetune_stage1_warmup \
    OUTPUT_DIR="${STAGE1_OUTPUT_DIR}" \
    MAX_TRAIN_STEPS="${ROBOTWIN_STAGE1_MAX_TRAIN_STEPS}" \
    WANDB_MODE="${WANDB_MODE:-offline}" \
    torchrun --nproc_per_node="${NGPU}" --master_port="${MASTER_PORT_STAGE1}" \
        "${SCRIPT_DIR}/train.py" \
        --teacher-model-path "${TEACHER_MODEL_PATH}" \
        --dataset-path "${DATASET_PATH}" \
        --gradient-accumulation-steps "${ACCUM}"
fi

if [ "$RUN_STAGE2" = "1" ]; then
    echo "===== RobotWin Stage 2 OPD continuation ====="
    CONFIG_FILE=distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow \
    RESUME_FROM_PATH="${STAGE2_RESUME_FROM_PATH}" \
    RESUME_ONLINE_FROM_TARGET="${RESUME_ONLINE_FROM_TARGET:-1}" \
    OUTPUT_DIR="${STAGE2_OUTPUT_DIR}" \
    MAX_TRAIN_STEPS="${ROBOTWIN_STAGE2_MAX_TRAIN_STEPS}" \
    WANDB_MODE="${WANDB_MODE:-offline}" \
    torchrun --nproc_per_node="${NGPU}" --master_port="${MASTER_PORT_STAGE2}" \
        "${SCRIPT_DIR}/train.py" \
        --teacher-model-path "${TEACHER_MODEL_PATH}" \
        --dataset-path "${DATASET_PATH}" \
        --gradient-accumulation-steps "${ACCUM}"
fi
