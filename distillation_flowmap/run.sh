#!/bin/bash
# Launch Flash-WAM FlowMap distillation.
#   TEACHER_PATH=/path/to/teacher DATASET_PATH=/path/to/dataset \
#   NGPU=4 bash distillation_flowmap/run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${TEACHER_PATH:-}}"
DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:-}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

NGPU="${NGPU:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"

# 多卡时自动换算 gradient_accumulation_steps，保持等效 batch size 不变
# 原始配置：batch_size=1, gradient_accumulation_steps=8, 等效 batch=8
# 多卡等效 batch = batch_size × (8/NGPU) × NGPU = 8
ORIG_ACCUM=8
ACCUM=$((ORIG_ACCUM / NGPU))
[ "$ACCUM" -lt 1 ] && ACCUM=1

echo "GPUs: ${NGPU}, gradient_accumulation_steps: ${ACCUM} (effective batch: $((1 * ACCUM * NGPU)))"

ARGS=""
[ -n "$TEACHER_MODEL_PATH" ] && ARGS="$ARGS --teacher-model-path $TEACHER_MODEL_PATH"
[ -n "$DATASET_PATH" ]       && ARGS="$ARGS --dataset-path $DATASET_PATH"
[ -n "$OUTPUT_DIR" ]         && ARGS="$ARGS --output-dir $OUTPUT_DIR"
[ -n "$RESUME_FROM_STEP" ]   && ARGS="$ARGS --resume-from-step $RESUME_FROM_STEP"
[ -n "$RESUME_FROM_PATH" ]   && ARGS="$ARGS --resume-from-path $RESUME_FROM_PATH"

torchrun \
    --nproc_per_node="${NGPU}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/train.py" \
    --gradient-accumulation-steps "${ACCUM}" \
    $ARGS
