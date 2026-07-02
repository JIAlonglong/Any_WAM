#!/bin/bash
# ============================================================
# Flash-WAM FlowMap 蒸馏 —— 串行训练脚本
#
# 依次运行 RobotWin 和 LIBERO 两个任务的蒸馏训练。
# 支持单卡/多卡，支持跳过某个任务、从检查点恢复等。
#
# 用法：
#   # 4 卡串行跑两个任务（默认）
#   bash distillation_flowmap/run_all.sh
#
#   # 单卡
#   NGPU=1 bash distillation_flowmap/run_all.sh
#
#   # 只跑 RobotWin
#   RUN_ROBOTWIN=1 RUN_LIBERO=0 bash distillation_flowmap/run_all.sh
#
#   # 只跑 LIBERO
#   RUN_ROBOTWIN=0 RUN_LIBERO=1 bash distillation_flowmap/run_all.sh
#
#   # 从检查点恢复（分别指定）
#   RW_RESUME_STEP=5000 LIBERO_RESUME_STEP=3000 bash distillation_flowmap/run_all.sh
#
#   # 自定义输出根目录
#   OUTPUT_ROOT=/data/outputs bash distillation_flowmap/run_all.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================
# 全局参数
# ============================================================
NGPU="${NGPU:-4}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29501}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}}"

# 控制是否运行某个任务
RUN_ROBOTWIN="${RUN_ROBOTWIN:-1}"
RUN_LIBERO="${RUN_LIBERO:-1}"

# 各任务的恢复步数（空=不恢复）
RW_RESUME_STEP="${RW_RESUME_STEP:-}"
RW_RESUME_PATH="${RW_RESUME_PATH:-}"
LIBERO_RESUME_STEP="${LIBERO_RESUME_STEP:-}"
LIBERO_RESUME_PATH="${LIBERO_RESUME_PATH:-}"

# ============================================================
# 辅助函数
# ============================================================
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

log() {
    echo "[$(timestamp)] $*"
}

separator() {
    echo ""
    echo "================================================================"
    echo "  $*"
    echo "================================================================"
    echo ""
}

# ============================================================
# 预检
# ============================================================
log "Flash-WAM FlowMap 串行蒸馏"
log "GPUs: ${NGPU}"
log "Output root: ${OUTPUT_ROOT}"
log "Run RobotWin: ${RUN_ROBOTWIN}"
log "Run LIBERO:   ${RUN_LIBERO}"

# 多卡 gradient_accumulation_steps 换算
ORIG_ACCUM="${ORIG_ACCUM:-8}"
ACCUM=$((ORIG_ACCUM / NGPU))
[ "$ACCUM" -lt 1 ] && ACCUM=1
log "gradient_accumulation_steps: ${ACCUM} (effective batch factor: $((ACCUM * NGPU)))"

# ============================================================
# Task 1: RobotWin
# ============================================================
TASK1_STATUS="SKIPPED"
if [ "$RUN_ROBOTWIN" = "1" ]; then
    separator "Task 1/2: RobotWin FlowMap Distillation"

    export TEACHER_PATH="${RW_TEACHER_PATH:-${PROJECT_ROOT}/checkpoints/base}"
    export DATASET_PATH="${RW_DATASET_PATH:-${PROJECT_ROOT}/training_data/lerobot_robotwin_eef_aug_500}"
    export OUTPUT_DIR="${RW_OUTPUT_DIR:-${OUTPUT_ROOT}/output_robotwin}"
    export CONFIG_FILE="distillation_flowmap.config"
    export DISTILL_MODE="${DISTILL_MODE:-flashwam}"

    log "Teacher:  ${TEACHER_PATH}"
    log "Dataset:  ${DATASET_PATH}"
    log "Output:   ${OUTPUT_DIR}"

    RW_ARGS=""
    [ -n "$RW_RESUME_STEP" ] && RW_ARGS="$RW_ARGS --resume-from-step $RW_RESUME_STEP"
    [ -n "$RW_RESUME_PATH" ] && RW_ARGS="$RW_ARGS --resume-from-path $RW_RESUME_PATH"

    if torchrun \
        --nproc_per_node="${NGPU}" \
        --master_port="${MASTER_PORT_BASE}" \
        "${SCRIPT_DIR}/train.py" \
        --teacher-model-path "$TEACHER_PATH" \
        --dataset-path "$DATASET_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --gradient-accumulation-steps "${ACCUM}" \
        $RW_ARGS; then
        TASK1_STATUS="OK"
        log "RobotWin 训练完成 ✓"
    else
        TASK1_STATUS="FAILED (exit $?)"
        log "RobotWin 训练失败 ✗ (继续运行下一个任务)"
    fi
else
    log "RobotWin 已跳过 (RUN_ROBOTWIN=0)"
fi

# ============================================================
# Task 2: LIBERO
# ============================================================
TASK2_STATUS="SKIPPED"
if [ "$RUN_LIBERO" = "1" ]; then
    separator "Task 2/2: LIBERO FlowMap Distillation"

    export TEACHER_PATH="${LIBERO_TEACHER_PATH:-/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero}"
    export DATASET_PATH="${LIBERO_DATASET_PATH:-${PROJECT_ROOT}/training_data/libero-long-lerobot}"
    export OUTPUT_DIR="${LIBERO_OUTPUT_DIR:-${OUTPUT_ROOT}/output_libero}"
    export CONFIG_FILE="distillation_flowmap.config_libero"
    export DISTILL_MODE="${DISTILL_MODE:-flashwam}"

    log "Teacher:  ${TEACHER_PATH}"
    log "Dataset:  ${DATASET_PATH}"
    log "Output:   ${OUTPUT_DIR}"

    LIBERO_ARGS=""
    [ -n "$LIBERO_RESUME_STEP" ] && LIBERO_ARGS="$LIBERO_ARGS --resume-from-step $LIBERO_RESUME_STEP"
    [ -n "$LIBERO_RESUME_PATH" ] && LIBERO_ARGS="$LIBERO_ARGS --resume-from-path $LIBERO_RESUME_PATH"

    if torchrun \
        --nproc_per_node="${NGPU}" \
        --master_port="$((MASTER_PORT_BASE + 1))" \
        "${SCRIPT_DIR}/train.py" \
        --teacher-model-path "$TEACHER_PATH" \
        --dataset-path "$DATASET_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --gradient-accumulation-steps "${ACCUM}" \
        $LIBERO_ARGS; then
        TASK2_STATUS="OK"
        log "LIBERO 训练完成 ✓"
    else
        TASK2_STATUS="FAILED (exit $?)"
        log "LIBERO 训练失败 ✗"
    fi
else
    log "LIBERO 已跳过 (RUN_LIBERO=0)"
fi

# ============================================================
# 汇总
# ============================================================
separator "训练汇总"
log "RobotWin: ${TASK1_STATUS}"
log "LIBERO:   ${TASK2_STATUS}"

if [[ "$TASK1_STATUS" == *"FAILED"* ]] || [[ "$TASK2_STATUS" == *"FAILED"* ]]; then
    log "部分任务失败，请检查日志"
    exit 1
fi

log "全部完成 ✓"
