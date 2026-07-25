#!/usr/bin/env bash
# Serial pipeline: Stage-2 training, then teacher/Stage-1/Stage-2 closed-loop eval.
# Every model is evaluated on all 40 standard LIBERO tasks at matched
# video/action budgets 1/1, 2/2, and 4/4.

set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GIT_COMMON="$(git -C "${PROJECT_ROOT}" rev-parse --git-common-dir)"
[[ "${GIT_COMMON}" = /* ]] || GIT_COMMON="${PROJECT_ROOT}/${GIT_COMMON}"
SHARED_ROOT="$(cd "$(dirname "${GIT_COMMON}")" && pwd)"

PHASE=all
DRY_RUN=0
STEPS=10000
SAVE_INTERVAL=1000
EPISODES=10
MASTER_PORT=29659
EVAL_MASTER_PORT_BASE=29680
EVAL_WS_PORT_BASE=29780
RUN_TAG="${RUN_TAG:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) PHASE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --steps) STEPS="$2"; shift 2 ;;
        --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --eval-master-port-base) EVAL_MASTER_PORT_BASE="$2"; shift 2 ;;
        --eval-ws-port-base) EVAL_WS_PORT_BASE="$2"; shift 2 ;;
        --run-tag) RUN_TAG="$2"; shift 2 ;;
        *) die "Unknown argument: $1" ;;
    esac
done
case "$PHASE" in train|eval|all) ;; *) die "--phase must be train, eval, or all" ;; esac
for value in "$STEPS" "$SAVE_INTERVAL" "$EPISODES" "$MASTER_PORT" "$EVAL_MASTER_PORT_BASE" "$EVAL_WS_PORT_BASE"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "numeric arguments must be positive integers"
done

PYTHON="${PYTHON:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"
STAGE1_CKPT="${STAGE1_CKPT:-${SHARED_ROOT}/distillation_flowmap/output_libero_fullft_stage1_warmup/checkpoints/step_2000}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero}"
DEFAULT_OUTPUT="${SHARED_ROOT}/distillation_flowmap/output_libero_lingbotva_stage2_video_only_opd_universal_from_stage1_step2000_steps${STEPS}"
[[ -z "$RUN_TAG" ]] || DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_${RUN_TAG}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${SHARED_ROOT}/evaluation/results/libero_lingbotva_video_only_opd_${RUN_TAG:-formal}}"
NAIVE_CKPT="${NAIVE_CKPT:-}"
EVAL_CONTRACT="${EVAL_OUTPUT_ROOT}/evaluation_contract.json"

TRAIN_SCRIPT="${SCRIPT_DIR}/run_libero_video_only_opd_stage2_8gpu.sh"
EVAL_SCRIPT="${PROJECT_ROOT}/evaluation/libero/run_lingbotva_4suite_124_eval_8gpu.sh"
TEACHER_TRANSFORMER="${TEACHER_MODEL_PATH}/transformer"
STAGE1_TRANSFORMER="${STAGE1_CKPT}/target_student/transformer"
STAGE2_TRANSFORMER="${OUTPUT_DIR}/checkpoints/step_${STEPS}/target_student/transformer"

train_cmd=(
    bash "$TRAIN_SCRIPT"
    --steps "$STEPS"
    --save-interval "$SAVE_INTERVAL"
    --master-port "$MASTER_PORT"
    --output-dir "$OUTPUT_DIR"
)
[[ -z "$RUN_TAG" ]] || train_cmd+=(--run-tag "$RUN_TAG")

print_eval() {
    local name="$1" ckpt="$2"
    echo "EVAL ${name} checkpoint=${ckpt} matched_budgets=1/1,2/2,4/4 suites=libero_10,libero_spatial,libero_object,libero_goal episodes_per_task=${EPISODES}"
}

run_eval() {
    local name="$1" ckpt="$2"
    [[ -d "$ckpt" ]] || die "Missing ${name} transformer: ${ckpt}"
    bash "$EVAL_SCRIPT" \
        --model-name "$name" \
        --checkpoint "$ckpt" \
        --episodes "$EPISODES" \
        --output-root "$EVAL_OUTPUT_ROOT" \
        --master-port-base "$EVAL_MASTER_PORT_BASE" \
        --ws-port-base "$EVAL_WS_PORT_BASE"
}

if [[ "$PHASE" == train || "$PHASE" == all ]]; then
    echo "TRAIN output=${OUTPUT_DIR} steps=${STEPS} save_interval=${SAVE_INTERVAL}"
    if [[ "$DRY_RUN" == 0 ]]; then
        STAGE1_CKPT="$STAGE1_CKPT" \
        TEACHER_MODEL_PATH="$TEACHER_MODEL_PATH" \
        OUTPUT_DIR="$OUTPUT_DIR" \
        PYTHON="$PYTHON" \
        "${train_cmd[@]}"
    fi
fi

if [[ "$PHASE" == eval || "$PHASE" == all ]]; then
    print_eval teacher "$TEACHER_TRANSFORMER"
    print_eval stage1_only "$STAGE1_TRANSFORMER"
    print_eval stage2 "$STAGE2_TRANSFORMER"
    if [[ -n "$NAIVE_CKPT" ]]; then
        print_eval naive_composition "$NAIVE_CKPT"
    else
        echo "EVAL_CONTRACT naive_composition=missing reason=no_explicit_no_distillation_checkpoint"
    fi
    if [[ "$DRY_RUN" == 0 ]]; then
        run_eval teacher "$TEACHER_TRANSFORMER"
        run_eval stage1_only "$STAGE1_TRANSFORMER"
        run_eval stage2 "$STAGE2_TRANSFORMER"
        if [[ -n "$NAIVE_CKPT" ]]; then
            run_eval naive_composition "$NAIVE_CKPT"
        fi
        contract_cmd=(
            "$PYTHON" "${PROJECT_ROOT}/evaluation/libero/write_lingbotva_eval_contract.py"
            --output "$EVAL_CONTRACT"
            --teacher "$TEACHER_TRANSFORMER"
            --stage1-only "$STAGE1_TRANSFORMER"
            --stage2 "$STAGE2_TRANSFORMER"
            --episodes-per-task "$EPISODES"
        )
        [[ -z "$NAIVE_CKPT" ]] || contract_cmd+=(--naive "$NAIVE_CKPT")
        "${contract_cmd[@]}"
    fi
fi
