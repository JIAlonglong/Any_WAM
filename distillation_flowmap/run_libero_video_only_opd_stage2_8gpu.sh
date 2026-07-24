#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash distillation_flowmap/run_libero_video_only_opd_stage2_8gpu.sh [options]

Options:
  --dry-run
  --steps N
  --save-interval N
  --master-port PORT
  --output-dir PATH
  --run-tag TAG
  --resume-step N
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
positive() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer"; }
require_file() { [[ -f "$2" ]] || die "Missing $1: $2"; }

DRY_RUN=0
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
MASTER_PORT="${MASTER_PORT:-29659}"
OUTPUT_OVERRIDE="${OUTPUT_DIR:-}"
RUN_TAG="${RUN_TAG:-}"
RESUME_STEP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --steps) [[ $# -ge 2 ]] || die "$1 requires a value"; MAX_TRAIN_STEPS="$2"; shift 2 ;;
        --save-interval) [[ $# -ge 2 ]] || die "$1 requires a value"; SAVE_INTERVAL="$2"; shift 2 ;;
        --master-port) [[ $# -ge 2 ]] || die "$1 requires a value"; MASTER_PORT="$2"; shift 2 ;;
        --output-dir) [[ $# -ge 2 ]] || die "$1 requires a value"; OUTPUT_OVERRIDE="$2"; shift 2 ;;
        --run-tag) [[ $# -ge 2 ]] || die "$1 requires a value"; RUN_TAG="$2"; shift 2 ;;
        --resume-step) [[ $# -ge 2 ]] || die "$1 requires a value"; RESUME_STEP="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

positive MAX_TRAIN_STEPS "${MAX_TRAIN_STEPS}"
positive SAVE_INTERVAL "${SAVE_INTERVAL}"
positive MASTER_PORT "${MASTER_PORT}"
(( MASTER_PORT <= 65535 )) || die "invalid master port: ${MASTER_PORT}"
if [[ -n "${RESUME_STEP}" ]]; then positive RESUME_STEP "${RESUME_STEP}"; fi
if [[ -n "${RUN_TAG}" && ! "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    die "invalid run tag: ${RUN_TAG}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GIT_COMMON="$(git -C "${PROJECT_ROOT}" rev-parse --git-common-dir)"
[[ "${GIT_COMMON}" = /* ]] || GIT_COMMON="${PROJECT_ROOT}/${GIT_COMMON}"
SHARED_ROOT="$(cd "$(dirname "${GIT_COMMON}")" && pwd)"

PYTHON="${PYTHON:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"
STAGE1_CKPT="${STAGE1_CKPT:-${SHARED_ROOT}/distillation_flowmap/output_libero_fullft_stage1_warmup/checkpoints/step_2000}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${SHARED_ROOT}/../lingbot-va/checkpoints/libero}"
DATASET_PATH="${DATASET_PATH:-${SHARED_ROOT}/training_data/libero-long-lerobot}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-${DATASET_PATH}/empty_emb.pt}"
DEFAULT_OUTPUT="${SHARED_ROOT}/distillation_flowmap/output_libero_lingbotva_stage2_video_only_opd_universal_from_stage1_step2000_steps${MAX_TRAIN_STEPS}"
[[ -z "${RUN_TAG}" ]] || DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_${RUN_TAG}"
OUTPUT_DIR="${OUTPUT_OVERRIDE:-${DEFAULT_OUTPUT}}"

[[ -x "${PYTHON}" ]] || die "PYTHON is not executable: ${PYTHON}"
require_file "teacher transformer config" "${TEACHER_MODEL_PATH}/transformer/config.json"
require_file "dataset episodes metadata" "${DATASET_PATH}/meta/episodes.jsonl"
require_file "empty embedding" "${EMPTY_EMB_PATH}"
for variant in online_student target_student; do
    require_file "${variant} transformer config" "${STAGE1_CKPT}/${variant}/transformer/config.json"
    require_file "${variant} diffusion weights" "${STAGE1_CKPT}/${variant}/transformer/diffusion_pytorch_model.safetensors"
done

GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<<"${GPU_LIST}"
[[ ${#GPUS[@]} -eq 8 ]] || die "CUDA_VISIBLE_DEVICES must contain exactly 8 GPUs"
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
    [[ -n "${gpu}" ]] || die "empty GPU id"
    [[ -z "${SEEN[${gpu}]:-}" ]] || die "duplicate GPU id: ${gpu}"
    SEEN["${gpu}"]=1
done

RESUME_FROM_PATH="${STAGE1_CKPT}"
RESUME_ONLINE_FROM_TARGET=1
RESET_RESUME_STEP=1
RESUME_OPTIMIZER_STATE=0
if [[ -n "${RESUME_STEP}" ]]; then
    RESUME_FROM_PATH="${OUTPUT_DIR}/checkpoints/step_${RESUME_STEP}"
    require_file "resume online transformer config" "${RESUME_FROM_PATH}/online_student/transformer/config.json"
    require_file "resume optimizer" "${RESUME_FROM_PATH}/optimizer.pt"
    RESUME_ONLINE_FROM_TARGET=0
    RESET_RESUME_STEP=0
    RESUME_OPTIMIZER_STATE=1
else
    [[ ! -e "${OUTPUT_DIR}" ]] || die "Refusing to reuse existing OUTPUT_DIR: ${OUTPUT_DIR}"
fi

CONFIG_FILE="distillation_flowmap.config_libero_fullfinetune_stage2_video_only_opd"
launch_env=(
    "CONFIG_FILE=${CONFIG_FILE}"
    "OUTPUT_DIR=${OUTPUT_DIR}"
    "RESUME_FROM_PATH=${RESUME_FROM_PATH}"
    "RESUME_ONLINE_FROM_TARGET=${RESUME_ONLINE_FROM_TARGET}"
    "RESET_RESUME_STEP=${RESET_RESUME_STEP}"
    "RESUME_OPTIMIZER_STATE=${RESUME_OPTIMIZER_STATE}"
    "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
    "SAVE_INTERVAL=${SAVE_INTERVAL}"
    "TRAIN_SEED=42"
    "OPD_QUERY_MODE=danceopd"
    "OPD_ROLLOUT_STEP_PAIRS=8,1;8,2;8,4"
    "OPD_DANCEOPD_ROLLOUT_STEPS=2,4"
    "OPD_DANCEOPD_ENDPOINT_WEIGHT=1.0"
    "OPD_DANCEOPD_VELOCITY_WEIGHT=1.0"
    "OPD_DANCEOPD_ACTION_VELOCITY_WEIGHT=0.0"
    "OPD_AUX_ACTION=0"
    "OPD_JOINT_ACTION_ROLLOUT=0"
    "OPD_AUX_INTERVAL=4"
    "OPD_AUX_PROB=1.0"
    "OPD_DANCEOPD_TERMINAL_PRIOR_TOLERANCE=2e-6"
    "MECHANISM_DIAGNOSTICS=1"
    "MECHANISM_DIAGNOSTIC_INTERVAL=50"
    "MECHANISM_DIAGNOSTIC_SEED=42"
    "MECHANISM_DIAGNOSTIC_R=500"
    "MECHANISM_DIAGNOSTIC_S=250"
    "MECHANISM_DIAGNOSTIC_TEACHER_STEPS=8"
    "GRADIENT_CHECKPOINTING=1"
    "OPD_AUX_GRADIENT_CHECKPOINTING=1"
    "USE_FSDP1=1"
    "SKIP_TEACHER_COMPILE=1"
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    "HF_DATASETS_OFFLINE=1"
    "TRANSFORMERS_OFFLINE=1"
    "HF_HUB_OFFLINE=1"
    "WANDB_MODE=offline"
    "ENABLE_WANDB=1"
    "WANDB_DIR=${OUTPUT_DIR}/wandb"
    "ENABLE_LIGHT_EVAL=0"
    "ENABLE_ROLLOUT_EVAL=0"
    "CUDA_VISIBLE_DEVICES=${GPU_LIST}"
)
command=(
    "${PYTHON}" -m torch.distributed.run
    "--nproc_per_node=8"
    "--master_port=${MASTER_PORT}"
    "${SCRIPT_DIR}/train.py"
    --teacher-model-path "${TEACHER_MODEL_PATH}"
    --dataset-path "${DATASET_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --resume-from-path "${RESUME_FROM_PATH}"
    --gradient-accumulation-steps 2
)

for item in "${launch_env[@]}"; do printf '%s\n' "${item}"; done
printf 'MASTER_PORT=%s\n' "${MASTER_PORT}"
printf 'COMMAND='
printf '%q ' "${command[@]}"
printf '\n'
if [[ "${DRY_RUN}" == 1 ]]; then exit 0; fi

mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${launch_env[@]}" >"${OUTPUT_DIR}/launch_env.txt"
printf '%q ' "${command[@]}" >"${OUTPUT_DIR}/launch_command.txt"
printf '\n' >>"${OUTPUT_DIR}/launch_command.txt"
cd "${PROJECT_ROOT}"
exec env "${launch_env[@]}" "${command[@]}"
