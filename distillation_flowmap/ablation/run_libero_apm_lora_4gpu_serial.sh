#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if common_git_dir="$(git -C "${PROJECT_ROOT}" rev-parse --git-common-dir 2>/dev/null)"; then
  if [[ "${common_git_dir}" != /* ]]; then
    common_git_dir="${PROJECT_ROOT}/${common_git_dir}"
  fi
  SHARED_ROOT="$(cd "$(dirname "${common_git_dir}")" && pwd)"
else
  SHARED_ROOT="${PROJECT_ROOT}"
fi

PHASE="all"
STEPS=500
SAVE_INTERVAL=100
EPISODES=20
GPU_IDS="0,1,2,3"
ARMS_CSV="stage1_only,anchor_only,field_only,apm"
MASTER_PORT_BASE=29670
OUTPUT_ROOT="${SHARED_ROOT}/distillation_flowmap/output_libero_apm_lora_ablation_20260724"
PYTHON="${PYTHON:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"
TORCHRUN="${TORCHRUN:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/torchrun}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${SHARED_ROOT}/../lingbot-va/checkpoints/libero}"
DATASET_PATH="${DATASET_PATH:-${SHARED_ROOT}/training_data/libero-long-lerobot}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-${DATASET_PATH}/empty_emb.pt}"
STAGE1_CKPT="${STAGE1_CKPT:-${SHARED_ROOT}/distillation_flowmap/output_libero_fullft_stage1_warmup/checkpoints/step_2000}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: run_libero_apm_lora_4gpu_serial.sh [options]

  --phase train|offline-eval|closed-loop|all
  --steps N                    Stage-2 steps (default: 500)
  --save-interval N            Checkpoint interval (default: 100)
  --episodes N                 Closed-loop episodes per job (default: 20)
  --gpu-ids 0,1,2,3            Exactly four unique local GPU IDs
  --arms CSV                   Ordered subset of stage1_only,anchor_only,field_only,apm
  --master-port-base PORT
  --output-root PATH
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="${2:?missing value for --phase}"; shift 2 ;;
    --steps) STEPS="${2:?missing value for --steps}"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="${2:?missing value for --save-interval}"; shift 2 ;;
    --episodes) EPISODES="${2:?missing value for --episodes}"; shift 2 ;;
    --gpu-ids) GPU_IDS="${2:?missing value for --gpu-ids}"; shift 2 ;;
    --arms) ARMS_CSV="${2:?missing value for --arms}"; shift 2 ;;
    --master-port-base) MASTER_PORT_BASE="${2:?missing value for --master-port-base}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?missing value for --output-root}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "${PHASE}" in
  train|offline-eval|closed-loop|all) ;;
  *) die "invalid phase ${PHASE}; expected train, offline-eval, closed-loop, or all" ;;
esac
for numeric in "${STEPS}" "${SAVE_INTERVAL}" "${EPISODES}" "${MASTER_PORT_BASE}"; do
  [[ "${numeric}" =~ ^[1-9][0-9]*$ ]] || die "steps, intervals, episodes, and ports must be positive integers"
done

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
[[ "${#GPU_LIST[@]}" -eq 4 ]] || die "gpu selection must contain exactly four unique IDs"
declare -A SEEN_GPUS=()
for gpu in "${GPU_LIST[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "GPU IDs must be non-negative integers"
  [[ -z "${SEEN_GPUS[${gpu}]:-}" ]] || die "gpu selection must contain exactly four unique IDs"
  SEEN_GPUS["${gpu}"]=1
done

IFS=',' read -r -a ARMS <<< "${ARMS_CSV}"
[[ "${#ARMS[@]}" -gt 0 ]] || die "at least one arm is required"
declare -A SEEN_ARMS=()
for arm in "${ARMS[@]}"; do
  case "${arm}" in
    stage1_only|anchor_only|field_only|apm) ;;
    *) die "unknown arm: ${arm}" ;;
  esac
  [[ -z "${SEEN_ARMS[${arm}]:-}" ]] || die "duplicate arm: ${arm}"
  SEEN_ARMS["${arm}"]=1
done

PLANNER="${SCRIPT_DIR}/launch_libero_apm_ablation.py"
OFFLINE_EVAL="${PROJECT_ROOT}/distillation_flowmap/rollout_eval_video_stage2.py"
CLOSED_LOOP_EVAL="${PROJECT_ROOT}/evaluation/libero/run_lingbotva_task0_124_eval_4gpu.sh"
PROTOCOL_ROOT="${OUTPUT_ROOT}/protocol"
TRAIN_MANIFEST="${PROTOCOL_ROOT}/train_manifest.json"
HELDOUT_MANIFEST="${PROTOCOL_ROOT}/heldout_manifest.json"

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

run_training() {
  local index="$1"
  local arm="$2"
  local port="$(( MASTER_PORT_BASE + index ))"
  local -a command=(
    "${PYTHON}" "${PLANNER}"
    --variant "${arm}"
    --output-root "${OUTPUT_ROOT}"
    --teacher-model-path "${TEACHER_MODEL_PATH}"
    --dataset-path "${DATASET_PATH}"
    --empty-emb-path "${EMPTY_EMB_PATH}"
    --stage1-ckpt "${STAGE1_CKPT}"
    --torchrun "${TORCHRUN}"
    --steps "${STEPS}"
    --save-interval "${SAVE_INTERVAL}"
    --master-port "${port}"
    --gpu-ids "${GPU_IDS}"
  )
  echo "TRAIN arm=${arm} port=${port} manifest=${TRAIN_MANIFEST}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${command[@]}" --dry-run
    return
  fi
  mkdir -p "${OUTPUT_ROOT}/launcher_logs"
  PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${command[@]}" --execute \
    > "${OUTPUT_ROOT}/launcher_logs/${arm}_train.log" 2>&1
}

run_offline_eval() {
  local arm="$1"
  local stage2_dir="${OUTPUT_ROOT}/${arm}/seed_42/stage2"
  local checkpoint="${stage2_dir}/checkpoints/step_${STEPS}"
  local result_dir="${OUTPUT_ROOT}/${arm}/seed_42/offline_eval"
  local result_json="${result_dir}/heldout.json"
  local -a command=(
    "${PYTHON}" "${OFFLINE_EVAL}"
    --config distillation_flowmap.config_libero_apm_lora_ablation
    --teacher-model-path "${TEACHER_MODEL_PATH}"
    --dataset-path "${DATASET_PATH}"
    --empty-emb-path "${EMPTY_EMB_PATH}"
    --output-dir "${stage2_dir}"
    --resume-from-path "${checkpoint}"
    --result-json "${result_json}"
    --num-batches 10
    --eval-manifest "${HELDOUT_MANIFEST}"
    --split-name heldout
    --student-steps 1 2 4
    --teacher-steps 1 2 4
    --load-target-student
  )
  echo "OFFLINE_EVAL arm=${arm} checkpoint=${checkpoint} manifest=${HELDOUT_MANIFEST}"
  print_command "${command[@]}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return
  fi
  [[ -d "${checkpoint}" ]] || die "missing final checkpoint for ${arm}: ${checkpoint}"
  [[ -f "${HELDOUT_MANIFEST}" ]] || die "missing heldout manifest: ${HELDOUT_MANIFEST}"
  mkdir -p "${result_dir}"
  DATASET_SAMPLE_MANIFEST="${HELDOUT_MANIFEST}" \
  RESUME_ONLINE_FROM_TARGET=1 \
  HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  WANDB_MODE=offline \
    "${command[@]}" > "${result_dir}/heldout.log" 2>&1
}

run_closed_loop() {
  local eval_root="${OUTPUT_ROOT}/closed_loop_task0"
  local -a command=(
    bash "${CLOSED_LOOP_EVAL}"
    --stage1-checkpoint "${STAGE1_CKPT}"
    --ablation-root "${OUTPUT_ROOT}"
    --train-steps "${STEPS}"
    --episodes "${EPISODES}"
    --gpu-ids "${GPU_IDS}"
    --master-port-base "$(( MASTER_PORT_BASE + 100 ))"
    --ws-port-base "$(( MASTER_PORT_BASE + 200 ))"
    --output-root "${eval_root}"
  )
  echo "CLOSED_LOOP models=stage1,stage1_only,anchor_only,field_only,apm task=libero_10:0 episodes=${EPISODES} budgets=1,2,4"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    CHECK_ONLY=1 "${command[@]}"
    return
  fi
  "${command[@]}"
}

cd "${PROJECT_ROOT}"
if [[ "${PHASE}" == "train" || "${PHASE}" == "all" ]]; then
  for index in "${!ARMS[@]}"; do
    run_training "${index}" "${ARMS[${index}]}"
  done
fi

if [[ "${PHASE}" == "offline-eval" || "${PHASE}" == "all" ]]; then
  for arm in "${ARMS[@]}"; do
    run_offline_eval "${arm}"
  done
fi

if [[ "${PHASE}" == "closed-loop" || "${PHASE}" == "all" ]]; then
  run_closed_loop
fi
