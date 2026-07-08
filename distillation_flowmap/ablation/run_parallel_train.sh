#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ROOT="${ROOT:-${PROJECT_ROOT}/distillation_flowmap/output_robotwin_stepwam_ablation/protocol_core4}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${PROJECT_ROOT}/checkpoints/lingbot-va-posttrain-robotwin}"
DATASET_PATH="${DATASET_PATH:-/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/empty_emb.pt}"
TORCHRUN="${TORCHRUN:-/root/nas/junjie/conda_envs/any_wam/bin/torchrun}"

STAGE1_STEPS="${STAGE1_STEPS:-5000}"
STAGE2_STEPS="${STAGE2_STEPS:-5000}"
VARIANTS="full_stepwam,w_o_opd,endpoint_only_opd,velocity_only_opd"
SEEDS="0,1,2"
TASK_PRESET="${TASK_PRESET:-core4}"
MAX_EPISODES_PER_TASK="${MAX_EPISODES_PER_TASK:-50}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-100}"
TRAIN_SAMPLES_PER_TASK="${TRAIN_SAMPLES_PER_TASK:-80}"
HELDOUT_SAMPLES_PER_TASK="${HELDOUT_SAMPLES_PER_TASK:-20}"
PROTOCOL_SEED="${PROTOCOL_SEED:-0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29660}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
STAGE="${STAGE:-both}"
DRY_RUN=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --stage1-steps) STAGE1_STEPS="$2"; shift 2 ;;
    --stage2-steps) STAGE2_STEPS="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --task-preset) TASK_PRESET="$2"; shift 2 ;;
    --max-episodes-per-task) MAX_EPISODES_PER_TASK="$2"; shift 2 ;;
    --max-samples-per-task) MAX_SAMPLES_PER_TASK="$2"; shift 2 ;;
    --train-samples-per-task) TRAIN_SAMPLES_PER_TASK="$2"; shift 2 ;;
    --heldout-samples-per-task) HELDOUT_SAMPLES_PER_TASK="$2"; shift 2 ;;
    --protocol-seed) PROTOCOL_SEED="$2"; shift 2 ;;
    --master-port-base) MASTER_PORT_BASE="$2"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --priority) shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "${STAGE}" != "both" ] && [ "${STAGE}" != "stage1" ] && [ "${STAGE}" != "stage2" ]; then
  echo "--stage must be one of: both, stage1, stage2" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}"
IFS=',' read -r -a variant_list <<< "${VARIANTS}"
IFS=',' read -r -a seed_list <<< "${SEEDS}"
pids_file="${ROOT}/train_pids.txt"
: > "${pids_file}"

common_args=(
  --root "${ROOT}"
  --teacher-model-path "${TEACHER_MODEL_PATH}"
  --dataset-path "${DATASET_PATH}"
  --empty-emb-path "${EMPTY_EMB_PATH}"
  --torchrun "${TORCHRUN}"
  --task-preset "${TASK_PRESET}"
  --max-episodes-per-task "${MAX_EPISODES_PER_TASK}"
  --max-samples-per-task "${MAX_SAMPLES_PER_TASK}"
  --protocol-seed "${PROTOCOL_SEED}"
  --train-samples-per-task "${TRAIN_SAMPLES_PER_TASK}"
  --heldout-samples-per-task "${HELDOUT_SAMPLES_PER_TASK}"
  --gradient-accumulation-steps 1
  --ngpu 1
  --use-shared-stage1
)

wait_batch() {
  local active_count="$1"
  if [ "${active_count}" -gt 0 ]; then
    wait
  fi
}

launch_job() {
  local phase="$1"
  local variant="$2"
  local seed="$3"
  local gpu="$4"
  local port="$5"
  local log_dir="${ROOT}/${variant}/seed_${seed}/logs"
  local log_file
  mkdir -p "${log_dir}"
  log_file="${log_dir}/${phase}_$(date +%Y%m%d_%H%M%S).log"
  echo "Launching ${phase} ${variant} seed ${seed} on GPU ${gpu}; log=${log_file}"
  local cmd=(
    /root/nas/junjie/conda_envs/any_wam/bin/python
    distillation_flowmap/ablation/launch_robotwin_stepwam_ablation.py
    --variant "${variant}"
    --seed "${seed}"
    "${common_args[@]}"
    --stage1-steps "${STAGE1_STEPS}"
    --stage2-steps "${STAGE2_STEPS}"
    --master-port "${port}"
    --stage "${phase}"
  )
  if [ "${DRY_RUN}" = "1" ]; then
    cmd+=(--dry-run)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${log_file}" 2>&1 &
  echo "$!" >> "${pids_file}"
}

if [ "${STAGE}" = "both" ] || [ "${STAGE}" = "stage1" ]; then
  active=0
  job_idx=0
  for seed in "${seed_list[@]}"; do
    gpu=$(( job_idx % MAX_PARALLEL ))
    port=$(( MASTER_PORT_BASE + job_idx * 4 ))
    launch_job stage1 full_stepwam "${seed}" "${gpu}" "${port}"
    active=$(( active + 1 ))
    job_idx=$(( job_idx + 1 ))
    if [ "${active}" -ge "${MAX_PARALLEL}" ]; then
      wait_batch "${active}"
      active=0
    fi
  done
  wait_batch "${active}"
fi

if [ "${STAGE}" = "both" ] || [ "${STAGE}" = "stage2" ]; then
  active=0
  job_idx=0
  for seed in "${seed_list[@]}"; do
    for variant in "${variant_list[@]}"; do
      gpu=$(( job_idx % MAX_PARALLEL ))
      port=$(( MASTER_PORT_BASE + 100 + job_idx * 4 ))
      launch_job stage2 "${variant}" "${seed}" "${gpu}" "${port}"
      active=$(( active + 1 ))
      job_idx=$(( job_idx + 1 ))
      if [ "${active}" -ge "${MAX_PARALLEL}" ]; then
        wait_batch "${active}"
        active=0
      fi
    done
  done
  wait_batch "${active}"
fi

echo "All requested RobotWin mini-ablation jobs finished. PIDs were written to ${pids_file}"
