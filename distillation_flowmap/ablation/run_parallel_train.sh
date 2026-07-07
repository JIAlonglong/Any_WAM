#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROOT="${ROOT:-${PROJECT_ROOT}/distillation_flowmap/output_robotwin_stepwam_ablation}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${PROJECT_ROOT}/checkpoints/lingbot-va-posttrain-robotwin}"
DATASET_PATH="${DATASET_PATH:-/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/empty_emb.pt}"
TORCHRUN="${TORCHRUN:-/root/nas/junjie/conda_envs/any_wam/bin/torchrun}"
STAGE1_STEPS=5000
STAGE2_STEPS=5000
VARIANTS="full_stepwam,w_o_opd,endpoint_only_opd,velocity_only_opd,local_adjacent_only,action_only"
SEEDS="0"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29660}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --stage1-steps) STAGE1_STEPS="$2"; shift 2 ;;
    --stage2-steps) STAGE2_STEPS="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --priority) shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "${PROJECT_ROOT}"
IFS=',' read -r -a variant_list <<< "${VARIANTS}"
IFS=',' read -r -a seed_list <<< "${SEEDS}"

job_idx=0
pids=()
for seed in "${seed_list[@]}"; do
  for variant in "${variant_list[@]}"; do
    gpu=$(( job_idx % 8 ))
    port=$(( MASTER_PORT_BASE + job_idx * 4 ))
    log_dir="${ROOT}/${variant}/seed_${seed}/logs"
    mkdir -p "${log_dir}"
    log_file="${log_dir}/train_$(date +%Y%m%d_%H%M%S).log"
    echo "Launching ${variant} seed ${seed} on GPU ${gpu}; log=${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    /root/nas/junjie/conda_envs/any_wam/bin/python \
      distillation_flowmap/ablation/launch_robotwin_stepwam_ablation.py \
      --variant "${variant}" \
      --seed "${seed}" \
      --root "${ROOT}" \
      --teacher-model-path "${TEACHER_MODEL_PATH}" \
      --dataset-path "${DATASET_PATH}" \
      --empty-emb-path "${EMPTY_EMB_PATH}" \
      --torchrun "${TORCHRUN}" \
      --stage1-steps "${STAGE1_STEPS}" \
      --stage2-steps "${STAGE2_STEPS}" \
      --master-port "${port}" > "${log_file}" 2>&1 &
    pids+=("$!")
    job_idx=$(( job_idx + 1 ))
  done
done

printf "%s\n" "${pids[@]}" > "${ROOT}/train_pids.txt"
wait
