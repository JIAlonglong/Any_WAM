#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ROOT="${ROOT:-${PROJECT_ROOT}/distillation_flowmap/output_robotwin_stepwam_ablation/protocol_opd_mechanism_calibration_v1}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${PROJECT_ROOT}/checkpoints/lingbot-va-posttrain-robotwin}"
DATASET_PATH="${DATASET_PATH:-/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/empty_emb.pt}"
SHARED_STAGE1_CKPT="${SHARED_STAGE1_CKPT:-${PROJECT_ROOT}/distillation_flowmap/output_robotwin_stepwam_ablation/protocol_final12_5000/shared_stage1/representative/seed_0/stage1/checkpoints/step_5000}"
PYTHON="${PYTHON:-/root/nas/junjie/conda_envs/any_wam/bin/python}"
TORCHRUN="${TORCHRUN:-/root/nas/junjie/conda_envs/any_wam/bin/torchrun}"

VARIANTS="${VARIANTS:-calib_w_o_opd,calib_endpoint_only,calib_velocity_only,calib_full_last_step,calib_full_suffix_grad}"
SEED="${SEED:-0}"
TASK_PRESET="${TASK_PRESET:-core2}"
STAGE2_STEPS="${STAGE2_STEPS:-750}"
MAX_EPISODES_PER_TASK="${MAX_EPISODES_PER_TASK:-30}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-30}"
TRAIN_SAMPLES_PER_TASK="${TRAIN_SAMPLES_PER_TASK:-20}"
HELDOUT_SAMPLES_PER_TASK="${HELDOUT_SAMPLES_PER_TASK:-10}"
PROTOCOL_SEED="${PROTOCOL_SEED:-0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-33900}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_LARGE_CALIBRATION="${ALLOW_LARGE_CALIBRATION:-0}"

if [[ ! "${STAGE2_STEPS}" =~ ^[0-9]+$ ]] \
  || [[ ! "${TRAIN_SAMPLES_PER_TASK}" =~ ^[0-9]+$ ]] \
  || [[ ! "${HELDOUT_SAMPLES_PER_TASK}" =~ ^[0-9]+$ ]]; then
  echo "Calibration steps and sample counts must be non-negative integers." >&2
  exit 2
fi

expanded_protocol=0
if [[ "${TASK_PRESET}" != "core2" ]] \
  || (( TRAIN_SAMPLES_PER_TASK > 20 )) \
  || (( HELDOUT_SAMPLES_PER_TASK > 10 )) \
  || (( MAX_EPISODES_PER_TASK > 30 )) \
  || (( MAX_SAMPLES_PER_TASK > 30 )) \
  || (( STAGE2_STEPS > 1000 )); then
  expanded_protocol=1
fi
if (( expanded_protocol == 1 )) && [[ "${ALLOW_LARGE_CALIBRATION}" != "1" ]]; then
  echo "Refusing expanded calibration; set ALLOW_LARGE_CALIBRATION=1 explicitly." >&2
  exit 2
fi

if [[ ! -d "${SHARED_STAGE1_CKPT}" ]]; then
  echo "Shared Stage1 checkpoint not found: ${SHARED_STAGE1_CKPT}" >&2
  exit 2
fi

TASKS_JSON="${SCRIPT_DIR}/robotwin_stepwam_tasks.json"
task_count="$(
  "${PYTHON}" -c \
    'import json, sys; data=json.load(open(sys.argv[1])); print(len(data["presets"][sys.argv[2]]))' \
    "${TASKS_JSON}" "${TASK_PRESET}"
)"
if [[ "${task_count}" != "2" ]] && [[ "${ALLOW_LARGE_CALIBRATION}" != "1" ]]; then
  echo "Refusing expanded calibration: ${TASK_PRESET} contains ${task_count} tasks." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}/preflight" "${ROOT}/logs"
IFS=',' read -r -a variant_list <<< "${VARIANTS}"

common_args=(
  --root "${ROOT}"
  --teacher-model-path "${TEACHER_MODEL_PATH}"
  --dataset-path "${DATASET_PATH}"
  --empty-emb-path "${EMPTY_EMB_PATH}"
  --torchrun "${TORCHRUN}"
  --task-preset "${TASK_PRESET}"
  --max-episodes-per-task "${MAX_EPISODES_PER_TASK}"
  --max-samples-per-task "${MAX_SAMPLES_PER_TASK}"
  --train-samples-per-task "${TRAIN_SAMPLES_PER_TASK}"
  --heldout-samples-per-task "${HELDOUT_SAMPLES_PER_TASK}"
  --protocol-seed "${PROTOCOL_SEED}"
  --stage2-steps "${STAGE2_STEPS}"
  --gradient-accumulation-steps 1
  --ngpu 1
  --stage1-ckpt "${SHARED_STAGE1_CKPT}"
  --stage "stage2"
)

for index in "${!variant_list[@]}"; do
  variant="${variant_list[$index]}"
  preflight_path="${ROOT}/preflight/${variant}_seed_${SEED}.txt"
  "${PYTHON}" distillation_flowmap/ablation/launch_robotwin_stepwam_ablation.py \
    --variant "${variant}" \
    --seed "${SEED}" \
    --master-port "$(( MASTER_PORT_BASE + index * 4 ))" \
    "${common_args[@]}" \
    --dry-run > "${preflight_path}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    cat "${preflight_path}"
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Calibration dry-run complete: ${ROOT}/preflight"
  exit 0
fi

pids_file="${ROOT}/train_pids.txt"
: > "${pids_file}"
active_pids=()
failures=0

wait_batch() {
  local pid
  local rc
  set +e
  for pid in "${active_pids[@]}"; do
    wait "${pid}"
    rc="$?"
    if [[ "${rc}" -ne 0 ]]; then
      failures=$(( failures + 1 ))
      echo "Calibration job ${pid} failed with exit code ${rc}." >&2
    fi
  done
  set -e
  active_pids=()
}

for index in "${!variant_list[@]}"; do
  variant="${variant_list[$index]}"
  gpu="$(( index % MAX_PARALLEL ))"
  port="$(( MASTER_PORT_BASE + 100 + index * 4 ))"
  log_path="${ROOT}/logs/${variant}_seed_${SEED}_$(date +%Y%m%d_%H%M%S).log"
  echo "Launching ${variant} seed ${SEED} on GPU ${gpu}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
    distillation_flowmap/ablation/launch_robotwin_stepwam_ablation.py \
    --variant "${variant}" \
    --seed "${SEED}" \
    --master-port "${port}" \
    "${common_args[@]}" > "${log_path}" 2>&1 &
  pid="$!"
  echo "${pid}" >> "${pids_file}"
  active_pids+=("${pid}")
  if [[ "${#active_pids[@]}" -ge "${MAX_PARALLEL}" ]]; then
    wait_batch
  fi
done
if [[ "${#active_pids[@]}" -gt 0 ]]; then
  wait_batch
fi

if [[ "${failures}" -ne 0 ]]; then
  echo "Calibration finished with ${failures} failed job(s)." >&2
  exit 1
fi
echo "All OPD mechanism calibration jobs finished under ${ROOT}."
