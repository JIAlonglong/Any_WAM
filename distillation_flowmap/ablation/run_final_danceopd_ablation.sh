#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ROOT="${ROOT:-${PROJECT_ROOT}/distillation_flowmap/output_robotwin_stepwam_ablation/protocol_final12_danceopd_i1_single_seed_v1}"
SHARED_STAGE1_CKPT="${SHARED_STAGE1_CKPT:-${PROJECT_ROOT}/distillation_flowmap/output_robotwin_stepwam_ablation/protocol_final12_5000/shared_stage1/representative/seed_0/stage1/checkpoints/step_5000}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${PROJECT_ROOT}/checkpoints/lingbot-va-posttrain-robotwin}"
DATASET_PATH="${DATASET_PATH:-/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/empty_emb.pt}"
PYTHON="${PYTHON:-/root/nas/junjie/conda_envs/any_wam/bin/python}"

STAGE1_STEPS="${STAGE1_STEPS:-5000}"
STAGE2_STEPS="${STAGE2_STEPS:-5000}"
TASK_PRESET="${TASK_PRESET:-representative}"
MAX_EPISODES_PER_TASK="${MAX_EPISODES_PER_TASK:-50}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-50}"
TRAIN_SAMPLES_PER_TASK="${TRAIN_SAMPLES_PER_TASK:-40}"
HELDOUT_SAMPLES_PER_TASK="${HELDOUT_SAMPLES_PER_TASK:-10}"
PROTOCOL_SEED="${PROTOCOL_SEED:-0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-38000}"
MODE="${MODE:-all}"
DRY_RUN=0

MAIN_VARIANTS=(
  final_w_o_opd
  final_endpoint_only_danceopd
  final_danceopd_velocity_only
  final_stepwam_danceopd
)
CONTROL_VARIANTS=(
  final_local_adjacent_only
  final_action_only
)

usage() {
  printf '%s\n' "Usage: $0 [--main-only|--controls-only|--dry-run] [--root PATH] [--shared-stage1-ckpt PATH]"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --main-only) MODE="main"; shift ;;
    --controls-only) MODE="controls"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --root) ROOT="$2"; shift 2 ;;
    --shared-stage1-ckpt) SHARED_STAGE1_CKPT="$2"; shift 2 ;;
    --stage1-steps) STAGE1_STEPS="$2"; shift 2 ;;
    --stage2-steps) STAGE2_STEPS="$2"; shift 2 ;;
    --protocol-seed) PROTOCOL_SEED="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ "${MODE}" != "all" ] && [ "${MODE}" != "main" ] && [ "${MODE}" != "controls" ]; then
  echo "MODE must be all, main, or controls" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"

prepare_protocol() {
  "${PYTHON}" - "${ROOT}" "${TASK_PRESET}" "${PROTOCOL_SEED}" "${TRAIN_SAMPLES_PER_TASK}" "${HELDOUT_SAMPLES_PER_TASK}" "${MAIN_VARIANTS[@]}" <<'PY'
import sys
from pathlib import Path

from distillation_flowmap.ablation.launch_robotwin_stepwam_ablation import (
    load_metadata,
    task_preset_filter,
)
from distillation_flowmap.ablation.robotwin_mini_protocol import (
    DEFAULT_EVAL_PAIRS,
    write_protocol_manifests,
)

root = Path(sys.argv[1])
task_preset = sys.argv[2]
protocol_seed = int(sys.argv[3])
train_samples = int(sys.argv[4])
heldout_samples = int(sys.argv[5])
expected_variants = sys.argv[6:]

metadata, tasks = load_metadata()
variants = {entry["name"]: entry for entry in metadata["variants"]}
missing = [name for name in expected_variants if name not in variants]
if missing:
    raise SystemExit(f"Missing final-ablation variants: {missing}")
for name in expected_variants:
    if variants[name].get("seeds") != [0]:
        raise SystemExit(f"{name} must be frozen to seeds=[0], got {variants[name].get('seeds')}")

task_names = task_preset_filter(tasks, task_preset)
if len(task_names) != 12:
    raise SystemExit(f"Expected 12 representative tasks, got {len(task_names)}: {task_names}")
write_protocol_manifests(
    root=root,
    task_preset=task_preset,
    protocol_seed=protocol_seed,
    task_names=task_names,
    train_samples_per_task=train_samples,
    heldout_samples_per_task=heldout_samples,
    pairs=DEFAULT_EVAL_PAIRS,
)
PY
}

require_path() {
  local path="$1"
  if [ ! -e "${path}" ]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
}

check_target_is_fresh() {
  local variant="$1"
  local checkpoint="${ROOT}/${variant}/seed_0/stage2/checkpoints/step_${STAGE2_STEPS}"
  if [ -e "${checkpoint}" ]; then
    echo "Refusing to overwrite completed checkpoint: ${checkpoint}" >&2
    exit 1
  fi
}

declare -a pids=()
declare -a labels=()
failures=0
job_index=0

launch_job() {
  local variant="$1"
  local stage="$2"
  local gpu="$3"
  local uses_shared_stage1="$4"
  local log_dir="${ROOT}/${variant}/seed_0/logs"
  local log_file="${log_dir}/${stage}_$(date +%Y%m%d_%H%M%S).log"
  local port=$(( MASTER_PORT_BASE + job_index * 4 ))
  local -a command=(
    "${PYTHON}"
    distillation_flowmap/ablation/launch_robotwin_stepwam_ablation.py
    --variant "${variant}"
    --seed 0
    --root "${ROOT}"
    --teacher-model-path "${TEACHER_MODEL_PATH}"
    --dataset-path "${DATASET_PATH}"
    --empty-emb-path "${EMPTY_EMB_PATH}"
    --stage1-steps "${STAGE1_STEPS}"
    --stage2-steps "${STAGE2_STEPS}"
    --task-preset "${TASK_PRESET}"
    --max-episodes-per-task "${MAX_EPISODES_PER_TASK}"
    --max-samples-per-task "${MAX_SAMPLES_PER_TASK}"
    --train-samples-per-task "${TRAIN_SAMPLES_PER_TASK}"
    --heldout-samples-per-task "${HELDOUT_SAMPLES_PER_TASK}"
    --protocol-seed "${PROTOCOL_SEED}"
    --gradient-accumulation-steps 1
    --ngpu 1
    --master-port "${port}"
    --stage "${stage}"
  )
  if [ "${uses_shared_stage1}" = "1" ]; then
    command+=(--stage1-ckpt "${SHARED_STAGE1_CKPT}")
  fi
  job_index=$(( job_index + 1 ))

  if [ "${DRY_RUN}" = "1" ]; then
    printf "DRY_RUN [%s]: env CUDA_VISIBLE_DEVICES=%q" "${variant}" "${gpu}"
    printf " %q" "${command[@]}"
    printf "\n"
    return
  fi

  mkdir -p "${log_dir}"
  echo "Launching ${variant} (${stage}) on GPU ${gpu}; log=${log_file}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${command[@]}"
  ) > "${log_file}" 2>&1 &
  pids+=("$!")
  labels+=("${variant}:${stage}:gpu${gpu}:${log_file}")
}

if [ "${DRY_RUN}" = "0" ]; then
  require_path "${SHARED_STAGE1_CKPT}"
  require_path "${TEACHER_MODEL_PATH}"
  require_path "${DATASET_PATH}"
  require_path "${EMPTY_EMB_PATH}"
  if [ "${MODE}" = "all" ] || [ "${MODE}" = "main" ]; then
    for variant in "${MAIN_VARIANTS[@]}"; do
      check_target_is_fresh "${variant}"
    done
  fi
  if [ "${MODE}" = "all" ] || [ "${MODE}" = "controls" ]; then
    for variant in "${CONTROL_VARIANTS[@]}"; do
      check_target_is_fresh "${variant}"
    done
  fi
  prepare_protocol
fi

if [ "${MODE}" = "all" ] || [ "${MODE}" = "main" ]; then
  launch_job final_w_o_opd stage2 0 1
  launch_job final_endpoint_only_danceopd stage2 1 1
  launch_job final_danceopd_velocity_only stage2 2 1
  launch_job final_stepwam_danceopd stage2 3 1
fi

if [ "${MODE}" = "all" ] || [ "${MODE}" = "controls" ]; then
  launch_job final_local_adjacent_only both 4 0
  launch_job final_action_only both 5 0
fi

if [ "${DRY_RUN}" = "1" ]; then
  exit 0
fi

for idx in "${!pids[@]}"; do
  if wait "${pids[${idx}]}"; then
    echo "Completed ${labels[${idx}]}"
  else
    echo "FAILED ${labels[${idx}]}" >&2
    failures=$(( failures + 1 ))
  fi
done

if [ "${failures}" -ne 0 ]; then
  echo "Final ablation finished with ${failures} failed job(s)." >&2
  exit 1
fi

echo "Final ablation training completed. Run ablation/run_final_danceopd_eval.sh after cached video metrics are available."
