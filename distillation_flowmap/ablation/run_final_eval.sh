#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ROOT="${ROOT:-${PROJECT_ROOT}/distillation_flowmap/output_robotwin_stepwam_ablation/protocol_final12_5000}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${PROJECT_ROOT}/checkpoints/lingbot-va-posttrain-robotwin}"
DATASET_PATH="${DATASET_PATH:-/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/empty_emb.pt}"
TORCHRUN="${TORCHRUN:-/root/nas/junjie/conda_envs/any_wam/bin/torchrun}"
PYTHON="${PYTHON:-/root/nas/junjie/conda_envs/any_wam/bin/python}"

STAGE2_STEPS="${STAGE2_STEPS:-5000}"
VARIANTS="${VARIANTS:-full_stepwam,w_o_opd,endpoint_only_opd,velocity_only_opd,local_adjacent_only,action_only}"
SEEDS="${SEEDS:-0,1,2}"
PROTOCOL_SEED="${PROTOCOL_SEED:-0}"
TASK_PRESET="${TASK_PRESET:-representative}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-34000}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
EVAL_GPUS="${EVAL_GPUS:-6,7}"
TEACHER_STEPS="${TEACHER_STEPS:-4}"
TEACHER_CACHE_SOURCE_VARIANT="${TEACHER_CACHE_SOURCE_VARIANT:-full_stepwam}"
TEACHER_CACHE_SOURCE_SEED="${TEACHER_CACHE_SOURCE_SEED:-0}"
BASELINE_VARIANT="${BASELINE_VARIANT:-w_o_opd}"

RUN_HELDOUT_EVAL="${RUN_HELDOUT_EVAL:-1}"
RUN_TRAIN_EVAL="${RUN_TRAIN_EVAL:-1}"
RUN_VIDEO_EVAL="${RUN_VIDEO_EVAL:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
TRAIN_EVAL_SAMPLES_PER_TASK="${TRAIN_EVAL_SAMPLES_PER_TASK:-10}"
VIDEO_VARIANTS="${VIDEO_VARIANTS:-${VARIANTS}}"
VIDEO_SEEDS="${VIDEO_SEEDS:-0}"
VIDEO_MAX_PAIRS="${VIDEO_MAX_PAIRS:-1}"
DRY_RUN=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --stage2-steps) STAGE2_STEPS="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --eval-gpus) EVAL_GPUS="$2"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --skip-heldout-eval) RUN_HELDOUT_EVAL=0; shift ;;
    --skip-train-eval) RUN_TRAIN_EVAL=0; shift ;;
    --skip-video-eval) RUN_VIDEO_EVAL=0; shift ;;
    --skip-summary) RUN_SUMMARY=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "${PROJECT_ROOT}"
IFS=',' read -r -a variant_list <<< "${VARIANTS}"
IFS=',' read -r -a seed_list <<< "${SEEDS}"
IFS=',' read -r -a eval_gpu_list <<< "${EVAL_GPUS}"
IFS=',' read -r -a video_variant_list <<< "${VIDEO_VARIANTS}"
IFS=',' read -r -a video_seed_list <<< "${VIDEO_SEEDS}"

if [ "${#eval_gpu_list[@]}" -eq 0 ]; then
  echo "EVAL_GPUS must contain at least one GPU id" >&2
  exit 2
fi

PROTOCOL_DIR="${ROOT}/protocol/manifests/${TASK_PRESET}_protocol_seed_${PROTOCOL_SEED}"
TRAIN_MANIFEST="${PROTOCOL_DIR}/train_manifest.json"
HELDOUT_MANIFEST="${PROTOCOL_DIR}/heldout_eval_manifest.json"
TRAIN_EVAL_MANIFEST="${PROTOCOL_DIR}/train_eval_manifest.json"
EVAL_PAIRS="${PROTOCOL_DIR}/eval_pairs.json"
TEACHER_CACHE="${TEACHER_CACHE:-${ROOT}/protocol/teacher_cache_heldout_final.pt}"
TRAIN_TEACHER_CACHE="${TRAIN_TEACHER_CACHE:-${ROOT}/protocol/teacher_cache_train_eval_final.pt}"
SUMMARY_DIR="${SUMMARY_DIR:-${ROOT}/summary_final}"
mkdir -p "${ROOT}/protocol" "${SUMMARY_DIR}"

run_or_print() {
  if [ "${DRY_RUN}" = "1" ]; then
    printf "DRY_RUN:"
    printf " %q" "$@"
    printf "\n"
  else
    "$@"
  fi
}

require_path() {
  local path="$1"
  if [ "${DRY_RUN}" = "1" ]; then
    return
  fi
  if [ ! -e "${path}" ]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
}

stage2_ckpt() {
  local variant="$1"
  local seed="$2"
  printf "%s/%s/seed_%s/stage2/checkpoints/step_%s" "${ROOT}" "${variant}" "${seed}" "${STAGE2_STEPS}"
}

make_balanced_train_eval_manifest() {
  require_path "${TRAIN_MANIFEST}"
  if [ -f "${TRAIN_EVAL_MANIFEST}" ]; then
    return
  fi
  run_or_print "${PYTHON}" - "${TRAIN_MANIFEST}" "${TRAIN_EVAL_MANIFEST}" "${TRAIN_EVAL_SAMPLES_PER_TASK}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
samples_per_task = int(sys.argv[3])
payload = json.loads(src.read_text(encoding="utf-8"))
payload["schema"] = "robotwin_stepwam_train_eval_manifest_v1"
payload["split"] = "train_eval"
payload["samples_per_task"] = samples_per_task
for task in payload.get("tasks", []):
    task["indices"] = list(task.get("indices", []))[:samples_per_task]
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

build_teacher_cache() {
  local cache_path="$1"
  local manifest_path="$2"
  local split_name="$3"
  local ckpt
  ckpt="$(stage2_ckpt "${TEACHER_CACHE_SOURCE_VARIANT}" "${TEACHER_CACHE_SOURCE_SEED}")"
  require_path "${ckpt}"
  require_path "${manifest_path}"
  require_path "${EVAL_PAIRS}"
  run_or_print env CUDA_VISIBLE_DEVICES="${eval_gpu_list[0]}" "${TORCHRUN}" \
    --nproc_per_node=1 \
    --master_port="${MASTER_PORT_BASE}" \
    distillation_flowmap/rollout_eval_stage2.py \
    --config distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow \
    --teacher-model-path "${TEACHER_MODEL_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --empty-emb-path "${EMPTY_EMB_PATH}" \
    --output-dir "${ROOT}/eval/teacher_cache_${split_name}" \
    --resume-from-path "${ckpt}" \
    --result-json "${ROOT}/eval/teacher_cache_${split_name}.json" \
    --teacher-cache-path "${cache_path}" \
    --teacher-cache-only \
    --eval-manifest "${manifest_path}" \
    --eval-pairs-json "${EVAL_PAIRS}" \
    --split-name "${split_name}" \
    --num-batches 0 \
    --student-steps 4 \
    --teacher-steps "${TEACHER_STEPS}" \
    --disable-eval-gradient-checkpointing \
    --disable-eval-force-cfg \
    --eval-rollout-grad-mode endpoint \
    --eval-empty-cache
}

active_pids=()
failures=0

wait_batch() {
  local label="$1"
  local pid
  local rc
  local batch_failures=0
  if [ "${#active_pids[@]}" -eq 0 ]; then
    return
  fi
  set +e
  for pid in "${active_pids[@]}"; do
    wait "${pid}"
    rc="$?"
    if [ "${rc}" -ne 0 ]; then
      batch_failures=$(( batch_failures + 1 ))
      echo "Eval job ${pid} in ${label} failed with exit code ${rc}; inspect ${ROOT}" >&2
    fi
  done
  set -e
  active_pids=()
  if [ "${batch_failures}" -ne 0 ]; then
    failures=$(( failures + batch_failures ))
  fi
}

launch_rollout_eval() {
  local split_name="$1"
  local manifest_path="$2"
  local result_name="$3"
  local cache_path="$4"
  local variant="$5"
  local seed="$6"
  local job_idx="$7"
  local gpu="${eval_gpu_list[$(( job_idx % ${#eval_gpu_list[@]} ))]}"
  local port=$(( MASTER_PORT_BASE + 10 + job_idx ))
  local run_dir="${ROOT}/${variant}/seed_${seed}"
  local ckpt
  ckpt="$(stage2_ckpt "${variant}" "${seed}")"
  require_path "${ckpt}"
  mkdir -p "${run_dir}/metrics" "${run_dir}/eval_${split_name}"
  echo "Launching ${split_name} rollout eval ${variant} seed ${seed} on GPU ${gpu}"
  if [ "${DRY_RUN}" = "1" ]; then
    run_or_print env CUDA_VISIBLE_DEVICES="${gpu}" "${TORCHRUN}" \
      --nproc_per_node=1 --master_port="${port}" distillation_flowmap/rollout_eval_stage2.py \
      --config distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow \
      --teacher-model-path "${TEACHER_MODEL_PATH}" --dataset-path "${DATASET_PATH}" \
      --empty-emb-path "${EMPTY_EMB_PATH}" --output-dir "${run_dir}/eval_${split_name}" \
      --resume-from-path "${ckpt}" --result-json "${run_dir}/metrics/${result_name}.json" \
      --teacher-cache-path "${cache_path}" --eval-manifest "${manifest_path}" \
      --eval-pairs-json "${EVAL_PAIRS}" --split-name "${split_name}" --num-batches 0 \
      --student-steps 4 --teacher-steps "${TEACHER_STEPS}" --disable-eval-gradient-checkpointing \
      --disable-eval-force-cfg --eval-rollout-grad-mode endpoint --eval-empty-cache
    return
  fi
  env CUDA_VISIBLE_DEVICES="${gpu}" "${TORCHRUN}" \
    --nproc_per_node=1 \
    --master_port="${port}" \
    distillation_flowmap/rollout_eval_stage2.py \
    --config distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow \
    --teacher-model-path "${TEACHER_MODEL_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --empty-emb-path "${EMPTY_EMB_PATH}" \
    --output-dir "${run_dir}/eval_${split_name}" \
    --resume-from-path "${ckpt}" \
    --result-json "${run_dir}/metrics/${result_name}.json" \
    --teacher-cache-path "${cache_path}" \
    --eval-manifest "${manifest_path}" \
    --eval-pairs-json "${EVAL_PAIRS}" \
    --split-name "${split_name}" \
    --num-batches 0 \
    --student-steps 4 \
    --teacher-steps "${TEACHER_STEPS}" \
    --disable-eval-gradient-checkpointing \
    --disable-eval-force-cfg \
    --eval-rollout-grad-mode endpoint \
    --eval-empty-cache > "${run_dir}/metrics/${result_name}.log" 2>&1 &
  active_pids+=("$!")
}

run_rollout_grid() {
  local split_name="$1"
  local manifest_path="$2"
  local result_name="$3"
  local cache_path="$4"
  local active=0
  local job_idx=0
  for seed in "${seed_list[@]}"; do
    for variant in "${variant_list[@]}"; do
      launch_rollout_eval "${split_name}" "${manifest_path}" "${result_name}" "${cache_path}" "${variant}" "${seed}" "${job_idx}"
      if [ "${DRY_RUN}" = "1" ]; then
        job_idx=$(( job_idx + 1 ))
        continue
      fi
      active=$(( active + 1 ))
      job_idx=$(( job_idx + 1 ))
      if [ "${active}" -ge "${MAX_PARALLEL}" ]; then
        wait_batch "${split_name}"
        active=0
      fi
    done
  done
  wait_batch "${split_name}"
}

run_video_eval() {
  local job_idx=0
  local gpu
  local port
  local run_dir
  local ckpt
  require_path "${HELDOUT_MANIFEST}"
  require_path "${EVAL_PAIRS}"
  for seed in "${video_seed_list[@]}"; do
    for variant in "${video_variant_list[@]}"; do
      gpu="${eval_gpu_list[$(( job_idx % ${#eval_gpu_list[@]} ))]}"
      port=$(( MASTER_PORT_BASE + 1000 + job_idx ))
      run_dir="${ROOT}/${variant}/seed_${seed}"
      ckpt="$(stage2_ckpt "${variant}" "${seed}")"
      require_path "${ckpt}"
      mkdir -p "${run_dir}/metrics" "${run_dir}/videos"
      echo "Running video eval ${variant} seed ${seed} on GPU ${gpu}"
      run_or_print env CUDA_VISIBLE_DEVICES="${gpu}" "${TORCHRUN}" \
        --nproc_per_node=1 \
        --master_port="${port}" \
        distillation_flowmap/rollout_eval_video_stage2.py \
        --config distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow \
        --teacher-model-path "${TEACHER_MODEL_PATH}" \
        --dataset-path "${DATASET_PATH}" \
        --empty-emb-path "${EMPTY_EMB_PATH}" \
        --output-dir "${run_dir}/videos" \
        --resume-from-path "${ckpt}" \
        --result-json "${run_dir}/metrics/video_mse.json" \
        --video-dir "${run_dir}/videos" \
        --eval-manifest "${HELDOUT_MANIFEST}" \
        --eval-pairs-json "${EVAL_PAIRS}" \
        --split-name heldout \
        --num-batches 0 \
        --student-steps 4 \
        --teacher-steps "${TEACHER_STEPS}" \
        --video-max-pairs "${VIDEO_MAX_PAIRS}" \
        --video-decode-device cpu \
        --disable-eval-gradient-checkpointing \
        --disable-eval-force-cfg \
        --eval-rollout-grad-mode endpoint \
        --eval-empty-cache
      job_idx=$(( job_idx + 1 ))
    done
  done
}

if [ "${RUN_HELDOUT_EVAL}" = "1" ]; then
  build_teacher_cache "${TEACHER_CACHE}" "${HELDOUT_MANIFEST}" heldout
  run_rollout_grid heldout "${HELDOUT_MANIFEST}" offline_rollout "${TEACHER_CACHE}"
fi

if [ "${RUN_TRAIN_EVAL}" = "1" ]; then
  make_balanced_train_eval_manifest
  build_teacher_cache "${TRAIN_TEACHER_CACHE}" "${TRAIN_EVAL_MANIFEST}" train_eval
  run_rollout_grid train_eval "${TRAIN_EVAL_MANIFEST}" train_rollout "${TRAIN_TEACHER_CACHE}"
fi

if [ "${RUN_VIDEO_EVAL}" = "1" ]; then
  run_video_eval
fi

if [ "${failures}" -ne 0 ]; then
  echo "Final eval finished with ${failures} failed rollout job(s)." >&2
  exit 1
fi

if [ "${RUN_SUMMARY}" = "1" ]; then
  run_or_print "${PYTHON}" distillation_flowmap/ablation/summarize_robotwin_ablation.py \
    --root "${ROOT}" \
    --out "${SUMMARY_DIR}" \
    --baseline-variant "${BASELINE_VARIANT}"
fi

echo "Final RobotWin ablation eval finished. Summary: ${SUMMARY_DIR}"
