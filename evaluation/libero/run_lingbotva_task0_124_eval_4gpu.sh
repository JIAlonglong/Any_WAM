#!/usr/bin/env bash
# Evaluate the Stage-1 baseline plus four LIBERO APM arms on the trained task.
# Each job uses matched video/action sampling steps in {1,2,4}; four jobs run per wave.

set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_SCRIPT="${PROJECT_ROOT}/evaluation/libero/run_eval_new.sh"
MERGE_SCRIPT="${PROJECT_ROOT}/evaluation/libero/merge_lingbotva_task0_ablation.py"
SERVER_PYTHON="${SERVER_PYTHON:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"
CLIENT_PYTHON="${CLIENT_PYTHON:-/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TRAIN_STEPS=500
EPISODES=20
MASTER_PORT_BASE=30670
WS_PORT_BASE=31670
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero}"
STAGE1_CHECKPOINT=""
ABLATION_ROOT=""
OUTPUT_ROOT=""

usage() {
  cat <<'EOF'
Usage: run_lingbotva_task0_124_eval_4gpu.sh \
  --stage1-checkpoint CHECKPOINT_ROOT \
  --ablation-root ABLATION_OUTPUT_ROOT \
  --output-root EVAL_ROOT [options]

Options:
  --train-steps N
  --episodes N
  --gpu-ids 0,1,2,3
  --master-port-base PORT
  --ws-port-base PORT
  --base-model-path PATH

Set CHECK_ONLY=1 to validate and print all 15 jobs without starting services.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage1-checkpoint) STAGE1_CHECKPOINT="${2:?missing value}"; shift 2 ;;
    --ablation-root) ABLATION_ROOT="${2:?missing value}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?missing value}"; shift 2 ;;
    --train-steps) TRAIN_STEPS="${2:?missing value}"; shift 2 ;;
    --episodes) EPISODES="${2:?missing value}"; shift 2 ;;
    --gpu-ids) GPU_IDS="${2:?missing value}"; shift 2 ;;
    --master-port-base) MASTER_PORT_BASE="${2:?missing value}"; shift 2 ;;
    --ws-port-base) WS_PORT_BASE="${2:?missing value}"; shift 2 ;;
    --base-model-path) BASE_MODEL_PATH="${2:?missing value}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${STAGE1_CHECKPOINT}" && -n "${ABLATION_ROOT}" && -n "${OUTPUT_ROOT}" ]] \
  || { usage >&2; exit 2; }
for numeric in "${TRAIN_STEPS}" "${EPISODES}" "${MASTER_PORT_BASE}" "${WS_PORT_BASE}"; do
  [[ "${numeric}" =~ ^[1-9][0-9]*$ ]] || die "steps, episodes, and ports must be positive integers"
done

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
[[ "${#GPUS[@]}" -eq 4 ]] || die "gpu selection must contain exactly four unique IDs"
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "GPU IDs must be non-negative integers"
  [[ -z "${SEEN_GPUS[${gpu}]:-}" ]] || die "gpu selection must contain exactly four unique IDs"
  SEEN_GPUS["${gpu}"]=1
done

MODELS=(stage1 stage1_only anchor_only field_only apm)
BUDGETS=(1 2 4)
declare -A CHECKPOINTS=()
CHECKPOINTS[stage1]="${STAGE1_CHECKPOINT}/target_student/transformer"
for arm in stage1_only anchor_only field_only apm; do
  CHECKPOINTS["${arm}"]="${ABLATION_ROOT}/${arm}/seed_42/stage2/checkpoints/step_${TRAIN_STEPS}/target_student/transformer"
done

if [[ "${CHECK_ONLY:-0}" != "1" ]]; then
  for component in transformer vae tokenizer text_encoder; do
    [[ -d "${BASE_MODEL_PATH}/${component}" ]] \
      || die "base model component not found: ${BASE_MODEL_PATH}/${component}"
  done
  for model in "${MODELS[@]}"; do
    [[ -d "${CHECKPOINTS[${model}]}" ]] \
      || die "checkpoint not found for ${model}: ${CHECKPOINTS[${model}]}"
  done
fi

job_index=0
wave_pids=()
wave_labels=()

cleanup_active_jobs() {
  local pid
  for pid in "${wave_pids[@]:-}"; do
    [[ -n "${pid}" ]] || continue
    pkill -TERM -P "${pid}" 2>/dev/null || true
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup_active_jobs EXIT INT TERM

wait_wave() {
  local index rc failed=0
  set +e
  for index in "${!wave_pids[@]}"; do
    wait "${wave_pids[${index}]}"
    rc="$?"
    if [[ "${rc}" -ne 0 ]]; then
      echo "Evaluation job failed (${wave_labels[${index}]}); see its log." >&2
      failed=1
    fi
  done
  set -e
  wave_pids=()
  wave_labels=()
  [[ "${failed}" -eq 0 ]] || return 1
}

launch_job() {
  local model="$1"
  local steps="$2"
  local slot="$3"
  local gpu="${GPUS[${slot}]}"
  local master_port="$(( MASTER_PORT_BASE + job_index ))"
  local ws_port="$(( WS_PORT_BASE + job_index ))"
  local save_root="${OUTPUT_ROOT}/${model}/steps_${steps}"
  local log_path="${save_root}.log"
  local result_json="${save_root}/results/libero_eval/libero_10_0.json"
  echo "JOB model=${model} steps=${steps} suite=libero_10 task=0:1 gpu=${gpu} video_steps=${steps} action_steps=${steps} master_port=${master_port} ws_port=${ws_port} base_model=${BASE_MODEL_PATH}"
  if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
    job_index=$(( job_index + 1 ))
    return
  fi
  if [[ -f "${result_json}" ]] && "${SERVER_PYTHON}" -c \
    'import json, sys; d=json.load(open(sys.argv[1])); n=int(float(d["total_num"])); s=float(d["succ_num"]); raise SystemExit(0 if n == int(sys.argv[2]) and 0 <= s <= n else 1)' \
    "${result_json}" "${EPISODES}"; then
    echo "SKIP model=${model} steps=${steps} validated=${result_json}"
    job_index=$(( job_index + 1 ))
    return
  fi
  mkdir -p "$(dirname "${save_root}")"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
    SERVER_PYTHON="${SERVER_PYTHON}" \
    CLIENT_PYTHON="${CLIENT_PYTHON}" \
    STUDENT_CKPT="${CHECKPOINTS[${model}]}" \
    TEACHER_CKPT="${CHECKPOINTS[${model}]}" \
    WAN22_PRETRAINED_PATH="${BASE_MODEL_PATH}" \
    LIBERO_BENCHMARK=libero_10 \
    EVAL_MODE=success \
    NUM_STEPS="${steps}" \
    ACTION_NUM_STEPS="${steps}" \
    TEST_NUM="${EPISODES}" \
    TASK_START=0 \
    TASK_END=1 \
    PORT="${ws_port}" \
    MASTER_PORT="${master_port}" \
    SAVE_ROOT="${save_root}" \
      bash "${EVAL_SCRIPT}" external "${model}"
  ) > "${log_path}" 2>&1 &
  wave_pids+=("$!")
  wave_labels+=("${model}/steps_${steps}")
  job_index=$(( job_index + 1 ))
}

for model in "${MODELS[@]}"; do
  for steps in "${BUDGETS[@]}"; do
    slot="${#wave_pids[@]}"
    launch_job "${model}" "${steps}" "${slot}"
    if [[ "${CHECK_ONLY:-0}" != "1" && "${#wave_pids[@]}" -eq 4 ]]; then
      wait_wave
    fi
  done
done
if [[ "${CHECK_ONLY:-0}" != "1" && "${#wave_pids[@]}" -gt 0 ]]; then
  wait_wave
fi

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  echo "CHECK_ONLY=1: verified 15 jobs (5 models x matched 1/2/4 budgets) on libero_10 task 0."
else
  "${SERVER_PYTHON}" "${MERGE_SCRIPT}" \
    --input-root "${OUTPUT_ROOT}" \
    --expected-episodes "${EPISODES}" \
    --output "${OUTPUT_ROOT}/summary.json"
  echo "Completed LIBERO task-0 ablation family: ${OUTPUT_ROOT}/summary.json"
fi
