#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ROOT="${ROOT:-${PROJECT_ROOT}/distillation_flowmap/output_robotwin_stepwam_ablation/protocol_final12_5000}"
STAGE1_STEPS="${STAGE1_STEPS:-5000}"
STAGE2_STEPS="${STAGE2_STEPS:-5000}"
VARIANTS="${VARIANTS:-full_stepwam,w_o_opd,endpoint_only_opd,velocity_only_opd,local_adjacent_only,action_only}"
SEEDS="${SEEDS:-0,1,2}"
TASK_PRESET="${TASK_PRESET:-representative}"
MAX_EPISODES_PER_TASK="${MAX_EPISODES_PER_TASK:-50}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-50}"
TRAIN_SAMPLES_PER_TASK="${TRAIN_SAMPLES_PER_TASK:-40}"
HELDOUT_SAMPLES_PER_TASK="${HELDOUT_SAMPLES_PER_TASK:-10}"
PROTOCOL_SEED="${PROTOCOL_SEED:-0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-33000}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
STAGE="${STAGE:-both}"

bash "${SCRIPT_DIR}/run_parallel_train.sh" \
  --root "${ROOT}" \
  --stage1-steps "${STAGE1_STEPS}" \
  --stage2-steps "${STAGE2_STEPS}" \
  --variants "${VARIANTS}" \
  --seeds "${SEEDS}" \
  --task-preset "${TASK_PRESET}" \
  --max-episodes-per-task "${MAX_EPISODES_PER_TASK}" \
  --max-samples-per-task "${MAX_SAMPLES_PER_TASK}" \
  --train-samples-per-task "${TRAIN_SAMPLES_PER_TASK}" \
  --heldout-samples-per-task "${HELDOUT_SAMPLES_PER_TASK}" \
  --protocol-seed "${PROTOCOL_SEED}" \
  --master-port-base "${MASTER_PORT_BASE}" \
  --max-parallel "${MAX_PARALLEL}" \
  --stage "${STAGE}" \
  "$@"
