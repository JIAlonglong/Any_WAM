#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ROOT="${ROOT:-${PROJECT_ROOT}/distillation_flowmap/output_robotwin_stepwam_ablation/protocol_opd_mechanism_calibration_v1}"
STAGE2_STEPS="${STAGE2_STEPS:-750}"
VARIANTS="${VARIANTS:-calib_w_o_opd,calib_endpoint_only,calib_velocity_only,calib_full_last_step,calib_full_suffix_grad}"
SEEDS="${SEEDS:-0}"
TASK_PRESET="${TASK_PRESET:-core2}"
PROTOCOL_SEED="${PROTOCOL_SEED:-0}"
TEACHER_STEPS="${TEACHER_STEPS:-8}"
TEACHER_CACHE_SOURCE_VARIANT="${TEACHER_CACHE_SOURCE_VARIANT:-calib_w_o_opd}"
TEACHER_CACHE_SOURCE_SEED="${TEACHER_CACHE_SOURCE_SEED:-0}"
BASELINE_VARIANT="${BASELINE_VARIANT:-calib_w_o_opd}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3,4}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
VIDEO_VARIANTS="${VIDEO_VARIANTS:-${VARIANTS}}"
VIDEO_SEEDS="${VIDEO_SEEDS:-0}"
VIDEO_MAX_PAIRS="${VIDEO_MAX_PAIRS:-1}"
ALLOW_LARGE_CALIBRATION_EVAL="${ALLOW_LARGE_CALIBRATION_EVAL:-0}"

if [[ "${TASK_PRESET}" != "core2" || "${STAGE2_STEPS}" -gt 1000 ]] \
  && [[ "${ALLOW_LARGE_CALIBRATION_EVAL}" != "1" ]]; then
  echo "Refusing expanded calibration eval; set ALLOW_LARGE_CALIBRATION_EVAL=1 explicitly." >&2
  exit 2
fi

export ROOT STAGE2_STEPS VARIANTS SEEDS TASK_PRESET PROTOCOL_SEED
export TEACHER_STEPS TEACHER_CACHE_SOURCE_VARIANT TEACHER_CACHE_SOURCE_SEED
export BASELINE_VARIANT EVAL_GPUS MAX_PARALLEL VIDEO_VARIANTS VIDEO_SEEDS
export VIDEO_MAX_PAIRS
export TEACHER_CACHE="${TEACHER_CACHE:-${ROOT}/protocol/teacher_cache_heldout_calib_manifest_first_t${TEACHER_STEPS}.pt}"
export TRAIN_TEACHER_CACHE="${TRAIN_TEACHER_CACHE:-${ROOT}/protocol/teacher_cache_train_calib_manifest_first_t${TEACHER_STEPS}.pt}"
export SUMMARY_DIR="${SUMMARY_DIR:-${ROOT}/summary_calibration}"

exec bash "${SCRIPT_DIR}/run_final_eval.sh" "$@"
