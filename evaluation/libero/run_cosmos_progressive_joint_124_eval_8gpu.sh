#!/usr/bin/env bash
# Sequential full LIBERO evaluation for joint video/action K=1, K=2, and K=4.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

usage() {
    cat <<'USAGE'
Usage: bash evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh run|dry-run
Required: MATRIX_ROOT, S4_CKPT_ROOT, COSMOS_POLICY_PATH, S4_DATASET_PATH, S4_EMPTY_EMBEDDING,
and an S4_PROMPT_TABLE covering all 40 tasks.
S4_EPISODES_PER_TASK defaults to 50, so each K evaluates 2,000 episodes
across the four ten-task LIBERO suites.
USAGE
}

die() { printf 'ERROR=%s\n' "$*" >&2; exit 2; }
require_env() {
    local name="$1"
    [[ -n "${!name:-}" ]] || die "set ${name}"
}
emit_kv() { printf '%s=%s\n' "$1" "$2"; }
positive() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer"
}

[[ $# -eq 1 ]] || { usage >&2; exit 2; }
MODE="$1"
case "${MODE}" in
    run|dry-run) ;;
    *) usage >&2; die "mode must be run or dry-run" ;;
esac

require_env "MATRIX_ROOT"
require_env "S4_CKPT_ROOT"
require_env "S4_DATASET_PATH"
require_env "S4_EMPTY_EMBEDDING"
require_env "S4_PROMPT_TABLE"

[[ ! -e "${MATRIX_ROOT}" ]] || \
    die "MATRIX_ROOT already exists; no-overwrite policy requires a new root: ${MATRIX_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
S4_FORMAL_LAUNCHER="${S4_FORMAL_LAUNCHER:-${SCRIPT_DIR}/run_cosmos_progressive_s4_eval.sh}"
[[ -x "${S4_FORMAL_LAUNCHER}" ]] || die "formal launcher is not executable: ${S4_FORMAL_LAUNCHER}"
S4_EPISODES_PER_TASK="${S4_EPISODES_PER_TASK:-50}"
positive S4_EPISODES_PER_TASK "$S4_EPISODES_PER_TASK"
export S4_EPISODES_PER_TASK

readonly STEPS=(1 2 4)
readonly SUITES=(libero_10 libero_spatial libero_object libero_goal)
COSMOS_POLICY_PATH="${COSMOS_POLICY_PATH:-}"
COSMOS_POLICY_TEACHER_LOCK="${COSMOS_POLICY_TEACHER_LOCK:-}"
S4_MATRIX_ROLES="${S4_MATRIX_ROLES:-stage2_target,official_teacher}"
case "${S4_MATRIX_ROLES}" in
    stage2_target)
        ROLES=(stage2_target)
        DUAL_ROLE_MATRIX=0
        ;;
    official_teacher)
        require_env "COSMOS_POLICY_PATH"
        require_env "COSMOS_POLICY_TEACHER_LOCK"
        ROLES=(official_teacher)
        DUAL_ROLE_MATRIX=0
        ;;
    stage2_target,official_teacher)
        require_env "COSMOS_POLICY_PATH"
        require_env "COSMOS_POLICY_TEACHER_LOCK"
        ROLES=(stage2_target official_teacher)
        DUAL_ROLE_MATRIX=1
        ;;
    *) die "S4_MATRIX_ROLES must be stage2_target, official_teacher, or stage2_target,official_teacher" ;;
esac
S4_ALIGNMENT_VERIFIED="${S4_ALIGNMENT_VERIFIED:-0}"
S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH="${S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH:-0}"
export S4_ALIGNMENT_VERIFIED S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH
case "${S4_ALIGNMENT_VERIFIED}" in 0|1) ;; *) die "S4_ALIGNMENT_VERIFIED must be 0 or 1" ;; esac
case "${S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH}" in 0|1) ;; *) die "S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH must be 0 or 1" ;; esac
ALIGNMENT_BLOCKED=0
if [[ "${S4_ALIGNMENT_VERIFIED}" == "1" ]]; then
    export S4_EVAL_CLASSIFICATION="formal_verified"
    export S4_EVAL_IS_FORMAL=1
elif [[ "${S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH}" == "1" ]]; then
    export S4_EVAL_CLASSIFICATION="diagnostic_known_alignment_mismatch"
    export S4_EVAL_IS_FORMAL=0
else
    export S4_EVAL_CLASSIFICATION="blocked_known_alignment_mismatch"
    export S4_EVAL_IS_FORMAL=0
    ALIGNMENT_BLOCKED=1
fi
export S4_FORMAL_NUM_SHARDS=4
export S4_VIDEO_SEEDS="${S4_VIDEO_SEEDS:-0,1}"

if [[ "${MODE}" == "dry-run" ]]; then
    export S4_DRY_RUN=1
else
    export S4_DRY_RUN=0
fi

emit_kv "MATRIX_MODE" "${MODE}"
emit_kv "MATRIX_ROOT" "${MATRIX_ROOT}"
emit_kv "EPISODES_PER_TASK" "${S4_EPISODES_PER_TASK}"
emit_kv "ALIGNMENT_BLOCKED" "${ALIGNMENT_BLOCKED}"
emit_kv "EVALUATION_CLASSIFICATION" "${S4_EVAL_CLASSIFICATION}"
emit_kv "EVALUATION_IS_FORMAL" "${S4_EVAL_IS_FORMAL}"
if [[ "${MODE}" == "run" && "${ALIGNMENT_BLOCKED}" == "1" ]]; then
    die "known training/action alignment mismatch: set S4_ALIGNMENT_VERIFIED=1 only after verification, or S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH=1 for a non-formal diagnostic run"
fi

for role in "${ROLES[@]}"; do
    export S4_MODEL_ROLE="${role}"
    emit_kv "MATRIX_ROLE" "${role}"
    for k in "${STEPS[@]}"; do
        export S4_STUDENT_STEPS="${k}"
        emit_kv "MATRIX_STEP" "${k}"
        for suite in "${SUITES[@]}"; do
            export S4_LIBERO_BENCHMARK="${suite}"
            if (( DUAL_ROLE_MATRIX )); then
                export EVAL_ROOT="${MATRIX_ROOT}/${role}/k${k}/${suite}"
            else
                export EVAL_ROOT="${MATRIX_ROOT}/k${k}/${suite}"
            fi
            emit_kv "MATRIX_SUITE" "${suite}"
            "${S4_FORMAL_LAUNCHER}" formal
        done
    done
done

if [[ "${MODE}" == "dry-run" ]]; then
    emit_kv "MATRIX_SUMMARY_PLAN" "${MATRIX_ROOT}/matrix_summary.json"
    emit_kv "MATRIX_SUMMARY_CSV_PLAN" "${MATRIX_ROOT}/matrix_summary.csv"
    exit 0
fi

"${PYTHON_BIN}" - "${MATRIX_ROOT}" "${S4_CKPT_ROOT}" "${COSMOS_POLICY_PATH}" \
    "${S4_EVAL_CLASSIFICATION}" "${S4_EVAL_IS_FORMAL}" \
    "${S4_EPISODES_PER_TASK}" "${S4_MATRIX_ROLES}" <<'PY'
import sys
from pathlib import Path

from evaluation.libero.cosmos_progressive_eval_summary import (
    merge_complete_matrix,
    merge_student_matrix,
)

root = Path(sys.argv[1])
checkpoints = {
    "stage2_target": str(Path(sys.argv[2]).resolve()),
    "official_teacher": str(Path(sys.argv[3]).resolve()),
}
expected_classification = sys.argv[4]
expected_is_formal = sys.argv[5] == "1"
episodes_per_task = int(sys.argv[6])
roles = sys.argv[7]
if roles == "stage2_target,official_teacher":
    merge_complete_matrix(
        root=root,
        checkpoints=checkpoints,
        evaluation_classification=expected_classification,
        is_formal=expected_is_formal,
        episodes_per_task=episodes_per_task,
    )
else:
    role = roles
    merge_student_matrix(
        root=root,
        checkpoint=checkpoints[role],
        evaluation_classification=expected_classification,
        is_formal=expected_is_formal,
        model_role=role,
        episodes_per_task=episodes_per_task,
    )
print(f"MATRIX_SUMMARY={root / 'matrix_summary.json'}")
print(f"MATRIX_SUMMARY_CSV={root / 'matrix_summary.csv'}")
PY
