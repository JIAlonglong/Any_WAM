#!/usr/bin/env bash
# Resume the official Cosmos teacher K=1/2/4 full40 matrix on four GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

die() {
    printf 'ERROR=%s\n' "$*" >&2
    exit 2
}

[[ $# -eq 1 ]] || \
    die "usage: bash evaluation/libero/run_cosmos_teacher_124_eval_4gpu_resume.sh MATRIX_ROOT"
MATRIX_ROOT="$1"
[[ -n "${MATRIX_ROOT}" ]] || die "MATRIX_ROOT must not be empty"
export MATRIX_ROOT

# Pin the known audited runtime while preserving explicit caller overrides.
export COSMOS_PREDICT2_REPO="${COSMOS_PREDICT2_REPO:-/kpfs-intern/jialongliu/projects/cosmos-predict2.5-k2-nfe-fix}"
export COSMOS_POLICY_TEACHER_LOCK="${COSMOS_POLICY_TEACHER_LOCK:-/kpfs-intern/jialongliu/results/cosmos_teacher_eval_assets/teacher.lock.json}"
export PYTHON_BIN="${PYTHON_BIN:-/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

# shellcheck source=evaluation/libero/cosmos_progressive_s4_env.sh
source "${SCRIPT_DIR}/cosmos_progressive_s4_env.sh"

# The evaluator preflight starts COSMOS_POLICY_PYTHON with inherited
# PYTHONPATH. Make the CUDA/OSS extras visible there as well as to the raw
# worker, matching the audited long-form launch command.
export PYTHONPATH="${PROJECT_ROOT}:${COSMOS_POLICY_EXTRA_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${COSMOS_EVAL_PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export S4_CKPT_ROOT="${COSMOS_POLICY_PATH}"
export S4_PROMPT_TABLE="${S4_PROMPT_TABLE:-${S4_EMPTY_EMBEDDING}}"
export S4_MODEL_ROLE=official_teacher
export S4_ALIGNMENT_VERIFIED=1
export S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH=0
export S4_EVAL_CLASSIFICATION=formal_verified
export S4_EVAL_IS_FORMAL=1
export S4_FORMAL_NUM_SHARDS=4
export S4_FORMAL_GPU_LAYOUT=colocated
export S4_VIDEO_SEEDS="${S4_VIDEO_SEEDS:-0}"
export S4_EPISODES_PER_TASK="${S4_EPISODES_PER_TASK:-50}"

[[ "${S4_EPISODES_PER_TASK}" =~ ^[1-9][0-9]*$ ]] || \
    die "S4_EPISODES_PER_TASK must be a positive integer"
[[ -x "${PYTHON_BIN}" ]] || die "PYTHON_BIN is not executable: ${PYTHON_BIN}"

FORMAL_LAUNCHER="${COSMOS_TEACHER_FORMAL_LAUNCHER:-${SCRIPT_DIR}/run_cosmos_progressive_s4_eval.sh}"
[[ -x "${FORMAL_LAUNCHER}" ]] || die "formal launcher is not executable: ${FORMAL_LAUNCHER}"

SELECTED_STEPS="${COSMOS_TEACHER_STEPS:-}"
case "${SELECTED_STEPS}" in
    "")
        STEPS=(1 2 4)
        DEFER_MATRIX_MERGE=0
        ;;
    1|2|4)
        STEPS=("${SELECTED_STEPS}")
        DEFER_MATRIX_MERGE=1
        ;;
    *)
        die "COSMOS_TEACHER_STEPS must be one of 1, 2, or 4 when set"
        ;;
esac
readonly STEPS
readonly SUITES=(libero_10 libero_spatial libero_object libero_goal)
DRY_RUN="${S4_DRY_RUN:-0}"
case "${DRY_RUN}" in
    0|1) ;;
    *) die "S4_DRY_RUN must be 0 or 1" ;;
esac

validate_completed_cell() {
    local summary="$1" suite="$2" steps="$3"
    "${PYTHON_BIN}" - \
        "${summary}" "${suite}" "${steps}" "${S4_EPISODES_PER_TASK}" \
        "${COSMOS_POLICY_PATH}" <<'PY'
import json
import re
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected_suite = sys.argv[2]
expected_steps = int(sys.argv[3])
episodes_per_task = int(sys.argv[4])
expected_checkpoint = str(Path(sys.argv[5]).resolve())
payload = json.loads(summary_path.read_text(encoding="utf-8"))

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)

require(payload.get("model_role") == "official_teacher", "model role mismatch")
require(payload.get("libero_benchmark") == expected_suite, "suite mismatch")
require(
    payload.get("video_steps") == expected_steps
    and payload.get("action_steps") == expected_steps,
    "video/action step mismatch",
)
require(payload.get("num_tasks") == 10, "task count mismatch")
require(
    payload.get("num_records") == 10 * episodes_per_task,
    "record count mismatch",
)
require(
    payload.get("seeds_per_task") == episodes_per_task,
    "episodes-per-task mismatch",
)
require(payload.get("is_formal") is True, "formal status mismatch")
require(
    payload.get("evaluation_classification") == "formal_verified",
    "formal classification mismatch",
)
require(
    str(Path(payload.get("checkpoint", "")).resolve()) == expected_checkpoint,
    "checkpoint mismatch",
)
identity = payload.get("checkpoint_contract_identity")
require(
    isinstance(identity, str) and re.fullmatch(r"[0-9a-f]{64}", identity) is not None,
    "checkpoint contract identity missing",
)
PY
}

for steps in "${STEPS[@]}"; do
    for suite in "${SUITES[@]}"; do
        export S4_STUDENT_STEPS="${steps}"
        export S4_LIBERO_BENCHMARK="${suite}"
        export EVAL_ROOT="${MATRIX_ROOT}/k${steps}/${suite}"
        summary="${EVAL_ROOT}/formal_summary.json"

        if [[ -f "${summary}" ]]; then
            validate_completed_cell "${summary}" "${suite}" "${steps}"
            printf 'SKIP_COMPLETED=%s,K=%s,records=%s\n' \
                "${suite}" "${steps}" "$((10#$S4_EPISODES_PER_TASK * 10))"
            continue
        fi
        [[ ! -e "${EVAL_ROOT}" ]] || \
            die "incomplete existing cell: ${EVAL_ROOT}"

        if [[ "${DRY_RUN}" == "1" ]]; then
            "${FORMAL_LAUNCHER}" dry-run
        else
            "${FORMAL_LAUNCHER}" formal
        fi
    done
done

if [[ "${DEFER_MATRIX_MERGE}" == "1" ]]; then
    printf 'MATRIX_SUMMARY_DEFERRED=%s/matrix_summary.json\n' "${MATRIX_ROOT}"
    printf 'MATRIX_SUMMARY_CSV_DEFERRED=%s/matrix_summary.csv\n' "${MATRIX_ROOT}"
    exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'MATRIX_SUMMARY_PLAN=%s/matrix_summary.json\n' "${MATRIX_ROOT}"
    printf 'MATRIX_SUMMARY_CSV_PLAN=%s/matrix_summary.csv\n' "${MATRIX_ROOT}"
    exit 0
fi

"${PYTHON_BIN}" - \
    "${MATRIX_ROOT}" "${COSMOS_POLICY_PATH}" "${S4_EPISODES_PER_TASK}" <<'PY'
import sys
from pathlib import Path

from evaluation.libero.cosmos_progressive_eval_summary import merge_student_matrix

root = Path(sys.argv[1])
merge_student_matrix(
    root=root,
    checkpoint=str(Path(sys.argv[2]).resolve()),
    evaluation_classification="formal_verified",
    is_formal=True,
    model_role="official_teacher",
    episodes_per_task=int(sys.argv[3]),
)
print(f"MATRIX_SUMMARY={root / 'matrix_summary.json'}")
print(f"MATRIX_SUMMARY_CSV={root / 'matrix_summary.csv'}")
PY
