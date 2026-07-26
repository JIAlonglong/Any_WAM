#!/usr/bin/env bash
# Official Cosmos teacher-only matched-budget K=1/2/4 evaluation on 8 GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'USAGE'
Usage:
  bash evaluation/libero/run_cosmos_official_teacher_124_eval_8gpu.sh run|dry-run

Evaluates only the native official Cosmos teacher on all 40 LIBERO tasks at
matched video/action solver budgets K=1, K=2, and K=4. The default is 50
episodes per task. Set MATRIX_ROOT to a fresh output path and provide
COSMOS_POLICY_TEACHER_LOCK plus an all-40-task S4_PROMPT_TABLE.
USAGE
}

die() {
    printf 'ERROR=%s\n' "$*" >&2
    exit 2
}

[[ $# -eq 1 ]] || {
    usage >&2
    die "expected exactly one mode: run or dry-run"
}
MODE="$1"
case "${MODE}" in
    run|dry-run) ;;
    *)
        usage >&2
        die "mode must be run or dry-run"
        ;;
esac

# Callers may override any runtime path before invoking this wrapper. The
# shared environment provides the audited cu128 worker and official model
# defaults without requiring a separate `source` command.
source "${SCRIPT_DIR}/cosmos_progressive_s4_env.sh"

: "${MATRIX_ROOT:?set a new teacher evaluation root}"
: "${COSMOS_POLICY_PATH:?set the official Cosmos teacher root}"
: "${COSMOS_POLICY_TEACHER_LOCK:?set the verified teacher provenance lock}"
: "${S4_DATASET_PATH:?set the LIBERO dataset root}"
: "${S4_EMPTY_EMBEDDING:?set the empty embedding}"
: "${S4_PROMPT_TABLE:?set the all-40-task prompt table}"

[[ ! -e "${MATRIX_ROOT}" ]] || \
    die "MATRIX_ROOT already exists; no-overwrite policy requires a new root: ${MATRIX_ROOT}"

# These values are the experiment definition, not user-tunable defaults.
export S4_MATRIX_ROLES=official_teacher
export S4_ALIGNMENT_VERIFIED=1
export S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH=0
COSMOS_TEACHER_FORMAL_NUM_SHARDS="${COSMOS_TEACHER_FORMAL_NUM_SHARDS:-4}"
case "${COSMOS_TEACHER_FORMAL_NUM_SHARDS}" in
    2|4) ;;
    *) die "COSMOS_TEACHER_FORMAL_NUM_SHARDS must be 2 or 4" ;;
esac
export S4_FORMAL_NUM_SHARDS="${COSMOS_TEACHER_FORMAL_NUM_SHARDS}"
export S4_EPISODES_PER_TASK="${S4_EPISODES_PER_TASK:-50}"

# The delegated matrix parser requires S4_CKPT_ROOT syntactically. In the
# official_teacher role it uses COSMOS_POLICY_PATH as the evaluated checkpoint,
# so bind the unused compatibility value to that same root rather than a
# student checkpoint.
export S4_CKPT_ROOT="${COSMOS_POLICY_PATH}"

MATRIX_LAUNCHER="${COSMOS_TEACHER_MATRIX_LAUNCHER:-${SCRIPT_DIR}/run_cosmos_progressive_joint_124_eval_8gpu.sh}"
[[ -x "${MATRIX_LAUNCHER}" ]] || die "matrix launcher is not executable: ${MATRIX_LAUNCHER}"

exec "${MATRIX_LAUNCHER}" "${MODE}"
