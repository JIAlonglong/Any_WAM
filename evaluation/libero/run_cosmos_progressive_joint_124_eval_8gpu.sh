#!/usr/bin/env bash
# Sequential full LIBERO evaluation for joint video/action K=1, K=2, and K=4.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

usage() {
    cat <<'USAGE'
Usage: bash evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh run|dry-run
Required: MATRIX_ROOT, S4_CKPT_ROOT, S4_DATASET_PATH, S4_EMPTY_EMBEDDING.
Each K evaluates all 10 LIBERO tasks with 50 shared seeds (500 episodes).
USAGE
}

die() { printf 'ERROR=%s\n' "$*" >&2; exit 2; }
require_env() {
    local name="$1"
    [[ -n "${!name:-}" ]] || die "set ${name}"
}
emit_kv() { printf '%s=%s\n' "$1" "$2"; }

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

[[ ! -e "${MATRIX_ROOT}" ]] || \
    die "MATRIX_ROOT already exists; no-overwrite policy requires a new root: ${MATRIX_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
S4_FORMAL_LAUNCHER="${S4_FORMAL_LAUNCHER:-${SCRIPT_DIR}/run_cosmos_progressive_s4_eval.sh}"
[[ -x "${S4_FORMAL_LAUNCHER}" ]] || die "formal launcher is not executable: ${S4_FORMAL_LAUNCHER}"

readonly STEPS=(1 2 4)
export S4_FORMAL_NUM_SHARDS=4
export S4_VIDEO_SEEDS="${S4_VIDEO_SEEDS:-0,1}"

if [[ "${MODE}" == "dry-run" ]]; then
    export S4_DRY_RUN=1
else
    export S4_DRY_RUN=0
fi

emit_kv "MATRIX_MODE" "${MODE}"
emit_kv "MATRIX_ROOT" "${MATRIX_ROOT}"

caller_prompt_table="${S4_PROMPT_TABLE:-}"
for k in "${STEPS[@]}"; do
    export S4_STUDENT_STEPS="${k}"
    export EVAL_ROOT="${MATRIX_ROOT}/k${k}"
    emit_kv "MATRIX_STEP" "${k}"
    "${S4_FORMAL_LAUNCHER}" formal

    if [[ "${k}" == "1" && -z "${caller_prompt_table}" ]]; then
        export S4_PROMPT_TABLE="${MATRIX_ROOT}/k1/prompt_embeddings.pt"
        if [[ "${MODE}" == "run" && ! -f "${S4_PROMPT_TABLE}" ]]; then
            die "K=1 did not materialize the expected prompt table: ${S4_PROMPT_TABLE}"
        fi
    fi
done

if [[ "${MODE}" == "dry-run" ]]; then
    emit_kv "MATRIX_SUMMARY_PLAN" "${MATRIX_ROOT}/matrix_summary.json"
    exit 0
fi

"${PYTHON_BIN}" - "${MATRIX_ROOT}" "${S4_CKPT_ROOT}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_checkpoint = str(Path(sys.argv[2]).resolve())
steps = (1, 2, 4)
summary_paths = {
    str(step): root / f"k{step}" / "formal_summary.json" for step in steps
}

for step in steps:
    path = summary_paths[str(step)]
    if not path.is_file():
        raise SystemExit(f"missing child summary for K={step}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("num_records", -1)) != 500:
        raise SystemExit(
            f"num_records mismatch for K={step}: "
            f"expected=500 got={payload.get('num_records')!r}"
        )
    if int(payload.get("student_steps", -1)) != step:
        raise SystemExit(
            f"student_steps mismatch for K={step}: "
            f"got={payload.get('student_steps')!r}"
        )
    reported_checkpoint = payload.get("checkpoint")
    if not isinstance(reported_checkpoint, str):
        raise SystemExit(
            f"checkpoint mismatch for K={step}: got={reported_checkpoint!r}"
        )
    if str(Path(reported_checkpoint).resolve()) != expected_checkpoint:
        raise SystemExit(
            f"checkpoint mismatch for K={step}: expected={expected_checkpoint!r} "
            f"got={reported_checkpoint!r}"
        )

matrix_summary = {
    "schema": "cosmos_progressive_joint_124_matrix_v1",
    "checkpoint": expected_checkpoint,
    "steps": list(steps),
    "summaries": {key: str(path) for key, path in summary_paths.items()},
}
output_path = root / "matrix_summary.json"
temporary_path = root / ".matrix_summary.json.tmp"
temporary_path.write_text(
    json.dumps(matrix_summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary_path, output_path)
print(f"MATRIX_SUMMARY={output_path}")
PY
