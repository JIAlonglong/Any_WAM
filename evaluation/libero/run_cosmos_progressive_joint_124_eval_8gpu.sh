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
emit_kv "ALIGNMENT_BLOCKED" "${ALIGNMENT_BLOCKED}"
emit_kv "EVALUATION_CLASSIFICATION" "${S4_EVAL_CLASSIFICATION}"
emit_kv "EVALUATION_IS_FORMAL" "${S4_EVAL_IS_FORMAL}"
if [[ "${MODE}" == "run" && "${ALIGNMENT_BLOCKED}" == "1" ]]; then
    die "known training/action alignment mismatch: set S4_ALIGNMENT_VERIFIED=1 only after verification, or S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH=1 for a non-formal diagnostic run"
fi

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

"${PYTHON_BIN}" - "${MATRIX_ROOT}" "${S4_CKPT_ROOT}" \
    "${S4_EVAL_CLASSIFICATION}" "${S4_EVAL_IS_FORMAL}" "${S4_MODEL_ROLE:-stage2_target}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_checkpoint = str(Path(sys.argv[2]).resolve())
expected_classification = sys.argv[3]
expected_is_formal = sys.argv[4] == "1"
expected_model_role = sys.argv[5]
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
    if payload.get("model_role") != expected_model_role:
        raise SystemExit(f"model_role mismatch for K={step}: {payload.get('model_role')!r}")
    if int(payload.get("video_steps", -1)) != step or int(payload.get("action_steps", -1)) != step:
        raise SystemExit(f"video/action step mismatch for K={step}")
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
    if payload.get("evaluation_classification") != expected_classification:
        raise SystemExit(
            f"evaluation classification mismatch for K={step}: "
            f"expected={expected_classification!r} "
            f"got={payload.get('evaluation_classification')!r}"
        )
    if payload.get("is_formal") is not expected_is_formal:
        raise SystemExit(
            f"is_formal mismatch for K={step}: expected={expected_is_formal!r} "
            f"got={payload.get('is_formal')!r}"
        )

identities = {json.loads(path.read_text(encoding="utf-8")).get("checkpoint_contract_identity") for path in summary_paths.values()}
if len(identities) != 1 or None in identities:
    raise SystemExit(f"checkpoint_contract_identity mismatch across K: {sorted(identities)!r}")

matrix_summary = {
    "schema": "cosmos_progressive_joint_124_matrix_v1",
    "checkpoint": expected_checkpoint,
    "evaluation_classification": expected_classification,
    "is_formal": expected_is_formal,
    "steps": list(steps),
    "model_role": expected_model_role,
    "checkpoint_contract_identity": next(iter(identities)),
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
