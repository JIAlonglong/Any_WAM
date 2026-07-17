#!/usr/bin/env bash
# Run the fixed Cosmos Progressive S4 paper-offline and formal closed-loop suite.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

usage() {
    cat <<'USAGE'
Usage: bash evaluation/libero/run_cosmos_progressive_s4_suite.sh run|dry-run

Required environment:
  SUITE_ROOT, S4_CKPT_ROOT, S4_DATASET_PATH, S4_EMPTY_EMBEDDING,
  COSMOS_POLICY_PATH, COSMOS_POLICY_PYTHON, COSMOS_PREDICT2_REPO

The live run requires a new, non-existing SUITE_ROOT. It serially creates a
fixed protocol, Cosmos teacher cache, paper-offline metric result, and formal
closed-loop result. Dry-run only prints the plan; it creates no files and
executes no child process.
USAGE
}

die() { printf 'ERROR=%s\n' "$*" >&2; exit 2; }
require_env() {
    local name="$1"
    [[ -n "${!name:-}" ]] || die "set ${name}"
}
emit_kv() { printf '%s=%s\n' "$1" "$2"; }

emit_command() {
    printf 'COMMAND='
    printf '%q ' "$@"
    printf '\n'
}

run_local_command() {
    emit_command "$@"
    (( DRY_RUN )) && return 0
    "$@"
}

run_cosmos_pair_command() {
    local student_gpu="$1" worker_gpu="$2"
    shift 2
    emit_command \
        "CUDA_VISIBLE_DEVICES=${student_gpu}" \
        "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES=${worker_gpu}" \
        "$@"
    (( DRY_RUN )) && return 0
    CUDA_VISIBLE_DEVICES="${student_gpu}" \
    COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES="${worker_gpu}" \
        "$@"
}

write_phase_status() {
    local event="$1" phase="$2"
    (( DRY_RUN )) && return 0
    printf '{"event":"%s","phase":"%s"}\n' "${event}" "${phase}" >> "${STATUS_PATH}"
}

run_phase() {
    local phase="$1"
    shift
    emit_kv "PHASE_PLAN" "${phase}"
    write_phase_status started "${phase}"
    "$@"
    write_phase_status completed "${phase}"
}

ensure_suite_root_absent() {
    (( DRY_RUN )) && return 0
    if [[ -e "${SUITE_ROOT}" || -L "${SUITE_ROOT}" ]]; then
        die "SUITE_ROOT already exists; no-overwrite policy requires a new root: ${SUITE_ROOT}"
    fi
}

create_new_suite_root() {
    (( DRY_RUN )) && return 0
    mkdir -p "$(dirname "${SUITE_ROOT}")"
    if ! mkdir "${SUITE_ROOT}"; then
        die "SUITE_ROOT could not be created exclusively; choose a new root: ${SUITE_ROOT}"
    fi
    STATUS_PATH="${SUITE_ROOT}/suite_status.jsonl"
}

require_directory() {
    local name="$1" path="$2"
    [[ -d "${path}" ]] || die "${name} is not a directory: ${path}"
}

require_file() {
    local name="$1" path="$2"
    [[ -f "${path}" ]] || die "${name} is not a file: ${path}"
}

validate_live_inputs() {
    (( DRY_RUN )) && return 0
    require_directory S4_CKPT_ROOT "${S4_CKPT_ROOT}"
    require_directory S4_DATASET_PATH "${S4_DATASET_PATH}"
    require_file S4_EMPTY_EMBEDDING "${S4_EMPTY_EMBEDDING}"
    require_directory COSMOS_POLICY_PATH "${COSMOS_POLICY_PATH}"
    [[ -x "${COSMOS_POLICY_PYTHON}" ]] || die "COSMOS_POLICY_PYTHON is not executable: ${COSMOS_POLICY_PYTHON}"
    require_directory COSMOS_PREDICT2_REPO "${COSMOS_PREDICT2_REPO}"
    require_file FORMAL_LAUNCHER "${FORMAL_LAUNCHER}"
    if [[ -n "${S4_PROMPT_TABLE}" ]]; then
        require_file S4_PROMPT_TABLE "${S4_PROMPT_TABLE}"
    fi
    if [[ -n "${S4_INITIAL_STATES_JSON}" ]]; then
        require_file S4_INITIAL_STATES_JSON "${S4_INITIAL_STATES_JSON}"
    fi
    if [[ "${PYTHON_BIN}" == */* ]]; then
        [[ -x "${PYTHON_BIN}" ]] || die "PYTHON_BIN is not executable: ${PYTHON_BIN}"
    else
        command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "PYTHON_BIN is not on PATH: ${PYTHON_BIN}"
    fi
}

run_runtime_preflight() {
    local prompt_table="${S4_PROMPT_TABLE:-${FORMAL_ROOT}/prompt_embeddings.pt}"
    emit_kv "PREFLIGHT_STUDENT_GPU" 0
    emit_kv "PREFLIGHT_COSMOS_WORKER_GPU" 1
    local -a command=(
        env
        "COSMOS_POLICY_PYTHON=${COSMOS_POLICY_PYTHON}"
        "COSMOS_PREDICT2_REPO=${COSMOS_PREDICT2_REPO}"
        "${PYTHON_BIN}" -m evaluation.libero.rollout_cosmos_progressive_s4
        --checkpoint-transformer "${S4_CKPT_ROOT}"
        --config "${S4_CONFIG}"
        --prompt-table "${prompt_table}"
        --empty-embedding "${S4_EMPTY_EMBEDDING}"
        --teacher-model-path "${COSMOS_POLICY_PATH}"
        --device cuda:0
        --preflight
    )
    run_cosmos_pair_command 0 1 "${command[@]}"
}

run_protocol() {
    local -a command=(
        "${PYTHON_BIN}" -m distillation_flowmap.prepare_cosmos_progressive_protocol
        --config "${S4_CONFIG}"
        --dataset-path "${S4_DATASET_PATH}"
        --output-dir "${PROTOCOL_ROOT}"
        --selection-per-task 3
        --test-per-task 5
        --protocol-seed 20260714
        --pairs 1000,0
    )
    run_local_command "${command[@]}"
}

run_teacher_cache() {
    emit_kv "OFFLINE_STUDENT_GPU" 0
    emit_kv "OFFLINE_COSMOS_WORKER_GPU" 1
    local -a command=(
        env
        "COSMOS_POLICY_PYTHON=${COSMOS_POLICY_PYTHON}"
        "COSMOS_PREDICT2_REPO=${COSMOS_PREDICT2_REPO}"
        "${PYTHON_BIN}" -m distillation_flowmap.build_cosmos_progressive_teacher_cache
        --config "${S4_CONFIG}"
        --dataset-path "${S4_DATASET_PATH}"
        --manifest "${PROTOCOL_ROOT}/test_manifest.json"
        --pairs "${PROTOCOL_ROOT}/eval_pairs.json"
        --cache-dir "${PROTOCOL_ROOT}/teacher_cache/test"
        --teacher-steps 8
        --teacher-model-path "${COSMOS_POLICY_PATH}"
    )
    run_cosmos_pair_command 0 1 "${command[@]}"
}

run_offline_paper() {
    emit_kv "OFFLINE_STUDENT_GPU" 0
    emit_kv "OFFLINE_COSMOS_WORKER_GPU" 1
    local -a command=(
        env
        "COSMOS_POLICY_PYTHON=${COSMOS_POLICY_PYTHON}"
        "COSMOS_PREDICT2_REPO=${COSMOS_PREDICT2_REPO}"
        "${PYTHON_BIN}" -m distillation_flowmap.eval_cosmos_progressive_s4_paper
        --checkpoint-transformer "${S4_CKPT_ROOT}"
        --config "${S4_CONFIG}"
        --dataset-path "${S4_DATASET_PATH}"
        --manifest "${PROTOCOL_ROOT}/test_manifest.json"
        --pairs "${PROTOCOL_ROOT}/eval_pairs.json"
        --cache-dir "${PROTOCOL_ROOT}/teacher_cache/test"
        --output-dir "${OFFLINE_ROOT}"
        --teacher-model-path "${COSMOS_POLICY_PATH}"
        --device cuda:0
        --student-steps 4
        --teacher-steps 8
    )
    run_cosmos_pair_command 0 1 "${command[@]}"
}

run_closed_loop_formal() {
    local -a command=(
        env
        "EVAL_ROOT=${FORMAL_ROOT}"
        "S4_CKPT_ROOT=${S4_CKPT_ROOT}"
        "S4_DATASET_PATH=${S4_DATASET_PATH}"
        "S4_EMPTY_EMBEDDING=${S4_EMPTY_EMBEDDING}"
        "S4_CONFIG=${S4_CONFIG}"
        "COSMOS_POLICY_PATH=${COSMOS_POLICY_PATH}"
        "COSMOS_POLICY_PYTHON=${COSMOS_POLICY_PYTHON}"
        "COSMOS_PREDICT2_REPO=${COSMOS_PREDICT2_REPO}"
        "PYTHON_BIN=${PYTHON_BIN}"
        "S4_LIBERO_BENCHMARK=${S4_LIBERO_BENCHMARK}"
    )
    if [[ -n "${S4_PROMPT_TABLE}" ]]; then
        command+=("S4_PROMPT_TABLE=${S4_PROMPT_TABLE}")
    fi
    if [[ -n "${S4_INITIAL_STATES_JSON}" ]]; then
        command+=("S4_INITIAL_STATES_JSON=${S4_INITIAL_STATES_JSON}")
    fi
    if (( DRY_RUN )); then
        command+=("S4_DRY_RUN=1")
    else
        # Do not inherit a caller's S4_DRY_RUN=1 into a live suite result.
        command+=("S4_DRY_RUN=0")
    fi
    command+=(bash "${FORMAL_LAUNCHER}" formal)
    emit_command "${command[@]}"

    if (( DRY_RUN )); then
        emit_kv "FORMAL_SHARD_0_STUDENT_GPU" 0
        emit_kv "FORMAL_SHARD_0_COSMOS_WORKER_GPU" 1
        emit_kv "FORMAL_SHARD_0_TASK_RANGE" "0,5"
        emit_kv "FORMAL_SHARD_1_STUDENT_GPU" 2
        emit_kv "FORMAL_SHARD_1_COSMOS_WORKER_GPU" 3
        emit_kv "FORMAL_SHARD_1_TASK_RANGE" "5,10"
        emit_kv "FORMAL_REQUESTED_SEEDS_PER_TASK" 50
        emit_kv "FORMAL_REQUESTED_RECORDS" 500
        if [[ -n "${S4_PROMPT_TABLE}" ]]; then
            emit_kv "FORMAL_PROMPT_TABLE" "${S4_PROMPT_TABLE}"
        else
            emit_kv "FORMAL_PROMPT_TABLE_PLAN" "${FORMAL_ROOT}/prompt_embeddings.pt"
        fi
        return 0
    fi
    "${command[@]}"
}

[[ $# -eq 1 ]] || { usage >&2; exit 2; }
MODE="$1"
case "${MODE}" in
    run) DRY_RUN=0 ;;
    dry-run) DRY_RUN=1 ;;
    *) usage >&2; die "mode must be run or dry-run" ;;
esac

SUITE_ROOT="${SUITE_ROOT:-}"
S4_CKPT_ROOT="${S4_CKPT_ROOT:-}"
S4_DATASET_PATH="${S4_DATASET_PATH:-}"
S4_EMPTY_EMBEDDING="${S4_EMPTY_EMBEDDING:-}"
COSMOS_POLICY_PATH="${COSMOS_POLICY_PATH:-}"
COSMOS_POLICY_PYTHON="${COSMOS_POLICY_PYTHON:-}"
COSMOS_PREDICT2_REPO="${COSMOS_PREDICT2_REPO:-}"
for name in \
    SUITE_ROOT S4_CKPT_ROOT S4_DATASET_PATH S4_EMPTY_EMBEDDING \
    COSMOS_POLICY_PATH COSMOS_POLICY_PYTHON COSMOS_PREDICT2_REPO; do
    require_env "${name}"
done

PYTHON_BIN="${PYTHON_BIN:-python}"
S4_CONFIG="${S4_CONFIG:-distillation_flowmap.config_libero_cosmos_policy_stage2_progressive}"
S4_PROMPT_TABLE="${S4_PROMPT_TABLE:-}"
S4_INITIAL_STATES_JSON="${S4_INITIAL_STATES_JSON:-}"
S4_LIBERO_BENCHMARK="${S4_LIBERO_BENCHMARK:-libero_10}"
PROTOCOL_ROOT="${SUITE_ROOT}/protocol"
OFFLINE_ROOT="${SUITE_ROOT}/offline_paper"
FORMAL_ROOT="${SUITE_ROOT}/closed_loop_formal"
FORMAL_LAUNCHER="${PROJECT_ROOT}/evaluation/libero/run_cosmos_progressive_s4_eval.sh"
STATUS_PATH=""

emit_kv "MODE" "${MODE}"
emit_kv "DRY_RUN" "${DRY_RUN}"
emit_kv "SUITE_ROOT" "${SUITE_ROOT}"
emit_kv "S4_CKPT_ROOT" "${S4_CKPT_ROOT}"
emit_kv "PROTOCOL_ROOT" "${PROTOCOL_ROOT}"
emit_kv "OFFLINE_ROOT" "${OFFLINE_ROOT}"
emit_kv "FORMAL_ROOT" "${FORMAL_ROOT}"

ensure_suite_root_absent
validate_live_inputs
emit_kv "PHASE_PLAN" runtime_preflight
run_runtime_preflight
create_new_suite_root
run_phase protocol run_protocol
run_phase teacher_cache run_teacher_cache
run_phase offline_paper run_offline_paper
run_phase closed_loop_formal run_closed_loop_formal
emit_kv "SUITE_COMPLETE" 1
