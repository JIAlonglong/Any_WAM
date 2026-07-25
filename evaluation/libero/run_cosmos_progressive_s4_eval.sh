#!/usr/bin/env bash
# Reproducible Cosmos Progressive S4 evaluation launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

usage() {
    cat <<'USAGE'
Usage: bash evaluation/libero/run_cosmos_progressive_s4_eval.sh smoke|gate|formal|dry-run
Required: S4_CKPT_ROOT (public S4 transformer), EVAL_ROOT (new result root).
Smoke additionally requires S4_SMOKE_DATASET_PATH, S4_SMOKE_MANIFEST,
S4_SMOKE_PAIRS, and S4_SMOKE_CACHE_DIR.  The cache must be prebuilt by the
official Cosmos environment; this launcher never synthesizes teacher caches.
Gate/formal require S4_EMPTY_EMBEDDING and either S4_PROMPT_TABLE or S4_DATASET_PATH for live-only materialization from existing latent embeddings.
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

emit_command() {
    local student_gpu="$1" worker_gpu="$2"
    shift 2
    printf 'COMMAND='
    printf '%q ' "CUDA_VISIBLE_DEVICES=${student_gpu}" "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES=${worker_gpu}"
    printf '%q ' "$@"
    printf '\n'
}
emit_smoke_command() {
    local student_gpu="$1"
    shift
    printf 'COMMAND='
    printf '%q ' "CUDA_VISIBLE_DEVICES=${student_gpu}"
    printf '%q ' "$@"
    printf '\n'
}
emit_local_command() {
    printf 'COMMAND='
    printf '%q ' "$@"
    printf '\n'
}
run_live_command() {
    local student_gpu="$1" worker_gpu="$2"
    shift 2
    emit_command "${student_gpu}" "${worker_gpu}" "$@"
    (( DRY_RUN )) && return 0
    CUDA_VISIBLE_DEVICES="${student_gpu}" COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES="${worker_gpu}" "$@"
}
run_smoke_command() {
    local student_gpu="$1"
    shift
    emit_smoke_command "${student_gpu}" "$@"
    (( DRY_RUN )) && return 0
    CUDA_VISIBLE_DEVICES="${student_gpu}" "$@"
}
ensure_new_output_root() {
    (( DRY_RUN )) && return 0
    [[ ! -e "${EVAL_ROOT}" ]] || die "EVAL_ROOT already exists; no-overwrite policy requires a new root: ${EVAL_ROOT}"
    mkdir -p "${EVAL_ROOT}"
}

verify_smoke_summary() {
    local summary_path="$1"
    emit_local_command "${PYTHON_BIN}" - "${summary_path}" "<verify-cache-only-summary>"
    (( DRY_RUN )) && return 0
    "${PYTHON_BIN}" - "${summary_path}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("is_paper_metric") is not False:
    raise SystemExit("cache-only smoke must be explicitly non-paper")
if int(summary.get("student_steps", -1)) != 4:
    raise SystemExit("cache-only smoke did not report K=4")
for key in ("video_x0", "video_noise", "teacher_x_r", "teacher_v_r", "student_video", "student_field"):
    shape = summary.get("validated_video_shapes", {}).get(key)
    if not isinstance(shape, list) or len(shape) != 5 or int(shape[1]) != 16:
        raise SystemExit(f"cache-only smoke missing 16-channel shape for {key}: {shape!r}")
print("SMOKE_SUMMARY_VALID=1")
PY
}

verify_formal_results() {
    local seed_count="$1" shard_plan="$2"
    emit_local_command \
        "${PYTHON_BIN}" - "${EVAL_ROOT}" "${S4_CKPT_ROOT}" "${seed_count}" \
        "${S4_STUDENT_STEPS}" "${shard_plan}" "${S4_EVAL_CLASSIFICATION}" \
        "${S4_EVAL_IS_FORMAL}" "<merge-formal-records>"
    (( DRY_RUN )) && return 0
    "${PYTHON_BIN}" - "${EVAL_ROOT}" "${S4_CKPT_ROOT}" "${seed_count}" \
        "${S4_STUDENT_STEPS}" "${shard_plan}" "${S4_EVAL_CLASSIFICATION}" \
        "${S4_EVAL_IS_FORMAL}" <<'PY'
import os
import sys
from pathlib import Path

from evaluation.libero.cosmos_progressive_eval_summary import merge_formal_records

root = Path(sys.argv[1])
expected_checkpoint = str(Path(sys.argv[2]).resolve())
seed_count = int(sys.argv[3])
requested_steps = int(sys.argv[4])
evaluation_classification = sys.argv[6]
is_formal = sys.argv[7] == "1"
expected_model_role = os.environ.get("S4_MODEL_ROLE", "stage2_target")
suite = os.environ.get("S4_LIBERO_BENCHMARK", "libero_10")
merge_formal_records(
    root=root,
    checkpoint=expected_checkpoint,
    suite=suite,
    seed_count=seed_count,
    requested_steps=requested_steps,
    shard_plan=sys.argv[5],
    evaluation_classification=evaluation_classification,
    is_formal=is_formal,
    model_role=expected_model_role,
)
print(f"MERGE_SUMMARY={root / 'formal_summary.json'}")
print(f"MERGE_SUMMARY_CSV={root / 'formal_summary.csv'}")
PY
}


prepare_prompt_table() {
    if [[ -n "${S4_PROMPT_TABLE}" ]]; then
        emit_kv "PROMPT_TABLE" "${S4_PROMPT_TABLE}"
        return 0
    fi
    require_env "S4_DATASET_PATH"
    S4_PROMPT_TABLE="${EVAL_ROOT}/prompt_embeddings.pt"
    if (( DRY_RUN )); then
        emit_kv "PROMPT_TABLE_PLAN" "${S4_PROMPT_TABLE}"
        return 0
    fi
    emit_kv "PROMPT_TABLE_MATERIALIZED" "${S4_PROMPT_TABLE}"
    emit_local_command "${PYTHON_BIN}" - "${S4_DATASET_PATH}" "${S4_PROMPT_TABLE}" "<materialize-task-prompt-embeddings>"
    "${PYTHON_BIN}" - "${S4_DATASET_PATH}" "${S4_PROMPT_TABLE}" <<'PY'
import json
import sys
from pathlib import Path

import torch

dataset_root = Path(sys.argv[1])
output_path = Path(sys.argv[2])
tasks_path = dataset_root / "meta" / "tasks.jsonl"
if output_path.exists():
    raise SystemExit(f"refusing to overwrite prompt table: {output_path}")
if not tasks_path.is_file():
    raise SystemExit(f"missing task metadata: {tasks_path}")

expected = set()
for line in tasks_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    payload = json.loads(line)
    text = payload.get("task", payload.get("text"))
    if not isinstance(text, str) or not text.strip():
        raise SystemExit(f"invalid task entry in {tasks_path}: {payload!r}")
    expected.add(text)
if len(expected) != 10:
    raise SystemExit(f"expected exactly 10 LIBERO task prompts, found {len(expected)}")

embeddings = {}
for latent_path in sorted((dataset_root / "latents").glob("**/*.pth")):
    payload = torch.load(latent_path, map_location="cpu", weights_only=False)
    text = payload.get("text")
    embedding = payload.get("text_emb")
    if text not in expected or text in embeddings:
        continue
    if not torch.is_tensor(embedding) or tuple(embedding.shape) != (512, 4096):
        raise SystemExit(
            f"invalid text_emb for task {text!r} in {latent_path}: "
            f"{getattr(embedding, 'shape', None)!r}"
        )
    embeddings[text] = embedding.detach().cpu().unsqueeze(0)

missing = sorted(expected - set(embeddings))
if missing:
    raise SystemExit(f"missing prompt embeddings for tasks: {missing}")
torch.save({"embeddings": embeddings}, output_path)
print(f"PROMPT_TABLE_WRITTEN={output_path}")
PY
}

preflight_shard() {
    local shard="$1" student_gpu="$2" worker_gpu="$3"
    local -a command=(
        "${PYTHON_BIN}" -m evaluation.libero.rollout_cosmos_progressive_s4
        --checkpoint-transformer "${S4_CKPT_ROOT}"
        --config "${S4_CONFIG}"
        --prompt-table "${S4_PROMPT_TABLE}"
        --empty-embedding "${S4_EMPTY_EMBEDDING}"
        --teacher-model-path "${COSMOS_POLICY_PATH}"
        --device cuda:0
        --model-role "${S4_MODEL_ROLE}"
        --video-steps "${S4_STUDENT_STEPS}"
        --action-steps "${S4_STUDENT_STEPS}"
        --preflight
    )
    emit_kv "PREFLIGHT_SHARD" "${shard}"
    run_live_command "${student_gpu}" "${worker_gpu}" "${command[@]}"
}

launch_shard() {
    local shard="$1" student_gpu="$2" worker_gpu="$3" task_start="$4" task_end="$5" seed_count="$6"
    local seed
    emit_kv "SHARD_${shard}_STUDENT_GPU" "${student_gpu}"
    emit_kv "SHARD_${shard}_COSMOS_WORKER_GPU" "${worker_gpu}"
    emit_kv "SHARD_${shard}_TASK_RANGE" "${task_start},${task_end}"
    for ((seed = 0; seed < seed_count; seed += 1)); do
        local seed_root="${EVAL_ROOT}/shard_${shard}/seed_${seed}"
        local -a command=(
            "${PYTHON_BIN}" -m evaluation.libero.rollout_cosmos_progressive_s4
            --checkpoint-transformer "${S4_CKPT_ROOT}"
            --config "${S4_CONFIG}"
            --prompt-table "${S4_PROMPT_TABLE}"
            --empty-embedding "${S4_EMPTY_EMBEDDING}"
            --teacher-model-path "${COSMOS_POLICY_PATH}"
            --device cuda:0
            --output-dir "${seed_root}"
            --anchor-record-dir "${seed_root}/anchors"
            --libero-benchmark "${S4_LIBERO_BENCHMARK}"
            --task-range "${task_start}" "${task_end}"
            --episodes 1
            --env-seed "${seed}"
            --episode-index-offset "${seed}"
            --model-role "${S4_MODEL_ROLE}"
            --video-steps "${S4_STUDENT_STEPS}"
            --action-steps "${S4_STUDENT_STEPS}"
        )
        if [[ -n "${S4_VIDEO_SEED_SET[${seed}]:-}" ]]; then
            command+=(--save-video)
        fi
        if [[ -n "${S4_INITIAL_STATES_JSON}" ]]; then
            command+=(--initial-states-json "${S4_INITIAL_STATES_JSON}")
        fi
        run_live_command "${student_gpu}" "${worker_gpu}" "${command[@]}"
    done
}

run_live_evaluation() {
    local live_mode="$1" seed_count="$2"
    ensure_new_output_root
    prepare_prompt_table
    emit_kv "REQUESTED_SEEDS_PER_TASK" "${seed_count}"
    emit_kv "REQUESTED_RECORDS" "$((10 * seed_count))"

    if [[ "${live_mode}" == "gate" ]]; then
        # Gate is deliberately single-pair: student GPU 0 and worker GPU 1
        # cover all ten tasks for the shared five-seed acceptance check.
        preflight_shard 0 0 1
        launch_shard 0 0 1 0 10 "${seed_count}"
        return 0
    fi

    local -a shard_ids student_gpus worker_gpus task_starts task_ends
    if [[ "${S4_FORMAL_NUM_SHARDS}" == "2" ]]; then
        shard_ids=(0 1)
        student_gpus=(0 2)
        worker_gpus=(1 3)
        task_starts=(0 5)
        task_ends=(5 10)
    else
        shard_ids=(0 1 2 3)
        student_gpus=(0 2 4 6)
        worker_gpus=(1 3 5 7)
        task_starts=(0 3 6 8)
        task_ends=(3 6 8 10)
    fi
    local shard_plan="" index
    for index in "${!shard_ids[@]}"; do
        [[ -z "${shard_plan}" ]] || shard_plan+=";"
        shard_plan+="${shard_ids[index]}:${task_starts[index]}:${task_ends[index]}"
        preflight_shard \
            "${shard_ids[index]}" "${student_gpus[index]}" "${worker_gpus[index]}"
    done
    if (( DRY_RUN )); then
        for index in "${!shard_ids[@]}"; do
            launch_shard \
                "${shard_ids[index]}" "${student_gpus[index]}" "${worker_gpus[index]}" \
                "${task_starts[index]}" "${task_ends[index]}" "${seed_count}"
        done
    else
        local -a pids=()
        for index in "${!shard_ids[@]}"; do
            launch_shard \
                "${shard_ids[index]}" "${student_gpus[index]}" "${worker_gpus[index]}" \
                "${task_starts[index]}" "${task_ends[index]}" "${seed_count}" &
            pids+=("$!")
        done
        local status=0 pid
        for pid in "${pids[@]}"; do
            if ! wait "${pid}"; then
                status=1
            fi
        done
        (( status == 0 )) || return "${status}"
    fi
    verify_formal_results "${seed_count}" "${shard_plan}"
}

run_smoke() {
    require_env "S4_SMOKE_DATASET_PATH"
    require_env "S4_SMOKE_MANIFEST"
    require_env "S4_SMOKE_PAIRS"
    require_env "S4_SMOKE_CACHE_DIR"
    ensure_new_output_root
    local smoke_root="${EVAL_ROOT}/smoke"
    local smoke_gpu="${S4_SMOKE_STUDENT_GPU:-0}"
    emit_kv "SMOKE_STUDENT_GPU" "${smoke_gpu}"
    emit_kv "SMOKE_MANIFEST" "${S4_SMOKE_MANIFEST}"
    emit_kv "SMOKE_PAIRS" "${S4_SMOKE_PAIRS}"
    emit_kv "SMOKE_CACHE_DIR" "${S4_SMOKE_CACHE_DIR}"
    emit_kv "SMOKE_RESULT" "non-paper-cache-only"
    local -a command=(
        "${PYTHON_BIN}" -m distillation_flowmap.eval_cosmos_progressive_s4_paper
        --checkpoint-transformer "${S4_CKPT_ROOT}"
        --config "${S4_CONFIG}"
        --dataset-path "${S4_SMOKE_DATASET_PATH}"
        --manifest "${S4_SMOKE_MANIFEST}"
        --pairs "${S4_SMOKE_PAIRS}"
        --cache-dir "${S4_SMOKE_CACHE_DIR}"
        --output-dir "${smoke_root}"
        --device cuda:0
        --student-steps 4
        --teacher-steps 8
        --limit 1
        --cache-only-smoke
        --skip-same-state-velocity
    )
    run_smoke_command "${smoke_gpu}" "${command[@]}"
    verify_smoke_summary "${smoke_root}/summary.json"
}

[[ $# -eq 1 ]] || { usage >&2; exit 2; }
MODE="$1"
case "${MODE}" in smoke|gate|formal|dry-run) ;; *) usage >&2; die "mode must be smoke, gate, formal, or dry-run" ;; esac

: "${S4_CKPT_ROOT:?set the downloaded public S4 transformer root}"
: "${EVAL_ROOT:?set a new empty result root}"
PYTHON_BIN="${PYTHON_BIN:-python}"
S4_CONFIG="${S4_CONFIG:-distillation_flowmap.config_libero_cosmos_policy_stage2_progressive}"
COSMOS_POLICY_PATH="${COSMOS_POLICY_PATH:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
S4_PROMPT_TABLE="${S4_PROMPT_TABLE:-}"
S4_DATASET_PATH="${S4_DATASET_PATH:-}"
S4_EMPTY_EMBEDDING="${S4_EMPTY_EMBEDDING:-}"
S4_SMOKE_DATASET_PATH="${S4_SMOKE_DATASET_PATH:-}"
S4_SMOKE_MANIFEST="${S4_SMOKE_MANIFEST:-}"
S4_SMOKE_PAIRS="${S4_SMOKE_PAIRS:-}"
S4_SMOKE_CACHE_DIR="${S4_SMOKE_CACHE_DIR:-}"
S4_INITIAL_STATES_JSON="${S4_INITIAL_STATES_JSON:-}"
S4_LIBERO_BENCHMARK="${S4_LIBERO_BENCHMARK:-libero_10}"
case "${S4_LIBERO_BENCHMARK}" in
    libero_10|libero_spatial|libero_object|libero_goal) ;;
    *) die "S4_LIBERO_BENCHMARK must be libero_10, libero_spatial, libero_object, or libero_goal" ;;
esac
S4_STUDENT_STEPS="${S4_STUDENT_STEPS:-4}"
case "${S4_STUDENT_STEPS}" in
    1|2|4) ;;
    *) die "S4_STUDENT_STEPS must be 1, 2, or 4" ;;
esac
S4_EPISODES_PER_TASK="${S4_EPISODES_PER_TASK:-50}"
positive S4_EPISODES_PER_TASK "$S4_EPISODES_PER_TASK"
export S4_EPISODES_PER_TASK
S4_MODEL_ROLE="${S4_MODEL_ROLE:-stage2_target}"
case "${S4_MODEL_ROLE}" in
    stage2_online|stage2_target) ;;
    *) die "S4_MODEL_ROLE must be stage2_online or stage2_target" ;;
esac
export S4_MODEL_ROLE
S4_EVAL_CLASSIFICATION="${S4_EVAL_CLASSIFICATION:-unclassified}"
S4_EVAL_IS_FORMAL="${S4_EVAL_IS_FORMAL:-0}"
case "${S4_EVAL_IS_FORMAL}" in
    0|1) ;;
    *) die "S4_EVAL_IS_FORMAL must be 0 or 1" ;;
esac
if [[ "${S4_EVAL_CLASSIFICATION}" == "formal_verified" ]]; then
    [[ "${S4_EVAL_IS_FORMAL}" == "1" ]] || \
        die "formal_verified classification requires S4_EVAL_IS_FORMAL=1"
elif [[ "${S4_EVAL_IS_FORMAL}" == "1" ]]; then
    die "S4_EVAL_IS_FORMAL=1 requires formal_verified classification"
fi
S4_FORMAL_NUM_SHARDS="${S4_FORMAL_NUM_SHARDS:-2}"
case "${S4_FORMAL_NUM_SHARDS}" in
    2|4) ;;
    *) die "S4_FORMAL_NUM_SHARDS must be 2 or 4" ;;
esac
S4_VIDEO_SEEDS="${S4_VIDEO_SEEDS:-}"
declare -A S4_VIDEO_SEED_SET=()
if [[ -n "${S4_VIDEO_SEEDS}" ]]; then
    [[ "${S4_VIDEO_SEEDS}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || \
        die "S4_VIDEO_SEEDS must be comma-separated integers in 0..49"
    IFS=',' read -r -a requested_video_seeds <<< "${S4_VIDEO_SEEDS}"
    for video_seed in "${requested_video_seeds[@]}"; do
        if [[ ! "${video_seed}" =~ ^[0-9]+$ ]] || (( video_seed > 49 )); then
            die "S4_VIDEO_SEEDS must be comma-separated integers in 0..49"
        fi
        S4_VIDEO_SEED_SET["${video_seed}"]=1
    done
fi
DRY_RUN=0
[[ "${MODE}" == "dry-run" || "${S4_DRY_RUN:-0}" == "1" ]] && DRY_RUN=1
S4_ALIGNMENT_VERIFIED="${S4_ALIGNMENT_VERIFIED:-0}"
S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH="${S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH:-0}"
case "${S4_ALIGNMENT_VERIFIED}" in
    0|1) ;;
    *) die "S4_ALIGNMENT_VERIFIED must be 0 or 1" ;;
esac
case "${S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH}" in
    0|1) ;;
    *) die "S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH must be 0 or 1" ;;
esac
if (( ! DRY_RUN )); then
    case "${S4_EVAL_CLASSIFICATION}" in
        formal_verified)
            [[ "${S4_ALIGNMENT_VERIFIED}" == "1" ]] || \
                die "formal_verified requires S4_ALIGNMENT_VERIFIED=1"
            ;;
        diagnostic_known_alignment_mismatch)
            [[ "${S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH}" == "1" ]] || \
                die "diagnostic classification requires S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH=1"
            ;;
        blocked_known_alignment_mismatch)
            die "blocked classification cannot run live"
            ;;
        *)
            die "live evaluation classification is not authorized: ${S4_EVAL_CLASSIFICATION}"
            ;;
    esac
fi

emit_kv "MODE" "${MODE}"
emit_kv "DRY_RUN" "${DRY_RUN}"
emit_kv "S4_CKPT_ROOT" "${S4_CKPT_ROOT}"
emit_kv "EVAL_ROOT" "${EVAL_ROOT}"
emit_kv "S4_STUDENT_STEPS" "${S4_STUDENT_STEPS}"
emit_kv "S4_LIBERO_BENCHMARK" "${S4_LIBERO_BENCHMARK}"
emit_kv "S4_EPISODES_PER_TASK" "${S4_EPISODES_PER_TASK}"
emit_kv "S4_FORMAL_NUM_SHARDS" "${S4_FORMAL_NUM_SHARDS}"
emit_kv "S4_VIDEO_SEEDS" "${S4_VIDEO_SEEDS}"
emit_kv "EVALUATION_CLASSIFICATION" "${S4_EVAL_CLASSIFICATION}"
emit_kv "EVALUATION_IS_FORMAL" "${S4_EVAL_IS_FORMAL}"

case "${MODE}" in
    smoke)
        run_smoke
        ;;
    gate)
        require_env "S4_EMPTY_EMBEDDING"
        run_live_evaluation gate 5
        ;;
    formal)
        require_env "S4_EMPTY_EMBEDDING"
        run_live_evaluation formal "${S4_EPISODES_PER_TASK}"
        ;;
    dry-run)
        require_env "S4_PROMPT_TABLE"
        require_env "S4_EMPTY_EMBEDDING"
        emit_kv "PLAN_MODE" "formal"
        run_live_evaluation formal "${S4_EPISODES_PER_TASK}"
        ;;
esac
