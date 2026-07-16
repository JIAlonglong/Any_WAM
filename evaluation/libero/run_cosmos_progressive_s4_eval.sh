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
    local seed_count="$1"
    emit_local_command "${PYTHON_BIN}" - "${EVAL_ROOT}" "${S4_CKPT_ROOT}" "${seed_count}" "<merge-formal-records>"
    (( DRY_RUN )) && return 0
    "${PYTHON_BIN}" - "${EVAL_ROOT}" "${S4_CKPT_ROOT}" "${seed_count}" <<'PY'
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
expected_checkpoint = str(Path(sys.argv[2]).resolve())
seed_count = int(sys.argv[3])
expected_tasks = tuple(range(10))
expected = {(task, seed) for task in expected_tasks for seed in range(seed_count)}
seen = {}
per_task = defaultdict(list)

for record_path in sorted(root.glob("shard_*/seed_*/records/task_*_episode_*.json")):
    relative_parts = record_path.relative_to(root).parts
    shard_match = next((re.fullmatch(r"shard_(\d+)", part) for part in relative_parts if re.fullmatch(r"shard_(\d+)", part)), None)
    seed_match = next((re.fullmatch(r"seed_(\d+)", part) for part in relative_parts if re.fullmatch(r"seed_(\d+)", part)), None)
    if shard_match is None or seed_match is None:
        raise SystemExit(f"seed mismatch: record path lacks shard/seed identity: {record_path}")
    shard, seed = int(shard_match.group(1)), int(seed_match.group(1))
    if shard not in (0, 1) or seed not in range(seed_count):
        raise SystemExit(f"seed mismatch: unexpected shard={shard} seed={seed} in {record_path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    task = int(record.get("task_idx", -1))
    allowed = range(0, 5) if shard == 0 else range(5, 10)
    if task not in allowed:
        raise SystemExit(f"seed mismatch: task {task} is not assigned to shard {shard}")
    if int(record.get("episode_idx", -1)) != 0:
        raise SystemExit(f"seed mismatch: expected episode_idx=0 for shared seed {seed}")
    if "seed" not in record:
        raise SystemExit(f"seed mismatch: missing record seed for task={task} path seed={seed}")
    try:
        record_seed = int(record["seed"])
    except (TypeError, ValueError):
        raise SystemExit(f"seed mismatch: invalid record seed={record['seed']!r} path seed={seed}")
    if record_seed != seed:
        raise SystemExit(f"seed mismatch: record seed={record_seed} path seed={seed}")
    if record.get("s4_checkpoint") != expected_checkpoint:
        raise SystemExit(f"checkpoint mismatch: task={task} seed={seed} expected={expected_checkpoint!r} got={record.get('s4_checkpoint')!r}")
    key = (task, seed)
    if key in seen:
        raise SystemExit(f"duplicate record: task={task} seed={seed}: {seen[key]} and {record_path}")
    seen[key] = str(record_path)
    per_task[task].append(1.0 if bool(record.get("success", False)) else 0.0)

unexpected = set(seen) - expected
if unexpected:
    raise SystemExit(f"seed mismatch: unexpected task/seed records: {sorted(unexpected)}")
missing = expected - set(seen)
if missing:
    raise SystemExit(f"missing record: {len(missing)} required task/seed records absent, first={sorted(missing)[:5]}")
if len(seen) != len(expected):
    raise SystemExit(f"duplicate record: expected {len(expected)} records, found {len(seen)}")

task_means = {str(task): sum(per_task[task]) / seed_count for task in expected_tasks}
macro_success = sum(task_means[str(task)] for task in expected_tasks) / len(expected_tasks)
rng = random.Random(0)
bootstrap = []
for _ in range(10000):
    sampled_means = [sum(rng.choice(per_task[task]) for _ in range(seed_count)) / seed_count for task in expected_tasks]
    bootstrap.append(sum(sampled_means) / len(sampled_means))
bootstrap.sort()
summary = {
    "schema": "cosmos_progressive_s4_formal_eval_v1",
    "checkpoint": expected_checkpoint,
    "num_records": len(seen),
    "seeds_per_task": seed_count,
    "per_task_success": task_means,
    "macro_success": macro_success,
    "bootstrap_ci_95": [bootstrap[int(0.025 * len(bootstrap))], bootstrap[int(0.975 * len(bootstrap))]],
}
path = root / "formal_summary.json"
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"MERGE_SUMMARY={path}")
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
        )
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

    # Formal uses the only four-GPU split: two non-overlapping task shards.
    preflight_shard 0 0 1
    preflight_shard 1 2 3
    if (( DRY_RUN )); then
        launch_shard 0 0 1 0 5 "${seed_count}"
        launch_shard 1 2 3 5 10 "${seed_count}"
    else
        launch_shard 0 0 1 0 5 "${seed_count}" &
        local pid0=$!
        launch_shard 1 2 3 5 10 "${seed_count}" &
        local pid1=$!
        wait "${pid0}"
        wait "${pid1}"
    fi
    verify_formal_results "${seed_count}"
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
DRY_RUN=0
[[ "${MODE}" == "dry-run" || "${S4_DRY_RUN:-0}" == "1" ]] && DRY_RUN=1

emit_kv "MODE" "${MODE}"
emit_kv "DRY_RUN" "${DRY_RUN}"
emit_kv "S4_CKPT_ROOT" "${S4_CKPT_ROOT}"
emit_kv "EVAL_ROOT" "${EVAL_ROOT}"

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
        run_live_evaluation formal 50
        ;;
    dry-run)
        require_env "S4_PROMPT_TABLE"
        require_env "S4_EMPTY_EMBEDDING"
        emit_kv "PLAN_MODE" "formal"
        run_live_evaluation formal 50
        ;;
esac
