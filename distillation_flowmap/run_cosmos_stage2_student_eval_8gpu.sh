#!/usr/bin/env bash
# Run an independently completed Cosmos Stage-1 through Stage-2 and student-only Full40 eval.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash distillation_flowmap/run_cosmos_stage2_student_eval_8gpu.sh \
  --stage1-root PATH --output-root PATH --run-tag TAG [options]

Options:
  --phase all|stage2|eval|check  default: all
  --stage1-step N                default: 3000
  --stage2-steps N               default: 5000
  --save-interval N              default: 1000
  --episodes N                   default: 50
  --master-port PORT             default: 29672
  --dry-run                      validate and print the plan; write nothing
  --check-only                   validate and print the plan; write nothing

The Wan base model is derived from validated Stage-1 metadata. The dataset and
complete Cosmos worker runtime default to the reviewed shared production paths.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
positive() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a strict positive decimal integer"; }
require_dir() { [[ -d "$2" && ! -L "$2" ]] || die "$1 must be a plain directory: $2"; }
require_file() { [[ -f "$2" && ! -L "$2" ]] || die "$1 is missing: $2"; }
canonical() {
    "$PYTHON_BIN" -c \
        'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=False))' \
        "$1"
}
absolute_lexical_from() {
    "$PYTHON_BIN" -c \
        'import os, sys; print(os.path.abspath(os.path.join(sys.argv[1], sys.argv[2])))' \
        "$1" "$2"
}
resolve_executable() {
    local label="$1" raw="$2" resolved
    [[ -e "$raw" && -x "$raw" ]] || die "$label is not executable: $raw"
    resolved="$(readlink -f -- "$raw")" || die "$label cannot be resolved: $raw"
    [[ -f "$resolved" && -x "$resolved" ]] || \
        die "$label does not resolve to an executable file: $raw"
    printf '%s\n' "$resolved"
}
resolve_path_list() {
    local label="$1" raw="$2" entry
    local -a entries resolved=()
    [[ -n "$raw" ]] || die "$label must not be empty"
    [[ "$raw" != :* && "$raw" != *: && "$raw" != *::* ]] || \
        die "$label contains an empty path component"
    IFS=':' read -r -a entries <<< "$raw"
    for entry in "${entries[@]}"; do
        require_dir "$label entry" "$entry"
        resolved+=("$(cd "$entry" && pwd -P)")
    done
    local IFS=':'
    printf '%s\n' "${resolved[*]}"
}
print_command() {
    local label="$1"
    shift
    printf '%s=' "$label"
    printf ' %q' "$@"
    printf '\n'
}

ACTIVE_PID=""
forward_signal() {
    local signal_name="$1" exit_code="$2"
    trap - INT TERM
    if [[ -n "$ACTIVE_PID" ]]; then
        /bin/kill -s "$signal_name" -- "-$ACTIVE_PID" 2>/dev/null || true
        wait "$ACTIVE_PID" 2>/dev/null || true
    fi
    exit "$exit_code"
}
trap 'forward_signal INT 130' INT
trap 'forward_signal TERM 143' TERM
run_child() {
    setsid --wait "$@" &
    ACTIVE_PID="$!"
    local status=0
    wait "$ACTIVE_PID" || status=$?
    ACTIVE_PID=""
    return "$status"
}

PHASE=all
STAGE1_ROOT=""
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
RUN_TAG=""
STAGE1_STEP=3000
STAGE2_STEPS=5000
SAVE_INTERVAL=1000
EPISODES=50
MASTER_PORT=29672
READ_ONLY_MODE=""

while (( $# > 0 )); do
    case "$1" in
        --phase) PHASE="${2:-}"; shift 2 ;;
        --stage1-root) STAGE1_ROOT="${2:-}"; shift 2 ;;
        --stage1-step) STAGE1_STEP="${2:-}"; shift 2 ;;
        --stage2-steps) STAGE2_STEPS="${2:-}"; shift 2 ;;
        --save-interval) SAVE_INTERVAL="${2:-}"; shift 2 ;;
        --episodes) EPISODES="${2:-}"; shift 2 ;;
        --master-port) MASTER_PORT="${2:-}"; shift 2 ;;
        --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
        --run-tag) RUN_TAG="${2:-}"; shift 2 ;;
        --dry-run|--check-only)
            [[ -z "$READ_ONLY_MODE" ]] || die "choose only one read-only mode"
            READ_ONLY_MODE="${1#--}"
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$PHASE" in all|stage2|eval|check) ;; *) die "--phase must be all, stage2, eval, or check" ;; esac
if [[ "$PHASE" == check ]]; then
    [[ -z "$READ_ONLY_MODE" ]] || die "--phase check is already read-only"
    READ_ONLY_MODE=check-only
    PHASE=all
fi
[[ -n "$STAGE1_ROOT" ]] || die "--stage1-root is required"
[[ -n "$OUTPUT_ROOT" ]] || die "--output-root is required"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
    die "--run-tag must be one path-safe component"
positive STAGE1_STEP "$STAGE1_STEP"
positive STAGE2_STEPS "$STAGE2_STEPS"
positive SAVE_INTERVAL "$SAVE_INTERVAL"
positive EPISODES "$EPISODES"
positive MASTER_PORT "$MASTER_PORT"
(( 10#$MASTER_PORT <= 65535 )) || die "--master-port must be at most 65535"
(( 10#$STAGE2_STEPS % 10#$SAVE_INTERVAL == 0 )) || \
    die "Stage-2 steps must be divisible by save interval"

command -v setsid >/dev/null 2>&1 || die "setsid is required for process-group signal forwarding"
PYTHON_BIN="${PYTHON_BIN:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"
[[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
export PYTHON_BIN
CALLER_CWD="$(pwd -P)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
HELPER_PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/wan_va${PYTHONPATH:+:$PYTHONPATH}"

RAW_STAGE1_ROOT="$(absolute_lexical_from "$CALLER_CWD" "$STAGE1_ROOT")"
RAW_OUTPUT_ROOT="$(absolute_lexical_from "$CALLER_CWD" "$OUTPUT_ROOT")"
RAW_STAGE1_CHECKPOINT="$RAW_STAGE1_ROOT/checkpoints/step_$STAGE1_STEP"
RAW_RUN_ROOT="$RAW_OUTPUT_ROOT/$RUN_TAG"
if ! (
    cd "$PROJECT_ROOT"
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$HELPER_PYTHONPATH" \
        "$PYTHON_BIN" - "$RAW_STAGE1_CHECKPOINT" \
            "$RAW_RUN_ROOT/universal-video-action" <<'PY'
import sys
from pathlib import Path

from distillation_flowmap.cosmos_stage2_lineage import (
    validate_stage2_path_isolation,
)

validate_stage2_path_isolation(
    stage1_root=Path(sys.argv[1]),
    output_dir=Path(sys.argv[2]),
    resume_checkpoint=None,
)
PY
); then
    die "RUN_ROOT/Stage-2 path isolation or symlink-component preflight failed"
fi

STAGE1_ROOT="$(canonical "$RAW_STAGE1_ROOT")"
OUTPUT_ROOT="$(canonical "$RAW_OUTPUT_ROOT")"
STAGE1_CHECKPOINT="$STAGE1_ROOT/checkpoints/step_$STAGE1_STEP"
RUN_ROOT="$OUTPUT_ROOT/$RUN_TAG"
STAGE2_OUTPUT="$RUN_ROOT/universal-video-action"
STAGE2_TARGET_TRANSFORMER="$STAGE2_OUTPUT/checkpoints/step_$STAGE2_STEPS/target_student/transformer"
LOCK_ROOT="$RUN_ROOT/provenance-locks"
MATRIX_ROOT="$RUN_ROOT/eval/student_only"
PROMPT_TABLE="$RUN_ROOT/libero_wan_prompt_embeddings_all40.pt"

stage1_lineage_output="$(
    cd "$PROJECT_ROOT"
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$HELPER_PYTHONPATH" \
        "$PYTHON_BIN" - "$STAGE1_CHECKPOINT" "$STAGE1_STEP" \
            "${WAN_STUDENT_BASE_MODEL_PATH:-}" "${COSMOS_POLICY_PATH:-}" <<'PY'
import json
import sys
from pathlib import Path

from distillation_flowmap.cosmos_hybrid_backend import (
    validate_cosmos_teacher_model_path,
    validate_wan_student_base_model_path,
)
from distillation_flowmap.cosmos_stage2_lineage import (
    validate_stage1_parent,
    validated_stage1_hybrid_model_paths,
)

checkpoint, expected_step, configured_wan, configured_teacher = sys.argv[1:]
parent = validate_stage1_parent(Path(checkpoint), expected_step=int(expected_step))
wan_base, teacher = validated_stage1_hybrid_model_paths(parent)
if configured_wan:
    actual_wan = str(validate_wan_student_base_model_path(configured_wan))
    if actual_wan != wan_base:
        raise ValueError(
            "WAN_STUDENT_BASE_MODEL_PATH does not match validated Stage-1 metadata"
        )
if configured_teacher:
    actual_teacher = str(validate_cosmos_teacher_model_path(configured_teacher))
    if actual_teacher != teacher:
        raise ValueError(
            "COSMOS_POLICY_PATH does not match validated Stage-1 metadata"
        )
lineage = json.dumps(
    {
        "parent_stage1_contract_identity": parent.contract_identity,
        "parent_stage1_path": parent.canonical_path,
    },
    sort_keys=True,
    separators=(",", ":"),
)
print("PARENT_STAGE1_PATH=" + parent.canonical_path)
print("PARENT_STAGE1_CONTRACT_IDENTITY=" + parent.contract_identity)
print("STAGE2_LINEAGE_JSON=" + lineage)
print("WAN_STUDENT_BASE_MODEL_PATH=" + wan_base)
print("COSMOS_POLICY_PATH=" + teacher)
PY
)" || die "real Stage-1 lineage and hybrid-root validation failed"
while IFS='=' read -r key value; do
    case "$key" in
        PARENT_STAGE1_PATH|PARENT_STAGE1_CONTRACT_IDENTITY|STAGE2_LINEAGE_JSON|WAN_STUDENT_BASE_MODEL_PATH|COSMOS_POLICY_PATH)
            printf -v "$key" '%s' "$value"
            ;;
        *) die "unexpected Stage-1 lineage preflight output: $key" ;;
    esac
done <<< "$stage1_lineage_output"
for required in PARENT_STAGE1_PATH PARENT_STAGE1_CONTRACT_IDENTITY STAGE2_LINEAGE_JSON WAN_STUDENT_BASE_MODEL_PATH COSMOS_POLICY_PATH; do
    [[ -n "${!required:-}" ]] || die "Stage-1 lineage preflight omitted $required"
done
STAGE1_CHECKPOINT="$PARENT_STAGE1_PATH"
STAGE1_TARGET="$STAGE1_CHECKPOINT/target_student"

DATASET_PATH="${DATASET_PATH:-/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-$DATASET_PATH/empty_emb.pt}"
COSMOS_PREDICT25_LOCAL_MODEL_DIR="${COSMOS_PREDICT25_LOCAL_MODEL_DIR:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World}"
DATASET_PATH="$(canonical "$DATASET_PATH")"
EMPTY_EMB_PATH="$(canonical "$EMPTY_EMB_PATH")"
COSMOS_PREDICT25_LOCAL_MODEL_DIR="$(canonical "$COSMOS_PREDICT25_LOCAL_MODEL_DIR")"
require_dir DATASET_PATH "$DATASET_PATH"
require_file EMPTY_EMB_PATH "$EMPTY_EMB_PATH"
require_dir COSMOS_PREDICT25_LOCAL_MODEL_DIR "$COSMOS_PREDICT25_LOCAL_MODEL_DIR"

COSMOS_WORKER_ENV_ROOT="${COSMOS_WORKER_ENV_ROOT:-/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310}"
COSMOS_POLICY_PYTHON="${COSMOS_POLICY_PYTHON:-$COSMOS_WORKER_ENV_ROOT/bin/python}"
COSMOS_PREDICT2_REPO="${COSMOS_PREDICT2_REPO:-/kpfs-intern/jialongliu/projects/cosmos-predict2.5-formal-compat-441b897}"
COSMOS_POLICY_PYTHON="$(resolve_executable COSMOS_POLICY_PYTHON "$COSMOS_POLICY_PYTHON")"
COSMOS_WORKER_ENV_ROOT="$(canonical "$COSMOS_WORKER_ENV_ROOT")"
COSMOS_PREDICT2_REPO="$(canonical "$COSMOS_PREDICT2_REPO")"
require_dir COSMOS_WORKER_ENV_ROOT "$COSMOS_WORKER_ENV_ROOT"
require_dir COSMOS_PREDICT2_REPO "$COSMOS_PREDICT2_REPO"
case "$COSMOS_POLICY_PYTHON" in
    "$COSMOS_WORKER_ENV_ROOT"/*) ;;
    *) die "COSMOS_POLICY_PYTHON must belong to COSMOS_WORKER_ENV_ROOT" ;;
esac
COSMOS_WORKER_SITE_PACKAGES="${COSMOS_WORKER_SITE_PACKAGES:-$COSMOS_WORKER_ENV_ROOT/lib/python3.10/site-packages}"
COSMOS_WORKER_SITE_PACKAGES="$(canonical "$COSMOS_WORKER_SITE_PACKAGES")"
require_dir COSMOS_WORKER_SITE_PACKAGES "$COSMOS_WORKER_SITE_PACKAGES"
COSMOS_POLICY_EXTRA_PYTHONPATH="${COSMOS_POLICY_EXTRA_PYTHONPATH:-$COSMOS_PREDICT2_REPO/packages/cosmos-cuda:$COSMOS_PREDICT2_REPO/packages/cosmos-oss}"
COSMOS_POLICY_EXTRA_PYTHONPATH="$(
    resolve_path_list COSMOS_POLICY_EXTRA_PYTHONPATH "$COSMOS_POLICY_EXTRA_PYTHONPATH"
)"
DEFAULT_COSMOS_WORKER_CUDA_LIBRARY_PATH="$COSMOS_WORKER_SITE_PACKAGES/nvidia/cublas/lib:$COSMOS_WORKER_SITE_PACKAGES/nvidia/cuda_cupti/lib:$COSMOS_WORKER_SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$COSMOS_WORKER_SITE_PACKAGES/nvidia/cuda_runtime/lib:$COSMOS_WORKER_SITE_PACKAGES/nvidia/cudnn/lib:$COSMOS_WORKER_SITE_PACKAGES/nvidia/cufft/lib:$COSMOS_WORKER_SITE_PACKAGES/nvidia/cufile/lib:$COSMOS_WORKER_SITE_PACKAGES/nvidia/curand/lib:$COSMOS_WORKER_SITE_PACKAGES/nvidia/cusolver/lib:$COSMOS_WORKER_SITE_PACKAGES/nvidia/cusparse/lib:$COSMOS_WORKER_SITE_PACKAGES/nvidia/nccl/lib:$COSMOS_WORKER_SITE_PACKAGES/nvidia/nvjitlink/lib:$COSMOS_WORKER_SITE_PACKAGES/nvidia/nvtx/lib"
COSMOS_WORKER_CUDA_LIBRARY_PATH="$(
    resolve_path_list COSMOS_WORKER_CUDA_LIBRARY_PATH \
        "${COSMOS_WORKER_CUDA_LIBRARY_PATH:-$DEFAULT_COSMOS_WORKER_CUDA_LIBRARY_PATH}"
)"
WORKER_LD_LIBRARY_PATH="$COSMOS_WORKER_CUDA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
AUDITED_COSMOS_PREDICT2_REPO_COMMIT=1eb8457072b4a1adfe1f83c3076e4aa5452cbab2
COSMOS_PREDICT2_REPO_COMMIT="$(
    GIT_OPTIONAL_LOCKS=0 \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_CONFIG_SYSTEM=/dev/null \
    GIT_CONFIG_NOSYSTEM=1 \
        /usr/bin/git -C "$COSMOS_PREDICT2_REPO" rev-parse HEAD
)" || die "COSMOS_PREDICT2_REPO must be a readable Git repository"
[[ "$COSMOS_PREDICT2_REPO_COMMIT" == "$AUDITED_COSMOS_PREDICT2_REPO_COMMIT" ]] || \
    die "COSMOS_PREDICT2_REPO must be at audited commit $AUDITED_COSMOS_PREDICT2_REPO_COMMIT"
cosmos_repo_status="$(
    GIT_OPTIONAL_LOCKS=0 \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_CONFIG_SYSTEM=/dev/null \
    GIT_CONFIG_NOSYSTEM=1 \
        /usr/bin/git -C "$COSMOS_PREDICT2_REPO" status \
            --porcelain=v1 --untracked-files=all
)" || die "COSMOS_PREDICT2_REPO must be a readable Git repository"
[[ -z "$cosmos_repo_status" ]] || die "COSMOS_PREDICT2_REPO must be clean"

WORKER_PROBE_PYTHONPATH="$COSMOS_PREDICT2_REPO:$COSMOS_POLICY_EXTRA_PYTHONPATH:$COSMOS_WORKER_SITE_PACKAGES"
worker_probe_output="$(
    /usr/bin/env -i \
        PATH=/usr/bin:/bin \
        HOME=/nonexistent \
        LANG=C.UTF-8 \
        LC_ALL=C.UTF-8 \
        PYTHONNOUSERSITE=1 \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="$WORKER_PROBE_PYTHONPATH" \
        LD_LIBRARY_PATH="$WORKER_LD_LIBRARY_PATH" \
        COSMOS_PREDICT2_REPO="$COSMOS_PREDICT2_REPO" \
        "$COSMOS_POLICY_PYTHON" -c '
import re
import torch
import cosmos_predict2

cuda = torch.version.cuda or ""
match = re.fullmatch(r"([0-9]+)\.([0-9]+)(?:\.[0-9]+)?", cuda)
if match is None or (int(match.group(1)), int(match.group(2))) < (12, 8):
    raise SystemExit(
        f"Cosmos worker CUDA runtime {cuda!r} is incompatible; requires CUDA >=12.8"
    )
print("COSMOS_WORKER_RUNTIME_OK=" + cuda)
'
)" || die "Cosmos worker runtime probe failed before any output write"
[[ "$worker_probe_output" == COSMOS_WORKER_RUNTIME_OK=* ]] || \
    die "Cosmos worker runtime probe returned an invalid response"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
[[ "$CUDA_VISIBLE_DEVICES" == "0,1,2,3,4,5,6,7" ]] || \
    die "CUDA_VISIBLE_DEVICES must be exactly 0,1,2,3,4,5,6,7"
export COSMOS_STAGE1_EXPECTED_STEP="$STAGE1_STEP"

LOCK_PREPARER="${COSMOS_LOCK_PREPARER:-$SCRIPT_DIR/prepare_cosmos_libero_provenance_locks.py}"
STAGE2_LAUNCHER="${COSMOS_STAGE2_LAUNCHER:-$SCRIPT_DIR/run_cosmos_libero_train_8gpu.sh}"
PROMPT_TABLE_BUILDER="${COSMOS_WAN_PROMPT_TABLE_BUILDER:-$SCRIPT_DIR/build_cosmos_wan_prompt_table.py}"
EVAL_LAUNCHER="${COSMOS_JOINT124_EVAL_LAUNCHER:-$PROJECT_ROOT/evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh}"
for executable in "$LOCK_PREPARER" "$STAGE2_LAUNCHER" "$PROMPT_TABLE_BUILDER" "$EVAL_LAUNCHER"; do
    [[ -x "$executable" ]] || die "required executable is missing: $executable"
done

if [[ "$LOCK_PREPARER" == *.py ]]; then
    lock_command=(env "PYTHONDONTWRITEBYTECODE=1" "PYTHONPATH=$HELPER_PYTHONPATH" "$PYTHON_BIN" "$LOCK_PREPARER")
else
    lock_command=("$LOCK_PREPARER")
fi
lock_command+=(
    --output-root "$LOCK_ROOT"
    --dataset-root "$DATASET_PATH"
    --teacher-root "$COSMOS_POLICY_PATH"
    --local-model-root "$COSMOS_PREDICT25_LOCAL_MODEL_DIR"
    --stage1-target-root "$STAGE1_TARGET"
)
lock_plan=("${lock_command[@]}" --dry-run)

if [[ "$PROMPT_TABLE_BUILDER" == *.py ]]; then
    prompt_command=(env "PYTHONDONTWRITEBYTECODE=1" "PYTHONPATH=$HELPER_PYTHONPATH" "$PYTHON_BIN" "$PROMPT_TABLE_BUILDER")
else
    prompt_command=("$PROMPT_TABLE_BUILDER")
fi
prompt_command+=(--wan-base-model "$WAN_STUDENT_BASE_MODEL_PATH" --output "$PROMPT_TABLE")
PROMPT_TABLE_MODE=build
if [[ -e "$PROMPT_TABLE" || -L "$PROMPT_TABLE" ]]; then
    [[ -f "$PROMPT_TABLE" && ! -L "$PROMPT_TABLE" ]] || \
        die "prompt table must be a plain file: $PROMPT_TABLE"
    PROMPT_TABLE_MODE=validate
    prompt_command+=(--validate-only)
    prompt_plan=("${prompt_command[@]}")
else
    prompt_plan=("${prompt_command[@]}" --dry-run)
fi

common_runtime_environment=(
    "COSMOS_PREDICT2_REPO=$COSMOS_PREDICT2_REPO"
    "COSMOS_PREDICT2_REPO_COMMIT=$COSMOS_PREDICT2_REPO_COMMIT"
    "COSMOS_POLICY_PYTHON=$COSMOS_POLICY_PYTHON"
    "COSMOS_POLICY_EXTRA_PYTHONPATH=$COSMOS_POLICY_EXTRA_PYTHONPATH"
    "COSMOS_WORKER_ENV_ROOT=$COSMOS_WORKER_ENV_ROOT"
    "COSMOS_WORKER_SITE_PACKAGES=$COSMOS_WORKER_SITE_PACKAGES"
    "COSMOS_WORKER_CUDA_LIBRARY_PATH=$COSMOS_WORKER_CUDA_LIBRARY_PATH"
    "LD_LIBRARY_PATH=$WORKER_LD_LIBRARY_PATH"
)
common_lineage_environment=(
    "STUDENT_BASE_MODEL_PATH=$STAGE1_TARGET"
    "WAN_STUDENT_BASE_MODEL_PATH=$WAN_STUDENT_BASE_MODEL_PATH"
    "COSMOS_POLICY_PATH=$COSMOS_POLICY_PATH"
    "RESUME_FROM_PATH=$STAGE1_CHECKPOINT"
    "PARENT_STAGE1_PATH=$PARENT_STAGE1_PATH"
    "PARENT_STAGE1_CONTRACT_IDENTITY=$PARENT_STAGE1_CONTRACT_IDENTITY"
    "COSMOS_STAGE1_EXPECTED_STEP=$STAGE1_STEP"
    "STAGE2_LINEAGE_JSON=$STAGE2_LINEAGE_JSON"
    "RESUME_ONLINE_FROM_TARGET=1"
    "RESET_RESUME_STEP=1"
    "RESUME_OPTIMIZER_STATE=0"
)
stage2_environment=(
    env
    "COSMOS_STAGE1_ROOT=$STAGE1_CHECKPOINT"
    "${common_lineage_environment[@]}"
    "COSMOS_PROVENANCE_LOCK_ROOT=$LOCK_ROOT"
    "VERIFY_LARGE_ARTIFACT_DIGESTS=1"
    "DATASET_PATH=$DATASET_PATH"
    "EMPTY_EMB_PATH=$EMPTY_EMB_PATH"
    "COSMOS_PREDICT25_LOCAL_MODEL_DIR=$COSMOS_PREDICT25_LOCAL_MODEL_DIR"
    "${common_runtime_environment[@]}"
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128"
    "PIPELINE_RUN_ROOT=$RUN_ROOT"
)
stage2_command=(
    bash "$STAGE2_LAUNCHER" universal-video-action
    --steps "$STAGE2_STEPS"
    --save-interval "$SAVE_INTERVAL"
    --master-port "$MASTER_PORT"
    --output-root "$OUTPUT_ROOT"
    --run-tag "$RUN_TAG"
)
stage2_execution=("${stage2_environment[@]}" "${stage2_command[@]}")
eval_environment=(
    env
    "${common_lineage_environment[@]}"
    "${common_runtime_environment[@]}"
    "MATRIX_ROOT=$MATRIX_ROOT"
    "S4_CKPT_ROOT=$STAGE2_TARGET_TRANSFORMER"
    "S4_MATRIX_ROLES=stage2_target"
    "S4_FORMAL_NUM_SHARDS=4"
    "S4_FORMAL_GPU_LAYOUT=paired"
    "S4_VIDEO_SEEDS=0"
    "S4_EPISODES_PER_TASK=$EPISODES"
    "S4_ALIGNMENT_VERIFIED=1"
    "S4_DATASET_PATH=$DATASET_PATH"
    "S4_EMPTY_EMBEDDING=$EMPTY_EMB_PATH"
    "S4_PROMPT_TABLE=$PROMPT_TABLE"
    "COSMOS_PREDICT25_LOCAL_MODEL_DIR=$COSMOS_PREDICT25_LOCAL_MODEL_DIR"
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128"
    "PYTHON_BIN=$PYTHON_BIN"
)
eval_execution=("${eval_environment[@]}" bash "$EVAL_LAUNCHER" run)
eval_plan=("${eval_environment[@]}" bash "$EVAL_LAUNCHER" dry-run)

preflight_stage2_inference() {
    (
        cd "$PROJECT_ROOT"
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="$HELPER_PYTHONPATH" \
            "$PYTHON_BIN" - "$STAGE2_TARGET_TRANSFORMER" \
                "$STAGE2_STEPS" "$PARENT_STAGE1_PATH" "$STAGE1_STEP" \
                "$WAN_STUDENT_BASE_MODEL_PATH" "$COSMOS_POLICY_PATH" <<'PY'
import sys
from pathlib import Path

from distillation_flowmap.cosmos_stage2_lineage import (
    bind_stage2_inference_runtime,
    resolve_cosmos_inference_checkpoint,
)

(
    transformer_raw,
    expected_step_raw,
    expected_parent,
    expected_parent_step_raw,
    expected_wan,
    expected_teacher,
) = sys.argv[1:]
transformer = Path(transformer_raw)
resolved = resolve_cosmos_inference_checkpoint(
    model_role="stage2_target",
    checkpoint_transformer=transformer,
)
expected_checkpoint = transformer.parent.parent.resolve(strict=True)
if Path(resolved.checkpoint_path) != expected_checkpoint:
    raise ValueError("Stage-2 inference checkpoint path does not match requested target")
if expected_checkpoint.name != f"step_{int(expected_step_raw)}":
    raise ValueError("Stage-2 inference checkpoint directory does not match expected step")
if resolved.parent_stage1_path != expected_parent:
    raise ValueError("Stage-2 inference parent does not match requested Stage-1")
if resolved.parent_stage1_expected_step != int(expected_parent_step_raw):
    raise ValueError("Stage-2 inference parent step does not match requested Stage-1")
if resolved.wan_student_base_model_path != expected_wan:
    raise ValueError("Stage-2 inference Wan base does not match validated Stage-1")
if resolved.cosmos_teacher_model_path != expected_teacher:
    raise ValueError("Stage-2 inference Teacher does not match validated Stage-1")
bind_stage2_inference_runtime(
    resolved,
    environment={},
    configured_teacher_model_path=expected_teacher,
)
print(
    "STAGE2_CHECKPOINT_CONTRACT_IDENTITY="
    + resolved.checkpoint_contract_identity
)
PY
    ) || die "real Stage-2 inference checkpoint preflight failed"
}

printf 'PHASE=%s\n' "$PHASE"
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
printf 'STAGE1_CHECKPOINT=%s\n' "$STAGE1_CHECKPOINT"
printf 'PARENT_STAGE1_CONTRACT_IDENTITY=%s\n' "$PARENT_STAGE1_CONTRACT_IDENTITY"
printf 'COSMOS_STAGE1_EXPECTED_STEP=%s\n' "$STAGE1_STEP"
printf 'WAN_STUDENT_BASE_MODEL_PATH=%s\n' "$WAN_STUDENT_BASE_MODEL_PATH"
printf 'DATASET_PATH=%s\n' "$DATASET_PATH"
printf 'COSMOS_PREDICT2_REPO=%s\n' "$COSMOS_PREDICT2_REPO"
printf 'COSMOS_PREDICT2_REPO_COMMIT=%s\n' "$COSMOS_PREDICT2_REPO_COMMIT"
printf 'COSMOS_POLICY_PYTHON=%s\n' "$COSMOS_POLICY_PYTHON"
printf 'COSMOS_POLICY_EXTRA_PYTHONPATH=%s\n' "$COSMOS_POLICY_EXTRA_PYTHONPATH"
printf 'COSMOS_WORKER_ENV_ROOT=%s\n' "$COSMOS_WORKER_ENV_ROOT"
printf 'COSMOS_WORKER_SITE_PACKAGES=%s\n' "$COSMOS_WORKER_SITE_PACKAGES"
printf 'COSMOS_WORKER_CUDA_LIBRARY_PATH=%s\n' "$COSMOS_WORKER_CUDA_LIBRARY_PATH"
printf 'WORKER_LD_LIBRARY_PATH=%s\n' "$WORKER_LD_LIBRARY_PATH"
printf 'STAGE2_STEPS=%s\n' "$STAGE2_STEPS"
printf 'MAX_TRAIN_STEPS=%s\n' "$STAGE2_STEPS"
printf 'STAGE2_TARGET_TRANSFORMER=%s\n' "$STAGE2_TARGET_TRANSFORMER"
printf 'EVAL_CHECKPOINT_ROLES=stage2_target\n'
printf 'S4_MATRIX_ROLES=stage2_target\n'
printf 'EVAL_PROTOCOL=student-only Full40 K=1,2,4; %s episodes per task\n' "$EPISODES"
printf 'PROMPT_TABLE_MODE=%s\n' "$PROMPT_TABLE_MODE"
printf 'S4_PROMPT_TABLE=%s\n' "$PROMPT_TABLE"
print_command LOCK_COMMAND "${lock_command[@]}"
print_command STAGE2_COMMAND "${stage2_execution[@]}"
print_command PROMPT_TABLE_COMMAND "${prompt_command[@]}"
print_command EVAL_COMMAND "${eval_execution[@]}"

if [[ -n "$READ_ONLY_MODE" ]]; then
    printf 'PIPELINE_MODE=%s\n' "$READ_ONLY_MODE"
    if [[ "$PHASE" == all || "$PHASE" == stage2 ]]; then
        print_command LOCK_PREFLIGHT "${lock_plan[@]}"
        run_child "${lock_plan[@]}"
    fi
    if [[ "$PHASE" == eval ]]; then
        preflight_stage2_inference
    fi
    if [[ "$PHASE" == all || "$PHASE" == eval ]]; then
        print_command PROMPT_TABLE_PREFLIGHT "${prompt_plan[@]}"
        run_child "${prompt_plan[@]}"
        print_command EVAL_PLAN "${eval_plan[@]}"
        run_child "${eval_plan[@]}"
    fi
    exit 0
fi

if [[ "$PHASE" == all || "$PHASE" == stage2 ]]; then
    [[ ! -e "$STAGE2_OUTPUT" && ! -L "$STAGE2_OUTPUT" ]] || \
        die "Stage-2 output already exists: $STAGE2_OUTPUT"
    [[ ! -e "$LOCK_ROOT" && ! -L "$LOCK_ROOT" ]] || \
        die "provenance lock root already exists: $LOCK_ROOT"
fi
if [[ "$PHASE" == all || "$PHASE" == eval ]]; then
    [[ ! -e "$MATRIX_ROOT" && ! -L "$MATRIX_ROOT" ]] || \
        die "evaluation matrix already exists: $MATRIX_ROOT"
fi

if [[ "$PHASE" == eval ]]; then
    preflight_stage2_inference
fi
if [[ "$PHASE" == all || "$PHASE" == stage2 ]]; then
    run_child "${lock_plan[@]}"
fi
if [[ "$PHASE" == all || "$PHASE" == eval ]]; then
    run_child "${prompt_plan[@]}"
fi

if [[ "$PHASE" == all || "$PHASE" == stage2 ]]; then
    mkdir -p "$RUN_ROOT"
    run_child "${lock_command[@]}"
    run_child "${stage2_execution[@]}"
    preflight_stage2_inference
fi
[[ "$PHASE" != stage2 ]] || exit 0

run_child "${prompt_command[@]}"
run_child "${eval_execution[@]}"
