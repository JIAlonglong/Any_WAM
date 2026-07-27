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
  --dry-run                      print the plan; write nothing
  --check-only                   validate and print the plan; write nothing
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
positive() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a strict positive decimal integer"; }
require_dir() { [[ -d "$2" && ! -L "$2" ]] || die "$1 must be a plain directory: $2"; }
require_file() { [[ -f "$2" && ! -L "$2" ]] || die "$1 is missing: $2"; }
require_transformer() {
    local label="$1" root="$2"
    require_dir "$label" "$root"
    require_file "$label/config.json" "$root/config.json"
    [[ -f "$root/diffusion_pytorch_model.safetensors" || \
       -f "$root/diffusion_pytorch_model.safetensors.index.json" ]] || \
        die "$label weights are missing: $root"
}
canonical() {
    "$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=False))' "$1"
}
validate_devices() {
    local raw="$1" ordinal
    local -a ordinals
    local -A seen=()
    IFS=',' read -r -a ordinals <<< "$raw"
    (( ${#ordinals[@]} == 8 )) || die "CUDA_VISIBLE_DEVICES must contain exactly 8 comma-separated GPU ordinals"
    for ordinal in "${ordinals[@]}"; do
        [[ "$ordinal" =~ ^[0-9]+$ ]] || die "CUDA_VISIBLE_DEVICES contains a non-numeric GPU ordinal: $ordinal"
        [[ -z "${seen[$ordinal]:-}" ]] || die "CUDA_VISIBLE_DEVICES contains duplicate GPU ordinal: $ordinal"
        seen["$ordinal"]=1
    done
}
print_command() { local label="$1"; shift; printf '%s=' "$label"; printf ' %q' "$@"; printf '\n'; }

ACTIVE_PID=""
forward_signal() {
    local signal="$1"
    if [[ -n "$ACTIVE_PID" ]]; then
        kill "-$signal" "$ACTIVE_PID" 2>/dev/null || true
        wait "$ACTIVE_PID" 2>/dev/null || true
    fi
    exit 130
}
trap 'forward_signal INT' INT
trap 'forward_signal TERM' TERM
run_child() {
    "$@" &
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
            READ_ONLY_MODE="${1#--}"; shift ;;
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
[[ "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "--run-tag must be one path-safe component"
positive STAGE1_STEP "$STAGE1_STEP"
positive STAGE2_STEPS "$STAGE2_STEPS"
positive SAVE_INTERVAL "$SAVE_INTERVAL"
positive EPISODES "$EPISODES"
positive MASTER_PORT "$MASTER_PORT"
(( 10#$MASTER_PORT <= 65535 )) || die "--master-port must be at most 65535"
(( 10#$STAGE2_STEPS % 10#$SAVE_INTERVAL == 0 )) || die "Stage-2 steps must be divisible by save interval"

PYTHON_BIN="${PYTHON_BIN:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"
[[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
export PYTHON_BIN
STAGE1_ROOT="$(canonical "$STAGE1_ROOT")"
OUTPUT_ROOT="$(canonical "$OUTPUT_ROOT")"
STAGE1_CHECKPOINT="$STAGE1_ROOT/checkpoints/step_$STAGE1_STEP"
STAGE1_TARGET="$STAGE1_CHECKPOINT/target_student"
LOCK_ROOT="$OUTPUT_ROOT/$RUN_TAG/provenance-locks"
RUN_ROOT="$OUTPUT_ROOT/$RUN_TAG"
STAGE2_OUTPUT="$RUN_ROOT/universal-video-action"
STAGE2_TARGET_TRANSFORMER="$STAGE2_OUTPUT/checkpoints/step_$STAGE2_STEPS/target_student/transformer"
MATRIX_ROOT="$RUN_ROOT/eval/student_only"

require_transformer "Stage-1 target" "$STAGE1_TARGET/transformer"
[[ -n "${WAN_STUDENT_BASE_MODEL_PATH:-}" ]] || die "WAN_STUDENT_BASE_MODEL_PATH must be set explicitly"
WAN_STUDENT_BASE_MODEL_PATH="$(canonical "$WAN_STUDENT_BASE_MODEL_PATH")"
require_transformer "WAN_STUDENT_BASE_MODEL_PATH/transformer" "$WAN_STUDENT_BASE_MODEL_PATH/transformer"
DATASET_PATH="${DATASET_PATH:-$PWD/training_data/libero-long-lerobot}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-$DATASET_PATH/empty_emb.pt}"
COSMOS_POLICY_PATH="${COSMOS_POLICY_PATH:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
COSMOS_PREDICT25_LOCAL_MODEL_DIR="${COSMOS_PREDICT25_LOCAL_MODEL_DIR:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World}"
DATASET_PATH="$(canonical "$DATASET_PATH")"
EMPTY_EMB_PATH="$(canonical "$EMPTY_EMB_PATH")"
COSMOS_POLICY_PATH="$(canonical "$COSMOS_POLICY_PATH")"
COSMOS_PREDICT25_LOCAL_MODEL_DIR="$(canonical "$COSMOS_PREDICT25_LOCAL_MODEL_DIR")"
require_dir DATASET_PATH "$DATASET_PATH"
require_file EMPTY_EMB_PATH "$EMPTY_EMB_PATH"
require_dir COSMOS_POLICY_PATH "$COSMOS_POLICY_PATH"
require_dir COSMOS_PREDICT25_LOCAL_MODEL_DIR "$COSMOS_PREDICT25_LOCAL_MODEL_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
validate_devices "$CUDA_VISIBLE_DEVICES"
export COSMOS_STAGE1_EXPECTED_STEP="$STAGE1_STEP"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
LOCK_PREPARER="${COSMOS_LOCK_PREPARER:-$SCRIPT_DIR/prepare_cosmos_libero_provenance_locks.py}"
STAGE2_LAUNCHER="${COSMOS_STAGE2_LAUNCHER:-$SCRIPT_DIR/run_cosmos_libero_train_8gpu.sh}"
PROMPT_TABLE_BUILDER="${COSMOS_WAN_PROMPT_TABLE_BUILDER:-$SCRIPT_DIR/build_cosmos_wan_prompt_table.py}"
EVAL_LAUNCHER="${COSMOS_JOINT124_EVAL_LAUNCHER:-$PROJECT_ROOT/evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh}"
for executable in "$LOCK_PREPARER" "$STAGE2_LAUNCHER" "$PROMPT_TABLE_BUILDER" "$EVAL_LAUNCHER"; do
    [[ -x "$executable" ]] || die "required executable is missing: $executable"
done

HELPER_PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/wan_va${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$LOCK_PREPARER" == *.py ]]; then lock_command=(env "PYTHONPATH=$HELPER_PYTHONPATH" "$PYTHON_BIN" "$LOCK_PREPARER"); else lock_command=("$LOCK_PREPARER"); fi
lock_command+=(--output-root "$LOCK_ROOT" --dataset-root "$DATASET_PATH" --teacher-root "$COSMOS_POLICY_PATH" --local-model-root "$COSMOS_PREDICT25_LOCAL_MODEL_DIR" --stage1-target-root "$STAGE1_TARGET")
if [[ "$PROMPT_TABLE_BUILDER" == *.py ]]; then prompt_command=(env "PYTHONPATH=$HELPER_PYTHONPATH" "$PYTHON_BIN" "$PROMPT_TABLE_BUILDER"); else prompt_command=("$PROMPT_TABLE_BUILDER"); fi
PROMPT_TABLE="$RUN_ROOT/libero_wan_prompt_embeddings_all40.pt"
prompt_command+=(--wan-base-model "$WAN_STUDENT_BASE_MODEL_PATH" --output "$PROMPT_TABLE")
PROMPT_TABLE_MODE=build
if [[ -e "$PROMPT_TABLE" ]]; then
    [[ -f "$PROMPT_TABLE" && ! -L "$PROMPT_TABLE" ]] || die "prompt table must be a plain file: $PROMPT_TABLE"
    PROMPT_TABLE_MODE=validate
    prompt_command+=(--validate-only)
fi

audited_environment=(
    env
    "COSMOS_STAGE1_ROOT=$STAGE1_CHECKPOINT"
    "STUDENT_BASE_MODEL_PATH=$STAGE1_TARGET"
    "WAN_STUDENT_BASE_MODEL_PATH=$WAN_STUDENT_BASE_MODEL_PATH"
    "COSMOS_PROVENANCE_LOCK_ROOT=$LOCK_ROOT"
    "COSMOS_STAGE1_EXPECTED_STEP=$STAGE1_STEP"
    "DATASET_PATH=$DATASET_PATH"
    "EMPTY_EMB_PATH=$EMPTY_EMB_PATH"
    "COSMOS_POLICY_PATH=$COSMOS_POLICY_PATH"
    "COSMOS_PREDICT25_LOCAL_MODEL_DIR=$COSMOS_PREDICT25_LOCAL_MODEL_DIR"
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128"
    "PIPELINE_RUN_ROOT=$RUN_ROOT"
)
for name in COSMOS_PREDICT2_REPO COSMOS_POLICY_PYTHON COSMOS_POLICY_EXTRA_PYTHONPATH COSMOS_WORKER_ENV_ROOT COSMOS_WORKER_SITE_PACKAGES COSMOS_WORKER_CUDA_LIBRARY_PATH; do
    [[ -z "${!name:-}" ]] || audited_environment+=("$name=${!name}")
done
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then audited_environment+=("LD_LIBRARY_PATH=$LD_LIBRARY_PATH"); fi
stage2_command=(bash "$STAGE2_LAUNCHER" universal-video-action --steps "$STAGE2_STEPS" --save-interval "$SAVE_INTERVAL" --master-port "$MASTER_PORT" --output-root "$OUTPUT_ROOT" --run-tag "$RUN_TAG")
stage2_execution=("${audited_environment[@]}" "${stage2_command[@]}")
eval_environment=(
    env "MATRIX_ROOT=$MATRIX_ROOT" "S4_CKPT_ROOT=$STAGE2_TARGET_TRANSFORMER"
    "S4_MATRIX_ROLES=stage2_target" "S4_FORMAL_NUM_SHARDS=4" "S4_FORMAL_GPU_LAYOUT=paired"
    "S4_VIDEO_SEEDS=0" "S4_EPISODES_PER_TASK=$EPISODES" "S4_ALIGNMENT_VERIFIED=1"
    "S4_DATASET_PATH=$DATASET_PATH" "S4_EMPTY_EMBEDDING=$EMPTY_EMB_PATH" "S4_PROMPT_TABLE=$PROMPT_TABLE"
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128" "PYTHON_BIN=$PYTHON_BIN"
)
for name in WAN_STUDENT_BASE_MODEL_PATH COSMOS_PREDICT2_REPO COSMOS_POLICY_PYTHON COSMOS_POLICY_EXTRA_PYTHONPATH COSMOS_WORKER_ENV_ROOT COSMOS_WORKER_SITE_PACKAGES COSMOS_WORKER_CUDA_LIBRARY_PATH; do
    [[ -z "${!name:-}" ]] || eval_environment+=("$name=${!name}")
done
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then eval_environment+=("LD_LIBRARY_PATH=$LD_LIBRARY_PATH"); fi
eval_execution=("${eval_environment[@]}" bash "$EVAL_LAUNCHER" run)
eval_plan=("${eval_environment[@]}" bash "$EVAL_LAUNCHER" dry-run)

printf 'PHASE=%s\n' "$PHASE"
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
printf 'STAGE1_CHECKPOINT=%s\n' "$STAGE1_CHECKPOINT"
printf 'STAGE2_STEPS=%s\n' "$STAGE2_STEPS"
printf 'MAX_TRAIN_STEPS=%s\n' "$STAGE2_STEPS"
printf 'STAGE2_TARGET_TRANSFORMER=%s\n' "$STAGE2_TARGET_TRANSFORMER"
printf 'EVAL_CHECKPOINT_ROLES=stage2_target\n'
printf 'EVAL_PROTOCOL=student-only Full40 K=1,2,4; %s episodes per task\n' "$EPISODES"
printf 'PROMPT_TABLE_MODE=%s\n' "$PROMPT_TABLE_MODE"
printf 'S4_PROMPT_TABLE=%s\n' "$PROMPT_TABLE"
print_command LOCK_COMMAND "${lock_command[@]}"
print_command STAGE2_COMMAND "${stage2_execution[@]}"
print_command PROMPT_TABLE_COMMAND "${prompt_command[@]}"
print_command EVAL_COMMAND "${eval_execution[@]}"

if [[ -n "$READ_ONLY_MODE" ]]; then
    printf 'PIPELINE_MODE=%s\n' "$READ_ONLY_MODE"
    print_command EVAL_PLAN "${eval_plan[@]}"
    run_child "${eval_plan[@]}"
    exit 0
fi

if [[ "$PHASE" == all || "$PHASE" == stage2 ]]; then
    [[ ! -e "$STAGE2_OUTPUT" && ! -L "$STAGE2_OUTPUT" ]] || die "Stage-2 output already exists: $STAGE2_OUTPUT"
    [[ ! -e "$LOCK_ROOT" && ! -L "$LOCK_ROOT" ]] || die "provenance lock root already exists: $LOCK_ROOT"
fi
if [[ "$PHASE" == all || "$PHASE" == eval ]]; then
    [[ ! -e "$MATRIX_ROOT" && ! -L "$MATRIX_ROOT" ]] || die "evaluation matrix already exists: $MATRIX_ROOT"
fi

if [[ "$PHASE" == all || "$PHASE" == stage2 ]]; then
    mkdir -p "$RUN_ROOT"
    run_child "${lock_command[@]}"
    run_child "${stage2_execution[@]}"
    require_transformer "Stage-2 target" "$STAGE2_TARGET_TRANSFORMER"
fi
[[ "$PHASE" != stage2 ]] || exit 0

if [[ "$PHASE" == eval ]]; then
    require_transformer "Stage-2 target" "$STAGE2_TARGET_TRANSFORMER"
fi
run_child "${prompt_command[@]}"
run_child "${eval_execution[@]}"
