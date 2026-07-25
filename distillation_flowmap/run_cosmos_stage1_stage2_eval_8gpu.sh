#!/usr/bin/env bash
# One-allocation Cosmos-teacher / LingBotVA-Wan-student Stage1→Stage2→eval chain.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh \
    --phase all|stage1|stage2|eval --output-root PATH --run-tag TAG [options]

Options:
  --stage1-steps N          default: 5000
  --stage2-steps N          default: 10000
  --save-interval N         default: 1000
  --stage1-master-port PORT default: 29671
  --stage2-master-port PORT default: 29672
  --resume-stage stage1|stage2 --resume-step N
  --dry-run                 print the complete plan; write nothing
  --check-only              validate static inputs and print the plan; write nothing

This is not a pure-Cosmos student run. It explicitly initializes the Flash-WAM/
Wan student from CLEAN_STUDENT_BASE_MODEL_PATH and uses a Cosmos policy teacher.
Formal evaluation is LIBERO-10 only: 500 episodes for each matched K=1,2,4.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

require_env() {
    local name="$1"
    [[ -n "${!name:-}" ]] || die "$name must be set explicitly"
}

positive() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer"
}

port() {
    positive "$1" "$2"
    (( 10#$2 <= 65535 )) || die "$1 must be at most 65535"
}

require_dir() {
    [[ -d "$2" && ! -L "$2" ]] || die "$1 must be a plain directory: $2"
}

require_transformer() {
    local label="$1"
    local root="$2"
    require_dir "$label" "$root"
    [[ -f "$root/config.json" && ! -L "$root/config.json" ]] || \
        die "$label config.json is missing: $root/config.json"
    if [[ -f "$root/diffusion_pytorch_model.safetensors" ]]; then
        return
    fi
    [[ -f "$root/diffusion_pytorch_model.safetensors.index.json" ]] || \
        die "$label weights are missing: $root"
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

PHASE=""
OUTPUT_ROOT=""
RUN_TAG=""
STAGE1_STEPS=5000
STAGE2_STEPS=10000
SAVE_INTERVAL=1000
STAGE1_MASTER_PORT=29671
STAGE2_MASTER_PORT=29672
RESUME_STAGE=""
RESUME_STEP=""
READ_ONLY_MODE=""

while (( $# > 0 )); do
    case "$1" in
        --phase) PHASE="${2:-}"; shift 2 ;;
        --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
        --run-tag) RUN_TAG="${2:-}"; shift 2 ;;
        --stage1-steps) STAGE1_STEPS="${2:-}"; shift 2 ;;
        --stage2-steps) STAGE2_STEPS="${2:-}"; shift 2 ;;
        --save-interval) SAVE_INTERVAL="${2:-}"; shift 2 ;;
        --stage1-master-port) STAGE1_MASTER_PORT="${2:-}"; shift 2 ;;
        --stage2-master-port) STAGE2_MASTER_PORT="${2:-}"; shift 2 ;;
        --resume-stage) RESUME_STAGE="${2:-}"; shift 2 ;;
        --resume-step) RESUME_STEP="${2:-}"; shift 2 ;;
        --dry-run)
            [[ -z "$READ_ONLY_MODE" ]] || die "choose only one read-only mode"
            READ_ONLY_MODE=dry-run
            shift
            ;;
        --check-only)
            [[ -z "$READ_ONLY_MODE" ]] || die "choose only one read-only mode"
            READ_ONLY_MODE=check-only
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$PHASE" in all|stage1|stage2|eval) ;; *) die "--phase must be all, stage1, stage2, or eval" ;; esac
[[ -n "$OUTPUT_ROOT" ]] || die "--output-root is required"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "--run-tag must be one path-safe component"
positive STAGE1_STEPS "$STAGE1_STEPS"
positive STAGE2_STEPS "$STAGE2_STEPS"
positive SAVE_INTERVAL "$SAVE_INTERVAL"
port STAGE1_MASTER_PORT "$STAGE1_MASTER_PORT"
port STAGE2_MASTER_PORT "$STAGE2_MASTER_PORT"
(( 10#$STAGE1_STEPS % 10#$SAVE_INTERVAL == 0 )) || die "Stage-1 steps must be divisible by save interval"
(( 10#$STAGE2_STEPS % 10#$SAVE_INTERVAL == 0 )) || die "Stage-2 steps must be divisible by save interval"
if [[ -n "$RESUME_STAGE" || -n "$RESUME_STEP" ]]; then
    [[ "$RESUME_STAGE" == stage1 || "$RESUME_STAGE" == stage2 ]] || die \
        "--resume-stage must be stage1 or stage2"
    positive RESUME_STEP "$RESUME_STEP"
    [[ "$PHASE" == "$RESUME_STAGE" ]] || die \
        "--phase must equal --resume-stage for an exact resume"
fi

require_env CLEAN_STUDENT_BASE_MODEL_PATH
require_env COSMOS_PREDICT2_REPO
require_dir CLEAN_STUDENT_BASE_MODEL_PATH "$CLEAN_STUDENT_BASE_MODEL_PATH"
require_transformer \
    CLEAN_STUDENT_BASE_MODEL_PATH/transformer \
    "$CLEAN_STUDENT_BASE_MODEL_PATH/transformer"
require_dir COSMOS_PREDICT2_REPO "$COSMOS_PREDICT2_REPO"

git_status="$(
    GIT_OPTIONAL_LOCKS=0 \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_CONFIG_SYSTEM=/dev/null \
    GIT_CONFIG_NOSYSTEM=1 \
    /usr/bin/git -C "$COSMOS_PREDICT2_REPO" status \
        --porcelain=v1 --untracked-files=all
)" || die "COSMOS_PREDICT2_REPO must be a readable Git repository"
[[ -z "$git_status" ]] || die "COSMOS_PREDICT2_REPO must be clean"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"
STAGE1_LAUNCHER="${COSMOS_RAW_STAGE1_LAUNCHER:-$SCRIPT_DIR/run_cosmos_raw_stage1_8gpu.sh}"
STAGE2_LAUNCHER="${COSMOS_STAGE2_LAUNCHER:-$SCRIPT_DIR/run_cosmos_libero_train_8gpu.sh}"
EVAL_LAUNCHER="${COSMOS_JOINT124_EVAL_LAUNCHER:-$PROJECT_ROOT/evaluation/libero/run_cosmos_progressive_joint_124_eval_8gpu.sh}"
LOCK_PREPARER="${COSMOS_LOCK_PREPARER:-$SCRIPT_DIR/prepare_cosmos_libero_provenance_locks.py}"
for executable in "$STAGE1_LAUNCHER" "$STAGE2_LAUNCHER" "$EVAL_LAUNCHER" "$LOCK_PREPARER"; do
    [[ -x "$executable" ]] || die "required executable is missing: $executable"
done

DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/training_data/libero-long-lerobot}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-$DATASET_PATH/empty_emb.pt}"
COSMOS_POLICY_PATH="${COSMOS_POLICY_PATH:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
COSMOS_PREDICT25_LOCAL_MODEL_DIR="${COSMOS_PREDICT25_LOCAL_MODEL_DIR:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World}"
require_dir DATASET_PATH "$DATASET_PATH"
[[ -f "$EMPTY_EMB_PATH" ]] || die "EMPTY_EMB_PATH is missing: $EMPTY_EMB_PATH"
require_dir COSMOS_POLICY_PATH "$COSMOS_POLICY_PATH"
require_dir COSMOS_PREDICT25_LOCAL_MODEL_DIR "$COSMOS_PREDICT25_LOCAL_MODEL_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
(( ${#devices[@]} == 8 )) || die "CUDA_VISIBLE_DEVICES must contain exactly 8 devices"

OUTPUT_ROOT="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$OUTPUT_ROOT")"
RUN_ROOT="$OUTPUT_ROOT/$RUN_TAG"
STAGE1_OUTPUT="$RUN_ROOT/stage1"
STAGE1_CHECKPOINT="$STAGE1_OUTPUT/checkpoints/step_$STAGE1_STEPS"
STAGE1_TARGET="$STAGE1_CHECKPOINT/target_student"
LOCK_ROOT="$RUN_ROOT/provenance-locks"
STAGE2_OUTPUT="$RUN_ROOT/universal-video-action"
STAGE2_CHECKPOINT="$STAGE2_OUTPUT/checkpoints/step_$STAGE2_STEPS"
EVAL_TRANSFORMER="$STAGE2_CHECKPOINT/target_student/transformer"
MATRIX_ROOT="$RUN_ROOT/eval/stage2_target"

stage1_command=(
    "$STAGE1_LAUNCHER" run
    --steps "$STAGE1_STEPS"
    --save-interval "$SAVE_INTERVAL"
    --master-port "$STAGE1_MASTER_PORT"
    --output-dir "$STAGE1_OUTPUT"
    --run-tag "$RUN_TAG"
)
stage2_command=(
    "$STAGE2_LAUNCHER" universal-video-action
    --steps "$STAGE2_STEPS"
    --save-interval "$SAVE_INTERVAL"
    --master-port "$STAGE2_MASTER_PORT"
    --output-root "$OUTPUT_ROOT"
    --run-tag "$RUN_TAG"
)
if [[ "$RESUME_STAGE" == stage1 ]]; then
    stage1_command+=(--resume-step "$RESUME_STEP")
elif [[ "$RESUME_STAGE" == stage2 ]]; then
    stage2_command+=(--resume-step "$RESUME_STEP")
fi
if [[ "$LOCK_PREPARER" == *.py ]]; then
    lock_command=("$PYTHON_BIN" "$LOCK_PREPARER")
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
eval_command=("$EVAL_LAUNCHER" run)

printf 'PIPELINE_SEMANTICS=LingBotVA/Wan student init + Cosmos teacher\n'
printf 'PHASE=%s\n' "$PHASE"
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
printf 'STAGE1_CHECKPOINT=%s\n' "$STAGE1_CHECKPOINT"
printf 'STAGE2_CHECKPOINT=%s\n' "$STAGE2_CHECKPOINT"
printf 'EVAL_CHECKPOINT_ROLE=stage2_target\n'
printf 'EVAL_PROTOCOL=LIBERO-10 matched joint 1/2/4, 500 episodes per K\n'
print_command STAGE1_COMMAND "${stage1_command[@]}"
print_command LOCK_COMMAND "${lock_command[@]}"
print_command STAGE2_COMMAND "${stage2_command[@]}"
print_command EVAL_COMMAND "${eval_command[@]}"

if [[ -n "$READ_ONLY_MODE" ]]; then
    printf 'PIPELINE_MODE=%s\n' "$READ_ONLY_MODE"
    exit 0
fi

if [[ "$PHASE" == all ]]; then
    [[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || die "RUN_ROOT already exists: $RUN_ROOT"
elif [[ "$PHASE" == stage1 ]]; then
    if [[ "$RESUME_STAGE" == stage1 ]]; then
        [[ -d "$STAGE1_OUTPUT/checkpoints/step_$RESUME_STEP" ]] || die \
            "exact Stage-1 resume checkpoint is missing"
    else
        [[ ! -e "$STAGE1_OUTPUT" && ! -L "$STAGE1_OUTPUT" ]] || die \
            "Stage-1 output already exists: $STAGE1_OUTPUT"
    fi
elif [[ "$PHASE" == stage2 ]]; then
    require_transformer Stage1-target "$STAGE1_TARGET/transformer"
    if [[ "$RESUME_STAGE" == stage2 ]]; then
        [[ -d "$STAGE2_OUTPUT/checkpoints/step_$RESUME_STEP" ]] || die \
            "exact Stage-2 resume checkpoint is missing"
        require_dir provenance-locks "$LOCK_ROOT"
    else
        [[ ! -e "$STAGE2_OUTPUT" && ! -L "$STAGE2_OUTPUT" ]] || die \
            "Stage-2 output already exists: $STAGE2_OUTPUT"
    fi
else
    require_transformer Stage2-target "$EVAL_TRANSFORMER"
    [[ ! -e "$MATRIX_ROOT" && ! -L "$MATRIX_ROOT" ]] || die \
        "evaluation matrix already exists: $MATRIX_ROOT"
fi

export PIPELINE_RUN_ROOT="$RUN_ROOT"
export PIPELINE_STAGE2_STEPS="$STAGE2_STEPS"

if [[ "$PHASE" == all || "$PHASE" == stage1 ]]; then
    run_child "${stage1_command[@]}"
    require_transformer Stage1-target "$STAGE1_TARGET/transformer"
fi
[[ "$PHASE" != stage1 ]] || exit 0

if [[ "$PHASE" == all || ( "$PHASE" == stage2 && -z "$RESUME_STAGE" ) ]]; then
    [[ ! -e "$LOCK_ROOT" && ! -L "$LOCK_ROOT" ]] || die \
        "provenance lock root already exists: $LOCK_ROOT"
    run_child "${lock_command[@]}"
fi

if [[ "$PHASE" == all || "$PHASE" == stage2 ]]; then
    export COSMOS_STAGE1_ROOT="$STAGE1_CHECKPOINT"
    export STUDENT_BASE_MODEL_PATH="$STAGE1_TARGET"
    export COSMOS_PROVENANCE_LOCK_ROOT="$LOCK_ROOT"
    export VERIFY_LARGE_ARTIFACT_DIGESTS=1
    run_child "${stage2_command[@]}"
    require_transformer Stage2-target "$EVAL_TRANSFORMER"
fi
[[ "$PHASE" != stage2 ]] || exit 0

export MATRIX_ROOT
export S4_CKPT_ROOT="$EVAL_TRANSFORMER"
export S4_MODEL_ROLE=stage2_target
export S4_DATASET_PATH="$DATASET_PATH"
export S4_EMPTY_EMBEDDING="$EMPTY_EMB_PATH"
export S4_ALIGNMENT_VERIFIED=1
run_child "${eval_command[@]}"
