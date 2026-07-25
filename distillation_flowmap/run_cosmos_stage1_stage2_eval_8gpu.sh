#!/usr/bin/env bash
# One-allocation Cosmos-teacher / LingBotVA-Wan-student Stage1→Stage2→eval chain.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh \
    --phase all|stage1|stage2|eval|check --output-root PATH --run-tag TAG [options]

Options:
  --stage1-steps N          default: 5000
  --stage2-steps N          default: 10000
  --steps N                 smoke alias: use N for both training stages
  --episodes N              episodes per LIBERO task (default: 50)
  --save-interval N         default: 1000
  --stage1-master-port PORT default: 29671
  --stage2-master-port PORT default: 29672
  --resume-stage stage1|stage2 --resume-step N
  --dry-run                 print the complete plan; write nothing
  --check-only              validate static inputs and print the plan; write nothing

This is not a pure-Cosmos student run. It explicitly initializes the Flash-WAM/
Wan student from WAN_STUDENT_BASE_MODEL_PATH and uses a Cosmos policy teacher.
Formal evaluation covers all 40 LIBERO tasks for both Stage-2 Student and
official Teacher at matched K=1,2,4.
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
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
RUN_TAG=""
STAGE1_STEPS=5000
STAGE2_STEPS=10000
SAVE_INTERVAL=1000
STAGE1_MASTER_PORT=29671
STAGE2_MASTER_PORT=29672
RESUME_STAGE=""
RESUME_STEP=""
READ_ONLY_MODE=""
EVAL_EPISODES=50

while (( $# > 0 )); do
    case "$1" in
        --phase) PHASE="${2:-}"; shift 2 ;;
        --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
        --run-tag) RUN_TAG="${2:-}"; shift 2 ;;
        --stage1-steps) STAGE1_STEPS="${2:-}"; shift 2 ;;
        --stage2-steps) STAGE2_STEPS="${2:-}"; shift 2 ;;
        --steps)
            STAGE1_STEPS="${2:-}"
            STAGE2_STEPS="${2:-}"
            shift 2
            ;;
        --episodes) EVAL_EPISODES="${2:-}"; shift 2 ;;
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

if [[ "$PHASE" == check ]]; then
    [[ -z "$READ_ONLY_MODE" ]] || die "--phase check is already read-only"
    READ_ONLY_MODE=check-only
    PHASE=all
fi
case "$PHASE" in all|stage1|stage2|eval) ;; *) die "--phase must be all, stage1, stage2, eval, or check" ;; esac
if [[ -z "$OUTPUT_ROOT" && -n "$READ_ONLY_MODE" ]]; then
    OUTPUT_ROOT="$(pwd -P)/output_cosmos_stage1_stage2_eval"
fi
[[ -n "$OUTPUT_ROOT" ]] || die "--output-root is required"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "--run-tag must be one path-safe component"
positive STAGE1_STEPS "$STAGE1_STEPS"
positive STAGE2_STEPS "$STAGE2_STEPS"
positive SAVE_INTERVAL "$SAVE_INTERVAL"
positive EVAL_EPISODES "$EVAL_EPISODES"
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

if [[ -z "${WAN_STUDENT_BASE_MODEL_PATH:-}" ]]; then
    [[ -n "${CLEAN_STUDENT_BASE_MODEL_PATH:-}" ]] || \
        die "WAN_STUDENT_BASE_MODEL_PATH must be set explicitly"
    WAN_STUDENT_BASE_MODEL_PATH="$CLEAN_STUDENT_BASE_MODEL_PATH"
    printf '%s\n' \
        "WARNING: CLEAN_STUDENT_BASE_MODEL_PATH is deprecated; use WAN_STUDENT_BASE_MODEL_PATH" \
        >&2
elif [[ -n "${CLEAN_STUDENT_BASE_MODEL_PATH:-}" && \
        "$WAN_STUDENT_BASE_MODEL_PATH" != "$CLEAN_STUDENT_BASE_MODEL_PATH" ]]; then
    die "WAN_STUDENT_BASE_MODEL_PATH and deprecated CLEAN_STUDENT_BASE_MODEL_PATH disagree"
fi
require_env COSMOS_PREDICT2_REPO
require_dir WAN_STUDENT_BASE_MODEL_PATH "$WAN_STUDENT_BASE_MODEL_PATH"
require_transformer \
    WAN_STUDENT_BASE_MODEL_PATH/transformer \
    "$WAN_STUDENT_BASE_MODEL_PATH/transformer"
require_dir COSMOS_PREDICT2_REPO "$COSMOS_PREDICT2_REPO"

COSMOS_PREDICT2_REPO_COMMIT="$(
    GIT_OPTIONAL_LOCKS=0 \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_CONFIG_SYSTEM=/dev/null \
    GIT_CONFIG_NOSYSTEM=1 \
    /usr/bin/git -C "$COSMOS_PREDICT2_REPO" rev-parse HEAD
)" || die "COSMOS_PREDICT2_REPO must be a readable Git repository"
AUDITED_COSMOS_PREDICT2_REPO_COMMIT=1eb8457072b4a1adfe1f83c3076e4aa5452cbab2
[[ "$COSMOS_PREDICT2_REPO_COMMIT" == "$AUDITED_COSMOS_PREDICT2_REPO_COMMIT" ]] || \
    die "COSMOS_PREDICT2_REPO must be at audited commit $AUDITED_COSMOS_PREDICT2_REPO_COMMIT"
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
PROMPT_TABLE_BUILDER="${COSMOS_WAN_PROMPT_TABLE_BUILDER:-$SCRIPT_DIR/build_cosmos_wan_prompt_table.py}"
for executable in "$STAGE1_LAUNCHER" "$STAGE2_LAUNCHER" "$EVAL_LAUNCHER" "$LOCK_PREPARER" "$PROMPT_TABLE_BUILDER"; do
    [[ -x "$executable" ]] || die "required executable is missing: $executable"
done

DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/training_data/libero-long-lerobot}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-$DATASET_PATH/empty_emb.pt}"
S4_PROMPT_TABLE="${S4_PROMPT_TABLE:-}"
COSMOS_POLICY_PATH="${COSMOS_POLICY_PATH:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
COSMOS_PREDICT25_LOCAL_MODEL_DIR="${COSMOS_PREDICT25_LOCAL_MODEL_DIR:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World}"
require_dir DATASET_PATH "$DATASET_PATH"
[[ -f "$EMPTY_EMB_PATH" ]] || die "EMPTY_EMB_PATH is missing: $EMPTY_EMB_PATH"
if [[ -n "$S4_PROMPT_TABLE" ]]; then
    [[ -f "$S4_PROMPT_TABLE" && ! -L "$S4_PROMPT_TABLE" ]] || \
        die "S4_PROMPT_TABLE must be a plain all-40 prompt table: $S4_PROMPT_TABLE"
fi
require_dir COSMOS_POLICY_PATH "$COSMOS_POLICY_PATH"
require_dir COSMOS_PREDICT25_LOCAL_MODEL_DIR "$COSMOS_PREDICT25_LOCAL_MODEL_DIR"
if ! (
    cd "$PROJECT_ROOT"
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/wan_va:${PYTHONPATH:-}" \
        "$PYTHON_BIN" -c \
        'import sys
from distillation_flowmap.cosmos_hybrid_backend import (
    validate_cosmos_teacher_model_path,
    validate_wan_student_base_model_path,
)
validate_wan_student_base_model_path(sys.argv[1])
validate_cosmos_teacher_model_path(sys.argv[2])' \
        "$WAN_STUDENT_BASE_MODEL_PATH" "$COSMOS_POLICY_PATH"
); then
    die "hybrid Student/Teacher backend validation failed"
fi

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
MATRIX_ROOT="$RUN_ROOT/eval/student_teacher"
PROMPT_TABLE_WAS_EXPLICIT=0
if [[ -n "$S4_PROMPT_TABLE" ]]; then
    PROMPT_TABLE_WAS_EXPLICIT=1
else
    S4_PROMPT_TABLE="$RUN_ROOT/libero_wan_prompt_embeddings_all40.pt"
fi

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
HELPER_PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/wan_va${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$LOCK_PREPARER" == *.py ]]; then
    lock_command=(env "PYTHONPATH=$HELPER_PYTHONPATH" "$PYTHON_BIN" "$LOCK_PREPARER")
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
if [[ "$PROMPT_TABLE_BUILDER" == *.py ]]; then
    prompt_table_command=(
        env "PYTHONPATH=$HELPER_PYTHONPATH"
        "$PYTHON_BIN" "$PROMPT_TABLE_BUILDER"
    )
else
    prompt_table_command=("$PROMPT_TABLE_BUILDER")
fi
prompt_table_command+=(
    --wan-base-model "$WAN_STUDENT_BASE_MODEL_PATH"
    --output "$S4_PROMPT_TABLE"
)
PROMPT_TABLE_MODE=build
if (( PROMPT_TABLE_WAS_EXPLICIT )) || [[ -f "$S4_PROMPT_TABLE" ]]; then
    PROMPT_TABLE_MODE=validate
    prompt_table_command+=(--validate-only)
fi
prompt_table_plan=("${prompt_table_command[@]}" --dry-run)
eval_command=("$EVAL_LAUNCHER" run)
stage1_environment=(
    env
    "WAN_STUDENT_BASE_MODEL_PATH=$WAN_STUDENT_BASE_MODEL_PATH"
    "DATASET_PATH=$DATASET_PATH"
    "EMPTY_EMB_PATH=$EMPTY_EMB_PATH"
    "COSMOS_POLICY_PATH=$COSMOS_POLICY_PATH"
    "COSMOS_POLICY_TEACHER_LOCK=$LOCK_ROOT/teacher.lock.json"
    "COSMOS_PREDICT2_REPO=$COSMOS_PREDICT2_REPO"
    "COSMOS_PREDICT2_REPO_COMMIT=$COSMOS_PREDICT2_REPO_COMMIT"
    "COSMOS_PREDICT25_LOCAL_MODEL_DIR=$COSMOS_PREDICT25_LOCAL_MODEL_DIR"
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
)
stage2_environment=(
    env
    "COSMOS_STAGE1_ROOT=$STAGE1_CHECKPOINT"
    "STUDENT_BASE_MODEL_PATH=$STAGE1_TARGET"
    "WAN_STUDENT_BASE_MODEL_PATH=$WAN_STUDENT_BASE_MODEL_PATH"
    "RESUME_FROM_PATH=$STAGE1_CHECKPOINT"
    "PARENT_STAGE1_PATH=$STAGE1_CHECKPOINT"
    "PARENT_STAGE1_CONTRACT_IDENTITY=__DERIVED_AFTER_STAGE1__"
    "STAGE2_LINEAGE_JSON=__DERIVED_AFTER_STAGE1__"
    "RESUME_ONLINE_FROM_TARGET=1"
    "RESET_RESUME_STEP=1"
    "RESUME_OPTIMIZER_STATE=0"
    "COSMOS_PROVENANCE_LOCK_ROOT=$LOCK_ROOT"
    "VERIFY_LARGE_ARTIFACT_DIGESTS=1"
    "DATASET_PATH=$DATASET_PATH"
    "COSMOS_POLICY_PATH=$COSMOS_POLICY_PATH"
    "COSMOS_PREDICT2_REPO=$COSMOS_PREDICT2_REPO"
    "COSMOS_PREDICT2_REPO_COMMIT=$COSMOS_PREDICT2_REPO_COMMIT"
    "COSMOS_PREDICT25_LOCAL_MODEL_DIR=$COSMOS_PREDICT25_LOCAL_MODEL_DIR"
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "ALIGNED_VIDEO_OPD_INTERVAL=4"
    "OPD_AUX_INTERVAL=4"
    "OPD_DANCEOPD_ROLLOUT_STEPS=2,4"
    "OPD_DANCEOPD_ANCHOR_TEACHER_STEPS=8"
    "OPD_DANCEOPD_ENDPOINT_WEIGHT=1.0"
    "OPD_DANCEOPD_VELOCITY_WEIGHT=1.0"
    "ACTION_DOWNSAMPLE_FACTOR=4"
    "VIDEO_ACTION_BRIDGE=0"
)
eval_environment=(
    env
    "MATRIX_ROOT=$MATRIX_ROOT"
    "S4_CKPT_ROOT=$EVAL_TRANSFORMER"
    "COSMOS_POLICY_PATH=$COSMOS_POLICY_PATH"
    "COSMOS_POLICY_TEACHER_LOCK=$LOCK_ROOT/teacher.lock.json"
    "COSMOS_PREDICT2_REPO=$COSMOS_PREDICT2_REPO"
    "COSMOS_PREDICT2_REPO_COMMIT=$COSMOS_PREDICT2_REPO_COMMIT"
    "WAN_STUDENT_BASE_MODEL_PATH=$WAN_STUDENT_BASE_MODEL_PATH"
    "STUDENT_BASE_MODEL_PATH=$STAGE1_TARGET"
    "RESUME_FROM_PATH=$STAGE1_CHECKPOINT"
    "PARENT_STAGE1_PATH=$STAGE1_CHECKPOINT"
    "PARENT_STAGE1_CONTRACT_IDENTITY=__DERIVED_AFTER_STAGE1__"
    "STAGE2_LINEAGE_JSON=__DERIVED_AFTER_STAGE1__"
    "RESUME_ONLINE_FROM_TARGET=1"
    "RESET_RESUME_STEP=1"
    "RESUME_OPTIMIZER_STATE=0"
    "S4_MATRIX_ROLES=stage2_target,official_teacher"
    "S4_DATASET_PATH=$DATASET_PATH"
    "S4_EMPTY_EMBEDDING=$EMPTY_EMB_PATH"
    "S4_PROMPT_TABLE=$S4_PROMPT_TABLE"
    "S4_ALIGNMENT_VERIFIED=1"
    "S4_EPISODES_PER_TASK=$EVAL_EPISODES"
)
stage1_dry_command=("${stage1_command[@]}")
stage1_dry_command[1]=dry-run
stage1_execution=("${stage1_environment[@]}" "${stage1_command[@]}")
stage1_plan=("${stage1_environment[@]}" "${stage1_dry_command[@]}")
lock_plan=("${lock_command[@]}" --dry-run)
stage2_execution=("${stage2_environment[@]}" "${stage2_command[@]}")
stage2_plan=("${stage2_environment[@]}" "${stage2_command[@]}" --dry-run)
eval_execution=("${eval_environment[@]}" "${eval_command[@]}")
eval_plan=("${eval_environment[@]}" "$EVAL_LAUNCHER" dry-run)

replace_env_assignment() {
    local array_name="$1"
    local key="$2"
    local value="$3"
    local -n assignments="$array_name"
    local index
    for index in "${!assignments[@]}"; do
        if [[ "${assignments[$index]}" == "$key="* ]]; then
            assignments[$index]="$key=$value"
            return
        fi
    done
    die "internal environment assignment is missing: $key"
}

derive_stage1_lineage() {
    local -a derived
    mapfile -t derived < <(
        cd "$PROJECT_ROOT"
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/wan_va:${PYTHONPATH:-}" \
            "$PYTHON_BIN" - "$STAGE1_CHECKPOINT" "$STAGE1_STEPS" <<'PY'
import json
import sys
from pathlib import Path
from distillation_flowmap.cosmos_stage2_lineage import validate_stage1_parent

parent = validate_stage1_parent(Path(sys.argv[1]), expected_step=int(sys.argv[2]))
print(parent.contract_identity)
print(json.dumps(
    {
        "parent_stage1_contract_identity": parent.contract_identity,
        "parent_stage1_path": parent.canonical_path,
    },
    sort_keys=True,
    separators=(",", ":"),
))
PY
    )
    (( ${#derived[@]} == 2 )) || die "failed to derive Stage-1 lineage"
    PARENT_STAGE1_CONTRACT_IDENTITY="${derived[0]}"
    STAGE2_LINEAGE_JSON="${derived[1]}"
    local array_name
    for array_name in stage2_environment eval_environment; do
        replace_env_assignment "$array_name" \
            PARENT_STAGE1_CONTRACT_IDENTITY \
            "$PARENT_STAGE1_CONTRACT_IDENTITY"
        replace_env_assignment "$array_name" STAGE2_LINEAGE_JSON \
            "$STAGE2_LINEAGE_JSON"
    done
    stage2_execution=("${stage2_environment[@]}" "${stage2_command[@]}")
    eval_execution=("${eval_environment[@]}" "${eval_command[@]}")
}

printf 'PIPELINE_SEMANTICS=LingBotVA/Wan student init + Cosmos teacher\n'
printf 'PHASE=%s\n' "$PHASE"
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
printf 'STAGE1_CHECKPOINT=%s\n' "$STAGE1_CHECKPOINT"
printf 'STAGE2_CHECKPOINT=%s\n' "$STAGE2_CHECKPOINT"
printf 'EVAL_CHECKPOINT_ROLES=stage2_target,official_teacher\n'
printf 'EVAL_PROTOCOL=40 tasks, matched joint 1/2/4, %s episodes per role/K\n' \
    "$((10#$EVAL_EPISODES * 40))"
printf 'EVAL_TOTAL_EPISODES=%s\n' "$((10#$EVAL_EPISODES * 40 * 3 * 2))"
printf 'EVAL_EPISODES=%s\n' "$EVAL_EPISODES"
printf 'COSMOS_PREDICT2_REPO_COMMIT=%s\n' "$COSMOS_PREDICT2_REPO_COMMIT"
printf 'ALIGNED_VIDEO_OPD_INTERVAL=4\n'
printf 'OPD_AUX_INTERVAL=4\n'
printf 'OPD_DANCEOPD_ROLLOUT_STEPS=2,4\n'
printf 'OPD_DANCEOPD_ANCHOR_TEACHER_STEPS=8\n'
printf 'OPD_DANCEOPD_ENDPOINT_WEIGHT=1.0\n'
printf 'OPD_DANCEOPD_VELOCITY_WEIGHT=1.0\n'
printf 'ACTION_DOWNSAMPLE_FACTOR=4\n'
printf 'VIDEO_ACTION_BRIDGE=0\n'
printf 'PROMPT_TABLE_MODE=%s\n' "$PROMPT_TABLE_MODE"
printf 'S4_PROMPT_TABLE=%s\n' "$S4_PROMPT_TABLE"

if [[ -n "$READ_ONLY_MODE" ]]; then
    printf 'PIPELINE_MODE=%s\n' "$READ_ONLY_MODE"
    print_command STAGE1_COMMAND "${stage1_plan[@]}"
    print_command LOCK_COMMAND "${lock_plan[@]}"
    print_command STAGE2_COMMAND "${stage2_plan[@]}"
    if [[ "$PHASE" == all || "$PHASE" == eval ]]; then
        print_command PROMPT_TABLE_COMMAND "${prompt_table_plan[@]}"
        run_child "${prompt_table_plan[@]}"
    fi
    print_command EVAL_COMMAND "${eval_plan[@]}"
    exit 0
fi
print_command STAGE1_COMMAND "${stage1_execution[@]}"
print_command LOCK_COMMAND "${lock_command[@]}"
print_command STAGE2_COMMAND "${stage2_execution[@]}"
if [[ "$PHASE" == all || "$PHASE" == eval ]]; then
    print_command PROMPT_TABLE_COMMAND "${prompt_table_command[@]}"
fi
print_command EVAL_COMMAND "${eval_execution[@]}"

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

if (( PROMPT_TABLE_WAS_EXPLICIT )) && \
        [[ "$PHASE" == all || "$PHASE" == eval ]]; then
    run_child "${prompt_table_command[@]}"
fi

export PIPELINE_RUN_ROOT="$RUN_ROOT"
export PIPELINE_STAGE2_STEPS="$STAGE2_STEPS"
export ALIGNED_VIDEO_OPD_INTERVAL=4
export OPD_AUX_INTERVAL=4
export OPD_DANCEOPD_ROLLOUT_STEPS=2,4
export OPD_DANCEOPD_ANCHOR_TEACHER_STEPS=8
export OPD_DANCEOPD_ENDPOINT_WEIGHT=1.0
export OPD_DANCEOPD_VELOCITY_WEIGHT=1.0
export ACTION_DOWNSAMPLE_FACTOR=4
export VIDEO_ACTION_BRIDGE=0

if [[ "$PHASE" == all || "$PHASE" == stage1 ]]; then
    run_child "${stage1_execution[@]}"
    require_transformer Stage1-target "$STAGE1_TARGET/transformer"
fi
[[ "$PHASE" != stage1 ]] || exit 0

derive_stage1_lineage

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
    run_child "${stage2_execution[@]}"
    require_transformer Stage2-target "$EVAL_TRANSFORMER"
fi
[[ "$PHASE" != stage2 ]] || exit 0

if (( ! PROMPT_TABLE_WAS_EXPLICIT )); then
    run_child "${prompt_table_command[@]}"
fi

export MATRIX_ROOT
export S4_CKPT_ROOT="$EVAL_TRANSFORMER"
export COSMOS_POLICY_PATH
export COSMOS_POLICY_TEACHER_LOCK="$LOCK_ROOT/teacher.lock.json"
export S4_MATRIX_ROLES=stage2_target,official_teacher
export S4_DATASET_PATH="$DATASET_PATH"
export S4_EMPTY_EMBEDDING="$EMPTY_EMB_PATH"
export S4_PROMPT_TABLE
export S4_ALIGNMENT_VERIFIED=1
export S4_EPISODES_PER_TASK="$EVAL_EPISODES"
run_child "${eval_execution[@]}"
