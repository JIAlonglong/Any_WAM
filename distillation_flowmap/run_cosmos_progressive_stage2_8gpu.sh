#!/usr/bin/env bash
# Launch one fresh or explicitly resumed aligned Cosmos progressive stage.
set -euo pipefail


die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}


require_dir() {
    local label="$1"
    local path="$2"
    [[ -d "$path" ]] || die "$label is not a directory: $path"
}


require_file() {
    local label="$1"
    local path="$2"
    [[ -f "$path" ]] || die "$label is missing: $path"
}


require_transformer() {
    local label="$1"
    local root="$2"
    require_dir "$label" "$root"
    require_file "$label/config.json" "$root/config.json"
    require_file "$label/diffusion_pytorch_model.safetensors" \
        "$root/diffusion_pytorch_model.safetensors"
}


validate_port() {
    local value="$1"
    [[ "$value" =~ ^[0-9]+$ ]] || die "invalid master port: $value"
    (( 10#$value >= 1 && 10#$value <= 65535 )) || die "invalid master port: $value"
}


validate_devices() {
    local label="$1"
    local raw="$2"
    local ordinal
    local -a ordinals
    local -A seen=()

    IFS=',' read -r -a ordinals <<< "$raw"
    (( ${#ordinals[@]} == 8 )) || die \
        "$label must contain exactly 8 comma-separated GPU ordinals: $raw"
    for ordinal in "${ordinals[@]}"; do
        [[ "$ordinal" =~ ^[0-9]+$ ]] || die \
            "$label contains a non-numeric GPU ordinal: $ordinal"
        [[ -z "${seen[$ordinal]:-}" ]] || die \
            "$label contains duplicate GPU ordinal: $ordinal"
        seen["$ordinal"]=1
    done
}


print_assignment() {
    printf '%s=%s\n' "$1" "$2"
}


PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ -n "${COSMOS_STAGE1_ROOT:-}" ]] || die \
    "COSMOS_STAGE1_ROOT must be explicitly set"
[[ -n "${STUDENT_BASE_MODEL_PATH:-}" ]] || die \
    "STUDENT_BASE_MODEL_PATH must be explicitly set"
STAGE1_ROOT="$COSMOS_STAGE1_ROOT"
STUDENT_BASE_MODEL_PATH="$STUDENT_BASE_MODEL_PATH"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_cosmos_independent_dance_4way_8gpu_20260721}"
DATASET_PATH="${DATASET_PATH:-/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot}"
COSMOS_POLICY_PATH="${COSMOS_POLICY_PATH:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
COSMOS_WORKER_ENV_ROOT="${COSMOS_WORKER_ENV_ROOT:-/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310}"
COSMOS_POLICY_PYTHON="${COSMOS_POLICY_PYTHON:-${COSMOS_WORKER_ENV_ROOT}/bin/python}"
COSMOS_PREDICT2_REPO="${COSMOS_PREDICT2_REPO:-/kpfs-intern/jialongliu/projects/cosmos-predict2.5}"
COSMOS_PREDICT25_LOCAL_MODEL_DIR="${COSMOS_PREDICT25_LOCAL_MODEL_DIR:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World}"
COSMOS_WORKER_SITE_PACKAGES="${COSMOS_WORKER_SITE_PACKAGES:-${COSMOS_WORKER_ENV_ROOT}/lib/python3.10/site-packages}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES="${COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
PREFLIGHT_BIN="${PREFLIGHT_BIN:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"

stage="${1:-}"
if (( $# > 0 )); then
    shift
fi

dry_run=0
resume_step=""
master_port=""
while (( $# > 0 )); do
    case "$1" in
        --dry-run)
            dry_run=1
            shift
            ;;
        --master-port)
            (( $# >= 2 )) || die "--master-port requires a value"
            master_port="$2"
            shift 2
            ;;
        --resume-step)
            (( $# >= 2 )) || die "--resume-step requires a value"
            resume_step="$2"
            shift 2
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

train_seed=""
case "$stage" in
    s1)
        progressive_stage=s1
        max_steps=3000; default_port=29663
        action_endpoint_weight="${OPD_DANCEOPD_ACTION_ENDPOINT_WEIGHT:-1.0}"
        opd_aux_action="${OPD_AUX_ACTION:-0}"
        cosmos_use_teacher_action_anchor="${COSMOS_USE_TEACHER_ACTION_ANCHOR:-1}"
        opd_joint_action_rollout="${OPD_JOINT_ACTION_ROLLOUT:-1}" ;;
    s2)
        progressive_stage=s2
        max_steps=3000; default_port=29662
        action_endpoint_weight="${OPD_DANCEOPD_ACTION_ENDPOINT_WEIGHT:-1.0}"
        opd_aux_action="${OPD_AUX_ACTION:-0}"
        cosmos_use_teacher_action_anchor="${COSMOS_USE_TEACHER_ACTION_ANCHOR:-1}"
        opd_joint_action_rollout="${OPD_JOINT_ACTION_ROLLOUT:-1}" ;;
    s4)
        progressive_stage=s4
        max_steps=5000; default_port=29661
        action_endpoint_weight="${OPD_DANCEOPD_ACTION_ENDPOINT_WEIGHT:-1.0}"
        opd_aux_action="${OPD_AUX_ACTION:-0}"
        cosmos_use_teacher_action_anchor="${COSMOS_USE_TEACHER_ACTION_ANCHOR:-1}"
        opd_joint_action_rollout="${OPD_JOINT_ACTION_ROLLOUT:-1}" ;;
    universal)
        progressive_stage=universal
        max_steps=5000; default_port=29664
        action_endpoint_weight="${OPD_DANCEOPD_ACTION_ENDPOINT_WEIGHT:-1.0}"
        opd_aux_action="${OPD_AUX_ACTION:-0}"
        cosmos_use_teacher_action_anchor="${COSMOS_USE_TEACHER_ACTION_ANCHOR:-1}"
        opd_joint_action_rollout="${OPD_JOINT_ACTION_ROLLOUT:-1}" ;;
    universal-video)
        progressive_stage=universal
        max_steps=5000; default_port=29665
        action_endpoint_weight=0.0
        opd_aux_action=0
        cosmos_use_teacher_action_anchor=1
        opd_joint_action_rollout=1
        train_seed=42 ;;
    universal-video-action)
        progressive_stage=universal
        max_steps=5000; default_port=29666
        action_endpoint_weight=1.0
        opd_aux_action=0
        cosmos_use_teacher_action_anchor=1
        opd_joint_action_rollout=1
        train_seed=42 ;;
    *)
        die "stage must be one of: s1, s2, s4, universal, universal-video, universal-video-action" ;;
esac

master_port="${master_port:-$default_port}"
validate_port "$master_port"
MASTER_PORT="$master_port"
output_dir="${OUTPUT_DIR:-$OUTPUT_ROOT/$stage}"

validate_devices CUDA_VISIBLE_DEVICES "$CUDA_VISIBLE_DEVICES"
validate_devices COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES \
    "$COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES"
[[ "$COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES" == "$CUDA_VISIBLE_DEVICES" ]] || die \
    "worker GPU list must equal CUDA_VISIBLE_DEVICES"

require_dir DATASET_PATH "$DATASET_PATH"
require_file empty_emb.pt "$DATASET_PATH/empty_emb.pt"
require_dir COSMOS_POLICY_PATH "$COSMOS_POLICY_PATH"
require_dir COSMOS_PREDICT2_REPO "$COSMOS_PREDICT2_REPO"
require_dir COSMOS_PREDICT25_LOCAL_MODEL_DIR "$COSMOS_PREDICT25_LOCAL_MODEL_DIR"
[[ -x "$COSMOS_POLICY_PYTHON" ]] || die \
    "COSMOS_POLICY_PYTHON is not executable: $COSMOS_POLICY_PYTHON"
if [[ "$(basename "$STUDENT_BASE_MODEL_PATH")" == "transformer" ]]; then
    require_transformer STUDENT_BASE_MODEL_PATH "$STUDENT_BASE_MODEL_PATH"
else
    require_transformer STUDENT_BASE_MODEL_PATH \
        "$STUDENT_BASE_MODEL_PATH/transformer"
fi
expected_student_base="$STAGE1_ROOT/target_student"
[[ "$(cd "$STUDENT_BASE_MODEL_PATH" && pwd -P)" == \
   "$(cd "$expected_student_base" && pwd -P)" ]] || die \
    "STUDENT_BASE_MODEL_PATH must identify validated Stage-1 target_student"

if [[ -n "$resume_step" ]]; then
    [[ "$resume_step" =~ ^[1-9][0-9]*$ ]] && (( resume_step < max_steps )) || die \
        "resume step must be in [1, $((max_steps - 1))]"
    resume_from_path="$output_dir/checkpoints/step_$resume_step"
    resume_online_from_target=0
    reset_resume_step=0
    resume_optimizer_state=1
else
    resume_from_path="$STAGE1_ROOT"
    resume_online_from_target=1
    reset_resume_step=1
    resume_optimizer_state=0
fi

lineage_code='import json, os, sys
from pathlib import Path
from distillation_flowmap.cosmos_stage2_lineage import (
    validate_stage1_parent,
    validate_stage2_path_isolation,
    validate_stage2_resume,
)
stage1 = Path(os.environ["COSMOS_STAGE1_ROOT"])
output = Path(os.environ["OUTPUT_DIR"])
resume_raw = os.environ.get("STAGE2_RESUME_CHECKPOINT", "")
resume = Path(resume_raw) if resume_raw else None
parent = validate_stage1_parent(stage1, expected_step=5000)
validate_stage2_path_isolation(
    stage1_root=stage1,
    output_dir=output,
    resume_checkpoint=resume,
)
if resume is not None:
    validate_stage2_resume(
        resume,
        arm_root=output,
        expected_step=int(os.environ["STAGE2_RESUME_STEP"]),
        expected_parent=parent,
    )
payload = {
    "parent_stage1_path": parent.canonical_path,
    "parent_stage1_contract_identity": parent.contract_identity,
}
print("PARENT_STAGE1_PATH=" + parent.canonical_path)
print("PARENT_STAGE1_CONTRACT_IDENTITY=" + parent.contract_identity)
print("STAGE2_LINEAGE_JSON=" + json.dumps(payload, sort_keys=True, separators=(",", ":")))'
lineage_output="$(
    cd "$PROJECT_ROOT"
    env \
        COSMOS_STAGE1_ROOT="$STAGE1_ROOT" \
        OUTPUT_DIR="$output_dir" \
        STAGE2_RESUME_CHECKPOINT="${resume_step:+$resume_from_path}" \
        STAGE2_RESUME_STEP="${resume_step:-0}" \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/wan_va:$PROJECT_ROOT/distillation_flowmap:${PYTHONPATH:-}" \
        "$PREFLIGHT_BIN" -c "$lineage_code"
)" || die "Stage-2 lineage preflight failed"
while IFS='=' read -r key value; do
    case "$key" in
        PARENT_STAGE1_PATH|PARENT_STAGE1_CONTRACT_IDENTITY|STAGE2_LINEAGE_JSON)
            printf -v "$key" '%s' "$value"
            export "$key"
            ;;
        *) die "unexpected lineage preflight output: $key" ;;
    esac
done <<< "$lineage_output"
if [[ -z "$resume_step" ]]; then
    [[ ! -e "$output_dir" && ! -L "$output_dir" ]] || die \
        "Refusing fresh run with existing OUTPUT_DIR: $output_dir"
fi

COSMOS_POLICY_EXTRA_PYTHONPATH="${COSMOS_POLICY_EXTRA_PYTHONPATH:-$COSMOS_PREDICT2_REPO/packages/cosmos-cuda:$COSMOS_PREDICT2_REPO/packages/cosmos-oss}"
COSMOS_WORKER_CUDA_LIBRARY_PATH="${COSMOS_WORKER_CUDA_LIBRARY_PATH:-${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cublas/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cuda_cupti/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cuda_nvrtc/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cuda_runtime/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cudnn/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cufft/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cufile/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/curand/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cusolver/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cusparse/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cusparselt/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/nccl/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/nvjitlink/lib}"

export CONFIG_FILE=distillation_flowmap.config_libero_cosmos_policy_stage2_progressive
export COSMOS_PROGRESSIVE_STAGE="$progressive_stage"
export COSMOS_PROGRESSIVE_OUTPUT_ROOT="$OUTPUT_ROOT"
export OUTPUT_DIR="$output_dir"
export MAX_TRAIN_STEPS="$max_steps"
export SAVE_INTERVAL=1000
export RESUME_FROM_PATH="$resume_from_path"
export RESUME_ONLINE_FROM_TARGET="$resume_online_from_target"
export RESET_RESUME_STEP="$reset_resume_step"
export RESUME_OPTIMIZER_STATE="$resume_optimizer_state"
export USE_FSDP1=1
export GRADIENT_CHECKPOINTING=1
export OPD_AUX_GRADIENT_CHECKPOINTING=1
export OPD_SERIAL_STUDENT_CFG=1
export OPD_AUX_STANDALONE_STEP=1
export OPD_COSMOS_SPATIAL_CROP_SIZE=28
export OPD_DANCEOPD_ACTION_ENDPOINT_WEIGHT="$action_endpoint_weight"
export OPD_AUX_ACTION="$opd_aux_action"
export COSMOS_USE_TEACHER_ACTION_ANCHOR="$cosmos_use_teacher_action_anchor"
export OPD_JOINT_ACTION_ROLLOUT="$opd_joint_action_rollout"
if [[ -n "$train_seed" ]]; then
    export TRAIN_SEED="$train_seed"
fi
export ENABLE_WANDB=0
export WANDB_MODE=offline
export ATTN_MODE="${ATTN_MODE:-flex}"
export COSMOS_POLICY_INFERENCE_MODE="${COSMOS_POLICY_INFERENCE_MODE:-subprocess}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/hf_cache}"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES
export COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES
export COSMOS_POLICY_PATH
export STUDENT_BASE_MODEL_PATH
export COSMOS_POLICY_PYTHON
export COSMOS_PREDICT2_REPO
export COSMOS_PREDICT25_LOCAL_MODEL_DIR
export COSMOS_POLICY_EXTRA_PYTHONPATH
export COSMOS_WORKER_CUDA_LIBRARY_PATH
export LD_LIBRARY_PATH="${COSMOS_WORKER_CUDA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

train_cmd=(
    "$TORCHRUN_BIN"
    --nproc_per_node=8
    "--master_port=$master_port"
    distillation_flowmap/train.py
    --teacher-model-path "$COSMOS_POLICY_PATH"
    --dataset-path "$DATASET_PATH"
    --output-dir "$output_dir"
    --resume-from-path "$resume_from_path"
    --gradient-accumulation-steps 1
)

for key in \
    COSMOS_PROGRESSIVE_STAGE \
    MASTER_PORT \
    OUTPUT_DIR \
    MAX_TRAIN_STEPS \
    RESUME_FROM_PATH \
    RESUME_ONLINE_FROM_TARGET \
    RESET_RESUME_STEP \
    RESUME_OPTIMIZER_STATE \
    SAVE_INTERVAL \
    OPD_DANCEOPD_ACTION_ENDPOINT_WEIGHT \
    OPD_AUX_ACTION \
    COSMOS_USE_TEACHER_ACTION_ANCHOR \
    OPD_JOINT_ACTION_ROLLOUT \
    STUDENT_BASE_MODEL_PATH \
    PARENT_STAGE1_PATH \
    PARENT_STAGE1_CONTRACT_IDENTITY \
    STAGE2_LINEAGE_JSON \
    CUDA_VISIBLE_DEVICES \
    COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES; do
    print_assignment "$key" "${!key}"
done
if [[ -n "$train_seed" ]]; then
    print_assignment TRAIN_SEED "$TRAIN_SEED"
fi
printf 'command='
printf '%q ' "${train_cmd[@]}"
printf '\n'

if (( dry_run )); then
    exit 0
fi

output_parent="$(dirname "$output_dir")"
mkdir -p "$output_parent"
if [[ -z "$resume_step" ]]; then
    mkdir "$output_dir" || die "failed to atomically claim OUTPUT_DIR: $output_dir"
fi
cd "$PROJECT_ROOT"
exec "${train_cmd[@]}"
