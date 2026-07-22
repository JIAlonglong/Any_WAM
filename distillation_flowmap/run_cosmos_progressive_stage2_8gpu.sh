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
STAGE1_ROOT="${COSMOS_STAGE1_ROOT:-/kpfs-intern/jialongliu/models/modelscope/JIAlonglong/any-wam-cosmos-checkpoints/raw_stage1_5000}"
STUDENT_BASE_MODEL_PATH="${STUDENT_BASE_MODEL_PATH:-${STAGE1_ROOT}/target_student}"
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

case "$stage" in
    s1)
        max_steps=3000; default_port=29663 ;;
    s2)
        max_steps=3000; default_port=29662 ;;
    s4)
        max_steps=5000; default_port=29661 ;;
    universal)
        max_steps=5000; default_port=29664 ;;
    *)
        die "stage must be one of: s1, s2, s4, universal" ;;
esac

master_port="${master_port:-$default_port}"
validate_port "$master_port"
output_dir="$OUTPUT_ROOT/$stage"

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

if [[ -n "$resume_step" ]]; then
    [[ "$resume_step" =~ ^[1-9][0-9]*$ ]] && (( resume_step < max_steps )) || die \
        "resume step must be in [1, $((max_steps - 1))]"
    resume_from_path="$output_dir/checkpoints/step_$resume_step"
    require_transformer "resume online_student/transformer" \
        "$resume_from_path/online_student/transformer"
    require_file resume_optimizer "$resume_from_path/optimizer.pt"
    resume_online_from_target=0
    reset_resume_step=0
    resume_optimizer_state=1
else
    if [[ -e "$output_dir/checkpoints" ]]; then
        [[ -d "$output_dir/checkpoints" ]] || die \
            "fresh checkpoint path is not a directory: $output_dir/checkpoints"
        # A failed run can create this directory before its first checkpoint.
        # Preserve its logs and allow a safe fresh restart only while it is empty.
        checkpoint_entry="$(find "$output_dir/checkpoints" -mindepth 1 -maxdepth 1 -print -quit)" || die \
            "cannot inspect fresh checkpoint path: $output_dir/checkpoints"
        [[ -z "$checkpoint_entry" ]] || die \
            "Refusing fresh run with existing checkpoints: $output_dir/checkpoints"
    fi
    resume_from_path="$STAGE1_ROOT"
    resume_online_from_target=1
    reset_resume_step=1
    resume_optimizer_state=0
    require_transformer "Stage-1 online_student/transformer" \
        "$resume_from_path/online_student/transformer"
    require_transformer "Stage-1 target_student/transformer" \
        "$resume_from_path/target_student/transformer"
fi

COSMOS_POLICY_EXTRA_PYTHONPATH="${COSMOS_POLICY_EXTRA_PYTHONPATH:-$COSMOS_PREDICT2_REPO/packages/cosmos-cuda:$COSMOS_PREDICT2_REPO/packages/cosmos-oss}"
COSMOS_WORKER_CUDA_LIBRARY_PATH="${COSMOS_WORKER_CUDA_LIBRARY_PATH:-${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cublas/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cuda_cupti/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cuda_nvrtc/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cuda_runtime/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cudnn/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cufft/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cufile/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/curand/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cusolver/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cusparse/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cusparselt/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/nccl/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/nvjitlink/lib}"

export CONFIG_FILE=distillation_flowmap.config_libero_cosmos_policy_stage2_progressive
export COSMOS_PROGRESSIVE_STAGE="$stage"
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
    MAX_TRAIN_STEPS \
    RESUME_FROM_PATH \
    RESUME_ONLINE_FROM_TARGET \
    RESET_RESUME_STEP \
    RESUME_OPTIMIZER_STATE \
    SAVE_INTERVAL \
    STUDENT_BASE_MODEL_PATH \
    CUDA_VISIBLE_DEVICES \
    COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES; do
    print_assignment "$key" "${!key}"
done
printf 'command='
printf '%q ' "${train_cmd[@]}"
printf '\n'

if (( dry_run )); then
    exit 0
fi

mkdir -p "$output_dir"
cd "$PROJECT_ROOT"
exec "${train_cmd[@]}"
