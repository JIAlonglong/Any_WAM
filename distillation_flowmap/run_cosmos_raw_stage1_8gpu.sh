#!/usr/bin/env bash
# Launch corrected Cosmos raw Stage-1 from the clean WanVA student base.
set -euo pipefail


usage() {
    cat <<'EOF'
Usage: bash distillation_flowmap/run_cosmos_raw_stage1_8gpu.sh run|dry-run [options]

Options:
  --steps N
  --save-interval N
  --master-port PORT
  --output-dir PATH
  --run-tag TAG
  --resume-step N
EOF
}


die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}


positive() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer"
}


require_dir() {
    [[ -d "$2" ]] || die "$1 is not a directory: $2"
}


require_file() {
    [[ -f "$2" ]] || die "$1 is missing: $2"
}


require_plain_dir() {
    [[ ! -L "$2" ]] || die "$1 must not be a symlink: $2"
    require_dir "$1" "$2"
}


require_plain_file() {
    [[ ! -L "$2" ]] || die "$1 must not be a symlink: $2"
    require_file "$1" "$2"
}


require_transformer() {
    local label="$1"
    local root="$2"
    require_plain_dir "$label" "$root"
    require_plain_file "$label/config.json" "$root/config.json"
    if [[ -f "$root/diffusion_pytorch_model.safetensors" ]]; then
        require_plain_file "$label/diffusion_pytorch_model.safetensors" \
            "$root/diffusion_pytorch_model.safetensors"
        return
    fi

    local index_path="$root/diffusion_pytorch_model.safetensors.index.json"
    require_plain_file "$label/diffusion_pytorch_model.safetensors.index.json" \
        "$index_path"
    local shard_validation_code='import json, os, sys
from pathlib import Path

index = Path(sys.argv[1])
payload = json.loads(index.read_text(encoding="utf-8"))
weight_map = payload.get("weight_map")
if not isinstance(weight_map, dict) or not weight_map:
    raise ValueError(f"{index} must contain a nonempty weight_map")
root = index.parent
for raw_name in set(weight_map.values()):
    if (
        not isinstance(raw_name, str)
        or raw_name in ("", ".", "..")
        or "/" in raw_name
        or os.path.isabs(raw_name)
        or Path(raw_name).name != raw_name
    ):
        raise ValueError(f"invalid shard name in {index}: {raw_name!r}")
    shard = root / raw_name
    if shard.is_symlink():
        raise ValueError(f"declared transformer shard must not be a symlink: {shard}")
    if not shard.is_file():
        raise FileNotFoundError(f"missing declared transformer shard: {shard}")'
    if ! PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -c \
        "$shard_validation_code" "$index_path"; then
        die "$label has an invalid sharded weight layout"
    fi
}


validate_port() {
    local value="$1"
    [[ "$value" =~ ^[0-9]+$ ]] || die "invalid master port: $value"
    (( 10#$value >= 1 && 10#$value <= 65535 )) || die \
        "invalid master port: $value"
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
        [[ "$ordinal" =~ ^(0|[1-9][0-9]*)$ ]] || die \
            "$label contains a non-canonical GPU ordinal: $ordinal"
        [[ -z "${seen[$ordinal]:-}" ]] || die \
            "$label contains duplicate GPU ordinal: $ordinal"
        seen["$ordinal"]=1
    done
}


mode="${1:-}"
if (( $# > 0 )); then
    shift
fi
case "$mode" in
    run) dry_run=0 ;;
    dry-run) dry_run=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "mode must be run or dry-run" ;;
esac

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
MASTER_PORT="${MASTER_PORT:-29671}"
OUTPUT_OVERRIDE="${STAGE1_OUTPUT:-}"
RUN_TAG="${RUN_TAG:-}"
RESUME_STEP=""

while (( $# > 0 )); do
    case "$1" in
        --steps)
            (( $# >= 2 )) || die "$1 requires a value"
            MAX_TRAIN_STEPS="$2"
            shift 2
            ;;
        --save-interval)
            (( $# >= 2 )) || die "$1 requires a value"
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --master-port)
            (( $# >= 2 )) || die "$1 requires a value"
            MASTER_PORT="$2"
            shift 2
            ;;
        --output-dir)
            (( $# >= 2 )) || die "$1 requires a value"
            OUTPUT_OVERRIDE="$2"
            shift 2
            ;;
        --run-tag)
            (( $# >= 2 )) || die "$1 requires a value"
            RUN_TAG="$2"
            shift 2
            ;;
        --resume-step)
            (( $# >= 2 )) || die "$1 requires a value"
            RESUME_STEP="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

positive MAX_TRAIN_STEPS "$MAX_TRAIN_STEPS"
positive SAVE_INTERVAL "$SAVE_INTERVAL"
validate_port "$MASTER_PORT"
if [[ -n "$RESUME_STEP" ]]; then
    positive RESUME_STEP "$RESUME_STEP"
    (( 10#$RESUME_STEP < 10#$MAX_TRAIN_STEPS )) || die \
        "RESUME_STEP must be less than MAX_TRAIN_STEPS"
fi
if [[ -n "$RUN_TAG" && ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    die "invalid run tag: $RUN_TAG"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
GIT_COMMON="$(git -C "$PROJECT_ROOT" rev-parse --git-common-dir)"
[[ "$GIT_COMMON" = /* ]] || GIT_COMMON="$PROJECT_ROOT/$GIT_COMMON"
SHARED_ROOT="$(cd "$(dirname "$GIT_COMMON")" && pwd -P)"

PYTHON_BIN="${PYTHON_BIN:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"
PREFLIGHT_BIN="${PREFLIGHT_BIN:-$PYTHON_BIN}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/torchrun}"
WAN_STUDENT_BASE_MODEL_PATH="${WAN_STUDENT_BASE_MODEL_PATH:-}"
LEGACY_CLEAN_STUDENT_BASE_MODEL_PATH="${CLEAN_STUDENT_BASE_MODEL_PATH:-}"
if [[ -z "$WAN_STUDENT_BASE_MODEL_PATH" ]]; then
    [[ -n "$LEGACY_CLEAN_STUDENT_BASE_MODEL_PATH" ]] || die \
        "WAN_STUDENT_BASE_MODEL_PATH must be set explicitly"
    WAN_STUDENT_BASE_MODEL_PATH="$LEGACY_CLEAN_STUDENT_BASE_MODEL_PATH"
    printf '%s\n' \
        "WARNING: CLEAN_STUDENT_BASE_MODEL_PATH is deprecated; use WAN_STUDENT_BASE_MODEL_PATH" \
        >&2
elif [[ -n "$LEGACY_CLEAN_STUDENT_BASE_MODEL_PATH" ]]; then
    [[ "$WAN_STUDENT_BASE_MODEL_PATH" == "$LEGACY_CLEAN_STUDENT_BASE_MODEL_PATH" ]] || die \
        "WAN_STUDENT_BASE_MODEL_PATH and deprecated CLEAN_STUDENT_BASE_MODEL_PATH disagree"
fi
CLEAN_STUDENT_BASE_MODEL_PATH="$WAN_STUDENT_BASE_MODEL_PATH"
DATASET_PATH="${DATASET_PATH:-${SHARED_ROOT}/training_data/libero-long-lerobot}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-${DATASET_PATH}/empty_emb.pt}"
COSMOS_POLICY_PATH="${COSMOS_POLICY_PATH:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
COSMOS_WORKER_ENV_ROOT="${COSMOS_WORKER_ENV_ROOT:-/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310}"
COSMOS_POLICY_PYTHON="${COSMOS_POLICY_PYTHON:-${COSMOS_WORKER_ENV_ROOT}/bin/python}"
COSMOS_PREDICT2_REPO="${COSMOS_PREDICT2_REPO:-/kpfs-intern/jialongliu/projects/cosmos-predict2.5}"
COSMOS_PREDICT25_LOCAL_MODEL_DIR="${COSMOS_PREDICT25_LOCAL_MODEL_DIR:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World}"
COSMOS_WORKER_SITE_PACKAGES="${COSMOS_WORKER_SITE_PACKAGES:-${COSMOS_WORKER_ENV_ROOT}/lib/python3.10/site-packages}"
COSMOS_POLICY_EXTRA_PYTHONPATH="$COSMOS_PREDICT2_REPO/packages/cosmos-cuda:$COSMOS_PREDICT2_REPO/packages/cosmos-oss"
COSMOS_WORKER_CUDA_LIBRARY_PATH="${COSMOS_WORKER_CUDA_LIBRARY_PATH:-${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cublas/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cuda_cupti/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cuda_nvrtc/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cuda_runtime/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cudnn/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cufft/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cufile/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/curand/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cusolver/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cusparse/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/nccl/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/nvjitlink/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/nvtx/lib}"
WORKER_LD_LIBRARY_PATH="$COSMOS_WORKER_CUDA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
TRAIN_SEED="${TRAIN_SEED:-42}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES="${COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"

DEFAULT_OUTPUT="${SHARED_ROOT}/distillation_flowmap/output_libero_cosmos_raw_stage1_v2_steps${MAX_TRAIN_STEPS}"
[[ -z "$RUN_TAG" ]] || DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_${RUN_TAG}"
OUTPUT_DIR="${OUTPUT_OVERRIDE:-$DEFAULT_OUTPUT}"

positive TRAIN_SEED "$TRAIN_SEED"
validate_devices CUDA_VISIBLE_DEVICES "$CUDA_VISIBLE_DEVICES"
validate_devices COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES \
    "$COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES"
[[ "$COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES" == "$CUDA_VISIBLE_DEVICES" ]] || die \
    "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES must equal CUDA_VISIBLE_DEVICES"

[[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
[[ -x "$PREFLIGHT_BIN" ]] || die "PREFLIGHT_BIN is not executable: $PREFLIGHT_BIN"
[[ -x "$TORCHRUN_BIN" ]] || die "TORCHRUN_BIN is not executable: $TORCHRUN_BIN"
require_dir CLEAN_STUDENT_BASE_MODEL_PATH "$WAN_STUDENT_BASE_MODEL_PATH"
require_dir COSMOS_POLICY_PATH "$COSMOS_POLICY_PATH"
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
require_transformer "CLEAN_STUDENT_BASE_MODEL_PATH/transformer" \
    "$CLEAN_STUDENT_BASE_MODEL_PATH/transformer"
require_dir DATASET_PATH "$DATASET_PATH"
require_file empty_emb.pt "$EMPTY_EMB_PATH"
[[ -x "$COSMOS_POLICY_PYTHON" ]] || die \
    "COSMOS_POLICY_PYTHON is not executable: $COSMOS_POLICY_PYTHON"
require_dir COSMOS_PREDICT2_REPO "$COSMOS_PREDICT2_REPO"
require_dir COSMOS_PREDICT25_LOCAL_MODEL_DIR "$COSMOS_PREDICT25_LOCAL_MODEL_DIR"
require_dir "Cosmos CUDA package" "$COSMOS_PREDICT2_REPO/packages/cosmos-cuda"
require_dir "Cosmos OSS package" "$COSMOS_PREDICT2_REPO/packages/cosmos-oss"
require_dir COSMOS_WORKER_SITE_PACKAGES "$COSMOS_WORKER_SITE_PACKAGES"
IFS=':' read -r -a worker_cuda_library_dirs <<< "$COSMOS_WORKER_CUDA_LIBRARY_PATH"
(( ${#worker_cuda_library_dirs[@]} > 0 )) || die \
    "COSMOS_WORKER_CUDA_LIBRARY_PATH must declare at least one directory"
for worker_cuda_library_dir in "${worker_cuda_library_dirs[@]}"; do
    require_dir "COSMOS_WORKER_CUDA_LIBRARY_PATH entry" \
        "$worker_cuda_library_dir"
done
cosmos_git_status="$(
    GIT_OPTIONAL_LOCKS=0 git -C "$COSMOS_PREDICT2_REPO" status \
        --porcelain=v1 --untracked-files=all
)" || die "COSMOS_PREDICT2_REPO must be a readable Git repository"
[[ -z "$cosmos_git_status" ]] || die "COSMOS_PREDICT2_REPO must be clean"

output_safety_code='import os, sys
from pathlib import Path

mode = sys.argv[1]
raw_output = Path(sys.argv[2])
if os.path.lexists(raw_output) and raw_output.is_symlink():
    raise ValueError(f"OUTPUT_DIR must not be a symlink: {raw_output}")
output = raw_output.resolve(strict=(mode == "resume"))
if mode == "fresh":
    for raw_protected in sys.argv[3:]:
        protected = Path(raw_protected).resolve(strict=True)
        if output == protected or protected in output.parents:
            raise ValueError(
                f"OUTPUT_DIR {output} equals or is nested beneath protected input {protected}"
            )
elif not output.is_dir():
    raise ValueError(f"resume OUTPUT_DIR is not a directory: {output}")
print(output)'
output_mode=fresh
[[ -z "$RESUME_STEP" ]] || output_mode=resume
if ! OUTPUT_DIR="$(
    PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -c "$output_safety_code" \
        "$output_mode" "$OUTPUT_DIR" \
        "$CLEAN_STUDENT_BASE_MODEL_PATH" \
        "$DATASET_PATH" \
        "$COSMOS_POLICY_PATH" \
        "$COSMOS_PREDICT2_REPO" \
        "$COSMOS_PREDICT25_LOCAL_MODEL_DIR" \
        "$COSMOS_WORKER_ENV_ROOT"
)"; then
    die "OUTPUT_DIR canonical safety validation failed"
fi

RESUME_FROM_PATH=""
RESUME_ONLINE_FROM_TARGET=0
RESET_RESUME_STEP=0
RESUME_OPTIMIZER_STATE=0
if [[ -n "$RESUME_STEP" ]]; then
    require_plain_dir "resume OUTPUT_DIR" "$OUTPUT_DIR"
    require_plain_dir "resume checkpoints" "$OUTPUT_DIR/checkpoints"
    RESUME_FROM_PATH="$OUTPUT_DIR/checkpoints/step_$RESUME_STEP"
    require_plain_dir "resume checkpoint" "$RESUME_FROM_PATH"
    for variant in online_student target_student; do
        require_plain_dir "resume $variant" "$RESUME_FROM_PATH/$variant"
        require_transformer "resume $variant/transformer" \
            "$RESUME_FROM_PATH/$variant/transformer"
    done
    require_plain_file optimizer.pt "$RESUME_FROM_PATH/optimizer.pt"
    require_plain_file lr_scheduler.pt "$RESUME_FROM_PATH/lr_scheduler.pt"

    resume_path_code='import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
checkpoint = Path(sys.argv[2]).resolve(strict=True)
expected = (root / "checkpoints" / f"step_{sys.argv[3]}").resolve(strict=True)
if checkpoint != expected or root not in checkpoint.parents:
    raise ValueError(
        f"resume checkpoint {checkpoint} is not the exact checkpoint inside {root}"
    )'
    if ! PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -c "$resume_path_code" \
        "$OUTPUT_DIR" "$RESUME_FROM_PATH" "$RESUME_STEP"; then
        die "resume checkpoint canonical containment validation failed"
    fi

    resume_validation_code='import json, sys
from pathlib import Path
from distillation_flowmap.cosmos_training_contract import validate_contract_metadata

expected_step = int(sys.argv[3])
for variant, raw_path in zip(("online_student", "target_student"), sys.argv[1:3]):
    path = Path(raw_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_contract_metadata(payload, required_stage="raw_stage1")
    actual_step = payload.get("checkpoint_step")
    if type(actual_step) is not int or actual_step != expected_step:
        raise ValueError(
            f"{variant} checkpoint_step must be exactly {expected_step}, got {actual_step!r}"
        )
print(f"validated corrected raw Stage-1 checkpoint step {expected_step}")'
    if ! (
        cd "$PROJECT_ROOT"
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/wan_va:$PROJECT_ROOT/distillation_flowmap:${PYTHONPATH:-}" \
            "$PYTHON_BIN" -c "$resume_validation_code" \
            "$RESUME_FROM_PATH/online_student/transformer/config.json" \
            "$RESUME_FROM_PATH/target_student/transformer/config.json" \
            "$RESUME_STEP"
    ); then
        die "resume checkpoint metadata validation failed"
    fi
    RESUME_ONLINE_FROM_TARGET=0
    RESET_RESUME_STEP=0
    RESUME_OPTIMIZER_STATE=1
else
    [[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || die \
        "Refusing fresh run with existing OUTPUT_DIR: $OUTPUT_DIR"
fi

CONFIG_FILE="distillation_flowmap.config_libero_cosmos_policy_stage1"
launch_env=(
    "CONFIG_FILE=$CONFIG_FILE"
    "OUTPUT_DIR=$OUTPUT_DIR"
    "STUDENT_BACKEND=wan_flowmap"
    "WAN_STUDENT_BASE_MODEL_PATH=$WAN_STUDENT_BASE_MODEL_PATH"
    "STUDENT_BASE_MODEL_PATH=$CLEAN_STUDENT_BASE_MODEL_PATH"
    "RESUME_FROM_PATH=$RESUME_FROM_PATH"
    "RESUME_ONLINE_FROM_TARGET=$RESUME_ONLINE_FROM_TARGET"
    "RESET_RESUME_STEP=$RESET_RESUME_STEP"
    "RESUME_OPTIMIZER_STATE=$RESUME_OPTIMIZER_STATE"
    "MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS"
    "SAVE_INTERVAL=$SAVE_INTERVAL"
    "TRAIN_SEED=$TRAIN_SEED"
    "COSMOS_POLICY_PATH=$COSMOS_POLICY_PATH"
    "COSMOS_POLICY_USE_RAW_INFERENCE=${COSMOS_POLICY_USE_RAW_INFERENCE:-1}"
    "COSMOS_POLICY_INFERENCE_MODE=${COSMOS_POLICY_INFERENCE_MODE:-subprocess}"
    "COSMOS_POLICY_PYTHON=$COSMOS_POLICY_PYTHON"
    "COSMOS_PREDICT2_REPO=$COSMOS_PREDICT2_REPO"
    "COSMOS_PREDICT25_LOCAL_MODEL_DIR=$COSMOS_PREDICT25_LOCAL_MODEL_DIR"
    "COSMOS_POLICY_EXTRA_PYTHONPATH=$COSMOS_POLICY_EXTRA_PYTHONPATH"
    "COSMOS_WORKER_SITE_PACKAGES=$COSMOS_WORKER_SITE_PACKAGES"
    "COSMOS_WORKER_CUDA_LIBRARY_PATH=$COSMOS_WORKER_CUDA_LIBRARY_PATH"
    "LD_LIBRARY_PATH=$WORKER_LD_LIBRARY_PATH"
    "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES=$COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES"
    "COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION=${COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION:-5}"
    "GRADIENT_CHECKPOINTING=1"
    "USE_FSDP1=1"
    "SKIP_TEACHER_COMPILE=1"
    "ENABLE_WANDB=${ENABLE_WANDB:-0}"
    "WANDB_MODE=offline"
    "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    "HF_DATASETS_OFFLINE=1"
    "TRANSFORMERS_OFFLINE=1"
    "HF_HUB_OFFLINE=1"
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
)

command=(
    "$TORCHRUN_BIN"
    "--nproc_per_node=8"
    "--master_port=$MASTER_PORT"
    "$SCRIPT_DIR/train.py"
    --teacher-model-path "$COSMOS_POLICY_PATH"
    --dataset-path "$DATASET_PATH"
    --output-dir "$OUTPUT_DIR"
)
if [[ -n "$RESUME_FROM_PATH" ]]; then
    command+=(--resume-from-path "$RESUME_FROM_PATH")
fi
command+=(--gradient-accumulation-steps 4)

if ! PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$COSMOS_PREDICT2_REPO:$COSMOS_POLICY_EXTRA_PYTHONPATH" \
    LD_LIBRARY_PATH="$WORKER_LD_LIBRARY_PATH" \
    "$COSMOS_POLICY_PYTHON" -c \
    'import cosmos_cuda, cosmos_predict2; from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import get_action'; then
    die "worker import preflight failed"
fi

preflight_code='import os
from importlib import import_module
from distillation_flowmap.cosmos_training_contract import (
    contract_metadata,
    validate_contract_metadata,
)

cfg = import_module(os.environ["CONFIG_FILE"]).cfg
metadata = contract_metadata(cfg, stage="raw_stage1")
validate_contract_metadata(metadata, required_stage="raw_stage1")
resume_path = os.environ["RESUME_FROM_PATH"] or None
expected = {
    "student_backend": "wan_flowmap",
    "teacher_backend": "cosmos_policy",
    "training_contract_stage": "raw_stage1",
    "student_base_model_path": os.environ["WAN_STUDENT_BASE_MODEL_PATH"],
    "teacher_model_path": os.environ["COSMOS_POLICY_PATH"],
    "resume_from_path": resume_path,
    "resume_online_from_target": False,
    "reset_resume_step": False,
    "resume_optimizer_state": resume_path is not None,
    "seed": int(os.environ["TRAIN_SEED"]),
    "max_train_steps": int(os.environ["MAX_TRAIN_STEPS"]),
    "save_interval": int(os.environ["SAVE_INTERVAL"]),
}
for field, expected_value in expected.items():
    actual_value = getattr(cfg, field, None)
    if type(actual_value) is not type(expected_value) or actual_value != expected_value:
        raise RuntimeError(
            f"{field} must be exactly {expected_value!r}, got {actual_value!r}"
        )
print("corrected Cosmos raw Stage-1 config preflight passed")'
if ! (
    cd "$PROJECT_ROOT"
    env "${launch_env[@]}" \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/wan_va:$PROJECT_ROOT/distillation_flowmap:${PYTHONPATH:-}" \
        "$PREFLIGHT_BIN" -c "$preflight_code"
); then
    die "config preflight failed"
fi

for item in "${launch_env[@]}"; do
    printf '%s\n' "$item"
done
printf 'MASTER_PORT=%s\n' "$MASTER_PORT"
printf 'RUN_TAG=%s\n' "$RUN_TAG"
printf 'COMMAND='
printf '%q ' "${command[@]}"
printf '\n'

if (( dry_run )); then
    exit 0
fi

if [[ -z "$RESUME_STEP" ]]; then
    OUTPUT_PARENT="$(dirname "$OUTPUT_DIR")"
    mkdir -p "$OUTPUT_PARENT" || die \
        "failed to create OUTPUT_DIR parent: $OUTPUT_PARENT"
    mkdir "$OUTPUT_DIR" || die \
        "failed to atomically claim OUTPUT_DIR: $OUTPUT_DIR"
fi
launch_env_text=""
for item in "${launch_env[@]}"; do
    launch_env_text+="${item}"$'\n'
done
printf -v launch_command_text '%q ' "${command[@]}"
launch_command_text+=$'\n'
artifact_write_code='import os, sys, tempfile
from pathlib import Path

root = Path(sys.argv[1])

def atomic_write(name, content):
    destination = root / name
    descriptor, temp_name = tempfile.mkstemp(
        dir=root, prefix=f".{name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)

atomic_write("launch_env.txt", sys.argv[2])
atomic_write("launch_command.txt", sys.argv[3])
directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)'
if ! PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -c "$artifact_write_code" \
    "$OUTPUT_DIR" "$launch_env_text" "$launch_command_text"; then
    die "failed to atomically write launch artifacts in OUTPUT_DIR: $OUTPUT_DIR"
fi
cd "$PROJECT_ROOT"
exec env "${launch_env[@]}" "${command[@]}"
