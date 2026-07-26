#!/usr/bin/env bash
# Resume the aligned Cosmos LIBERO Stage-1 run, then continue Stage-2 and eval.
set -euo pipefail

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PIPELINE_LAUNCHER="${COSMOS_PIPELINE_LAUNCHER:-$SCRIPT_DIR/run_cosmos_stage1_stage2_eval_8gpu.sh}"

DRY_RUN=0
while (( $# > 0 )); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            cat <<'EOF'
Usage:
  bash distillation_flowmap/resume_cosmos_aligned_full40_8gpu.sh [--dry-run]

Resumes the aligned Cosmos LIBERO Stage-1 run, then serially runs Stage-2 and
the full matched-budget student/teacher evaluation. Settings may be overridden
with environment variables.
EOF
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ -f "$PIPELINE_LAUNCHER" ]] || \
    die "pipeline launcher is missing: $PIPELINE_LAUNCHER"

REPO="${REPO:-/kpfs-intern/jialongliu/projects/cosmos-predict2.5-formal-compat-441b897}"
PYTHON_BIN="${PYTHON_BIN:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"
WAN_STUDENT_BASE_MODEL_PATH="${WAN_STUDENT_BASE_MODEL_PATH:-/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero}"
COSMOS_POLICY_PATH="${COSMOS_POLICY_PATH:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
COSMOS_PREDICT2_REPO="${COSMOS_PREDICT2_REPO:-$REPO}"
COSMOS_POLICY_PYTHON="${COSMOS_POLICY_PYTHON:-/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/bin/python}"
COSMOS_POLICY_EXTRA_PYTHONPATH="${COSMOS_POLICY_EXTRA_PYTHONPATH:-$COSMOS_PREDICT2_REPO/packages/cosmos-cuda:$COSMOS_PREDICT2_REPO/packages/cosmos-oss}"
DATASET_PATH="${DATASET_PATH:-/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-$DATASET_PATH/empty_emb.pt}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_cosmos_aligned_stage1_stage2_full40_8gpu_20260725}"
RUN_TAG="${RUN_TAG:-aligned-anchor-field-full40-20260725}"
RESUME_STEP="${RESUME_STEP:-1000}"
STAGE1_STEPS="${STAGE1_STEPS:-5000}"
STAGE2_STEPS="${STAGE2_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
EPISODES="${EPISODES:-50}"
STAGE1_MASTER_PORT="${STAGE1_MASTER_PORT:-29671}"
STAGE2_MASTER_PORT="${STAGE2_MASTER_PORT:-29672}"

[[ "$RESUME_STEP" =~ ^[1-9][0-9]*$ ]] || \
    die "RESUME_STEP must be a positive integer"

if (( ! DRY_RUN )); then
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    (( ${#devices[@]} == 8 )) || \
        die "CUDA_VISIBLE_DEVICES must contain exactly eight devices"
fi

resume_transformer="$OUTPUT_ROOT/$RUN_TAG/stage1/checkpoints/step_$RESUME_STEP/target_student/transformer"
[[ -f "$resume_transformer/config.json" ]] || \
    die "Stage-1 resume checkpoint config is missing: $resume_transformer/config.json"
if [[ ! -f "$resume_transformer/diffusion_pytorch_model.safetensors" && \
      ! -f "$resume_transformer/diffusion_pytorch_model.safetensors.index.json" ]]; then
    die "Stage-1 resume checkpoint weights are missing: $resume_transformer"
fi

export REPO
export PYTHON_BIN
export WAN_STUDENT_BASE_MODEL_PATH
export COSMOS_POLICY_PATH
export COSMOS_PREDICT2_REPO
export COSMOS_POLICY_PYTHON
export COSMOS_POLICY_EXTRA_PYTHONPATH
export DATASET_PATH
export EMPTY_EMB_PATH
export CUDA_VISIBLE_DEVICES
# step_1000 was written by the former FSDP2 path, so its DTensor Adam moments
# cannot be loaded into the corrected FSDP1 training path. Resume model weights
# and the global step, then rebuild optimizer moments locally.
export STAGE1_RESUME_OPTIMIZER_STATE=0

common=(
    --output-root "$OUTPUT_ROOT"
    --run-tag "$RUN_TAG"
    --stage1-steps "$STAGE1_STEPS"
    --stage2-steps "$STAGE2_STEPS"
    --save-interval "$SAVE_INTERVAL"
    --episodes "$EPISODES"
    --stage1-master-port "$STAGE1_MASTER_PORT"
    --stage2-master-port "$STAGE2_MASTER_PORT"
)
dry_run=()
if (( DRY_RUN )); then
    dry_run=(--dry-run)
fi

printf 'RESUME_CHAIN=stage1(step_%s)->stage2->eval\n' "$RESUME_STEP"
printf 'RUN_ROOT=%s/%s\n' "$OUTPUT_ROOT" "$RUN_TAG"

printf 'RESUME_PHASE=stage1\n'
bash "$PIPELINE_LAUNCHER" \
    --phase stage1 \
    "${common[@]}" \
    --resume-stage stage1 \
    --resume-step "$RESUME_STEP" \
    "${dry_run[@]}"

printf 'RESUME_PHASE=stage2\n'
bash "$PIPELINE_LAUNCHER" \
    --phase stage2 \
    "${common[@]}" \
    "${dry_run[@]}"

printf 'RESUME_PHASE=eval\n'
bash "$PIPELINE_LAUNCHER" \
    --phase eval \
    "${common[@]}" \
    "${dry_run[@]}"
