#!/usr/bin/env bash
# Launch exactly one aligned Cosmos LIBERO Stage-2 experiment arm on 8 GPUs.
set -euo pipefail


die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}


require_dir() {
    [[ -d "$2" ]] || die "$1 is not a directory: $2"
}


require_file() {
    [[ -f "$2" ]] || die "$1 is missing: $2"
}


validate_positive_integer() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer: $2"
}


validate_port() {
    validate_positive_integer "master port" "$1"
    (( 10#$1 <= 65535 )) || die "master port must be in [1, 65535]: $1"
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


validate_binary_flag() {
    [[ "$2" == "0" || "$2" == "1" ]] || die "$1 must be 0 or 1: $2"
}


print_assignment() {
    printf '%s=%s\n' "$1" "$2"
}


PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PREFLIGHT_BIN="${PREFLIGHT_BIN:-/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
[[ -n "${COSMOS_STAGE1_ROOT:-}" ]] || die "COSMOS_STAGE1_ROOT must be explicitly set"
[[ -n "${STUDENT_BASE_MODEL_PATH:-}" ]] || die "STUDENT_BASE_MODEL_PATH must be explicitly set"
STAGE1_ROOT="$COSMOS_STAGE1_ROOT"
DATASET_PATH="${DATASET_PATH:-/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot}"
COSMOS_POLICY_PATH="${COSMOS_POLICY_PATH:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
COSMOS_WORKER_ENV_ROOT="${COSMOS_WORKER_ENV_ROOT:-/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310}"
COSMOS_POLICY_PYTHON="${COSMOS_POLICY_PYTHON:-${COSMOS_WORKER_ENV_ROOT}/bin/python}"
COSMOS_PREDICT2_REPO="${COSMOS_PREDICT2_REPO:-/kpfs-intern/jialongliu/projects/cosmos-predict2.5}"
COSMOS_PREDICT25_LOCAL_MODEL_DIR="${COSMOS_PREDICT25_LOCAL_MODEL_DIR:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World}"
COSMOS_WORKER_SITE_PACKAGES="${COSMOS_WORKER_SITE_PACKAGES:-${COSMOS_WORKER_ENV_ROOT}/lib/python3.10/site-packages}"
COSMOS_POLICY_EXTRA_PYTHONPATH="$COSMOS_PREDICT2_REPO/packages/cosmos-cuda:$COSMOS_PREDICT2_REPO/packages/cosmos-oss"
[[ -n "${COSMOS_PROVENANCE_LOCK_ROOT:-}" ]] || die \
    "COSMOS_PROVENANCE_LOCK_ROOT must be explicitly set"
VERIFY_LARGE_ARTIFACT_DIGESTS="${VERIFY_LARGE_ARTIFACT_DIGESTS:-1}"
validate_binary_flag VERIFY_LARGE_ARTIFACT_DIGESTS \
    "$VERIFY_LARGE_ARTIFACT_DIGESTS"
[[ "$VERIFY_LARGE_ARTIFACT_DIGESTS" == "1" ]] || die \
    "VERIFY_LARGE_ARTIFACT_DIGESTS must be 1 until a reviewed immutable-store attestation exists"
for test_only_name in "${!COSMOS_PROVENANCE_TEST_@}"; do
    die "test-only provenance environment is forbidden in the production launcher: $test_only_name"
done
FLASHWAM_PROVENANCE_REPO="$PROJECT_ROOT"

arm="${1:-}"
[[ -n "$arm" ]] || die "an experiment arm is required"
shift

steps=""
save_interval=1000
master_port=""
output_root="${OUTPUT_ROOT:-/kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_cosmos_aligned_8gpu}"
run_tag="${RUN_TAG:-default}"
resume_step=""
dry_run=0
check_only=0
while (( $# > 0 )); do
    case "$1" in
        --steps)
            (( $# >= 2 )) || die "--steps requires a value"
            steps="$2"; shift 2 ;;
        --save-interval)
            (( $# >= 2 )) || die "--save-interval requires a value"
            save_interval="$2"; shift 2 ;;
        --master-port)
            (( $# >= 2 )) || die "--master-port requires a value"
            master_port="$2"; shift 2 ;;
        --output-root)
            (( $# >= 2 )) || die "--output-root requires a value"
            output_root="$2"; shift 2 ;;
        --run-tag)
            (( $# >= 2 )) || die "--run-tag requires a value"
            run_tag="$2"; shift 2 ;;
        --resume-step)
            (( $# >= 2 )) || die "--resume-step requires a value"
            resume_step="$2"; shift 2 ;;
        --dry-run)
            dry_run=1; shift ;;
        --check-only)
            check_only=1; shift ;;
        *)
            die "unknown option: $1" ;;
    esac
done
(( ! (dry_run && check_only) )) || die "--dry-run and --check-only are mutually exclusive"
[[ -z "$steps" ]] || validate_positive_integer "steps" "$steps"
validate_positive_integer "save interval" "$save_interval"
[[ -z "$master_port" ]] || validate_port "$master_port"
[[ -z "$resume_step" ]] || validate_positive_integer "resume step" "$resume_step"

provenance_code='import sys
from distillation_flowmap.cosmos_libero_provenance import (
    canonical_json,
    resolve_formal_provenance,
)
(
    dataset, teacher, video_vae, local_model, lock_root,
    flashwam_repo, cosmos_repo, worker_python, worker_site_packages,
    extra_pythonpath, verify_large,
) = sys.argv[1:]
payload = resolve_formal_provenance(
    dataset_root=dataset,
    dataset_lock=f"{lock_root}/dataset.lock.json",
    teacher_root=teacher,
    teacher_lock=f"{lock_root}/teacher.lock.json",
    video_vae_root=video_vae,
    video_vae_lock=f"{lock_root}/video_vae.lock.json",
    local_model_root=local_model,
    local_model_lock=f"{lock_root}/local_model.lock.json",
    flashwam_repo=flashwam_repo,
    cosmos_repo=cosmos_repo,
    worker_python=worker_python,
    worker_site_packages=worker_site_packages,
    extra_pythonpath=tuple(part for part in extra_pythonpath.split(":") if part),
    verify_large_artifact_digests=verify_large == "1",
)
print("PROVENANCE_IDENTITY_JSON=" + canonical_json(payload))
print("PROVENANCE_IDENTITY_SHA256=" + payload["identity_sha256"])'
provenance_output="$(
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/wan_va:$PROJECT_ROOT/distillation_flowmap:${PYTHONPATH:-}" \
        "$PREFLIGHT_BIN" -c "$provenance_code" \
        "$DATASET_PATH" "$COSMOS_POLICY_PATH" "$STUDENT_BASE_MODEL_PATH" \
        "$COSMOS_PREDICT25_LOCAL_MODEL_DIR" "$COSMOS_PROVENANCE_LOCK_ROOT" \
        "$FLASHWAM_PROVENANCE_REPO" "$COSMOS_PREDICT2_REPO" \
        "$COSMOS_POLICY_PYTHON" "$COSMOS_WORKER_SITE_PACKAGES" \
        "$COSMOS_POLICY_EXTRA_PYTHONPATH" "$VERIFY_LARGE_ARTIFACT_DIGESTS"
)" || die "formal provenance preflight failed"
while IFS='=' read -r key value; do
    case "$key" in
        PROVENANCE_IDENTITY_JSON|PROVENANCE_IDENTITY_SHA256)
            printf -v "$key" '%s' "$value"; export "$key" ;;
        *) die "unexpected provenance preflight output: $key" ;;
    esac
done <<< "$provenance_output"
[[ -n "${PROVENANCE_IDENTITY_JSON:-}" ]] || die \
    "provenance preflight omitted canonical JSON"
[[ -n "${PROVENANCE_IDENTITY_SHA256:-}" ]] || die \
    "provenance preflight omitted identity digest"
COSMOS_LIBERO_PROVENANCE_JSON="$PROVENANCE_IDENTITY_JSON"
export COSMOS_LIBERO_PROVENANCE_JSON

resolver_code='import sys
import json
from distillation_flowmap.cosmos_libero_variants import (
    VariantError, canonical_variant_json, resolve_variant,
)
(
    name, output_root, run_tag, steps_raw, save_raw, port_raw,
    dataset_path, teacher_path, vae_path, repo, worker_python,
    worker_pythonpath, local_model,
    provenance_json,
) = sys.argv[1:]
try:
    record = resolve_variant(
        name,
        output_root=output_root,
        run_tag=run_tag,
        steps=int(steps_raw) if steps_raw else None,
        save_interval=int(save_raw),
        master_port=int(port_raw) if port_raw else None,
        dataset_path=dataset_path,
        teacher_model_path=teacher_path,
        cosmos_video_vae_model_path=vae_path,
        cosmos_policy_repo=repo,
        cosmos_policy_python=worker_python,
        cosmos_policy_extra_pythonpath=worker_pythonpath,
        cosmos_policy_local_model_dir=local_model,
        attention_mode="flex",
        provenance=json.loads(provenance_json),
    )
except (VariantError, ValueError) as exc:
    raise SystemExit(str(exc))
pairs = ";".join(",".join(str(v) for v in pair) for pair in record["rollout_step_pairs"])
choices = ",".join(str(v) for v in record["danceopd_rollout_steps"])
values = {
    "VARIANT_NAME": record["name"],
    "RUN_TAG": record["run_tag"],
    "COSMOS_PROGRESSIVE_STAGE": record["progressive_stage"],
    "MAX_TRAIN_STEPS": record["max_train_steps"],
    "SAVE_INTERVAL": record["save_interval"],
    "MASTER_PORT": record["master_port"],
    "OUTPUT_DIR": record["output_dir"],
    "OPD_ROLLOUT_STEP_PAIRS": pairs,
    "OPD_DANCEOPD_ROLLOUT_STEPS": choices,
    "OPD_DANCEOPD_ENDPOINT_WEIGHT": record["video_endpoint_weight"],
    "OPD_DANCEOPD_VELOCITY_WEIGHT": record["video_velocity_weight"],
    "OPD_DANCEOPD_ACTION_ENDPOINT_WEIGHT": record["action_endpoint_weight"],
    "OPD_AUX_ACTION": int(record["action_opd_enabled"]),
    "USE_OPD_AUX": int(record["use_opd_aux"]),
    "OPD_AUX_STANDALONE_STEP": int(record["opd_aux_standalone_step"]),
    "LEARNING_RATE": record["learning_rate"],
    "OPD_AUX_WEIGHT": record["opd_aux_weight"],
    "OPD_AUX_WARMUP_STEPS": record["opd_aux_warmup_steps"],
    "OPD_AUX_PROB": record["opd_aux_prob"],
    "OPD_ROLLOUT_GRAD_MODE": record["opd_rollout_grad_mode"],
    "OPD_ROLLOUT_GRAD_STEPS": record["opd_rollout_grad_steps"],
    "OPD_ENDPOINT_FOCUS_PROB": record["opd_endpoint_focus_prob"],
    "OPD_DANCEOPD_QUERY_ALPHA": record["opd_danceopd_query_alpha"],
    "OPD_DANCEOPD_QUERY_BETA": record["opd_danceopd_query_beta"],
    "TRAIN_SEED": record["train_seed"],
    "COSMOS_LIBERO_VARIANT_JSON": canonical_variant_json(record),
}
identity_exports = {
    "BETA1": "beta1",
    "BETA2": "beta2",
    "EMA_DECAY": "ema_decay",
    "EMA_WARMUP_STEPS": "ema_warmup_steps",
    "DROP_TEXT_RATIO": "drop_text_ratio",
    "FUSE_GUIDANCE_SCALE": "fuse_guidance_scale",
    "CFG_MIN": "cfg_min",
    "CFG_MAX": "cfg_max",
    "MAX_GRAD_NORM": "max_grad_norm",
    "WARMUP_STEPS": "warmup_steps",
    "NUM_DDIM_TIMESTEPS_ACTION": "num_ddim_timesteps_action",
    "DIFFUSION_RATIO": "diffusion_ratio",
    "CONSISTENCY_RATIO": "consistency_ratio",
    "FLOWMAP_RATIO": "flowmap_ratio",
    "VIDEO_LOSS_WEIGHT": "video_loss_weight",
    "ACTION_LOSS_WEIGHT": "action_loss_weight",
    "ACTION_BLOCK_WEIGHT": "action_block_weight",
    "COSMOS_POLICY_USE_RAW_INFERENCE": "cosmos_policy_use_raw_inference",
    "SKIP_TARGET_STUDENT_FOR_COSMOS_LATENT": "skip_target_student_for_cosmos_latent",
    "COSMOS_LATENT_CDIFF_LOSS_WEIGHT": "cosmos_latent_cdiff_loss_weight",
    "COSMOS_LATENT_ENDPOINT_LOSS_WEIGHT": "cosmos_latent_endpoint_loss_weight",
    "COSMOS_LATENT_EPSILON": "cosmos_latent_epsilon",
    "COSMOS_LATENT_T_MIN": "cosmos_latent_t_min",
    "COSMOS_LATENT_T_MAX": "cosmos_latent_t_max",
    "COSMOS_LATENT_CHANNELS": "cosmos_latent_channels",
    "COSMOS_LATENT_FRAMES": "cosmos_latent_frames",
    "COSMOS_LATENT_HEIGHT": "cosmos_latent_height",
    "COSMOS_LATENT_WIDTH": "cosmos_latent_width",
    "COSMOS_LATENT_CENTER_VELOCITY_MODE": "cosmos_latent_center_velocity_mode",
    "COSMOS_LATENT_TARGET_MODE": "cosmos_latent_target_mode",
    "COSMOS_LATENT_CDIFF_INTERVAL": "cosmos_latent_cdiff_interval",
    "OPD_ACTION_ROLLOUT_GRAD_MODE": "opd_action_rollout_grad_mode",
    "OPD_AUX_INTERVAL": "opd_aux_interval",
    "OPD_DANCEOPD_VERIFY_TERMINAL_PRIOR": "opd_danceopd_verify_terminal_prior",
    "OPD_DANCEOPD_TERMINAL_PRIOR_TOLERANCE": "opd_danceopd_terminal_prior_tolerance",
    "OPD_DANCEOPD_TERMINAL_PRIOR_WARN_FACTOR": "opd_danceopd_terminal_prior_warn_factor",
    "OPD_COSMOS_SPATIAL_CROP_SIZE": "opd_cosmos_spatial_crop_size",
    "COSMOS_USE_TEACHER_ACTION_ANCHOR": "cosmos_use_teacher_action_anchor",
    "OPD_JOINT_ACTION_ROLLOUT": "opd_joint_action_rollout",
    "MECHANISM_DIAGNOSTICS": "mechanism_diagnostics",
    "MECHANISM_DIAGNOSTIC_INTERVAL": "mechanism_diagnostic_interval",
    "MECHANISM_DIAGNOSTIC_SEED": "mechanism_diagnostic_seed",
    "MECHANISM_DIAGNOSTIC_R": "mechanism_diagnostic_r",
    "MECHANISM_DIAGNOSTIC_S": "mechanism_diagnostic_s",
    "MECHANISM_DIAGNOSTIC_TEACHER_STEPS": "mechanism_diagnostic_teacher_steps",
    "MECHANISM_COSMOS_T_MIN": "mechanism_cosmos_t_min",
    "MECHANISM_COSMOS_T_MAX": "mechanism_cosmos_t_max",
    "DATASET_PATH": "dataset_path",
    "COSMOS_POLICY_PATH": "teacher_model_path",
    "COSMOS_VIDEO_VAE_MODEL_PATH": "cosmos_video_vae_model_path",
    "COSMOS_PREDICT2_REPO": "cosmos_policy_repo",
    "COSMOS_POLICY_PYTHON": "cosmos_policy_python",
    "COSMOS_POLICY_EXTRA_PYTHONPATH": "cosmos_policy_extra_pythonpath",
    "COSMOS_PREDICT25_LOCAL_MODEL_DIR": "cosmos_policy_local_model_dir",
    "COSMOS_POLICY_CONFIG_NAME": "cosmos_policy_config_name",
    "COSMOS_POLICY_CONFIG_FILE": "cosmos_policy_config_file",
    "COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION": "cosmos_policy_num_denoising_steps_action",
    "COSMOS_POLICY_SEED": "cosmos_policy_seed",
    "COSMOS_POLICY_PRIMARY_IMAGE_KEY": "cosmos_policy_primary_image_key",
    "COSMOS_POLICY_WRIST_IMAGE_KEY": "cosmos_policy_wrist_image_key",
    "COSMOS_POLICY_INFERENCE_MODE": "cosmos_policy_inference_mode",
    "ATTN_MODE": "attention_mode",
}
for env_name, field in identity_exports.items():
    value = record[field]
    if isinstance(value, bool):
        value = int(value)
    values[env_name] = value
for key, value in values.items():
    print(f"{key}={value}")'
resolved="$(
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/wan_va:$PROJECT_ROOT/distillation_flowmap:${PYTHONPATH:-}" \
        "$PREFLIGHT_BIN" -c "$resolver_code" \
        "$arm" "$output_root" "$run_tag" "$steps" "$save_interval" "$master_port" \
        "$DATASET_PATH" "$COSMOS_POLICY_PATH" "$STUDENT_BASE_MODEL_PATH" \
        "$COSMOS_PREDICT2_REPO" "$COSMOS_POLICY_PYTHON" \
        "$COSMOS_POLICY_EXTRA_PYTHONPATH" "$COSMOS_PREDICT25_LOCAL_MODEL_DIR" \
        "$COSMOS_LIBERO_PROVENANCE_JSON"
)" || die "variant resolution failed"
declare -A seen_resolver_keys=()
while IFS='=' read -r key value; do
    [[ "$key" =~ ^[A-Z][A-Z0-9_]*$ ]] || die "invalid resolver output key: $key"
    [[ -z "${seen_resolver_keys[$key]:-}" ]] || die \
        "duplicate resolver output key: $key"
    printf -v "$key" '%s' "$value"
    export "$key"
    seen_resolver_keys["$key"]=1
done <<< "$resolved"
for key in VARIANT_NAME RUN_TAG COSMOS_PROGRESSIVE_STAGE MAX_TRAIN_STEPS \
    SAVE_INTERVAL MASTER_PORT OUTPUT_DIR COSMOS_LIBERO_VARIANT_JSON; do
    [[ -n "${seen_resolver_keys[$key]:-}" ]] || die \
        "variant resolver omitted required field: $key"
done
validate_port "$MASTER_PORT"
if [[ -n "$resume_step" ]]; then
    (( resume_step < MAX_TRAIN_STEPS )) || die \
        "resume step must be smaller than MAX_TRAIN_STEPS"
fi

[[ -n "${COSMOS_STAGE1_ROOT:-}" ]] || die "COSMOS_STAGE1_ROOT must be explicitly set"
[[ -n "${STUDENT_BASE_MODEL_PATH:-}" ]] || die "STUDENT_BASE_MODEL_PATH must be explicitly set"
STAGE1_ROOT="$COSMOS_STAGE1_ROOT"
raw_output_preflight_code='import sys
from pathlib import Path
from distillation_flowmap.cosmos_stage2_lineage import validate_stage2_path_isolation
validate_stage2_path_isolation(
    stage1_root=Path(sys.argv[1]),
    output_dir=Path(sys.argv[2]) / sys.argv[3] / sys.argv[4],
    resume_checkpoint=None,
)'
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/wan_va:$PROJECT_ROOT/distillation_flowmap:${PYTHONPATH:-}" \
    "$PREFLIGHT_BIN" -c "$raw_output_preflight_code" \
    "$STAGE1_ROOT" "$output_root" "$run_tag" "$arm" || die \
    "raw Stage-2 output path preflight failed"
DATASET_PATH="${DATASET_PATH:-/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot}"
COSMOS_POLICY_PATH="${COSMOS_POLICY_PATH:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
COSMOS_WORKER_ENV_ROOT="${COSMOS_WORKER_ENV_ROOT:-/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310}"
COSMOS_POLICY_PYTHON="${COSMOS_POLICY_PYTHON:-${COSMOS_WORKER_ENV_ROOT}/bin/python}"
COSMOS_PREDICT2_REPO="${COSMOS_PREDICT2_REPO:-/kpfs-intern/jialongliu/projects/cosmos-predict2.5}"
COSMOS_PREDICT25_LOCAL_MODEL_DIR="${COSMOS_PREDICT25_LOCAL_MODEL_DIR:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World}"
COSMOS_WORKER_SITE_PACKAGES="${COSMOS_WORKER_SITE_PACKAGES:-${COSMOS_WORKER_ENV_ROOT}/lib/python3.10/site-packages}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES="${COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"

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
require_dir STUDENT_BASE_MODEL_PATH "$STUDENT_BASE_MODEL_PATH"
require_file STUDENT_BASE_MODEL_PATH/config.json \
    "$STUDENT_BASE_MODEL_PATH/transformer/config.json"

if [[ -n "$resume_step" ]]; then
    RESUME_FROM_PATH="$OUTPUT_DIR/checkpoints/step_$resume_step"
    RESUME_ONLINE_FROM_TARGET=0
    RESET_RESUME_STEP=0
    RESUME_OPTIMIZER_STATE=1
else
    RESUME_FROM_PATH="$STAGE1_ROOT"
    RESUME_ONLINE_FROM_TARGET=1
    RESET_RESUME_STEP=1
    RESUME_OPTIMIZER_STATE=0
fi
export RESUME_FROM_PATH RESUME_ONLINE_FROM_TARGET RESET_RESUME_STEP
export RESUME_OPTIMIZER_STATE

lineage_code='import json, os
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
expected_variant = os.environ["COSMOS_LIBERO_VARIANT_JSON"]
parent = validate_stage1_parent(stage1, expected_step=5000)
validate_stage2_path_isolation(
    stage1_root=stage1, output_dir=output, resume_checkpoint=resume,
)
if resume is not None:
    validate_stage2_resume(
        resume,
        arm_root=output,
        expected_step=int(os.environ["STAGE2_RESUME_STEP"]),
        expected_parent=parent,
    )
    for student in ("online_student", "target_student"):
        config = resume / student / "transformer" / "config.json"
        payload = json.loads(config.read_text(encoding="utf-8"))
        if payload.get("cosmos_libero_variant_json") != expected_variant:
            raise ValueError(
                f"{student} canonical Cosmos LIBERO variant does not match selected arm"
            )
        if payload.get("cosmos_libero_provenance_json") != os.environ[
            "COSMOS_LIBERO_PROVENANCE_JSON"
        ]:
            raise ValueError(
                f"{student} Cosmos LIBERO provenance does not match current inputs"
            )
    def read_required_canonical_manifest(path, *, label):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required {label} manifest is missing or not a plain file")
        raw = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"required {label} manifest is malformed JSON") from exc
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if raw != canonical + "\n":
            raise ValueError(f"required {label} manifest is not exact canonical JSON")
        return canonical

    actual_variant = read_required_canonical_manifest(
        output / "cosmos_libero_variant.json",
        label="variant",
    )
    if actual_variant != expected_variant:
        raise ValueError("run-level canonical variant manifest does not match selected arm")
    actual_provenance = read_required_canonical_manifest(
        output / "cosmos_libero_provenance.json",
        label="provenance",
    )
    if actual_provenance != os.environ["COSMOS_LIBERO_PROVENANCE_JSON"]:
        raise ValueError(
            "run-level provenance manifest does not match current inputs"
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
        OUTPUT_DIR="$OUTPUT_DIR" \
        COSMOS_LIBERO_VARIANT_JSON="$COSMOS_LIBERO_VARIANT_JSON" \
        COSMOS_LIBERO_PROVENANCE_JSON="$COSMOS_LIBERO_PROVENANCE_JSON" \
        STAGE2_RESUME_CHECKPOINT="${resume_step:+$RESUME_FROM_PATH}" \
        STAGE2_RESUME_STEP="${resume_step:-0}" \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/wan_va:$PROJECT_ROOT/distillation_flowmap:${PYTHONPATH:-}" \
        "$PREFLIGHT_BIN" -c "$lineage_code"
)" || die "Stage-2 lineage/variant preflight failed"
while IFS='=' read -r key value; do
    case "$key" in
        PARENT_STAGE1_PATH|PARENT_STAGE1_CONTRACT_IDENTITY|STAGE2_LINEAGE_JSON)
            printf -v "$key" '%s' "$value"; export "$key" ;;
        *) die "unexpected lineage preflight output: $key" ;;
    esac
done <<< "$lineage_output"

expected_student_base="$STAGE1_ROOT/target_student"
[[ "$(cd "$STUDENT_BASE_MODEL_PATH" && pwd -P)" == \
   "$(cd "$expected_student_base" && pwd -P)" ]] || die \
    "STUDENT_BASE_MODEL_PATH must identify validated Stage-1 target_student"
if [[ -z "$resume_step" ]]; then
    [[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || die \
        "Refusing fresh run with existing OUTPUT_DIR: $OUTPUT_DIR"
fi

export CONFIG_FILE=distillation_flowmap.config_libero_cosmos_policy_stage2_progressive
export OUTPUT_DIR MAX_TRAIN_STEPS SAVE_INTERVAL MASTER_PORT TRAIN_SEED
export COSMOS_PROGRESSIVE_STAGE
export COSMOS_PROGRESSIVE_OUTPUT_ROOT="$output_root"
export OUTPUT_ROOT="$output_root"
export STUDENT_BASE_MODEL_PATH COSMOS_LIBERO_VARIANT_JSON
export COSMOS_LIBERO_PROVENANCE_JSON
export OPD_COSMOS_SPATIAL_CROP_SIZE=28
export OPD_ROLLOUT_STEP_PAIRS OPD_DANCEOPD_ROLLOUT_STEPS
export OPD_DANCEOPD_ENDPOINT_WEIGHT OPD_DANCEOPD_VELOCITY_WEIGHT
export OPD_DANCEOPD_ACTION_ENDPOINT_WEIGHT OPD_AUX_ACTION USE_OPD_AUX
export OPD_AUX_STANDALONE_STEP
export LEARNING_RATE OPD_AUX_WEIGHT OPD_AUX_WARMUP_STEPS OPD_AUX_PROB
export OPD_ROLLOUT_GRAD_MODE OPD_ROLLOUT_GRAD_STEPS
export OPD_ENDPOINT_FOCUS_PROB OPD_DANCEOPD_QUERY_ALPHA
export OPD_DANCEOPD_QUERY_BETA
export COSMOS_USE_TEACHER_ACTION_ANCHOR OPD_JOINT_ACTION_ROLLOUT
export MECHANISM_DIAGNOSTICS MECHANISM_DIAGNOSTIC_INTERVAL
export MECHANISM_DIAGNOSTIC_SEED MECHANISM_DIAGNOSTIC_R
export MECHANISM_DIAGNOSTIC_S MECHANISM_DIAGNOSTIC_TEACHER_STEPS
export MECHANISM_COSMOS_T_MIN MECHANISM_COSMOS_T_MAX
unset ACTION_AWARE_WEIGHT GT_REGRESSION_WEIGHT ACTION_TRANSITION_PARAM
unset ACTION_LOCAL_FM_WEIGHT ACTION_TRANSITION_BLOCK_WEIGHT
unset ACTION_LOCAL_FM_BLOCK_WEIGHT
export ATTN_MODE
export COSMOS_POLICY_INFERENCE_MODE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/kpfs-intern/jialongliu/models/cosmos_predict2_5/hf_cache
export CUDA_VISIBLE_DEVICES COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES
export COSMOS_POLICY_PATH COSMOS_POLICY_PYTHON COSMOS_PREDICT2_REPO
export COSMOS_PREDICT25_LOCAL_MODEL_DIR
export COSMOS_POLICY_EXTRA_PYTHONPATH="${COSMOS_POLICY_EXTRA_PYTHONPATH:-$COSMOS_PREDICT2_REPO/packages/cosmos-cuda:$COSMOS_PREDICT2_REPO/packages/cosmos-oss}"
export COSMOS_WORKER_CUDA_LIBRARY_PATH="${COSMOS_WORKER_CUDA_LIBRARY_PATH:-${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cublas/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cuda_cupti/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cuda_nvrtc/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cuda_runtime/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cudnn/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cufft/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cufile/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/curand/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cusolver/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cusparse/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cusparselt/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/nccl/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/nvjitlink/lib}"
export LD_LIBRARY_PATH="${COSMOS_WORKER_CUDA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

schema_actions_code='import os
from distillation_flowmap.cosmos_progressive_env_schema import (
    CLEARED_LEGACY_ENV,
    resolve_pinned_environment,
)
for name, value in sorted(resolve_pinned_environment(os.environ).items()):
    if "\t" in value or "\n" in value:
        raise SystemExit(f"unsafe pinned environment value: {name}")
    print(f"SET_PINNED\t{name}\t{value}")
for name in sorted(CLEARED_LEGACY_ENV):
    print(f"UNSET_LEGACY\t{name}\t")'
schema_actions_output="$(
    cd "$PROJECT_ROOT"
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/wan_va:$PROJECT_ROOT/distillation_flowmap:${PYTHONPATH:-}" \
        "$PREFLIGHT_BIN" -c "$schema_actions_code"
)" || die "failed to resolve schema-owned launcher actions"
while IFS=$'\t' read -r action key value; do
    [[ "$key" =~ ^[A-Z][A-Z0-9_]*$ ]] || die \
        "invalid schema-owned launcher action key: $key"
    case "$action" in
        SET_PINNED)
            printf -v "$key" '%s' "$value"
            export "$key"
            ;;
        UNSET_LEGACY)
            unset "$key"
            ;;
        *)
            die "unknown schema-owned launcher action: $action"
            ;;
    esac
done <<< "$schema_actions_output"

env_contract_code='import json, os
from distillation_flowmap.cosmos_progressive_env_schema import (
    launcher_action_manifest,
    validate_launcher_environment,
)
actions = launcher_action_manifest(os.environ)
validate_launcher_environment(os.environ, actions=actions)
print(
    "ENV_CONTRACT_ACTIONS_JSON="
    + json.dumps(
        actions, sort_keys=True, separators=(",", ":")
    )
)'
env_contract_output="$(
    cd "$PROJECT_ROOT"
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/wan_va:$PROJECT_ROOT/distillation_flowmap:${PYTHONPATH:-}" \
        "$PREFLIGHT_BIN" -c "$env_contract_code"
)" || die "launcher environment contract preflight failed"
while IFS='=' read -r key value; do
    case "$key" in
        ENV_CONTRACT_ACTIONS_JSON)
            printf -v "$key" '%s' "$value"; export "$key" ;;
        *) die "unexpected environment contract output: $key" ;;
    esac
done <<< "$env_contract_output"

config_preflight_code='import json, os
from importlib import import_module
cfg = import_module(os.environ["CONFIG_FILE"]).cfg
required = {
    "student_base_model_path": os.environ["STUDENT_BASE_MODEL_PATH"],
    "resume_from_path": os.environ["RESUME_FROM_PATH"],
    "parent_stage1_path": os.environ["PARENT_STAGE1_PATH"],
    "parent_stage1_contract_identity": os.environ["PARENT_STAGE1_CONTRACT_IDENTITY"],
    "stage2_lineage_json": os.environ["STAGE2_LINEAGE_JSON"],
    "cosmos_libero_variant_json": os.environ["COSMOS_LIBERO_VARIANT_JSON"],
    "cosmos_libero_provenance_json": os.environ["COSMOS_LIBERO_PROVENANCE_JSON"],
}
for field, expected in required.items():
    actual = getattr(cfg, field, None)
    if actual != expected:
        raise RuntimeError(f"{field} must be exactly {expected!r}, got {actual!r}")
print(
    "CONFIG_IDENTITY_JSON="
    + json.dumps(
        cfg.cosmos_libero_variant_identity,
        sort_keys=True,
        separators=(",", ":"),
    )
)'
(
    cd "$PROJECT_ROOT"
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/wan_va:$PROJECT_ROOT/distillation_flowmap:${PYTHONPATH:-}" \
        "$PREFLIGHT_BIN" -c "$config_preflight_code"
) || die "config preflight failed"

train_cmd=(
    "$TORCHRUN_BIN"
    --nproc_per_node=8
    "--master_port=$MASTER_PORT"
    distillation_flowmap/train.py
    --teacher-model-path "$COSMOS_POLICY_PATH"
    --dataset-path "$DATASET_PATH"
    --output-dir "$OUTPUT_DIR"
    --resume-from-path "$RESUME_FROM_PATH"
    --gradient-accumulation-steps 1
)

for key in \
    VARIANT_NAME RUN_TAG COSMOS_PROGRESSIVE_STAGE MASTER_PORT OUTPUT_ROOT OUTPUT_DIR \
    MAX_TRAIN_STEPS SAVE_INTERVAL RESUME_FROM_PATH \
    RESUME_ONLINE_FROM_TARGET RESET_RESUME_STEP RESUME_OPTIMIZER_STATE \
    OPD_ROLLOUT_STEP_PAIRS OPD_DANCEOPD_ROLLOUT_STEPS \
    OPD_DANCEOPD_ENDPOINT_WEIGHT OPD_DANCEOPD_VELOCITY_WEIGHT \
    OPD_DANCEOPD_ACTION_ENDPOINT_WEIGHT OPD_AUX_ACTION USE_OPD_AUX \
    OPD_AUX_STANDALONE_STEP LEARNING_RATE OPD_AUX_WEIGHT \
    OPD_AUX_WARMUP_STEPS OPD_AUX_PROB OPD_ROLLOUT_GRAD_MODE \
    OPD_ROLLOUT_GRAD_STEPS OPD_ENDPOINT_FOCUS_PROB \
    OPD_DANCEOPD_QUERY_ALPHA OPD_DANCEOPD_QUERY_BETA \
    TRAIN_SEED ENABLE_TENSORBOARD ENABLE_WANDB \
    WANDB_MODE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_OFFLINE \
    CONFIG_FILE COSMOS_POLICY_PATH DATASET_PATH COSMOS_POLICY_PYTHON \
    COSMOS_PREDICT2_REPO COSMOS_PREDICT25_LOCAL_MODEL_DIR \
    COSMOS_POLICY_EXTRA_PYTHONPATH COSMOS_WORKER_CUDA_LIBRARY_PATH \
    ATTN_MODE \
    STUDENT_BASE_MODEL_PATH PARENT_STAGE1_PATH \
    PARENT_STAGE1_CONTRACT_IDENTITY STAGE2_LINEAGE_JSON \
    COSMOS_LIBERO_VARIANT_JSON CUDA_VISIBLE_DEVICES \
    COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES \
    PROVENANCE_IDENTITY_JSON PROVENANCE_IDENTITY_SHA256 \
    COSMOS_LIBERO_PROVENANCE_JSON COSMOS_PROVENANCE_LOCK_ROOT \
    VERIFY_LARGE_ARTIFACT_DIGESTS; do
    print_assignment "$key" "${!key}"
done
print_assignment ENV_CONTRACT_ACTIONS_JSON "$ENV_CONTRACT_ACTIONS_JSON"
for key in \
    MECHANISM_DIAGNOSTICS MECHANISM_DIAGNOSTIC_INTERVAL \
    MECHANISM_DIAGNOSTIC_SEED MECHANISM_DIAGNOSTIC_R \
    MECHANISM_DIAGNOSTIC_S MECHANISM_DIAGNOSTIC_TEACHER_STEPS \
    MECHANISM_COSMOS_T_MIN MECHANISM_COSMOS_T_MAX; do
    print_assignment "$key" "${!key}"
done
for key in \
    BETA1 BETA2 EMA_DECAY EMA_WARMUP_STEPS DROP_TEXT_RATIO \
    FUSE_GUIDANCE_SCALE CFG_MIN CFG_MAX MAX_GRAD_NORM WARMUP_STEPS \
    NUM_DDIM_TIMESTEPS_ACTION DIFFUSION_RATIO CONSISTENCY_RATIO \
    FLOWMAP_RATIO VIDEO_LOSS_WEIGHT ACTION_LOSS_WEIGHT \
    ACTION_BLOCK_WEIGHT COSMOS_POLICY_USE_RAW_INFERENCE \
    SKIP_TARGET_STUDENT_FOR_COSMOS_LATENT \
    COSMOS_LATENT_CDIFF_LOSS_WEIGHT COSMOS_LATENT_ENDPOINT_LOSS_WEIGHT \
    COSMOS_LATENT_EPSILON COSMOS_LATENT_T_MIN COSMOS_LATENT_T_MAX \
    COSMOS_LATENT_CHANNELS COSMOS_LATENT_FRAMES COSMOS_LATENT_HEIGHT \
    COSMOS_LATENT_WIDTH COSMOS_LATENT_CENTER_VELOCITY_MODE \
    COSMOS_LATENT_TARGET_MODE COSMOS_LATENT_CDIFF_INTERVAL \
    OPD_ACTION_ROLLOUT_GRAD_MODE OPD_AUX_INTERVAL \
    OPD_DANCEOPD_VERIFY_TERMINAL_PRIOR \
    OPD_DANCEOPD_TERMINAL_PRIOR_TOLERANCE \
    OPD_DANCEOPD_TERMINAL_PRIOR_WARN_FACTOR \
    OPD_COSMOS_SPATIAL_CROP_SIZE COSMOS_USE_TEACHER_ACTION_ANCHOR \
    OPD_JOINT_ACTION_ROLLOUT \
    COSMOS_POLICY_CONFIG_NAME COSMOS_POLICY_CONFIG_FILE \
    COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION COSMOS_POLICY_SEED \
    COSMOS_POLICY_PRIMARY_IMAGE_KEY COSMOS_POLICY_WRIST_IMAGE_KEY \
    COSMOS_VIDEO_VAE_MODEL_PATH COSMOS_POLICY_INFERENCE_MODE; do
    print_assignment "$key" "${!key}"
done
for key in \
    OPD_AUX_EMPTY_CACHE COSMOS_TRAIN_STEP_PROFILE OPD_PROFILE \
    SKIP_TEACHER_COMPILE CACHE_DATASET_IN_MEMORY \
    COSMOS_POLICY_VALIDATE_WEIGHTS ENABLE_LIGHT_EVAL ENABLE_ROLLOUT_EVAL \
    ENABLE_STAGE1_START_EVAL ENABLE_STAGE1_START_EVAL_BASELINE \
    LIGHT_EVAL_INTERVAL LIGHT_EVAL_NUM_BATCHES LIGHT_EVAL_SEED \
    LIGHT_EVAL_START_INDEX \
    STOP_AFTER_STEP GRADIENT_CHECKPOINTING USE_FSDP1 \
    OPD_AUX_GRADIENT_CHECKPOINTING OPD_SERIAL_STUDENT_CFG \
    PYTORCH_CUDA_ALLOC_CONF HF_HOME; do
    print_assignment "$key" "${!key}"
done
print_assignment DATASET_SAMPLE_MANIFEST "<unset>"
print_assignment STAGE1_CKPT_NAME "<unset>"
for key in \
    DISTILL_MODE TEACHER_PATH COSMOS_PROGRESSIVE_RUN_ID \
    OPD_AUX_VARIANT OPD_TEACHER_TARGET_MODE ROLLOUT_STEP_PAIRS \
    VIDEO_TRANSITION_PARAM VIDEO_TRANSITION_WEIGHT \
    OPD_ENDPOINT_AUX_WEIGHT LOCAL_FM_WEIGHT \
    OPD_TRANSITION_GROUP_WEIGHT OPD_ANCHOR_CAP_RATIO \
    OPD_AUX_USE_NOFSDP_ROLLOUT ACTION_AWARE_WEIGHT \
    GT_REGRESSION_WEIGHT ACTION_TRANSITION_PARAM \
    ACTION_LOCAL_FM_WEIGHT ACTION_TRANSITION_BLOCK_WEIGHT \
    ACTION_LOCAL_FM_BLOCK_WEIGHT; do
    print_assignment "$key" "<unset>"
done
printf 'COMMAND='
printf '%q ' "${train_cmd[@]}"
printf '\n'

if (( dry_run || check_only )); then
    exit 0
fi

if [[ -z "$resume_step" ]]; then
    mkdir -p "$(dirname "$OUTPUT_DIR")"
    mkdir "$OUTPUT_DIR" || die "failed to atomically claim OUTPUT_DIR: $OUTPUT_DIR"
    manifest_code='import os
from pathlib import Path
output = Path(os.environ["OUTPUT_DIR"])
for filename, variable in (
    ("cosmos_libero_variant.json", "COSMOS_LIBERO_VARIANT_JSON"),
    ("cosmos_libero_provenance.json", "COSMOS_LIBERO_PROVENANCE_JSON"),
):
    destination = output / filename
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(os.environ[variable] + "\n")
        handle.flush()
        os.fsync(handle.fileno())'
    "$PREFLIGHT_BIN" -c "$manifest_code" || die \
        "failed to persist canonical variant manifest"
fi
cd "$PROJECT_ROOT"
exec "${train_cmd[@]}"
