#!/usr/bin/env bash
# Safe, explicit launcher for the independent Cosmos mixed-step policies.
#
# This file only creates a tmux session and an immutable per-session worker
# script when a user invokes `preflight`, `start`, or `eval`.  It never removes
# an output, creates a cache, or quietly starts an existing S4 experiment.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
readonly RUNNER_PATH="$SCRIPT_DIR/run_cosmos_mixed_step_policy.py"
readonly PREFLIGHT_PORT=29860
readonly PREFLIGHT_FORCE_SEQUENCE="s1,s2,s4,s1,s2,s4,s1,s2,s4"
readonly TRAIN_STEPS=5000
readonly TRAIN_SAVE_INTERVAL=250
readonly RESERVATION_ROOT_NAME=".cosmos_mixed_step_reservations"
readonly DEFAULT_STAGE1_CHECKPOINT="/root/nas/junjie/jj/Any_WAM/distillation_flowmap/output_libero_cosmos_policy_stage1_cosmos_latent_cdiff_8gpu_20260706_cosmos_latent_s1s2_8gpu/checkpoints/step_5000"
readonly -a SUPPORTED_POLICIES=("universe" "s2" "s1")
declare -Ar POLICY_PORTS=( ["universe"]=29861 ["s2"]=29862 ["s1"]=29863 )

TMUX_BIN="${TMUX_BIN:-tmux}"
PYTHON_BIN="${PYTHON_BIN:-/root/nas/junjie/conda_envs/any_wam/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/nas/junjie/conda_envs/any_wam/bin/torchrun}"
DEVICE_LIST="${DEVICE_LIST:-0,1,2,3,4,5,6,7}"
EVAL_DEVICE_LIST="${EVAL_DEVICE_LIST:-0}"
EVAL_WORKER_DEVICE_LIST="${EVAL_WORKER_DEVICE_LIST:-0}"
TRAIN_SEED="${TRAIN_SEED:-20260716}"

ROOT_BASE=""
PROTOCOL_SOURCE_ROOT=""
DATASET_PATH=""
TEACHER_MODEL_PATH=""
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-}"
ACTIVE_RESERVATION_DIR=""
ACTIVE_RESERVATION_FAILURE_MARKER=""
ACTIVE_RESERVATION_OPERATION=""


usage() {
    cat <<'USAGE'
Usage:
  launch_cosmos_mixed_step_8gpu.sh preflight REQUIRED_INPUTS
  launch_cosmos_mixed_step_8gpu.sh start {universe|s2|s1} REQUIRED_INPUTS
  launch_cosmos_mixed_step_8gpu.sh eval {universe|s2|s1} CHECKPOINT EVAL_INPUTS
  launch_cosmos_mixed_step_8gpu.sh status --root-base ROOT_BASE

Required training/preflight inputs (all paths must already exist):
  --root-base PATH              New output parent. The launcher never deletes it.
  --protocol-source-root PATH   Read-only protocol JSON plus t4/t8 fixed caches.
  --dataset-path PATH           Read-only LIBERO dataset path.
  --teacher-model-path PATH     Read-only Cosmos Policy model directory.
  --stage1-checkpoint PATH      Common Stage-1 online-student checkpoint.

Required evaluation inputs:
  --root-base PATH --protocol-source-root PATH --dataset-path PATH
  --teacher-model-path PATH
  --stage1-checkpoint PATH      Optional override; the default is still checked
                                as an immutable non-overlapping source.

Optional explicit execution settings:
  --python PATH                 Default: $PYTHON_BIN or the Any-WAM environment.
  --torchrun PATH               Default: $TORCHRUN_BIN or the Any-WAM environment.
  --device-list LIST            Default: 0,1,2,3,4,5,6,7 (must be exactly 8 unique IDs).
  --eval-device-list LIST       Default: 0.
  --eval-worker-device-list LIST Default: 0.
  --train-seed INTEGER          Default: 20260716.

`preflight` writes ROOT_BASE/PREFLIGHT_COMPLETE only after the 9-step Universe
run, target-free online checkpoint check, and all three fixed-cache proxy
evaluations have completed.  `start` schedules exactly one fresh policy; it
does not queue S4 or any later policy.  `eval` only invokes the runner's
eval-only S1/S2/S4 proxy path and never supplies a training command.
USAGE
}


die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}


require_directory() {
    local path="$1"
    [[ -n "$path" && -d "$path" ]] || die "Required directory is missing: $path"
}


require_program() {
    local program="$1"
    if [[ "$program" == */* ]]; then
        [[ -x "$program" ]] || die "Required executable is missing: $program"
    else
        command -v "$program" >/dev/null 2>&1 || die "Required command is missing: $program"
    fi
}


canonical_directory() {
    local path="$1"
    require_directory "$path"
    (cd "$path" && pwd -P)
}


require_marker() {
    local marker="$1"
    local purpose="$2"
    [[ -f "$marker" ]] || die "$purpose requires completion marker: $marker"
}


ensure_marker_absent() {
    local marker="$1"
    [[ ! -e "$marker" && ! -L "$marker" ]] || die "Refusing to overwrite existing marker: $marker"
}


assert_supported_policy() {
    local policy="$1"
    case "$policy" in
        universe|s2|s1)
            ;;
        *)
            die "Unsupported policy: $policy (expected universe, s2, or s1)"
            ;;
    esac
}


validate_eight_devices() {
    local device_list="$1"
    local -a devices=()
    local device=""
    IFS=',' read -r -a devices <<< "$device_list"
    (( ${#devices[@]} == 8 )) || die "--device-list must contain exactly 8 CUDA device IDs"
    declare -A seen=()
    for device in "${devices[@]}"; do
        [[ "$device" =~ ^[0-9]+$ ]] || die "CUDA device IDs must be numeric: $device"
        [[ -z "${seen[$device]+x}" ]] || die "CUDA device IDs must be unique: $device"
        seen["$device"]=1
    done
}


parse_options() {
    while (( $# > 0 )); do
        case "$1" in
            --root-base)
                ROOT_BASE="${2:-}"
                shift 2
                ;;
            --protocol-source-root)
                PROTOCOL_SOURCE_ROOT="${2:-}"
                shift 2
                ;;
            --dataset-path)
                DATASET_PATH="${2:-}"
                shift 2
                ;;
            --teacher-model-path)
                TEACHER_MODEL_PATH="${2:-}"
                shift 2
                ;;
            --stage1-checkpoint)
                STAGE1_CHECKPOINT="${2:-}"
                shift 2
                ;;
            --python)
                PYTHON_BIN="${2:-}"
                shift 2
                ;;
            --torchrun)
                TORCHRUN_BIN="${2:-}"
                shift 2
                ;;
            --device-list)
                DEVICE_LIST="${2:-}"
                shift 2
                ;;
            --eval-device-list)
                EVAL_DEVICE_LIST="${2:-}"
                shift 2
                ;;
            --eval-worker-device-list)
                EVAL_WORKER_DEVICE_LIST="${2:-}"
                shift 2
                ;;
            --train-seed)
                TRAIN_SEED="${2:-}"
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                die "Unknown option: $1"
                ;;
        esac
    done
}


validate_root_base() {
    ROOT_BASE="$(canonical_directory "$ROOT_BASE")"
}


assert_root_base_input_isolation() {
    local root_base="$1"
    shift
    local input_path=""
    for input_path in "$@"; do
        if [[ "$root_base" == "$input_path" || "$root_base" == "$input_path"/* || "$input_path" == "$root_base"/* ]]; then
            die "Output root base must not overlap immutable input: root=$root_base input=$input_path"
        fi
    done
}


resolve_path_maybe_missing() {
    realpath -m -- "$1"
}


assert_eval_artifact_paths_isolated() {
    local policy_root="$1"
    local protocol_source_root="$2"
    local dataset_path="$3"
    local teacher_model_path="$4"
    local stage1_checkpoint="$5"
    shift 5
    local canonical_policy_root
    canonical_policy_root="$(resolve_path_maybe_missing "$policy_root")"
    protocol_source_root="$(resolve_path_maybe_missing "$protocol_source_root")"
    dataset_path="$(resolve_path_maybe_missing "$dataset_path")"
    teacher_model_path="$(resolve_path_maybe_missing "$teacher_model_path")"
    stage1_checkpoint="$(resolve_path_maybe_missing "$stage1_checkpoint")"
    local artifact_path=""
    local resolved_artifact=""
    for artifact_path in "$@"; do
        resolved_artifact="$(resolve_path_maybe_missing "$artifact_path")"
        if [[ "$resolved_artifact" != "$canonical_policy_root" && "$resolved_artifact" != "$canonical_policy_root"/* ]]; then
            die "Evaluation artifact must resolve under policy root: artifact=$resolved_artifact root=$canonical_policy_root"
        fi
        assert_root_base_input_isolation \
            "$resolved_artifact" "$protocol_source_root" "$dataset_path" \
            "$teacher_model_path" "$stage1_checkpoint"
    done
}


validate_training_inputs() {
    validate_root_base
    PROTOCOL_SOURCE_ROOT="$(canonical_directory "$PROTOCOL_SOURCE_ROOT")"
    DATASET_PATH="$(canonical_directory "$DATASET_PATH")"
    TEACHER_MODEL_PATH="$(canonical_directory "$TEACHER_MODEL_PATH")"
    STAGE1_CHECKPOINT="$(canonical_directory "$STAGE1_CHECKPOINT")"
    assert_root_base_input_isolation \
        "$ROOT_BASE" "$PROTOCOL_SOURCE_ROOT" "$DATASET_PATH" \
        "$TEACHER_MODEL_PATH" "$STAGE1_CHECKPOINT"
    require_program "$PYTHON_BIN"
    require_program "$TORCHRUN_BIN"
    [[ -f "$RUNNER_PATH" ]] || die "Missing mixed-step runner: $RUNNER_PATH"
    [[ -f "$STAGE1_CHECKPOINT/online_student/transformer/config.json" ]] || die \
        "Stage-1 checkpoint has no online_student transformer config: $STAGE1_CHECKPOINT"
    validate_eight_devices "$DEVICE_LIST"
    [[ "$TRAIN_SEED" =~ ^[0-9]+$ ]] || die "--train-seed must be a non-negative integer"
    [[ -n "$EVAL_DEVICE_LIST" && -n "$EVAL_WORKER_DEVICE_LIST" ]] || die \
        "Evaluation device lists must be non-empty"
}


validate_eval_inputs() {
    validate_root_base
    PROTOCOL_SOURCE_ROOT="$(canonical_directory "$PROTOCOL_SOURCE_ROOT")"
    DATASET_PATH="$(canonical_directory "$DATASET_PATH")"
    TEACHER_MODEL_PATH="$(canonical_directory "$TEACHER_MODEL_PATH")"
    STAGE1_CHECKPOINT="$(canonical_directory "${STAGE1_CHECKPOINT:-$DEFAULT_STAGE1_CHECKPOINT}")"
    assert_root_base_input_isolation \
        "$ROOT_BASE" "$PROTOCOL_SOURCE_ROOT" "$DATASET_PATH" \
        "$TEACHER_MODEL_PATH" "$STAGE1_CHECKPOINT"
    require_program "$PYTHON_BIN"
    require_program realpath
    [[ -f "$RUNNER_PATH" ]] || die "Missing mixed-step runner: $RUNNER_PATH"
    [[ -n "$EVAL_DEVICE_LIST" && -n "$EVAL_WORKER_DEVICE_LIST" ]] || die \
        "Evaluation device lists must be non-empty"
}


write_marker_once() {
    local marker="$1"
    local operation="$2"
    local status="$3"
    local checkpoint_dir="${4:-}"
    [[ ! -e "$marker" && ! -L "$marker" ]] || return 1
    mkdir -p "$(dirname "$marker")"
    (
        set -o noclobber
        printf 'operation=%s\nstatus=%s\ncheckpoint=%s\nrecorded_at=%s\n' \
            "$operation" "$status" "$checkpoint_dir" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$marker"
    )
}


release_active_reservation() {
    if [[ -n "$ACTIVE_RESERVATION_DIR" && -d "$ACTIVE_RESERVATION_DIR" ]]; then
        if ! rmdir "$ACTIVE_RESERVATION_DIR"; then
            printf 'ERROR: leaving in-flight reservation for manual inspection: %s\n' \
                "$ACTIVE_RESERVATION_DIR" >&2
            return 1
        fi
    fi
    ACTIVE_RESERVATION_DIR=""
    ACTIVE_RESERVATION_FAILURE_MARKER=""
    ACTIVE_RESERVATION_OPERATION=""
}


launcher_exit_cleanup() {
    local exit_code=$?
    trap - EXIT
    if [[ -n "$ACTIVE_RESERVATION_DIR" ]]; then
        if ! write_marker_once "$ACTIVE_RESERVATION_FAILURE_MARKER" \
            "$ACTIVE_RESERVATION_OPERATION" "launcher_setup_failed" ""; then
            printf 'ERROR: unable to create launcher failure marker (or marker already exists): %s\n' \
                "$ACTIVE_RESERVATION_FAILURE_MARKER" >&2
        fi
        release_active_reservation || true
    fi
    exit "$exit_code"
}


reserve_operation() {
    local root_base="$1"
    local operation_id="$2"
    local failure_marker="$3"
    [[ "$operation_id" =~ ^[a-z0-9_-]+$ ]] || die "Invalid reservation operation identity: $operation_id"
    local reservation_root="$root_base/$RESERVATION_ROOT_NAME"
    local reservation_dir="$reservation_root/$operation_id"
    mkdir -p "$reservation_root"
    if ! mkdir "$reservation_dir"; then
        die "Operation already has an in-flight reservation: $operation_id ($reservation_dir)"
    fi
    ACTIVE_RESERVATION_DIR="$reservation_dir"
    ACTIVE_RESERVATION_FAILURE_MARKER="$failure_marker"
    ACTIVE_RESERVATION_OPERATION="$operation_id"
}


transfer_active_reservation_to_worker() {
    ACTIVE_RESERVATION_DIR=""
    ACTIVE_RESERVATION_FAILURE_MARKER=""
    ACTIVE_RESERVATION_OPERATION=""
}


new_run_tag() {
    date -u +%Y%m%dT%H%M%SZ-"$$"
}


prepare_artifact_paths() {
    local root_base="$1"
    local session_name="$2"
    local log_dir="$root_base/logs"
    LOG_FILE="$log_dir/${session_name}.log"
    WORKER_SCRIPT="$log_dir/${session_name}.worker.sh"
    [[ ! -e "$LOG_FILE" && ! -L "$LOG_FILE" ]] || die "Refusing to overwrite log: $LOG_FILE"
    [[ ! -e "$WORKER_SCRIPT" && ! -L "$WORKER_SCRIPT" ]] || die \
        "Refusing to overwrite worker script: $WORKER_SCRIPT"
    mkdir -p "$log_dir"
}


write_worker_script() {
    local worker_script="$1"
    local log_file="$2"
    local success_marker="$3"
    local failure_marker="$4"
    local reservation_dir="$5"
    local operation="$6"
    local checkpoint_dir="$7"
    local selection_proxy="$8"
    shift 8
    local -a runner_argv=("$@")

    ensure_marker_absent "$success_marker"
    ensure_marker_absent "$failure_marker"
    [[ ! -e "$worker_script" && ! -L "$worker_script" ]] || die \
        "Refusing to overwrite worker script: $worker_script"
    [[ ! -e "$log_file" && ! -L "$log_file" ]] || die "Refusing to overwrite log: $log_file"
    (
        set -o noclobber
        : > "$log_file"
    )
    umask 077
    (
        set -o noclobber
        {
        printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail'
        printf 'LOG_FILE=%q\n' "$log_file"
        printf 'SUCCESS_MARKER=%q\n' "$success_marker"
        printf 'FAILURE_MARKER=%q\n' "$failure_marker"
        printf 'RESERVATION_DIR=%q\n' "$reservation_dir"
        printf 'OPERATION=%q\n' "$operation"
        printf 'CHECKPOINT_DIR=%q\n' "$checkpoint_dir"
        printf 'SELECTION_PROXY=%q\n' "$selection_proxy"
        cat <<'WORKER_HEADER'
write_failure_marker() {
    if [[ ! -e "$FAILURE_MARKER" && ! -L "$FAILURE_MARKER" ]]; then
        mkdir -p "$(dirname "$FAILURE_MARKER")"
        (
            set -o noclobber
            printf 'operation=%s\nstatus=worker_failed\nfinished_at=%s\n' \
                "$OPERATION" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$FAILURE_MARKER"
        ) || printf 'unable to create worker failure marker: %s\n' "$FAILURE_MARKER" >&2
    fi
}
release_reservation() {
    if ! rmdir "$RESERVATION_DIR"; then
        printf 'leaving in-flight reservation for manual inspection: %s\n' "$RESERVATION_DIR" >&2
    fi
}
on_exit() {
    local exit_code=$?
    if (( exit_code != 0 )); then
        write_failure_marker
    fi
    release_reservation
}
trap 'exit 130' INT
trap 'exit 143' TERM HUP
trap on_exit EXIT
exec >>"$LOG_FILE" 2>&1
printf 'starting %s at %s\n' "$OPERATION" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
WORKER_HEADER
        printf 'cd %q\n' "$REPO_ROOT"
        printf '%q ' "${runner_argv[@]}"
        printf '\n'
        cat <<'WORKER_VALIDATION'
if [[ ! -f "$CHECKPOINT_DIR/online_student/transformer/config.json" ]]; then
    printf 'missing online_student transformer after %s: %s\n' "$OPERATION" "$CHECKPOINT_DIR" >&2
    exit 1
fi
if [[ -e "$CHECKPOINT_DIR/target_student" || -L "$CHECKPOINT_DIR/target_student" ]]; then
    printf 'target_student is forbidden for target-free %s: %s\n' "$OPERATION" "$CHECKPOINT_DIR" >&2
    exit 1
fi
if [[ ! -f "$SELECTION_PROXY" ]]; then
    printf 'missing all-budget selection proxy after %s: %s\n' "$OPERATION" "$SELECTION_PROXY" >&2
    exit 1
fi
for budget in s1 s2 s4; do
    if [[ ! -f "${SELECTION_PROXY%/selection_proxy.json}/${budget}.json" ]]; then
        printf 'missing %s fixed-cache proxy result after %s\n' "$budget" "$OPERATION" >&2
        exit 1
    fi
done
if [[ -e "$SUCCESS_MARKER" || -L "$SUCCESS_MARKER" ]]; then
    printf 'refusing to overwrite success marker: %s\n' "$SUCCESS_MARKER" >&2
    exit 1
fi
mkdir -p "$(dirname "$SUCCESS_MARKER")"
(
    set -o noclobber
    printf 'operation=%s\ncheckpoint=%s\ncompleted_at=%s\n' \
        "$OPERATION" "$CHECKPOINT_DIR" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$SUCCESS_MARKER"
) || exit 1
printf 'completed %s at %s\n' "$OPERATION" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
WORKER_VALIDATION
        } > "$worker_script"
    )
    chmod 700 "$worker_script"
}


launch_tmux_session() {
    local session_name="$1"
    local worker_script="$2"
    require_program "$TMUX_BIN"
    if "$TMUX_BIN" has-session -t "$session_name" 2>/dev/null; then
        die "Refusing to reuse active tmux session: $session_name"
    fi
    "$TMUX_BIN" new-session -d -s "$session_name" "$worker_script"
    printf 'Created tmux session: %s\nLog: %s\nWorker: %s\n' \
        "$session_name" "$LOG_FILE" "$worker_script"
}


show_status() {
    local root_base="$1"
    printf 'Cosmos mixed-step launcher status for %s\n' "$root_base"
    for marker in \
        "$root_base/PREFLIGHT_COMPLETE" \
        "$root_base/PREFLIGHT_FAILED" \
        "$root_base/universe/TRAINING_COMPLETE" \
        "$root_base/universe/TRAINING_FAILED" \
        "$root_base/s2/TRAINING_COMPLETE" \
        "$root_base/s2/TRAINING_FAILED" \
        "$root_base/s1/TRAINING_COMPLETE" \
        "$root_base/s1/TRAINING_FAILED"; do
        if [[ -e "$marker" || -L "$marker" ]]; then
            printf 'marker: %s\n' "$marker"
        fi
    done
    local reservation=""
    local reservation_root="$root_base/$RESERVATION_ROOT_NAME"
    if [[ -d "$reservation_root" ]]; then
        shopt -s nullglob
        for reservation in "$reservation_root"/*; do
            if [[ -d "$reservation" ]]; then
                printf 'in-flight reservation: %s\n' "$reservation"
            fi
        done
        shopt -u nullglob
    fi
    if command -v "$TMUX_BIN" >/dev/null 2>&1; then
        "$TMUX_BIN" list-sessions 2>/dev/null || true
    else
        printf 'tmux command unavailable: %s\n' "$TMUX_BIN" >&2
    fi
    if [[ -d "$root_base/logs" ]]; then
        local log_file=""
        shopt -s nullglob
        for log_file in "$root_base"/logs/*.log; do
            printf '\n--- tail: %s ---\n' "$log_file"
            tail -n 20 "$log_file" || true
        done
        shopt -u nullglob
    fi
}


start_preflight() {
    validate_training_inputs
    ensure_marker_absent "$ROOT_BASE/PREFLIGHT_COMPLETE"
    ensure_marker_absent "$ROOT_BASE/PREFLIGHT_FAILED"
    local run_tag
    run_tag="$(new_run_tag)"
    local preflight_root="$ROOT_BASE/preflight-universe-${run_tag}"
    local session_name="cosmos-mixed-preflight-universe-${run_tag}"
    [[ ! -e "$preflight_root" && ! -L "$preflight_root" ]] || die \
        "Refusing to reuse preflight root: $preflight_root"
    reserve_operation "$ROOT_BASE" "preflight" "$ROOT_BASE/PREFLIGHT_FAILED"
    prepare_artifact_paths "$ROOT_BASE" "$session_name"
    local checkpoint_dir="$preflight_root/checkpoints/step_9"
    local selection_proxy="$preflight_root/metrics/selection/step_9/selection_proxy.json"
    local -a runner_argv=(
        "$PYTHON_BIN" "$RUNNER_PATH"
        "--policy" "universe"
        "--root" "$preflight_root"
        "--dataset-path" "$DATASET_PATH"
        "--protocol-source-root" "$PROTOCOL_SOURCE_ROOT"
        "--stage1-checkpoint" "$STAGE1_CHECKPOINT"
        "--max-train-steps" "9"
        "--save-interval" "9"
        "--stop-after-step" "9"
        "--current-step" "0"
        "--chunk-size" "9"
        "--master-port" "$PREFLIGHT_PORT"
        "--train-seed" "$TRAIN_SEED"
        "--torchrun" "$TORCHRUN_BIN"
        "--teacher-model-path" "$TEACHER_MODEL_PATH"
        "--device-list" "$DEVICE_LIST"
        "--world-size" "8"
        "--force-sequence" "$PREFLIGHT_FORCE_SEQUENCE"
        "--eval-device-list" "$EVAL_DEVICE_LIST"
        "--eval-worker-device-list" "$EVAL_WORKER_DEVICE_LIST"
        "--run"
    )
    write_worker_script "$WORKER_SCRIPT" "$LOG_FILE" \
        "$ROOT_BASE/PREFLIGHT_COMPLETE" "$ROOT_BASE/PREFLIGHT_FAILED" \
        "$ACTIVE_RESERVATION_DIR" "universe-preflight" "$checkpoint_dir" "$selection_proxy" \
        "${runner_argv[@]}"
    launch_tmux_session "$session_name" "$WORKER_SCRIPT"
    transfer_active_reservation_to_worker
}


start_policy() {
    local policy="$1"
    assert_supported_policy "$policy"
    validate_training_inputs
    local root_base="$ROOT_BASE"
    case "$policy" in
        universe)
            require_marker "$root_base/PREFLIGHT_COMPLETE" "Universe start"
            ;;
        s2)
            require_marker "$root_base/universe/TRAINING_COMPLETE" "S2 start"
            ;;
        s1)
            require_marker "$root_base/s2/TRAINING_COMPLETE" "S1 start"
            ;;
    esac
    local policy_root="$root_base/$policy"
    [[ ! -e "$policy_root" && ! -L "$policy_root" ]] || die \
        "Refusing fresh $policy start because policy root already exists: $policy_root"
    reserve_operation "$root_base" "policy-${policy}" "$policy_root/TRAINING_FAILED"
    local run_tag
    run_tag="$(new_run_tag)"
    local session_name="cosmos-mixed-${policy}-${run_tag}"
    prepare_artifact_paths "$root_base" "$session_name"
    local checkpoint_dir="$policy_root/checkpoints/step_${TRAIN_STEPS}"
    local selection_proxy="$policy_root/metrics/selection/step_${TRAIN_STEPS}/selection_proxy.json"
    local -a runner_argv=(
        "$PYTHON_BIN" "$RUNNER_PATH"
        "--policy" "$policy"
        "--root" "$policy_root"
        "--dataset-path" "$DATASET_PATH"
        "--protocol-source-root" "$PROTOCOL_SOURCE_ROOT"
        "--stage1-checkpoint" "$STAGE1_CHECKPOINT"
        "--max-train-steps" "$TRAIN_STEPS"
        "--save-interval" "$TRAIN_SAVE_INTERVAL"
        "--current-step" "0"
        "--master-port" "${POLICY_PORTS[$policy]}"
        "--train-seed" "$TRAIN_SEED"
        "--torchrun" "$TORCHRUN_BIN"
        "--teacher-model-path" "$TEACHER_MODEL_PATH"
        "--device-list" "$DEVICE_LIST"
        "--world-size" "8"
        "--eval-device-list" "$EVAL_DEVICE_LIST"
        "--eval-worker-device-list" "$EVAL_WORKER_DEVICE_LIST"
        "--run"
    )
    write_worker_script "$WORKER_SCRIPT" "$LOG_FILE" \
        "$policy_root/TRAINING_COMPLETE" "$policy_root/TRAINING_FAILED" \
        "$ACTIVE_RESERVATION_DIR" "${policy}-full-5000" "$checkpoint_dir" "$selection_proxy" \
        "${runner_argv[@]}"
    launch_tmux_session "$session_name" "$WORKER_SCRIPT"
    transfer_active_reservation_to_worker
}


start_eval() {
    local policy="$1"
    local checkpoint="$2"
    assert_supported_policy "$policy"
    validate_eval_inputs
    local root_base="$ROOT_BASE"
    local policy_root
    policy_root="$(canonical_directory "$root_base/$policy")"
    assert_root_base_input_isolation \
        "$policy_root" "$PROTOCOL_SOURCE_ROOT" "$DATASET_PATH" \
        "$TEACHER_MODEL_PATH" "$STAGE1_CHECKPOINT"
    checkpoint="$(canonical_directory "$checkpoint")"
    [[ "$(dirname "$checkpoint")" == "$policy_root/checkpoints" ]] || die \
        "Evaluation checkpoint must be directly under $policy_root/checkpoints"
    [[ "$(basename "$checkpoint")" =~ ^step_[0-9]+$ ]] || die \
        "Evaluation checkpoint must have a step_N directory name: $checkpoint"
    [[ -f "$policy_root/policy_manifest.json" ]] || die \
        "Evaluation requires the policy's provenance manifest: $policy_root/policy_manifest.json"
    local checkpoint_label
    checkpoint_label="$(basename "$checkpoint")"
    local metrics_dir="$policy_root/metrics/selection/$checkpoint_label"
    local selection_proxy="$metrics_dir/selection_proxy.json"
    local eval_complete_marker="$metrics_dir/EVAL_COMPLETE"
    local eval_failed_marker="$metrics_dir/EVAL_FAILED"
    assert_eval_artifact_paths_isolated \
        "$policy_root" "$PROTOCOL_SOURCE_ROOT" "$DATASET_PATH" \
        "$TEACHER_MODEL_PATH" "$STAGE1_CHECKPOINT" \
        "$metrics_dir" "$selection_proxy" \
        "$metrics_dir/s1.json" "$metrics_dir/s2.json" "$metrics_dir/s4.json" \
        "$eval_complete_marker" "$eval_failed_marker"
    ensure_marker_absent "$eval_complete_marker"
    ensure_marker_absent "$eval_failed_marker"
    [[ ! -e "$selection_proxy" && ! -L "$selection_proxy" ]] || die \
        "Refusing to overwrite existing selection proxy: $selection_proxy"
    local budget=""
    for budget in s1 s2 s4; do
        [[ ! -e "$metrics_dir/${budget}.json" && ! -L "$metrics_dir/${budget}.json" ]] || die \
            "Refusing to overwrite existing fixed-cache proxy result: $metrics_dir/${budget}.json"
    done
    reserve_operation "$root_base" "eval-${policy}-${checkpoint_label}" "$eval_failed_marker"
    local run_tag
    run_tag="$(new_run_tag)"
    local session_name="cosmos-mixed-eval-${policy}-${checkpoint_label}-${run_tag}"
    prepare_artifact_paths "$root_base" "$session_name"
    local -a runner_argv=(
        "$PYTHON_BIN" "$RUNNER_PATH"
        "--policy" "$policy"
        "--root" "$policy_root"
        "--dataset-path" "$DATASET_PATH"
        "--protocol-source-root" "$PROTOCOL_SOURCE_ROOT"
        "--stage1-checkpoint" "$STAGE1_CHECKPOINT"
        "--teacher-model-path" "$TEACHER_MODEL_PATH"
        "--eval-device-list" "$EVAL_DEVICE_LIST"
        "--eval-worker-device-list" "$EVAL_WORKER_DEVICE_LIST"
        "--eval-checkpoint" "$checkpoint"
        "--run"
    )
    write_worker_script "$WORKER_SCRIPT" "$LOG_FILE" \
        "$eval_complete_marker" "$eval_failed_marker" \
        "$ACTIVE_RESERVATION_DIR" "${policy}-offline-proxy-${checkpoint_label}" \
        "$checkpoint" "$selection_proxy" \
        "${runner_argv[@]}"
    launch_tmux_session "$session_name" "$WORKER_SCRIPT"
    transfer_active_reservation_to_worker
}


main() {
    local command="${1:-}"
    case "$command" in
        preflight)
            shift
            parse_options "$@"
            start_preflight
            ;;
        start)
            local policy="${2:-}"
            [[ -n "$policy" ]] || die "start requires a policy name"
            shift 2
            parse_options "$@"
            start_policy "$policy"
            ;;
        eval)
            local policy="${2:-}"
            local checkpoint="${3:-}"
            [[ -n "$policy" && -n "$checkpoint" ]] || die \
                "eval requires a policy name and checkpoint directory"
            shift 3
            parse_options "$@"
            start_eval "$policy" "$checkpoint"
            ;;
        status)
            shift
            parse_options "$@"
            validate_root_base
            show_status "$ROOT_BASE"
            ;;
        --help|-h|help)
            usage
            ;;
        "")
            usage >&2
            exit 2
            ;;
        *)
            die "Unknown command: $command"
            ;;
    esac
}


trap launcher_exit_cleanup EXIT
main "$@"
