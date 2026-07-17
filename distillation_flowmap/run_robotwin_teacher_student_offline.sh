#!/usr/bin/env bash
# Guarded, offline-only RobotWin teacher/student metric matrix.
#
# This launcher intentionally does nothing but print a reproducible JSON plan
# unless --run is supplied.  It never starts a RobotWin environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
CANONICAL_ANY_WAM_ROOT="/root/nas/junjie/jj/Any_WAM"
TRAINING_ARTIFACT_ROOT="${CANONICAL_ANY_WAM_ROOT}/distillation_flowmap/output_robotwin_stepwam_ablation"

PYTHON="/root/nas/junjie/conda_envs/any_wam/bin/python"
TORCHRUN="/root/nas/junjie/conda_envs/any_wam/bin/torchrun"
EVALUATOR="${REPO_ROOT}/distillation_flowmap/rollout_eval_video_stage2.py"
SUMMARIZER="${REPO_ROOT}/distillation_flowmap/summarize_robotwin_teacher_student.py"

DATASET_ROOT="/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500"
EMPTY_EMB_PATH="/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/empty_emb.pt"
TEACHER_MODEL_PATH="/root/nas/junjie/jj/Any_WAM/checkpoints/lingbot-va-posttrain-robotwin"

DANCE_ROOT="/root/nas/junjie/jj/Any_WAM/distillation_flowmap/output_robotwin_stepwam_ablation/protocol_final12_danceopd_i1_single_seed_v1"
FULL_ROOT="/root/nas/junjie/jj/Any_WAM/distillation_flowmap/output_robotwin_stepwam_ablation/protocol_final12_5000"
CORE2_ROOT="/root/nas/junjie/jj/Any_WAM/distillation_flowmap/archive_robotwin_stepwam_ablation_nonfinal_20260716_114435/protocol_dance_endpoint_2x2_seed0_v1"

DANCE_CHECKPOINT="${DANCE_ROOT}/final_stepwam_danceopd/seed_0/stage2/checkpoints/step_5000"
FULL_CHECKPOINT="${FULL_ROOT}/full_stepwam/seed_0/stage2/checkpoints/step_5000"
CORE2_PROTOCOL_DIR="${CORE2_ROOT}/protocol/manifests/core2_protocol_seed_0"
DANCE_FINAL12_PROTOCOL_DIR="${DANCE_ROOT}/protocol/manifests/representative_protocol_seed_0"
FULL_FINAL12_PROTOCOL_DIR="${FULL_ROOT}/protocol/manifests/representative_protocol_seed_0"

RUN=0
SPLIT="core2"
CHECKPOINT_SELECTION="all"
SOURCE_SELECTION="all"
K_SELECTION="all"
OUTPUT_ROOT=""
CORE2_GATE_PLAN=""
CORE2_GATE_PLAN_SHA256=""
REQUESTED_DATASET_ROOT="${DATASET_ROOT}"
CUDA_VISIBLE_DEVICES_VALUE="0"
MASTER_PORT_BASE="39200"

usage() {
  cat <<'USAGE'
Usage:
  distillation_flowmap/run_robotwin_teacher_student_offline.sh \
    --output-root /absolute/fresh/output/root [options]

Default mode is plan-only: it writes no output directory and prints a JSON plan
to stdout.  Redirect that JSON to a review file before using --run.

Options:
  --run                         Execute the planned offline evaluator commands.
  --split core2|final12          Fixed held-out protocol (default: core2).
  --checkpoint danceopd|full_stepwam|all
                                Fixed Stage-2 step_5000 ablation checkpoint(s).
  --source target|online|all     Target is primary; online is labelled supplemental.
  --k 1|2|4|all                  Equal teacher/student NFE budget(s).
  --output-root PATH             Required fresh absolute output root.
  --core2-gate PATH              Required for --run --split final12. Must be an
                                 executed Core2 plan covering every selected
                                 checkpoint/source/K entry.
  --dataset-path PATH            Must equal the fixed RobotWin ablation dataset root.
  --cuda-visible-devices GPU     One GPU ordinal used only with --run (default: 0).
  --master-port-base PORT        Base port for sequential single-process torchrun.
  -h, --help                     Show this message.

The plan uses rollout_eval_video_stage2.py without --video-dir, so it records
video latent metrics and teacher/student action metrics without decoding video.
USAGE
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

require_value() {
  [ "$#" -ge 2 ] || die "$1 requires a value"
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

inventory_sha256() {
  local path="$1"
  (
    cd "$path"
    find -P . -type f -printf '%P\t%s\t%T@\n' | LC_ALL=C sort | sha256sum | awk '{print $1}'
  )
}

assert_exists() {
  local path="$1"
  [ -e "$path" ] || die "required protocol input is missing: $path"
}

assert_not_protected_output_path() {
  local candidate="$1" protected
  for protected in \
    "$REPO_ROOT" "$CANONICAL_ANY_WAM_ROOT" "$TRAINING_ARTIFACT_ROOT" \
    "$DATASET_ROOT" "$TEACHER_MODEL_PATH" "$DANCE_ROOT" "$FULL_ROOT" "$CORE2_ROOT"; do
    case "$candidate" in
      "$protected"|"$protected"/*) die "--output-root cannot be inside an immutable code, dataset, teacher, checkpoint, or protocol root" ;;
    esac
  done
}

assert_fresh_output_root() {
  local parent base parent_real
  [ -n "$OUTPUT_ROOT" ] || die "--output-root is required"
  case "$OUTPUT_ROOT" in
    /*) ;;
    *) die "--output-root must be an absolute path" ;;
  esac
  case "$OUTPUT_ROOT" in
    *$'\n'*|*$'\r'*|*$'\t'*|*$'\037'*) die "--output-root contains an unsupported control character" ;;
  esac
  assert_not_protected_output_path "$OUTPUT_ROOT"
  parent="$(dirname "$OUTPUT_ROOT")"
  base="$(basename "$OUTPUT_ROOT")"
  [ "$base" != "." ] && [ "$base" != ".." ] || die "--output-root must name a new directory"
  [ -d "$parent" ] || die "parent directory for --output-root does not exist: $parent"
  parent_real="$(cd "$parent" && pwd -P)"
  OUTPUT_ROOT="${parent_real}/${base}"
  [ ! -e "$OUTPUT_ROOT" ] || die "--output-root must be a fresh, non-existent directory: $OUTPUT_ROOT"
  assert_not_protected_output_path "$OUTPUT_ROOT"
}

resolve_core2_gate_plan() {
  local gate_parent gate_name
  [ -n "$CORE2_GATE_PLAN" ] || return 0
  [ "$SPLIT" = "final12" ] || die "--core2-gate is only valid with --split final12"
  case "$CORE2_GATE_PLAN" in
    /*) ;;
    *) die "--core2-gate must be an absolute plan.json path" ;;
  esac
  [ -f "$CORE2_GATE_PLAN" ] || die "--core2-gate must name an existing Core2 plan: $CORE2_GATE_PLAN"
  gate_parent="$(dirname "$CORE2_GATE_PLAN")"
  gate_name="$(basename "$CORE2_GATE_PLAN")"
  CORE2_GATE_PLAN="$(cd "$gate_parent" && pwd -P)/${gate_name}"
  CORE2_GATE_PLAN_SHA256="$(sha256_file "$CORE2_GATE_PLAN")"
}

expand_checkpoints() {
  case "$CHECKPOINT_SELECTION" in
    danceopd) CHECKPOINTS=(danceopd) ;;
    full_stepwam) CHECKPOINTS=(full_stepwam) ;;
    all) CHECKPOINTS=(danceopd full_stepwam) ;;
    *) die "--checkpoint must be danceopd, full_stepwam, or all" ;;
  esac
}

expand_sources() {
  case "$SOURCE_SELECTION" in
    target) SOURCES=(target) ;;
    online) SOURCES=(online) ;;
    all) SOURCES=(target online) ;;
    *) die "--source must be target, online, or all" ;;
  esac
}

expand_steps() {
  case "$K_SELECTION" in
    1|2|4) STEPS=("$K_SELECTION") ;;
    all) STEPS=(1 2 4) ;;
    *) die "--k must be 1, 2, 4, or all" ;;
  esac
}

resolve_checkpoint() {
  local name="$1"
  case "$name" in
    danceopd) printf '%s\n' "$DANCE_CHECKPOINT" ;;
    full_stepwam) printf '%s\n' "$FULL_CHECKPOINT" ;;
    *) die "unknown fixed checkpoint mapping: $name" ;;
  esac
}

resolve_protocol_dir() {
  local checkpoint_name="$1"
  case "$SPLIT" in
    core2) printf '%s\n' "$CORE2_PROTOCOL_DIR" ;;
    final12)
      case "$checkpoint_name" in
        danceopd) printf '%s\n' "$DANCE_FINAL12_PROTOCOL_DIR" ;;
        full_stepwam) printf '%s\n' "$FULL_FINAL12_PROTOCOL_DIR" ;;
        *) die "unknown final12 checkpoint mapping: $checkpoint_name" ;;
      esac
      ;;
    *) die "--split must be core2 or final12" ;;
  esac
}

resolve_protocol_note() {
  case "$SPLIT" in
    core2) printf '%s\n' "archive_core2_subset_diagnostic" ;;
    final12) printf '%s\n' "checkpoint_owned_final12_heldout" ;;
    *) die "--split must be core2 or final12" ;;
  esac
}

build_entries() {
  local checkpoint_name checkpoint_path protocol_dir protocol_note source source_subdir
  local source_env step manifest_path pair_path checkpoint_inventory source_inventory
  local manifest_digest pair_digest entry_dir result_path log_path evaluator_output port
  local index=0
  ENTRY_ROWS=()

  for checkpoint_name in "${CHECKPOINTS[@]}"; do
    checkpoint_path="$(resolve_checkpoint "$checkpoint_name")"
    protocol_dir="$(resolve_protocol_dir "$checkpoint_name")"
    protocol_note="$(resolve_protocol_note)"
    manifest_path="${protocol_dir}/heldout_eval_manifest.json"
    pair_path="${protocol_dir}/eval_pairs.json"
    assert_exists "$checkpoint_path"
    assert_exists "$manifest_path"
    assert_exists "$pair_path"
    checkpoint_inventory="$(inventory_sha256 "$checkpoint_path")"
    manifest_digest="$(sha256_file "$manifest_path")"
    pair_digest="$(sha256_file "$pair_path")"

    for source in "${SOURCES[@]}"; do
      case "$source" in
        target)
          source_subdir="target_student/transformer"
          source_env=1
          ;;
        online)
          source_subdir="online_student/transformer"
          source_env=0
          ;;
        *) die "unknown rollout source: $source" ;;
      esac
      assert_exists "${checkpoint_path}/${source_subdir}"
      source_inventory="$(inventory_sha256 "${checkpoint_path}/${source_subdir}")"

      for step in "${STEPS[@]}"; do
        port=$((MASTER_PORT_BASE + index))
        [ "$port" -le 65535 ] || die "master port range exceeds 65535"
        entry_dir="${OUTPUT_ROOT}/${SPLIT}/${checkpoint_name}/${source}/k${step}"
        result_path="${entry_dir}/result.json"
        log_path="${entry_dir}/evaluator.log"
        evaluator_output="${entry_dir}/evaluator_output"
        ENTRY_ROWS+=("${SPLIT}"$'\037'"${protocol_note}"$'\037'"${checkpoint_name}"$'\037'"${checkpoint_path}"$'\037'"${checkpoint_inventory}"$'\037'"${source}"$'\037'"${source_subdir}"$'\037'"${source_inventory}"$'\037'"${source_env}"$'\037'"${step}"$'\037'"${manifest_path}"$'\037'"${manifest_digest}"$'\037'"${pair_path}"$'\037'"${pair_digest}"$'\037'"${entry_dir}"$'\037'"${result_path}"$'\037'"${log_path}"$'\037'"${evaluator_output}"$'\037'"${port}")
        index=$((index + 1))
      done
    done
  done
}

emit_plan() {
  local destination="$1"
  local joined
  joined="$(printf '%s\n' "${ENTRY_ROWS[@]}")"
  PLAN_ENTRY_ROWS="$joined" \
  PLAN_DESTINATION="$destination" \
  PLAN_OUTPUT_ROOT="$OUTPUT_ROOT" \
  PLAN_SPLIT="$SPLIT" \
  PLAN_CHECKPOINT_SELECTION="$CHECKPOINT_SELECTION" \
  PLAN_SOURCE_SELECTION="$SOURCE_SELECTION" \
  PLAN_K_SELECTION="$K_SELECTION" \
  PLAN_CORE2_GATE_PLAN="$CORE2_GATE_PLAN" \
  PLAN_CORE2_GATE_PLAN_SHA256="$CORE2_GATE_PLAN_SHA256" \
  PLAN_EXECUTION_REQUESTED="$RUN" \
  PLAN_DATASET_ROOT="$DATASET_ROOT" \
  PLAN_EMPTY_EMB_PATH="$EMPTY_EMB_PATH" \
  PLAN_TEACHER_MODEL_PATH="$TEACHER_MODEL_PATH" \
  PLAN_TORCHRUN="$TORCHRUN" \
  PLAN_EVALUATOR="$EVALUATOR" \
  PLAN_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
  "$PYTHON" - <<'PY'
import json
import os
import shlex
import sys
from pathlib import Path

separator = "\x1f"
field_names = (
    "split",
    "protocol_note",
    "checkpoint_name",
    "checkpoint_path",
    "checkpoint_inventory_sha256",
    "rollout_source",
    "student_checkpoint_subdir",
    "student_checkpoint_inventory_sha256",
    "resume_online_from_target",
    "student_steps",
    "manifest_path",
    "manifest_sha256",
    "pair_path",
    "pair_sha256",
    "entry_dir",
    "expected_raw_result",
    "log_path",
    "evaluator_output_dir",
    "master_port",
)

entries = []
for line in os.environ["PLAN_ENTRY_ROWS"].splitlines():
    fields = line.split(separator)
    if len(fields) != len(field_names):
        raise RuntimeError("internal error: malformed offline plan entry")
    item = dict(zip(field_names, fields))
    for number_name in ("student_steps", "master_port", "resume_online_from_target"):
        item[number_name] = int(item[number_name])
    item["teacher_steps"] = item["student_steps"]
    item["dataset_path"] = os.environ["PLAN_DATASET_ROOT"]
    item["teacher_model_path"] = os.environ["PLAN_TEACHER_MODEL_PATH"]
    item["empty_emb_path"] = os.environ["PLAN_EMPTY_EMB_PATH"]
    item["video_decode_enabled"] = False
    item["teacher_action_gt_enabled"] = True
    item["teacher_cache_path"] = None
    command_argv = [
        os.environ["PLAN_TORCHRUN"],
        "--nproc_per_node=1",
        "--master_port",
        str(item["master_port"]),
        os.environ["PLAN_EVALUATOR"],
        "--config",
        "distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow",
        "--teacher-model-path",
        item["teacher_model_path"],
        "--dataset-path",
        item["dataset_path"],
        "--empty-emb-path",
        item["empty_emb_path"],
        "--output-dir",
        item["evaluator_output_dir"],
        "--resume-from-path",
        item["checkpoint_path"],
        "--result-json",
        item["expected_raw_result"],
        "--eval-manifest",
        item["manifest_path"],
        "--eval-pairs-json",
        item["pair_path"],
        "--split-name",
        "heldout",
        "--num-batches",
        "0",
        "--student-steps",
        str(item["student_steps"]),
        "--teacher-steps",
        str(item["teacher_steps"]),
        "--rollout-student",
        item["rollout_source"],
        "--emit-teacher-action-gt",
        "--disable-eval-gradient-checkpointing",
        "--disable-eval-force-cfg",
        "--eval-rollout-grad-mode",
        "endpoint",
        "--eval-empty-cache",
    ]
    command_env = {
        "CUDA_VISIBLE_DEVICES": os.environ["PLAN_CUDA_VISIBLE_DEVICES"],
        "RESUME_ONLINE_FROM_TARGET": str(item["resume_online_from_target"]),
    }
    item["command_env"] = command_env
    item["command_argv"] = command_argv
    item["command"] = shlex.join(
        ["env", *(f"{key}={value}" for key, value in command_env.items()), *command_argv]
    )
    entries.append(item)

plan = {
    "schema": "robotwin_teacher_student_offline_plan_v1",
    "execution_requested": os.environ["PLAN_EXECUTION_REQUESTED"] == "1",
    "output_root": os.environ["PLAN_OUTPUT_ROOT"],
    "selection": {
        "split": os.environ["PLAN_SPLIT"],
        "checkpoint": os.environ["PLAN_CHECKPOINT_SELECTION"],
        "source": os.environ["PLAN_SOURCE_SELECTION"],
        "k": os.environ["PLAN_K_SELECTION"],
    },
    "core2_gate_plan": os.environ["PLAN_CORE2_GATE_PLAN"] or None,
    "core2_gate_plan_sha256": os.environ["PLAN_CORE2_GATE_PLAN_SHA256"] or None,
    "entries": entries,
}
payload = json.dumps(plan, indent=2, sort_keys=True) + "\n"
destination = os.environ["PLAN_DESTINATION"]
if destination == "-":
    sys.stdout.write(payload)
else:
    path = Path(destination)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing plan: {path}")
    path.write_text(payload, encoding="utf-8")
PY
}

run_entries() {
  local row split protocol_note checkpoint_name checkpoint_path checkpoint_inventory source
  local source_subdir source_inventory source_env step manifest_path manifest_digest pair_path
  local pair_digest entry_dir result_path log_path evaluator_output port
  local old_ifs="$IFS"
  IFS=$'\037'
  for row in "${ENTRY_ROWS[@]}"; do
    read -r split protocol_note checkpoint_name checkpoint_path checkpoint_inventory source \
      source_subdir source_inventory source_env step manifest_path manifest_digest pair_path \
      pair_digest entry_dir result_path log_path evaluator_output port <<< "$row"
    [ ! -e "$result_path" ] || die "refusing to overwrite planned raw result: $result_path"
    [ ! -e "$log_path" ] || die "refusing to overwrite planned evaluator log: $log_path"
    mkdir -p "$evaluator_output"
    printf 'Running offline metric-only evaluator: split=%s checkpoint=%s source=%s K=%s\n' \
      "$split" "$checkpoint_name" "$source" "$step" >&2
    env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}" \
      "RESUME_ONLINE_FROM_TARGET=${source_env}" \
      "$TORCHRUN" \
      --nproc_per_node=1 \
      --master_port "$port" \
      "$EVALUATOR" \
      --config distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow \
      --teacher-model-path "$TEACHER_MODEL_PATH" \
      --dataset-path "$DATASET_ROOT" \
      --empty-emb-path "$EMPTY_EMB_PATH" \
      --output-dir "$evaluator_output" \
      --resume-from-path "$checkpoint_path" \
      --result-json "$result_path" \
      --eval-manifest "$manifest_path" \
      --eval-pairs-json "$pair_path" \
      --split-name heldout \
      --num-batches 0 \
      --student-steps "$step" \
      --teacher-steps "$step" \
      --rollout-student "$source" \
      --emit-teacher-action-gt \
      --disable-eval-gradient-checkpointing \
      --disable-eval-force-cfg \
      --eval-rollout-grad-mode endpoint \
      --eval-empty-cache > "$log_path" 2>&1
  done
  IFS="$old_ifs"
}

validate_core2_gate() {
  local row split protocol_note checkpoint_name checkpoint_path checkpoint_inventory source
  local source_subdir source_inventory source_env step manifest_path manifest_digest pair_path
  local pair_digest entry_dir result_path log_path evaluator_output port
  local old_ifs="$IFS"
  local command
  [ "$RUN" -eq 1 ] && [ "$SPLIT" = "final12" ] || return 0
  [ -n "$CORE2_GATE_PLAN" ] || die "--run --split final12 requires --core2-gate /absolute/core2/output/plan.json"
  command=("$PYTHON" "$SUMMARIZER" --verify-core2-gate "$CORE2_GATE_PLAN")
  IFS=$'\037'
  for row in "${ENTRY_ROWS[@]}"; do
    read -r split protocol_note checkpoint_name checkpoint_path checkpoint_inventory source \
      source_subdir source_inventory source_env step manifest_path manifest_digest pair_path \
      pair_digest entry_dir result_path log_path evaluator_output port <<< "$row"
    command+=(--require-entry "${checkpoint_name}:${source}:${step}")
  done
  IFS="$old_ifs"
  "${command[@]}" || die "Core2 gate verification failed; final12 evaluator was not started"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run) RUN=1; shift ;;
    --split) require_value "$@"; SPLIT="$2"; shift 2 ;;
    --checkpoint) require_value "$@"; CHECKPOINT_SELECTION="$2"; shift 2 ;;
    --source) require_value "$@"; SOURCE_SELECTION="$2"; shift 2 ;;
    --k) require_value "$@"; K_SELECTION="$2"; shift 2 ;;
    --output-root) require_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --core2-gate) require_value "$@"; CORE2_GATE_PLAN="$2"; shift 2 ;;
    --dataset-path) require_value "$@"; REQUESTED_DATASET_ROOT="$2"; shift 2 ;;
    --cuda-visible-devices) require_value "$@"; CUDA_VISIBLE_DEVICES_VALUE="$2"; shift 2 ;;
    --master-port-base) require_value "$@"; MASTER_PORT_BASE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ "$SPLIT" = "core2" ] || [ "$SPLIT" = "final12" ] || die "--split must be core2 or final12"
[ "$REQUESTED_DATASET_ROOT" = "$DATASET_ROOT" ] || die "dataset path is fixed to the RobotWin ablation dataset: $DATASET_ROOT"
[[ "$CUDA_VISIBLE_DEVICES_VALUE" =~ ^[0-9]+$ ]] || die "--cuda-visible-devices must be one GPU ordinal"
[[ "$MASTER_PORT_BASE" =~ ^[0-9]+$ ]] || die "--master-port-base must be an integer"
[ "$MASTER_PORT_BASE" -ge 1024 ] && [ "$MASTER_PORT_BASE" -le 65535 ] || die "--master-port-base must be between 1024 and 65535"

assert_fresh_output_root
resolve_core2_gate_plan
assert_exists "$PYTHON"
assert_exists "$TORCHRUN"
assert_exists "$EVALUATOR"
assert_exists "$SUMMARIZER"
assert_exists "$DATASET_ROOT"
assert_exists "$EMPTY_EMB_PATH"
assert_exists "$TEACHER_MODEL_PATH"

expand_checkpoints
expand_sources
expand_steps
build_entries
validate_core2_gate

if [ "$RUN" -eq 0 ]; then
  emit_plan "-"
  exit 0
fi

mkdir "$OUTPUT_ROOT"
emit_plan "${OUTPUT_ROOT}/plan.json"
printf 'Wrote execution plan: %s\n' "${OUTPUT_ROOT}/plan.json" >&2
run_entries
"$PYTHON" "$SUMMARIZER" --plan "${OUTPUT_ROOT}/plan.json" --out "${OUTPUT_ROOT}/summary"
