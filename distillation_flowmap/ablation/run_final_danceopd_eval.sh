#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ROOT="${ROOT:-${PROJECT_ROOT}/distillation_flowmap/output_robotwin_stepwam_ablation/protocol_final12_danceopd_i1_single_seed_v1}"
SHARED_STAGE1_CKPT="${SHARED_STAGE1_CKPT:-${PROJECT_ROOT}/distillation_flowmap/output_robotwin_stepwam_ablation/protocol_final12_5000/shared_stage1/representative/seed_0/stage1/checkpoints/step_5000}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${PROJECT_ROOT}/checkpoints/lingbot-va-posttrain-robotwin}"
DATASET_PATH="${DATASET_PATH:-/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/empty_emb.pt}"
TORCHRUN="${TORCHRUN:-/root/nas/junjie/conda_envs/any_wam/bin/torchrun}"
PYTHON="${PYTHON:-/root/nas/junjie/conda_envs/any_wam/bin/python}"

STAGE2_STEPS="${STAGE2_STEPS:-5000}"
TASK_PRESET="${TASK_PRESET:-representative}"
PROTOCOL_SEED="${PROTOCOL_SEED:-0}"
EVAL_GPUS="${EVAL_GPUS:-6,7}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-39200}"

MAIN_VARIANTS=(
  final_w_o_opd
  final_endpoint_only_danceopd
  final_danceopd_velocity_only
  final_stepwam_danceopd
)
CONTROL_VARIANTS=(
  final_local_adjacent_only
  final_action_only
)
ASSET_VARIANTS=(
  final_w_o_opd
  final_stepwam_danceopd
)

MODE=all
RUN_ASSETS="${RUN_ASSETS:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
DRY_RUN=0
CACHE_ONLY=0

usage() {
  cat <<'USAGE'
Usage: run_final_danceopd_eval.sh [options]

Options:
  --main-only             Evaluate only the strict four-variant OPD table.
  --controls-only         Evaluate local-adjacent and action-only separately.
  --skip-assets           Skip teacher/baseline/full native WanVAE assets.
  --skip-summary          Skip the final report invocation.
  --cache-only            Build/validate the shared teacher cache and provenance only.
  --dry-run               Print commands without requiring paths or launching jobs.
  --root PATH             Override the frozen final-ablation output root.
  --eval-gpus IDS         Comma-separated evaluation GPUs (default: 6,7).
  --max-parallel N        Maximum concurrent eval jobs (default: 2).
  --help                  Show this message.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --main-only) MODE=main; shift ;;
    --controls-only) MODE=controls; shift ;;
    --skip-assets) RUN_ASSETS=0; shift ;;
    --skip-summary) RUN_SUMMARY=0; shift ;;
    --cache-only) CACHE_ONLY=1; RUN_ASSETS=0; RUN_SUMMARY=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --root) ROOT="$2"; shift 2 ;;
    --eval-gpus) EVAL_GPUS="$2"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ "${MODE}" != all ] && [ "${MODE}" != main ] && [ "${MODE}" != controls ]; then
  echo "MODE must be all, main, or controls" >&2
  exit 2
fi
if ! [[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
IFS=',' read -r -a EVAL_GPU_LIST <<< "${EVAL_GPUS}"
if [ "${#EVAL_GPU_LIST[@]}" -eq 0 ] || [ -z "${EVAL_GPU_LIST[0]}" ]; then
  echo "EVAL_GPUS must contain at least one GPU id" >&2
  exit 2
fi

PROTOCOL_DIR="${ROOT}/protocol/manifests/${TASK_PRESET}_protocol_seed_${PROTOCOL_SEED}"
HELDOUT_MANIFEST="${PROTOCOL_DIR}/heldout_eval_manifest.json"
EVAL_PAIRS="${PROTOCOL_DIR}/eval_pairs.json"
ASSET_MANIFEST="${PROTOCOL_DIR}/heldout_asset_manifest.json"
ASSET_PAIRS="${PROTOCOL_DIR}/asset_eval_pairs_t1000_r0.json"
TEACHER_CACHE="${TEACHER_CACHE:-${ROOT}/protocol/teacher_cache_heldout_equal_nfe_trajectories.pt}"
TEACHER_CACHE_LOG="${ROOT}/protocol/teacher_cache_heldout_equal_nfe_trajectories.log"
PROVENANCE_JSON="${ROOT}/protocol/final_danceopd_eval_provenance.json"
SUMMARY_DIR="${SUMMARY_DIR:-${ROOT}/summary_final_danceopd}"

STUDENT_STEPS=(1 2 4)
TEACHER_STEPS=(1 2 4 8)
TRAJECTORY_TEACHER_STEPS=(1 2 4)
CACHE_SOURCE_VARIANT="${CACHE_SOURCE_VARIANT:-final_w_o_opd}"

print_command() {
  printf 'DRY_RUN:'
  printf ' %q' "$@"
  printf '\n'
}

require_path() {
  local path="$1"
  if [ "${DRY_RUN}" = 1 ]; then
    return
  fi
  if [ ! -e "${path}" ]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
}

stage2_checkpoint() {
  local variant="$1"
  printf '%s/%s/seed_0/stage2/checkpoints/step_%s' "${ROOT}" "${variant}" "${STAGE2_STEPS}"
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

prepare_asset_protocol() {
  if [ "${DRY_RUN}" = 1 ]; then
    echo "DRY_RUN: create one-heldout-record-per-task asset manifest ${ASSET_MANIFEST}"
    echo "DRY_RUN: create clean 1000->0 asset pair file ${ASSET_PAIRS}"
    return
  fi

  require_path "${HELDOUT_MANIFEST}"
  require_path "${EVAL_PAIRS}"
  "${PYTHON}" - "${HELDOUT_MANIFEST}" "${EVAL_PAIRS}" "${ASSET_MANIFEST}" "${ASSET_PAIRS}" <<'PY'
import json
import sys
from pathlib import Path

heldout_path, pairs_path, asset_manifest_path, asset_pairs_path = map(Path, sys.argv[1:])
heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
pairs_payload = json.loads(pairs_path.read_text(encoding="utf-8"))

tasks = []
for task in heldout.get("tasks", []):
    indices = list(task.get("indices", []))
    if not indices:
        raise SystemExit(f"Task has no held-out indices: {task.get('task')!r}")
    tasks.append({"task": str(task["task"]), "indices": [int(indices[0])]})
if len(tasks) != 12:
    raise SystemExit(f"Expected 12 asset tasks, got {len(tasks)}")

asset_manifest = dict(heldout)
asset_manifest["schema"] = "robotwin_final_danceopd_asset_manifest_v1"
asset_manifest["samples_per_task"] = 1
asset_manifest["tasks"] = tasks

raw_pairs = pairs_payload.get("pairs", pairs_payload)
clean_pairs = [
    pair for pair in raw_pairs
    if float(pair["t"]) == 1000.0 and float(pair["r"]) == 0.0
]
if len(clean_pairs) != 1:
    raise SystemExit(f"Expected exactly one clean 1000->0 pair, got {clean_pairs!r}")
asset_pairs = {
    "schema": "robotwin_final_danceopd_asset_pairs_v1",
    "task_preset": pairs_payload.get("task_preset"),
    "protocol_seed": pairs_payload.get("protocol_seed"),
    "pairs": clean_pairs,
}

for path, payload in (
    (asset_manifest_path, asset_manifest),
    (asset_pairs_path, asset_pairs),
):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

build_teacher_cache() {
  local checkpoint
  checkpoint="$(stage2_checkpoint "${CACHE_SOURCE_VARIANT}")"
  require_path "${checkpoint}"
  require_path "${HELDOUT_MANIFEST}"
  require_path "${EVAL_PAIRS}"

  local command=(
    env "CUDA_VISIBLE_DEVICES=${EVAL_GPU_LIST[0]}" "${TORCHRUN}"
    --nproc_per_node=1
    "--master_port=${MASTER_PORT_BASE}"
    distillation_flowmap/rollout_eval_stage2.py
    --config distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow
    --teacher-model-path "${TEACHER_MODEL_PATH}"
    --dataset-path "${DATASET_PATH}"
    --empty-emb-path "${EMPTY_EMB_PATH}"
    --output-dir "${ROOT}/eval/teacher_cache_heldout_equal_nfe"
    --resume-from-path "${checkpoint}"
    --result-json "${ROOT}/eval/teacher_cache_heldout_equal_nfe.json"
    --teacher-cache-path "${TEACHER_CACHE}"
    --teacher-cache-only
    --cache-teacher-trajectories
    --trajectory-teacher-steps "${TRAJECTORY_TEACHER_STEPS[@]}"
    --eval-manifest "${HELDOUT_MANIFEST}"
    --eval-pairs-json "${EVAL_PAIRS}"
    --split-name heldout
    --num-batches 0
    --student-steps "${STUDENT_STEPS[@]}"
    --teacher-steps "${TEACHER_STEPS[@]}"
    --disable-eval-gradient-checkpointing
    --disable-eval-force-cfg
    --eval-rollout-grad-mode endpoint
    --eval-empty-cache
  )

  if [ "${DRY_RUN}" = 1 ]; then
    print_command "${command[@]}"
    return
  fi

  mkdir -p "${ROOT}/protocol" "${ROOT}/eval"
  echo "Building shared held-out teacher cache on GPU ${EVAL_GPU_LIST[0]}"
  "${command[@]}" > "${TEACHER_CACHE_LOG}" 2>&1
  require_path "${TEACHER_CACHE}"
}

write_provenance() {
  if [ "${DRY_RUN}" = 1 ]; then
    echo "DRY_RUN: write provenance ${PROVENANCE_JSON} with teacher-cache SHA256"
    return
  fi

  require_path "${TEACHER_CACHE}"
  local cache_hash
  local manifest_hash
  local pairs_hash
  local git_hash
  cache_hash="$(sha256_file "${TEACHER_CACHE}")"
  manifest_hash="$(sha256_file "${HELDOUT_MANIFEST}")"
  pairs_hash="$(sha256_file "${EVAL_PAIRS}")"
  git_hash="$(git rev-parse HEAD)"

  "${PYTHON}" - "${PROVENANCE_JSON}" "${ROOT}" "${git_hash}" "${TEACHER_CACHE}" "${cache_hash}" "${HELDOUT_MANIFEST}" "${manifest_hash}" "${EVAL_PAIRS}" "${pairs_hash}" "${SHARED_STAGE1_CKPT}" "${STAGE2_STEPS}" "${TASK_PRESET}" "${PROTOCOL_SEED}" <<'PY'
import json
import sys
from pathlib import Path

(
    out_path,
    root,
    git_hash,
    cache_path,
    cache_hash,
    heldout_manifest,
    manifest_hash,
    eval_pairs,
    pairs_hash,
    shared_stage1,
    stage2_steps,
    task_preset,
    protocol_seed,
) = sys.argv[1:]

payload = {
    "schema": "robotwin_final_danceopd_eval_provenance_v1",
    "git_hash": git_hash,
    "root": root,
    "task_preset": task_preset,
    "protocol_seed": int(protocol_seed),
    "stage2_steps": int(stage2_steps),
    "shared_stage1_checkpoint": shared_stage1,
    "heldout_manifest": {"path": heldout_manifest, "sha256": manifest_hash},
    "eval_pairs": {"path": eval_pairs, "sha256": pairs_hash},
    "teacher_cache": {"path": cache_path, "sha256": cache_hash},
    "main_variants": [
        "final_w_o_opd",
        "final_endpoint_only_danceopd",
        "final_danceopd_velocity_only",
        "final_stepwam_danceopd",
    ],
    "structural_controls": [
        "final_local_adjacent_only",
        "final_action_only",
    ],
    "student_steps": [1, 2, 4],
    "teacher_steps": [1, 2, 4, 8],
    "trajectory_teacher_steps": [1, 2, 4],
    "decoded_video_metrics": {
        "backend": "native_wanvae",
        "device": "cuda",
        "lpips": "alexnet-lpips-0.1.4",
    },
}
out = Path(out_path)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

ACTIVE_PIDS=()
ACTIVE_LABELS=()
FAILURES=0
JOB_INDEX=0

wait_active_jobs() {
  local label="$1"
  local index
  local status=0
  for index in "${!ACTIVE_PIDS[@]}"; do
    if wait "${ACTIVE_PIDS[${index}]}"; then
      echo "Completed ${ACTIVE_LABELS[${index}]}"
    else
      echo "FAILED ${ACTIVE_LABELS[${index}]}" >&2
      status=1
      FAILURES=$(( FAILURES + 1 ))
    fi
  done
  ACTIVE_PIDS=()
  ACTIVE_LABELS=()
  if [ "${status}" -ne 0 ]; then
    return 1
  fi
}

launch_numeric_eval() {
  local variant="$1"
  local result_name="$2"
  local decoded_metrics="$3"
  local diagnostics="$4"
  local gpu="${EVAL_GPU_LIST[$(( JOB_INDEX % ${#EVAL_GPU_LIST[@]} ))]}"
  local port=$(( MASTER_PORT_BASE + 20 + JOB_INDEX ))
  local checkpoint
  local run_dir
  local command

  checkpoint="$(stage2_checkpoint "${variant}")"
  run_dir="${ROOT}/${variant}/seed_0"
  require_path "${checkpoint}"

  command=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "${TORCHRUN}"
    --nproc_per_node=1
    "--master_port=${port}"
    distillation_flowmap/rollout_eval_stage2.py
    --config distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow
    --teacher-model-path "${TEACHER_MODEL_PATH}"
    --dataset-path "${DATASET_PATH}"
    --empty-emb-path "${EMPTY_EMB_PATH}"
    --output-dir "${run_dir}/eval_heldout_final"
    --resume-from-path "${checkpoint}"
    --result-json "${run_dir}/metrics/${result_name}.json"
    --teacher-cache-path "${TEACHER_CACHE}"
    --eval-manifest "${HELDOUT_MANIFEST}"
    --eval-pairs-json "${EVAL_PAIRS}"
    --split-name heldout
    --num-batches 0
    --student-steps "${STUDENT_STEPS[@]}"
    --teacher-steps "${TEACHER_STEPS[@]}"
    --disable-eval-gradient-checkpointing
    --disable-eval-force-cfg
    --eval-rollout-grad-mode endpoint
    --eval-empty-cache
  )
  if [ "${diagnostics}" = 1 ]; then
    command+=(--rollout-drift --same-state-velocity)
  fi
  if [ "${decoded_metrics}" = 1 ]; then
    command+=(
      --decoded-video-metrics
      --decoded-video-device cuda
      --decoded-video-lpips
    )
  fi

  JOB_INDEX=$(( JOB_INDEX + 1 ))
  echo "Launching held-out eval ${variant} on GPU ${gpu}"
  if [ "${DRY_RUN}" = 1 ]; then
    print_command "${command[@]}"
    return
  fi

  mkdir -p "${run_dir}/metrics" "${run_dir}/eval_heldout_final"
  "${command[@]}" > "${run_dir}/metrics/${result_name}.log" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LABELS+=("${variant}:${result_name}:gpu${gpu}")
}

run_numeric_group() {
  local group_name="$1"
  shift
  local active=0
  local variant
  for variant in "$@"; do
    if [ "${variant}" = final_action_only ]; then
      launch_numeric_eval "${variant}" offline_rollout_action_only 0 0
    else
      launch_numeric_eval "${variant}" offline_rollout 1 1
    fi
    if [ "${DRY_RUN}" = 1 ]; then
      continue
    fi
    active=$(( active + 1 ))
    if [ "${active}" -ge "${MAX_PARALLEL}" ]; then
      wait_active_jobs "${group_name}"
      active=0
    fi
  done
  if [ "${DRY_RUN}" = 0 ] && [ "${#ACTIVE_PIDS[@]}" -gt 0 ]; then
    wait_active_jobs "${group_name}"
  fi
}

write_action_only_scope() {
  local out_path="${ROOT}/final_action_only/seed_0/metrics/control_scope.json"
  if [ "${DRY_RUN}" = 1 ]; then
    echo "DRY_RUN: mark action-only control as action/closed-loop only at ${out_path}"
    return
  fi
  mkdir -p "$(dirname "${out_path}")"
  "${PYTHON}" - "${out_path}" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "schema": "robotwin_final_danceopd_action_only_scope_v1",
            "offline_report": "action endpoint metrics only",
            "decoded_video_metrics": "not_applicable",
            "main_opd_table": False,
            "closed_loop_required": True,
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY
}

launch_asset_eval() {
  local variant="$1"
  local gpu="${EVAL_GPU_LIST[$(( JOB_INDEX % ${#EVAL_GPU_LIST[@]} ))]}"
  local port=$(( MASTER_PORT_BASE + 200 + JOB_INDEX ))
  local checkpoint
  local run_dir
  local command

  checkpoint="$(stage2_checkpoint "${variant}")"
  run_dir="${ROOT}/${variant}/seed_0"
  require_path "${checkpoint}"

  command=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "${TORCHRUN}"
    --nproc_per_node=1
    "--master_port=${port}"
    distillation_flowmap/rollout_eval_video_stage2.py
    --config distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow
    --teacher-model-path "${TEACHER_MODEL_PATH}"
    --dataset-path "${DATASET_PATH}"
    --empty-emb-path "${EMPTY_EMB_PATH}"
    --output-dir "${run_dir}/videos/final_heldout_assets"
    --resume-from-path "${checkpoint}"
    --result-json "${run_dir}/metrics/final_video_assets.json"
    --video-dir "${run_dir}/videos/final_heldout_assets"
    --eval-manifest "${ASSET_MANIFEST}"
    --eval-pairs-json "${ASSET_PAIRS}"
    --split-name heldout
    --num-batches 0
    --student-steps "${STUDENT_STEPS[@]}"
    --teacher-steps "${TEACHER_STEPS[@]}"
    --video-max-pairs 12
    --video-decode-device cuda
    --condition-first-frame
    --disable-eval-gradient-checkpointing
    --disable-eval-force-cfg
    --eval-rollout-grad-mode endpoint
    --eval-empty-cache
  )

  JOB_INDEX=$(( JOB_INDEX + 1 ))
  echo "Launching asset export ${variant} on GPU ${gpu}"
  if [ "${DRY_RUN}" = 1 ]; then
    print_command "${command[@]}"
    return
  fi

  mkdir -p "${run_dir}/metrics" "${run_dir}/videos/final_heldout_assets"
  "${command[@]}" > "${run_dir}/metrics/final_video_assets.log" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LABELS+=("${variant}:final_video_assets:gpu${gpu}")
}

run_assets() {
  local active=0
  local variant
  prepare_asset_protocol
  for variant in "${ASSET_VARIANTS[@]}"; do
    launch_asset_eval "${variant}"
    if [ "${DRY_RUN}" = 1 ]; then
      continue
    fi
    active=$(( active + 1 ))
    if [ "${active}" -ge "${MAX_PARALLEL}" ]; then
      wait_active_jobs assets
      active=0
    fi
  done
  if [ "${DRY_RUN}" = 0 ] && [ "${#ACTIVE_PIDS[@]}" -gt 0 ]; then
    wait_active_jobs assets
  fi
}

run_summary() {
  local command=(
    "${PYTHON}"
    distillation_flowmap/ablation/summarize_final_danceopd_ablation.py
    --root "${ROOT}"
    --out "${SUMMARY_DIR}"
    --provenance "${PROVENANCE_JSON}"
  )
  if [ "${DRY_RUN}" = 1 ]; then
    print_command "${command[@]}"
  else
    "${command[@]}"
  fi
}

if [ "${DRY_RUN}" = 0 ]; then
  require_path "${SHARED_STAGE1_CKPT}"
  require_path "${TEACHER_MODEL_PATH}"
  require_path "${DATASET_PATH}"
  require_path "${EMPTY_EMB_PATH}"
fi

build_teacher_cache
write_provenance

if [ "${CACHE_ONLY}" = 0 ]; then
  if [ "${MODE}" = all ] || [ "${MODE}" = main ]; then
    run_numeric_group main "${MAIN_VARIANTS[@]}"
  fi
  if [ "${MODE}" = all ] || [ "${MODE}" = controls ]; then
    run_numeric_group controls "${CONTROL_VARIANTS[@]}"
    write_action_only_scope
  fi
  if [ "${RUN_ASSETS}" = 1 ] && { [ "${MODE}" = all ] || [ "${MODE}" = main ]; }; then
    run_assets
  fi
fi

if [ "${FAILURES}" -ne 0 ]; then
  echo "Final DanceOPD eval finished with ${FAILURES} failed job(s)." >&2
  exit 1
fi

if [ "${CACHE_ONLY}" = 0 ] && [ "${RUN_SUMMARY}" = 1 ]; then
  run_summary
fi

echo "Final DanceOPD eval finished. Provenance: ${PROVENANCE_JSON}"
