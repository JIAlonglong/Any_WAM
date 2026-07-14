#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ROOT="${ROOT:-${PROJECT_ROOT}/distillation_flowmap/output_robotwin_stepwam_ablation/protocol_final12_danceopd_i1_single_seed_v1}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/root/nas/junjie/third_party/RoboTwin_official}"
PYTHON="${PYTHON:-/root/nas/junjie/conda_envs/any_wam/bin/python}"
TORCHRUN="${TORCHRUN:-/root/nas/junjie/conda_envs/any_wam/bin/torchrun}"

STAGE2_STEPS="${STAGE2_STEPS:-5000}"
STUDENT_STEPS=4
TEST_NUM=50
TASK_CONFIG=demo_clean
CLOSED_LOOP_GPUS="${CLOSED_LOOP_GPUS:-0}"
BASE_PORT="${BASE_PORT:-39900}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-39980}"

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
TASKS=(
  place_a2b_right
  put_object_cabinet
  stack_bowls_three
  lift_pot
  place_can_basket
  handover_block
  open_microwave
  open_laptop
  pick_dual_bottles
  blocks_ranking_size
  place_burger_fries
  rotate_qrcode
)

MODE=main
ONLY_VARIANT=""
DRY_RUN=0
SMOKE=0
SERVER_PIDS=()

usage() {
  cat <<'USAGE'
Usage: run_final_danceopd_closed_loop.sh [options]

Runs the official RoboTwin closed-loop client against one K=4 student server per
selected GPU. Tasks are serial within a server because its streaming KV cache
is stateful; servers process distinct task queues in parallel.

Options:
  --main-only             Run only the four strict OPD variants (default).
  --controls-only         Run local-adjacent and action-only separately.
  --all                   Run main variants and structural controls.
  --variant NAME          Restrict the selected group to one variant.
  --smoke                 One task and one valid episode; never a final metric.
  --dry-run               Print frozen commands without launching processes.
  --root PATH             Override the final-ablation output root.
  --robotwin-root PATH    Override the official RoboTwin checkout.
  --gpus IDS              Comma-separated GPU IDs (default: 0).
  --help                  Show this message.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --main-only) MODE=main; shift ;;
    --controls-only) MODE=controls; shift ;;
    --all) MODE=all; shift ;;
    --variant) ONLY_VARIANT="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --root) ROOT="$2"; shift 2 ;;
    --robotwin-root) ROBOTWIN_ROOT="$2"; shift 2 ;;
    --gpus) CLOSED_LOOP_GPUS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ "${MODE}" != main ] && [ "${MODE}" != controls ] && [ "${MODE}" != all ]; then
  echo "MODE must be main, controls, or all" >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${CLOSED_LOOP_GPUS}"
if [ "${#GPU_LIST[@]}" -eq 0 ] || [ -z "${GPU_LIST[0]}" ]; then
  echo "CLOSED_LOOP_GPUS must contain at least one GPU id" >&2
  exit 2
fi
for gpu in "${GPU_LIST[@]}"; do
  if ! [[ "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id: ${gpu}" >&2
    exit 2
  fi
done

case "${MODE}" in
  main) SELECTED_VARIANTS=("${MAIN_VARIANTS[@]}") ;;
  controls) SELECTED_VARIANTS=("${CONTROL_VARIANTS[@]}") ;;
  all) SELECTED_VARIANTS=("${MAIN_VARIANTS[@]}" "${CONTROL_VARIANTS[@]}") ;;
esac

if [ -n "${ONLY_VARIANT}" ]; then
  found=0
  for variant in "${SELECTED_VARIANTS[@]}"; do
    if [ "${variant}" = "${ONLY_VARIANT}" ]; then
      found=1
      break
    fi
  done
  if [ "${found}" -ne 1 ]; then
    echo "Variant ${ONLY_VARIANT} is not in the selected mode ${MODE}" >&2
    exit 2
  fi
  SELECTED_VARIANTS=("${ONLY_VARIANT}")
fi

RUN_TASKS=("${TASKS[@]}")
EPISODES_PER_TASK="${TEST_NUM}"
if [ "${SMOKE}" -eq 1 ]; then
  RUN_TASKS=("${TASKS[0]}")
  EPISODES_PER_TASK=1
fi

cd "${PROJECT_ROOT}"

print_command() {
  printf 'DRY_RUN:'
  printf ' %q' "$@"
  printf '\n'
}

require_path() {
  local path="$1"
  if [ "${DRY_RUN}" -eq 1 ]; then
    return
  fi
  if [ ! -e "${path}" ]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
}

checkpoint_for_variant() {
  local variant="$1"
  printf '%s/%s/seed_0/stage2/checkpoints/step_%s/target_student/transformer' \
    "${ROOT}" "${variant}" "${STAGE2_STEPS}"
}

server_port() {
  local variant_index="$1"
  local gpu_index="$2"
  printf '%d' "$((BASE_PORT + variant_index * 32 + gpu_index))"
}

server_master_port() {
  local variant_index="$1"
  local gpu_index="$2"
  printf '%d' "$((MASTER_PORT_BASE + variant_index * 32 + gpu_index))"
}

server_env_pythonpath() {
  printf '%s' "${PROJECT_ROOT}:${PROJECT_ROOT}/wan_va:${PROJECT_ROOT}/distillation_flowmap${PYTHONPATH:+:${PYTHONPATH}}"
}

build_server_command() {
  local -n command_ref="$1"
  local gpu="$2"
  local port="$3"
  local master_port="$4"
  local checkpoint="$5"
  local save_root="$6"
  command_ref=(
    env
    "CUDA_VISIBLE_DEVICES=${gpu}"
    "PYTHONPATH=$(server_env_pythonpath)"
    "${TORCHRUN}"
    --nproc_per_node=1
    "--master_port=${master_port}"
    wan_va/wan_va_server.py
    --config-name robotwin
    --checkpoint-path "${checkpoint}"
    --num-steps "${STUDENT_STEPS}"
    --action-num-steps "${STUDENT_STEPS}"
    --port "${port}"
    --save-root "${save_root}"
  )
}

build_client_command() {
  local -n command_ref="$1"
  local gpu="$2"
  local port="$3"
  local task="$4"
  local save_root="$5"
  local variant="$6"
  command_ref=(
    env
    "CUDA_VISIBLE_DEVICES=${gpu}"
    "ROBOTWIN_ROOT=${ROBOTWIN_ROOT}"
    "PYTHONPATH=$(server_env_pythonpath)"
    "PYTHONWARNINGS=ignore::UserWarning"
    "${PYTHON}"
    -m evaluation.robotwin.eval_polict_client_openpi
    --config "${ROBOTWIN_ROOT}/policy/ACT/deploy_policy.yml"
    --no-save-visualization
    --overrides
    --task_name "${task}"
    --task_config "${TASK_CONFIG}"
    --train_config_name final_danceopd
    --model_name "${variant}"
    --ckpt_setting "step_${STAGE2_STEPS}_k${STUDENT_STEPS}"
    --seed 0
    --policy_name LingBotVA
    --save_root "${save_root}"
    --video_guidance_scale 5
    --action_guidance_scale 1
    --test_num "${EPISODES_PER_TASK}"
    --port "${port}"
    --nfe "${STUDENT_STEPS}"
  )
}

cleanup_servers() {
  local pid
  for pid in "${SERVER_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${SERVER_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  SERVER_PIDS=()
}

trap cleanup_servers EXIT INT TERM

wait_for_server() {
  local pid="$1"
  local port="$2"
  local log_file="$3"
  local attempt
  for attempt in $(seq 1 180); do
    if nc -z 127.0.0.1 "${port}" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "Server exited before listening on port ${port}: ${log_file}" >&2
      tail -100 "${log_file}" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "Timed out waiting for server port ${port}: ${log_file}" >&2
  tail -100 "${log_file}" >&2 || true
  return 1
}

start_server() {
  local gpu="$1"
  local port="$2"
  local master_port="$3"
  local checkpoint="$4"
  local save_root="$5"
  local log_file="$6"
  local -a command=()
  build_server_command command "${gpu}" "${port}" "${master_port}" "${checkpoint}" "${save_root}"

  mkdir -p "$(dirname "${log_file}")" "${save_root}"
  "${command[@]}" >"${log_file}" 2>&1 &
  SERVER_PIDS+=("$!")
}

run_client_task() {
  local gpu="$1"
  local port="$2"
  local task="$3"
  local save_root="$4"
  local variant="$5"
  local log_file="$6"
  local -a command=()
  build_client_command command "${gpu}" "${port}" "${task}" "${save_root}" "${variant}"

  mkdir -p "$(dirname "${log_file}")"
  "${command[@]}" >"${log_file}" 2>&1
}

write_variant_summary() {
  local variant="$1"
  local run_dir="$2"
  "${PYTHON}" - "${run_dir}" "${variant}" "${STUDENT_STEPS}" \
    "${EPISODES_PER_TASK}" "${RUN_TASKS[@]}" <<'PY'
import json
import sys
from pathlib import Path

from evaluation.robotwin.closed_loop_metrics import build_closed_loop_summary

run_dir = Path(sys.argv[1])
variant = sys.argv[2]
student_steps = int(sys.argv[3])
episodes_per_task = int(sys.argv[4])
tasks = sys.argv[5:]

records = []
for task in tasks:
    path = run_dir / "metrics" / task / "closed_loop_metrics.json"
    if not path.is_file():
        raise SystemExit(f"Missing closed-loop metric for {task}: {path}")
    records.append(json.loads(path.read_text(encoding="utf-8")))

summary = build_closed_loop_summary(
    records,
    expected_task_count=len(tasks),
)
summary.update(
    {
        "variant": variant,
        "student_steps": student_steps,
        "episodes_per_task": episodes_per_task,
        "teacher_reference": {
            "status": "not_evaluated",
            "reason": (
                "This runner records student closed-loop metrics only. "
                "Teacher retention and speedup require a separate native "
                "LingBot-VA teacher reference run under the same task seeds."
            ),
        },
    }
)
out = run_dir / "closed_loop_summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Wrote {out}")
PY
}

print_dry_run() {
  local variant_index=0
  local variant
  local gpu_index
  local gpu
  local port
  local master_port
  local task_index
  local task
  local checkpoint
  local run_dir
  local -a command=()

  for variant in "${SELECTED_VARIANTS[@]}"; do
    checkpoint="$(checkpoint_for_variant "${variant}")"
    run_dir="${ROOT}/${variant}/seed_0/stage2/closed_loop"
    echo "VARIANT ${variant} checkpoint=${checkpoint}"
    for gpu_index in "${!GPU_LIST[@]}"; do
      gpu="${GPU_LIST[${gpu_index}]}"
      port="$(server_port "${variant_index}" "${gpu_index}")"
      master_port="$(server_master_port "${variant_index}" "${gpu_index}")"
      build_server_command command "${gpu}" "${port}" "${master_port}" \
        "${checkpoint}" "${run_dir}/server_outputs/gpu_${gpu}"
      print_command "${command[@]}"
    done
    for task_index in "${!RUN_TASKS[@]}"; do
      task="${RUN_TASKS[${task_index}]}"
      gpu_index=$((task_index % ${#GPU_LIST[@]}))
      gpu="${GPU_LIST[${gpu_index}]}"
      port="$(server_port "${variant_index}" "${gpu_index}")"
      build_client_command command "${gpu}" "${port}" "${task}" \
        "${run_dir}" "${variant}"
      print_command "${command[@]}"
    done
    variant_index=$((variant_index + 1))
  done
}

run_variant() {
  local variant_index="$1"
  local variant="$2"
  local checkpoint
  local run_dir
  local gpu_index
  local gpu
  local port
  local master_port
  local server_log
  local task_index
  local task
  local last_server_index
  local -a worker_pids=()
  local -a worker_labels=()
  local failures=0
  local pid
  local index

  checkpoint="$(checkpoint_for_variant "${variant}")"
  run_dir="${ROOT}/${variant}/seed_0/stage2/closed_loop"
  require_path "${checkpoint}"
  mkdir -p "${run_dir}/logs"

  echo "Starting ${variant}: K=${STUDENT_STEPS}, tasks=${#RUN_TASKS[@]}, episodes=${EPISODES_PER_TASK}"
  for gpu_index in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[${gpu_index}]}"
    port="$(server_port "${variant_index}" "${gpu_index}")"
    master_port="$(server_master_port "${variant_index}" "${gpu_index}")"
    server_log="${run_dir}/logs/server_gpu_${gpu}.log"
    start_server "${gpu}" "${port}" "${master_port}" "${checkpoint}" \
      "${run_dir}/server_outputs/gpu_${gpu}" "${server_log}"
    last_server_index=$((${#SERVER_PIDS[@]} - 1))
    if ! wait_for_server "${SERVER_PIDS[${last_server_index}]}" "${port}" "${server_log}"; then
      cleanup_servers
      return 1
    fi
  done

  for gpu_index in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[${gpu_index}]}"
    port="$(server_port "${variant_index}" "${gpu_index}")"
    (
      worker_status=0
      for task_index in "${!RUN_TASKS[@]}"; do
        if [ "$((task_index % ${#GPU_LIST[@]}))" -ne "${gpu_index}" ]; then
          continue
        fi
        task="${RUN_TASKS[${task_index}]}"
        echo "${variant} task=${task} gpu=${gpu} port=${port}"
        if ! run_client_task "${gpu}" "${port}" "${task}" "${run_dir}" \
          "${variant}" "${run_dir}/logs/${task}_gpu_${gpu}.log"; then
          worker_status=1
          break
        fi
      done
      exit "${worker_status}"
    ) &
    worker_pids+=("$!")
    worker_labels+=("gpu_${gpu}")
  done

  for index in "${!worker_pids[@]}"; do
    pid="${worker_pids[${index}]}"
    if ! wait "${pid}"; then
      echo "Closed-loop worker failed: ${worker_labels[${index}]}" >&2
      failures=1
    fi
  done

  cleanup_servers
  if [ "${failures}" -ne 0 ]; then
    return 1
  fi

  write_variant_summary "${variant}" "${run_dir}"
  echo "Completed ${variant}"
}

if [ "${DRY_RUN}" -eq 1 ]; then
  print_dry_run
  exit 0
fi

require_path "${ROBOTWIN_ROOT}"
require_path "${ROBOTWIN_ROOT}/policy/ACT/deploy_policy.yml"
require_path "${PROJECT_ROOT}/wan_va/wan_va_server.py"

for variant in "${SELECTED_VARIANTS[@]}"; do
  require_path "$(checkpoint_for_variant "${variant}")"
done

variant_index=0
for variant in "${SELECTED_VARIANTS[@]}"; do
  run_variant "${variant_index}" "${variant}"
  variant_index=$((variant_index + 1))
done
