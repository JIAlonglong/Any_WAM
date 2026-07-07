#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${PROJECT_ROOT}/checkpoints/lingbot-va-posttrain-robotwin}"
DATASET_PATH="${DATASET_PATH:-/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500}"
EMPTY_EMB_PATH="${EMPTY_EMB_PATH:-/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/empty_emb.pt}"
TORCHRUN="${TORCHRUN:-/root/nas/junjie/conda_envs/any_wam/bin/torchrun}"
SMOKE_ROOT="${SMOKE_ROOT:-${PROJECT_ROOT}/distillation_flowmap/output_robotwin_stepwam_ablation/smoke}"
SMOKE_STAGE1_STEPS="${SMOKE_STAGE1_STEPS:-20}"
SMOKE_STAGE2_STEPS="${SMOKE_STAGE2_STEPS:-20}"
SMOKE_TASK="${SMOKE_TASK:-place_a2b_right}"
SMOKE_EPISODES="${SMOKE_EPISODES:-5}"
MASTER_PORT="${MASTER_PORT:-29640}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-${SMOKE_STAGE2_STEPS}}"

cd "${PROJECT_ROOT}"

/root/nas/junjie/conda_envs/any_wam/bin/python \
  distillation_flowmap/ablation/launch_robotwin_stepwam_ablation.py \
  --variant full_stepwam \
  --seed 0 \
  --root "${SMOKE_ROOT}" \
  --teacher-model-path "${TEACHER_MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --empty-emb-path "${EMPTY_EMB_PATH}" \
  --torchrun "${TORCHRUN}" \
  --stage1-steps "${SMOKE_STAGE1_STEPS}" \
  --stage2-steps "${SMOKE_STAGE2_STEPS}" \
  --master-port "${MASTER_PORT}"

RUN_DIR="${SMOKE_ROOT}/full_stepwam/seed_0"
mkdir -p "${RUN_DIR}/metrics" "${RUN_DIR}/videos"

"${TORCHRUN}" --nproc_per_node=1 --master_port="$((MASTER_PORT + 2))" \
  distillation_flowmap/rollout_eval_stage2.py \
  --config distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow \
  --teacher-model-path "${TEACHER_MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${RUN_DIR}/eval" \
  --resume-from-path "${RUN_DIR}/stage2/checkpoints/step_${SMOKE_STAGE2_STEPS}" \
  --result-json "${RUN_DIR}/metrics/offline_rollout.json" \
  --num-batches 1 \
  --student-steps 4 \
  --teacher-steps 4 \
  --pairs 1000,0

"${TORCHRUN}" --nproc_per_node=1 --master_port="$((MASTER_PORT + 3))" \
  distillation_flowmap/rollout_eval_video_stage2.py \
  --config distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow \
  --teacher-model-path "${TEACHER_MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${RUN_DIR}/videos" \
  --resume-from-path "${RUN_DIR}/stage2/checkpoints/step_${SMOKE_STAGE2_STEPS}" \
  --result-json "${RUN_DIR}/metrics/video_mse.json" \
  --num-batches 1 \
  --student-steps 4 \
  --teacher-steps 4 \
  --pairs 1000,0

if [ "${RUN_ROBOTWIN_ENV_SMOKE:-0}" = "1" ]; then
  PYTHONWARNINGS=ignore::UserWarning \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  python -m evaluation.robotwin.eval_polict_client_openpi \
    --config policy/ACT/deploy_policy.yml \
    --overrides \
    --task_name "${SMOKE_TASK}" \
    --task_config "${ROBOTWIN_TASK_CONFIG:-demo_clean}" \
    --train_config_name 0 \
    --model_name 0 \
    --ckpt_setting full_stepwam_seed_0_smoke \
    --seed 0 \
    --policy_name ACT \
    --save_root "${RUN_DIR}/robotwin_env" \
    --video_guidance_scale 5 \
    --action_guidance_scale 1 \
    --test_num "${SMOKE_EPISODES}" \
    --port "${ROBOTWIN_PORT:-29556}"
fi
