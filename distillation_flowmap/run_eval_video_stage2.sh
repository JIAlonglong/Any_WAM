#!/bin/bash
# 统一的 Stage 2 视频评估脚本
# 用法: bash run_eval_video_stage2.sh [robotwin|libero] [checkpoint_step]

source /kpfs-intern/jialongliu/miniforge3/etc/profile.d/conda.sh
conda activate flashwam

cd /root/intern/jialongliu/projects/Flash-WAM
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 默认参数
MODEL=${1:-"libero"}
CKPT_STEP=${2:-"latest"}

# 根据模型选择配置
if [ "$MODEL" = "robotwin" ]; then
    CONFIG="distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow"
    TEACHER_PATH="/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/robotwin"
    DATASET_PATH="/root/intern/jialongliu/projects/Flash-WAM/training_data/robotwin_hf"
    OUTPUT_DIR="/root/intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_robotwin_stage2_anyflow_from_stage1_step4000"
    # 获取最新的checkpoint
    if [ "$CKPT_STEP" = "latest" ]; then
        CKPT_STEP=$(ls ${OUTPUT_DIR}/checkpoints/ | grep step_ | sed 's/step_//' | sort -n | tail -1)
    fi
elif [ "$MODEL" = "libero" ]; then
    CONFIG="distillation_flowmap.config_libero_fullfinetune_stage2_anyflow"
    TEACHER_PATH="/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero"
    DATASET_PATH="/root/intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot"
    OUTPUT_DIR="/root/intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_fullft_stage2_anyflow_from_stage1_step1000"
    # 获取最新的checkpoint
    if [ "$CKPT_STEP" = "latest" ]; then
        CKPT_STEP=$(ls ${OUTPUT_DIR}/checkpoints/ | grep step_ | sed 's/step_//' | sort -n | tail -1)
    fi
else
    echo "未知模型: $MODEL, 支持: robotwin, libero"
    exit 1
fi

RESUME_PATH="${OUTPUT_DIR}/checkpoints/step_${CKPT_STEP}"
VIDEO_DIR="${OUTPUT_DIR}/eval_videos_step${CKPT_STEP}"
RESULT_JSON="${OUTPUT_DIR}/eval_metrics_step${CKPT_STEP}.json"

echo "========================================"
echo "评估模型: $MODEL"
echo "Checkpoint: step_${CKPT_STEP}"
echo "========================================"

mkdir -p ${VIDEO_DIR}

torchrun --nproc_per_node=1 \
    distillation_flowmap/rollout_eval_video_stage2.py \
    --config ${CONFIG} \
    --teacher-model-path ${TEACHER_PATH} \
    --dataset-path ${DATASET_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --resume-from-path ${RESUME_PATH} \
    --result-json ${RESULT_JSON} \
    --num-batches 1 \
    --seed 42 \
    --cfg-scale 5.0 \
    --pairs "1000,0" \
    --student-steps 1 2 4 \
    --teacher-steps 1 2 4 \
    --video-dir ${VIDEO_DIR} \
    --video-fps 10 \
    --video-max-pairs 1 \
    --video-sample-index 0
