# LIBERO FlowMap 蒸馏指南

## 概述

本文档说明如何在单卡 A800 80G 上使用 FlowMap 方法蒸馏 LIBERO 版本的 Flash-WAM 模型。

## 环境准备

### 1. 数据集准备

LIBERO 数据集已经预处理好，位于：
```
training_data/libero-long-lerobot/
├── latents/chunk-000/
│   ├── observation.images.agentview_rgb/   # VAE 编码后的 latent
│   └── observation.images.eye_in_hand_rgb/
├── data/chunk-000/                         # 动作数据 (parquet)
├── meta/info.json                          # 数据集元信息
└── videos/chunk-000/                       # 原始视频
```

### 2. 生成 empty_emb.pt

empty_emb.pt 是空文本嵌入文件，用于 CFG 无条件推理。由于 LIBERO 和 RobotWin 使用相同的 Text Encoder，可以直接复用 RobotWin 的 empty_emb.pt：

```bash
cp training_data/robotwin_hf/empty_emb.pt training_data/libero-long-lerobot/empty_emb.pt
```

或者使用预处理脚本生成（需要 GPU）：
```bash
python scripts/preprocess_robotwin.py \
    --data-root /path/to/libero/data \
    --checkpoint-dir checkpoints/base \
    --output-dir training_data/libero-long-lerobot \
    --generate-empty-emb-only
```

### 3. 教师模型

教师模型位于：
```
checkpoints/libero/
├── transformer/        # Transformer 模型权重
├── text_encoder/       # Text Encoder
├── tokenizer/          # Tokenizer
└── vae/                # VAE 编码器
```

## 训练配置

### 配置文件

- **主配置**: `distillation_flowmap/config_libero.py`
- **核心参数**:
  - 分辨率: 128×128
  - 动作维度: 7 (单臂 6 关节 + 夹爪)
  - action_per_frame: 4
  - attn_window: 30
  - frame_chunk_size: 4
  - LoRA: rank=256, alpha=128

### 显存优化（单卡 A800 80G）

```python
# config_libero.py 中的关键配置
cfg.batch_size = 1                    # 每个 GPU 的 batch size
cfg.gradient_accumulation_steps = 8   # 梯度累积步数（等效 batch = 8）
cfg.use_lora = True                   # 启用 LoRA 微调（减少显存占用）
cfg.lora_rank = 256                   # LoRA rank
cfg.lora_alpha = 128                  # LoRA alpha
```

### FlowMap 蒸馏参数

```python
# 三种目标的混合比例
cfg.diffusion_ratio = 0.5    # 50% 扩散目标（标准 FlowMatch）
cfg.consistency_ratio = 0.25 # 25% 一致性目标（LCM 端点映射）
cfg.flowmap_ratio = 0.25     # 25% 流映射目标（AnyFlow 核心）

# 中心差分参数
cfg.epsilon = 1.0            # 扰动步长
cfg.gt_regression_weight = 0.1  # GT 回归 loss 权重
```

## 启动训练

### 单卡训练

```bash
# 设置环境变量
export TEACHER_PATH=checkpoints/libero
export DATASET_PATH=training_data/libero-long-lerobot
export OUTPUT_DIR=distillation_flowmap/output_libero
export CONFIG_FILE=distillation_flowmap.config_libero
export DISTILL_MODE=flashwam

# 启动训练（单卡）
NGPU=1 bash distillation_flowmap/run.sh
```

### 多卡训练（如果可用）

```bash
# 设置环境变量
export TEACHER_PATH=checkpoints/libero
export DATASET_PATH=training_data/libero-long-lerobot
export OUTPUT_DIR=distillation_flowmap/output_libero
export CONFIG_FILE=distillation_flowmap.config_libero
export DISTILL_MODE=flashwam

# 启动训练（4卡）
NGPU=4 bash distillation_flowmap/run.sh
```

### 自定义启动脚本

创建 `distillation_flowmap/run_libero.sh`：

```bash
#!/bin/bash
# LIBERO FlowMap 蒸馏启动脚本（单卡 A800 80G）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 路径配置
export TEACHER_PATH="${PROJECT_ROOT}/checkpoints/libero"
export DATASET_PATH="${PROJECT_ROOT}/training_data/libero-long-lerobot"
export OUTPUT_DIR="${SCRIPT_DIR}/output_libero"
export CONFIG_FILE="distillation_flowmap.config_libero"
export DISTILL_MODE="flashwam"

# 训练参数
NGPU="${NGPU:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:-}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

ARGS=""
[ -n "$RESUME_FROM_STEP" ] && ARGS="$ARGS --resume-from-step $RESUME_FROM_STEP"
[ -n "$RESUME_FROM_PATH" ] && ARGS="$ARGS --resume-from-path $RESUME_FROM_PATH"

echo "=========================================="
echo "LIBERO FlowMap Distillation"
echo "=========================================="
echo "Teacher: ${TEACHER_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Output:  ${OUTPUT_DIR}"
echo "GPUs:    ${NGPU}"
echo "=========================================="

torchrun \
    --nproc_per_node="${NGPU}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/train.py" \
    --teacher-model-path "$TEACHER_PATH" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    $ARGS
```

## 监控训练

### WandB 日志

如果启用了 WandB（默认开启），训练过程会记录以下指标：

- `loss/total`: 总损失
- `loss/video_consistency`: 视频一致性损失
- `loss/action_consistency`: 动作一致性损失
- `loss/action_aware`: 动作感知正则损失
- `loss/gt_regression`: GT 回归损失
- `train/grad_norm`: 梯度范数
- `train/lr`: 学习率

### 检查点保存

检查点保存在：
```
output_libero/checkpoints/
├── step_1000/
│   ├── online_student/transformer/   # 在线学生模型
│   └── target_student/transformer/   # 目标学生模型（EMA）
├── step_2000/
│   └── ...
└── ...
```

## 推理测试

训练完成后，可以使用 `distillation_flowmap/inference.py` 进行推理测试：

```bash
python distillation_flowmap/inference.py \
    --model-path output_libero/checkpoints/step_10000/target_student/transformer \
    --config-name libero \
    --port 29056
```

然后使用评测脚本：
```bash
bash evaluation/libero/launch_server.sh
bash evaluation/libero/launch_client.sh
```

## 常见问题

### 1. 显存不足

如果遇到 OOM，可以尝试：
- 减小 `batch_size`（已设为 1）
- 增加 `gradient_accumulation_steps`（已设为 8）
- 减小 `lora_rank`（如从 256 降到 128）
- 启用 `gradient_checkpointing`（已通过 `apply_ac` 启用）

### 2. empty_emb.pt 缺失

确保 empty_emb.pt 文件存在于数据集目录：
```bash
ls -la training_data/libero-long-lerobot/empty_emb.pt
```

如果缺失，从 RobotWin 复制：
```bash
cp training_data/robotwin_hf/empty_emb.pt training_data/libero-long-lerobot/
```

### 3. 训练不稳定

如果训练不稳定（loss 震荡或 NaN），可以尝试：
- 降低学习率（`cfg.learning_rate = 1e-6`）
- 增加 warmup 步数（`cfg.warmup_steps = 200`）
- 调整 FlowMap 比例（增加 `diffusion_ratio`）

## 参考

- FlowMap 原始论文: AnyFlow
- Flash-WAM 论文: Flash-WAM
- 配置文件: `distillation_flowmap/config_libero.py`
- 核心实现: `distillation_flowmap/flowmap_trainer.py`, `distillation_flowmap/flowmap_step.py`
