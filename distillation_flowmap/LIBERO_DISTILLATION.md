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

检查点保存在当前配置的 `cfg.output_dir` 下。例如：
```
distillation_flowmap/output_libero_fullft_stage1_warmup/checkpoints/
distillation_flowmap/output_libero_fullft_stage2_anyflow/checkpoints/
```
每个 step 通常包含 `online_student/transformer/` 和 `target_student/transformer/`。

## 推理测试

训练完成后，真实 LIBERO 环境视频和成功率评估使用统一入口：

```bash
bash evaluation/libero/run_eval_new.sh step_5000 online_student
EVAL_MODE=success TEST_NUM=50 bash evaluation/libero/run_eval_new.sh step_5000 online_student
```

离线 teacher/student rollout 指标使用：

```bash
python distillation_flowmap/rollout_eval_stage2.py --help
```


## Stage 1 / Stage 2 OPD 与 i2va Demo 说明

本节记录当前 LIBERO full-parameter Stage 1 -> Stage 2 的固定用法。不要把这些入口当作临时 debug 代码删除；如果改 OPD 或 demo 生成逻辑，需要同步更新这里。

更完整的中文 method 风格说明见 `distillation_flowmap/FLOWMAP_METHOD_CN.md`。

### Stage 1 checkpoint 结构

Stage 1 warmup checkpoint 目录形如：

```text
distillation_flowmap/output_libero_fullft_stage1_warmup/checkpoints/step_N/
├── online_student/transformer/
└── target_student/transformer/
```

Stage 2 continuation 的 `RESUME_FROM_PATH` 应该指向 `step_N` 目录本身，而不是内部的 `target_student/transformer` 或 `online_student/transformer`。当前常用起点是：

```text
/kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_fullft_stage1_warmup/checkpoints/step_1000
```

通常设置 `RESUME_ONLINE_FROM_TARGET=1`，让 Stage 2 的 online/target student 都从 Stage 1 target 权重起步。

### Stage 2 OPD 启动命令

从 Stage 1 `step_1000` 启动 Stage 2 的一行命令：

```bash
source /kpfs-intern/jialongliu/miniforge3/bin/activate && conda activate flashwam && cd /kpfs-intern/jialongliu/projects/Flash-WAM && CONFIG_FILE=distillation_flowmap.config_libero_fullfinetune_stage2_anyflow RESUME_FROM_PATH=/kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_fullft_stage1_warmup/checkpoints/step_1000 RESUME_ONLINE_FROM_TARGET=1 OUTPUT_DIR=/kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_fullft_stage2_anyflow_from_stage1_step1000 MAX_TRAIN_STEPS=10000 WANDB_MODE=offline torchrun --nproc_per_node=8 --master_port=29620 distillation_flowmap/train.py --teacher-model-path /kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero --dataset-path /kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot
```

关键默认值：

- `MAX_TRAIN_STEPS`: Stage 2 总迭代数。默认在 `config_libero_fullfinetune_stage2_anyflow.py` 中是 `5000`；上面命令显式设为 `10000`。
- `SAVE_INTERVAL`: checkpoint 保存间隔，默认 `1000`。
- `GRADIENT_CHECKPOINTING=0`: 已在 config 中默认关闭。OPD 有额外 student backward 路径，PyTorch FSDP2 在 activation checkpoint recompute 中可能触发 `aten.addmm.default: got mixed torch.Tensor and DTensor`。
- `SKIP_TEACHER_COMPILE=1`: 已在 config 中默认开启，Stage 2 不再额外编译 teacher。
- `OPD_AUX_INTERVAL=8`: 已在 config 中默认设置。每 8 个 optimizer step 跑一次 OPD，并且只在 gradient accumulation 的最后一个 microbatch 跑，避免被 `ACCUM` 放大。
- `OPD_ROLLOUT_STEP_PAIRS=1,1;1,2;1,4`: 已在 config 中默认设置。在 `student_state` 模式下保留 1-step/2-step/4-step student-induced state。此模式里 teacher 不做 N-step endpoint rollout，pair 的第一项不会增加 teacher 步数；若要做 endpoint teacher rollout，需要切 `OPD_TEACHER_TARGET_MODE=endpoint`，同时 transition 语义会回到 x0/endpoint。
- OPD 性能约束：不要改回每个 microbatch 都跑，否则耗时会乘以 `gradient_accumulation_steps`。由于 OPD 只跑最后一个 microbatch，OPD loss 本身不再除以 `gradient_accumulation_steps`。
- action teacher transition 使用 conditional-only teacher forward；action transition 不使用 CFG uncond 分支，避免一整次无用 teacher forward。
- `OPD_PROFILE=1`: 临时打开 OPD 分段计时，会同步 CUDA 并打印 `prepare_batch/video_student_rollout/video_teacher/action_opd/backward` 等耗时。只用于诊断，不建议常开。

### 当前 Stage 2 OPD 默认语义

当前 Stage 2 配置让 OPD 的 transition/field matching 主导，endpoint/local-FM 只做 anchor：

```text
OPD_TEACHER_TARGET_MODE=student_state
VIDEO_TRANSITION_PARAM=velocity
ACTION_TRANSITION_PARAM=velocity
LOCAL_FM_WEIGHT=1e-4
ACTION_LOCAL_FM_WEIGHT=0.003
ACTION_TRANSITION_BLOCK_WEIGHT=4.0
ACTION_LOCAL_FM_BLOCK_WEIGHT=1.0
OPD_TRANSITION_GROUP_WEIGHT=25.0
OPD_ANCHOR_CAP_RATIO=0.25
OPD_QUERY_BIAS=low_t
OPD_AUX_INTERVAL=8
OPD_ROLLOUT_STEP_PAIRS=1,1;1,2;1,4
```

`student_state` 模式下，video/action 的 teacher 和 student 应该在同一个 student-induced state 上比较 velocity。不要把 action transition 改回 teacher endpoint rollout 后仍保留 velocity loss；如果切 endpoint 语义，transition 参数也要对应回 x0/endpoint。

### LIBERO i2va 直接生成长视频

仓库里保留了 LingBot 风格的 i2va demo 入口：

- `wan_va/configs/va_libero_i2va.py`: LIBERO i2va 配置。
- `wan_va/configs/__init__.py`: 注册 `libero_i2av`。
- `wan_va/wan_va_server.py`: `generate()`、`load_init_obs()`、`decode_one_video()`。
- `wan_va/configs/va_libero_cfg.py`: LIBERO 推理必须保持 `action_downsample_factor=4`，和 Stage 1 训练对齐。

这个模式会生成多个 chunk 的 latent，先沿时间维拼 latent，再一次性 VAE decode 成 `demo.mp4`。这比逐段 decode 后再拼 MP4 更接近原 LingBot demo。它是生成视频，不是 LIBERO 环境 rollout。

Target student 示例：

```bash
source /kpfs-intern/jialongliu/miniforge3/bin/activate && conda activate flashwam && cd /kpfs-intern/jialongliu/projects/Flash-WAM && WAN22_PRETRAINED_PATH=/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29616 wan_va/wan_va_server.py --config-name libero_i2av --checkpoint-path /kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_fullft_stage1_warmup/checkpoints/step_1000/target_student/transformer --num-steps 8 --action-num-steps 8 --input-img-path /kpfs-intern/jialongliu/projects/lingbot-va/example/libero --num-chunks-to-infer 10 --save-root /kpfs-intern/jialongliu/projects/Flash-WAM/evaluation/outputs/libero_i2va_stage1_step1000_target_aligned_s8
```

Online student 只需要把 checkpoint 和输出目录换成：

```bash
--checkpoint-path /kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_fullft_stage1_warmup/checkpoints/step_1000/online_student/transformer
--save-root /kpfs-intern/jialongliu/projects/Flash-WAM/evaluation/outputs/libero_i2va_stage1_step1000_online_aligned_s8
```

判断 checkpoint 是否正确加载，看日志里是否有：

```text
Full model weights loaded: 847 keys, 0 missing, 0 unexpected
FlowMap inference kwargs: {..., action_downsample_factor: 4}
```

Stage 1 `step_1000` 的 `2-step` i2va 可能仍然偏糊，`8-step`/`16-step` 更适合作诊断。2-step 糊不代表 checkpoint 没加载。


## 常见问题

### 1. 显存不足

如果遇到 OOM，可以尝试：
- 减小 `batch_size`（已设为 1）
- 增加 `gradient_accumulation_steps`（已设为 8）
- Stage 1 可以按需启用 `gradient_checkpointing`
- Stage 2 OPD 默认必须保持 `GRADIENT_CHECKPOINTING=0`；如果显存不够，优先调 batch/accumulation，不要直接打开 checkpointing，否则可能触发 FSDP2 DTensor recompute 报错

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
