"""
Flash-WAM Flow Map 蒸馏配置文件 —— LIBERO 环境

基于 config.py（robotwin），修改为 libero 环境的参数：
  - 教师模型：libero 后训练 checkpoint
  - 数据集：robbyant/libero-long-lerobot（HuggingFace）
  - 分辨率：128×128（libero 标准）
  - 相机：agentview_rgb + eye_in_hand_rgb（2 个）
  - 动作：7 维（单臂 6 关节 + 夹爪）
  - action_per_frame=4, attn_window=30, frame_chunk_size=4
  - LoRA 微调：rank=256, alpha=128
"""

import os
import torch
from easydict import EasyDict

cfg = EasyDict(__name__="Config: Flash-WAM FlowMap Distillation (LIBERO)")

# ============================================================
# 路径配置
# ============================================================
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)

# 教师模型路径：libero 后训练 checkpoint
cfg.teacher_model_path = os.environ.get(
    "TEACHER_PATH",
    os.path.join(_project_root, "checkpoints", "libero"))

# 输出目录
cfg.output_dir = os.environ.get(
    "OUTPUT_DIR", os.path.join(_this_dir, "output_libero_0613"))

# 训练数据集路径（下载后的 libero lerobot 数据集）
cfg.dataset_path = os.environ.get(
    "DATASET_PATH",
    os.path.join(_project_root, "training_data", "libero_long_lerobot"))

# 空文本嵌入路径
cfg.empty_emb_path = os.path.join(cfg.dataset_path, "empty_emb.pt")

# ============================================================
# 模型架构配置（来自 va_libero_cfg）
# ============================================================
cfg.patch_size = (1, 2, 2)
cfg.param_dtype = torch.bfloat16
cfg.env_type = "none"               # libero 环境类型
cfg.height = 128                    # libero 分辨率 128×128
cfg.width = 128
cfg.num_frames = 128                 # 从 128 减半，latent_T ≈ 16，显著加速
cfg.action_dim = 30                 # 动作维度（与 robotwin 一致，但只用前 7 维）
cfg.action_per_frame = 4            # libero: 每帧 4 步动作
cfg.frame_chunk_size = 4            # libero: 帧分块大小 4
cfg.attn_window = 30                # libero: 注意力窗口 30

# 使用的相机视角键名（libero: 2 个摄像头）
cfg.obs_cam_keys = [
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
]

# 使用的动作通道 ID 列表（libero: 单臂 7 维 = 6 关节 + 夹爪）
cfg.used_action_channel_ids = list(range(0, 7))

# 反向映射
_inv = [len(cfg.used_action_channel_ids)] * cfg.action_dim
for _i, _j in enumerate(cfg.used_action_channel_ids):
    _inv[_j] = _i
cfg.inverse_used_action_channel_ids = _inv

# 动作归一化（来自 va_libero_cfg）
cfg.action_norm_method = "quantiles"
cfg.norm_stat = {
    "q01": [
        -0.6589285731315613, -0.84375, -0.9375,
        -0.12107142806053162, -0.15964286029338837,
        -0.26571428775787354, -1.0,
    ] + [0.0] * 23,
    "q99": [
        0.8999999761581421, 0.8544642925262451, 0.9375,
        0.17142857611179352, 0.1842857152223587,
        0.34392857551574707, 1.0,
    ] + [0.0] * 23,
}

# ============================================================
# FlowMatch 调度器配置
# ============================================================
cfg.snr_shift = 5.0                 # 视频 SNR 偏移
cfg.action_snr_shift = 0.05         # libero 动作 SNR 偏移（比 robotwin 小很多）
cfg.num_train_timesteps = 1000

# ============================================================
# LCM 蒸馏核心参数
# ============================================================
cfg.num_ddim_timesteps = 2

cfg.distill_mode = os.environ.get("DISTILL_MODE", "flashwam")

_mode = cfg.distill_mode
cfg.distill_video = _mode in ("video", "joint", "video_action_aware", "flashwam")
cfg.distill_action = _mode in ("action", "joint", "flashwam")
cfg.action_aware = _mode in ("video_action_aware", "flashwam")

cfg.num_ddim_timesteps_action = 2
cfg.action_loss_weight = 1.0
cfg.action_distill_mode = "x0"
cfg.action_aware_weight = 0.01

# ============================================================
# Flow Map 蒸馏参数
# ============================================================
cfg.diffusion_ratio = 0.5
cfg.consistency_ratio = 0.25
cfg.flowmap_ratio = 0.25

cfg.epsilon = 1.0
cfg.gate_value = 0.0
cfg.deltatime_type = 'r'
cfg.weight_type = 'beta08'
cfg.gt_regression_weight = 0.1

# ============================================================
# 消融实验开关
# ============================================================
cfg.use_flowmap = True
cfg.use_gt_regression = True
cfg.use_central_diff = True
cfg.selective_cdiff = True
cfg.action_use_flowmap = False
cfg.action_epsilon = getattr(cfg, "epsilon", 1.0)  # action-side central-difference radius


# ============================================================
# DMD（On-Policy Distribution Matching Distillation）参数
# ============================================================
cfg.use_dmd = False                 # 是否启用 DMD 第二阶段（默认关闭，需要先完成第一阶段）
cfg.dmd_weight = 0.1                # DMD loss 权重（相对于主 loss）
cfg.dmd_warmup_steps = 0            # DMD 预热步数（从 checkpoint 恢复时设为 0）
cfg.dmd_rollout_steps_min = 2       # on-policy rollout 最小步数
cfg.dmd_rollout_steps_max = 8       # on-policy rollout 最大步数
cfg.dmd_cfg_scale = 5.0             # rollout 时的 CFG 引导强度
cfg.dmd_discriminator_lr = 1e-5     # 判别器学习率（通常比学生学习率大 2x）
cfg.dmd_discriminator_steps = 1     # 判别器更新频率（每 N 个梯度累积步）
cfg.dmd_discriminator_warmup = 200  # 判别器预热步数（先只训判别器，不注入 DMD 梯度）
cfg.dmd_hidden_dim = 256            # 判别器隐藏维度
cfg.dmd_num_layers = 4              # 判别器 Transformer 层数
cfg.dmd_num_heads = 8               # 判别器注意力头数
cfg.dmd_dropout = 0.1               # 判别器 dropout

# ============================================================
# LoRA 配置（libero 微调默认开启）
# ============================================================
cfg.use_lora = True                 # libero 微调默认启用 LoRA
cfg.lora_rank = 128
cfg.lora_alpha = 64
cfg.lora_dropout = 0.0
cfg.lora_target_modules = [
    "to_q", "to_k", "to_v",
    "to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
    "time_proj",
    "delta_embedder.linear_1",
    "delta_embedder.linear_2",
]

# ============================================================
# LCM 超参数
# ============================================================
cfg.ema_decay = 0.995
cfg.loss_type = "huber"
cfg.huber_c = 0.001
cfg.sigma_data = 0.5
cfg.cfg_min = 2.0
cfg.cfg_max = 10.0

# ============================================================
# 训练超参数
# ============================================================
cfg.learning_rate = 5e-6
cfg.beta1 = 0.9
cfg.beta2 = 0.999
cfg.weight_decay = 0.0
cfg.max_grad_norm = 2.0
cfg.warmup_steps = 100
cfg.max_train_steps = 10000
cfg.batch_size = 1
cfg.gradient_accumulation_steps = 2
cfg.load_worker = 2

cfg.noisy_cond_prob = 0.0
cfg.cfg_prob = 0.0

# ============================================================
# 检查点与日志
# ============================================================
cfg.save_interval = 500
cfg.gc_interval = 50
cfg.enable_wandb = True
cfg.wandb_entity = None
cfg.seed = 42
