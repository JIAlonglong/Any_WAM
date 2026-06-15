"""
Flash-WAM 配置文件：模态感知的 LCM 蒸馏配置（LingBot-VA 联合视频+动作）

核心设计思路：
  - 原始的 1000 步 FlowMatch schedule 被压缩为 25 个锚点
  - 这 25 个锚点是从 1000 步 schedule 中均匀采样得到的
  - 所有噪声添加和时间步嵌入仍然使用原始的 1000 步系统
  - LCM 蒸馏的目标是让模型只需 2 步即可生成高质量结果
"""

import os
import torch
from easydict import EasyDict

cfg = EasyDict(__name__="Config: LCM Video Distillation v2")

# ============================================================
# 路径配置
# ============================================================
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)

# 教师模型路径（预训练好的 LingBot-VA 模型）
cfg.teacher_model_path = os.environ.get(
    "TEACHER_PATH", os.path.join(_project_root, "checkpoints", "lingbot-va-posttrain-robotwin"))

# 输出目录（存放蒸馏过程中的检查点和日志）
cfg.output_dir = os.environ.get(
    "OUTPUT_DIR", os.path.join(_this_dir, "output"))

# 训练数据集路径（包含 latent 表示的 LeRobot 数据集）
cfg.dataset_path = os.environ.get(
    "DATASET_PATH", os.path.join(_project_root, "training_data", "lerobot_robotwin_eef_aug_500"))

# 空文本嵌入路径（用于 CFG 无条件推理时替换文本嵌入）
cfg.empty_emb_path = os.path.join(cfg.dataset_path, "empty_emb.pt")

# ============================================================
# 模型架构配置（来自 va_robotwin_cfg）
# ============================================================
cfg.patch_size = (1, 2, 2)          # 视频 token 的 patch 大小：时间×高×宽
cfg.param_dtype = torch.bfloat16    # 模型参数精度（bf16 节省显存）
cfg.env_type = "robotwin_tshape"    # 环境类型（机器人操作场景）
cfg.height = 256                    # 视频高度（像素）
cfg.width = 320                     # 视频宽度（像素）
cfg.action_dim = 30                 # 动作维度（机器人自由度）
cfg.action_per_frame = 16           # 每帧对应的动作步数
cfg.frame_chunk_size = 2            # 帧分块大小（注意力窗口的分块单位）
cfg.attn_window = 72                # 注意力窗口大小（滑动窗口注意力）

# 使用的相机视角键名（3 个摄像头：俯视、左腕、右腕）
cfg.obs_cam_keys = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]

# 使用的动作通道 ID 列表
# 前 7 维：左臂关节 + 夹爪；后 7 维：右臂关节 + 夹爪
cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))

# 反向映射：从完整动作维度到实际使用通道的索引
_inv = [len(cfg.used_action_channel_ids)] * cfg.action_dim
for _i, _j in enumerate(cfg.used_action_channel_ids):
    _inv[_j] = _i
cfg.inverse_used_action_channel_ids = _inv

# 动作归一化方法（使用分位数归一化）
cfg.action_norm_method = "quantiles"
cfg.norm_stat = {
    # q01: 1% 分位数（动作范围下界）
    "q01": [
        -0.06172713458538055, -3.6716461181640625e-05, -0.08783501386642456,
        -1, -1, -1, -1, -0.3547105032205582, -1.3113021850585938e-06,
        -0.11975435614585876, -1, -1, -1, -1,
    ] + [0.0] * 16,
    # q99: 99% 分位数（动作范围上界）
    "q99": [
        0.3462600058317184, 0.39966784834861746, 0.14745532035827624, 1, 1, 1,
        1, 0.034201726913452024, 0.39142737388610793, 0.1792279863357542, 1, 1,
        1, 1,
    ] + [0.0] * 14 + [1.0, 1.0],
}

# ============================================================
# FlowMatch 调度器配置（与原生训练保持一致）
# ============================================================
cfg.snr_shift = 5.0                 # 视频的 SNR 偏移系数（控制噪声 schedule 的形状）
cfg.action_snr_shift = 1.0          # 动作的 SNR 偏移系数
cfg.num_train_timesteps = 1000      # 训练时的时间步总数（1000 步完整 schedule）

# ============================================================
# LCM 蒸馏核心参数
# ============================================================
# 2 个锚点 → k = 1000/2 = 500 步长（目标：2 步生成）
cfg.num_ddim_timesteps = 2

# 蒸馏模式选择：
#   flashwam           — Flash-WAM（论文方法）：模态感知联合蒸馏
#                        = 视频一致性损失 + 动作一致性损失 + 动作 MSE 正则
#   joint              — 天真联合 LCM：视频一致性 + 动作一致性（消融实验）
#   video              — 仅视频 LCM 一致性损失（消融实验）
#   video_action_aware — 仅视频 LCM + 小权重动作 MSE 正则（消融实验）
#   action             — 仅动作一致性，student 初始化为 video-LCM 检查点
cfg.distill_mode = os.environ.get("DISTILL_MODE", "flashwam")

_mode = cfg.distill_mode
cfg.distill_video = _mode in ("video", "joint", "video_action_aware", "flashwam")   # 是否蒸馏视频
cfg.distill_action = _mode in ("action", "joint", "flashwam")                        # 是否蒸馏动作
cfg.action_aware = _mode in ("video_action_aware", "flashwam")                       # 是否使用动作感知正则

cfg.num_ddim_timesteps_action = 2   # 动作的锚点数（k_action = 1000/2 = 500）
cfg.action_loss_weight = 1.0        # 动作一致性损失的权重
cfg.action_distill_mode = "x0"      # 动作一致性函数的参数化方式（"x0" 直接预测干净样本）
cfg.action_aware_weight = 0.01      # 动作感知正则化损失的权重（较小，仅起辅助作用）

# ============================================================
# LCM 超参数
# ============================================================
cfg.ema_decay = 0.995               # EMA 衰减系数（target student 的更新速度）
cfg.loss_type = "huber"             # 损失类型："huber"（鲁棒）或 "l2"（MSE）
cfg.huber_c = 0.001                 # Huber 损失的阈值参数
cfg.sigma_data = 0.5                # 数据噪声水平（用于边界条件缩放）
cfg.cfg_min = 2.0                   # 教师 CFG 引导强度的最小值
cfg.cfg_max = 10.0                  # 教师 CFG 引导强度的最大值

# ============================================================
# 训练超参数
# ============================================================
cfg.learning_rate = 5e-6            # 学习率
cfg.beta1 = 0.9                     # AdamW 的 beta1
cfg.beta2 = 0.999                   # AdamW 的 beta2
cfg.weight_decay = 0.0              # 权重衰减
cfg.max_grad_norm = 2.0             # 梯度裁剪范数上限
cfg.warmup_steps = 100              # 学习率预热步数
cfg.max_train_steps = 10000         # 最大训练步数
cfg.batch_size = 1                  # 每个 GPU 的 batch size
cfg.gradient_accumulation_steps = 8 # 梯度累积步数（等效 batch = 1×8 = 8）
cfg.load_worker = 0                 # 数据加载的 worker 数量

# 训练时的数据增强概率（蒸馏时不使用）
cfg.noisy_cond_prob = 0.0           # 条件加噪概率（蒸馏时关闭）
cfg.cfg_prob = 0.0                  # CFG 随机丢弃概率（蒸馏时关闭，教师显式处理 CFG）

# ============================================================
# 检查点与日志
# ============================================================
cfg.save_interval = 1000            # 每隔多少步保存一次检查点
cfg.gc_interval = 50                # 每隔多少步做一次垃圾回收和显存清理
cfg.enable_wandb = True             # 是否启用 WandB 日志记录
cfg.wandb_entity = None             # WandB 实体名（团队/个人）
cfg.seed = 42                       # 随机种子
