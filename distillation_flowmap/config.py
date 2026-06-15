"""
Flash-WAM Flow Map 蒸馏配置文件（基于 AnyFlow 核心思想）

核心设计思路：
  - 原始的 1000 步 FlowMatch schedule 被压缩为 25 个锚点
  - 这 25 个锚点是从 1000 步 schedule 中均匀采样得到的
  - 所有噪声添加和时间步嵌入仍然使用原始的 1000 步系统
  - LCM 蒸馏的目标是让模型只需 2 步即可生成高质量结果

与原配置（distillation/config.py）的差异：
  - 新增 Flow Map 蒸馏相关参数：扩散目标、一致性目标、流映射目标的混合比例
  - 新增中心差分扰动步长 epsilon（用于计算 flow map 的数值梯度）
  - 新增 delta_emb_gate 初始值（控制 r 信息的使用时机）
  - 新增 deltatime_type（控制 delta 时间步的计算方式）
  - 新增 weight_type（时间步采样权重策略）
  - 新增 gt_regression_weight（GT 回归 loss 的辅助权重）
  - loss_type 改为 "l2"（FlowMap 蒸馏更适合 L2 损失）
  - distill_mode 默认为 "flashwam"（与原配置一致）
"""

import os
import torch
from easydict import EasyDict

cfg = EasyDict(__name__="Config: Flash-WAM FlowMap Distillation")

# ============================================================
# 路径配置（与原配置完全一致）
# ============================================================
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)

# 教师模型路径（预训练好的 LingBot-VA 模型）
cfg.teacher_model_path = os.environ.get(
    "TEACHER_PATH", os.path.join(_project_root, "checkpoints", "base"))

# 输出目录（存放蒸馏过程中的检查点和日志）
cfg.output_dir = os.environ.get(
    "OUTPUT_DIR", os.path.join(_this_dir, "output"))

# 训练数据集路径（包含 latent 表示的 LeRobot 数据集）
cfg.dataset_path = os.environ.get(
    "DATASET_PATH", os.path.join(_project_root, "training_data", "lerobot_robotwin_eef_aug_500"))

# 空文本嵌入路径（用于 CFG 无条件推理时替换文本嵌入）
cfg.empty_emb_path = os.path.join(cfg.dataset_path, "empty_emb.pt")

# ============================================================
# 模型架构配置（来自 va_robotwin_cfg，与原配置完全一致）
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
# FlowMatch 调度器配置（与原配置完全一致）
# ============================================================
cfg.snr_shift = 5.0                 # 视频的 SNR 偏移系数（控制噪声 schedule 的形状）
cfg.action_snr_shift = 1.0          # 动作的 SNR 偏移系数
cfg.num_train_timesteps = 1000      # 训练时的时间步总数（1000 步完整 schedule）

# ============================================================
# LCM 蒸馏核心参数（与原配置一致，但 loss_type 有变更）
# ============================================================
# 2 个锚点 → k = 1000/2 = 500 步长（目标：2 步生成）
# 注意：FlowMap 蒸馏使用混合采样（diffusion + consistency + flowmap），
# 不再严格依赖 num_ddim_timesteps，但保留此参数用于向后兼容和计算 k 值。
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
cfg.action_aware_weight = 0.1       # 动作感知正则化损失的权重（较小，仅起辅助作用）
                                    # 改进：从 0.01 增加到 0.1，加强动作正则化

# ============================================================
# Flow Map 蒸馏参数（新增 —— 基于 AnyFlow 核心思想）
# ============================================================
# 这些参数控制 Flow Map 蒸馏中三种目标的混合比例：
#   1. 扩散目标（r=t）：标准 FlowMatch 去噪目标，学习从噪声到数据的映射
#   2. 一致性目标（r=0）：LCM 端点映射，学习从噪声直接到数据的一致性函数
#   3. 流映射目标（r<t）：AnyFlow 核心，学习任意中间时间步之间的流映射
#
# 三种目标的比例之和应为 1.0，通过随机采样 r 来决定每次迭代使用哪种目标。

cfg.diffusion_ratio = 0.5           # 扩散目标占比（r=t，标准 FlowMatch）
                                     # 当采样的 r 比例落入此区间时，使用标准扩散去噪目标
cfg.consistency_ratio = 0.25         # 一致性目标占比（r=0，LCM 端点映射）
                                     # 当采样的 r 比例落入此区间时，使用 LCM 一致性目标
cfg.flowmap_ratio = 0.25            # 流映射目标占比（r<t，AnyFlow 核心）
                                     # 当采样的 r 比例落入此区间时，使用 flow map 中间映射目标

cfg.epsilon = 1.0                   # 中心差分的扰动步长
                                     # 用于计算 flow map 的数值梯度：f(x+eps) - f(x-eps) / (2*eps)
                                     # 较大的 epsilon 提供更稳定的梯度估计，但可能引入偏差

cfg.gate_value = 0.1                # delta_emb_gate 初始值（0 = 初期不使用 r 信息）
                                     # 控制时间差嵌入（delta embedding）的初始门控值
                                     # 设为 0 表示训练初期不引入 r 相关信息，随训练逐渐打开
                                     # 改进：从 0.0 增加到 0.1，让模型尽早学习 r 信息

cfg.deltatime_type = 'r'            # delta 时间步类型
                                     # 'r'   — 使用 r 作为 delta 时间步（直接使用采样的中间时间步）
                                     # 't-r' — 使用 t-r 作为 delta 时间步（使用从起点到 r 的距离）

cfg.weight_type = 'beta08'          # 时间步采样权重策略
                                     # 'gaussian' — 高斯权重，中间时间步权重较高
                                     # 'beta08'   — Beta(0.8, 0.8) 分布权重，两端权重较高（推荐）
                                     # 'uniform'  — 均匀权重，所有时间步等概率采样

cfg.gt_regression_weight = 0.5      # GT 回归 loss 权重
                                     # 辅助损失：直接回归到 ground truth 的 MSE 损失
                                     # 改进：从 0.1 增加到 0.5，加强 GT 回归的正则化作用

# ============================================================
# 消融实验开关（用于控制不同组件的启用/禁用）
# ============================================================
# 这些开关用于消融实验，验证每个组件的独立贡献：
#   Ablation 1: use_flowmap=False, use_gt_regression=True  → 纯 GT 回归
#   Ablation 2: use_flowmap=True,  use_gt_regression=True  → 流映射 + GT 回归
#   Ablation 3: use_flowmap=True,  use_gt_regression=False → 流映射（无 GT 回归）
#   Ablation 4: gate_value=0.0/0.1/0.5                    → 不同 gate 初始值
#   Ablation 5: epsilon=0.1/0.5/1.0                       → 不同中心差分步长
#   Ablation 6: selective_cdiff=True/False                 → 选择性 vs 全量中心差分

cfg.use_flowmap = True              # 是否使用流映射目标（False = 退化为 LCM 一致性）
                                    # 控制训练时是否采样 r<t 的中间映射目标
                                    # 关闭后，flowmap_ratio 自动失效，仅保留 diffusion + consistency

cfg.use_gt_regression = True        # 是否使用 GT 动作回归 loss
                                    # 开启时，额外计算动作的 MSE 回归损失（权重由 gt_regression_weight 控制）
                                    # 关闭后，仅依赖一致性目标学习动作映射

cfg.use_central_diff = True         # 是否使用中心差分法（False = 不计算 dF/dt，退化为标准 FlowMatch）
                                    # 中心差分是 AnyFlow 流映射目标的核心：通过数值微分
                                    # f(x+eps) - f(x-eps) / (2*eps) 近似 flow map 的梯度
                                    # 关闭后，流映射目标退化为标准的 FlowMatch 单点去噪

cfg.selective_cdiff = True          # 是否使用选择性中心差分（True = 只对流映射 batch 计算）
                                    # True  — 仅对采样到流映射目标（r<t）的 batch 计算中心差分，
                                    #          节约计算开销（扩散和一致性 batch 不需要）
                                    # False — 对所有 batch 都计算中心差分（消融用，验证是否需要全量计算）

cfg.action_use_flowmap = False      # 动作是否使用流映射目标（False = 动作用 GT 回归 + x0）
                                    # 动作维度低（30 维），中心差分信号可能不稳定，默认关闭
                                    # True  — 动作也使用流映射中间目标（与视频对称）
                                    # False — 动作仅使用 GT 回归 + x0 一致性目标

cfg.use_action_distill = True       # 是否对 action 使用教师蒸馏（核心改进）
                                    # True  — 使用教师 Euler 步推进 + target_student 生成目标（原始 LCM 蒸馏）
                                    # False — 直接使用 GT 动作作为目标（原始 FlowMap 行为）
                                    # 这是解决阶段1 action 质量差的关键开关

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
# LoRA 配置（可选 —— 用于降低显存开销）
# ============================================================
# 启用 LoRA 后，只训练 LoRA adapter 参数（~几十 M）而非全量参数（~1.3B），
# 显著降低显存占用，可以跑更大 batch 或更大模型。
# 与 AnyFlow 的 LoRA 策略一致（rank=256, alpha=128）。

cfg.use_lora = False                # 是否启用 LoRA（False = 全量微调，与原 Flash-WAM 一致）
cfg.lora_rank = 256                 # LoRA 秩（rank），越大表达能力越强，显存开销越高
cfg.lora_alpha = 128                # LoRA 缩放系数（alpha/rank = 实际缩放比例）
cfg.lora_dropout = 0.0              # LoRA dropout（0 = 不丢弃）
cfg.lora_target_modules = [         # LoRA 目标模块名称（匹配 nn.Linear 的名字）
    "to_q", "to_k", "to_v",        # 自注意力的 Q/K/V 投影
    "to_out.0",                     # 自注意力输出投影
    "ffn.net.0.proj",               # FFN 第一层（GEGLU 中的 Linear）
    "ffn.net.2",                    # FFN 第二层（输出 Linear）
    "time_proj",                    # 时间步投影（condition_embedder 中）
    "delta_embedder.linear_1",      # delta 时间步嵌入第一层（Flow Map 新增）
    "delta_embedder.linear_2",      # delta 时间步嵌入第二层（Flow Map 新增）
]

# ============================================================
# LCM 超参数（与原配置基本一致，loss_type 有变更）
# ============================================================
cfg.ema_decay = 0.995               # EMA 衰减系数（target student 的更新速度）
cfg.loss_type = "l2"                # 损失类型：FlowMap 蒸馏使用 "l2"（MSE），而非原配置的 "huber"
                                     # 原因：FlowMap 的混合目标对损失函数更敏感，L2 更稳定
cfg.huber_c = 0.001                 # Huber 损失的阈值参数（保留向后兼容，l2 模式下不使用）
cfg.sigma_data = 0.5                # 数据噪声水平（用于边界条件缩放）
cfg.cfg_min = 2.0                   # 教师 CFG 引导强度的最小值
cfg.cfg_max = 10.0                  # 教师 CFG 引导强度的最大值

# ============================================================
# 训练超参数（与原配置完全一致）
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
cfg.load_worker = 8                 # 数据加载的 worker 数量
                                    # 改进：从 0 增加到 8，避免数据加载成为瓶颈
cfg.pin_memory = True               # 启用 pin_memory 加速 CPU->GPU 传输
cfg.prefetch_factor = 4             # 预取 4 个 batch
cfg.persistent_workers = True       # 保持 worker 进程，避免每 epoch 重新 fork
cfg.cache_dataset_in_memory = True  # 缓存数据集到内存（如果内存允许）

cfg.use_torch_compile = False       # 是否使用 torch.compile 加速（实验性）
                                    # True  — 使用 torch.compile 编译学生模型，可能加速 10-30%
                                    # False — 不使用编译（默认，更稳定）
                                    # 注意：torch.compile 可能导致某些 PyTorch 版本下出现问题

# 训练时的数据增强概率（蒸馏时不使用）
cfg.noisy_cond_prob = 0.0           # 条件加噪概率（蒸馏时关闭）
cfg.cfg_prob = 0.0                  # CFG 随机丢弃概率（蒸馏时关闭，教师显式处理 CFG）

# ============================================================
# 检查点与日志（与原配置完全一致）
# ============================================================
cfg.save_interval = 1000            # 每隔多少步保存一次检查点
cfg.gc_interval = 50                # 每隔多少步做一次垃圾回收和显存清理
cfg.enable_wandb = True             # 是否启用 WandB 日志记录
cfg.wandb_entity = None             # WandB 实体名（团队/个人）
cfg.seed = 42                       # 随机种子
