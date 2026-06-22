"""
Flash-WAM Flow Map 蒸馏配置文件 —— LIBERO 环境（Stage 2 优化版）

基于 config_libero_optimized.py（Stage 1），启用 DMD 进行第二阶段训练：
  - 启用 DMD（Distribution Matching Distillation）
  - 降低学习率（Stage 2 微调需要更精细的学习率）
  - 从 Stage 1 checkpoint 恢复训练

Stage 2: FlowMap 蒸馏 + DMD 对抗训练
"""

import os
import torch
from easydict import EasyDict

cfg = EasyDict(__name__="Config: Flash-WAM FlowMap Distillation (LIBERO Stage 2)")

# ============================================================
# 路径配置
# ============================================================
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)

cfg.teacher_model_path = os.environ.get(
    "TEACHER_PATH",
    os.path.join(_project_root, "checkpoints", "libero"))

cfg.output_dir = os.environ.get(

    "OUTPUT_DIR", os.path.join(_this_dir, "output_libero_optimized_stage2"))

# Stage 1 checkpoint resume (LoRA adapter weights)
cfg.resume_from_path = os.path.join(_project_root, "checkpoints", "lingbot_va_flowmap_distill", "flowmap_distill_stage1")
cfg.resume_from_step = 700

cfg.dataset_path = os.environ.get(
    "DATASET_PATH",
    os.path.join(_project_root, "training_data", "libero-long-lerobot"))

cfg.empty_emb_path = os.path.join(cfg.dataset_path, "empty_emb.pt")

# ============================================================
# 模型架构配置（与 Stage 1 一致）
# ============================================================
cfg.patch_size = (1, 2, 2)
cfg.param_dtype = torch.bfloat16
cfg.env_type = "none"
cfg.height = 128
cfg.width = 128
cfg.num_frames = 32                # 與 Stage 1 一致
cfg.action_dim = 30
cfg.action_per_frame = 4
cfg.frame_chunk_size = 4
cfg.attn_window = 30

cfg.obs_cam_keys = [
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
]

cfg.used_action_channel_ids = list(range(0, 7))

_inv = [len(cfg.used_action_channel_ids)] * cfg.action_dim
for _i, _j in enumerate(cfg.used_action_channel_ids):
    _inv[_j] = _i
cfg.inverse_used_action_channel_ids = _inv

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
# FlowMatch 调度器配置（与 Stage 1 一致）
# ============================================================
cfg.snr_shift = 5.0
cfg.action_snr_shift = 0.05
cfg.num_train_timesteps = 1000

# ============================================================
# LCM 蒸馏核心参数（与 Stage 1 一致）
# ============================================================
cfg.num_ddim_timesteps = 2

cfg.distill_mode = os.environ.get("DISTILL_MODE", "flashwam")

_mode = cfg.distill_mode
cfg.distill_video = _mode in ("video", "joint", "video_action_aware", "flashwam", "onpolicy_transition")
cfg.distill_action = _mode in ("action", "joint", "flashwam", "onpolicy_transition")
cfg.action_aware = _mode in ("video_action_aware", "flashwam", "onpolicy_transition")

cfg.num_ddim_timesteps_action = 2
cfg.action_loss_weight = 5.0      # 适度提高 action 重要性
cfg.action_distill_mode = "x0"
cfg.action_aware_weight = 0.5     # 同步提高到 5x

# ============================================================
# Flow Map 蒸馏参数（与 Stage 1 一致）
# ============================================================
cfg.diffusion_ratio = 0.5
cfg.consistency_ratio = 0.25
cfg.flowmap_ratio = 0.25

cfg.epsilon = 5.0
cfg.gate_value = 0.25                # 提升 delta_embedder 贡献 (原 0.1)
cfg.deltatime_type = 'r'
cfg.weight_type = 'beta08'
cfg.gt_regression_weight = 1.0    # 同步提高到 2x

# Video loss 权重（Stage 2 新增）
# video loss 在潜空间计算，值域（100-1600）比其他 loss（0.1-5）大 100 倍
# 设置 0.1 使 video loss 贡献与其他 loss 平衡
cfg.video_loss_weight = 0.1

# ============================================================
# 消融实验开关（与 Stage 1 一致）
# ============================================================
cfg.use_flowmap = True
cfg.use_gt_regression = True
cfg.use_central_diff = True
cfg.selective_cdiff = True
cfg.action_use_flowmap = False
cfg.use_action_distill = True       # DMD 需要动作蒸馏

# ============================================================
# On-Policy Transition Matching 参数
# ============================================================
cfg.use_onpolicy_transition = (_mode == "onpolicy_transition")
cfg.rollout_step_pairs = [
    [1, 1],
    [1, 1],
    [1, 2],  # Change 3: 偏向 [1,1] 减少 teacher 前向次数
    ]
cfg.teacher_micro_steps = 1
cfg.teacher_solver = "euler"
cfg.transition_loss_type = "huber"
cfg.transition_huber_c = 5e-4  # 缩小平坦区，恢复梯度
cfg.video_transition_weight = 1.0
cfg.local_fm_weight = 0.05
cfg.action_state_mode = "data"
cfg.action_onpolicy_prob = 0.0
cfg.composition_weight = 0.0
cfg.onpolicy_warmup_steps = 0

# ============================================================
# DMD 参数（Stage 2 启用，仅 flashwam 模式使用）
# ============================================================
cfg.use_dmd = (_mode == "flashwam")   # 仅 flashwam 模式启用 DMD
cfg.dmd_weight = 0.1
cfg.dmd_warmup_steps = 0            # DMD 从第一步开始

# ============================================================
# Stage 2 继训注意事项
# ============================================================
# 若从旧 checkpoint (gate=0.1) resume，需手动覆盖 gate 参数：
#   model.condition_embedder.delta_emb_gate.data.fill_(0.3)
#   model.condition_embedder_action.delta_emb_gate.data.fill_(0.3)
# 或在 flowmap_trainer.py 的 load_checkpoint 后添加上述逻辑。
# 注意: gate 已改为 register_buffer(persistent=False)，checkpoint 不保存 gate，
#        resume 时自动使用 config 中的 gate_value，无需手动覆盖。
cfg.dmd_rollout_steps_min = 2
cfg.dmd_rollout_steps_max = 8
cfg.dmd_cfg_scale = 5.0
cfg.dmd_discriminator_lr = 1e-5
cfg.dmd_discriminator_steps = 1
cfg.dmd_discriminator_warmup = 1  # 判别器预热 200 步
cfg.dmd_hidden_dim = 256
cfg.dmd_num_layers = 4
cfg.dmd_num_heads = 8
cfg.dmd_dropout = 0.1

# ============================================================
# LoRA 配置（与 Stage 1 一致）
# ============================================================
cfg.use_lora = True
cfg.lora_rank = 512
cfg.lora_alpha = 256
cfg.lora_dropout = 0.0
cfg.lora_target_modules = [
    "to_q", "to_k", "to_v",
    "to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
    "time_proj",
    "delta_embedder.linear_1",
    "delta_embedder.linear_2",
    "time_embedder.linear_1",
    "time_embedder.linear_2",
]

# ============================================================
# LCM 超参数（与 Stage 1 一致）
# ============================================================
cfg.ema_decay = 0.995
cfg.loss_type = "huber"
cfg.huber_c = 0.0005  # 缩小平坦区
cfg.sigma_data = 0.5
cfg.cfg_min = 2.0
cfg.cfg_max = 10.0

# ============================================================
# 训练超参数（Stage 2 优化版）
# ============================================================
cfg.learning_rate = 5e-6            # Stage 2: scaled by sqrt(2) for batch_size=2
cfg.beta1 = 0.9
cfg.beta2 = 0.999
cfg.weight_decay = 0.0
cfg.max_grad_norm = 1.0             # 与 Stage 1 一致
cfg.warmup_steps = 100              # Stage 2: 减少 warmup（已有良好初始化）
cfg.max_train_steps = 20000         # Stage 2: 更多 steps 让 delta_embedder 收敛
cfg.batch_size = 1  # bs=2 + checkpointing: mask bug unresolvable
cfg.gradient_accumulation_steps = 2                 # Change 1: 减半以配合 batch_size=2
cfg.load_worker = 8                 # 与 Stage 1 一致
cfg.pin_memory = True
cfg.prefetch_factor = 4
cfg.cache_dataset_in_memory = True
cfg.use_torch_compile = False  # compile CUDA graphs + kernel cache OOM even with bs=1       # Change 2: 重新启用 torch.compile（无 checkpointing 时安全）

# Loss clipping（与 Stage 1 一致）
cfg.loss_clip_value = 10.0
cfg.loss_clip_enabled = True

cfg.noisy_cond_prob = 0.0
cfg.cfg_prob = 0.0

# Action 时间下采样（与 Stage 1 一致）
cfg.action_downsample_factor = 4

# ============================================================
# 检查点与日志
# ============================================================
cfg.save_interval = 100      # 每 100 步保存一次 checkpoint
cfg.gc_interval = 50
cfg.enable_wandb = True
cfg.wandb_entity = None
cfg.seed = 42
