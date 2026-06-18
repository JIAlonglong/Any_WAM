"""
Flash-WAM Flow Map 蒸馏配置文件 —— LIBERO 环境（Stage 2 无 DMD 版本）

用于测试禁用 DMD 后的训练速度，对比 DMD 的开销。
"""

import os
import torch
from easydict import EasyDict

cfg = EasyDict(__name__="Config: Flash-WAM FlowMap Distillation (LIBERO Stage 2 No DMD)")

# ============================================================
# 路径配置
# ============================================================
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)

cfg.teacher_model_path = os.environ.get(
    "TEACHER_PATH",
    os.path.join(_project_root, "checkpoints", "libero"))

cfg.output_dir = os.environ.get(
    "OUTPUT_DIR", os.path.join(_this_dir, "output_libero_optimized_stage2_nodmd"))

cfg.dataset_path = os.environ.get(
    "DATASET_PATH",
    os.path.join(_project_root, "training_data", "libero-long-lerobot"))

cfg.empty_emb_path = os.path.join(cfg.dataset_path, "empty_emb.pt")

# ============================================================
# 模型架构配置（与 Stage 2 一致）
# ============================================================
cfg.patch_size = (1, 2, 2)
cfg.param_dtype = torch.bfloat16
cfg.env_type = "none"
cfg.height = 128
cfg.width = 128
cfg.num_frames = 64
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
# FlowMatch 调度器配置（与 Stage 2 一致）
# ============================================================
cfg.snr_shift = 5.0
cfg.action_snr_shift = 0.05
cfg.num_train_timesteps = 1000

# ============================================================
# LCM 蒸馏核心参数（与 Stage 2 一致）
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
cfg.action_aware_weight = 0.1

# ============================================================
# Flow Map 蒸馏参数（与 Stage 2 一致）
# ============================================================
cfg.diffusion_ratio = 0.5
cfg.consistency_ratio = 0.25
cfg.flowmap_ratio = 0.25

cfg.epsilon = 1.0
cfg.gate_value = 0.1
cfg.deltatime_type = 'r'
cfg.weight_type = 'beta08'
cfg.gt_regression_weight = 0.5

# ============================================================
# 消融实验开关（与 Stage 2 一致）
# ============================================================
cfg.use_flowmap = True
cfg.use_gt_regression = True
cfg.use_central_diff = True
cfg.selective_cdiff = True
cfg.action_use_flowmap = False
cfg.use_action_distill = True

# ============================================================
# DMD 参数（禁用 DMD 用于速度测试）
# ============================================================
cfg.use_dmd = False                 # 禁用 DMD
cfg.dmd_weight = 0.1
cfg.dmd_warmup_steps = 0
cfg.dmd_rollout_steps_min = 2
cfg.dmd_rollout_steps_max = 4
cfg.dmd_cfg_scale = 5.0
cfg.dmd_discriminator_lr = 1e-5
cfg.dmd_discriminator_steps = 1
cfg.dmd_discriminator_warmup = 200
cfg.dmd_hidden_dim = 256
cfg.dmd_num_layers = 4
cfg.dmd_num_heads = 8
cfg.dmd_dropout = 0.1

# ============================================================
# LoRA 配置（与 Stage 2 一致）
# ============================================================
cfg.use_lora = True
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
# LCM 超参数（与 Stage 2 一致）
# ============================================================
cfg.ema_decay = 0.995
cfg.loss_type = "huber"
cfg.huber_c = 0.001
cfg.sigma_data = 0.5
cfg.cfg_min = 2.0
cfg.cfg_max = 10.0

# ============================================================
# 训练超参数（与 Stage 2 一致）
# ============================================================
cfg.learning_rate = 5e-6
cfg.beta1 = 0.9
cfg.beta2 = 0.999
cfg.weight_decay = 0.0
cfg.max_grad_norm = 1.0
cfg.warmup_steps = 100
cfg.max_train_steps = 20  # 速度测试：只运行 20 步
cfg.batch_size = 1
cfg.gradient_accumulation_steps = 16
cfg.load_worker = 8
cfg.pin_memory = True
cfg.prefetch_factor = 4
cfg.cache_dataset_in_memory = True
cfg.use_torch_compile = False

# Loss clipping
cfg.loss_clip_value = 10.0
cfg.loss_clip_enabled = True

cfg.noisy_cond_prob = 0.0
cfg.cfg_prob = 0.0

# Action 时间下采样
cfg.action_downsample_factor = 4

# ============================================================
# 检查点与日志
# ============================================================
cfg.save_interval = 500
cfg.gc_interval = 50
cfg.enable_wandb = True
cfg.wandb_entity = None
cfg.seed = 42
