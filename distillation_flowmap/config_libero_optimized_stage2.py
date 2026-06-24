"""
Stage 2: On-Policy Distillation（基准）

与 Stage 3 控制变量对比：唯一区别是没有 KTO 自适应权重。
其他所有参数完全一致。
"""
import os
import copy
from distillation_flowmap.config_libero_optimized import cfg as _base_cfg
cfg = copy.deepcopy(_base_cfg)
_this_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# On-Policy Transition（与 Stage 3 完全一致）
# ============================================================
cfg.use_onpolicy_transition = True
cfg.onpolicy_warmup_steps = 500
cfg.rollout_step_pairs = [[1, 1]]
cfg.teacher_micro_steps = 2
cfg.video_transition_weight = 0.2
cfg.local_fm_weight = 0.05
cfg.transition_loss_type = 'huber'
cfg.transition_huber_c = 1e-3

# ============================================================
# KTO 自适应权重（Stage 2 关闭，作为 baseline）
# ============================================================
cfg.kto_adaptive = False

# ============================================================
# 训练参数（与 Stage 3 完全一致）
# ============================================================
cfg.max_train_steps = 5000
cfg.learning_rate = 5e-6
cfg.save_interval = 250
cfg.warmup_steps = 100
cfg.max_grad_norm = 0.3
cfg.loss_clip_value = 5.0
cfg.loss_clip_enabled = True

# ============================================================
# 路径
# ============================================================
_stage1_ckpt = os.path.join(_this_dir, 'output_libero_stage1_retrain_20260624', 'checkpoints', 'step_1500')
cfg.resume_from_path = os.environ.get('RESUME_FROM_PATH', _stage1_ckpt)
cfg.output_dir = os.environ.get('OUTPUT_DIR', os.path.join(_this_dir, 'output_libero_stage2_opd'))
cfg.wandb_name_prefix = "stage2_opd"
cfg.gt_regression_weight = 0.5
