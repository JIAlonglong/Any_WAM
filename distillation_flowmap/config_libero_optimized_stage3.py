"""
Stage 3: KTO-PAOPD（Pointwise Adaptive On-Policy Distillation）

基于 Stage 2 OPD，添加 KTO-style 自适应权重：
  - 已学好（学生≈教师）的 token → 低权重 0.3（维持即可）
  - 还没学好的 token → 高权重 1.0（重点纠正）
  - 动态阈值：每个 batch 的中位数
  - "损失比收益更痛"：bad weight > good weight

基于论文：KTO (Ethayarajh et al., ICML 2024), APO (NeurIPS 2025)
"""
import os
import copy
from distillation_flowmap.config_libero_optimized import cfg as _base_cfg
cfg = copy.deepcopy(_base_cfg)
_this_dir = os.path.dirname(os.path.abspath(__file__))

# Stage 2: On-Policy Transition
cfg.use_onpolicy_transition = True
cfg.onpolicy_warmup_steps = 500
cfg.rollout_step_pairs = [[1, 1]]
cfg.teacher_micro_steps = 2
cfg.video_transition_weight = 0.2
cfg.local_fm_weight = 0.05
cfg.transition_loss_type = 'huber'
cfg.transition_huber_c = 1e-3

# ============================================================
# KTO-style pointwise adaptive weighting（Stage 3 独有）
# ============================================================
cfg.kto_adaptive = True           # 启用 KTO 自适应权重
cfg.kto_good_weight = 0.3         # 已学好 token 的低权重（维持）
cfg.kto_bad_weight = 1.0          # 未学好 token 的高权重（重点纠正）
# cfg.kto_threshold = None → 动态用当前 batch 的中位数

# Stage 2 训练参数
cfg.max_train_steps = 5000
cfg.learning_rate = 5e-6
cfg.save_interval = 250
cfg.warmup_steps = 100
cfg.max_grad_norm = 0.3
cfg.loss_clip_value = 5.0
cfg.loss_clip_enabled = True

# 路径
_stage1_ckpt = os.path.join(_this_dir, 'output_libero_stage1_retrain_20260624', 'checkpoints', 'step_1500')
cfg.resume_from_path = os.environ.get('RESUME_FROM_PATH', _stage1_ckpt)
cfg.output_dir = os.environ.get('OUTPUT_DIR', os.path.join(_this_dir, 'output_libero_stage3_kto'))
cfg.wandb_name_prefix = "stage3_kto"
cfg.gt_regression_weight = 0.5
