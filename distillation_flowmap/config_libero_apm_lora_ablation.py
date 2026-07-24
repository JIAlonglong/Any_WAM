"""LoRA small-sample variant of the LIBERO video-only APM experiment."""

import copy
import math
import os

from distillation_flowmap.config_libero_fullfinetune_stage2_video_only_opd import (
    cfg as _base_cfg,
)


cfg = copy.deepcopy(_base_cfg)

cfg.wandb_name_prefix = "libero_apm_lora_small_sample"
cfg.use_lora = True
cfg.lora_rank = int(os.environ.get("LORA_RANK", 128))
cfg.lora_alpha = int(os.environ.get("LORA_ALPHA", 64))
cfg.lora_dropout = float(os.environ.get("LORA_DROPOUT", 0.0))
cfg.learning_rate = float(os.environ.get("LEARNING_RATE", 5e-6))
cfg.dataset_sample_manifest = os.environ.get("DATASET_SAMPLE_MANIFEST")
cfg.dataset_task_filter = None
cfg.dataset_max_episodes_per_task = None
cfg.dataset_max_samples_per_task = None

if not cfg.dataset_sample_manifest:
    raise ValueError(
        "DATASET_SAMPLE_MANIFEST is required for the LIBERO APM ablation"
    )
if (
    cfg.lora_rank <= 0
    or cfg.lora_alpha <= 0
    or not math.isfinite(cfg.lora_dropout)
    or cfg.lora_dropout < 0
    or cfg.lora_dropout >= 1
    or not math.isfinite(cfg.learning_rate)
    or cfg.learning_rate <= 0
):
    raise ValueError("LoRA settings must be finite and positive")

# Reassert the experimental invariant after importing all environment-aware
# parent configs. The ablation isolates video endpoint/field supervision only.
cfg.opd_aux_action = False
cfg.opd_joint_action_rollout = False
cfg.opd_danceopd_action_velocity_weight = 0.0
