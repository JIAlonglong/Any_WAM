"""Stage 2 KTO-PAOPD normalized focal reweighting variant for LIBERO full fine-tuning."""
import copy
import os

from distillation_flowmap.config_libero_fullfinetune_stage2_anyflow import cfg as _base_cfg

cfg = copy.deepcopy(_base_cfg)
_this_dir = os.path.dirname(os.path.abspath(__file__))

cfg.opd_aux_variant = os.environ.get("OPD_AUX_VARIANT", "kto_paopd_norm_focal")
cfg.kto_adaptive = True
cfg.kto_reweight_mode = os.environ.get("KTO_REWEIGHT_MODE", "normalized_focal")
cfg.kto_good_weight = float(os.environ.get("KTO_GOOD_WEIGHT", 0.3))
cfg.kto_bad_weight = float(os.environ.get("KTO_BAD_WEIGHT", 1.0))
_kto_threshold = os.environ.get("KTO_THRESHOLD")
cfg.kto_threshold = float(_kto_threshold) if _kto_threshold not in (None, "") else None
cfg.kto_eps = float(os.environ.get("KTO_EPS", 1e-5))
cfg.kto_alpha = float(os.environ.get("KTO_ALPHA", 0.5))
cfg.kto_temperature = float(os.environ.get("KTO_TEMPERATURE", 0.10))
cfg.kto_min_weight = float(os.environ.get("KTO_MIN_WEIGHT", 0.5))
cfg.kto_max_weight = float(os.environ.get("KTO_MAX_WEIGHT", 1.8))
cfg.kto_threshold_quantile = float(os.environ.get("KTO_THRESHOLD_QUANTILE", 0.70))
cfg.kto_threshold_ema_decay = float(os.environ.get("KTO_THRESHOLD_EMA_DECAY", 0.90))
cfg.kto_use_ema_threshold = os.environ.get(
    "KTO_USE_EMA_THRESHOLD", "1").lower() in ("1", "true", "yes", "on")
cfg.kto_warmup_steps = int(os.environ.get("KTO_WARMUP_STEPS", 0))
cfg.kto_ramp_steps = int(os.environ.get("KTO_RAMP_STEPS", 0))
cfg.kto_alpha_decay_hold_steps = int(os.environ.get("KTO_ALPHA_DECAY_HOLD_STEPS", 20))
cfg.kto_alpha_decay_ramp_steps = int(os.environ.get("KTO_ALPHA_DECAY_RAMP_STEPS", 20))
cfg.kto_video_scale_start = float(os.environ.get("KTO_VIDEO_SCALE_START", 0.85))
cfg.kto_video_scale_end = float(os.environ.get("KTO_VIDEO_SCALE_END", 1.0))
cfg.kto_video_scale_hold_steps = int(os.environ.get("KTO_VIDEO_SCALE_HOLD_STEPS", 20))
cfg.kto_video_scale_ramp_steps = int(os.environ.get("KTO_VIDEO_SCALE_RAMP_STEPS", 20))
cfg.kto_main_video_reweight = os.environ.get(
    "KTO_MAIN_VIDEO_REWEIGHT", "0").lower() in ("1", "true", "yes", "on")
cfg.kto_main_alpha = float(os.environ.get("KTO_MAIN_ALPHA", 1.0))
cfg.kto_main_temperature = float(os.environ.get("KTO_MAIN_TEMPERATURE", cfg.kto_temperature))
cfg.kto_main_min_weight = float(os.environ.get("KTO_MAIN_MIN_WEIGHT", cfg.kto_min_weight))
cfg.kto_main_max_weight = float(os.environ.get("KTO_MAIN_MAX_WEIGHT", cfg.kto_max_weight))
cfg.kto_main_threshold_quantile = float(os.environ.get("KTO_MAIN_THRESHOLD_QUANTILE", 0.70))
cfg.kto_main_threshold_ema_decay = float(os.environ.get("KTO_MAIN_THRESHOLD_EMA_DECAY", 0.90))
cfg.kto_main_use_ema_threshold = os.environ.get(
    "KTO_MAIN_USE_EMA_THRESHOLD", "1").lower() in ("1", "true", "yes", "on")

cfg.stage2_sampler = os.environ.get("STAGE2_SAMPLER", "default").lower()
cfg.stage2_group_by = os.environ.get("STAGE2_GROUP_BY", "task").lower()
_stage2_samples_per_group = os.environ.get("STAGE2_SAMPLES_PER_GROUP")
cfg.stage2_samples_per_group = (
    int(_stage2_samples_per_group)
    if _stage2_samples_per_group not in (None, "")
    else None
)

cfg.output_dir = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_this_dir, "output_libero_fullft_stage2_kto_paopd_norm_focal"),
)
cfg.wandb_name_prefix = os.environ.get(
    "WANDB_NAME_PREFIX",
    "stage2_fullft_kto_paopd_norm_focal",
)
