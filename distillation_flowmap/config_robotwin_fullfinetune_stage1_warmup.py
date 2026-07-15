"""Stage 1 warmup for RobotWin FlowMap full-parameter fine-tuning."""
import copy
import os

from distillation_flowmap.config import cfg as _base_cfg, _env_bool

cfg = copy.deepcopy(_base_cfg)
_this_dir = os.path.dirname(os.path.abspath(__file__))

cfg.output_dir = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_this_dir, "output_robotwin_fullft_stage1_warmup"),
)
cfg.wandb_name_prefix = "robotwin_stage1_fullft_warmup"

# Full-parameter fine-tuning. RobotWin base already defaults to this, but keep
# it explicit so the stage configs mirror the LIBERO full-finetune path.
cfg.use_lora = False
cfg.lora_rank = 0
cfg.lora_alpha = 0
cfg.lora_dropout = 0.0

# Stage 1 keeps the standard AnyFlow/FlowMap objective and does not add OPD.
cfg.use_onpolicy_transition = False
cfg.use_opd_aux = False
cfg.use_dmd = False


def _env_list(name):
    value = os.environ.get(name)
    if not value:
        return None
    return [v.strip() for v in value.replace(";", ",").split(",") if v.strip()]


def _env_positive_int(name):
    value = os.environ.get(name)
    if value in (None, "", "0"):
        return None
    value = int(value)
    return value if value > 0 else None


cfg.dataset_task_filter = _env_list("DATASET_TASK_FILTER")
cfg.dataset_sample_manifest = os.environ.get("DATASET_SAMPLE_MANIFEST")
cfg.dataset_max_episodes_per_task = _env_positive_int("DATASET_MAX_EPISODES_PER_TASK")
cfg.dataset_max_samples_per_task = _env_positive_int("DATASET_MAX_SAMPLES_PER_TASK")


def _parse_adjacent_grid(text):
    return [int(v.strip()) for v in text.split(",") if v.strip()]


cfg.flowmap_pair_mode = os.environ.get("FLOWMAP_PAIR_MODE", "arbitrary").lower()
cfg.opd_pair_mode = os.environ.get("OPD_PAIR_MODE", cfg.flowmap_pair_mode).lower()
cfg.flowmap_adjacent_grid = _parse_adjacent_grid(
    os.environ.get("FLOWMAP_ADJACENT_GRID", "1000,750,500,250,0")
)

# Keep the single-node RobotWin effective batch when launched with torchrun.
cfg.auto_scale_gradient_accumulation = _env_bool("AUTO_SCALE_GRADIENT_ACCUMULATION", True)
cfg.gradient_accumulation_reference = int(os.environ.get(
    "GRADIENT_ACCUMULATION_REFERENCE",
    cfg.gradient_accumulation_steps,
))

# Conservative full-model warmup defaults; all can be overridden by env.
cfg.learning_rate = float(os.environ.get("LEARNING_RATE", 1e-6))
cfg.beta1 = float(os.environ.get("BETA1", cfg.beta1))
cfg.beta2 = float(os.environ.get("BETA2", cfg.beta2))
cfg.ema_decay = float(os.environ.get("EMA_DECAY", cfg.ema_decay))
cfg.ema_warmup_steps = int(os.environ.get("EMA_WARMUP_STEPS", 100))
cfg.drop_text_ratio = float(os.environ.get("DROP_TEXT_RATIO", 0.1))
cfg.fuse_guidance_scale = float(os.environ.get("FUSE_GUIDANCE_SCALE", 5.0))
cfg.cfg_min = float(os.environ.get("CFG_MIN", cfg.fuse_guidance_scale))
cfg.cfg_max = float(os.environ.get("CFG_MAX", cfg.fuse_guidance_scale))
cfg.max_grad_norm = float(os.environ.get("MAX_GRAD_NORM", 0.5))
cfg.warmup_steps = int(os.environ.get("WARMUP_STEPS", 100))
cfg.max_train_steps = int(os.environ.get("MAX_TRAIN_STEPS", 5000))
cfg.save_interval = int(os.environ.get("SAVE_INTERVAL", 1000))
cfg.resume_optimizer_state = _env_bool("RESUME_OPTIMIZER_STATE", False)
cfg.reset_resume_step = _env_bool("RESET_RESUME_STEP", True)
cfg.skip_teacher_compile = _env_bool("SKIP_TEACHER_COMPILE", True)

# RobotWin sequences are heavier than LIBERO; keep checkpointing enabled by
# default unless explicitly disabled.
cfg.gradient_checkpointing = _env_bool("GRADIENT_CHECKPOINTING", True)

# Disabled by default to avoid extra evaluation memory during warmup.
cfg.enable_light_eval = _env_bool("ENABLE_LIGHT_EVAL", False)
cfg.enable_rollout_eval = _env_bool("ENABLE_ROLLOUT_EVAL", False)
cfg.enable_grad_branch_diagnostics = _env_bool("ENABLE_GRAD_BRANCH_DIAGNOSTICS", False)
