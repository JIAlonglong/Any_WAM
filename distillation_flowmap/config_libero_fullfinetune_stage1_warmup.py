"""Stage 1 warmup for LIBERO FlowMap full-parameter fine-tuning.

This keeps the AnyFlow-style mixed (t, r) objective from
config_libero_optimized, but disables LoRA and runs only a short warmup before
Stage 2 continues from the produced full-model checkpoint.
"""
import copy
import os

from distillation_flowmap.config_libero_optimized import cfg as _base_cfg

cfg = copy.deepcopy(_base_cfg)
_this_dir = os.path.dirname(os.path.abspath(__file__))

cfg.output_dir = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_this_dir, "output_libero_fullft_stage1_warmup"),
)
cfg.wandb_name_prefix = "stage1_fullft_warmup"

# Full-parameter fine-tuning. Do not build or save LoRA adapters.
cfg.use_lora = False
cfg.lora_rank = 0
cfg.lora_alpha = 0
cfg.lora_dropout = 0.0

# Full-model fine-tuning needs a much smaller LR than the old LoRA run.
cfg.learning_rate = float(os.environ.get("LEARNING_RATE", 1e-6))
cfg.beta1 = float(os.environ.get("BETA1", 0.9))
cfg.beta2 = float(os.environ.get("BETA2", 0.95))
cfg.ema_decay = float(os.environ.get("EMA_DECAY", 0.999))
cfg.ema_warmup_steps = int(os.environ.get("EMA_WARMUP_STEPS", 100))
cfg.drop_text_ratio = float(os.environ.get("DROP_TEXT_RATIO", 0.1))
cfg.fuse_guidance_scale = float(os.environ.get("FUSE_GUIDANCE_SCALE", 3.0))
cfg.cfg_min = float(os.environ.get("CFG_MIN", cfg.fuse_guidance_scale))
cfg.cfg_max = float(os.environ.get("CFG_MAX", cfg.fuse_guidance_scale))
cfg.max_grad_norm = float(os.environ.get("MAX_GRAD_NORM", 0.3))
cfg.warmup_steps = int(os.environ.get("WARMUP_STEPS", 100))
cfg.max_train_steps = int(os.environ.get("MAX_TRAIN_STEPS", 5000))
cfg.save_interval = int(os.environ.get("SAVE_INTERVAL", 1000))
cfg.skip_teacher_compile = os.environ.get(
    "SKIP_TEACHER_COMPILE", "0").lower() in ("1", "true", "yes", "on")

# Lightweight deterministic video/action eval. Enabled by default in Stage 1
# so the warmup checkpoint has a quick fixed-batch quality signal.
cfg.enable_light_eval = os.environ.get(
    "ENABLE_LIGHT_EVAL", "1").lower() in ("1", "true", "yes", "on")
cfg.light_eval_interval = int(os.environ.get("LIGHT_EVAL_INTERVAL", cfg.save_interval))
cfg.light_eval_num_batches = int(os.environ.get("LIGHT_EVAL_NUM_BATCHES", 1))
cfg.light_eval_seed = int(os.environ.get("LIGHT_EVAL_SEED", 42))
cfg.light_eval_start_index = int(os.environ.get("LIGHT_EVAL_START_INDEX", 0)) 
cfg.light_eval_pairs = [(1000, 1000), (1000, 0), (750, 250)]

# Stay close to AnyFlow: use mixed t/r sampling without explicit endpoint oversampling.

# Preserve the current action-side AnyFlow path.
cfg.action_use_flowmap = True
cfg.num_ddim_timesteps_action = int(os.environ.get("NUM_DDIM_TIMESTEPS_ACTION", 1))
