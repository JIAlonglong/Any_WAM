"""Stage 2 continuation for Cosmos/WanVA-cdiff Stage 1 checkpoints.

This config resumes from the Cosmos Stage 1 WanVA-cdiff checkpoint, but keeps
the Stage 2 training path aligned with LingBotVA: WanVA teacher, mixed
AnyFlow/FlowMap objective, action FlowMap, target-student action distillation,
and OPD auxiliary correction. Use RESUME_FROM_PATH to point at a concrete
timestamped Stage 1 run.
"""
import copy
import os

from distillation_flowmap.config_libero_fullfinetune_stage2_anyflow import (
    cfg as _base_cfg,
)

cfg = copy.deepcopy(_base_cfg)
_this_dir = os.path.dirname(os.path.abspath(__file__))


def _env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


_stage1_ckpt = os.path.join(
    _this_dir,
    "output_libero_cosmos_policy_stage1_wanva_cdiff",
    "checkpoints",
    "step_5000",
)

# Use the regular WanVA/LingBotVA teacher backend so FlowMapStepMixin._train_step
# follows the same main Stage 2 AnyFlow/OPD path as LingBotVA. Cosmos-specific
# endpoint-target steps are intentionally not used here.
cfg.teacher_backend = "wanva"
cfg.teacher_model_path = os.environ.get(
    "TEACHER_PATH",
    "/root/nas/junjie/jj/Any_WAM/checkpoints/lingbot-va-posttrain-libero",
)

cfg.resume_from_path = os.environ.get("RESUME_FROM_PATH", _stage1_ckpt)
cfg.resume_online_from_target = _env_bool("RESUME_ONLINE_FROM_TARGET", True)
cfg.reset_resume_step = _env_bool("RESET_RESUME_STEP", True)
cfg.output_dir = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_this_dir, "output_libero_cosmos_policy_stage2_wanva_cdiff"),
)
cfg.wandb_name_prefix = "stage2_cosmos_wanva_cdiff_lingbotva"
cfg.enable_wandb = _env_bool("ENABLE_WANDB", False)

cfg.distill_mode = "flashwam"
cfg.distill_video = True
cfg.distill_action = True
cfg.action_aware = True
cfg.use_action_distill = True
cfg.use_gt_regression = True
cfg.use_flowmap = True
cfg.use_central_diff = True
cfg.selective_cdiff = True
cfg.action_use_flowmap = True
cfg.action_epsilon = getattr(cfg, "epsilon", 1.0)

# Keep LingBotVA Stage 2 OPD defaults from config_libero_fullfinetune_stage2_anyflow,
# but make the main toggles explicit for import tests and launch scripts.
cfg.use_opd_aux = _env_bool("USE_OPD_AUX", True)
cfg.opd_aux_action = _env_bool("OPD_AUX_ACTION", False)
cfg.use_onpolicy_transition = _env_bool("USE_ONPOLICY_TRANSITION", False)
cfg.use_dmd = _env_bool("USE_DMD", False)

cfg.max_train_steps = int(os.environ.get("MAX_TRAIN_STEPS", 5000))
cfg.save_interval = int(os.environ.get("SAVE_INTERVAL", 1000))
cfg.learning_rate = float(os.environ.get("LEARNING_RATE", 5e-7))
cfg.skip_teacher_compile = _env_bool("SKIP_TEACHER_COMPILE", True)
cfg.gradient_checkpointing = _env_bool("GRADIENT_CHECKPOINTING", False)

# This config is for training; Cosmos official-eval/video scripts can still be
# run separately against saved checkpoints.
cfg.enable_light_eval = _env_bool("ENABLE_LIGHT_EVAL", False)
cfg.enable_rollout_eval = _env_bool("ENABLE_ROLLOUT_EVAL", False)
cfg.enable_stage1_start_eval = _env_bool("ENABLE_STAGE1_START_EVAL", False)
cfg.enable_stage1_start_eval_baseline = _env_bool(
    "ENABLE_STAGE1_START_EVAL_BASELINE", False)
