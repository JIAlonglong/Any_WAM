"""Stage 2 continuation with Cosmos-only video and action targets.

This keeps the teacher signal purely from Cosmos Policy:
  - raw actions from official Cosmos Policy inference
  - future_image_predictions encoded as latent x0 video anchors

It intentionally disables WanVA/LingBotVA teacher velocity, central-diff,
action FlowMap, OPD, and DMD. The WanVA base/VAE is still used as the student
architecture and latent tokenizer.
"""
import copy
import os

from distillation_flowmap.config_libero_cosmos_policy_stage1_all_cosmos_flowmap import (
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
    "output_libero_cosmos_policy_stage1_all_cosmos_flowmap",
    "checkpoints",
    "step_5000",
)

cfg.resume_from_path = os.environ.get("RESUME_FROM_PATH", _stage1_ckpt)
cfg.resume_online_from_target = _env_bool("RESUME_ONLINE_FROM_TARGET", True)
cfg.reset_resume_step = _env_bool("RESET_RESUME_STEP", True)
cfg.resume_optimizer_state = _env_bool("RESUME_OPTIMIZER_STATE", False)

cfg.output_dir = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_this_dir, "output_libero_cosmos_policy_stage2_all_cosmos_flowmap"),
)
cfg.wandb_name_prefix = "stage2_cosmos_all_cosmos_flowmap"
cfg.enable_wandb = _env_bool("ENABLE_WANDB", False)

cfg.distill_mode = "flashwam"
cfg.distill_video = True
cfg.distill_action = True
cfg.action_aware = False
cfg.use_action_distill = False
cfg.use_gt_regression = False
cfg.use_central_diff = False
cfg.selective_cdiff = False
cfg.action_use_flowmap = False
cfg.use_opd_aux = False
cfg.use_onpolicy_transition = False
cfg.use_dmd = False

cfg.learning_rate = float(os.environ.get("LEARNING_RATE", 2e-7))
cfg.max_train_steps = int(os.environ.get("MAX_TRAIN_STEPS", 5000))
cfg.save_interval = int(os.environ.get("SAVE_INTERVAL", 1000))
cfg.skip_teacher_compile = _env_bool("SKIP_TEACHER_COMPILE", True)
cfg.gradient_checkpointing = _env_bool("GRADIENT_CHECKPOINTING", False)

cfg.enable_light_eval = _env_bool("ENABLE_LIGHT_EVAL", False)
cfg.enable_rollout_eval = _env_bool("ENABLE_ROLLOUT_EVAL", False)
cfg.enable_stage1_start_eval = _env_bool("ENABLE_STAGE1_START_EVAL", False)
cfg.enable_stage1_start_eval_baseline = _env_bool(
    "ENABLE_STAGE1_START_EVAL_BASELINE", False)
