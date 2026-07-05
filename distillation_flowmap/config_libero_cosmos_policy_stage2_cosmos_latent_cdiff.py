"""Stage 2 LIBERO FlowMap with pure Cosmos latent central-diff targets.

This is the stage-2 continuation of
``config_libero_cosmos_policy_stage1_cosmos_latent_cdiff``. It keeps the
teacher signal Cosmos-only: official Cosmos Policy actions plus Cosmos latent
central-diff vector-field targets.
"""
import copy
import os

from distillation_flowmap.config_libero_cosmos_policy_stage2_all_cosmos_flowmap import (
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
    "output_libero_cosmos_policy_stage1_cosmos_latent_cdiff",
    "checkpoints",
    os.environ.get("STAGE1_CKPT_NAME", "step_5000"),
)

cfg.resume_from_path = os.environ.get("RESUME_FROM_PATH", _stage1_ckpt)
cfg.resume_online_from_target = _env_bool("RESUME_ONLINE_FROM_TARGET", True)
cfg.reset_resume_step = _env_bool("RESET_RESUME_STEP", True)
cfg.resume_optimizer_state = _env_bool("RESUME_OPTIMIZER_STATE", False)

cfg.output_dir = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_this_dir, "output_libero_cosmos_policy_stage2_cosmos_latent_cdiff"),
)
cfg.wandb_name_prefix = "stage2_cosmos_latent_cdiff"

cfg.cosmos_video_target = False
cfg.cosmos_latent_target = True
cfg.cosmos_policy_use_raw_inference = _env_bool("COSMOS_POLICY_USE_RAW_INFERENCE", True)
cfg.return_raw_observation = True

cfg.distill_mode = "flashwam"
cfg.distill_video = True
cfg.distill_action = True
cfg.use_central_diff = True
cfg.cosmos_video_cdiff_aux = False
cfg.cosmos_video_cdiff_mode = "primary"
cfg.cosmos_latent_cdiff_loss_weight = float(os.environ.get("COSMOS_LATENT_CDIFF_LOSS_WEIGHT", 1.0))
cfg.cosmos_latent_endpoint_loss_weight = float(os.environ.get("COSMOS_LATENT_ENDPOINT_LOSS_WEIGHT", 0.0))
cfg.cosmos_latent_epsilon = float(os.environ.get("COSMOS_LATENT_EPSILON", 0.001))
cfg.cosmos_latent_t_min = float(os.environ.get("COSMOS_LATENT_T_MIN", 4.0 / 5.0))
cfg.cosmos_latent_t_max = float(os.environ.get("COSMOS_LATENT_T_MAX", 80.0 / 81.0))
cfg.cosmos_latent_channels = int(os.environ.get("COSMOS_LATENT_CHANNELS", 16))
cfg.cosmos_latent_frames = int(os.environ.get("COSMOS_LATENT_FRAMES", 9))
cfg.cosmos_latent_height = int(os.environ.get("COSMOS_LATENT_HEIGHT", 28))
cfg.cosmos_latent_width = int(os.environ.get("COSMOS_LATENT_WIDTH", 28))
cfg.cosmos_latent_center_velocity_mode = os.environ.get(
    "COSMOS_LATENT_CENTER_VELOCITY_MODE", "symmetric_average"
).lower()
cfg.cosmos_policy_worker_cuda_visible_devices = (
    os.environ.get("COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES") or None
)

cfg.action_aware = False
cfg.use_action_distill = False
cfg.use_gt_regression = False
cfg.action_use_flowmap = False
cfg.action_aware_weight = 0.0
cfg.gt_regression_weight = 0.0

cfg.use_opd_aux = False
cfg.use_onpolicy_transition = False
cfg.use_dmd = False
cfg.gradient_checkpointing = _env_bool("GRADIENT_CHECKPOINTING", True)
