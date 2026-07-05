"""Stage 1 LIBERO FlowMap with Cosmos endpoints and WanVA central-diff field.

Cosmos Policy still supplies the raw action target and official future images.
The future images are encoded through the WanVA VAE as endpoint anchors. A
frozen WanVA/LingBotVA video teacher additionally supplies the same local
central-difference vector-field target used by the regular FlowMap path.
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


cfg.output_dir = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_this_dir, "output_libero_cosmos_policy_stage1_wanva_cdiff"),
)
cfg.wandb_name_prefix = "stage1_cosmos_wanva_cdiff"

# Use the LingBotVA/WanVA FlowMap local vector field as the primary video
# target, while keeping Cosmos future-image endpoints as a small anchor.
cfg.cosmos_video_cdiff_mode = os.environ.get(
    "COSMOS_VIDEO_CDIFF_MODE", "primary").lower()
if cfg.cosmos_video_cdiff_mode not in ("primary", "aux"):
    raise ValueError(
        "COSMOS_VIDEO_CDIFF_MODE must be either 'primary' or 'aux'."
    )
cfg.cosmos_video_cdiff_aux = _env_bool("COSMOS_VIDEO_CDIFF_AUX", True)
cfg.cosmos_video_cdiff_teacher_model_path = os.environ.get(
    "COSMOS_VIDEO_CDIFF_TEACHER_MODEL_PATH",
    cfg.student_base_model_path,
)
cfg.cosmos_video_cdiff_loss_weight = float(
    os.environ.get("COSMOS_VIDEO_CDIFF_LOSS_WEIGHT", 1.0)
)
_default_endpoint_weight = 1e-3 if cfg.cosmos_video_cdiff_mode == "primary" else 1.0
cfg.cosmos_video_endpoint_loss_weight = float(
    os.environ.get("COSMOS_VIDEO_ENDPOINT_LOSS_WEIGHT", _default_endpoint_weight)
)

cfg.use_central_diff = _env_bool("USE_CENTRAL_DIFF", True)
cfg.selective_cdiff = _env_bool("SELECTIVE_CDIFF", True)
cfg.epsilon = float(os.environ.get("EPSILON", getattr(cfg, "epsilon", 5.0)))

# Match the LingBotVA Stage 1 action objective by default: target-student
# action distillation, action-side FlowMap where available, local action FM,
# and the same light GT regularizer. Set COSMOS_ACTION_LINGBOTVA_STAGE1=0 to
# recover the older Cosmos raw-action endpoint objective.
cfg.cosmos_action_lingbotva_stage1 = _env_bool(
    "COSMOS_ACTION_LINGBOTVA_STAGE1", True)
if cfg.cosmos_action_lingbotva_stage1:
    cfg.action_use_flowmap = _env_bool("ACTION_USE_FLOWMAP", True)
    cfg.use_action_distill = _env_bool("USE_ACTION_DISTILL", True)
    cfg.action_aware = _env_bool("ACTION_AWARE", True)
    cfg.use_gt_regression = _env_bool("USE_GT_REGRESSION", True)
    cfg.gt_regression_weight = float(os.environ.get("GT_REGRESSION_WEIGHT", 0.15))
    cfg.action_aware_weight = float(os.environ.get("ACTION_AWARE_WEIGHT", 0.1))
else:
    cfg.action_use_flowmap = False
    cfg.use_action_distill = False
    cfg.action_aware = False
    cfg.use_gt_regression = False
    cfg.gt_regression_weight = 0.0
    cfg.action_aware_weight = 0.0

cfg.use_opd_aux = False
cfg.use_dmd = False
