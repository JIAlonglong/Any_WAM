"""Stage 1 action-only FlowMap chain using a Cosmos Policy checkpoint.

This is a parallel smoke/baseline path. The normal LIBERO WanVA stage configs
are unchanged. Cosmos Policy is validated as the teacher backend, while the
FlowMap student is still initialized from a WanVA transformer base.
"""
import copy
import os

from distillation_flowmap.config_libero_fullfinetune_stage1_warmup import cfg as _base_cfg

cfg = copy.deepcopy(_base_cfg)
_this_dir = os.path.dirname(os.path.abspath(__file__))


def _env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


cfg.teacher_backend = "cosmos_policy"
cfg.teacher_model_path = os.environ.get(
    "COSMOS_POLICY_PATH",
    "/root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
)
cfg.student_base_model_path = os.environ.get(
    "STUDENT_BASE_MODEL_PATH",
    "/root/nas/junjie/jj/Any_WAM/checkpoints/lingbot-va-posttrain-libero",
)
cfg.cosmos_policy_validate_weights = _env_bool("COSMOS_POLICY_VALIDATE_WEIGHTS", False)

cfg.output_dir = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_this_dir, "output_libero_cosmos_policy_stage1"),
)
cfg.wandb_name_prefix = "stage1_cosmos_policy_action"

# Cosmos Policy is not a WanVA video teacher. Keep this path action-only.
cfg.distill_mode = "action"
cfg.distill_video = False
cfg.distill_action = True
cfg.action_aware = True
cfg.use_action_distill = True
cfg.use_gt_regression = True
cfg.use_central_diff = False
cfg.action_use_flowmap = False

# Use r=t for the first compatibility chain. With the latent fallback action
# velocity, this makes the reconstructed x0 target exactly the dataset action.
cfg.diffusion_ratio = float(os.environ.get("DIFFUSION_RATIO", 1.0))
cfg.consistency_ratio = float(os.environ.get("CONSISTENCY_RATIO", 0.0))
cfg.flowmap_ratio = float(os.environ.get("FLOWMAP_RATIO", 0.0))

cfg.action_loss_weight = float(os.environ.get("ACTION_LOSS_WEIGHT", 1.0))
cfg.action_block_weight = float(os.environ.get("ACTION_BLOCK_WEIGHT", 1.0))
cfg.action_aware_weight = float(os.environ.get("ACTION_AWARE_WEIGHT", 0.01))
cfg.gt_regression_weight = float(os.environ.get("GT_REGRESSION_WEIGHT", 0.1))
cfg.learning_rate = float(os.environ.get("LEARNING_RATE", 5e-7))
cfg.max_train_steps = int(os.environ.get("MAX_TRAIN_STEPS", 100))
cfg.save_interval = int(os.environ.get("SAVE_INTERVAL", 50))
cfg.gradient_checkpointing = _env_bool("GRADIENT_CHECKPOINTING", False)
cfg.skip_teacher_compile = True

# Eval paths decode through WanVA VAE, which Cosmos Policy checkpoints do not
# provide. Keep them off for this backend.
cfg.enable_light_eval = False
cfg.enable_rollout_eval = False
cfg.enable_stage1_start_eval = False
cfg.enable_stage1_start_eval_baseline = False
cfg.use_opd_aux = False
cfg.use_dmd = False
