"""Stage 2 action-only continuation using a Cosmos Policy checkpoint."""
import copy
import os

from distillation_flowmap.config_libero_fullfinetune_stage2_anyflow import cfg as _base_cfg

cfg = copy.deepcopy(_base_cfg)
_this_dir = os.path.dirname(os.path.abspath(__file__))


def _env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


_stage1_ckpt = os.path.join(
    _this_dir,
    "output_libero_cosmos_policy_stage1",
    "checkpoints",
    "step_1",
)

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

cfg.resume_from_path = os.environ.get("RESUME_FROM_PATH", _stage1_ckpt)
cfg.resume_online_from_target = _env_bool("RESUME_ONLINE_FROM_TARGET", True)
cfg.reset_resume_step = _env_bool("RESET_RESUME_STEP", True)
cfg.output_dir = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_this_dir, "output_libero_cosmos_policy_stage2"),
)
cfg.wandb_name_prefix = "stage2_cosmos_policy_action"

cfg.distill_mode = "action"
cfg.distill_video = False
cfg.distill_action = True
cfg.action_aware = True
cfg.use_action_distill = True
cfg.use_gt_regression = True
cfg.use_central_diff = False
cfg.action_use_flowmap = False

cfg.diffusion_ratio = float(os.environ.get("DIFFUSION_RATIO", 1.0))
cfg.consistency_ratio = float(os.environ.get("CONSISTENCY_RATIO", 0.0))
cfg.flowmap_ratio = float(os.environ.get("FLOWMAP_RATIO", 0.0))

cfg.action_loss_weight = float(os.environ.get("ACTION_LOSS_WEIGHT", 1.0))
cfg.action_block_weight = float(os.environ.get("ACTION_BLOCK_WEIGHT", 1.0))
cfg.action_aware_weight = float(os.environ.get("ACTION_AWARE_WEIGHT", 0.01))
cfg.gt_regression_weight = float(os.environ.get("GT_REGRESSION_WEIGHT", 0.1))
cfg.learning_rate = float(os.environ.get("LEARNING_RATE", 2e-7))
cfg.max_train_steps = int(os.environ.get("MAX_TRAIN_STEPS", 100))
cfg.save_interval = int(os.environ.get("SAVE_INTERVAL", 50))
cfg.gradient_checkpointing = _env_bool("GRADIENT_CHECKPOINTING", False)
cfg.skip_teacher_compile = True

cfg.enable_light_eval = False
cfg.enable_rollout_eval = False
cfg.enable_stage1_start_eval = False
cfg.enable_stage1_start_eval_baseline = False
cfg.use_opd_aux = False
cfg.use_onpolicy_transition = False
cfg.use_dmd = False
