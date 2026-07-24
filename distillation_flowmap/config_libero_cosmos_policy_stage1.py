"""Stage 1 action-only FlowMap chain using a Cosmos Policy checkpoint.

This is a parallel smoke/baseline path. The normal LIBERO WanVA stage configs
are unchanged. Cosmos Policy is validated as the teacher backend, while the
FlowMap student is still initialized from a WanVA transformer base.
"""
import copy
import os

from distillation_flowmap.config_libero_fullfinetune_stage1_warmup import cfg as _base_cfg
from distillation_flowmap.cosmos_training_contract import (
    ACTION_PACKING_SCHEMA,
    CONTRACT_VERSION,
)

cfg = copy.deepcopy(_base_cfg)
_this_dir = os.path.dirname(os.path.abspath(__file__))

cfg.contract_version = CONTRACT_VERSION
cfg.action_packing_schema = ACTION_PACKING_SCHEMA
cfg.action_downsample_factor = 4
cfg.action_chunk_shape = [4, 4]
cfg.training_contract_stage = "raw_stage1"


def _env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    normalized = val.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        f"{name} must be a boolean (1/0, true/false, yes/no, or on/off), got {val!r}"
    )


def _env_int(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {val!r}") from None


cfg.resume_from_path = os.environ.get("RESUME_FROM_PATH") or None
cfg.resume_online_from_target = _env_bool("RESUME_ONLINE_FROM_TARGET", False)
cfg.reset_resume_step = _env_bool("RESET_RESUME_STEP", False)
cfg.resume_optimizer_state = _env_bool("RESUME_OPTIMIZER_STATE", False)
cfg.seed = _env_int("TRAIN_SEED", cfg.seed)


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
cfg.cosmos_policy_use_raw_inference = _env_bool("COSMOS_POLICY_USE_RAW_INFERENCE", False)
cfg.return_raw_observation = cfg.cosmos_policy_use_raw_inference
if cfg.cosmos_policy_use_raw_inference:
    cfg.cache_dataset_in_memory = _env_bool("CACHE_DATASET_IN_MEMORY", False)
cfg.raw_primary_image_key = os.environ.get(
    "COSMOS_POLICY_PRIMARY_IMAGE_KEY", "observation.images.agentview_rgb")
cfg.raw_wrist_image_key = os.environ.get(
    "COSMOS_POLICY_WRIST_IMAGE_KEY", "observation.images.eye_in_hand_rgb")
cfg.cosmos_policy_inference_mode = os.environ.get("COSMOS_POLICY_INFERENCE_MODE", "subprocess")
cfg.cosmos_policy_repo = os.environ.get(
    "COSMOS_PREDICT2_REPO",
    "/root/nas/junjie/cosmos_predict2_5/repos/cosmos-predict2.5",
)
cfg.cosmos_policy_python = os.environ.get(
    "COSMOS_POLICY_PYTHON",
    "/root/nas/junjie/cosmos_predict2_5/envs/predict2_py310/bin/python",
)
cfg.cosmos_policy_extra_pythonpath = os.environ.get(
    "COSMOS_POLICY_EXTRA_PYTHONPATH",
    "/root/nas/junjie/conda_envs/any_wam/lib/python3.10/site-packages",
)
cfg.cosmos_policy_local_model_dir = os.environ.get(
    "COSMOS_PREDICT25_LOCAL_MODEL_DIR",
    "/root/nas/junjie/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World",
)
cfg.cosmos_policy_config_name = os.environ.get(
    "COSMOS_POLICY_CONFIG_NAME", "cosmos_predict2_2b_480p_libero__inference_only")
cfg.cosmos_policy_config_file = os.environ.get(
    "COSMOS_POLICY_CONFIG_FILE",
    "cosmos_predict2/_src/predict2/cosmos_policy/config/config.py",
)
cfg.cosmos_policy_num_denoising_steps_action = int(
    os.environ.get("COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION", 5))
cfg.cosmos_policy_seed = int(os.environ.get("COSMOS_POLICY_SEED", 1))

cfg.output_dir = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_this_dir, "output_libero_cosmos_policy_stage1"),
)
cfg.wandb_name_prefix = "stage1_cosmos_policy_action"
cfg.enable_wandb = _env_bool("ENABLE_WANDB", False)

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
