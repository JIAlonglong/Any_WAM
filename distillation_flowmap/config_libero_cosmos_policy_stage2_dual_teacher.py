"""Stage 2 LIBERO FlowMap using Cosmos for actions and WanVA for video."""
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
    "output_libero_cosmos_policy_stage1_dual_teacher",
    "checkpoints",
    "step_5000",
)

cfg.teacher_backend = "cosmos_policy"
cfg.action_teacher_backend = "cosmos_policy"
cfg.teacher_model_path = os.environ.get(
    "COSMOS_POLICY_PATH",
    "/root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
)
cfg.student_base_model_path = os.environ.get(
    "STUDENT_BASE_MODEL_PATH",
    "/root/nas/junjie/jj/Any_WAM/checkpoints/lingbot-va-posttrain-libero",
)
cfg.video_teacher_backend = "wanva"
cfg.use_student_base_as_video_teacher = _env_bool("USE_STUDENT_BASE_AS_VIDEO_TEACHER", True)
cfg.video_teacher_model_path = os.environ.get(
    "VIDEO_TEACHER_MODEL_PATH",
    cfg.student_base_model_path,
)

cfg.cosmos_policy_validate_weights = _env_bool("COSMOS_POLICY_VALIDATE_WEIGHTS", False)
cfg.cosmos_policy_use_raw_inference = _env_bool("COSMOS_POLICY_USE_RAW_INFERENCE", True)
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

cfg.resume_from_path = os.environ.get("RESUME_FROM_PATH", _stage1_ckpt)
cfg.resume_online_from_target = _env_bool("RESUME_ONLINE_FROM_TARGET", True)
cfg.reset_resume_step = _env_bool("RESET_RESUME_STEP", True)
cfg.output_dir = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_this_dir, "output_libero_cosmos_policy_stage2_dual_teacher"),
)
cfg.wandb_name_prefix = "stage2_cosmos_dual_teacher"
cfg.enable_wandb = _env_bool("ENABLE_WANDB", False)

cfg.distill_mode = "flashwam"
cfg.distill_video = True
cfg.distill_action = True
cfg.action_aware = True
cfg.use_action_distill = True
cfg.use_gt_regression = True
cfg.use_central_diff = True
cfg.action_use_flowmap = True

cfg.diffusion_ratio = float(os.environ.get("DIFFUSION_RATIO", cfg.diffusion_ratio))
cfg.consistency_ratio = float(os.environ.get("CONSISTENCY_RATIO", cfg.consistency_ratio))
cfg.flowmap_ratio = float(os.environ.get("FLOWMAP_RATIO", cfg.flowmap_ratio))
cfg.use_opd_aux = _env_bool("USE_OPD_AUX", True)
cfg.opd_aux_action = _env_bool("OPD_AUX_ACTION", False)
cfg.learning_rate = float(os.environ.get("LEARNING_RATE", 5e-7))
cfg.max_train_steps = int(os.environ.get("MAX_TRAIN_STEPS", 5000))
cfg.save_interval = int(os.environ.get("SAVE_INTERVAL", 1000))
cfg.skip_teacher_compile = _env_bool("SKIP_TEACHER_COMPILE", True)
cfg.gradient_checkpointing = _env_bool("GRADIENT_CHECKPOINTING", False)
