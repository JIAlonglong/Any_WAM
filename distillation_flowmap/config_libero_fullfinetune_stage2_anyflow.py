"""Stage 2 AnyFlow-style continuation for LIBERO full-parameter fine-tuning."""
import copy
import os

from distillation_flowmap.config_libero_optimized import cfg as _base_cfg

cfg = copy.deepcopy(_base_cfg)
_this_dir = os.path.dirname(os.path.abspath(__file__))


def _env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")

_stage1_warmup_ckpt = os.path.join(
    _this_dir,
    "output_libero_fullft_stage1_warmup",
    "checkpoints",
    "step_500",
)

cfg.resume_from_path = os.environ.get("RESUME_FROM_PATH", _stage1_warmup_ckpt)
cfg.resume_online_from_target = _env_bool("RESUME_ONLINE_FROM_TARGET", True)
cfg.output_dir = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_this_dir, "output_libero_fullft_stage2_anyflow"),
)
cfg.wandb_name_prefix = "stage2_fullft_anyflow"

# Keep full-parameter mode when resuming from the full-model Stage 1 checkpoint.
cfg.use_lora = False
cfg.lora_rank = 0
cfg.lora_alpha = 0
cfg.lora_dropout = 0.0

# Stage 2 keeps the AnyFlow/FlowMap objective as the main loss. OPD is an
# auxiliary endpoint correction aligned with rollout_eval_video_stage2.py.
cfg.use_onpolicy_transition = _env_bool("USE_ONPOLICY_TRANSITION", False)
cfg.use_opd_aux = _env_bool("USE_OPD_AUX", True)
cfg.opd_aux_weight = float(os.environ.get("OPD_AUX_WEIGHT", 1.0))
cfg.opd_aux_warmup_steps = int(os.environ.get("OPD_AUX_WARMUP_STEPS", 0))
cfg.opd_aux_interval = int(os.environ.get("OPD_AUX_INTERVAL", 4))
cfg.opd_aux_prob = float(os.environ.get("OPD_AUX_PROB", 1.0))
cfg.opd_aux_use_nofsdp_rollout = _env_bool("OPD_AUX_USE_NOFSDP_ROLLOUT", False)
cfg.opd_profile = _env_bool("OPD_PROFILE", False)
cfg.opd_teacher_target_mode = os.environ.get("OPD_TEACHER_TARGET_MODE", "endpoint").lower()
_default_rollout_grad_mode = (
    "last_step" if cfg.opd_teacher_target_mode == "endpoint" else "endpoint"
)
cfg.opd_rollout_grad_mode = os.environ.get("OPD_ROLLOUT_GRAD_MODE", _default_rollout_grad_mode).lower()
cfg.opd_action_rollout_grad_mode = os.environ.get("OPD_ACTION_ROLLOUT_GRAD_MODE", cfg.opd_rollout_grad_mode).lower()
cfg.opd_same_state_velocity_weight = float(os.environ.get(
    "OPD_SAME_STATE_VELOCITY_WEIGHT", 0.0))
cfg.flowmap_aux_weight = float(os.environ.get("FLOWMAP_AUX_WEIGHT", 0.25))
_opd_aux_loss_clip = os.environ.get("OPD_AUX_LOSS_CLIP_VALUE")
cfg.opd_aux_loss_clip_value = (
    float(_opd_aux_loss_clip) if _opd_aux_loss_clip is not None else None
)
cfg.auto_scale_gradient_accumulation = _env_bool("AUTO_SCALE_GRADIENT_ACCUMULATION", True)
cfg.gradient_accumulation_reference = int(os.environ.get(
    "GRADIENT_ACCUMULATION_REFERENCE",
    cfg.gradient_accumulation_steps,
))

def _parse_step_pairs(text):
    return [
        [int(v) for v in pair.split(",")]
        for pair in text.split(";")
        if pair.strip()
    ]


# Endpoint mode makes the first pair item meaningful: N-step teacher endpoint
# vs K-step student endpoint. Defaults focus on compressed rollout rows.
cfg.rollout_step_pairs = (
    [[4, 1], [4, 2]]
    if cfg.opd_teacher_target_mode == "endpoint"
    else [[1, 1], [1, 2], [1, 4]]
)
_rollout_step_pairs = os.environ.get("ROLLOUT_STEP_PAIRS")
if _rollout_step_pairs:
    cfg.rollout_step_pairs = _parse_step_pairs(_rollout_step_pairs)
cfg.opd_rollout_step_pairs = list(cfg.rollout_step_pairs)
_opd_rollout_step_pairs = os.environ.get("OPD_ROLLOUT_STEP_PAIRS")
if _opd_rollout_step_pairs:
    cfg.opd_rollout_step_pairs = _parse_step_pairs(_opd_rollout_step_pairs)

# DanceOPD is opt-in for the LingBotVA Stage-2 continuation.  For a single
# S4 endpoint budget, query exactly the deployment rollout states.  The
# current sampler stores only pre-step states, so K=1/2 and mixed curricula
# require an explicit dense query grid rather than silently sampling a
# high-noise or mismatched trajectory.
cfg.opd_query_mode = os.environ.get("OPD_QUERY_MODE", "legacy").lower()
if cfg.opd_query_mode not in ("legacy", "danceopd"):
    raise ValueError("OPD_QUERY_MODE must be legacy or danceopd")
_danceopd_rollout_steps_env = os.environ.get("OPD_DANCEOPD_ROLLOUT_STEPS")
if _danceopd_rollout_steps_env is None:
    _danceopd_student_steps = {
        int(pair[1]) for pair in cfg.opd_rollout_step_pairs
    }
    if cfg.opd_query_mode == "danceopd":
        if len(_danceopd_student_steps) != 1:
            raise ValueError(
                "DanceOPD with mixed student budgets requires an explicit "
                "OPD_DANCEOPD_ROLLOUT_STEPS query grid"
            )
        _danceopd_default_steps = next(iter(_danceopd_student_steps))
        if _danceopd_default_steps < 4:
            raise ValueError(
                "Deployment-aligned DanceOPD defaults require K >= 4 because "
                "the current sampler has no terminal-state query; set an "
                "explicit OPD_DANCEOPD_ROLLOUT_STEPS dense grid for K=1/2"
            )
    else:
        _danceopd_default_steps = 16
else:
    _danceopd_default_steps = int(_danceopd_rollout_steps_env)
cfg.opd_danceopd_rollout_steps = int(_danceopd_default_steps)
if cfg.opd_danceopd_rollout_steps <= 0:
    raise ValueError("OPD_DANCEOPD_ROLLOUT_STEPS must be positive")
cfg.opd_danceopd_query_alpha = float(
    os.environ.get("OPD_DANCEOPD_QUERY_ALPHA", 5.0)
)
cfg.opd_danceopd_query_beta = float(
    os.environ.get("OPD_DANCEOPD_QUERY_BETA", 2.0)
)
cfg.opd_danceopd_velocity_weight = float(
    os.environ.get("OPD_DANCEOPD_VELOCITY_WEIGHT", 1.0)
)
cfg.opd_danceopd_endpoint_weight = float(
    os.environ.get("OPD_DANCEOPD_ENDPOINT_WEIGHT", 1.0)
)
cfg.opd_danceopd_verify_terminal_prior = _env_bool(
    "OPD_DANCEOPD_VERIFY_TERMINAL_PRIOR", True
)
cfg.opd_danceopd_terminal_prior_tolerance = float(
    os.environ.get("OPD_DANCEOPD_TERMINAL_PRIOR_TOLERANCE", 1e-6)
)
cfg.opd_danceopd_diagnostic_interval = int(
    os.environ.get("OPD_DANCEOPD_DIAGNOSTIC_INTERVAL", 50)
)

# Keep the AnyFlow mixed timestep distribution by default, while making v1/a1
# focused ablations possible without editing code.
cfg.diffusion_ratio = float(os.environ.get("DIFFUSION_RATIO", cfg.diffusion_ratio))
cfg.consistency_ratio = float(os.environ.get("CONSISTENCY_RATIO", cfg.consistency_ratio))
cfg.flowmap_ratio = float(os.environ.get("FLOWMAP_RATIO", cfg.flowmap_ratio))

cfg.enable_stage1_start_eval = (
    os.environ.get("ENABLE_STAGE1_START_EVAL", "0").lower()
    not in ("0", "false", "no", "off")
)
cfg.enable_stage1_start_eval_baseline = (
    os.environ.get("ENABLE_STAGE1_START_EVAL_BASELINE", "0").lower()
    not in ("0", "false", "no", "off")
)
cfg.stage1_start_eval_num_batches = int(os.environ.get("STAGE1_START_EVAL_NUM_BATCHES", 1))
cfg.stage1_start_eval_seed = int(os.environ.get("STAGE1_START_EVAL_SEED", 42))
cfg.stage1_start_eval_timesteps = [1000, 500]
cfg.stage1_start_eval_transition_pairs = [
    (1000, 500),
    (1000, 0),
]

# DanceOPD-style OPD query bias: train teacher/student matching mostly on
# low-noise student states instead of forcing high-noise one-step endpoints.
cfg.opd_query_bias = os.environ.get("OPD_QUERY_BIAS", "low_t").lower()
cfg.opd_query_bias_ratio = float(os.environ.get("OPD_QUERY_BIAS_RATIO", 1.0))
cfg.opd_low_noise_alpha = float(os.environ.get("OPD_LOW_NOISE_ALPHA", 5.0))
cfg.opd_low_noise_beta = float(os.environ.get("OPD_LOW_NOISE_BETA", 2.0))
cfg.opd_low_noise_max_sigma = float(os.environ.get("OPD_LOW_NOISE_MAX_SIGMA", 0.25))

# Conservative full-model continuation LR.
cfg.learning_rate = float(os.environ.get("LEARNING_RATE", 5e-7))
cfg.beta1 = float(os.environ.get("BETA1", 0.0))
cfg.beta2 = float(os.environ.get("BETA2", 0.999))
cfg.ema_decay = float(os.environ.get("EMA_DECAY", 0.99))
cfg.ema_warmup_steps = int(os.environ.get("EMA_WARMUP_STEPS", 200))
cfg.drop_text_ratio = float(os.environ.get("DROP_TEXT_RATIO", 0.1))
cfg.fuse_guidance_scale = float(os.environ.get("FUSE_GUIDANCE_SCALE", 3.0))
cfg.warmup_steps = int(os.environ.get("WARMUP_STEPS", 100))
cfg.max_train_steps = int(os.environ.get("MAX_TRAIN_STEPS", 5000))
cfg.save_interval = int(os.environ.get("SAVE_INTERVAL", 1000))
cfg.max_grad_norm = float(os.environ.get("MAX_GRAD_NORM", 0.3))
cfg.resume_optimizer_state = False
cfg.reset_resume_step = (
    os.environ.get("RESET_RESUME_STEP", "1").lower()
    not in ("0", "false", "no", "off")
)
cfg.skip_teacher_compile = os.environ.get(
    "SKIP_TEACHER_COMPILE", "1").lower() in ("1", "true", "yes", "on")
# OPD aux does an additional student backward path; PyTorch FSDP2 DTensor can
# hit mixed Tensor/DTensor dispatch during activation-checkpoint recompute.
# Keep it off by default for Stage 2 unless explicitly re-enabled.
cfg.gradient_checkpointing = os.environ.get(
    "GRADIENT_CHECKPOINTING", "0").lower() in ("1", "true", "yes", "on")
cfg.opd_aux_gradient_checkpointing = _env_bool("OPD_AUX_GRADIENT_CHECKPOINTING", False)

# Endpoint OPD must use x0/endpoint semantics. Velocity matching is only valid
# when teacher and student are evaluated at the same student-induced state.
_default_transition_param = (
    "x0" if cfg.opd_teacher_target_mode == "endpoint" else "velocity"
)
cfg.video_transition_param = os.environ.get("VIDEO_TRANSITION_PARAM", _default_transition_param)
cfg.action_transition_param = os.environ.get("ACTION_TRANSITION_PARAM", _default_transition_param)
cfg.video_transition_weight = float(os.environ.get("VIDEO_TRANSITION_WEIGHT", 1.0))
cfg.opd_endpoint_aux_weight = float(os.environ.get("OPD_ENDPOINT_AUX_WEIGHT", 0.1))
cfg.local_fm_weight = float(os.environ.get("LOCAL_FM_WEIGHT", 1e-4))
cfg.action_loss_weight = float(os.environ.get("ACTION_LOSS_WEIGHT", 1.0))
cfg.action_aware_weight = float(os.environ.get("ACTION_LOCAL_FM_WEIGHT", 0.003))
cfg.gt_regression_weight = float(os.environ.get("GT_REGRESSION_WEIGHT", 0.15))
cfg.action_transition_block_weight = float(os.environ.get("ACTION_TRANSITION_BLOCK_WEIGHT", 4.0))
cfg.action_local_fm_block_weight = float(os.environ.get("ACTION_LOCAL_FM_BLOCK_WEIGHT", 1.0))
cfg.opd_transition_group_weight = float(os.environ.get("OPD_TRANSITION_GROUP_WEIGHT", 25.0))
cfg.opd_anchor_cap_ratio = float(os.environ.get("OPD_ANCHOR_CAP_RATIO", 0.25))
# Kept for older configs/scripts; Stage 2 OPD uses the split weights above.
cfg.action_block_weight = float(os.environ.get("ACTION_BLOCK_WEIGHT", cfg.action_transition_block_weight))
cfg.cfg_min = float(os.environ.get("CFG_MIN", cfg.fuse_guidance_scale))
cfg.cfg_max = float(os.environ.get("CFG_MAX", cfg.fuse_guidance_scale))
cfg.opd_aux_action = (
    os.environ.get("OPD_AUX_ACTION", "0").lower()
    not in ("0", "false", "no", "off")
)

# Lightweight deterministic video/action eval. Disabled by default to avoid
# increasing peak memory on long Stage 2 runs; enable with ENABLE_LIGHT_EVAL=1.
cfg.enable_light_eval = os.environ.get(
    "ENABLE_LIGHT_EVAL", "0").lower() in ("1", "true", "yes", "on")
cfg.light_eval_interval = int(os.environ.get("LIGHT_EVAL_INTERVAL", cfg.save_interval))
cfg.light_eval_num_batches = int(os.environ.get("LIGHT_EVAL_NUM_BATCHES", 1))
cfg.light_eval_seed = int(os.environ.get("LIGHT_EVAL_SEED", 42))
cfg.light_eval_start_index = int(os.environ.get("LIGHT_EVAL_START_INDEX", 0))
cfg.light_eval_pairs = [(1000, 1000), (1000, 0), (750, 250)]

# Heavier deterministic rollout eval. It measures student multi-step rollout
# against teacher rollout/reference states, so keep the interval coarse.
cfg.enable_rollout_eval = os.environ.get(
    "ENABLE_ROLLOUT_EVAL", "0").lower() in ("1", "true", "yes", "on")
cfg.rollout_eval_interval = int(os.environ.get("ROLLOUT_EVAL_INTERVAL", 1000))
cfg.rollout_eval_num_batches = int(os.environ.get("ROLLOUT_EVAL_NUM_BATCHES", 1))
cfg.rollout_eval_seed = int(os.environ.get("ROLLOUT_EVAL_SEED", cfg.light_eval_seed))
cfg.rollout_eval_cfg_scale = float(os.environ.get("ROLLOUT_EVAL_CFG_SCALE", 5.0))
cfg.rollout_eval_teacher_steps = int(os.environ.get("ROLLOUT_EVAL_TEACHER_STEPS", 4))
_rollout_eval_student_steps = os.environ.get("ROLLOUT_EVAL_STUDENT_STEPS", "1,2")
cfg.rollout_eval_student_steps = [
    int(v) for v in _rollout_eval_student_steps.split(",") if v.strip()
]
_rollout_eval_pairs = os.environ.get("ROLLOUT_EVAL_PAIRS", "1000,0")
cfg.rollout_eval_pairs = [
    tuple(float(v) for v in pair.split(","))
    for pair in _rollout_eval_pairs.split(";")
    if pair.strip()
]
cfg.rollout_eval_save_videos = os.environ.get(
    "ROLLOUT_EVAL_SAVE_VIDEOS", "0").lower() in ("1", "true", "yes", "on")
cfg.rollout_eval_video_dir = os.environ.get("ROLLOUT_EVAL_VIDEO_DIR") or None
cfg.rollout_eval_video_fps = int(os.environ.get("ROLLOUT_EVAL_VIDEO_FPS", 10))
cfg.rollout_eval_video_max_pairs = int(os.environ.get("ROLLOUT_EVAL_VIDEO_MAX_PAIRS", 1))
cfg.rollout_eval_video_sample_index = int(os.environ.get("ROLLOUT_EVAL_VIDEO_SAMPLE_INDEX", 0))
