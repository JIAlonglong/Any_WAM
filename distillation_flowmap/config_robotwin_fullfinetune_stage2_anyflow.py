"""Stage 2 OPD continuation for RobotWin full-parameter fine-tuning."""
import copy
import os

from distillation_flowmap.config_robotwin_fullfinetune_stage1_warmup import (
    cfg as _stage1_cfg,
    _env_bool,
    _parse_adjacent_grid,
)

cfg = copy.deepcopy(_stage1_cfg)
_this_dir = os.path.dirname(os.path.abspath(__file__))

_stage1_default_ckpt = os.path.join(
    _this_dir,
    "output_robotwin_fullft_stage1_warmup",
    "checkpoints",
    "step_2000",
)

cfg.resume_from_path = os.environ.get("RESUME_FROM_PATH", _stage1_default_ckpt)
cfg.resume_online_from_target = _env_bool("RESUME_ONLINE_FROM_TARGET", True)
cfg.output_dir = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_this_dir, "output_robotwin_fullft_stage2_anyflow"),
)
cfg.wandb_name_prefix = "robotwin_stage2_fullft_anyflow"

# Stage 2 keeps the AnyFlow objective and adds endpoint OPD as an auxiliary
# teacher correction for the rollout lengths used by RobotWin eval.
cfg.use_onpolicy_transition = _env_bool("USE_ONPOLICY_TRANSITION", False)
cfg.use_opd_aux = _env_bool("USE_OPD_AUX", True)
cfg.opd_loss_composition = os.environ.get(
    "OPD_LOSS_COMPOSITION", "legacy"
).lower()
if cfg.opd_loss_composition not in ("legacy", "explicit_hybrid"):
    raise ValueError(
        "OPD_LOSS_COMPOSITION must be legacy or explicit_hybrid"
    )
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
cfg.opd_rollout_grad_mode = os.environ.get(
    "OPD_ROLLOUT_GRAD_MODE", _default_rollout_grad_mode).lower()
cfg.opd_action_rollout_grad_mode = os.environ.get(
    "OPD_ACTION_ROLLOUT_GRAD_MODE", cfg.opd_rollout_grad_mode).lower()
cfg.opd_rollout_grad_steps = int(os.environ.get(
    "OPD_ROLLOUT_GRAD_STEPS", 1))
if cfg.opd_rollout_grad_steps <= 0:
    raise ValueError("OPD_ROLLOUT_GRAD_STEPS must be positive")
cfg.opd_action_rollout_grad_steps = int(os.environ.get(
    "OPD_ACTION_ROLLOUT_GRAD_STEPS", cfg.opd_rollout_grad_steps))
if cfg.opd_action_rollout_grad_steps <= 0:
    raise ValueError("OPD_ACTION_ROLLOUT_GRAD_STEPS must be positive")
cfg.opd_fuse_action_teacher = _env_bool("OPD_FUSE_ACTION_TEACHER", True)
cfg.opd_serial_student_cfg = _env_bool("OPD_SERIAL_STUDENT_CFG", cfg.use_opd_aux)
cfg.opd_aux_empty_cache = _env_bool("OPD_AUX_EMPTY_CACHE", cfg.use_opd_aux)
cfg.opd_same_state_velocity_weight = float(os.environ.get(
    "OPD_SAME_STATE_VELOCITY_WEIGHT", 0.0))
cfg.flowmap_pair_mode = os.environ.get("FLOWMAP_PAIR_MODE", cfg.flowmap_pair_mode).lower()
cfg.opd_pair_mode = os.environ.get("OPD_PAIR_MODE", cfg.flowmap_pair_mode).lower()
cfg.flowmap_adjacent_grid = _parse_adjacent_grid(
    os.environ.get(
        "FLOWMAP_ADJACENT_GRID",
        ",".join(str(v) for v in cfg.flowmap_adjacent_grid),
    )
)

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
# vs K-step student endpoint. Defaults focus OPD on the harder compressed rows.
cfg.rollout_step_pairs = (
    [[4, 1], [4, 2]]
    if cfg.opd_teacher_target_mode == "endpoint"
    else [[1, 1], [2, 1], [4, 1], [4, 2]]
)
_rollout_step_pairs = os.environ.get("ROLLOUT_STEP_PAIRS")
if _rollout_step_pairs:
    cfg.rollout_step_pairs = _parse_step_pairs(_rollout_step_pairs)
cfg.opd_rollout_step_pairs = list(cfg.rollout_step_pairs)
_opd_rollout_step_pairs = os.environ.get("OPD_ROLLOUT_STEP_PAIRS")
if _opd_rollout_step_pairs:
    cfg.opd_rollout_step_pairs = _parse_step_pairs(_opd_rollout_step_pairs)

# DanceOPD-style low-noise query bias. Adjacent-grid OPD is the local
# transition ablation, so keep those endpoints exact unless explicitly
# overridden.
_default_opd_query_bias = (
    "none" if cfg.opd_pair_mode in ("adjacent_grid", "adjacent", "local") else "low_t"
)
cfg.opd_query_bias = os.environ.get("OPD_QUERY_BIAS", _default_opd_query_bias).lower()
cfg.opd_query_bias_ratio = float(os.environ.get("OPD_QUERY_BIAS_RATIO", 1.0))
cfg.opd_low_noise_alpha = float(os.environ.get("OPD_LOW_NOISE_ALPHA", 5.0))
cfg.opd_low_noise_beta = float(os.environ.get("OPD_LOW_NOISE_BETA", 2.0))
cfg.opd_low_noise_max_sigma = float(os.environ.get("OPD_LOW_NOISE_MAX_SIGMA", 0.25))

# Conservative continuation hyperparameters for full-model training.
cfg.learning_rate = float(os.environ.get("LEARNING_RATE", 5e-7))
cfg.beta1 = float(os.environ.get("BETA1", 0.0))
cfg.beta2 = float(os.environ.get("BETA2", 0.999))
cfg.ema_decay = float(os.environ.get("EMA_DECAY", 0.99))
cfg.ema_warmup_steps = int(os.environ.get("EMA_WARMUP_STEPS", 200))
cfg.drop_text_ratio = float(os.environ.get("DROP_TEXT_RATIO", 0.1))
cfg.fuse_guidance_scale = float(os.environ.get("FUSE_GUIDANCE_SCALE", 5.0))
cfg.cfg_min = float(os.environ.get("CFG_MIN", cfg.fuse_guidance_scale))
cfg.cfg_max = float(os.environ.get("CFG_MAX", cfg.fuse_guidance_scale))
cfg.warmup_steps = int(os.environ.get("WARMUP_STEPS", 100))
cfg.max_train_steps = int(os.environ.get("MAX_TRAIN_STEPS", 5000))
cfg.save_interval = int(os.environ.get("SAVE_INTERVAL", 1000))
cfg.max_grad_norm = float(os.environ.get("MAX_GRAD_NORM", 0.3))
cfg.use_8bit_optimizer = _env_bool("USE_8BIT_OPTIMIZER", True)
cfg.resume_optimizer_state = _env_bool("RESUME_OPTIMIZER_STATE", False)
cfg.reset_resume_step = _env_bool("RESET_RESUME_STEP", True)
cfg.skip_teacher_compile = _env_bool("SKIP_TEACHER_COMPILE", True)
cfg.gradient_checkpointing = _env_bool("GRADIENT_CHECKPOINTING", True)
cfg.use_fsdp1 = _env_bool("USE_FSDP1", True)
# OPD aux does an additional student backward path and needs activation
# checkpointing on a single H100. FSDP1 is forced above for this path, avoiding
# the FSDP2 DTensor/checkpoint recompute issue while keeping the memory peak low.
cfg.opd_aux_gradient_checkpointing = _env_bool("OPD_AUX_GRADIENT_CHECKPOINTING", True)

# OPD loss balance. Action OPD is off by default; enable it after checking
# stage2 speed with video OPD.
_default_transition_param = (
    "x0" if cfg.opd_teacher_target_mode == "endpoint" else "velocity"
)
cfg.video_transition_param = os.environ.get("VIDEO_TRANSITION_PARAM", _default_transition_param)
cfg.action_transition_param = os.environ.get("ACTION_TRANSITION_PARAM", _default_transition_param)
cfg.video_transition_weight = float(os.environ.get("VIDEO_TRANSITION_WEIGHT", 1.0))
cfg.opd_endpoint_aux_weight = float(os.environ.get("OPD_ENDPOINT_AUX_WEIGHT", 0.1))
cfg.local_fm_weight = float(os.environ.get("LOCAL_FM_WEIGHT", 1e-4))
cfg.action_loss_weight = float(os.environ.get("ACTION_LOSS_WEIGHT", cfg.action_loss_weight))
cfg.action_aware_weight = float(os.environ.get("ACTION_LOCAL_FM_WEIGHT", 0.01))
cfg.gt_regression_weight = float(os.environ.get("GT_REGRESSION_WEIGHT", 0.15))
cfg.action_transition_block_weight = float(os.environ.get(
    "ACTION_TRANSITION_BLOCK_WEIGHT", getattr(cfg, "action_block_weight", 1.0)))
cfg.action_local_fm_block_weight = float(os.environ.get("ACTION_LOCAL_FM_BLOCK_WEIGHT", 1.0))
cfg.opd_transition_group_weight = float(os.environ.get("OPD_TRANSITION_GROUP_WEIGHT", 25.0))
cfg.opd_anchor_cap_ratio = float(os.environ.get("OPD_ANCHOR_CAP_RATIO", 0.25))
cfg.action_block_weight = float(os.environ.get(
    "ACTION_BLOCK_WEIGHT", cfg.action_transition_block_weight))
cfg.opd_aux_action = (
    os.environ.get("OPD_AUX_ACTION", "0").lower()
    not in ("0", "false", "no", "off")
)

cfg.enable_light_eval = _env_bool("ENABLE_LIGHT_EVAL", False)
cfg.light_eval_interval = int(os.environ.get("LIGHT_EVAL_INTERVAL", cfg.save_interval))
cfg.light_eval_num_batches = int(os.environ.get("LIGHT_EVAL_NUM_BATCHES", 1))
cfg.light_eval_seed = int(os.environ.get("LIGHT_EVAL_SEED", 42))
cfg.light_eval_start_index = int(os.environ.get("LIGHT_EVAL_START_INDEX", 0))
cfg.light_eval_pairs = [(1000, 1000), (1000, 0), (750, 250)]

cfg.enable_rollout_eval = _env_bool("ENABLE_ROLLOUT_EVAL", False)
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
cfg.rollout_eval_save_videos = _env_bool("ROLLOUT_EVAL_SAVE_VIDEOS", False)
cfg.rollout_eval_video_dir = os.environ.get("ROLLOUT_EVAL_VIDEO_DIR")
cfg.rollout_eval_video_fps = int(os.environ.get("ROLLOUT_EVAL_VIDEO_FPS", 10))
cfg.rollout_eval_video_max_pairs = int(os.environ.get("ROLLOUT_EVAL_VIDEO_MAX_PAIRS", 1))
cfg.rollout_eval_video_sample_index = int(os.environ.get("ROLLOUT_EVAL_VIDEO_SAMPLE_INDEX", 0))
