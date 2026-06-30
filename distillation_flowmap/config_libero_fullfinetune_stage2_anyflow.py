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
# auxiliary teacher correction queried on the student-visited state.
cfg.use_onpolicy_transition = _env_bool("USE_ONPOLICY_TRANSITION", False)
cfg.use_opd_aux = _env_bool("USE_OPD_AUX", True)
cfg.opd_aux_weight = float(os.environ.get("OPD_AUX_WEIGHT", 0.05))
cfg.opd_aux_warmup_steps = int(os.environ.get("OPD_AUX_WARMUP_STEPS", 0))
cfg.opd_aux_interval = int(os.environ.get("OPD_AUX_INTERVAL", 8))
cfg.opd_aux_prob = float(os.environ.get("OPD_AUX_PROB", 1.0))
cfg.opd_aux_use_nofsdp_rollout = _env_bool("OPD_AUX_USE_NOFSDP_ROLLOUT", False)
_opd_aux_loss_clip = os.environ.get("OPD_AUX_LOSS_CLIP_VALUE")
cfg.opd_aux_loss_clip_value = (
    float(_opd_aux_loss_clip) if _opd_aux_loss_clip is not None else None
)

# Use the broader low-step rollout curriculum again. Pairs are [teacher_steps,
# student_steps], so these keep the teacher at least as strong as the student.
cfg.rollout_step_pairs = [
    [1, 1],
    [2, 1],
    [4, 1],
    [4, 2],
]
_rollout_step_pairs = os.environ.get("ROLLOUT_STEP_PAIRS")
if _rollout_step_pairs:
    cfg.rollout_step_pairs = [
        [int(v) for v in pair.split(",")]
        for pair in _rollout_step_pairs.split(";")
        if pair.strip()
    ]
_opd_rollout_step_pairs = os.environ.get("OPD_ROLLOUT_STEP_PAIRS")
if _opd_rollout_step_pairs:
    cfg.opd_rollout_step_pairs = [
        [int(v) for v in pair.split(",")]
        for pair in _opd_rollout_step_pairs.split(";")
        if pair.strip()
    ]

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

# Keep AnyFlow mixed sampling as the default; no explicit v1/a1 endpoint bias.
cfg.one_step_focus_ratio = float(os.environ.get("ONE_STEP_FOCUS_RATIO", 0.0))
cfg.one_step_t_min = float(os.environ.get("ONE_STEP_T_MIN", 0.85))
cfg.video_one_step_focus_ratio = float(os.environ.get(
    "VIDEO_ONE_STEP_FOCUS_RATIO", cfg.one_step_focus_ratio))
cfg.video_one_step_t_min = float(os.environ.get(
    "VIDEO_ONE_STEP_T_MIN", cfg.one_step_t_min))
cfg.action_one_step_focus_ratio = float(os.environ.get(
    "ACTION_ONE_STEP_FOCUS_RATIO", cfg.one_step_focus_ratio))
cfg.action_one_step_t_min = float(os.environ.get(
    "ACTION_ONE_STEP_T_MIN", 0.995))
cfg.opd_aux_one_step_focus_ratio = float(os.environ.get(
    "OPD_AUX_ONE_STEP_FOCUS_RATIO", cfg.one_step_focus_ratio))
cfg.opd_aux_one_step_t_min = float(os.environ.get(
    "OPD_AUX_ONE_STEP_T_MIN", cfg.one_step_t_min))

# Conservative full-model continuation LR.
cfg.learning_rate = float(os.environ.get("LEARNING_RATE", 5e-7))
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
    "SKIP_TEACHER_COMPILE", "0").lower() in ("1", "true", "yes", "on")

# Make OPD/local-FM a regularizer rather than the dominant Stage 2 signal.
# Stage 2 trajectory distillation compares K-step student endpoints against
# N-step teacher endpoints, so the video transition target is x0/endpoint-style
# rather than same-state velocity matching.
cfg.video_transition_param = os.environ.get("VIDEO_TRANSITION_PARAM", "x0")
cfg.video_transition_weight = float(os.environ.get("VIDEO_TRANSITION_WEIGHT", 1.0))
cfg.local_fm_weight = float(os.environ.get("LOCAL_FM_WEIGHT", 0.001))
cfg.action_loss_weight = float(os.environ.get("ACTION_LOSS_WEIGHT", 1.0))
cfg.action_aware_weight = float(os.environ.get("ACTION_LOCAL_FM_WEIGHT", 0.1))
cfg.gt_regression_weight = float(os.environ.get("GT_REGRESSION_WEIGHT", 0.15))
cfg.action_block_weight = float(os.environ.get("ACTION_BLOCK_WEIGHT", 4.0))
cfg.cfg_min = float(os.environ.get("CFG_MIN", cfg.cfg_min))
cfg.cfg_max = float(os.environ.get("CFG_MAX", cfg.cfg_max))
cfg.opd_aux_action = (
    os.environ.get("OPD_AUX_ACTION", "1").lower()
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
