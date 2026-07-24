"""Progressive Cosmos-only Stage 2 for K=4 -> K=2 -> K=1 deployment."""

import copy
import math
import os

from distillation_flowmap.config_libero_cosmos_policy_stage2_cosmos_latent_cdiff import (
    cfg as _base_cfg,
)
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
cfg.training_contract_stage = "progressive_stage2"


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _parse_positive_step_choices(text, *, env_name):
    parts = [part.strip() for part in text.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"{env_name} must be a comma-separated list of positive integers")
    try:
        choices = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            f"{env_name} must be a comma-separated list of positive integers"
        ) from exc
    if any(choice <= 0 for choice in choices) or len(set(choices)) != len(choices):
        raise ValueError(f"{env_name} must contain unique positive integers")
    return choices


_stage = os.environ.get("COSMOS_PROGRESSIVE_STAGE", "s4").strip().lower()
_stage_specs = {
    "s4": {
        "max_steps": 5000,
        "rollout_step_pairs": [[8, 4]],
        "focus_prob": 0.80,
        "velocity_weight": 1.0,
        "grad_mode": "suffix",
        "grad_steps": 2,
        "danceopd_rollout_steps": "4",
    },
    "s2": {
        "max_steps": 3000,
        "rollout_step_pairs": [[4, 2]],
        "focus_prob": 0.85,
        "velocity_weight": 1.0,
        "grad_mode": "last_step",
        "grad_steps": 1,
        "danceopd_rollout_steps": "2",
    },
    "s1": {
        "max_steps": 3000,
        "rollout_step_pairs": [[4, 1]],
        "focus_prob": 0.90,
        "velocity_weight": 0.0,
        "grad_mode": "last_step",
        "grad_steps": 1,
        "danceopd_rollout_steps": "1",
    },
    "universal": {
        "max_steps": 5000,
        "rollout_step_pairs": [[8, 1], [8, 2], [8, 4]],
        "focus_prob": 0.85,
        "velocity_weight": 1.0,
        "grad_mode": "last_step",
        "grad_steps": 1,
        "danceopd_rollout_steps": "2,4",
    },
}
if _stage not in _stage_specs:
    raise ValueError(
        "COSMOS_PROGRESSIVE_STAGE must be one of "
        f"{sorted(_stage_specs)}, got {_stage!r}"
    )
_spec = _stage_specs[_stage]

_stage1_checkpoint = (
    "/root/nas/junjie/jj/Any_WAM/distillation_flowmap/"
    "output_libero_cosmos_policy_stage1_cosmos_latent_cdiff_8gpu_20260706_"
    "cosmos_latent_s1s2_8gpu/checkpoints/step_5000"
)
_run_id = os.environ.get("COSMOS_PROGRESSIVE_RUN_ID", "20260714")
_output_root = os.environ.get(
    "COSMOS_PROGRESSIVE_OUTPUT_ROOT",
    os.path.join(_this_dir, f"output_libero_cosmos_policy_stage2_progressive_{_run_id}"),
)

cfg.resume_from_path = os.environ.get("RESUME_FROM_PATH", _stage1_checkpoint)
cfg.resume_online_from_target = _env_bool("RESUME_ONLINE_FROM_TARGET", False)
cfg.reset_resume_step = _env_bool("RESET_RESUME_STEP", True)
cfg.resume_optimizer_state = _env_bool("RESUME_OPTIMIZER_STATE", False)
cfg.output_dir = os.environ.get("OUTPUT_DIR", os.path.join(_output_root, _stage))
cfg.wandb_name_prefix = f"cosmos_progressive_{_stage}"
cfg.enable_wandb = _env_bool("ENABLE_WANDB", False)

cfg.learning_rate = float(os.environ.get("LEARNING_RATE", 2e-7))
cfg.max_train_steps = int(os.environ.get("MAX_TRAIN_STEPS", _spec["max_steps"]))
cfg.save_interval = int(os.environ.get("SAVE_INTERVAL", 250))
cfg.stop_after_step = int(os.environ.get("STOP_AFTER_STEP", 0) or 0)
cfg.dataset_sample_manifest = os.environ.get("DATASET_SAMPLE_MANIFEST")
cfg.gradient_checkpointing = _env_bool("GRADIENT_CHECKPOINTING", True)
# The trainer keeps this true under OPD + activation checkpointing as a second
# guard against FSDP2 DTensor recompute failures.
cfg.use_fsdp1 = _env_bool("USE_FSDP1", True)

cfg.cosmos_video_target = False
cfg.cosmos_latent_target = True
cfg.cosmos_policy_use_raw_inference = _env_bool("COSMOS_POLICY_USE_RAW_INFERENCE", True)
cfg.return_raw_observation = True
cfg.skip_target_student_for_cosmos_latent = _env_bool(
    "SKIP_TARGET_STUDENT_FOR_COSMOS_LATENT", True
)

cfg.distill_video = True
cfg.distill_action = True
# Keep the joint state wholly in Cosmos teacher coordinates during the
# progressive refinement; the default training paths retain dataset actions.
cfg.cosmos_use_teacher_action_anchor = _env_bool(
    "COSMOS_USE_TEACHER_ACTION_ANCHOR", True
)
cfg.use_opd_aux = _env_bool("USE_OPD_AUX", True)
cfg.opd_aux_variant = "default"
cfg.opd_teacher_target_mode = "cosmos_latent_full"
cfg.opd_aux_weight = float(os.environ.get("OPD_AUX_WEIGHT", 0.10))
cfg.opd_aux_warmup_steps = int(os.environ.get("OPD_AUX_WARMUP_STEPS", 8))
cfg.opd_aux_interval = 8
cfg.opd_aux_phase = 2
cfg.opd_aux_prob = float(os.environ.get("OPD_AUX_PROB", 1.0))
cfg.opd_aux_gradient_checkpointing = _env_bool(
    "OPD_AUX_GRADIENT_CHECKPOINTING", True
)
cfg.opd_aux_use_nofsdp_rollout = False
cfg.opd_profile = _env_bool("OPD_PROFILE", False)
# CFG=3 is required by the inherited policy setup. Running its conditional and
# unconditional branches serially is numerically identical to the doubled
# batch path while fitting the two-GPU FSDP1 budget beside the raw Cosmos worker.
cfg.opd_serial_student_cfg = _env_bool("OPD_SERIAL_STUDENT_CFG", True)
cfg.opd_aux_empty_cache = _env_bool("OPD_AUX_EMPTY_CACHE", True)
# FSDP1 fits the 2B student on two H100s for either objective, but not the
# main and full OPD backward graphs at once. Reserve scheduled updates for OPD
# so both objectives remain part of the same training run without peak overlap.
cfg.opd_aux_standalone_step = _env_bool("OPD_AUX_STANDALONE_STEP", True)
# Activation checkpointing keeps the full 28x28 Cosmos OPD objective within
# the two-H100 FSDP1 budget. A smaller value remains an explicit fallback for
# constrained hardware, not the default training objective.
cfg.opd_cosmos_spatial_crop_size = int(
    os.environ.get("OPD_COSMOS_SPATIAL_CROP_SIZE", 28)
)

cfg.opd_rollout_step_pairs = copy.deepcopy(_spec["rollout_step_pairs"])
cfg.rollout_step_pairs = copy.deepcopy(cfg.opd_rollout_step_pairs)
cfg.opd_rollout_grad_mode = os.environ.get(
    "OPD_ROLLOUT_GRAD_MODE", _spec["grad_mode"]
).lower()
cfg.opd_rollout_grad_steps = int(
    os.environ.get("OPD_ROLLOUT_GRAD_STEPS", _spec["grad_steps"])
)
cfg.opd_endpoint_focus_prob = float(
    os.environ.get("OPD_ENDPOINT_FOCUS_PROB", _spec["focus_prob"])
)

# DanceOPD samples its configured semantic-query rollout choices independently
# of the endpoint pair sampled by the main progressive OPD objective.
cfg.opd_danceopd_rollout_step_choices = _parse_positive_step_choices(
    os.environ.get("OPD_DANCEOPD_ROLLOUT_STEPS", _spec["danceopd_rollout_steps"]),
    env_name="OPD_DANCEOPD_ROLLOUT_STEPS",
)
cfg.opd_danceopd_rollout_steps = cfg.opd_danceopd_rollout_step_choices[0]
cfg.opd_danceopd_query_alpha = float(
    os.environ.get("OPD_DANCEOPD_QUERY_ALPHA", 5.0)
)
cfg.opd_danceopd_query_beta = float(
    os.environ.get("OPD_DANCEOPD_QUERY_BETA", 2.0)
)
cfg.opd_danceopd_verify_terminal_prior = _env_bool(
    "OPD_DANCEOPD_VERIFY_TERMINAL_PRIOR", True
)
cfg.opd_danceopd_terminal_prior_tolerance = float(
    os.environ.get("OPD_DANCEOPD_TERMINAL_PRIOR_TOLERANCE", 2e-6)
)
cfg.opd_danceopd_terminal_prior_warn_factor = float(
    os.environ.get("OPD_DANCEOPD_TERMINAL_PRIOR_WARN_FACTOR", 0.5)
)
if (
    not math.isfinite(cfg.opd_danceopd_terminal_prior_tolerance)
    or cfg.opd_danceopd_terminal_prior_tolerance < 0
):
    raise ValueError(
        "OPD_DANCEOPD_TERMINAL_PRIOR_TOLERANCE must be finite and "
        "non-negative"
    )
if (
    not math.isfinite(cfg.opd_danceopd_terminal_prior_warn_factor)
    or not 0 <= cfg.opd_danceopd_terminal_prior_warn_factor <= 1
):
    raise ValueError(
        "OPD_DANCEOPD_TERMINAL_PRIOR_WARN_FACTOR must be finite and in [0, 1]"
    )
cfg.opd_danceopd_endpoint_weight = float(
    os.environ.get("OPD_DANCEOPD_ENDPOINT_WEIGHT", 1.0)
)
cfg.opd_danceopd_action_endpoint_weight = float(
    os.environ.get("OPD_DANCEOPD_ACTION_ENDPOINT_WEIGHT", 1.0)
)
cfg.opd_danceopd_velocity_weight = float(
    os.environ.get("OPD_DANCEOPD_VELOCITY_WEIGHT", _spec["velocity_weight"])
)
cfg.opd_joint_action_rollout = _env_bool("OPD_JOINT_ACTION_ROLLOUT", True)

cfg.deployment_joint_rollout_enabled = True
cfg.deployment_joint_rollout_interval = 4
cfg.deployment_joint_steps = (1, 2, 4)
cfg.deployment_timestep_start = 1000
cfg.deployment_timestep_end = 0
cfg.deployment_action_weight = 1.0
cfg.raw_teacher_window_is_auxiliary = True

# Legacy Cosmos OPD controls remain neutral. The progressive path reports its
# endpoint and same-state velocity contributions directly.
cfg.video_transition_param = "velocity"
cfg.video_transition_weight = 0.0
cfg.opd_endpoint_aux_weight = 0.0
cfg.local_fm_weight = 0.0
cfg.opd_transition_group_weight = 1.0
cfg.opd_anchor_cap_ratio = -1.0
