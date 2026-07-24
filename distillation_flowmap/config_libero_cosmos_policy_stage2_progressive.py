"""Progressive Cosmos-only Stage 2 for K=4 -> K=2 -> K=1 deployment."""

import copy
import json
import math
import os
import re
from pathlib import Path

from distillation_flowmap.config_libero_cosmos_policy_stage2_cosmos_latent_cdiff import (
    cfg as _base_cfg,
)
from distillation_flowmap.cosmos_training_contract import (
    ACTION_PACKING_SCHEMA,
    CONTRACT_VERSION,
)
from distillation_flowmap.cosmos_stage2_lineage import (
    validate_stage1_parent,
    validate_stage2_resume,
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

_run_id = os.environ.get("COSMOS_PROGRESSIVE_RUN_ID", "20260714")
_output_root = os.environ.get(
    "COSMOS_PROGRESSIVE_OUTPUT_ROOT",
    os.path.join(_this_dir, f"output_libero_cosmos_policy_stage2_progressive_{_run_id}"),
)

if not os.environ.get("STUDENT_BASE_MODEL_PATH"):
    raise ValueError("STUDENT_BASE_MODEL_PATH must be explicitly set")
if not os.environ.get("RESUME_FROM_PATH"):
    raise ValueError("RESUME_FROM_PATH must be explicitly set")
cfg.student_base_model_path = os.environ["STUDENT_BASE_MODEL_PATH"]
cfg.resume_from_path = os.environ["RESUME_FROM_PATH"]
for _lineage_name in (
    "PARENT_STAGE1_PATH",
    "PARENT_STAGE1_CONTRACT_IDENTITY",
    "STAGE2_LINEAGE_JSON",
):
    if not os.environ.get(_lineage_name):
        raise ValueError(f"{_lineage_name} must be explicitly set")
cfg.parent_stage1_path = os.environ["PARENT_STAGE1_PATH"]
cfg.parent_stage1_contract_identity = os.environ[
    "PARENT_STAGE1_CONTRACT_IDENTITY"
]
cfg.stage2_lineage_json = os.environ["STAGE2_LINEAGE_JSON"]
_validated_parent = validate_stage1_parent(
    Path(cfg.parent_stage1_path), expected_step=5000
)
if (
    _validated_parent.contract_identity
    != cfg.parent_stage1_contract_identity
):
    raise ValueError(
        "PARENT_STAGE1_CONTRACT_IDENTITY does not match validated Stage-1"
    )
_expected_student_base = (
    Path(_validated_parent.canonical_path) / "target_student"
).resolve(strict=True)
if Path(cfg.student_base_model_path).resolve(strict=True) != _expected_student_base:
    raise ValueError(
        "STUDENT_BASE_MODEL_PATH must identify validated Stage-1 target_student"
    )
try:
    _lineage_payload = json.loads(cfg.stage2_lineage_json)
except json.JSONDecodeError as exc:
    raise ValueError("STAGE2_LINEAGE_JSON must be valid JSON") from exc
if _lineage_payload != {
    "parent_stage1_contract_identity": _validated_parent.contract_identity,
    "parent_stage1_path": _validated_parent.canonical_path,
}:
    raise ValueError("STAGE2_LINEAGE_JSON does not match validated Stage-1")
cfg.resume_online_from_target = _env_bool("RESUME_ONLINE_FROM_TARGET", False)
cfg.reset_resume_step = _env_bool("RESET_RESUME_STEP", True)
cfg.resume_optimizer_state = _env_bool("RESUME_OPTIMIZER_STATE", False)
cfg.output_dir = os.environ.get("OUTPUT_DIR", os.path.join(_output_root, _stage))
_resume_path = Path(cfg.resume_from_path)
_canonical_parent = Path(_validated_parent.canonical_path)
_resolved_resume = _resume_path.resolve(strict=True)
if _resolved_resume == _canonical_parent:
    _fresh_flags = {
        "RESUME_ONLINE_FROM_TARGET": cfg.resume_online_from_target,
        "RESET_RESUME_STEP": cfg.reset_resume_step,
        "RESUME_OPTIMIZER_STATE": not cfg.resume_optimizer_state,
    }
    for _flag_name, _is_valid in _fresh_flags.items():
        if not _is_valid:
            raise ValueError(
                f"{_flag_name} is inconsistent with fresh Stage-2 launch"
            )
elif re.fullmatch(r"step_([1-9][0-9]*)", _resume_path.name):
    _expected_resume_step = int(_resume_path.name.removeprefix("step_"))
    validate_stage2_resume(
        _resume_path,
        arm_root=Path(cfg.output_dir),
        expected_step=_expected_resume_step,
        expected_parent=_validated_parent,
    )
    _resume_flags = {
        "RESUME_ONLINE_FROM_TARGET": not cfg.resume_online_from_target,
        "RESET_RESUME_STEP": not cfg.reset_resume_step,
        "RESUME_OPTIMIZER_STATE": cfg.resume_optimizer_state,
    }
    for _flag_name, _is_valid in _resume_flags.items():
        if not _is_valid:
            raise ValueError(
                f"{_flag_name} is inconsistent with Stage-2 resume"
            )
else:
    raise ValueError(
        "fresh RESUME_FROM_PATH must equal the canonical validated Stage-1 "
        "parent; Stage-2 resume must be a step_N checkpoint"
    )
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

# Mechanism diagnostics are observation-only.  R/S retain the public
# LingBotVA-compatible student action-map times; they are never forwarded to
# the official Cosmos teacher.  Teacher field queries use only the separately
# validated normalized Cosmos band below.
cfg.mechanism_diagnostics = _env_bool("MECHANISM_DIAGNOSTICS", True)
cfg.mechanism_diagnostic_interval = int(
    os.environ.get("MECHANISM_DIAGNOSTIC_INTERVAL", 100)
)
cfg.mechanism_diagnostic_seed = int(
    os.environ.get("MECHANISM_DIAGNOSTIC_SEED", 42)
)
cfg.mechanism_diagnostic_r = float(
    os.environ.get("MECHANISM_DIAGNOSTIC_R", 500)
)
cfg.mechanism_diagnostic_s = float(
    os.environ.get("MECHANISM_DIAGNOSTIC_S", 250)
)
cfg.mechanism_diagnostic_teacher_steps = int(
    os.environ.get("MECHANISM_DIAGNOSTIC_TEACHER_STEPS", 8)
)
cfg.mechanism_cosmos_t_min = float(
    os.environ.get("MECHANISM_COSMOS_T_MIN", 4.0 / 5.0)
)
cfg.mechanism_cosmos_t_max = float(
    os.environ.get("MECHANISM_COSMOS_T_MAX", 80.0 / 81.0)
)
cfg.mechanism_teacher_joint_available = False
if cfg.mechanism_diagnostic_interval <= 0:
    raise ValueError("MECHANISM_DIAGNOSTIC_INTERVAL must be positive")
if not (
    math.isfinite(cfg.mechanism_diagnostic_s)
    and math.isfinite(cfg.mechanism_diagnostic_r)
    and 0 <= cfg.mechanism_diagnostic_s
    < cfg.mechanism_diagnostic_r
    <= cfg.num_train_timesteps
):
    raise ValueError(
        "MECHANISM_DIAGNOSTIC_S and MECHANISM_DIAGNOSTIC_R are student-only "
        "times and must satisfy 0 <= S < R <= num_train_timesteps"
    )
if cfg.mechanism_diagnostic_teacher_steps != 8:
    raise ValueError(
        "MECHANISM_DIAGNOSTIC_TEACHER_STEPS must be 8 for the official Cosmos teacher"
    )
_mechanism_calibrated_min = 4.0 / 5.0
_mechanism_calibrated_max = 80.0 / 81.0
if not (
    math.isfinite(cfg.mechanism_cosmos_t_min)
    and math.isfinite(cfg.mechanism_cosmos_t_max)
    and _mechanism_calibrated_min
    <= cfg.mechanism_cosmos_t_min
    < cfg.mechanism_cosmos_t_max
    <= _mechanism_calibrated_max
):
    raise ValueError(
        "MECHANISM_COSMOS_T_MIN and MECHANISM_COSMOS_T_MAX must stay inside "
        "the calibrated Cosmos [4/5, 80/81] teacher band"
    )

# Legacy Cosmos OPD controls remain neutral. The progressive path reports its
# endpoint and same-state velocity contributions directly.
cfg.video_transition_param = "velocity"
cfg.video_transition_weight = 0.0
cfg.opd_endpoint_aux_weight = 0.0
cfg.local_fm_weight = 0.0
cfg.opd_transition_group_weight = 1.0
cfg.opd_anchor_cap_ratio = -1.0
