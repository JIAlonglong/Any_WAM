"""Full LIBERO Stage-2 with universal video-only DanceOPD supervision."""

import copy
import math
import os


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _positive_unique_steps(value, *, name):
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"{name} must be a comma-separated list of positive integers")
    try:
        steps = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a comma-separated list of positive integers"
        ) from exc
    if any(step <= 0 for step in steps) or len(set(steps)) != len(steps):
        raise ValueError(f"{name} must contain unique positive integers")
    return steps


# The legacy LIBERO Stage-2 module accepts only one integer for this variable.
# Preserve the public variable while hiding it during the base import, then
# parse the universal curriculum here.
_rollout_steps_text = os.environ.pop("OPD_DANCEOPD_ROLLOUT_STEPS", None)
try:
    from distillation_flowmap.config_libero_fullfinetune_stage2_anyflow import (
        cfg as _base_cfg,
    )
finally:
    if _rollout_steps_text is not None:
        os.environ["OPD_DANCEOPD_ROLLOUT_STEPS"] = _rollout_steps_text


cfg = copy.deepcopy(_base_cfg)

cfg.wandb_name_prefix = "libero_stage2_video_only_opd_universal"
cfg.max_train_steps = int(os.environ.get("MAX_TRAIN_STEPS", 10000))
cfg.save_interval = int(os.environ.get("SAVE_INTERVAL", 1000))
cfg.seed = int(os.environ.get("TRAIN_SEED", 42))

cfg.use_opd_aux = True
cfg.opd_query_mode = "danceopd"
cfg.opd_teacher_target_mode = "endpoint"
cfg.opd_rollout_step_pairs = [[8, 1], [8, 2], [8, 4]]
cfg.rollout_step_pairs = list(cfg.opd_rollout_step_pairs)
cfg.opd_danceopd_rollout_step_choices = _positive_unique_steps(
    _rollout_steps_text if _rollout_steps_text is not None else "2,4",
    name="OPD_DANCEOPD_ROLLOUT_STEPS",
)
cfg.opd_danceopd_rollout_steps = cfg.opd_danceopd_rollout_step_choices[0]
cfg.opd_danceopd_endpoint_weight = float(
    os.environ.get("OPD_DANCEOPD_ENDPOINT_WEIGHT", 1.0)
)
cfg.opd_danceopd_velocity_weight = float(
    os.environ.get("OPD_DANCEOPD_VELOCITY_WEIGHT", 1.0)
)
if (
    not math.isfinite(cfg.opd_danceopd_endpoint_weight)
    or cfg.opd_danceopd_endpoint_weight < 0
    or not math.isfinite(cfg.opd_danceopd_velocity_weight)
    or cfg.opd_danceopd_velocity_weight < 0
):
    raise ValueError("video DanceOPD weights must be finite and non-negative")

# This experiment deliberately has no action OPD. Environment overrides cannot
# weaken that contract; action learning remains enabled through the main loss.
cfg.opd_aux_action = False
cfg.opd_danceopd_action_velocity_weight = 0.0
cfg.opd_joint_action_rollout = False
cfg.action_condition_on_student_video = True
cfg.action_loss_weight = float(os.environ.get("ACTION_LOSS_WEIGHT", 1.0))
if cfg.action_loss_weight <= 0:
    raise ValueError("ACTION_LOSS_WEIGHT must remain positive")
if cfg.gt_regression_weight <= 0:
    raise ValueError("GT regression must remain enabled for the action anchor")

cfg.opd_aux_interval = int(os.environ.get("OPD_AUX_INTERVAL", 4))
cfg.opd_aux_prob = float(os.environ.get("OPD_AUX_PROB", 1.0))
cfg.opd_danceopd_diagnostic_interval = int(
    os.environ.get("OPD_DANCEOPD_DIAGNOSTIC_INTERVAL", 50)
)

cfg.mechanism_diagnostics = _env_bool("MECHANISM_DIAGNOSTICS", True)
cfg.mechanism_diagnostic_interval = int(
    os.environ.get("MECHANISM_DIAGNOSTIC_INTERVAL", 50)
)
cfg.mechanism_diagnostic_seed = int(
    os.environ.get("MECHANISM_DIAGNOSTIC_SEED", 42)
)
cfg.mechanism_diagnostic_r = int(os.environ.get("MECHANISM_DIAGNOSTIC_R", 500))
cfg.mechanism_diagnostic_s = int(os.environ.get("MECHANISM_DIAGNOSTIC_S", 250))
cfg.mechanism_diagnostic_teacher_steps = int(
    os.environ.get("MECHANISM_DIAGNOSTIC_TEACHER_STEPS", 8)
)
if cfg.mechanism_diagnostic_interval < 1:
    raise ValueError("MECHANISM_DIAGNOSTIC_INTERVAL must be positive")
if not (
    cfg.num_train_timesteps
    > cfg.mechanism_diagnostic_r
    > cfg.mechanism_diagnostic_s
    > 0
):
    raise ValueError("mechanism diagnostic levels must satisfy T > r > s > 0")
