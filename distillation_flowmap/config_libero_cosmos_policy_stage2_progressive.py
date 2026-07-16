"""Progressive Cosmos-only Stage 2 for K=4 -> K=2 -> K=1 deployment."""

import copy
import os

from distillation_flowmap.config_libero_cosmos_policy_stage2_cosmos_latent_cdiff import (
    cfg as _base_cfg,
)
from distillation_flowmap.cosmos_mixed_step_policy import (
    get_mixed_step_policy_spec,
    parse_forced_indices,
)


cfg = copy.deepcopy(_base_cfg)
_this_dir = os.path.dirname(os.path.abspath(__file__))


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


_stage = os.environ.get("COSMOS_PROGRESSIVE_STAGE", "s4").strip().lower()
_stage_specs = {
    "s4": {
        "max_steps": 5000,
        "teacher_steps": 8,
        "student_steps": 4,
        "focus_prob": 0.80,
        "velocity_weight": 1.0,
        "grad_mode": "suffix",
        "grad_steps": 2,
    },
    "s2": {
        "max_steps": 3000,
        "teacher_steps": 4,
        "student_steps": 2,
        "focus_prob": 0.85,
        "velocity_weight": 0.50,
        "grad_mode": "last_step",
        "grad_steps": 1,
    },
    "s1": {
        "max_steps": 3000,
        "teacher_steps": 4,
        "student_steps": 1,
        "focus_prob": 0.90,
        "velocity_weight": 0.25,
        "grad_mode": "last_step",
        "grad_steps": 1,
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
cfg.opd_aux_interval = int(os.environ.get("OPD_AUX_INTERVAL", 8))
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

_mixed_policy_name = os.environ.get("COSMOS_MIXED_STEP_POLICY", "").strip().lower()
_forced_mixed_sequence = os.environ.get("COSMOS_MIXED_STEP_FORCE_SEQUENCE", "").strip()
if _mixed_policy_name:
    _mixed_policy_spec = get_mixed_step_policy_spec(_mixed_policy_name)
    cfg.cosmos_mixed_step_policy = _mixed_policy_spec.name
    cfg.opd_rollout_step_pairs = [list(pair) for pair in _mixed_policy_spec.rollout_step_pairs]
    cfg.opd_rollout_step_pair_weights = list(_mixed_policy_spec.weights)
    cfg.opd_rollout_step_forced_indices = parse_forced_indices(
        _forced_mixed_sequence,
        _mixed_policy_spec,
    )
    cfg.opd_rollout_selection_seed = int(
        os.environ.get(
            "COSMOS_MIXED_STEP_SELECTOR_SEED",
            os.environ.get("TRAIN_SEED", "0"),
        )
    )
    cfg.opd_rollout_selection_metrics_path = os.environ.get(
        "COSMOS_MIXED_STEP_METRICS_PATH",
        os.path.join(cfg.output_dir, "cosmos_mixed_step_opd.jsonl"),
    )
else:
    if _forced_mixed_sequence:
        raise ValueError(
            "COSMOS_MIXED_STEP_FORCE_SEQUENCE requires COSMOS_MIXED_STEP_POLICY"
        )
    # Keep legacy progressive stages byte-for-byte equivalent in their rollout
    # choice: no mixed policy, no forced selector, and no sidecar metrics file.
    cfg.cosmos_mixed_step_policy = None
    cfg.opd_rollout_step_pairs = [[_spec["teacher_steps"], _spec["student_steps"]]]
    cfg.opd_rollout_step_pair_weights = None
    cfg.opd_rollout_step_forced_indices = ()
    cfg.opd_rollout_selection_seed = int(
        os.environ.get("COSMOS_MIXED_STEP_SELECTOR_SEED", os.environ.get("TRAIN_SEED", "0"))
    )
    cfg.opd_rollout_selection_metrics_path = None
cfg.opd_selected_rollout_pair_context = None
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

cfg.opd_danceopd_rollout_steps = int(
    os.environ.get("OPD_DANCEOPD_ROLLOUT_STEPS", 16)
)
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
    os.environ.get("OPD_DANCEOPD_TERMINAL_PRIOR_TOLERANCE", 1e-6)
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

# Legacy Cosmos OPD controls remain neutral. The progressive path reports its
# endpoint and same-state velocity contributions directly.
cfg.video_transition_param = "velocity"
cfg.video_transition_weight = 0.0
cfg.opd_endpoint_aux_weight = 0.0
cfg.local_fm_weight = 0.0
cfg.opd_transition_group_weight = 1.0
cfg.opd_anchor_cap_ratio = -1.0
