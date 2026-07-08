"""Small diagnostics helpers for RobotWin StepWAM ablations."""


ACTION_BRANCH_MARKERS = (
    "action_embedder",
    "condition_embedder_action",
    "action_proj_out",
    "action_norm",
)
VIDEO_BRANCH_MARKERS = (
    "patch_embedding_mlp",
    "condition_embedder.",
    "proj_out",
)


def classify_parameter_branch(name):
    lowered = str(name).lower()
    if any(marker in lowered for marker in ACTION_BRANCH_MARKERS):
        return "action"
    if "condition_embedder_action" not in lowered:
        if any(marker in lowered for marker in VIDEO_BRANCH_MARKERS):
            return "video"
    return "shared"


def opd_diagnostic_aliases(metrics, config):
    def metric(name, default=0.0):
        return float(metrics.get(name, default))

    def cfg(name, default=0.0):
        return float(getattr(config, name, default))

    action_transition_weight = cfg("action_transition_block_weight", cfg("action_block_weight", 1.0)) * cfg(
        "action_loss_weight", 1.0)
    action_local_weight = cfg("action_local_fm_block_weight", 1.0) * cfg("action_aware_weight", 0.0)

    return {
        "opd_loss_scale/L_endpoint_video": metric("opd_endpoint_aux_loss"),
        "opd_loss_scale/L_velocity_video": metric("opd_same_state_velocity_loss"),
        "opd_loss_scale/L_endpoint_action": metric("opd_action_transition_loss"),
        "opd_loss_scale/L_velocity_action": 0.0,
        "opd_loss_scale/beta_end_video": cfg("opd_endpoint_aux_weight", 0.0),
        "opd_loss_scale/beta_vel_video": cfg("opd_same_state_velocity_weight", 0.0),
        "opd_loss_scale/beta_end_action": action_transition_weight,
        "opd_loss_scale/beta_vel_action": 0.0,
        "opd_loss_ratio/endpoint_video": metric("opd_endpoint_aux_ratio"),
        "opd_loss_ratio/velocity_video": metric("opd_same_state_velocity_ratio"),
        "opd_loss_ratio/endpoint_action": metric("opd_action_transition_ratio"),
        "opd_modality_weight/video_transition": cfg("video_transition_weight", 1.0),
        "opd_modality_weight/action_transition": action_transition_weight,
        "opd_modality_weight/action_local": action_local_weight,
    }
