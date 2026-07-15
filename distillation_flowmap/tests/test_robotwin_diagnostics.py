from types import SimpleNamespace

from distillation_flowmap.ablation.robotwin_diagnostics import (
    classify_parameter_branch,
    opd_diagnostic_aliases,
)


def test_classify_parameter_branch_separates_video_action_and_shared_names():
    assert classify_parameter_branch("module.action_embedder.weight") == "action"
    assert classify_parameter_branch("module.condition_embedder_action.delta.weight") == "action"
    assert classify_parameter_branch("module.action_proj_out.bias") == "action"

    assert classify_parameter_branch("module.patch_embedding_mlp.weight") == "video"
    assert classify_parameter_branch("module.condition_embedder.text_embedder.weight") == "video"
    assert classify_parameter_branch("module.proj_out.bias") == "video"

    assert classify_parameter_branch("module.blocks.0.attn1.to_q.weight") == "shared"


def test_opd_diagnostic_aliases_expose_endpoint_velocity_weights_and_ratios():
    metrics = {
        "opd_endpoint_aux_loss": 1.5,
        "opd_same_state_velocity_loss": 2.5,
        "opd_action_transition_loss": 3.5,
        "opd_endpoint_aux_ratio": 0.2,
        "opd_same_state_velocity_ratio": 0.3,
        "opd_action_transition_ratio": 0.4,
    }
    config = SimpleNamespace(
        opd_endpoint_aux_weight=0.1,
        opd_same_state_velocity_weight=0.2,
        video_transition_weight=1.0,
        action_transition_block_weight=4.0,
        action_loss_weight=0.5,
        action_local_fm_block_weight=2.0,
        action_aware_weight=0.25,
    )

    aliases = opd_diagnostic_aliases(metrics, config)

    assert aliases["opd_loss_scale/L_endpoint_video"] == 1.5
    assert aliases["opd_loss_scale/L_velocity_video"] == 2.5
    assert aliases["opd_loss_scale/L_endpoint_action"] == 3.5
    assert aliases["opd_loss_scale/beta_end_video"] == 0.1
    assert aliases["opd_loss_scale/beta_vel_video"] == 0.2
    assert aliases["opd_loss_scale/beta_end_action"] == 2.0
    assert aliases["opd_loss_ratio/velocity_video"] == 0.3
    assert aliases["opd_modality_weight/action_local"] == 0.5


def test_explicit_diagnostics_use_canonical_video_transition_endpoint():
    metrics = {
        "opd_video_transition_loss": 4.5,
        "opd_endpoint_aux_loss": 0.0,
        "opd_same_state_velocity_loss": 2.5,
        "opd_video_transition_ratio": 0.6,
        "opd_endpoint_aux_ratio": 0.0,
        "opd_same_state_velocity_ratio": 0.4,
    }
    config = SimpleNamespace(
        opd_loss_composition="explicit_hybrid",
        video_transition_weight=1.25,
        opd_endpoint_aux_weight=0.0,
        opd_same_state_velocity_weight=0.75,
    )

    aliases = opd_diagnostic_aliases(metrics, config)

    assert aliases["opd_loss_scale/L_endpoint_video"] == 4.5
    assert aliases["opd_loss_scale/beta_end_video"] == 1.25
    assert aliases["opd_loss_ratio/endpoint_video"] == 0.6
