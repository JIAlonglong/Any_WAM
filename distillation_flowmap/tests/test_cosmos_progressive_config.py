import importlib
import os
import sys
from pathlib import Path


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FLOWMAP_DIR = os.path.join(REPO_ROOT, "distillation_flowmap")
WANVA_DIR = os.path.join(REPO_ROOT, "wan_va")
for path in (FLOWMAP_DIR, WANVA_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def test_progressive_s4_defaults_to_full_cosmos_opd(monkeypatch):
    for name in (
        "COSMOS_PROGRESSIVE_STAGE",
        "OPD_ENDPOINT_FOCUS_PROB",
        "OPD_DANCEOPD_VELOCITY_WEIGHT",
        "OPD_DANCEOPD_ENDPOINT_WEIGHT",
        "OPD_DANCEOPD_ROLLOUT_STEPS",
        "OPD_AUX_INTERVAL",
        "OPD_AUX_PHASE",
    ):
        monkeypatch.delenv(name, raising=False)

    module = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    )
    module = importlib.reload(module)

    assert module.cfg.opd_teacher_target_mode == "cosmos_latent_full"
    assert module.cfg.opd_rollout_step_pairs == [[8, 4]]
    assert module.cfg.opd_endpoint_focus_prob > 0
    assert module.cfg.opd_danceopd_rollout_steps == 4
    assert module.cfg.opd_danceopd_endpoint_weight > 0
    assert module.cfg.opd_danceopd_velocity_weight == 1.0
    assert module.cfg.opd_serial_student_cfg is True
    assert module.cfg.opd_aux_standalone_step is True
    assert module.cfg.opd_aux_gradient_checkpointing is True
    assert module.cfg.opd_cosmos_spatial_crop_size == 28
    assert module.cfg.cosmos_use_teacher_action_anchor is True
    assert module.cfg.opd_joint_action_rollout is True
    assert module.cfg.opd_danceopd_action_endpoint_weight > 0
    assert module.cfg.deployment_joint_rollout_enabled is True
    assert module.cfg.deployment_joint_rollout_interval == 4
    assert module.cfg.deployment_joint_steps == (1, 2, 4)
    assert module.cfg.deployment_timestep_start == 1000
    assert module.cfg.deployment_timestep_end == 0
    assert module.cfg.deployment_action_weight == 1.0
    assert module.cfg.raw_teacher_window_is_auxiliary is True
    assert module.cfg.opd_aux_interval == 8
    assert module.cfg.opd_aux_phase == 2


def test_progressive_raw_auxiliary_schedule_ignores_conflicting_environment(
    monkeypatch,
):
    monkeypatch.setenv("COSMOS_PROGRESSIVE_STAGE", "s4")
    monkeypatch.setenv("OPD_AUX_INTERVAL", "4")
    monkeypatch.setenv("OPD_AUX_PHASE", "0")
    module = importlib.reload(importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    ))

    assert module.cfg.opd_aux_interval == 8
    assert module.cfg.opd_aux_phase == 2


def test_progressive_k1_is_endpoint_only(monkeypatch):
    monkeypatch.setenv("COSMOS_PROGRESSIVE_STAGE", "s1")
    monkeypatch.delenv("OPD_DANCEOPD_ROLLOUT_STEPS", raising=False)
    monkeypatch.delenv("OPD_DANCEOPD_VELOCITY_WEIGHT", raising=False)
    module = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    )
    module = importlib.reload(module)

    assert module.cfg.opd_rollout_step_pairs == [[4, 1]]
    assert module.cfg.opd_endpoint_focus_prob > 0
    assert module.cfg.opd_danceopd_rollout_steps == 1
    assert module.cfg.opd_danceopd_velocity_weight == 0.0


def test_progressive_k2_uses_a_two_step_danceopd_query_grid(monkeypatch):
    monkeypatch.setenv("COSMOS_PROGRESSIVE_STAGE", "s2")
    monkeypatch.delenv("OPD_DANCEOPD_ROLLOUT_STEPS", raising=False)
    monkeypatch.delenv("OPD_DANCEOPD_VELOCITY_WEIGHT", raising=False)
    module = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    )
    module = importlib.reload(module)

    assert module.cfg.opd_rollout_step_pairs == [[4, 2]]
    assert module.cfg.opd_danceopd_rollout_steps == 2
    assert module.cfg.opd_danceopd_velocity_weight == 1.0


def test_progressive_universal_retains_original_lingbotva_definition(monkeypatch):
    monkeypatch.setenv("COSMOS_PROGRESSIVE_STAGE", "universal")
    monkeypatch.delenv("OPD_DANCEOPD_ROLLOUT_STEPS", raising=False)
    module = importlib.reload(importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    ))
    assert module.cfg.opd_rollout_step_pairs == [[8, 1], [8, 2], [8, 4]]
    assert module.cfg.opd_danceopd_rollout_step_choices == (2, 4)
    assert module.cfg.opd_danceopd_rollout_steps == 2
    assert module.cfg.opd_danceopd_velocity_weight == 1.0


def test_cosmos_danceopd_uses_terminal_semantic_states_and_skips_s1_velocity():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_step.py"
    ).read_text(encoding="utf-8")
    cosmos_dance_block = source.split(
        "def _cosmos_danceopd_velocity_loss("
    )[1].split("def _cosmos_latent_full_opd_aux_transition_step(")[0]
    cosmos_full_block = source.split(
        "def _cosmos_latent_full_opd_aux_transition_step("
    )[1].split("def _cosmos_latent_opd_aux_transition_step(")[0]

    assert "sample_semantic_query_indices(" in cosmos_dance_block
    assert cosmos_dance_block.count("video_states.append(current_video.detach().clone())") >= 2
    assert "if velocity_weight > 0:" in cosmos_full_block


def test_progressive_exposes_chunk_stop_and_training_manifest(monkeypatch):
    monkeypatch.setenv("DATASET_SAMPLE_MANIFEST", "/tmp/cosmos_train.json")
    monkeypatch.setenv("STOP_AFTER_STEP", "250")
    module = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    )
    module = importlib.reload(module)

    assert module.cfg.dataset_sample_manifest == "/tmp/cosmos_train.json"
    assert module.cfg.stop_after_step == 250
