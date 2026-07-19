import importlib
import os
import sys

import torch
from wan_va.utils.scheduler import FlowMatchScheduler


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
    ):
        monkeypatch.delenv(name, raising=False)

    module = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    )
    module = importlib.reload(module)

    assert module.cfg.opd_teacher_target_mode == "cosmos_latent_full"
    assert module.cfg.opd_rollout_step_pairs == [[8, 4]]
    assert module.cfg.opd_endpoint_focus_prob > 0
    assert module.cfg.opd_danceopd_rollout_steps == 16
    assert module.cfg.opd_danceopd_endpoint_weight > 0
    assert module.cfg.opd_danceopd_velocity_weight > 0
    assert module.cfg.opd_serial_student_cfg is True
    assert module.cfg.opd_aux_standalone_step is True
    assert module.cfg.opd_aux_gradient_checkpointing is True
    assert module.cfg.opd_cosmos_spatial_crop_size == 28
    assert module.cfg.cosmos_use_teacher_action_anchor is True
    assert module.cfg.opd_joint_action_rollout is True
    assert module.cfg.opd_danceopd_action_endpoint_weight > 0


def test_progressive_k1_uses_four_step_teacher_not_two(monkeypatch):
    monkeypatch.setenv("COSMOS_PROGRESSIVE_STAGE", "s1")
    module = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    )
    module = importlib.reload(module)

    assert module.cfg.opd_rollout_step_pairs == [[4, 1]]
    assert module.cfg.opd_endpoint_focus_prob > 0


def test_progressive_exposes_chunk_stop_and_training_manifest(monkeypatch):
    monkeypatch.setenv("DATASET_SAMPLE_MANIFEST", "/tmp/cosmos_train.json")
    monkeypatch.setenv("STOP_AFTER_STEP", "250")
    module = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    )
    module = importlib.reload(module)

    assert module.cfg.dataset_sample_manifest == "/tmp/cosmos_train.json"
    assert module.cfg.stop_after_step == 250


def test_progressive_terminal_prior_tolerance_covers_libero_action_roundoff(
    monkeypatch,
):
    """The FP32 action schedule's shifted terminal sigma is slightly below one."""
    monkeypatch.delenv("OPD_DANCEOPD_TERMINAL_PRIOR_TOLERANCE", raising=False)
    module = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    )
    module = importlib.reload(module)

    scheduler = FlowMatchScheduler(
        shift=module.cfg.action_snr_shift,
        sigma_min=0.0,
        extra_one_step=True,
    )
    scheduler.set_timesteps(module.cfg.num_train_timesteps, training=True)
    clean = torch.full((1, 1, 1, 1, 1), 6.0, dtype=torch.float32)
    noise = torch.zeros_like(clean)
    terminal_t = torch.full((1, 1), float(module.cfg.num_train_timesteps))

    terminal_state = scheduler.add_noise(clean, noise, terminal_t, t_dim=2)
    terminal_error = float((terminal_state - noise).abs().amax())

    assert terminal_error > 1e-6
    assert terminal_error <= module.cfg.opd_danceopd_terminal_prior_tolerance
