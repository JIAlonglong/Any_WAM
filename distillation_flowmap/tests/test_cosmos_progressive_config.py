import importlib
import os
import sys


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


def test_progressive_k1_uses_four_step_teacher_not_two(monkeypatch):
    monkeypatch.setenv("COSMOS_PROGRESSIVE_STAGE", "s1")
    module = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    )
    module = importlib.reload(module)

    assert module.cfg.opd_rollout_step_pairs == [[4, 1]]
    assert module.cfg.opd_endpoint_focus_prob > 0
