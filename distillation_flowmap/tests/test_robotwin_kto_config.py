import importlib
import sys


def test_robotwin_kto_config_preserves_robotwin_base_and_exposes_sampler(monkeypatch):
    monkeypatch.setenv("STAGE2_SAMPLER", "group_balanced")
    monkeypatch.setenv("STAGE2_GROUP_BY", "episode")
    monkeypatch.setenv("STAGE2_SAMPLES_PER_GROUP", "3")
    sys.modules.pop("distillation_flowmap.config_robotwin_fullfinetune_stage2_kto_paopd", None)

    module = importlib.import_module("distillation_flowmap.config_robotwin_fullfinetune_stage2_kto_paopd")
    cfg = module.cfg

    assert cfg.env_type == "robotwin_tshape"
    assert cfg.height == 256
    assert cfg.width == 320
    assert cfg.obs_cam_keys == [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    assert cfg.opd_aux_variant == "kto_paopd_norm_focal"
    assert cfg.stage2_sampler == "group_balanced"
    assert cfg.stage2_group_by == "episode"
    assert cfg.stage2_samples_per_group == 3
