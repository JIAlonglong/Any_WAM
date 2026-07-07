import importlib
import os
import sys

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FLOWMAP_DIR = os.path.join(REPO_ROOT, "distillation_flowmap")
WANVA_DIR = os.path.join(REPO_ROOT, "wan_va")
for path in (FLOWMAP_DIR, WANVA_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from distillation_flowmap.flowmap_trainer import _set_video_channel_config_from_heads


class _FakeModel:
    def __init__(self):
        self.patch_size = [1, 2, 2]
        self.patch_embedding_mlp = torch.nn.Linear(64, 3072)
        self.proj_out = torch.nn.Linear(3072, 64)


def test_set_video_channel_config_from_heads_uses_adapted_cosmos_latent_heads():
    config_dict = {
        "in_channels": 48,
        "out_channels": 48,
        "patch_size": [1, 2, 2],
    }

    _set_video_channel_config_from_heads(config_dict, _FakeModel())

    assert config_dict["in_channels"] == 16
    assert config_dict["out_channels"] == 16


def test_cosmos_latent_stage2_defaults_are_memory_safe(monkeypatch):
    for name in (
        "GRADIENT_CHECKPOINTING",
        "OPD_AUX_WARMUP_STEPS",
        "OPD_ROLLOUT_STEP_PAIRS",
        "OPD_TRANSITION_GROUP_WEIGHT",
    ):
        monkeypatch.delenv(name, raising=False)

    module = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_cosmos_latent_cdiff"
    )
    module = importlib.reload(module)

    assert module.cfg.gradient_checkpointing is True
    assert module.cfg.opd_aux_warmup_steps >= 8
    assert module.cfg.opd_rollout_step_pairs == [[1, 1]]
    assert module.cfg.opd_transition_group_weight <= 1e-2
