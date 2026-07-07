import torch

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
