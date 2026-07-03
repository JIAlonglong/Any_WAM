"""Thin Cosmos Policy compatibility layer for FlowMap stage smoke runs.

The official Cosmos Policy checkpoint is not a Wan/Diffusers transformer
directory. This module keeps the checkpoint validation and action-shape fallback
isolated from the normal WanVA teacher path.
"""

import json
import os
from pathlib import Path

import torch


COSMOS_POLICY_WEIGHT_NAMES = (
    "Cosmos-Policy-LIBERO-Predict2-2B.pt",
    "Cosmos-Policy-RoboCasa-Predict2-2B.pt",
)


def resolve_cosmos_policy_assets(path):
    """Validate a local Cosmos Policy checkpoint directory and return assets."""
    if path is None:
        raise FileNotFoundError("cosmos_policy teacher_model_path is required.")

    root = Path(os.path.abspath(os.path.expanduser(path)))
    if not root.is_dir():
        raise FileNotFoundError(f"Cosmos Policy checkpoint directory not found: {root}")

    weight_path = None
    for name in COSMOS_POLICY_WEIGHT_NAMES:
        candidate = root / name
        if candidate.is_file():
            weight_path = candidate
            break
    if weight_path is None:
        pt_files = sorted(root.glob("Cosmos-Policy-*.pt"))
        if pt_files:
            weight_path = pt_files[0]
    if weight_path is None:
        raise FileNotFoundError(
            f"Invalid Cosmos Policy checkpoint: expected one of {COSMOS_POLICY_WEIGHT_NAMES} in {root}"
        )

    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Invalid Cosmos Policy checkpoint: missing {config_path}")

    stats_candidates = sorted(root.glob("*_dataset_statistics.json"))
    if not stats_candidates:
        raise FileNotFoundError(
            f"Invalid Cosmos Policy checkpoint: missing *_dataset_statistics.json in {root}"
        )

    embedding_candidates = sorted(root.glob("*_t5_embeddings.pkl"))
    if not embedding_candidates:
        raise FileNotFoundError(
            f"Invalid Cosmos Policy checkpoint: missing *_t5_embeddings.pkl in {root}"
        )

    return {
        "root": str(root),
        "weight_path": str(weight_path),
        "config_path": str(config_path),
        "dataset_stats_path": str(stats_candidates[0]),
        "t5_embeddings_path": str(embedding_candidates[0]),
    }


class CosmosPolicyActionTeacher:
    """Shape-compatible action teacher for FlowMap latent stage checks.

    This adapter loads and validates the official Cosmos Policy checkpoint
    assets, but it does not pretend that Cosmos is a WanVA video teacher. The
    current FlowMap dataset supplies latent/action tensors, not raw RGB/proprio,
    so the fallback teacher target is the action FlowMatch target already in the
    batch. A future raw-observation adapter can replace ``action_target_tokens``
    with official Cosmos policy inference without changing the stage plumbing.
    """

    def __init__(self, checkpoint_dir, dtype=torch.bfloat16, device="cpu"):
        self.assets = resolve_cosmos_policy_assets(checkpoint_dir)
        self.dtype = dtype
        self.device = torch.device(device)
        with open(self.assets["config_path"], "r") as f:
            self.policy_config = json.load(f)
        with open(self.assets["dataset_stats_path"], "r") as f:
            self.dataset_stats = json.load(f)
        self._state_metadata = None

    @property
    def weight_path(self):
        return self.assets["weight_path"]

    def load_state_metadata(self):
        """Load the checkpoint state dict once to prove the file is readable."""
        if self._state_metadata is None:
            state = torch.load(self.weight_path, map_location="cpu")
            if not isinstance(state, dict):
                raise ValueError(f"Unexpected Cosmos Policy checkpoint type: {type(state)!r}")
            keys = state.get("model", state)
            if not isinstance(keys, dict):
                raise ValueError("Cosmos Policy checkpoint does not contain a state dict.")
            sample_keys = list(keys.keys())[:8]
            has_net_keys = any(key.startswith("net.") for key in keys.keys())
            if not has_net_keys:
                raise ValueError("Cosmos Policy checkpoint does not contain expected 'net.' keys.")
            self._state_metadata = {
                "num_keys": len(keys),
                "sample_keys": sample_keys,
            }
            del state
        return self._state_metadata

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self

    def requires_grad_(self, requires_grad=False):
        return self

    def parameters(self):
        return iter(())

    def action_target_tokens(self, action_dict):
        """Return action teacher tokens with WanVA action sequence shape."""
        target = action_dict.get("targets")
        if target is None:
            target = action_dict["latent"]
        # [B, C, F, N, 1] -> [B, F*N, C]
        tokens = target.permute(0, 2, 3, 4, 1).reshape(
            target.shape[0],
            target.shape[2] * target.shape[3] * target.shape[4],
            target.shape[1],
        )
        return tokens.to(dtype=self.dtype, device=target.device)

    def __call__(self, input_dict, train_mode=True, **_kwargs):
        latent_dict = input_dict["latent_dict"]
        action_dict = input_dict["action_dict"]
        video_target = latent_dict.get("targets")
        if video_target is None:
            video_target = latent_dict["noisy_latents"]
        patch_t, patch_h, patch_w = (1, 2, 2)
        bsz, channels, frames, height, width = video_target.shape
        video_tokens = video_target.reshape(
            bsz,
            channels,
            frames // patch_t,
            patch_t,
            height // patch_h,
            patch_h,
            width // patch_w,
            patch_w,
        )
        video_tokens = video_tokens.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(
            bsz,
            (frames // patch_t) * (height // patch_h) * (width // patch_w),
            channels * patch_t * patch_h * patch_w,
        )
        return video_tokens.to(dtype=self.dtype), self.action_target_tokens(action_dict)
