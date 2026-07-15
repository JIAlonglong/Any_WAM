"""Explicit, rank-aware RNG setup for reproducible ablation training."""

import random

import numpy as np
import torch


def configure_training_seed(base_seed, rank=0):
    """Seed Python, NumPy, and Torch RNGs and return the rank-local seed."""
    effective_seed = int(base_seed) + int(rank)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    return effective_seed
