import random

import numpy as np
import torch

from distillation_flowmap.training_seed import configure_training_seed


def _draw():
    return (
        random.random(),
        float(np.random.rand()),
        torch.rand(4),
    )


def test_training_seed_replays_python_numpy_and_torch_rngs():
    assert configure_training_seed(37, rank=0) == 37
    first = _draw()

    assert configure_training_seed(37, rank=0) == 37
    second = _draw()

    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_training_seed_offsets_model_rng_by_rank():
    assert configure_training_seed(37, rank=2) == 39
