import pytest
from pathlib import Path

from distillation_flowmap.opd_rollout_grad import rollout_step_requires_grad


def grad_mask(mode, num_steps, suffix_steps=1):
    return [
        rollout_step_requires_grad(
            mode=mode,
            step_index=index,
            num_steps=num_steps,
            suffix_steps=suffix_steps,
        )
        for index in range(num_steps)
    ]


def test_existing_rollout_grad_modes_keep_their_masks():
    assert grad_mask("endpoint", 4) == [False, False, False, False]
    assert grad_mask("last_step", 4) == [False, False, False, True]
    assert grad_mask("full", 4) == [True, True, True, True]


def test_suffix_keeps_only_requested_final_steps():
    assert grad_mask("suffix", 4, suffix_steps=2) == [
        False,
        False,
        True,
        True,
    ]


def test_suffix_larger_than_rollout_keeps_full_graph():
    assert grad_mask("suffix", 3, suffix_steps=8) == [True, True, True]


def test_rollout_grad_rejects_invalid_mode_and_sizes():
    with pytest.raises(ValueError, match="Unsupported OPD rollout grad mode"):
        grad_mask("random", 4)
    with pytest.raises(ValueError, match="num_steps must be positive"):
        rollout_step_requires_grad(
            mode="suffix",
            step_index=0,
            num_steps=0,
            suffix_steps=2,
        )
    with pytest.raises(ValueError, match="suffix_steps must be positive"):
        grad_mask("suffix", 4, suffix_steps=0)


def test_flowmap_video_and_action_rollouts_share_gradient_selector():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_step.py"
    ).read_text(encoding="utf-8")

    assert source.count("rollout_step_requires_grad(") >= 2
    assert "opd_rollout_grad_steps" in source
    assert "opd_action_rollout_grad_steps" in source
