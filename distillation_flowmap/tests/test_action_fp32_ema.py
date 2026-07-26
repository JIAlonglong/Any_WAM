from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from distillation.ema import (
    SelectiveFp32EMA,
    load_selective_ema_rank_state,
    save_selective_ema_rank_state,
)


def _named_parameter(name, value, *, dtype=torch.bfloat16):
    return [(name, torch.nn.Parameter(torch.tensor(value, dtype=dtype)))]


def test_selective_fp32_ema_accumulates_updates_lost_by_bf16():
    name = "action_proj_out.weight"
    target_named = _named_parameter(name, [0.03125])
    source_named = _named_parameter(name, [0.03125])
    ema = SelectiveFp32EMA(target_named, source_named, {name})

    source_named[0][1].data.copy_(
        torch.tensor([0.03173828125], dtype=torch.bfloat16)
    )
    initial = target_named[0][1].detach().clone()
    for _ in range(200):
        stats = ema.update(0.99)

    assert not torch.equal(target_named[0][1], initial)
    assert stats["selected_count"] == 1.0
    assert stats["selected_moved_count"] == 1.0
    assert stats["master_delta_max"] > 0.0
    assert math.isfinite(stats["target_source_max"])


def test_selective_fp32_ema_keeps_existing_behavior_for_nonselected_parameters():
    action_name = "action_embedder.weight"
    shared_name = "blocks.0.attn1.to_q.weight"
    target_named = [
        (action_name, torch.nn.Parameter(torch.tensor([0.0], dtype=torch.bfloat16))),
        (shared_name, torch.nn.Parameter(torch.tensor([0.0], dtype=torch.bfloat16))),
    ]
    source_named = [
        (action_name, torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))),
        (shared_name, torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))),
    ]

    ema = SelectiveFp32EMA(target_named, source_named, {action_name})
    ema.update(0.5)

    assert target_named[0][1].item() == pytest.approx(0.5)
    assert target_named[1][1].item() == pytest.approx(0.5)


@pytest.mark.parametrize("rate", [-0.01, 1.01, math.nan])
def test_selective_fp32_ema_rejects_invalid_rate(rate):
    name = "action_embedder.weight"
    ema = SelectiveFp32EMA(
        _named_parameter(name, [0.0]),
        _named_parameter(name, [1.0]),
        {name},
    )

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ema.update(rate)


def test_selective_fp32_ema_rejects_parameter_name_mismatch():
    with pytest.raises(ValueError, match="parameter names"):
        SelectiveFp32EMA(
            _named_parameter("action_embedder.weight", [0.0]),
            _named_parameter("action_proj_out.weight", [0.0]),
            {"action_embedder.weight"},
        )


def test_selective_fp32_ema_rejects_parameter_shape_mismatch():
    name = "action_embedder.weight"
    with pytest.raises(ValueError, match="shape"):
        SelectiveFp32EMA(
            _named_parameter(name, [0.0]),
            _named_parameter(name, [0.0, 1.0]),
            {name},
        )


def test_selective_fp32_ema_state_round_trip_preserves_next_update():
    name = "condition_embedder_action.time_proj.weight"
    target_a = _named_parameter(name, [0.03125])
    source_a = _named_parameter(name, [0.03173828125])
    ema_a = SelectiveFp32EMA(target_a, source_a, {name})
    for _ in range(17):
        ema_a.update(0.99)
    state = ema_a.state_dict()

    target_b = _named_parameter(name, target_a[0][1].detach().tolist())
    source_b = _named_parameter(name, source_a[0][1].detach().tolist())
    ema_b = SelectiveFp32EMA(target_b, source_b, {name})
    ema_b.load_state_dict(state)

    stats_a = ema_a.update(0.99)
    stats_b = ema_b.update(0.99)

    assert torch.equal(target_a[0][1], target_b[0][1])
    assert stats_a == stats_b


def test_selective_fp32_ema_rejects_incompatible_state_names():
    name = "action_embedder.weight"
    ema = SelectiveFp32EMA(
        _named_parameter(name, [0.0]),
        _named_parameter(name, [1.0]),
        {name},
    )
    state = ema.state_dict()
    state["selected_names"] = ("action_proj_out.weight",)

    with pytest.raises(ValueError, match="selected names"):
        ema.load_state_dict(state)


def test_selective_fp32_ema_rejects_nonfinite_state():
    name = "action_embedder.weight"
    ema = SelectiveFp32EMA(
        _named_parameter(name, [0.0]),
        _named_parameter(name, [1.0]),
        {name},
    )
    state = ema.state_dict()
    state["masters"][name] = torch.tensor([math.nan], dtype=torch.float32)

    with pytest.raises(ValueError, match="non-finite"):
        ema.load_state_dict(state)


def test_selective_fp32_ema_rank_checkpoint_round_trip(tmp_path):
    name = "action_proj_out.weight"
    target_a = _named_parameter(name, [0.03125])
    source_a = _named_parameter(name, [0.03173828125])
    ema_a = SelectiveFp32EMA(target_a, source_a, {name})
    ema_a.update(0.99)
    save_selective_ema_rank_state(
        ema_a,
        tmp_path,
        rank=0,
        world_size=1,
        step=50,
    )

    target_b = _named_parameter(name, [0.03125])
    source_b = _named_parameter(name, [0.03173828125])
    ema_b = SelectiveFp32EMA(target_b, source_b, {name})
    metadata = load_selective_ema_rank_state(
        ema_b,
        tmp_path,
        rank=0,
        world_size=1,
        expected_step=50,
    )

    assert metadata == {"rank": 0, "world_size": 1, "step": 50}
    assert ema_a.state_dict()["masters"][name].equal(
        ema_b.state_dict()["masters"][name]
    )


def test_selective_fp32_ema_rank_checkpoint_requires_file(tmp_path):
    name = "action_proj_out.weight"
    ema = SelectiveFp32EMA(
        _named_parameter(name, [0.0]),
        _named_parameter(name, [1.0]),
        {name},
    )

    with pytest.raises(FileNotFoundError, match="rank_00000"):
        load_selective_ema_rank_state(
            ema,
            tmp_path,
            rank=0,
            world_size=1,
            expected_step=50,
        )


@pytest.mark.parametrize(
    ("load_rank", "load_world_size", "load_step", "message"),
    [
        (1, 2, 50, "rank"),
        (0, 2, 50, "world size"),
        (0, 1, 51, "step"),
    ],
)
def test_selective_fp32_ema_rank_checkpoint_validates_metadata(
    tmp_path,
    load_rank,
    load_world_size,
    load_step,
    message,
):
    name = "action_proj_out.weight"
    ema = SelectiveFp32EMA(
        _named_parameter(name, [0.0]),
        _named_parameter(name, [1.0]),
        {name},
    )
    save_selective_ema_rank_state(
        ema,
        tmp_path,
        rank=0,
        world_size=1,
        step=50,
    )
    source = tmp_path / "rank_00000.pt"
    destination = tmp_path / f"rank_{load_rank:05d}.pt"
    if destination != source:
        destination.write_bytes(source.read_bytes())

    with pytest.raises(ValueError, match=message):
        load_selective_ema_rank_state(
            ema,
            tmp_path,
            rank=load_rank,
            world_size=load_world_size,
            expected_step=load_step,
        )


def test_flowmap_trainer_wires_fp32_ema_only_for_action_branch():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_trainer.py"
    ).read_text()

    assert "SelectiveFp32EMA" in source
    assert 'classify_parameter_branch(name) == "action"' in source
    assert "self._action_ema.update(ema_decay)" in source
    assert "update_ema(" in source


def test_flowmap_trainer_saves_and_strictly_restores_action_ema_state():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_trainer.py"
    ).read_text()

    assert "save_selective_ema_rank_state(" in source
    assert "load_selective_ema_rank_state(" in source
    assert '"action_ema_fp32"' in source
    assert "reset_resume_step" in source


def test_flowmap_trainer_logs_action_gradient_ratios():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_trainer.py"
    ).read_text()

    assert '"grad_norm/action_to_shared"' in source
    assert '"grad_norm/action_to_video"' in source
