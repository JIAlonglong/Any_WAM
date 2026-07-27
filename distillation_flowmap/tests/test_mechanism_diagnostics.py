import ast
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from distillation_flowmap.danceopd_query import aligned_anchor_mse
from distillation_flowmap.mechanism_diagnostics import (
    MECHANISM_RATIO_EPS,
    MechanismDiagnosticScheduler,
    compute_mechanism_metric_samples,
    diagnostic_runtime,
    diagnostic_seed,
    fan_out_mechanism_metrics,
    means_from_reduced_stats,
    pack_finite_metric_stats,
    reduce_metric_stats,
)


def test_aligned_anchor_is_squared_l2_while_training_endpoint_remains_mse():
    teacher_continuation = torch.tensor([[1.0, 2.0]])
    same_prior_endpoint = torch.tensor([[4.0, 6.0]])
    direct_route = torch.tensor([[1.0, 4.0]])
    composed_route = torch.tensor([[3.0, 1.0]])
    samples = compute_mechanism_metric_samples(
        teacher_continuation_video=teacher_continuation,
        same_prior_teacher_endpoint_video=same_prior_endpoint,
        direct_route_video=direct_route,
        composed_route_video=composed_route,
        teacher_endpoint_action=torch.zeros(1, 1),
        action_student_context=torch.zeros(1, 1),
        action_teacher_video_context=torch.zeros(1, 1),
        action_teacher_joint_context=None,
        action_mask=None,
        teacher_joint_available=False,
        shared_state_verified=True,
        same_prior_verified=True,
        effective_teacher_steps=8,
    )

    assert samples["mechanism/g_anchor"].item() == 25.0
    assert samples["mechanism/g_anchor_mse"].item() == 12.5
    assert samples["mechanism/g_comp"].item() == 13.0
    assert samples["mechanism/g_comp_mse"].item() == 6.5
    assert samples["mechanism/shared_state_verified"].item() == 1.0
    assert samples["mechanism/same_prior_verified"].item() == 1.0
    assert samples["mechanism/effective_teacher_steps_verified"].item() == 1.0

    training_endpoint = aligned_anchor_mse(
        torch.zeros_like(teacher_continuation),
        torch.zeros(1),
        torch.zeros_like(teacher_continuation),
        same_prior_endpoint,
    )
    assert training_endpoint.item() == 26.0
    assert training_endpoint.item() != samples["mechanism/g_anchor"].item()


def _samples(**overrides):
    values = {
        "teacher_continuation_video": torch.tensor([[3.0, 4.0], [2.0, 2.0]]),
        "same_prior_teacher_endpoint_video": torch.zeros(2, 2),
        "direct_route_video": torch.tensor([[2.0, 0.0], [1.0, 1.0]]),
        "composed_route_video": torch.zeros(2, 2),
        "teacher_endpoint_action": torch.zeros(2, 1),
        "action_student_context": torch.tensor([[3.0], [1.0]]),
        "action_teacher_video_context": torch.tensor([[1.0], [1.0]]),
        "action_teacher_joint_context": torch.zeros(2, 1),
        "action_mask": None,
    }
    values.update(overrides)
    return compute_mechanism_metric_samples(**values)


def test_exact_g_formulas_and_anchor_to_comp_ratio_direction():
    samples = _samples()

    assert torch.equal(samples["mechanism/g_anchor"], torch.tensor([25.0, 8.0]))
    assert torch.equal(
        samples["mechanism/g_anchor_mse"], torch.tensor([12.5, 4.0])
    )
    assert torch.equal(samples["mechanism/g_comp"], torch.tensor([4.0, 2.0]))
    assert torch.equal(samples["mechanism/g_comp_mse"], torch.tensor([2.0, 1.0]))
    assert torch.equal(
        samples["mechanism/g_anchor_to_comp_ratio"],
        torch.tensor([6.25, 4.0]),
    )


def test_action_gains_are_masked_signed_and_exact():
    samples = _samples(
        teacher_continuation_video=torch.zeros(1, 2),
        same_prior_teacher_endpoint_video=torch.zeros(1, 2),
        direct_route_video=torch.zeros(1, 2),
        composed_route_video=torch.zeros(1, 2),
        teacher_endpoint_action=torch.zeros(1, 2, 1),
        action_student_context=torch.tensor([[[3.0], [99.0]]]),
        action_teacher_video_context=torch.tensor([[[1.0], [88.0]]]),
        action_teacher_joint_context=torch.tensor([[[0.5], [77.0]]]),
        action_mask=torch.tensor([[[1.0], [0.0]]]),
    )

    assert samples["mechanism/video_to_action_oracle_gain"].item() == 8.0
    assert samples["mechanism/video_to_action_full_joint_gain"].item() == 8.75
    assert samples["mechanism/video_to_action_residual_action_gap"].item() == 0.75
    assert samples[
        "mechanism/video_to_action_recoverable_fraction"
    ].item() == pytest.approx(8.0 / 9.0)


def test_named_ratio_clamp_handles_zero_denominators():
    assert MECHANISM_RATIO_EPS > 0
    samples = _samples(
        direct_route_video=torch.ones(2, 2),
        composed_route_video=torch.ones(2, 2),
        action_student_context=torch.zeros(2, 1),
        action_teacher_video_context=torch.zeros(2, 1),
    )

    assert torch.isfinite(
        samples["mechanism/g_anchor_to_comp_ratio"]
    ).all()
    assert torch.equal(
        samples["mechanism/g_anchor_to_comp_ratio"],
        samples["mechanism/g_anchor_mse"] / MECHANISM_RATIO_EPS,
    )
    assert torch.equal(
        samples["mechanism/video_to_action_recoverable_fraction"],
        torch.zeros(2),
    )


def test_nonfinite_inputs_mark_invalid_but_ratio_leaves_remain_finite():
    contaminated = torch.tensor([[float("nan"), 0.0], [1.0, 1.0]])
    samples = _samples(teacher_continuation_video=contaminated)
    stats = pack_finite_metric_stats(samples)

    assert torch.isfinite(
        samples["mechanism/g_anchor_to_comp_ratio"]
    ).all()
    assert stats["mechanism/g_anchor_sum"].item() == 2.0
    assert stats["mechanism/g_anchor_count"].item() == 1.0
    assert stats["diagnostic_valid_count"].item() == 0.0
    assert all(torch.isfinite(value).all() for value in stats.values())


class _FakeReduceOp:
    SUM = "sum"


class _FakeDistributed:
    ReduceOp = _FakeReduceOp

    def __init__(self, peer):
        self.peer = peer
        self.calls = []

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def all_reduce(self, packed, op):
        self.calls.append((packed.clone(), op))
        packed.add_(
            torch.tensor(
                [self.peer[key] for key in sorted(self.peer)],
                dtype=packed.dtype,
                device=packed.device,
            )
        )


def test_distributed_reducer_uses_one_packed_global_sum_and_count():
    local = {
        "diagnostic_batch_count": torch.tensor(1.0),
        "diagnostic_valid_count": torch.tensor(1.0),
        "mechanism/g_anchor_count": torch.tensor(2.0),
        "mechanism/g_anchor_sum": torch.tensor(10.0),
    }
    peer = {
        "diagnostic_batch_count": 1.0,
        "diagnostic_valid_count": 1.0,
        "mechanism/g_anchor_count": 3.0,
        "mechanism/g_anchor_sum": 30.0,
    }
    fake = _FakeDistributed(peer)

    reduced = reduce_metric_stats(local, dist_module=fake)
    means = means_from_reduced_stats(reduced)

    assert len(fake.calls) == 1
    assert fake.calls[0][1] == _FakeReduceOp.SUM
    assert means["mechanism/g_anchor"] == 8.0
    assert means["mechanism/diagnostic_valid"] == 1.0


def test_zero_global_count_omits_unmeasured_mean_and_marks_unavailable():
    means = means_from_reduced_stats(
        {
            "diagnostic_batch_count": torch.tensor(1.0),
            "diagnostic_valid_count": torch.tensor(1.0),
            "mechanism/g_anchor_sum": torch.tensor(0.0),
            "mechanism/g_anchor_count": torch.tensor(0.0),
        }
    )

    assert "mechanism/g_anchor" not in means
    assert "mechanism/g_anchor_finite_count" not in means
    assert "mechanism/g_anchor_available" not in means
    assert means["mechanism/diagnostic_valid"] == 0.0
    assert all(math.isfinite(value) for value in means.values())


def test_mixed_clock_fanout_exposes_only_explicit_anchor_protocol_keys(
    tmp_path,
):
    samples = _samples(
        teacher_continuation_video=None,
        anchor_available=False,
        anchor_unavailable_mixed_clock=True,
        continuation_verified=False,
        effective_teacher_steps=None,
    )
    stats = pack_finite_metric_stats(samples)
    means = means_from_reduced_stats(stats)

    assert stats["mechanism/g_anchor_count"].item() == 0
    assert stats["mechanism/g_anchor_mse_count"].item() == 0
    assert means["mechanism/g_anchor_available"] == 0.0
    assert means["mechanism/g_anchor_unavailable_mixed_clock"] == 1.0
    assert means["mechanism/g_comp"] > 0
    assert means["mechanism/diagnostic_valid"] == 1.0
    anchor_public_keys = {
        key
        for key in means
        if key.startswith("mechanism/g_anchor")
        or key.startswith("mechanism/branch_dominance")
    }
    assert anchor_public_keys == {
        "mechanism/g_anchor_available",
        "mechanism/g_anchor_unavailable_mixed_clock",
    }

    jsonl = tmp_path / "mixed-clock.jsonl"
    tb_calls = []
    tb = SimpleNamespace(
        add_scalar=lambda key, value, step: tb_calls.append(
            (key, value, step)
        ),
        flush=mock.Mock(),
    )
    wandb_log = mock.Mock()
    wandb = SimpleNamespace(
        run=SimpleNamespace(settings=SimpleNamespace(mode="offline")),
        log=wandb_log,
    )
    fan_out_mechanism_metrics(
        means,
        step=10,
        rank=0,
        tb_writer=tb,
        wandb_module=wandb,
        jsonl_path=jsonl,
    )
    jsonl_record = json.loads(jsonl.read_text())
    jsonl_metrics = {
        key: value
        for key, value in jsonl_record.items()
        if key.startswith("mechanism/")
    }
    tb_metrics = {key: value for key, value, _ in tb_calls}
    wandb_metrics = wandb_log.call_args.args[0]
    assert jsonl_metrics == tb_metrics == wandb_metrics == means
    for public_mapping in (jsonl_metrics, tb_metrics, wandb_metrics):
        public_anchor_keys = {
            key
            for key in public_mapping
            if key.startswith("mechanism/g_anchor")
            or key.startswith("mechanism/branch_dominance")
        }
        assert public_anchor_keys == {
            "mechanism/g_anchor_available",
            "mechanism/g_anchor_unavailable_mixed_clock",
        }


def test_equal_clock_means_retain_measured_anchor_metrics_and_counts():
    means = means_from_reduced_stats(pack_finite_metric_stats(_samples()))

    assert means["mechanism/g_anchor"] == pytest.approx(16.5)
    assert means["mechanism/g_anchor_finite_count"] == 2.0
    assert means["mechanism/g_anchor_mse"] == pytest.approx(8.25)
    assert means["mechanism/g_anchor_mse_finite_count"] == 2.0
    assert means["mechanism/g_anchor_available"] == 1.0
    assert means["mechanism/g_anchor_unavailable_mixed_clock"] == 0.0
    assert "mechanism/g_anchor_available_finite_count" not in means
    assert "mechanism/g_anchor_available_available" not in means
    assert (
        "mechanism/g_anchor_unavailable_mixed_clock_finite_count"
        not in means
    )
    assert (
        "mechanism/g_anchor_unavailable_mixed_clock_available"
        not in means
    )


def test_scheduler_runs_due_only_after_success_and_defers_skipped_once():
    scheduler = MechanismDiagnosticScheduler(interval=100)

    assert not scheduler.observe(completed_step=0, optimizer_succeeded=True)
    assert not scheduler.observe(completed_step=99, optimizer_succeeded=True)
    assert not scheduler.observe(completed_step=100, optimizer_succeeded=False)
    assert scheduler.pending
    assert scheduler.observe(completed_step=101, optimizer_succeeded=True)
    assert not scheduler.observe(completed_step=101, optimizer_succeeded=True)
    assert not scheduler.pending
    assert scheduler.observe(completed_step=200, optimizer_succeeded=True)
    assert diagnostic_seed(42, 2, 3, 4) == 42 + 2 * 1009 + 3 * 97 + 4


@pytest.mark.parametrize("raise_inside", [False, True])
def test_diagnostic_runtime_restores_rng_and_model_modes(raise_inside):
    model_train = torch.nn.Linear(1, 1).train()
    model_eval = torch.nn.Linear(1, 1).eval()
    torch.manual_seed(123)
    random.seed(456)
    torch_before = torch.random.get_rng_state().clone()
    python_before = random.getstate()

    def run():
        with diagnostic_runtime(seed=999, models=(model_train, model_eval)):
            assert not model_train.training
            assert not model_eval.training
            torch.rand(4)
            random.random()
            if raise_inside:
                raise RuntimeError("probe failed")

    if raise_inside:
        with pytest.raises(RuntimeError, match="probe failed"):
            run()
    else:
        run()

    assert model_train.training
    assert not model_eval.training
    assert torch.equal(torch_before, torch.random.get_rng_state())
    assert python_before == random.getstate()


def test_fanout_uses_identical_mechanism_keys_for_offline_wandb_tb_jsonl(
    tmp_path,
):
    metrics = {
        "mechanism/g_anchor": 2.0,
        "mechanism/g_comp": 3.0,
        "mechanism/diagnostic_valid": 1.0,
    }
    tb_calls = []
    tb = SimpleNamespace(
        add_scalar=lambda key, value, step: tb_calls.append((key, value, step)),
        flush=mock.Mock(),
    )
    wandb_log = mock.Mock()
    wandb = SimpleNamespace(
        run=SimpleNamespace(settings=SimpleNamespace(mode="offline")),
        log=wandb_log,
    )
    jsonl = tmp_path / "diagnostics" / "mechanism_metrics.jsonl"

    fan_out_mechanism_metrics(
        metrics,
        step=100,
        rank=0,
        tb_writer=tb,
        wandb_module=wandb,
        jsonl_path=jsonl,
    )

    wandb_metrics = wandb_log.call_args.args[0]
    tb_metrics = {key: value for key, value, _ in tb_calls}
    record = json.loads(jsonl.read_text())
    jsonl_metrics = {
        key: value for key, value in record.items() if key.startswith("mechanism/")
    }
    assert wandb_metrics == tb_metrics == jsonl_metrics == metrics
    assert {step for _, _, step in tb_calls} == {100}
    wandb_log.assert_called_once_with(metrics, step=100)
    tb.flush.assert_called_once_with()


def test_fanout_rejects_online_wandb_and_nonzero_rank_writes_nothing(tmp_path):
    online = SimpleNamespace(
        run=SimpleNamespace(settings=SimpleNamespace(mode="online")),
        log=mock.Mock(),
    )
    path = tmp_path / "mechanism.jsonl"
    with pytest.raises(RuntimeError, match="offline"):
        fan_out_mechanism_metrics(
            {"mechanism/g_anchor": 1.0},
            step=1,
            rank=0,
            tb_writer=None,
            wandb_module=online,
            jsonl_path=path,
        )

    fan_out_mechanism_metrics(
        {"mechanism/g_anchor": 1.0},
        step=1,
        rank=1,
        tb_writer=None,
        wandb_module=online,
        jsonl_path=path,
    )
    assert not path.exists()


def test_trainer_calls_scheduler_after_completed_optimizer_boundary():
    trainer_path = (
        Path(__file__).resolve().parents[1] / "flowmap_trainer.py"
    )
    source = trainer_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FlowMapDistiller"
    )
    train = next(
        node
        for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    train_source = ast.get_source_segment(source, train)

    finish_pos = train_source.index("finish_accumulation_window(")
    step_pos = train_source.index("self.step += 1", finish_pos)
    diagnostic_pos = train_source.index(
        "self._maybe_run_mechanism_diagnostics(", step_pos
    )
    assert finish_pos < step_pos < diagnostic_pos
    assert "optimizer_succeeded=not skipped_optimizer_step" in train_source[
        diagnostic_pos : diagnostic_pos + 220
    ]
