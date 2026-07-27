from types import SimpleNamespace

import torch

from distillation_flowmap.distributed_safety import (
    NONFINITE_ORIGINS,
    accumulation_backward_allowed,
    all_ranks_finite,
    finish_accumulation_window,
    initialize_nonfinite_safety,
    record_nonfinite_origins,
    reduce_nonfinite_origins,
)


class _FakeReduceOp:
    MIN = "min"
    MAX = "max"


class _FakeDistributed:
    ReduceOp = _FakeReduceOp

    def __init__(self, peer_values):
        self.peer_values = list(peer_values)

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def all_reduce(self, tensor, op):
        peer = torch.as_tensor(
            self.peer_values.pop(0), device=tensor.device, dtype=tensor.dtype
        )
        if op == self.ReduceOp.MIN:
            tensor.copy_(torch.minimum(tensor, peer))
        elif op == self.ReduceOp.MAX:
            tensor.copy_(torch.maximum(tensor, peer))
        else:
            raise AssertionError(f"unexpected reduction {op}")


class _Counter:
    def __init__(self):
        self.calls = 0
        self.set_to_none = []

    def step(self):
        self.calls += 1

    def zero_grad(self, *, set_to_none):
        self.calls += 1
        self.set_to_none.append(set_to_none)


class _TrainerState(SimpleNamespace):
    def _update_ema(self):
        self.ema_calls += 1


def _finite_flags():
    return {origin: False for origin in NONFINITE_ORIGINS}


def test_all_ranks_finite_uses_min_semantics():
    assert all_ranks_finite(True, device=torch.device("cpu"))
    assert not all_ranks_finite(False, device=torch.device("cpu"))

    fake_dist = _FakeDistributed(peer_values=[0])
    assert not all_ranks_finite(
        True, device=torch.device("cpu"), dist_module=fake_dist
    )


def test_nonfinite_origins_use_max_semantics_and_remain_distinct():
    local = _finite_flags()
    local["video"] = True
    peer = [
        False,  # video
        True,  # action
        True,  # teacher_or_gt
        True,  # opd_endpoint
        True,  # opd_compositional
        True,  # opd_action
        False,  # gradient
    ]

    reduced = reduce_nonfinite_origins(
        local,
        device=torch.device("cpu"),
        dist_module=_FakeDistributed(peer_values=peer),
    )

    assert reduced == {
        "video": True,
        "action": True,
        "teacher_or_gt": True,
        "opd_endpoint": True,
        "opd_compositional": True,
        "opd_action": True,
        "gradient": False,
    }


def test_two_microbatch_nonfinite_window_is_sticky_and_skips_all_advances(
):
    trainer = _TrainerState(
        device=torch.device("cpu"),
        optimizer=_Counter(),
        lr_scheduler=_Counter(),
        ema_calls=0,
    )
    initialize_nonfinite_safety(trainer)

    auxiliary_backward_calls = 0
    dmd_backward_calls = 0
    first = _finite_flags()
    first["video"] = True
    record_nonfinite_origins(trainer, first, device=trainer.device)
    if accumulation_backward_allowed(trainer):
        auxiliary_backward_calls += 1
    if accumulation_backward_allowed(trainer):
        dmd_backward_calls += 1

    record_nonfinite_origins(trainer, _finite_flags(), device=trainer.device)
    if accumulation_backward_allowed(trainer):
        auxiliary_backward_calls += 1
    if accumulation_backward_allowed(trainer):
        dmd_backward_calls += 1

    skipped, total_norm, grad_branch_norms = finish_accumulation_window(
        trainer,
        total_norm=torch.tensor(1.0),
        grad_branch_norms={"video": 1.0},
        update_ema_fn=trainer._update_ema,
    )

    assert skipped is True
    assert total_norm.item() == 0.0
    assert grad_branch_norms == {}
    assert auxiliary_backward_calls == 0
    assert dmd_backward_calls == 0
    assert trainer.optimizer.calls == 1  # zero_grad only
    assert trainer.optimizer.set_to_none == [True]
    assert trainer.lr_scheduler.calls == 0
    assert trainer.ema_calls == 0
    assert trainer.nonfinite_skipped_windows == 1
    assert trainer.nonfinite_origin_counters["video"] == 1
    assert sum(trainer.nonfinite_origin_counters.values()) == 1
    assert trainer.skip_accumulation_window is False


def test_finite_window_advances_optimizer_scheduler_and_ema():
    trainer = _TrainerState(
        device=torch.device("cpu"),
        optimizer=_Counter(),
        lr_scheduler=_Counter(),
        ema_calls=0,
    )
    initialize_nonfinite_safety(trainer)

    record_nonfinite_origins(trainer, _finite_flags(), device=trainer.device)
    skipped, _, _ = finish_accumulation_window(
        trainer,
        total_norm=torch.tensor(1.0),
        grad_branch_norms={},
        update_ema_fn=trainer._update_ema,
    )

    assert skipped is False
    assert trainer.optimizer.calls == 2  # step + zero_grad
    assert trainer.optimizer.set_to_none == [True]
    assert trainer.lr_scheduler.calls == 1
    assert trainer.ema_calls == 1
    assert trainer.nonfinite_skipped_windows == 0
