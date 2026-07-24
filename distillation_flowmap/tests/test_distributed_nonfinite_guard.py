from pathlib import Path

import torch

from distillation_flowmap.distributed_safety import all_ranks_finite


def _trainer_source() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "flowmap_trainer.py").read_text(encoding="utf-8")


def test_nonfinite_microbatch_skip_is_collective_and_sticky_until_sync():
    source = _trainer_source()

    assert "from distillation_flowmap.distributed_safety import all_ranks_finite" in source
    assert "accumulation_skip_step = False" in source
    assert "accumulation_skip_step = not all_ranks_finite(" in source
    assert 'not (accumulation_skip_step or result.get("skip_step", False))' in source
    aux_gate = source.split(
        "if (not standalone_opd and self.use_opd_aux", 1
    )[1].split("opd_aux_interval =", 1)[0]
    assert "not accumulation_skip_step" in aux_gate
    assert "if accumulation_skip_step:" in source
    assert "skip_step = accumulation_skip_step" in source


def test_optimizer_skip_state_is_always_defined_and_logged_finitely():
    source = _trainer_source()
    optimizer_block = source.split(
        'if result["should_sync"]:', 1
    )[1].split("# 计算平均损失", 1)[0]

    assert "skipped_optimizer_step = accumulation_skip_step" in optimizer_block
    assert "total_norm = torch.zeros((), device=self.device)" in optimizer_block
    assert "skipped_optimizer_step = True" in optimizer_block
    assert "grad_branch_norms = {}" in optimizer_block
    assert "metric_values = torch.nan_to_num(" in source
    assert '"train/step_skipped": float(skipped_optimizer_step)' in source


def test_all_ranks_finite_works_without_an_initialized_process_group():
    assert all_ranks_finite(True, device=torch.device("cpu"))
    assert not all_ranks_finite(False, device=torch.device("cpu"))
