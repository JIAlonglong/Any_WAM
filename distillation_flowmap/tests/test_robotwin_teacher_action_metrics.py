from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from distillation_flowmap import rollout_eval_stage2
from distillation_flowmap import rollout_eval_video_stage2


EVALUATORS = (rollout_eval_stage2, rollout_eval_video_stage2)


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_rollout_student_target_sets_resume_online_from_target(evaluator):
    cfg = SimpleNamespace(resume_online_from_target=False)

    resolved = evaluator.apply_rollout_student_choice(cfg, "target")

    assert cfg.resume_online_from_target is True
    assert resolved == "target"


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_teacher_action_metrics_use_same_mask_and_gt(evaluator):
    metrics = evaluator.compute_teacher_action_gt_metrics(
        teacher_action_v=torch.tensor([[[2.0]]]),
        action_noisy_t=torch.tensor([[[1.0]]]),
        action_noisy_r=torch.tensor([[[3.0]]]),
        action_v_target=torch.tensor([[[4.0]]]),
        sigma_t=torch.tensor([0.0]),
        sigma_r=torch.tensor([1.0]),
        action_mask=torch.tensor([[True]]),
    )

    assert metrics["action_teacher_gt_xr_mse"].item() == pytest.approx(0.0)
    assert metrics["action_teacher_gt_v_mse"].item() == pytest.approx(4.0)


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_teacher_action_flag_rejects_legacy_teacher_cache(evaluator, tmp_path):
    with pytest.raises(ValueError, match="teacher action"):
        evaluator.validate_teacher_action_cache_compatibility(
            True, tmp_path / "cache.pt"
        )


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_teacher_action_cache_guard_allows_default_off(evaluator, tmp_path):
    evaluator.validate_teacher_action_cache_compatibility(
        False, tmp_path / "cache.pt"
    )


@pytest.mark.parametrize(
    "source_path",
    (
        Path(rollout_eval_stage2.__file__),
        Path(rollout_eval_video_stage2.__file__),
    ),
)
def test_evaluators_expose_metric_only_teacher_action_flags(source_path):
    source = source_path.read_text(encoding="utf-8")

    assert '"--rollout-student"' in source
    assert 'choices=("config", "online", "target")' in source
    assert '"--emit-teacher-action-gt"' in source
    assert '"rollout_source_requested"' in source
    assert '"rollout_source_resolved"' in source
