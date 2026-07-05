from types import SimpleNamespace
import sys
from pathlib import Path

import pytest

from distillation_flowmap.cosmos_teacher_roles import resolve_teacher_roles


def test_action_only_cosmos_does_not_require_video_teacher():
    cfg = SimpleNamespace(
        teacher_backend="cosmos_policy",
        teacher_model_path="/ckpts/cosmos",
        distill_video=False,
    )

    roles = resolve_teacher_roles(cfg)

    assert roles.action_backend == "cosmos_policy"
    assert roles.action_model_path == "/ckpts/cosmos"
    assert roles.video_backend is None
    assert roles.video_model_path is None
    assert roles.uses_separate_video_teacher is False


def test_dual_teacher_requires_wanva_video_model_path():
    cfg = SimpleNamespace(
        teacher_backend="cosmos_policy",
        teacher_model_path="/ckpts/cosmos",
        distill_video=True,
    )

    with pytest.raises(ValueError, match="video_teacher_model_path"):
        resolve_teacher_roles(cfg)


def test_dual_teacher_defaults_video_backend_to_wanva():
    cfg = SimpleNamespace(
        teacher_backend="cosmos_policy",
        teacher_model_path="/ckpts/cosmos",
        distill_video=True,
        video_teacher_model_path="/ckpts/wanva",
    )

    roles = resolve_teacher_roles(cfg)

    assert roles.action_backend == "cosmos_policy"
    assert roles.action_model_path == "/ckpts/cosmos"
    assert roles.video_backend == "wanva"
    assert roles.video_model_path == "/ckpts/wanva"
    assert roles.uses_separate_video_teacher is True


def test_rejects_cosmos_as_video_teacher_for_latent_flowmap():
    cfg = SimpleNamespace(
        teacher_backend="cosmos_policy",
        teacher_model_path="/ckpts/cosmos",
        distill_video=True,
        video_teacher_backend="cosmos_policy",
        video_teacher_model_path="/ckpts/cosmos",
    )

    with pytest.raises(
        ValueError,
        match="Cosmos Policy cannot be used as the FlowMap video teacher",
    ):
        resolve_teacher_roles(cfg)


def test_normal_wanva_config_keeps_single_teacher():
    cfg = SimpleNamespace(
        teacher_backend="wanva",
        teacher_model_path="/ckpts/wanva",
        distill_video=True,
    )

    roles = resolve_teacher_roles(cfg)

    assert roles.action_backend == "wanva"
    assert roles.action_model_path == "/ckpts/wanva"
    assert roles.video_backend == "wanva"
    assert roles.video_model_path == "/ckpts/wanva"
    assert roles.uses_separate_video_teacher is False


def test_dual_teacher_uses_student_base_as_default_video_teacher_when_requested():
    cfg = SimpleNamespace(
        teacher_backend="cosmos_policy",
        teacher_model_path="/ckpts/cosmos",
        student_base_model_path="/ckpts/wanva",
        distill_video=True,
        use_student_base_as_video_teacher=True,
    )

    roles = resolve_teacher_roles(cfg)

    assert roles.video_backend == "wanva"
    assert roles.video_model_path == "/ckpts/wanva"
    assert roles.uses_separate_video_teacher is True


def test_flowmap_step_accessors_split_action_and_video_teachers():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "wan_va"))

    from distillation_flowmap.flowmap_step import FlowMapStepMixin

    class DummyDistiller(FlowMapStepMixin):
        pass

    distiller = DummyDistiller()
    distiller.teacher = "fsdp_action_teacher"
    distiller._teacher_nofsdp = "action_teacher"
    distiller._video_teacher_nofsdp = "video_teacher"

    assert distiller._teacher_model == "action_teacher"
    assert distiller._action_teacher_model == "action_teacher"
    assert distiller._video_teacher_model == "video_teacher"


def test_flowmap_step_video_accessor_falls_back_to_default_teacher():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "wan_va"))

    from distillation_flowmap.flowmap_step import FlowMapStepMixin

    class DummyDistiller(FlowMapStepMixin):
        pass

    distiller = DummyDistiller()
    distiller.teacher = "fsdp_teacher"
    distiller._teacher_nofsdp = "default_teacher"

    assert distiller._video_teacher_model == "default_teacher"
