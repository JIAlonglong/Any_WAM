from types import SimpleNamespace

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
