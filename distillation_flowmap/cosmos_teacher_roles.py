from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_COSMOS_BACKENDS = {"cosmos", "cosmos_policy", "cosmos-policy"}


@dataclass(frozen=True)
class TeacherRoles:
    action_backend: str
    action_model_path: str | None
    video_backend: str | None
    video_model_path: str | None
    uses_separate_video_teacher: bool


def _get(config: Any, name: str, default: Any = None) -> Any:
    return getattr(config, name, default)


def _normalize_backend(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    return normalized or default


def _is_cosmos_backend(value: str | None) -> bool:
    return value in _COSMOS_BACKENDS


def resolve_teacher_roles(config: Any) -> TeacherRoles:
    teacher_backend = _normalize_backend(_get(config, "teacher_backend", "wanva"), "wanva")
    action_backend = _normalize_backend(
        _get(config, "action_teacher_backend", teacher_backend),
        teacher_backend,
    )
    distill_video = bool(_get(config, "distill_video", True))

    action_model_path = _get(config, "action_teacher_model_path", None)
    if action_model_path is None:
        action_model_path = _get(config, "teacher_model_path", None)

    video_backend = _normalize_backend(_get(config, "video_teacher_backend", None), None)
    video_model_path = _get(config, "video_teacher_model_path", None)

    if not distill_video:
        return TeacherRoles(
            action_backend=action_backend,
            action_model_path=action_model_path,
            video_backend=None,
            video_model_path=None,
            uses_separate_video_teacher=False,
        )

    if _is_cosmos_backend(action_backend):
        if video_backend is None:
            video_backend = "wanva"
        if _is_cosmos_backend(video_backend):
            raise ValueError(
                "Cosmos Policy cannot be used as the FlowMap video teacher; "
                "provide a WanVA video_teacher_model_path."
            )
        if not video_model_path:
            raise ValueError(
                "distill_video=True with teacher_backend='cosmos_policy' requires "
                "cfg.video_teacher_model_path pointing to a WanVA/LingbotVA checkpoint."
            )
        return TeacherRoles(
            action_backend=action_backend,
            action_model_path=action_model_path,
            video_backend=video_backend,
            video_model_path=video_model_path,
            uses_separate_video_teacher=True,
        )

    if video_backend is None:
        video_backend = teacher_backend
    if video_model_path is None:
        video_model_path = _get(config, "teacher_model_path", None)
    return TeacherRoles(
        action_backend=action_backend,
        action_model_path=action_model_path,
        video_backend=video_backend,
        video_model_path=video_model_path,
        uses_separate_video_teacher=False,
    )
