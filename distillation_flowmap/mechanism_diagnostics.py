from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def squared_l2_per_sample(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return (lhs.float() - rhs.float()).flatten(1).square().sum(1)


def mse_per_sample(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return (lhs.float() - rhs.float()).flatten(1).square().mean(1)


def masked_action_mse_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    diff2 = (pred.float() - target.float()).square()
    if mask is None:
        return diff2.flatten(1).mean(1)
    expanded = torch.broadcast_to(mask.to(diff2), diff2.shape)
    numerator = (diff2 * expanded).flatten(1).sum(1)
    denominator = expanded.flatten(1).sum(1).clamp_min(1)
    return numerator / denominator


def compute_mechanism_metric_samples(
    *,
    teacher_cont_video: torch.Tensor,
    teacher_endpoint_video: torch.Tensor,
    student_direct_video: torch.Tensor,
    student_composed_video: torch.Tensor,
    student_field_video: torch.Tensor,
    teacher_field_video: torch.Tensor,
    teacher_endpoint_action: torch.Tensor,
    action_student_context: torch.Tensor,
    action_teacher_video_context: torch.Tensor,
    action_teacher_joint_context: torch.Tensor,
    action_mask: torch.Tensor | None,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    action_error_student = masked_action_mse_per_sample(
        action_student_context, teacher_endpoint_action, action_mask
    )
    action_error_teacher_video = masked_action_mse_per_sample(
        action_teacher_video_context, teacher_endpoint_action, action_mask
    )
    action_error_teacher_joint = masked_action_mse_per_sample(
        action_teacher_joint_context, teacher_endpoint_action, action_mask
    )
    oracle_gain = action_error_student - action_error_teacher_video

    return {
        "mechanism/g_anchor": squared_l2_per_sample(
            teacher_cont_video, teacher_endpoint_video
        ),
        "mechanism/g_anchor_mse": mse_per_sample(
            teacher_cont_video, teacher_endpoint_video
        ),
        "mechanism/g_comp": squared_l2_per_sample(
            student_direct_video, student_composed_video
        ),
        "mechanism/g_comp_mse": mse_per_sample(
            student_direct_video, student_composed_video
        ),
        "mechanism/video_endpoint_error": squared_l2_per_sample(
            student_direct_video, teacher_endpoint_video
        ),
        "mechanism/video_field_match_error": squared_l2_per_sample(
            student_field_video, teacher_field_video
        ),
        "mechanism/action_error_student_context": action_error_student,
        "mechanism/action_error_teacher_video_context": action_error_teacher_video,
        "mechanism/action_error_teacher_joint_context": action_error_teacher_joint,
        "mechanism/video_to_action_oracle_gain": oracle_gain,
        "mechanism/video_to_action_recoverable_fraction": oracle_gain.clamp_min(0)
        / action_error_student.clamp_min(eps),
        "mechanism/video_to_action_full_joint_gain": action_error_student
        - action_error_teacher_joint,
        "mechanism/video_to_action_residual_action_gap": action_error_teacher_video
        - action_error_teacher_joint,
    }


def pack_finite_metric_stats(
    samples: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    stats: dict[str, torch.Tensor] = {}
    finite_groups = []
    for name, values in samples.items():
        finite = torch.isfinite(values)
        finite_groups.append(finite.flatten())
        stats[f"{name}_sum"] = torch.where(
            finite, values, torch.zeros_like(values)
        ).sum()
        stats[f"{name}_count"] = finite.sum().to(values.dtype)

    reference = next(iter(samples.values()))
    all_finite = torch.cat(finite_groups).all()
    stats["diagnostic_valid"] = all_finite.to(
        device=reference.device, dtype=reference.dtype
    )
    return stats


def means_from_reduced_stats(
    stats: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    means: dict[str, float] = {}
    for name, total in stats.items():
        if not name.endswith("_sum"):
            continue
        metric_name = name.removesuffix("_sum")
        count = float(stats[f"{metric_name}_count"].item())
        means[metric_name] = float(total.item()) / count if count else math.nan
    return means


def diagnostic_due(step: int, interval: int) -> bool:
    return step > 0 and step % interval == 0


def diagnostic_seed(
    base_seed: int,
    diagnostic_index: int,
    batch_index: int,
    rank: int,
) -> int:
    return base_seed + diagnostic_index * 1009 + batch_index * 97 + rank
