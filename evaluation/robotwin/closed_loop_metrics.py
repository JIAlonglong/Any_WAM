"""Closed-loop metric helpers shared by the RoboTwin evaluator and ablation report."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from typing import Any


def _finite_nonnegative(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return number


def build_closed_loop_task_metrics(
    *,
    task: str,
    success_count: int,
    episode_count: int,
    chunk_latencies_s: Iterable[float],
    action_steps: int,
    cache_update_seconds: float,
    nfe: int,
) -> dict[str, Any]:
    """Build one task's official-success and policy-throughput summary."""
    if not task:
        raise ValueError("task must be non-empty")
    if episode_count <= 0:
        raise ValueError("episode_count must be positive")
    if success_count < 0 or success_count > episode_count:
        raise ValueError("success_count must be within [0, episode_count]")
    if action_steps < 0:
        raise ValueError("action_steps must be non-negative")
    if nfe <= 0:
        raise ValueError("nfe must be positive")

    latencies = [
        _finite_nonnegative(value, name="chunk latency")
        for value in chunk_latencies_s
    ]
    cache_seconds = _finite_nonnegative(
        cache_update_seconds,
        name="cache_update_seconds",
    )
    policy_seconds = float(sum(latencies))
    chunk_count = len(latencies)
    latency_mean = policy_seconds / chunk_count if chunk_count else None
    policy_hz = float(action_steps) / policy_seconds if policy_seconds > 0 else None

    return {
        "schema": "robotwin_closed_loop_task_metrics_v1",
        "task": str(task),
        "success_count": int(success_count),
        "episode_count": int(episode_count),
        "success_rate": float(success_count) / float(episode_count),
        "policy_chunks": chunk_count,
        "action_steps": int(action_steps),
        "policy_inference_seconds": policy_seconds,
        "cache_update_seconds": cache_seconds,
        "latency_per_chunk_seconds": latency_mean,
        "latency_per_chunk_p50_seconds": (
            float(statistics.median(latencies)) if latencies else None
        ),
        "policy_hz": policy_hz,
        "nfe": int(nfe),
    }

def _required_metric_number(record: Mapping[str, Any], key: str) -> float:
    if key not in record:
        raise ValueError(f"closed-loop metric is missing {key!r}")
    return _finite_nonnegative(record[key], name=key)


def build_closed_loop_summary(
    task_metrics: Iterable[Mapping[str, Any]],
    *,
    expected_task_count: int | None = None,
) -> dict[str, Any]:
    """Aggregate closed-loop metrics with task-macro success reporting."""
    records = [dict(record) for record in task_metrics]
    if not records:
        raise ValueError("task_metrics must be non-empty")
    if expected_task_count is not None and len(records) != expected_task_count:
        raise ValueError(
            f"expected {expected_task_count} task metrics, got {len(records)}"
        )

    tasks = [str(record.get("task", "")) for record in records]
    if any(not task for task in tasks):
        raise ValueError("every closed-loop metric must name a task")
    if len(set(tasks)) != len(tasks):
        raise ValueError("closed-loop task metrics must have unique task names")

    success_counts = []
    episode_counts = []
    chunks = []
    action_steps = []
    policy_seconds = []
    cache_seconds = []
    nfe_values = set()

    for record in records:
        success_count = int(record.get("success_count", -1))
        episode_count = int(record.get("episode_count", 0))
        if episode_count <= 0 or success_count < 0 or success_count > episode_count:
            raise ValueError("invalid success or episode count in closed-loop metrics")

        success_counts.append(success_count)
        episode_counts.append(episode_count)
        chunks.append(int(_required_metric_number(record, "policy_chunks")))
        action_steps.append(int(_required_metric_number(record, "action_steps")))
        policy_seconds.append(_required_metric_number(record, "policy_inference_seconds"))
        cache_seconds.append(_required_metric_number(record, "cache_update_seconds"))

        nfe = int(record.get("nfe", 0))
        if nfe <= 0:
            raise ValueError("closed-loop metric nfe must be positive")
        nfe_values.add(nfe)

    if len(nfe_values) != 1:
        raise ValueError("all closed-loop task metrics must use the same nfe")

    total_successes = sum(success_counts)
    total_episodes = sum(episode_counts)
    total_chunks = sum(chunks)
    total_actions = sum(action_steps)
    total_policy_seconds = float(sum(policy_seconds))

    return {
        "schema": "robotwin_closed_loop_summary_v1",
        "task_count": len(records),
        "tasks": tasks,
        "per_task": {str(record["task"]): record for record in records},
        "success_rate_macro": float(
            sum(
                success_count / episode_count
                for success_count, episode_count in zip(success_counts, episode_counts)
            )
            / len(records)
        ),
        "success_rate_micro": float(total_successes) / float(total_episodes),
        "success_count": total_successes,
        "episode_count": total_episodes,
        "policy_chunks": total_chunks,
        "action_steps": total_actions,
        "policy_inference_seconds": total_policy_seconds,
        "cache_update_seconds": float(sum(cache_seconds)),
        "latency_per_chunk_seconds": (
            total_policy_seconds / total_chunks if total_chunks else None
        ),
        "policy_hz": (
            float(total_actions) / total_policy_seconds
            if total_policy_seconds > 0
            else None
        ),
        "nfe": nfe_values.pop(),
    }
