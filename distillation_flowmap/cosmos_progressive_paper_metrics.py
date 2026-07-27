"""Paper-facing metric helpers for the Cosmos Progressive S4 evaluator."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from math import isfinite
from numbers import Real
from typing import Any


DEPLOYMENT_GRID = (1000, 750, 500, 250, 0)
PAPER_METRIC_NAMES = ("g_anchor", "g_comp", "video_ep", "field_match")


def internal_rollout_nodes():
    """Return the deployed intermediate states used for paper diagnostics."""
    return DEPLOYMENT_GRID[1:-1]


def composition_pairs():
    """Return consecutive deployed intermediate-state pairs."""
    return tuple(zip(DEPLOYMENT_GRID[1:-2], DEPLOYMENT_GRID[2:-1]))


def mean_video_mse(left, right):
    """Return MSE for video latents only; callers never pass action latents."""
    if left.shape != right.shape:
        raise ValueError("video tensors must have equal shapes")
    return (left.float() - right.float()).square().mean().item()


def paper_metric_template():
    """Return the complete Table-3 metric layout for one evaluation record."""
    node_slots = {f"node_{node}": None for node in internal_rollout_nodes()}
    return {
        "g_anchor": dict(node_slots),
        "g_comp": {
            f"pair_{source}_{target}": None
            for source, target in composition_pairs()
        },
        "video_ep": dict(node_slots),
        "field_match": dict(node_slots),
    }


def _identity_value(record, name):
    if name not in record:
        raise ValueError(f"record is missing required identity key: {name}")
    value = record[name]
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"record has an empty identity key: {name}")
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(f"record identity key must be hashable: {name}") from exc
    return value


def _flatten_metric_leaves(metrics, *, prefix=""):
    if not isinstance(metrics, Mapping):
        raise ValueError("record metrics must be a mapping")
    for name, value in metrics.items():
        name = str(name)
        if not name:
            raise ValueError("metric leaf names must not be empty")
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(value, Mapping):
            yield from _flatten_metric_leaves(value, prefix=path)
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"metric {path!r} must be a finite scalar")
        value = float(value)
        if not isfinite(value):
            raise ValueError(f"metric {path!r} must be a finite scalar")
        yield path, value


def _record_metrics(record):
    if "metrics" not in record:
        raise ValueError("record is missing metrics")
    flattened = {}
    for name, value in _flatten_metric_leaves(record["metrics"]):
        if name in flattened:
            raise ValueError(f"record has duplicate metric leaf: {name}")
        flattened[name] = value
    if not flattened:
        raise ValueError("record metrics must contain at least one scalar leaf")
    return flattened


def merge_task_records(records: Iterable[Mapping[str, Any]]):
    """Validate records and macro-average scalar metric leaves across tasks."""
    task_sums = defaultdict(lambda: defaultdict(float))
    task_counts = defaultdict(lambda: defaultdict(int))
    identities = set()
    num_records = 0

    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("records must be mappings")
        task = _identity_value(record, "task")
        if not isinstance(task, str):
            raise ValueError("record task identity must be a non-empty string")
        identity = (
            task,
            _identity_value(record, "record_index"),
            _identity_value(record, "pair_id"),
        )
        if identity in identities:
            raise ValueError(f"duplicate record identity: {identity}")
        identities.add(identity)

        for name, value in _record_metrics(record).items():
            task_sums[task][name] += value
            task_counts[task][name] += 1
        num_records += 1

    per_task = {
        task: {
            name: task_sums[task][name] / task_counts[task][name]
            for name in sorted(task_sums[task])
        }
        for task in sorted(task_sums)
    }
    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)
    for metrics in per_task.values():
        for name, value in metrics.items():
            metric_sums[name] += value
            metric_counts[name] += 1

    return {
        "num_records": num_records,
        "per_task": per_task,
        "metrics": {
            name: metric_sums[name] / metric_counts[name]
            for name in sorted(metric_sums)
        },
    }
