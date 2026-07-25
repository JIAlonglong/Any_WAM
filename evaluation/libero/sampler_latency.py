"""Auditable latency records for complete joint video/action sampler calls."""

from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path
from typing import Iterable, Mapping


_REQUIRED_FIELDS = (
    "model",
    "suite",
    "task_idx",
    "episode_idx",
    "video_steps",
    "action_steps",
    "call_index",
    "elapsed_ms",
)


def build_sampler_latency_record(
    *,
    model: str,
    suite: str,
    task_idx: int,
    episode_idx: int,
    video_steps: int,
    action_steps: int,
    call_index: int,
    elapsed_ms: float,
) -> dict:
    record = {
        "model": str(model),
        "suite": str(suite),
        "task_idx": int(task_idx),
        "episode_idx": int(episode_idx),
        "video_steps": int(video_steps),
        "action_steps": int(action_steps),
        "call_index": int(call_index),
        "elapsed_ms": float(elapsed_ms),
        "measurement_scope": "joint_sampler_call",
    }
    for field in ("model", "suite"):
        if not record[field]:
            raise ValueError(f"{field} must be non-empty")
    for field in ("task_idx", "episode_idx", "call_index"):
        if record[field] < 0:
            raise ValueError(f"{field} must be non-negative")
    for field in ("video_steps", "action_steps"):
        if record[field] <= 0:
            raise ValueError(f"{field} must be positive")
    if not math.isfinite(record["elapsed_ms"]) or record["elapsed_ms"] <= 0:
        raise ValueError("elapsed_ms must be finite and positive")
    return record


def append_sampler_latency_record(path: Path | str, record: Mapping) -> None:
    path = Path(path)
    validated = build_sampler_latency_record(
        **{field: record[field] for field in _REQUIRED_FIELDS}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(validated, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


def load_sampler_latency_records(paths: Iterable[Path | str]) -> list[dict]:
    records = []
    for source in paths:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"missing sampler latency JSONL: {path}")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(
                    build_sampler_latency_record(
                        **{field: payload[field] for field in _REQUIRED_FIELDS}
                    )
                )
            except Exception as exc:
                raise ValueError(f"{path.name}:{line_number}: {exc}") from exc
    return records


def latency_p50_ms(records: Iterable[Mapping]) -> float:
    values = [float(record["elapsed_ms"]) for record in records]
    if not values:
        raise ValueError("cannot compute latency p50 from zero records")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("latency values must be finite and positive")
    return float(statistics.median(values))
