import json
from pathlib import Path

import pytest

from evaluation.libero.sampler_latency import (
    append_sampler_latency_record,
    build_sampler_latency_record,
    latency_p50_ms,
    load_sampler_latency_records,
)


def _record(**overrides):
    values = {
        "model": "stage2",
        "suite": "libero_10",
        "task_idx": 3,
        "episode_idx": 7,
        "video_steps": 2,
        "action_steps": 2,
        "call_index": 4,
        "elapsed_ms": 12.5,
    }
    values.update(overrides)
    return build_sampler_latency_record(**values)


def test_append_round_trips_auditable_sampler_records(tmp_path):
    output = tmp_path / "sampler_latency.jsonl"
    append_sampler_latency_record(output, _record(elapsed_ms=10.0))
    append_sampler_latency_record(output, _record(call_index=5, elapsed_ms=30.0))

    records = load_sampler_latency_records([output])

    assert [record["elapsed_ms"] for record in records] == [10.0, 30.0]
    assert all(record["measurement_scope"] == "joint_sampler_call" for record in records)
    assert latency_p50_ms(records) == 20.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", ""),
        ("suite", ""),
        ("task_idx", -1),
        ("episode_idx", -1),
        ("video_steps", 0),
        ("action_steps", 0),
        ("call_index", -1),
        ("elapsed_ms", 0.0),
    ],
)
def test_sampler_latency_rejects_invalid_provenance(field, value):
    with pytest.raises(ValueError, match=field):
        _record(**{field: value})


def test_loader_rejects_malformed_jsonl(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(_record()) + "\nnot-json\n")

    with pytest.raises(ValueError, match="bad.jsonl:2"):
        load_sampler_latency_records([path])
