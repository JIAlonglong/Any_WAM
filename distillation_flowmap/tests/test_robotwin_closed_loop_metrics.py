import pytest

from evaluation.robotwin.closed_loop_metrics import (
    build_closed_loop_summary,
    build_closed_loop_task_metrics,
)


def test_task_metrics_report_policy_latency_hz_and_nfe():
    metrics = build_closed_loop_task_metrics(
        task="place_a2b_right",
        success_count=3,
        episode_count=5,
        chunk_latencies_s=[0.5, 1.0, 1.5],
        action_steps=24,
        cache_update_seconds=0.75,
        nfe=4,
    )

    assert metrics["task"] == "place_a2b_right"
    assert metrics["success_rate"] == pytest.approx(0.6)
    assert metrics["policy_chunks"] == 3
    assert metrics["action_steps"] == 24
    assert metrics["policy_inference_seconds"] == pytest.approx(3.0)
    assert metrics["cache_update_seconds"] == pytest.approx(0.75)
    assert metrics["latency_per_chunk_seconds"] == pytest.approx(1.0)
    assert metrics["latency_per_chunk_p50_seconds"] == pytest.approx(1.0)
    assert metrics["policy_hz"] == pytest.approx(8.0)
    assert metrics["nfe"] == 4

from evaluation.robotwin.closed_loop_metrics import build_closed_loop_summary


def test_summary_uses_task_macro_success_and_weighted_throughput():
    first = build_closed_loop_task_metrics(
        task="easy",
        success_count=5,
        episode_count=5,
        chunk_latencies_s=[1.0, 1.0],
        action_steps=20,
        cache_update_seconds=0.5,
        nfe=4,
    )
    second = build_closed_loop_task_metrics(
        task="hard",
        success_count=0,
        episode_count=5,
        chunk_latencies_s=[2.0],
        action_steps=10,
        cache_update_seconds=0.25,
        nfe=4,
    )

    summary = build_closed_loop_summary([first, second], expected_task_count=2)

    assert summary["task_count"] == 2
    assert summary["success_rate_macro"] == pytest.approx(0.5)
    assert summary["success_rate_micro"] == pytest.approx(0.5)
    assert summary["policy_chunks"] == 3
    assert summary["action_steps"] == 30
    assert summary["policy_inference_seconds"] == pytest.approx(4.0)
    assert summary["cache_update_seconds"] == pytest.approx(0.75)
    assert summary["latency_per_chunk_seconds"] == pytest.approx(4.0 / 3.0)
    assert summary["policy_hz"] == pytest.approx(7.5)
    assert summary["nfe"] == 4
