import json

import pytest

from distillation_flowmap.ablation.summarize_final_danceopd_ablation import (
    MAIN_VARIANTS,
    PRIMARY_METRICS,
    require_complete_main,
    build_final_summary,
    write_final_report,
)


PAIR = "rollout_eval/t1000_r0_i0"


def _metric_key(step, suffix):
    return f"{PAIR}/s{step}_t{step}/{suffix}"


def _write_metrics(root, variant, per_task):
    run_dir = root / variant / "seed_0"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"variant": variant, "seed": 0, "run_dir": str(run_dir)}),
        encoding="utf-8",
    )
    (metrics_dir / "offline_rollout.json").write_text(
        json.dumps({"per_task": per_task}),
        encoding="utf-8",
    )


def test_final_summary_uses_paired_task_macro_deltas(tmp_path):
    baseline_tasks = {
        "task_a": {
            _metric_key(4, "video_teacher_x_mse"): 1.0,
            _metric_key(4, "action_gt_xr_mse"): 0.4,
        },
        "task_b": {
            _metric_key(4, "video_teacher_x_mse"): 3.0,
            _metric_key(4, "action_gt_xr_mse"): 0.8,
        },
    }
    full_tasks = {
        "task_a": {
            _metric_key(4, "video_teacher_x_mse"): 0.5,
            _metric_key(4, "action_gt_xr_mse"): 0.3,
        },
        "task_b": {
            _metric_key(4, "video_teacher_x_mse"): 2.5,
            _metric_key(4, "action_gt_xr_mse"): 0.7,
        },
    }
    _write_metrics(tmp_path, "final_w_o_opd", baseline_tasks)
    _write_metrics(tmp_path, "final_stepwam_danceopd", full_tasks)

    summary = build_final_summary(
        tmp_path,
        bootstrap_samples=200,
        expected_task_count=None,
    )
    metric = summary["main"]["final_stepwam_danceopd"]["primary"][
        "latent_endpoint_mse_to_teacher"
    ]

    assert metric["n_tasks"] == 2
    assert metric["macro_mean"] == pytest.approx(1.5)
    assert metric["baseline_macro_mean"] == pytest.approx(2.0)
    assert metric["macro_delta"] == pytest.approx(-0.5)
    assert metric["bootstrap_unit"] == "tasks"
    assert metric["bootstrap_ci_low"] <= -0.5 <= metric["bootstrap_ci_high"]


def test_final_summary_writes_separate_main_and_control_tables(tmp_path):
    per_task = {
        "task_a": {_metric_key(4, "video_teacher_x_mse"): 1.0},
        "task_b": {_metric_key(4, "video_teacher_x_mse"): 2.0},
    }
    _write_metrics(tmp_path, "final_w_o_opd", per_task)
    _write_metrics(tmp_path, "final_stepwam_danceopd", per_task)

    summary = build_final_summary(
        tmp_path,
        bootstrap_samples=20,
        expected_task_count=None,
    )
    output_dir = tmp_path / "summary"
    write_final_report(summary, output_dir)

    assert (output_dir / "main_primary_long.csv").exists()
    assert (output_dir / "structural_controls_long.csv").exists()
    report = (output_dir / "final_report.md").read_text(encoding="utf-8")
    assert "task coverage, not training-seed significance" in report



def test_strict_main_completion_ignores_absent_structural_controls(tmp_path):
    primary = {
        _metric_key(4, suffix): 1.0
        for suffix in PRIMARY_METRICS.values()
    }
    for variant in MAIN_VARIANTS:
        _write_metrics(tmp_path, variant, {"task_a": dict(primary)})

    summary = build_final_summary(
        tmp_path,
        bootstrap_samples=20,
        expected_task_count=None,
    )

    assert set(summary["missing_variants"]) == {
        "final_local_adjacent_only",
        "final_action_only",
    }
    require_complete_main(summary)


def test_strict_main_completion_rejects_missing_primary_metric(tmp_path):
    primary = {
        _metric_key(4, suffix): 1.0
        for suffix in PRIMARY_METRICS.values()
    }
    incomplete = dict(primary)
    incomplete.pop(_metric_key(4, "decoded_video_gt_lpips"))
    for variant in MAIN_VARIANTS:
        payload = incomplete if variant == "final_stepwam_danceopd" else primary
        _write_metrics(tmp_path, variant, {"task_a": dict(payload)})

    summary = build_final_summary(
        tmp_path,
        bootstrap_samples=20,
        expected_task_count=None,
    )

    with pytest.raises(ValueError, match="final_stepwam_danceopd: decoded_lpips_to_gt"):
        require_complete_main(summary)
