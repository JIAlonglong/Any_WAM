import csv
import json
import math
from pathlib import Path

import pytest

from distillation_flowmap.mechanism_checkpoint_sweep import (
    discover_target_checkpoints,
    paper_record,
    plot_figure4,
    write_records_atomic,
)


def _checkpoint(run_root: Path, step: int, *, metadata_step=None, weights=True):
    transformer = (
        run_root
        / "checkpoints"
        / f"step_{step}"
        / "target_student"
        / "transformer"
    )
    transformer.mkdir(parents=True)
    config = {
        "checkpoint_step": step if metadata_step is None else metadata_step,
        "action_downsample_factor": 1,
        "action_grid_protocol": "continuous_action_v1",
    }
    (transformer / "config.json").write_text(json.dumps(config))
    if weights:
        (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    return transformer


def _metrics(anchor=25.0, comp=4.0, student=9.0, video=4.0, joint=1.0):
    return {
        "mechanism/g_anchor": anchor,
        "mechanism/g_anchor_mse": anchor / 5,
        "mechanism/g_comp": comp,
        "mechanism/g_comp_mse": comp / 5,
        "mechanism/action_error_student_context": 90.0,
        "mechanism/action_error_teacher_video_context": 40.0,
        "mechanism/action_error_student_generated_history_context": student,
        "mechanism/action_error_teacher_video_generated_history_context": video,
        "mechanism/action_error_teacher_joint_context": joint,
    }


def test_checkpoint_discovery_uses_real_numeric_step_order(tmp_path):
    _checkpoint(tmp_path, 10000)
    _checkpoint(tmp_path, 2000)
    _checkpoint(tmp_path, 1000)

    found = discover_target_checkpoints(tmp_path)

    assert [item.step for item in found] == [1000, 2000, 10000]
    assert all(item.transformer.name == "transformer" for item in found)


def test_checkpoint_discovery_rejects_directory_metadata_mismatch(tmp_path):
    _checkpoint(tmp_path, 1000, metadata_step=900)

    with pytest.raises(ValueError, match="checkpoint_step"):
        discover_target_checkpoints(tmp_path)


def test_checkpoint_discovery_rejects_missing_weights(tmp_path):
    _checkpoint(tmp_path, 1000, weights=False)

    with pytest.raises(FileNotFoundError, match="safetensors"):
        discover_target_checkpoints(tmp_path)


def test_paper_record_uses_signed_and_clamped_action_derivations(tmp_path):
    record = paper_record(
        step=1000,
        checkpoint=tmp_path / "step_1000",
        metrics=_metrics(),
        valid_sample_count=1,
    )

    assert record["g_anchor_l2"] == 25.0
    assert record["g_anchor_mse"] == 5.0
    assert record["g_comp_l2"] == 4.0
    assert record["g_comp_mse"] == 0.8
    assert record["e_student"] == 9.0
    assert record["e_video"] == 4.0
    assert record["e_joint"] == 1.0
    assert record["delta_video"] == 5.0
    assert record["r_video"] == pytest.approx(5.0 / 9.0)
    assert record["delta_joint"] == 8.0
    assert record["g_residual"] == 3.0
    assert record["valid_sample_count"] == 1

    negative = paper_record(
        step=2000,
        checkpoint=tmp_path / "step_2000",
        metrics=_metrics(student=2.0, video=3.0),
        valid_sample_count=1,
    )
    assert negative["delta_video"] == -1.0
    assert negative["r_video"] == 0.0


def test_paper_record_uses_deployment_generated_action_history(tmp_path):
    metrics = _metrics(student=7.0, video=2.0, joint=1.0)

    record = paper_record(
        step=1000,
        checkpoint=tmp_path / "step_1000",
        metrics=metrics,
        valid_sample_count=1,
    )

    assert record["e_student"] == 7.0
    assert record["e_video"] == 2.0
    assert record["delta_video"] == 5.0
    assert record["r_video"] == pytest.approx(5.0 / 7.0)
    assert record["g_residual"] == 1.0


def test_paper_record_rejects_nonfinite_metric(tmp_path):
    bad = _metrics()
    bad["mechanism/g_comp"] = math.nan

    with pytest.raises(ValueError, match="g_comp"):
        paper_record(
            step=1000,
            checkpoint=tmp_path / "step_1000",
            metrics=bad,
            valid_sample_count=1,
        )


def test_atomic_outputs_sort_and_deduplicate_checkpoint_rows(tmp_path):
    records = [
        paper_record(
            step=2000,
            checkpoint=tmp_path / "step_2000",
            metrics=_metrics(anchor=20.0),
            valid_sample_count=1,
        ),
        paper_record(
            step=1000,
            checkpoint=tmp_path / "step_1000",
            metrics=_metrics(anchor=30.0),
            valid_sample_count=1,
        ),
        paper_record(
            step=2000,
            checkpoint=tmp_path / "step_2000",
            metrics=_metrics(anchor=19.0),
            valid_sample_count=1,
        ),
    ]

    write_records_atomic(records, tmp_path)

    jsonl_rows = [
        json.loads(line)
        for line in (tmp_path / "mechanism_metrics.jsonl").read_text().splitlines()
    ]
    with (tmp_path / "mechanism_metrics.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert [row["step"] for row in jsonl_rows] == [1000, 2000]
    assert jsonl_rows[1]["g_anchor_l2"] == 19.0
    assert [int(row["step"]) for row in csv_rows] == [1000, 2000]


def test_plot_exports_combined_vector_and_individual_panels(tmp_path):
    records = [
        paper_record(
            step=1000,
            checkpoint=tmp_path / "step_1000",
            metrics=_metrics(anchor=100.0, comp=10.0),
            valid_sample_count=1,
        ),
        paper_record(
            step=2000,
            checkpoint=tmp_path / "step_2000",
            metrics=_metrics(anchor=50.0, comp=5.0),
            valid_sample_count=1,
        ),
    ]
    original = json.loads(json.dumps(records))

    outputs = plot_figure4(records, tmp_path)

    assert records == original
    assert outputs["combined_png"].name == "figure4_mechanism_diagnostics.png"
    assert outputs["combined_pdf"].name == "figure4_mechanism_diagnostics.pdf"
    assert outputs["panel_a"].name == "figure4_panel_a.png"
    assert outputs["panel_b"].name == "figure4_panel_b.png"
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
