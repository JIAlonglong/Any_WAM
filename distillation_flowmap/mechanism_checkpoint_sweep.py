"""Pure helpers for the LIBERO Figure 4 checkpoint sweep."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
_RECORD_FIELDS = (
    "step",
    "checkpoint",
    "g_anchor_l2",
    "g_anchor_mse",
    "g_comp_l2",
    "g_comp_mse",
    "e_student",
    "e_video",
    "e_joint",
    "delta_video",
    "r_video",
    "delta_joint",
    "g_residual",
    "valid_sample_count",
)
_METRIC_MAP = {
    "g_anchor_l2": "mechanism/g_anchor",
    "g_anchor_mse": "mechanism/g_anchor_mse",
    "g_comp_l2": "mechanism/g_comp",
    "g_comp_mse": "mechanism/g_comp_mse",
    "e_student": "mechanism/action_error_student_generated_history_context",
    "e_video": "mechanism/action_error_teacher_video_generated_history_context",
    "e_joint": "mechanism/action_error_teacher_joint_context",
}


@dataclass(frozen=True)
class CheckpointSpec:
    step: int
    root: Path
    transformer: Path


def _parse_step(path: Path) -> int:
    prefix = "step_"
    if not path.name.startswith(prefix):
        raise ValueError(f"invalid checkpoint directory name: {path.name}")
    suffix = path.name[len(prefix) :]
    if not suffix.isdigit():
        raise ValueError(f"invalid checkpoint step: {path.name}")
    return int(suffix)


def discover_target_checkpoints(run_root: Path) -> list[CheckpointSpec]:
    """Return valid target-student checkpoints sorted by real optimizer step."""
    run_root = Path(run_root)
    found: dict[int, CheckpointSpec] = {}
    for root in (run_root / "checkpoints").glob("step_*"):
        if not root.is_dir():
            continue
        step = _parse_step(root)
        transformer = root / "target_student" / "transformer"
        config_path = transformer / "config.json"
        weights_path = transformer / _WEIGHTS_NAME
        if not config_path.is_file():
            raise FileNotFoundError(f"missing checkpoint config: {config_path}")
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"missing checkpoint safetensors weights: {weights_path}"
            )
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        metadata_step = config.get("checkpoint_step")
        if metadata_step is not None and int(metadata_step) != step:
            raise ValueError(
                f"checkpoint_step={metadata_step} does not match directory "
                f"step={step}: {config_path}"
            )
        if step in found:
            raise ValueError(f"duplicate checkpoint step: {step}")
        found[step] = CheckpointSpec(
            step=step,
            root=root.resolve(),
            transformer=transformer.resolve(),
        )
    return [found[step] for step in sorted(found)]


def _finite_float(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric {name}: {result}")
    return result


def paper_record(
    *,
    step: int,
    checkpoint: Path,
    metrics: Mapping[str, float],
    valid_sample_count: int | float,
) -> dict:
    """Map internal diagnostic names to the stable paper output schema."""
    record = {
        "step": int(step),
        "checkpoint": str(Path(checkpoint)),
    }
    for paper_name, metric_name in _METRIC_MAP.items():
        if metric_name not in metrics:
            raise KeyError(f"missing metric {metric_name}")
        record[paper_name] = _finite_float(metrics[metric_name], paper_name)
    count = _finite_float(valid_sample_count, "valid_sample_count")
    if count <= 0 or not count.is_integer():
        raise ValueError(f"invalid valid_sample_count: {count}")
    record["valid_sample_count"] = int(count)
    record["delta_video"] = record["e_student"] - record["e_video"]
    record["r_video"] = max(0.0, record["delta_video"]) / max(
        record["e_student"], 1e-8
    )
    record["delta_joint"] = record["e_student"] - record["e_joint"]
    record["g_residual"] = record["e_video"] - record["e_joint"]
    for name in _RECORD_FIELDS:
        if name in ("step", "checkpoint"):
            continue
        _finite_float(record[name], name)
    return {name: record[name] for name in _RECORD_FIELDS}


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _normalized_records(records: Iterable[Mapping]) -> list[dict]:
    by_step: dict[int, dict] = {}
    for source in records:
        record = dict(source)
        step = int(record["step"])
        record["step"] = step
        by_step[step] = record
    return [by_step[step] for step in sorted(by_step)]


def write_records_atomic(records: Iterable[Mapping], output_dir: Path) -> None:
    """Write sorted, de-duplicated JSONL and CSV result files atomically."""
    output_dir = Path(output_dir)
    ordered = _normalized_records(records)
    jsonl = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in ordered
    )
    _atomic_text(output_dir / "mechanism_metrics.jsonl", jsonl)

    csv_path = output_dir / "mechanism_metrics.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_RECORD_FIELDS))
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(csv_path)


def _maybe_log_axis(axis, values) -> None:
    positive = [float(value) for value in values if float(value) > 0]
    if positive and max(positive) / min(positive) >= 100:
        axis.set_yscale("log")


def _style_axis(axis, *, ylabel: str) -> None:
    axis.set_xlabel("Successful optimizer steps")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _draw_panel_a(axis, records) -> None:
    steps = [row["step"] for row in records]
    anchor = [row["g_anchor_l2"] for row in records]
    comp = [row["g_comp_l2"] for row in records]
    axis.plot(
        steps,
        anchor,
        color="#D55E00",
        linestyle="-",
        marker="o",
        linewidth=1.5,
        markersize=3.5,
        label=r"$G_{\mathrm{anchor}}$",
    )
    axis.plot(
        steps,
        comp,
        color="#0072B2",
        linestyle="--",
        marker="s",
        linewidth=1.5,
        markersize=3.5,
        label=r"$G_{\mathrm{comp}}$",
    )
    _style_axis(axis, ylabel="Squared latent distance")
    _maybe_log_axis(axis, anchor + comp)
    axis.legend(frameon=False, fontsize=7)


def _draw_panel_b(axis, records) -> None:
    steps = [row["step"] for row in records]
    series = (
        ("e_student", r"$E_{\mathrm{student}}$", "#0072B2", "-", "o"),
        ("e_video", r"$E_{\mathrm{video}}$", "#D55E00", "--", "s"),
        ("e_joint", r"$E_{\mathrm{joint}}$", "#333333", "-.", "^"),
    )
    all_values = []
    for key, label, color, linestyle, marker in series:
        values = [row[key] for row in records]
        all_values.extend(values)
        axis.plot(
            steps,
            values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.5,
            markersize=3.5,
            label=label,
        )
    _style_axis(axis, ylabel="Masked action MSE")
    _maybe_log_axis(axis, all_values)
    axis.legend(frameon=False, fontsize=7)


def plot_figure4(records: Iterable[Mapping], output_dir: Path) -> dict[str, Path]:
    """Export the raw two-panel Figure 4 as PNG/PDF plus panel PNGs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = _normalized_records(records)
    if not ordered:
        raise ValueError("cannot plot an empty checkpoint sweep")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.35), constrained_layout=True)
    _draw_panel_a(axes[0], ordered)
    _draw_panel_b(axes[1], ordered)
    axes[0].text(-0.16, 1.03, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.16, 1.03, "(b)", transform=axes[1].transAxes, fontweight="bold")
    combined_png = output_dir / "figure4_mechanism_diagnostics.png"
    combined_pdf = output_dir / "figure4_mechanism_diagnostics.pdf"
    figure.savefig(combined_png, dpi=300, bbox_inches="tight")
    figure.savefig(combined_pdf, bbox_inches="tight")
    plt.close(figure)

    panel_a = output_dir / "figure4_panel_a.png"
    figure_a, axis_a = plt.subplots(figsize=(3.25, 2.35), constrained_layout=True)
    _draw_panel_a(axis_a, ordered)
    figure_a.savefig(panel_a, dpi=300, bbox_inches="tight")
    plt.close(figure_a)

    panel_b = output_dir / "figure4_panel_b.png"
    figure_b, axis_b = plt.subplots(figsize=(3.25, 2.35), constrained_layout=True)
    _draw_panel_b(axis_b, ordered)
    figure_b.savefig(panel_b, dpi=300, bbox_inches="tight")
    plt.close(figure_b)
    return {
        "combined_png": combined_png,
        "combined_pdf": combined_pdf,
        "panel_a": panel_a,
        "panel_b": panel_b,
    }
