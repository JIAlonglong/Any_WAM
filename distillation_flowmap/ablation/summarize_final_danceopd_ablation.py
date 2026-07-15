#!/usr/bin/env python3
"""Summarize the frozen final RobotWin LingBot-VA DanceOPD ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


MAIN_VARIANTS = (
    "final_w_o_opd",
    "final_endpoint_only_danceopd",
    "final_danceopd_velocity_only",
    "final_stepwam_danceopd",
)
STRUCTURAL_VARIANTS = (
    "final_local_adjacent_only",
    "final_action_only",
)
CLEAN_PAIR_ID = "t1000_r0_i0"
PRIMARY_STEP = 4

PRIMARY_METRICS = {
    "latent_endpoint_mse_to_gt": "video_gt_x_mse",
    "latent_endpoint_l1_to_gt": "video_gt_x_l1",
    "latent_endpoint_mse_to_teacher": "video_teacher_x_mse",
    "latent_endpoint_l1_to_teacher": "video_teacher_x_l1",
    "same_state_velocity_mse": "video_same_state_teacher_v_mse",
    "same_state_velocity_l1": "video_same_state_teacher_v_l1",
    "rollout_drift_mse": "video_rollout_drift_mse",
    "rollout_drift_l1": "video_rollout_drift_l1",
    "action_endpoint_mse": "action_gt_xr_mse",
    "action_endpoint_l1": "action_gt_xr_l1",
    "decoded_lpips_to_gt": "decoded_video_gt_lpips",
    "decoded_pixel_mse_to_gt": "decoded_video_gt_pixel_mse",
    "decoded_pixel_l1_to_gt": "decoded_video_gt_pixel_l1",
    "decoded_psnr_to_gt": "decoded_video_gt_psnr",
    "decoded_ssim_to_gt": "decoded_video_gt_ssim",
    "decoded_temporal_difference_mse_to_gt": "decoded_video_gt_temporal_difference_mse",
}
ACTION_ONLY_METRICS = {
    "action_endpoint_mse": "action_gt_xr_mse",
    "action_endpoint_l1": "action_gt_xr_l1",
    "action_velocity_mse": "action_gt_v_mse",
    "action_velocity_l1": "action_gt_v_l1",
}
TEACHER_REFERENCE_METRICS = {
    "teacher_latent_endpoint_mse_to_gt": "video_teacher_gt_x_mse",
    "teacher_latent_endpoint_l1_to_gt": "video_teacher_gt_x_l1",
    "teacher_decoded_lpips_to_gt": "decoded_video_teacher_gt_lpips",
    "teacher_decoded_pixel_mse_to_gt": "decoded_video_teacher_gt_pixel_mse",
    "teacher_decoded_psnr_to_gt": "decoded_video_teacher_gt_psnr",
    "teacher_decoded_ssim_to_gt": "decoded_video_teacher_gt_ssim",
    "teacher_decoded_temporal_difference_mse_to_gt": (
        "decoded_video_teacher_gt_temporal_difference_mse"
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _flatten_numeric(payload: Any, prefix: str = "") -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    flattened: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_numeric(value, name))
        elif _is_number(value):
            flattened[name] = float(value)
    return flattened


def _equal_nfe_key(step: int, suffix: str) -> str:
    return f"rollout_eval/{CLEAN_PAIR_ID}/s{int(step)}_t{int(step)}/{suffix}"


def _teacher_reference_key(teacher_step: int, suffix: str) -> str:
    student_step = teacher_step if teacher_step in (1, 2, 4) else PRIMARY_STEP
    return f"rollout_eval/{CLEAN_PAIR_ID}/s{student_step}_t{teacher_step}/{suffix}"


def _result_path(root: Path, variant: str) -> Path:
    name = "offline_rollout_action_only.json" if variant == "final_action_only" else "offline_rollout.json"
    return root / variant / "seed_0" / "metrics" / name


def _load_variant_tasks(root: Path, variant: str) -> dict[str, dict[str, float]] | None:
    path = _result_path(root, variant)
    if not path.exists():
        return None
    payload = _read_json(path)
    raw_tasks = payload.get("per_task", {})
    if not isinstance(raw_tasks, dict):
        raise ValueError(f"per_task must be a mapping: {path}")
    return {
        str(task): _flatten_numeric(values)
        for task, values in raw_tasks.items()
        if isinstance(values, dict)
    }


def _bootstrap_interval(
    deltas: list[float],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not deltas:
        raise ValueError("Cannot bootstrap an empty delta list")
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    generator = random.Random(seed)
    count = len(deltas)
    means = []
    for _ in range(samples):
        means.append(
            sum(deltas[generator.randrange(count)] for _ in range(count)) / count
        )
    means.sort()
    lower = means[int(0.025 * (samples - 1))]
    upper = means[int(0.975 * (samples - 1))]
    return float(lower), float(upper)


def _metric_summary(
    values: dict[str, float],
    baseline_values: dict[str, float],
    *,
    expected_task_count: int | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    tasks = sorted(set(values).intersection(baseline_values))
    if not tasks:
        return None, []
    if expected_task_count is not None and len(tasks) != expected_task_count:
        raise ValueError(
            f"Expected {expected_task_count} paired tasks, got {len(tasks)}: {tasks}"
        )

    variant_values = [float(values[task]) for task in tasks]
    baseline = [float(baseline_values[task]) for task in tasks]
    deltas = [value - baseline_value for value, baseline_value in zip(variant_values, baseline)]
    lower, upper = _bootstrap_interval(deltas, bootstrap_samples, bootstrap_seed)
    details = [
        {
            "task": task,
            "value": value,
            "baseline_value": baseline_value,
            "delta_vs_baseline": delta,
        }
        for task, value, baseline_value, delta in zip(tasks, variant_values, baseline, deltas)
    ]
    return (
        {
            "n_tasks": len(tasks),
            "tasks": tasks,
            "macro_mean": float(statistics.mean(variant_values)),
            "baseline_macro_mean": float(statistics.mean(baseline)),
            "macro_delta": float(statistics.mean(deltas)),
            "bootstrap_ci_low": lower,
            "bootstrap_ci_high": upper,
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_unit": "tasks",
        },
        details,
    )


def _task_metric_values(
    task_payload: dict[str, dict[str, float]],
    metric_key: str,
) -> dict[str, float]:
    return {
        task: float(metrics[metric_key])
        for task, metrics in task_payload.items()
        if metric_key in metrics and _is_number(metrics[metric_key])
    }


def _summarize_variant_metrics(
    variant: str,
    variant_tasks: dict[str, dict[str, float]],
    baseline_tasks: dict[str, dict[str, float]],
    metric_definitions: dict[str, str],
    *,
    step: int,
    expected_task_count: int | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
    section: str,
    include_delta: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[str]]:
    metrics: dict[str, dict[str, Any]] = {}
    per_task_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for index, (metric_name, suffix) in enumerate(metric_definitions.items()):
        metric_key = _equal_nfe_key(step, suffix)
        values = _task_metric_values(variant_tasks, metric_key)
        baseline_values = _task_metric_values(baseline_tasks, metric_key)
        if not values:
            missing.append(metric_name)
            continue
        if include_delta:
            summary, details = _metric_summary(
                values,
                baseline_values,
                expected_task_count=expected_task_count,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + index,
            )
            if summary is None:
                missing.append(metric_name)
                continue
            metrics[metric_name] = summary
            for detail in details:
                per_task_rows.append(
                    {
                        "section": section,
                        "variant": variant,
                        "step": int(step),
                        "metric": metric_name,
                        **detail,
                    }
                )
        else:
            task_names = sorted(values)
            if expected_task_count is not None and len(task_names) != expected_task_count:
                raise ValueError(
                    f"Expected {expected_task_count} control tasks, got {len(task_names)}"
                )
            metrics[metric_name] = {
                "n_tasks": len(task_names),
                "tasks": task_names,
                "macro_mean": float(statistics.mean(values[task] for task in task_names)),
                "bootstrap_unit": "not_reported_for_structural_control",
            }
    return metrics, per_task_rows, missing


def _teacher_reference(
    baseline_tasks: dict[str, dict[str, float]],
    *,
    expected_task_count: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for teacher_step in (1, 2, 4, 8):
        for metric_name, suffix in TEACHER_REFERENCE_METRICS.items():
            values = _task_metric_values(
                baseline_tasks,
                _teacher_reference_key(teacher_step, suffix),
            )
            if not values:
                continue
            task_names = sorted(values)
            if expected_task_count is not None and len(task_names) != expected_task_count:
                raise ValueError(
                    f"Expected {expected_task_count} teacher-reference tasks, got {len(task_names)}"
                )
            rows.append(
                {
                    "teacher_steps": teacher_step,
                    "metric": metric_name,
                    "n_tasks": len(task_names),
                    "macro_mean": float(statistics.mean(values[task] for task in task_names)),
                    "role": "equal_nfe" if teacher_step in (1, 2, 4) else "high_quality_reference",
                }
            )
    return rows


def build_final_summary(
    root: str | Path,
    *,
    baseline_variant: str = "final_w_o_opd",
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 0,
    expected_task_count: int | None = 12,
) -> dict[str, Any]:
    root = Path(root)
    if baseline_variant not in MAIN_VARIANTS:
        raise ValueError(f"Baseline must be a main variant, got {baseline_variant!r}")

    loaded = {
        variant: _load_variant_tasks(root, variant)
        for variant in (*MAIN_VARIANTS, *STRUCTURAL_VARIANTS)
    }
    baseline_tasks = loaded.get(baseline_variant)
    if baseline_tasks is None:
        raise FileNotFoundError(
            f"Missing baseline held-out metrics: {_result_path(root, baseline_variant)}"
        )

    summary: dict[str, Any] = {
        "schema": "robotwin_final_danceopd_summary_v1",
        "root": str(root),
        "baseline_variant": baseline_variant,
        "bootstrap_unit": "tasks",
        "bootstrap_interpretation": "task coverage, not training-seed significance",
        "expected_task_count": expected_task_count,
        "main": {},
        "curves": {},
        "structural_controls": {},
        "teacher_reference": _teacher_reference(
            baseline_tasks,
            expected_task_count=expected_task_count,
        ),
        "per_task_deltas": [],
        "missing_variants": [],
        "missing_main_metrics": {},
    }

    for variant in MAIN_VARIANTS:
        variant_tasks = loaded.get(variant)
        if variant_tasks is None:
            summary["missing_variants"].append(variant)
            continue
        primary, details, missing = _summarize_variant_metrics(
            variant,
            variant_tasks,
            baseline_tasks,
            PRIMARY_METRICS,
            step=PRIMARY_STEP,
            expected_task_count=expected_task_count,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            section="main_primary",
            include_delta=True,
        )
        summary["main"][variant] = {"primary": primary}
        summary["per_task_deltas"].extend(details)
        summary["missing_main_metrics"][variant] = missing

        curve_rows: dict[str, dict[str, dict[str, Any]]] = {}
        for step in (1, 2, 4):
            curve_metrics, curve_details, _ = _summarize_variant_metrics(
                variant,
                variant_tasks,
                baseline_tasks,
                PRIMARY_METRICS,
                step=step,
                expected_task_count=expected_task_count,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + step * 100,
                section="main_curve",
                include_delta=True,
            )
            curve_rows[f"k{step}"] = curve_metrics
            summary["per_task_deltas"].extend(curve_details)
        summary["curves"][variant] = curve_rows

    for variant in STRUCTURAL_VARIANTS:
        variant_tasks = loaded.get(variant)
        if variant_tasks is None:
            summary["missing_variants"].append(variant)
            continue
        metric_definitions = (
            ACTION_ONLY_METRICS if variant == "final_action_only" else PRIMARY_METRICS
        )
        control_metrics, _, missing = _summarize_variant_metrics(
            variant,
            variant_tasks,
            baseline_tasks,
            metric_definitions,
            step=PRIMARY_STEP,
            expected_task_count=expected_task_count,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            section="structural_control",
            include_delta=False,
        )
        summary["structural_controls"][variant] = {
            "primary": control_metrics,
            "missing_metrics": missing,
            "comparison_scope": "structural_control_not_strict_opd",
        }

    return summary

def require_complete_main(summary: dict[str, Any]) -> None:
    missing = [
        str(variant)
        for variant in summary.get("missing_variants", ())
        if variant in MAIN_VARIANTS
    ]
    missing_metrics = summary.get("missing_main_metrics", {})
    if isinstance(missing_metrics, dict):
        for variant in MAIN_VARIANTS:
            metrics = missing_metrics.get(variant, ())
            if metrics:
                missing.append(f"{variant}: {', '.join(map(str, metrics))}")
    if missing:
        raise ValueError("Incomplete strict main report: " + "; ".join(missing))



def _flatten_summary_rows(
    summary: dict[str, Any],
    section: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if section == "main_primary":
        for variant, payload in summary["main"].items():
            for metric, values in payload["primary"].items():
                rows.append(
                    {
                        "section": section,
                        "variant": variant,
                        "student_steps": PRIMARY_STEP,
                        "metric": metric,
                        **values,
                    }
                )
    elif section == "main_curves":
        for variant, curve_payload in summary["curves"].items():
            for label, metrics in curve_payload.items():
                for metric, values in metrics.items():
                    rows.append(
                        {
                            "section": section,
                            "variant": variant,
                            "student_steps": int(label[1:]),
                            "metric": metric,
                            **values,
                        }
                    )
    elif section == "structural_controls":
        for variant, payload in summary["structural_controls"].items():
            for metric, values in payload["primary"].items():
                rows.append(
                    {
                        "section": section,
                        "variant": variant,
                        "student_steps": PRIMARY_STEP,
                        "metric": metric,
                        "comparison_scope": payload["comparison_scope"],
                        **values,
                    }
                )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    selected = (
        "latent_endpoint_mse_to_teacher",
        "same_state_velocity_mse",
        "rollout_drift_mse",
        "action_endpoint_mse",
        "decoded_lpips_to_gt",
    )
    lines = [
        "# Final RobotWin DanceOPD Ablation",
        "",
        "The bootstrap confidence intervals use paired task resampling and describe "
        "task coverage, not training-seed significance.",
        "",
        "## Main K=4 Table",
        "",
        "| Variant | Metric | Macro mean | Delta vs baseline | 95% paired task CI | Tasks |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for variant, payload in summary["main"].items():
        for metric in selected:
            values = payload["primary"].get(metric)
            if values is None:
                continue
            lines.append(
                "| {variant} | {metric} | {mean:.6g} | {delta:.6g} | "
                "[{low:.6g}, {high:.6g}] | {count} |".format(
                    variant=variant,
                    metric=metric,
                    mean=values["macro_mean"],
                    delta=values["macro_delta"],
                    low=values["bootstrap_ci_low"],
                    high=values["bootstrap_ci_high"],
                    count=values["n_tasks"],
                )
            )
    if summary["missing_variants"]:
        lines.extend(
            [
                "",
                "## Missing Inputs",
                "",
                ", ".join(summary["missing_variants"]),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(summary: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_rows = _flatten_summary_rows(summary, "main_primary")
    curve_rows = _flatten_summary_rows(summary, "main_curves")
    control_rows = _flatten_summary_rows(summary, "structural_controls")
    common_fields = [
        "section",
        "variant",
        "student_steps",
        "metric",
        "n_tasks",
        "macro_mean",
        "baseline_macro_mean",
        "macro_delta",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "bootstrap_samples",
        "bootstrap_unit",
        "comparison_scope",
    ]
    _write_csv(main_rows, output_dir / "main_primary_long.csv", common_fields)
    _write_csv(curve_rows, output_dir / "main_curves_long.csv", common_fields)
    _write_csv(control_rows, output_dir / "structural_controls_long.csv", common_fields)
    _write_csv(
        summary["per_task_deltas"],
        output_dir / "per_task_deltas.csv",
        [
            "section",
            "variant",
            "step",
            "metric",
            "task",
            "value",
            "baseline_value",
            "delta_vs_baseline",
        ],
    )
    _write_csv(
        summary["teacher_reference"],
        output_dir / "teacher_reference.csv",
        ["teacher_steps", "metric", "n_tasks", "macro_mean", "role"],
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _write_markdown(summary, output_dir / "final_report.md")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--provenance", default=None)
    parser.add_argument("--baseline-variant", default="final_w_o_opd")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--expected-task-count", type=int, default=12)
    parser.add_argument(
        "--require-complete-main",
        action="store_true",
        help="Fail if a strict main variant or any primary metric is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    expected = None if args.expected_task_count <= 0 else args.expected_task_count
    summary = build_final_summary(
        args.root,
        baseline_variant=args.baseline_variant,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        expected_task_count=expected,
    )
    if args.provenance is not None:
        provenance_path = Path(args.provenance)
        if not provenance_path.exists():
            raise FileNotFoundError(f"Missing evaluation provenance: {provenance_path}")
        summary["provenance"] = _read_json(provenance_path)
    if args.require_complete_main:
        require_complete_main(summary)
    write_final_report(summary, args.out)
    print(json.dumps({
        "out": str(args.out),
        "missing_variants": summary["missing_variants"],
        "missing_main_metrics": summary["missing_main_metrics"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
