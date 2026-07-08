#!/usr/bin/env python3
"""Summarize protocol-controlled RobotWin StepWAM mini-ablation runs."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PER_SEED_BASE_COLUMNS = [
    "variant",
    "seed",
    "git_hash",
    "task_preset",
    "task_count",
    "tasks",
    "episodes_per_task",
    "protocol_seed",
    "train_manifest_path",
    "heldout_eval_manifest_path",
    "eval_pairs_path",
    "stage1_ckpt",
    "stage2_ckpt",
    "run_dir",
]

SUMMARY_BASE_COLUMNS = [
    "variant",
    "metric",
    "n",
    "mean",
    "std",
    "min",
    "max",
    "seeds",
    "baseline_variant",
    "baseline_mean",
]

CORE_METRIC_HINTS = (
    "video_teacher_x_mse",
    "video_teacher_v_mse",
    "action_gt_xr_mse",
    "action_gt_v_mse",
    "video_gt_x_mse",
    "video_gt_v_mse",
    "SR_all",
    "teacher_retention",
    "latency",
    "Hz",
    "NFE",
    "speedup",
)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return read_json(path)


def is_number(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def flatten_numeric(payload, prefix=""):
    flat = {}
    if not isinstance(payload, dict):
        return flat
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_numeric(value, name))
        elif is_number(value):
            flat[name] = float(value)
    return flat


def task_names_from_manifest(manifest):
    selected = manifest.get("selected_task_filter") or []
    if selected:
        return [str(task) for task in selected]
    splits = manifest.get("task_list", {}).get("splits", {})
    tasks = []
    for split_tasks in splits.values():
        tasks.extend(str(task) for task in split_tasks)
    return tasks


def metric_sources(run_dir):
    metrics_dir = Path(run_dir) / "metrics"
    video_path = metrics_dir / "video_mse.json"
    if not video_path.exists() and (metrics_dir / "video_mse_with_videos.json").exists():
        video_path = metrics_dir / "video_mse_with_videos.json"
    return [
        ("heldout/offline_rollout", metrics_dir / "offline_rollout.json"),
        ("heldout/video_mse", video_path),
        ("train/offline_rollout", metrics_dir / "train_rollout.json"),
        ("closed_loop", metrics_dir / "robotwin_sr.json"),
    ]


def read_run_metrics(run_dir):
    metrics = {}
    for prefix, path in metric_sources(run_dir):
        payload = maybe_read_json(path)
        for key, value in flatten_numeric(payload).items():
            metrics[f"{prefix}/{key}"] = value

    train_prefix = "train/offline_rollout/"
    heldout_prefix = "heldout/offline_rollout/"
    for key, value in list(metrics.items()):
        if not key.startswith(train_prefix):
            continue
        suffix = key[len(train_prefix):]
        heldout_key = heldout_prefix + suffix
        if heldout_key in metrics:
            metrics[f"gap/train_minus_heldout/{suffix}"] = value - metrics[heldout_key]
    return metrics


def row_from_manifest(manifest_path):
    manifest = read_json(manifest_path)
    run_dir = Path(manifest["run_dir"])
    tasks = task_names_from_manifest(manifest)
    row = {
        "variant": manifest.get("variant", ""),
        "seed": manifest.get("seed", ""),
        "git_hash": manifest.get("git_hash", ""),
        "task_preset": manifest.get("task_preset", ""),
        "task_count": len(tasks),
        "tasks": ",".join(tasks),
        "episodes_per_task": manifest.get("task_list", {}).get("episodes_per_task", ""),
        "protocol_seed": manifest.get("protocol_seed", ""),
        "train_manifest_path": manifest.get("train_manifest_path", ""),
        "heldout_eval_manifest_path": manifest.get("heldout_eval_manifest_path", ""),
        "eval_pairs_path": manifest.get("eval_pairs_path", ""),
        "stage1_ckpt": manifest.get("stage1_ckpt", ""),
        "stage2_ckpt": manifest.get("stage2_ckpt", ""),
        "run_dir": str(run_dir),
    }
    row.update(read_run_metrics(run_dir))
    return row


def collect_rows(root):
    manifests = sorted(Path(root).glob("*/seed_*/run_manifest.json"))
    return [row_from_manifest(path) for path in manifests]


def numeric_metric_names(rows):
    names = set()
    for row in rows:
        for key, value in row.items():
            if key in PER_SEED_BASE_COLUMNS:
                continue
            if is_number(value):
                names.add(key)
    return sorted(names)


def write_csv(rows, out_path, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row)
        fieldnames = sorted(keys)
    with Path(out_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_jsonl(rows, out_path):
    with Path(out_path).open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, sort_keys=True)
            f.write("\n")


def mean_or_blank(values):
    if not values:
        return ""
    return statistics.mean(values)


def summarize_rows(rows, baseline_variant):
    metric_names = numeric_metric_names(rows)
    grouped = {}
    seeds = {}
    for row in rows:
        variant = str(row.get("variant", ""))
        seed = str(row.get("seed", ""))
        for metric in metric_names:
            value = row.get(metric)
            if not is_number(value):
                continue
            grouped.setdefault((variant, metric), []).append(float(value))
            seeds.setdefault((variant, metric), []).append(seed)

    baseline_means = {
        metric: statistics.mean(values)
        for (variant, metric), values in grouped.items()
        if variant == baseline_variant and values
    }
    delta_column = f"delta_vs_{baseline_variant}"
    summary = []
    for (variant, metric), values in sorted(grouped.items()):
        mean_value = statistics.mean(values)
        baseline_mean = baseline_means.get(metric, "")
        row = {
            "variant": variant,
            "metric": metric,
            "n": len(values),
            "mean": mean_value,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "seeds": ",".join(seeds.get((variant, metric), [])),
            "baseline_variant": baseline_variant,
            "baseline_mean": baseline_mean,
            delta_column: mean_value - baseline_mean if baseline_mean != "" else "",
        }
        summary.append(row)
    return summary, delta_column


def task_rows(rows):
    out = []
    metric_names = numeric_metric_names(rows)
    for row in rows:
        tasks = [task for task in str(row.get("tasks", "")).split(",") if task]
        for task in tasks:
            task_row = {
                "variant": row.get("variant", ""),
                "seed": row.get("seed", ""),
                "task": task,
                "run_dir": row.get("run_dir", ""),
            }
            for metric in metric_names:
                if metric in row:
                    task_row[metric] = row[metric]
            out.append(task_row)
    return out


def video_asset_rows(rows):
    assets = []
    for row in rows:
        video_dir = Path(str(row.get("run_dir", ""))) / "videos"
        if not video_dir.exists():
            continue
        for path in sorted(video_dir.glob("*")):
            if path.suffix.lower() not in (".mp4", ".png", ".jpg", ".jpeg"):
                continue
            if path.suffix.lower() == ".mp4":
                kind = "mp4"
            elif "contact_sheet" in path.name:
                kind = "contact_sheet"
            else:
                kind = path.suffix.lower().lstrip(".")
            assets.append({
                "variant": row.get("variant", ""),
                "seed": row.get("seed", ""),
                "kind": kind,
                "path": str(path),
                "bytes": path.stat().st_size,
            })
    return assets


def write_markdown_table(rows, fieldnames, out_path, limit=None):
    rows = rows[:limit] if limit is not None else rows
    with Path(out_path).open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join("---" for _ in fieldnames) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(key, "")) for key in fieldnames) + " |\n")


def is_core_metric(metric):
    return any(hint in metric for hint in CORE_METRIC_HINTS)


def write_report(summary_rows, asset_rows, out_path, delta_column):
    core_rows = [row for row in summary_rows if is_core_metric(str(row.get("metric", "")))]
    columns = ["variant", "metric", "n", "mean", "std", "baseline_mean", delta_column]
    with Path(out_path).open("w", encoding="utf-8") as f:
        f.write("# RobotWin StepWAM Mini-Ablation Report\n\n")
        f.write("Trend only: this report is for protocol sanity and early signal checking, not final ablation conclusions.\n\n")
        f.write(f"- Summary rows: {len(summary_rows)}\n")
        f.write(f"- Video/contact-sheet assets: {len(asset_rows)}\n")
        f.write("- Delta is raw variant mean minus baseline mean; lower-is-better metrics should be interpreted accordingly.\n\n")
        f.write("## Core Metrics\n\n")
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in core_rows[:80]:
            f.write("| " + " | ".join(str(row.get(key, "")) for key in columns) + " |\n")
        f.write("\n## Assets\n\n")
        for asset in asset_rows[:40]:
            f.write(f"- {asset['kind']}: {asset['path']}\n")


def summarize(root, out, baseline_variant="w_o_opd"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(root)
    metric_names = numeric_metric_names(rows)
    per_seed_columns = PER_SEED_BASE_COLUMNS + metric_names
    summary, delta_column = summarize_rows(rows, baseline_variant)
    summary_columns = SUMMARY_BASE_COLUMNS + [delta_column]
    per_task = task_rows(rows)
    per_task_columns = ["variant", "seed", "task", "run_dir"] + metric_names
    assets = video_asset_rows(rows)

    write_csv(rows, out / "mini_ablation_per_seed.csv", per_seed_columns)
    write_jsonl(rows, out / "mini_ablation_per_seed.jsonl")
    write_csv(summary, out / "mini_ablation_summary.csv", summary_columns)
    write_markdown_table(summary, summary_columns, out / "mini_ablation_summary.md")
    write_csv(per_task, out / "mini_ablation_per_task.csv", per_task_columns)
    write_jsonl(assets, out / "mini_ablation_video_assets.jsonl")
    write_report(summary, assets, out / "mini_ablation_report.md", delta_column)

    # Backward-compatible aliases used by earlier notes/scripts.
    write_csv(rows, out / "ablation_table.csv", per_seed_columns)
    write_markdown_table(rows, per_seed_columns, out / "ablation_table.md")
    with (out / "ablation_table.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")
    return {
        "rows": rows,
        "summary": summary,
        "per_task": per_task,
        "assets": assets,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=None, help="Kept for backward compatibility; unused.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-variant", default="w_o_opd")
    return parser.parse_args()


def main():
    args = parse_args()
    result = summarize(args.root, args.out, baseline_variant=args.baseline_variant)
    print(
        "Wrote "
        f"{len(result['rows'])} per-seed rows, "
        f"{len(result['summary'])} summary rows, "
        f"{len(result['assets'])} assets to {args.out}"
    )


if __name__ == "__main__":
    main()
