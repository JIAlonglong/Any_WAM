#!/usr/bin/env python3
"""Summarize RobotWin StepWAM ablation manifests and metric files."""

import argparse
import csv
import json
from pathlib import Path


SUMMARY_COLUMNS = [
    "variant",
    "seed",
    "task_count",
    "episodes_per_task",
    "SR_easy",
    "SR_hard",
    "SR_all",
    "teacher_retention",
    "endpoint_error",
    "same_state_velocity_error",
    "rollout_drift",
    "FVD_or_proxy",
    "LPIPS_or_proxy",
    "temporal_error",
    "latency_chunk_ms",
    "Hz",
    "NFE",
    "speedup",
    "stage1_ckpt",
    "stage2_ckpt",
]


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return read_json(path)


def row_from_manifest(manifest_path):
    manifest = read_json(manifest_path)
    run_dir = Path(manifest["run_dir"])
    task_splits = manifest.get("task_list", {}).get("splits", {})
    task_count = sum(len(tasks) for tasks in task_splits.values())
    episodes = manifest.get("task_list", {}).get("episodes_per_task", "")
    offline = maybe_read_json(run_dir / "metrics" / "offline_rollout.json")
    video = maybe_read_json(run_dir / "metrics" / "video_mse.json")
    sr = maybe_read_json(run_dir / "metrics" / "robotwin_sr.json")
    return {
        "variant": manifest.get("variant", ""),
        "seed": manifest.get("seed", ""),
        "task_count": task_count,
        "episodes_per_task": episodes,
        "SR_easy": sr.get("SR_easy", ""),
        "SR_hard": sr.get("SR_hard", ""),
        "SR_all": sr.get("SR_all", ""),
        "teacher_retention": sr.get("teacher_retention", ""),
        "endpoint_error": offline.get("endpoint_error", offline.get("endpoint_mse", "")),
        "same_state_velocity_error": offline.get(
            "same_state_velocity_error", offline.get("same_state_velocity_mse", "")
        ),
        "rollout_drift": offline.get("rollout_drift", ""),
        "FVD_or_proxy": video.get("FVD", video.get("fvd_proxy", "")),
        "LPIPS_or_proxy": video.get("LPIPS", video.get("lpips_proxy", "")),
        "temporal_error": video.get("temporal_error", ""),
        "latency_chunk_ms": sr.get("latency_chunk_ms", ""),
        "Hz": sr.get("Hz", ""),
        "NFE": sr.get("NFE", ""),
        "speedup": sr.get("speedup", ""),
        "stage1_ckpt": manifest.get("stage1_ckpt", ""),
        "stage2_ckpt": manifest.get("stage2_ckpt", ""),
    }


def write_csv(rows, out_path):
    with Path(out_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_COLUMNS})


def write_markdown(rows, out_path):
    with Path(out_path).open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(SUMMARY_COLUMNS) + " |\n")
        f.write("| " + " | ".join("---" for _ in SUMMARY_COLUMNS) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(key, "")) for key in SUMMARY_COLUMNS) + " |\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifests = sorted(args.root.glob("*/seed_*/run_manifest.json"))
    rows = [row_from_manifest(path) for path in manifests]
    write_csv(rows, args.out / "ablation_table.csv")
    write_markdown(rows, args.out / "ablation_table.md")
    with (args.out / "ablation_table.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
