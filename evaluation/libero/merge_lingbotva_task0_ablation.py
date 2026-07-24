#!/usr/bin/env python3
"""Strictly merge the 5-model x 3-budget LIBERO task-0 ablation family."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


MODELS = ("stage1", "stage1_only", "anchor_only", "field_only", "apm")
BUDGETS = (1, 2, 4)
BENCHMARK = "libero_10"
TASK_IDX = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def find_one(root: Path, model: str, steps: int) -> Path:
    budget_root = root / model / f"steps_{steps}"
    matches = list(budget_root.rglob(f"{BENCHMARK}_{TASK_IDX}.json"))
    if not matches:
        raise ValueError(f"Missing result for {model} at {steps} steps")
    if len(matches) != 1:
        raise ValueError(
            f"Duplicate results for {model} at {steps} steps: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def read_row(path: Path, model: str, steps: int, expected_episodes: int) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    successes = float(data["succ_num"])
    episodes = int(float(data["total_num"]))
    if episodes != expected_episodes:
        raise ValueError(
            f"Episode budget mismatch for {model} at {steps} steps: "
            f"expected {expected_episodes}, got {episodes}"
        )
    if not 0 <= successes <= episodes:
        raise ValueError(
            f"Invalid successes for {model} at {steps} steps: "
            f"{successes}/{episodes}"
        )
    return {
        "model": model,
        "video_steps": steps,
        "action_steps": steps,
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "source": str(path),
    }


def main() -> None:
    args = parse_args()
    if args.expected_episodes <= 0:
        raise ValueError("--expected-episodes must be positive")

    rows = [
        read_row(
            find_one(args.input_root, model, steps),
            model,
            steps,
            args.expected_episodes,
        )
        for model in MODELS
        for steps in BUDGETS
    ]
    stage1_rates = {
        row["video_steps"]: row["success_rate"]
        for row in rows
        if row["model"] == "stage1"
    }
    models = {}
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        step_table = {}
        for row in model_rows:
            delta = row["success_rate"] - stage1_rates[row["video_steps"]]
            row["delta_vs_stage1"] = delta
            step_table[str(row["video_steps"])] = {
                "successes": row["successes"],
                "episodes": row["episodes"],
                "success_rate": row["success_rate"],
                "delta_vs_stage1": delta,
                "source": row["source"],
            }
        models[model] = {
            "steps": step_table,
            "mean_success_rate": sum(row["success_rate"] for row in model_rows)
            / len(model_rows),
        }

    nonbaseline = [model for model in MODELS if model != "stage1"]
    best_model = max(nonbaseline, key=lambda name: models[name]["mean_success_rate"])
    summary = {
        "benchmark": BENCHMARK,
        "task_idx": TASK_IDX,
        "expected_episodes_per_job": args.expected_episodes,
        "num_jobs": len(rows),
        "budgets": list(BUDGETS),
        "models": models,
        "best_ablation_model": best_model,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(
        f"Merged {len(rows)} task-0 jobs; "
        f"best ablation={best_model} "
        f"mean_success={models[best_model]['mean_success_rate']:.4f}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
