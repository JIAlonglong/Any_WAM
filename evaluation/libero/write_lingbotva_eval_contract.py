#!/usr/bin/env python3
"""Write an honest manifest for the LingBot-VA/LIBERO paper evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


SUITES = ("libero_10", "libero_spatial", "libero_object", "libero_goal")
BUDGETS = ((1, 1), (2, 2), (4, 4))


def build_eval_contract(
    *,
    teacher: Path,
    stage1_only: Path,
    stage2: Path,
    naive: Path | None,
    episodes_per_task: int,
) -> dict:
    if int(episodes_per_task) <= 0:
        raise ValueError("episodes_per_task must be positive")
    stage1_only = Path(stage1_only)
    if naive is not None and Path(naive).resolve() == stage1_only.resolve():
        raise ValueError(
            "Stage-I-only is distilled and cannot be labelled naive composition"
        )
    models = {
        "teacher": {
            "status": "evaluated",
            "checkpoint": str(Path(teacher)),
        },
        "stage1_only": {
            "status": "evaluated",
            "checkpoint": str(stage1_only),
            "distilled": True,
            "stage2_correction": False,
        },
        "stage2": {
            "status": "evaluated",
            "checkpoint": str(Path(stage2)),
            "distilled": True,
            "stage2_correction": True,
        },
    }
    if naive is None:
        models["naive_composition"] = {
            "status": "missing",
            "distilled": False,
            "reason": "no explicit no-distillation naive checkpoint supplied",
        }
    else:
        models["naive_composition"] = {
            "status": "evaluated",
            "checkpoint": str(Path(naive)),
            "distilled": False,
        }
    return {
        "benchmark": "LIBERO",
        "suites": list(SUITES),
        "matched_video_action_budgets": [list(pair) for pair in BUDGETS],
        "episodes_per_task": int(episodes_per_task),
        "models": models,
        "excluded": {
            "default_teacher_20_50": True,
            "reason": "explicitly outside this evaluation run",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--stage1-only", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--naive", type=Path)
    parser.add_argument("--episodes-per-task", type=int, required=True)
    args = parser.parse_args()
    payload = build_eval_contract(
        teacher=args.teacher,
        stage1_only=args.stage1_only,
        stage2=args.stage2,
        naive=args.naive,
        episodes_per_task=args.episodes_per_task,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
