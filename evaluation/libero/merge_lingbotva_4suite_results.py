#!/usr/bin/env python3
"""Validate and merge a complete four-suite LIBERO closed-loop evaluation."""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from evaluation.libero.sampler_latency import (
        latency_p50_ms,
        load_sampler_latency_records,
    )
except ModuleNotFoundError:
    from sampler_latency import latency_p50_ms, load_sampler_latency_records


SUITES = ("libero_10", "libero_spatial", "libero_object", "libero_goal")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-video-steps", type=int, required=True)
    parser.add_argument("--expected-action-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def find_one(root: Path, suite: str, task_idx: int) -> Path:
    matches = list(root.rglob(f"{suite}_{task_idx}.json"))
    if not matches:
        raise ValueError(f"Missing result for {suite} task {task_idx}")
    if len(matches) > 1:
        raise ValueError(
            f"Duplicate results for {suite} task {task_idx}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def main():
    args = parse_args()
    if args.expected_episodes <= 0:
        raise ValueError("--expected-episodes must be positive")
    if args.expected_video_steps <= 0 or args.expected_action_steps <= 0:
        raise ValueError("expected sampler steps must be positive")

    suites = {}
    tasks = []
    total_successes = 0.0
    total_episodes = 0
    for suite in SUITES:
        suite_successes = 0.0
        suite_episodes = 0
        suite_task_rates = []
        for task_idx in range(10):
            path = find_one(args.input_root, suite, task_idx)
            data = json.loads(path.read_text())
            succ_num = float(data["succ_num"])
            episode_num = int(float(data["total_num"]))
            if episode_num != args.expected_episodes:
                raise ValueError(
                    f"Episode budget mismatch for {suite} task {task_idx}: "
                    f"expected {args.expected_episodes}, got {episode_num}"
                )
            if not 0 <= succ_num <= episode_num:
                raise ValueError(
                    f"Invalid successes for {suite} task {task_idx}: {succ_num}/{episode_num}"
                )
            rate = succ_num / episode_num
            tasks.append(
                {
                    "suite": suite,
                    "task_idx": task_idx,
                    "successes": succ_num,
                    "episodes": episode_num,
                    "success_rate": rate,
                    "source": str(path),
                }
            )
            suite_successes += succ_num
            suite_episodes += episode_num
            suite_task_rates.append(rate)
        suites[suite] = {
            "num_tasks": 10,
            "successes": suite_successes,
            "episodes": suite_episodes,
            "micro_success_rate": suite_successes / suite_episodes,
            "macro_task_success_rate": sum(suite_task_rates) / len(suite_task_rates),
        }
        total_successes += suite_successes
        total_episodes += suite_episodes

    latency_paths = sorted(args.input_root.rglob("sampler_latency.jsonl"))
    latency_records = load_sampler_latency_records(latency_paths)
    covered_episodes = set()
    for record in latency_records:
        if record["model"] != args.expected_model:
            raise ValueError(
                f"latency model mismatch: expected {args.expected_model}, "
                f"got {record['model']}"
            )
        if record["video_steps"] != args.expected_video_steps:
            raise ValueError(
                f"latency video_steps mismatch: expected "
                f"{args.expected_video_steps}, got {record['video_steps']}"
            )
        if record["action_steps"] != args.expected_action_steps:
            raise ValueError(
                f"latency action_steps mismatch: expected "
                f"{args.expected_action_steps}, got {record['action_steps']}"
            )
        if record["suite"] not in SUITES:
            raise ValueError(f"unexpected latency suite: {record['suite']}")
        if not 0 <= record["task_idx"] < 10:
            raise ValueError(f"unexpected latency task_idx: {record['task_idx']}")
        if not 0 <= record["episode_idx"] < args.expected_episodes:
            raise ValueError(
                f"unexpected latency episode_idx: {record['episode_idx']}"
            )
        covered_episodes.add(
            (record["suite"], record["task_idx"], record["episode_idx"])
        )
    expected_coverage = {
        (suite, task_idx, episode_idx)
        for suite in SUITES
        for task_idx in range(10)
        for episode_idx in range(args.expected_episodes)
    }
    if covered_episodes != expected_coverage:
        missing = sorted(expected_coverage - covered_episodes)
        unexpected = sorted(covered_episodes - expected_coverage)
        raise ValueError(
            "sampler latency episode coverage mismatch: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    summary = {
        "num_suites": len(SUITES),
        "num_tasks": len(tasks),
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "micro_success_rate": total_successes / total_episodes,
        "macro_task_success_rate": sum(task["success_rate"] for task in tasks) / len(tasks),
        "macro_suite_success_rate": sum(
            suite["macro_task_success_rate"] for suite in suites.values()
        )
        / len(suites),
        "latency": {
            "measurement_scope": "joint_sampler_call",
            "record_count": len(latency_records),
            "p50_ms": latency_p50_ms(latency_records),
            "source_files": [str(path) for path in latency_paths],
            "model": args.expected_model,
            "video_steps": args.expected_video_steps,
            "action_steps": args.expected_action_steps,
        },
        "suites": suites,
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.output)
    print(
        f"Merged {len(tasks)} tasks / {total_episodes} episodes: "
        f"success={summary['micro_success_rate']:.4f}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
