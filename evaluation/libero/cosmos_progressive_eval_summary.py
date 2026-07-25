"""Strict aggregation for the matched-budget Cosmos LIBERO evaluation matrix."""

from __future__ import annotations

import csv
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


LIBERO_SUITES = (
    "libero_10",
    "libero_spatial",
    "libero_object",
    "libero_goal",
)
STUDENT_STEPS = (1, 2, 4)


def _fail(message: str) -> None:
    raise SystemExit(message)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _parse_shard_plan(value: str) -> dict[int, range]:
    plan: dict[int, range] = {}
    assigned: list[int] = []
    for entry in value.split(";"):
        try:
            shard_text, start_text, end_text = entry.split(":")
            shard, start, end = int(shard_text), int(start_text), int(end_text)
        except (TypeError, ValueError):
            _fail(f"invalid shard plan: {value!r}")
        if shard in plan or start < 0 or end < start:
            _fail(f"invalid shard plan: {value!r}")
        plan[shard] = range(start, end)
        assigned.extend(range(start, end))
    if sorted(assigned) != list(range(10)):
        _fail(f"invalid shard plan coverage: {value!r}")
    return plan


def _require_suite(value: str) -> str:
    if value not in LIBERO_SUITES:
        _fail(f"unsupported LIBERO suite {value!r}; expected one of {LIBERO_SUITES}")
    return value


def merge_formal_records(
    *,
    root: str | Path,
    checkpoint: str | Path,
    suite: str,
    seed_count: int,
    requested_steps: int,
    shard_plan: str,
    evaluation_classification: str,
    is_formal: bool,
    model_role: str,
) -> dict[str, Any]:
    """Validate one suite/K cell and publish its JSON and CSV summaries."""

    root = Path(root)
    expected_checkpoint = str(Path(checkpoint).resolve())
    suite = _require_suite(suite)
    seed_count = int(seed_count)
    requested_steps = int(requested_steps)
    if seed_count <= 0:
        _fail("seed_count must be positive")
    if requested_steps not in STUDENT_STEPS:
        _fail(f"requested_steps must be one of {STUDENT_STEPS}")
    plan = _parse_shard_plan(shard_plan)
    expected = {(suite, task, episode) for task in range(10) for episode in range(seed_count)}
    seen: dict[tuple[str, int, int], str] = {}
    per_task: dict[int, list[float]] = defaultdict(list)
    contract_identities: set[str] = set()
    cosmos_repo_commits: set[str] = set()
    cosmos_source_digests: set[str] = set()

    pattern = "shard_*/seed_*/records/task_*_episode_*.json"
    for record_path in sorted(root.glob(pattern)):
        relative = record_path.relative_to(root).parts
        shard_match = next(
            (re.fullmatch(r"shard_(\d+)", part) for part in relative if re.fullmatch(r"shard_(\d+)", part)),
            None,
        )
        seed_match = next(
            (re.fullmatch(r"seed_(\d+)", part) for part in relative if re.fullmatch(r"seed_(\d+)", part)),
            None,
        )
        if shard_match is None or seed_match is None:
            _fail(f"seed mismatch: record path lacks shard/seed identity: {record_path}")
        shard, path_seed = int(shard_match.group(1)), int(seed_match.group(1))
        if shard not in plan or path_seed not in range(seed_count):
            _fail(f"seed mismatch: unexpected shard={shard} seed={path_seed} in {record_path}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record_suite = record.get("libero_benchmark")
        if record_suite != suite:
            _fail(
                f"suite mismatch: expected={suite!r} got={record_suite!r} in {record_path}"
            )
        task = int(record.get("task_idx", -1))
        if task not in plan[shard]:
            _fail(f"task mismatch: task {task} is not assigned to shard {shard}")
        episode = int(record.get("episode_idx", -1))
        if episode != path_seed:
            _fail(f"episode mismatch: expected episode_idx={path_seed} got={episode}")
        try:
            payload_seed = int(record["seed"])
        except KeyError:
            _fail(f"seed mismatch: missing record seed for suite={suite} task={task}")
        except (TypeError, ValueError):
            _fail(f"seed mismatch: invalid record seed={record.get('seed')!r}")
        if payload_seed != path_seed:
            _fail(f"seed mismatch: record seed={payload_seed} path seed={path_seed}")
        if model_role == "official_teacher":
            if "student_steps" in record:
                _fail("official_teacher record must not claim student_steps")
            if (
                int(record.get("requested_video_steps", -1)) != requested_steps
                or int(record.get("requested_action_steps", -1)) != requested_steps
                or int(record.get("effective_video_steps", -1)) != requested_steps
                or int(record.get("effective_action_steps", -1)) != requested_steps
                or int(record.get("observed_joint_nfe", -1)) != requested_steps
                or record.get("matched_budget_verified") is not True
            ):
                _fail("official_teacher effective matched-budget proof is missing")
            repo_commit = record.get("cosmos_repo_commit")
            source_digest = record.get("cosmos_source_sha256")
            if repo_commit != "1eb8457072b4a1adfe1f83c3076e4aa5452cbab2":
                _fail("official_teacher audited Cosmos repository commit is missing")
            if (
                not isinstance(source_digest, str)
                or len(source_digest) != 64
                or any(character not in "0123456789abcdef" for character in source_digest)
            ):
                _fail("official_teacher audited Cosmos source identity is missing")
            cosmos_repo_commits.add(repo_commit)
            cosmos_source_digests.add(source_digest)
        elif int(record.get("student_steps", -1)) != requested_steps:
            _fail(
                f"step mismatch: expected={requested_steps} got={record.get('student_steps')!r}"
            )
        if record.get("model_role") != model_role:
            _fail(
                f"model_role mismatch: expected={model_role!r} got={record.get('model_role')!r}"
            )
        if (
            int(record.get("video_steps", -1)) != requested_steps
            or int(record.get("action_steps", -1)) != requested_steps
        ):
            _fail("video/action steps mismatch in formal record")
        identity = record.get("checkpoint_contract_identity")
        if not isinstance(identity, str) or not identity:
            _fail("checkpoint_contract_identity missing in formal record")
        contract_identities.add(identity)
        if record.get("s4_checkpoint") != expected_checkpoint:
            _fail(
                f"checkpoint mismatch: suite={suite} task={task} episode={episode} "
                f"expected={expected_checkpoint!r} got={record.get('s4_checkpoint')!r}"
            )
        key = (suite, task, episode)
        if key in seen:
            _fail(f"duplicate record: {key}: {seen[key]} and {record_path}")
        seen[key] = str(record_path)
        success = record.get("success")
        if type(success) is not bool:
            _fail("success must be a plain boolean in every formal record")
        per_task[task].append(1.0 if success else 0.0)

    unexpected = set(seen) - expected
    if unexpected:
        _fail(f"unexpected suite/task/episode records: {sorted(unexpected)[:5]}")
    missing = expected - set(seen)
    if missing:
        _fail(f"missing record: {len(missing)} required records absent, first={sorted(missing)[:5]}")
    if len(contract_identities) != 1:
        _fail(
            "checkpoint_contract_identity mismatch across formal records: "
            f"{sorted(contract_identities)!r}"
        )
    if model_role == "official_teacher" and (
        len(cosmos_repo_commits) != 1 or len(cosmos_source_digests) != 1
    ):
        _fail("official_teacher Cosmos source identity mismatch across records")

    task_means = {
        f"{suite}:{task}": sum(per_task[task]) / seed_count for task in range(10)
    }
    macro_success = sum(task_means.values()) / len(task_means)
    rng = random.Random(0)
    bootstrap = []
    for _ in range(10000):
        sampled_means = [
            sum(rng.choice(per_task[task]) for _ in range(seed_count)) / seed_count
            for task in range(10)
        ]
        bootstrap.append(sum(sampled_means) / len(sampled_means))
    bootstrap.sort()
    summary = {
        "schema": "cosmos_progressive_s4_formal_eval_v2",
        "checkpoint": expected_checkpoint,
        "libero_benchmark": suite,
        "model_role": model_role,
        "video_steps": requested_steps,
        "action_steps": requested_steps,
        "checkpoint_contract_identity": next(iter(contract_identities)),
        "evaluation_classification": evaluation_classification,
        "is_formal": bool(is_formal),
        "num_tasks": 10,
        "num_records": len(seen),
        "seeds_per_task": seed_count,
        "per_task_success": task_means,
        "macro_success": macro_success,
        "bootstrap_ci_95": [
            bootstrap[int(0.025 * len(bootstrap))],
            bootstrap[int(0.975 * len(bootstrap))],
        ],
    }
    if model_role != "official_teacher":
        summary["student_steps"] = requested_steps
    else:
        summary["cosmos_repo_commit"] = next(iter(cosmos_repo_commits))
        summary["cosmos_source_sha256"] = next(iter(cosmos_source_digests))
    _atomic_json(root / "formal_summary.json", summary)
    _atomic_csv(
        root / "formal_summary.csv",
        [
            "libero_benchmark",
            "task_idx",
            "model_role",
            "video_steps",
            "action_steps",
            "episodes",
            "success_rate",
        ],
        [
            {
                "libero_benchmark": suite,
                "task_idx": task,
                "model_role": model_role,
                "video_steps": requested_steps,
                "action_steps": requested_steps,
                "episodes": seed_count,
                "success_rate": task_means[f"{suite}:{task}"],
            }
            for task in range(10)
        ],
    )
    return summary


def merge_student_matrix(
    *,
    root: str | Path,
    checkpoint: str | Path,
    evaluation_classification: str,
    is_formal: bool,
    model_role: str,
    episodes_per_task: int,
) -> dict[str, Any]:
    """Validate all 4 suites × matched K={1,2,4} and publish matrix summaries."""

    root = Path(root)
    expected_checkpoint = str(Path(checkpoint).resolve())
    episodes_per_task = int(episodes_per_task)
    summaries: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    unique_tasks: set[str] = set()
    task_budget_cells: set[tuple[str, int]] = set()
    cosmos_repo_commits: set[str] = set()
    cosmos_source_digests: set[str] = set()

    for step in STUDENT_STEPS:
        step_tasks: set[str] = set()
        for suite in LIBERO_SUITES:
            path = root / f"k{step}" / suite / "formal_summary.json"
            if not path.is_file():
                _fail(f"missing child summary for {suite} K={step}: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("libero_benchmark") != suite:
                _fail(f"suite mismatch in child summary for {suite} K={step}")
            if payload.get("model_role") != model_role:
                _fail(f"model_role mismatch for {suite} K={step}")
            if (
                int(payload.get("video_steps", -1)) != step
                or int(payload.get("action_steps", -1)) != step
            ):
                _fail(f"video/action step mismatch for {suite} K={step}")
            if model_role == "official_teacher":
                if "student_steps" in payload:
                    _fail(f"official_teacher child must not claim student_steps for {suite} K={step}")
                repo_commit = payload.get("cosmos_repo_commit")
                source_digest = payload.get("cosmos_source_sha256")
                if repo_commit != "1eb8457072b4a1adfe1f83c3076e4aa5452cbab2":
                    _fail(f"official_teacher Cosmos repo commit mismatch for {suite} K={step}")
                if (
                    not isinstance(source_digest, str)
                    or len(source_digest) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in source_digest
                    )
                ):
                    _fail(f"official_teacher Cosmos source identity missing for {suite} K={step}")
                cosmos_repo_commits.add(repo_commit)
                cosmos_source_digests.add(source_digest)
            elif int(payload.get("student_steps", -1)) != step:
                _fail(f"video/action step mismatch for {suite} K={step}")
            if int(payload.get("num_tasks", -1)) != 10:
                _fail(f"task count mismatch for {suite} K={step}")
            if int(payload.get("num_records", -1)) != 10 * episodes_per_task:
                _fail(f"num_records mismatch for {suite} K={step}")
            if int(payload.get("seeds_per_task", -1)) != episodes_per_task:
                _fail(f"seeds_per_task mismatch for {suite} K={step}")
            reported_checkpoint = payload.get("checkpoint")
            if not isinstance(reported_checkpoint, str) or str(
                Path(reported_checkpoint).resolve()
            ) != expected_checkpoint:
                _fail(f"checkpoint mismatch for {suite} K={step}")
            if payload.get("evaluation_classification") != evaluation_classification:
                _fail(f"evaluation classification mismatch for {suite} K={step}")
            if payload.get("is_formal") is not bool(is_formal):
                _fail(f"is_formal mismatch for {suite} K={step}")
            identity = payload.get("checkpoint_contract_identity")
            if not isinstance(identity, str) or not identity:
                _fail(f"checkpoint_contract_identity missing for {suite} K={step}")
            identities.add(identity)
            expected_task_keys = {f"{suite}:{task}" for task in range(10)}
            actual_task_keys = set(payload.get("per_task_success", {}))
            if actual_task_keys != expected_task_keys:
                _fail(f"task identity mismatch for {suite} K={step}")
            if step_tasks & actual_task_keys:
                _fail(f"duplicate task identity for {suite} K={step}")
            step_tasks.update(actual_task_keys)
            unique_tasks.update(actual_task_keys)
            task_budget_cells.update((task, step) for task in actual_task_keys)
            key = f"k{step}/{suite}"
            summaries[key] = str(path)
            rows.append(
                {
                    "libero_benchmark": suite,
                    "model_role": model_role,
                    "video_steps": step,
                    "action_steps": step,
                    "tasks": 10,
                    "episodes_per_task": episodes_per_task,
                    "num_records": 10 * episodes_per_task,
                    "macro_success": payload.get("macro_success", ""),
                    "summary_json": str(path),
                }
            )
        if len(step_tasks) != 40:
            _fail(f"expected 40 unique tasks for K={step}, got {len(step_tasks)}")

    if len(unique_tasks) != 40 or len(task_budget_cells) != 120:
        _fail(
            f"matrix uniqueness mismatch: tasks={len(unique_tasks)} cells={len(task_budget_cells)}"
        )
    if len(identities) != 1:
        _fail(f"checkpoint_contract_identity mismatch across matrix: {sorted(identities)!r}")

    summary = {
        "schema": "cosmos_progressive_joint_124_matrix_v2",
        "checkpoint": expected_checkpoint,
        "evaluation_classification": evaluation_classification,
        "is_formal": bool(is_formal),
        "suites": list(LIBERO_SUITES),
        "steps": list(STUDENT_STEPS),
        "episodes_per_task": episodes_per_task,
        "episodes_per_k": 40 * episodes_per_task,
        "model_role": model_role,
        "checkpoint_contract_identity": next(iter(identities)),
        "unique_tasks": len(unique_tasks),
        "task_budget_cells": len(task_budget_cells),
        "summaries": summaries,
    }
    if model_role == "official_teacher":
        if len(cosmos_repo_commits) != 1 or len(cosmos_source_digests) != 1:
            _fail("official_teacher Cosmos source identity mismatch across matrix")
        summary["cosmos_repo_commit"] = next(iter(cosmos_repo_commits))
        summary["cosmos_source_sha256"] = next(iter(cosmos_source_digests))
    _atomic_json(root / "matrix_summary.json", summary)
    _atomic_csv(
        root / "matrix_summary.csv",
        [
            "libero_benchmark",
            "model_role",
            "video_steps",
            "action_steps",
            "tasks",
            "episodes_per_task",
            "num_records",
            "macro_success",
            "summary_json",
        ],
        rows,
    )
    return summary


def merge_complete_matrix(
    *,
    root: str | Path,
    checkpoints: dict[str, str | Path],
    evaluation_classification: str,
    is_formal: bool,
    episodes_per_task: int,
) -> dict[str, Any]:
    """Validate the independent Student and official-Teacher 40-task matrices."""

    root = Path(root)
    roles = ("stage2_target", "official_teacher")
    if set(checkpoints) != set(roles):
        _fail(f"complete matrix checkpoints must be exactly {roles}")
    expected_child_paths = {
        (root / role / f"k{step}" / suite / "formal_summary.json").resolve()
        for role in roles
        for step in STUDENT_STEPS
        for suite in LIBERO_SUITES
    }
    actual_child_paths = {
        path.resolve() for path in root.rglob("formal_summary.json")
    }
    unexpected_paths = actual_child_paths - expected_child_paths
    if unexpected_paths:
        _fail(
            "unexpected duplicate/foreign formal summary: "
            f"{sorted(map(str, unexpected_paths))[:3]}"
        )
    missing_paths = expected_child_paths - actual_child_paths
    if missing_paths:
        _fail(
            "missing role/suite/K formal summary: "
            f"{sorted(map(str, missing_paths))[:3]}"
        )
    rows: list[dict[str, Any]] = []
    summaries: dict[str, str] = {}
    official_teacher_source: dict[str, str] = {}
    for role in roles:
        role_root = root / role
        role_summary = merge_student_matrix(
            root=role_root,
            checkpoint=checkpoints[role],
            evaluation_classification=evaluation_classification,
            is_formal=is_formal,
            model_role=role,
            episodes_per_task=episodes_per_task,
        )
        if role == "official_teacher":
            official_teacher_source = {
                "official_teacher_cosmos_repo_commit": role_summary[
                    "cosmos_repo_commit"
                ],
                "official_teacher_cosmos_source_sha256": role_summary[
                    "cosmos_source_sha256"
                ],
            }
        for cell, path in role_summary["summaries"].items():
            key = f"{role}/{cell}"
            if key in summaries:
                _fail(f"duplicate role/suite/K summary: {key}")
            summaries[key] = path
            child = json.loads(Path(path).read_text(encoding="utf-8"))
            rows.append(
                {
                    "model_role": role,
                    "libero_benchmark": child["libero_benchmark"],
                    "video_steps": child["video_steps"],
                    "action_steps": child["action_steps"],
                    "tasks": child["num_tasks"],
                    "episodes_per_task": child["seeds_per_task"],
                    "num_records": child["num_records"],
                    "macro_success": child.get("macro_success", ""),
                    "summary_json": path,
                }
            )
    if len(summaries) != 24:
        _fail(f"complete matrix requires 24 role/suite/K summaries, got {len(summaries)}")
    summary = {
        "schema": "cosmos_progressive_student_teacher_124_matrix_v1",
        "roles": list(roles),
        "suites": list(LIBERO_SUITES),
        "steps": list(STUDENT_STEPS),
        "episodes_per_task": int(episodes_per_task),
        "role_task_budget_cells": 2 * 40 * 3,
        "checkpoints": {
            role: str(Path(checkpoints[role]).resolve()) for role in roles
        },
        "summaries": summaries,
        "evaluation_classification": evaluation_classification,
        "is_formal": bool(is_formal),
        **official_teacher_source,
    }
    _atomic_json(root / "matrix_summary.json", summary)
    _atomic_csv(
        root / "matrix_summary.csv",
        [
            "model_role",
            "libero_benchmark",
            "video_steps",
            "action_steps",
            "tasks",
            "episodes_per_task",
            "num_records",
            "macro_success",
            "summary_json",
        ],
        rows,
    )
    return summary
