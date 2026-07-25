import csv
import json
from pathlib import Path

import pytest

from evaluation.libero.cosmos_progressive_eval_summary import (
    LIBERO_SUITES,
    merge_formal_records,
    merge_student_matrix,
)


def _write_formal_records(root: Path, checkpoint: Path, *, suite: str, steps: int, episodes: int):
    shard_ranges = ((0, 0, 3), (1, 3, 6), (2, 6, 8), (3, 8, 10))
    for shard, task_start, task_end in shard_ranges:
        for episode in range(episodes):
            records_dir = root / f"shard_{shard}" / f"seed_{episode}" / "records"
            records_dir.mkdir(parents=True, exist_ok=True)
            for task in range(task_start, task_end):
                record = {
                    "libero_benchmark": suite,
                    "task_idx": task,
                    "episode_idx": episode,
                    "seed": episode,
                    "s4_checkpoint": str(checkpoint.resolve()),
                    "student_steps": steps,
                    "model_role": "stage2_target",
                    "video_steps": steps,
                    "action_steps": steps,
                    "checkpoint_contract_identity": "contract-v1",
                    "success": (task + episode) % 2 == 0,
                }
                (records_dir / f"task_{task}_episode_{episode}.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )


def test_formal_merger_validates_suite_records_and_writes_json_and_csv(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    root = tmp_path / "formal"
    _write_formal_records(
        root, checkpoint, suite="libero_spatial", steps=2, episodes=2
    )

    summary = merge_formal_records(
        root=root,
        checkpoint=checkpoint,
        suite="libero_spatial",
        seed_count=2,
        requested_steps=2,
        shard_plan="0:0:3;1:3:6;2:6:8;3:8:10",
        evaluation_classification="formal_verified",
        is_formal=True,
        model_role="stage2_target",
    )

    assert summary["libero_benchmark"] == "libero_spatial"
    assert summary["num_records"] == 20
    assert summary["num_tasks"] == 10
    assert len(summary["bootstrap_ci_95"]) == 2
    assert set(summary["per_task_success"]) == {
        f"libero_spatial:{task}" for task in range(10)
    }
    assert (root / "formal_summary.json").is_file()
    assert not (root / ".formal_summary.json.tmp").exists()
    assert not (root / ".formal_summary.csv.tmp").exists()
    with (root / "formal_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert {(row["libero_benchmark"], int(row["task_idx"])) for row in rows} == {
        ("libero_spatial", task) for task in range(10)
    }


def test_formal_merger_rejects_record_from_wrong_suite(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    root = tmp_path / "formal"
    _write_formal_records(root, checkpoint, suite="libero_10", steps=1, episodes=1)
    record = root / "shard_0" / "seed_0" / "records" / "task_0_episode_0.json"
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["libero_benchmark"] = "libero_goal"
    record.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="suite mismatch"):
        merge_formal_records(
            root=root,
            checkpoint=checkpoint,
            suite="libero_10",
            seed_count=1,
            requested_steps=1,
            shard_plan="0:0:3;1:3:6;2:6:8;3:8:10",
            evaluation_classification="formal_verified",
            is_formal=True,
            model_role="stage2_target",
        )


def _write_child_summary(root: Path, checkpoint: Path, *, suite: str, steps: int, episodes: int):
    child = root / f"k{steps}" / suite
    child.mkdir(parents=True)
    payload = {
        "schema": "cosmos_progressive_s4_formal_eval_v2",
        "checkpoint": str(checkpoint.resolve()),
        "libero_benchmark": suite,
        "student_steps": steps,
        "model_role": "stage2_target",
        "video_steps": steps,
        "action_steps": steps,
        "checkpoint_contract_identity": "contract-v1",
        "evaluation_classification": "formal_verified",
        "is_formal": True,
        "num_tasks": 10,
        "num_records": 10 * episodes,
        "seeds_per_task": episodes,
        "per_task_success": {
            f"{suite}:{task}": (task + steps) / 20.0 for task in range(10)
        },
    }
    (child / "formal_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_student_matrix_requires_all_four_suites_at_matched_124_and_writes_csv(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    root = tmp_path / "matrix"
    for steps in (1, 2, 4):
        for suite in LIBERO_SUITES:
            _write_child_summary(
                root, checkpoint, suite=suite, steps=steps, episodes=3
            )

    summary = merge_student_matrix(
        root=root,
        checkpoint=checkpoint,
        evaluation_classification="formal_verified",
        is_formal=True,
        model_role="stage2_target",
        episodes_per_task=3,
    )

    assert summary["suites"] == list(LIBERO_SUITES)
    assert summary["steps"] == [1, 2, 4]
    assert summary["unique_tasks"] == 40
    assert summary["task_budget_cells"] == 120
    assert summary["episodes_per_k"] == 120
    assert len(summary["summaries"]) == 12
    with (root / "matrix_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert not (root / ".matrix_summary.json.tmp").exists()
    assert not (root / ".matrix_summary.csv.tmp").exists()
    assert len(rows) == 12
    assert {
        (row["libero_benchmark"], int(row["video_steps"]), int(row["action_steps"]))
        for row in rows
    } == {(suite, step, step) for suite in LIBERO_SUITES for step in (1, 2, 4)}


def test_student_matrix_fails_if_any_suite_is_missing(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    root = tmp_path / "matrix"
    for steps in (1, 2, 4):
        for suite in LIBERO_SUITES:
            if (suite, steps) != ("libero_goal", 4):
                _write_child_summary(
                    root, checkpoint, suite=suite, steps=steps, episodes=1
                )

    with pytest.raises(SystemExit, match="missing child summary.*libero_goal.*K=4"):
        merge_student_matrix(
            root=root,
            checkpoint=checkpoint,
            evaluation_classification="formal_verified",
            is_formal=True,
            model_role="stage2_target",
            episodes_per_task=1,
        )
