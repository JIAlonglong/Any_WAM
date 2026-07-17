"""Summarize guarded RobotWin teacher/student offline metric plans.

The input is deliberately a launcher-produced plan rather than a directory glob:
this prevents accidental mixing of full-RobotWin, cache-only, or unrelated runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


PLAN_SCHEMA = "robotwin_teacher_student_offline_plan_v1"
RESULT_PREFIX = re.compile(
    r"^rollout_eval/(?P<pair>[^/]+)/s(?P<student_steps>\d+)_t(?P<teacher_steps>\d+)/(?P<metric>[^/]+)$"
)

# The video evaluator currently reports velocity MSE (not velocity L1) for both
# teacher and student.  Every pair below therefore corresponds to a real, shared
# evaluator JSON key pair rather than an invented missing metric.
METRIC_PAIRS = (
    ("action", "action_gt_xr_mse", "action_teacher_gt_xr_mse"),
    ("action", "action_gt_xr_l1", "action_teacher_gt_xr_l1"),
    ("action", "action_gt_v_mse", "action_teacher_gt_v_mse"),
    ("action", "action_gt_v_l1", "action_teacher_gt_v_l1"),
    ("video", "video_gt_x_mse", "video_teacher_gt_x_mse"),
    ("video", "video_gt_x_l1", "video_teacher_gt_x_l1"),
    ("video", "video_gt_v_mse", "video_teacher_gt_v_mse"),
)


def student_minus_teacher(student: float, teacher: float) -> float:
    """Return the signed error change; a negative result favors the student."""
    return student - teacher


def summarize_metric_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add signed teacher/student comparisons without mutating callers' rows."""
    summarized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        student = float(item["student"])
        teacher = float(item["teacher"])
        if not math.isfinite(student) or not math.isfinite(teacher):
            raise ValueError(f"non-finite metric for {item.get('metric', '<unknown>')}")
        delta = student_minus_teacher(student, teacher)
        item["student"] = student
        item["teacher"] = teacher
        item["student_minus_teacher"] = delta
        item["student_better"] = delta < 0.0
        summarized.append(item)
    return summarized


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_pair_ids(entry: dict[str, Any]) -> set[str]:
    """Read the immutable protocol pair file named by a launcher plan entry."""
    pair_path_value = entry.get("pair_path")
    pair_digest = entry.get("pair_sha256")
    if not isinstance(pair_path_value, str) or not pair_path_value:
        raise ValueError("launcher plan entry is missing pair_path")
    if not isinstance(pair_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", pair_digest):
        raise ValueError("launcher plan entry is missing a valid pair_sha256")
    pair_path = Path(pair_path_value).resolve(strict=False)
    if not pair_path.is_absolute():
        raise ValueError("launcher plan pair_path must be absolute")
    if _sha256_file(pair_path) != pair_digest:
        raise ValueError("pair SHA-256 does not match the launcher plan")
    pair_payload = _read_object(pair_path, "fixed eval pair file")
    pairs = pair_payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("fixed eval pair file contains no pairs")
    pair_ids: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("fixed eval pair file has a malformed pair entry")
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("fixed eval pair file has a missing pair_id")
        if pair_id in pair_ids:
            raise ValueError(f"fixed eval pair file repeats pair_id {pair_id!r}")
        pair_ids.add(pair_id)
    return pair_ids


def _validate_plan(plan_path: Path) -> tuple[dict[str, Any], Path]:
    plan = _read_object(plan_path, "launcher plan")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("plan is not a launcher-produced RobotWin offline plan")
    output_root_value = plan.get("output_root")
    if not isinstance(output_root_value, str) or not output_root_value:
        raise ValueError("launcher plan is missing output_root")
    root = Path(output_root_value).resolve(strict=False)
    if not root.is_absolute():
        raise ValueError("launcher plan output_root must be absolute")
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("launcher plan contains no result entries")
    return plan, root


def _required_metrics_for_result(
    result: dict[str, Any], entry: dict[str, Any]
) -> list[dict[str, Any]]:
    source = entry.get("rollout_source")
    if source not in {"target", "online"}:
        raise ValueError("plan entry has an unsupported rollout_source")
    if result.get("rollout_source_requested") != source:
        raise ValueError(
            f"result rollout_source_requested does not match planned source {source!r}"
        )
    if result.get("rollout_source_resolved") != source:
        raise ValueError(
            f"result rollout_source_resolved does not match planned source {source!r}"
        )
    try:
        student_steps = int(entry["student_steps"])
        teacher_steps = int(entry["teacher_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("plan entry is missing integer equal-NFE steps") from exc
    if student_steps != teacher_steps or student_steps not in {1, 2, 4}:
        raise ValueError("plan entry is not an allowed equal-NFE K=1/2/4 comparison")

    expected_pair_ids = _expected_pair_ids(entry)

    by_pair: dict[str, dict[str, Any]] = {}
    for key, value in result.items():
        match = RESULT_PREFIX.match(key)
        if match is None:
            continue
        if (
            int(match.group("student_steps")) != student_steps
            or int(match.group("teacher_steps")) != teacher_steps
        ):
            continue
        by_pair.setdefault(match.group("pair"), {})[match.group("metric")] = value
    if not by_pair:
        raise ValueError(
            f"result has no rollout_eval pair for planned s{student_steps}_t{teacher_steps}"
        )
    observed_pair_ids = set(by_pair)
    missing_pair_ids = sorted(expected_pair_ids - observed_pair_ids)
    if missing_pair_ids:
        raise ValueError(
            "missing fixed eval pair(s): " + ", ".join(missing_pair_ids)
        )
    unexpected_pair_ids = sorted(observed_pair_ids - expected_pair_ids)
    if unexpected_pair_ids:
        raise ValueError(
            "unexpected eval pair(s): " + ", ".join(unexpected_pair_ids)
        )

    rows: list[dict[str, Any]] = []
    for pair, metrics in sorted(by_pair.items()):
        missing = [
            key
            for _, student_key, teacher_key in METRIC_PAIRS
            for key in (student_key, teacher_key)
            if key not in metrics
        ]
        if missing:
            raise ValueError(
                "missing required metrics for "
                f"{pair}/s{student_steps}_t{teacher_steps}: {', '.join(sorted(missing))}"
            )
        for modality, student_key, teacher_key in METRIC_PAIRS:
            rows.append(
                {
                    "split": entry.get("split"),
                    "checkpoint": entry.get("checkpoint_name"),
                    "source": source,
                    "k": student_steps,
                    "pair": pair,
                    "modality": modality,
                    "metric": student_key,
                    "teacher_metric": teacher_key,
                    "teacher": metrics[teacher_key],
                    "student": metrics[student_key],
                    "result_json": entry.get("expected_raw_result"),
                }
            )
    return summarize_metric_rows(rows)


def _plan_result_path(entry: dict[str, Any], output_root: Path) -> Path:
    value = entry.get("expected_raw_result")
    if not isinstance(value, str) or not value:
        raise ValueError("launcher plan entry is missing expected_raw_result")
    path = Path(value).resolve(strict=False)
    if not _inside(path, output_root):
        raise ValueError("launcher plan result path escapes its isolated output_root")
    return path


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, int]:
    checkpoint = entry.get("checkpoint_name")
    source = entry.get("rollout_source")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("launcher plan entry is missing checkpoint_name")
    if source not in {"target", "online"}:
        raise ValueError("launcher plan entry has an unsupported rollout_source")
    try:
        student_steps = int(entry["student_steps"])
        teacher_steps = int(entry["teacher_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("launcher plan entry is missing integer equal-NFE steps") from exc
    if student_steps != teacher_steps or student_steps not in {1, 2, 4}:
        raise ValueError("launcher plan entry is not an allowed equal-NFE K=1/2/4 comparison")
    return checkpoint, source, student_steps


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row["split"]),
            str(row["checkpoint"]),
            str(row["source"]),
            int(row["k"]),
            str(row["pair"]),
            str(row["metric"]),
        ),
    )


def validate_plan_results(
    plan_path: Path | str,
    *,
    expected_split: str | None = None,
    require_execution: bool = False,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    """Validate every launcher-owned raw result without creating a summary."""
    plan_path = Path(plan_path)
    plan, output_root = _validate_plan(plan_path)
    if require_execution and plan.get("execution_requested") is not True:
        raise ValueError("core2 gate plan was not produced by an executed launcher run")
    if expected_split is not None:
        selection = plan.get("selection")
        if not isinstance(selection, dict) or selection.get("split") != expected_split:
            raise ValueError(f"launcher plan is not a {expected_split} plan")

    rows: list[dict[str, Any]] = []
    for entry in plan["entries"]:
        if not isinstance(entry, dict):
            raise ValueError("launcher plan has a malformed result entry")
        if expected_split is not None and entry.get("split") != expected_split:
            raise ValueError(f"launcher plan contains a non-{expected_split} result entry")
        result_path = _plan_result_path(entry, output_root)
        result = _read_object(result_path, "launcher-owned raw result")
        rows.extend(_required_metrics_for_result(result, entry))
    return plan, output_root, _sort_rows(rows)


def verify_completed_core2_gate(
    gate_plan_path: Path | str,
    *,
    required_entries: set[tuple[str, str, int]],
) -> list[dict[str, Any]]:
    """Require a completed, fully valid Core2 plan to cover final12 entries."""
    gate_plan_path = Path(gate_plan_path).resolve(strict=False)
    plan, output_root, rows = validate_plan_results(
        gate_plan_path,
        expected_split="core2",
        require_execution=True,
    )
    if gate_plan_path != (output_root / "plan.json").resolve(strict=False):
        raise ValueError("core2 gate must be the output_root/plan.json written by --run")
    observed_entries = {_entry_identity(entry) for entry in plan["entries"]}
    missing_entries = sorted(required_entries - observed_entries)
    if missing_entries:
        rendered = ", ".join(
            f"{checkpoint}/{source}/K{k}"
            for checkpoint, source, k in missing_entries
        )
        raise ValueError(
            "core2 gate does not cover required final12 entries: " + rendered
        )
    return rows


def _write_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    fieldnames = [
        "split",
        "checkpoint",
        "source",
        "k",
        "pair",
        "modality",
        "metric",
        "teacher_metric",
        "teacher",
        "student",
        "student_minus_teacher",
        "student_better",
        "result_json",
    ]
    with destination.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).replace("|", "\\|")


def _write_markdown(rows: list[dict[str, Any]], destination: Path) -> None:
    columns = (
        "split",
        "checkpoint",
        "source",
        "k",
        "pair",
        "modality",
        "metric",
        "teacher",
        "student",
        "student_minus_teacher",
        "student_better",
    )
    lines = [
        "# RobotWin offline teacher–student errors",
        "",
        "`student_minus_teacher = student - teacher`; negative means student better.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(row[name]) for name in columns) + " |")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(plan_path: Path | str, out: Path | str) -> list[dict[str, Any]]:
    """Validate launcher-owned raw results and write CSV/Markdown comparisons."""
    plan_path = Path(plan_path)
    _, output_root, rows = validate_plan_results(plan_path)
    out = Path(out).resolve(strict=False)
    if not _inside(out, output_root):
        raise ValueError("summary output must remain inside the plan output_root")
    if out.exists():
        raise ValueError(f"refusing to overwrite existing summary output: {out}")
    out.mkdir(parents=False)
    _write_csv(rows, out / "summary.csv")
    _write_markdown(rows, out / "summary.md")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize only raw result JSON files named by a guarded RobotWin launcher plan."
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--verify-core2-gate",
        type=Path,
        help="Validate an executed Core2 plan without writing a summary.",
    )
    parser.add_argument(
        "--require-entry",
        action="append",
        default=[],
        metavar="CHECKPOINT:SOURCE:K",
        help="Required Core2 coverage when verifying a gate; repeatable.",
    )
    args = parser.parse_args()

    if args.verify_core2_gate is not None:
        if args.plan is not None or args.out is not None:
            parser.error("--verify-core2-gate cannot be combined with --plan or --out")
        required_entries: set[tuple[str, str, int]] = set()
        for value in args.require_entry:
            checkpoint, separator, suffix = value.partition(":")
            source, separator2, k_text = suffix.partition(":")
            if not separator or not separator2 or not checkpoint or source not in {"target", "online"}:
                parser.error(f"invalid --require-entry: {value!r}")
            try:
                k = int(k_text)
            except ValueError:
                parser.error(f"invalid --require-entry K: {value!r}")
            if k not in {1, 2, 4}:
                parser.error(f"invalid --require-entry K: {value!r}")
            required_entries.add((checkpoint, source, k))
        rows = verify_completed_core2_gate(
            args.verify_core2_gate,
            required_entries=required_entries,
        )
        print(f"Verified {len(rows)} Core2 teacher/student metric rows")
        return

    if args.plan is None or args.out is None:
        parser.error("--plan and --out are required unless --verify-core2-gate is used")
    rows = summarize(args.plan, args.out)
    print(f"Wrote {len(rows)} teacher/student metric rows to {args.out}")


if __name__ == "__main__":
    main()
