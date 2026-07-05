from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


_STEP_RE = re.compile(r"^step_(\d+)$")


def parse_step(path: Path) -> int | None:
    match = _STEP_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def discover_checkpoints(run_dir: Path) -> list[Path]:
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        return []
    paths = [
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir() and parse_step(path) is not None
    ]
    return sorted(paths, key=lambda path: parse_step(path) or -1)


def resolve_checkpoint_transformer(step_dir: Path, kind: str = "online_student") -> Path:
    candidates: list[Path]
    if kind == "auto":
        candidates = [
            step_dir / "online_student" / "transformer",
            step_dir / "target_student" / "transformer",
            step_dir / "transformer",
        ]
    elif kind in ("online_student", "target_student"):
        candidates = [
            step_dir / kind / "transformer",
            step_dir / "transformer",
        ]
    elif kind == "transformer":
        candidates = [step_dir / "transformer"]
    else:
        raise ValueError(
            f"Unsupported checkpoint kind {kind!r}; "
            "expected auto, online_student, target_student, or transformer."
        )

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No transformer checkpoint found under {step_dir} for kind={kind!r}"
    )


def _flatten(prefix: str, value, out: dict[str, object]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            _flatten(next_prefix, child, out)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        out[prefix] = value


def _load_metrics(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    flat: dict[str, object] = {}
    _flatten("", payload, flat)
    return flat


def write_csv(rows: Iterable[dict[str, object]], output_csv: Path) -> None:
    rows = list(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(rows: Iterable[dict[str, object]], output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def build_eval_command(args, checkpoint_transformer: Path, output_json: Path) -> list[str]:
    cmd = [
        args.python,
        "distillation_flowmap/eval_cosmos_policy_stage1_metrics.py",
        "--checkpoint-transformer",
        str(checkpoint_transformer),
        "--config-file",
        args.config_file,
        "--num-samples",
        str(args.num_samples),
        "--start-index",
        str(args.start_index),
        "--stride",
        str(args.stride),
        "--batch-size",
        str(args.batch_size),
        "--student-gpu",
        str(args.student_gpu),
        "--teacher-gpu",
        str(args.teacher_gpu),
        "--dtype",
        args.dtype,
        "--output-json",
        str(output_json),
    ]
    if args.dataset_path:
        cmd.extend(["--dataset-path", args.dataset_path])
    if args.teacher_model_path:
        cmd.extend(["--teacher-model-path", args.teacher_model_path])
    if args.pairs:
        cmd.append("--pairs")
        cmd.extend(args.pairs)
    return cmd


def run_eval(args, checkpoint_transformer: Path, output_json: Path) -> list[str]:
    cmd = build_eval_command(args, checkpoint_transformer, output_json)
    if args.dry_run:
        print("DRY_RUN", " ".join(cmd), flush=True)
        return cmd
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)
    return cmd


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-file", default="distillation_flowmap.config_libero_cosmos_policy_stage1")
    parser.add_argument("--checkpoint-kind", default="online_student",
                        choices=["auto", "online_student", "target_student", "transformer"])
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pairs", nargs="+", default=["1000:1000", "750:750", "500:500", "250:250"])
    parser.add_argument("--student-gpu", type=int, default=0)
    parser.add_argument("--teacher-gpu", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for run_dir_s in args.run_dir:
        run_dir = Path(run_dir_s)
        for step_dir in discover_checkpoints(run_dir):
            step = parse_step(step_dir)
            checkpoint_transformer = resolve_checkpoint_transformer(
                step_dir, kind=args.checkpoint_kind)
            metrics_json = output_dir / f"{run_dir.name}_step_{step}.json"
            cmd = run_eval(args, checkpoint_transformer, metrics_json)
            row: dict[str, object] = {
                "run_dir": str(run_dir),
                "step_dir": str(step_dir),
                "checkpoint_transformer": str(checkpoint_transformer),
                "step": step,
                "config_file": args.config_file,
                "metrics_json": str(metrics_json),
                "command": " ".join(cmd),
            }
            if metrics_json.exists():
                row.update(_load_metrics(metrics_json))
            rows.append(row)

    write_jsonl(rows, output_dir / "summary.jsonl")
    write_csv(rows, output_dir / "summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
