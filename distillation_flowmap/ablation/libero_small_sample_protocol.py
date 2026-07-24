"""Frozen small-sample protocol for the LIBERO APM ablation."""

from __future__ import annotations

import json
import os
from pathlib import Path


BENCHMARK = "libero_10"
TASK_INDEX = 0
TASK_LANGUAGE = "put both the alphabet soup and the tomato sauce in the basket"
TRAIN_INDICES = tuple(range(40))
HELDOUT_INDICES = tuple(range(40, 50))


def build_libero_task0_protocol() -> dict:
    return {
        "benchmark": BENCHMARK,
        "task_index": TASK_INDEX,
        "task_language": TASK_LANGUAGE,
        "train_indices": list(TRAIN_INDICES),
        "heldout_indices": list(HELDOUT_INDICES),
    }


def _manifest(indices: tuple[int, ...], split: str) -> dict:
    return {
        "protocol": {
            "benchmark": BENCHMARK,
            "task_index": TASK_INDEX,
            "task_language": TASK_LANGUAGE,
            "split": split,
        },
        "tasks": [{"task": "libero", "indices": list(indices)}],
    }


def _write_or_validate(path: Path, payload: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"Existing manifest does not match frozen protocol: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_libero_task0_manifests(root: Path) -> dict[str, Path]:
    root = Path(root)
    paths = {
        "train": root / "train_manifest.json",
        "heldout": root / "heldout_manifest.json",
    }
    _write_or_validate(paths["train"], _manifest(TRAIN_INDICES, "train"))
    _write_or_validate(paths["heldout"], _manifest(HELDOUT_INDICES, "heldout"))
    return paths
