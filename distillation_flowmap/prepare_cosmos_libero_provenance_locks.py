#!/usr/bin/env python3
"""Atomically prepare the four exact locks consumed by Cosmos Stage-2."""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from distillation_flowmap.cosmos_libero_provenance import (
    ProvenanceError,
    build_artifact_lock,
    canonical_json,
)


DATASET_COMPACT = (
    "empty_emb.pt",
    "meta/info.json",
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
    "meta/episodes_ori.jsonl",
    "meta/episodes_stats.jsonl",
)
TEACHER_COMPACT = ("config.json", "libero_dataset_statistics.json")
TEACHER_LARGE = (
    "Cosmos-Policy-LIBERO-Predict2-2B.pt",
    "libero_t5_embeddings.pkl",
)
LOCAL_MODEL_COMPACT = (
    "config.json",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
)
LOCAL_MODEL_LARGE = ("model-480p-16fps.pt", "tokenizer/tokenizer.pth")


def _all_plain_files(root: Path) -> tuple[str, ...]:
    values = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ProvenanceError(f"artifact payload must not contain symlinks: {path}")
        if path.is_file():
            values.append(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ProvenanceError(f"artifact payload entry is not a file or directory: {path}")
    return tuple(sorted(values))


def _partition(
    root: Path, compact: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    all_files = _all_plain_files(root)
    missing = sorted(set(compact) - set(all_files))
    if missing:
        raise ProvenanceError(f"artifact is missing required compact files: {missing}")
    large = tuple(path for path in all_files if path not in set(compact))
    if not large:
        raise ProvenanceError("artifact has no payload files")
    return compact, large


def prepare_payloads(args: argparse.Namespace) -> dict[str, dict]:
    dataset = Path(args.dataset_root).resolve(strict=True)
    teacher = Path(args.teacher_root).resolve(strict=True)
    local_model = Path(args.local_model_root).resolve(strict=True)
    stage1_target = Path(args.stage1_target_root).resolve(strict=True)
    dataset_compact, dataset_large = _partition(dataset, DATASET_COMPACT)
    stage1_compact_candidates = (
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    )
    stage1_files = set(_all_plain_files(stage1_target))
    stage1_compact = tuple(
        path for path in stage1_compact_candidates if path in stage1_files
    )
    if "transformer/config.json" not in stage1_compact:
        raise ProvenanceError("Stage-1 target is missing transformer/config.json")
    stage1_large = tuple(sorted(stage1_files - set(stage1_compact)))
    if not stage1_large:
        raise ProvenanceError("Stage-1 target has no transformer weight payload")
    return {
        "dataset.lock.json": build_artifact_lock(
            dataset,
            compact_paths=dataset_compact,
            large_paths=dataset_large,
            immutable_store=False,
        ),
        "teacher.lock.json": build_artifact_lock(
            teacher,
            compact_paths=TEACHER_COMPACT,
            large_paths=TEACHER_LARGE,
            immutable_store=False,
        ),
        "local_model.lock.json": build_artifact_lock(
            local_model,
            compact_paths=LOCAL_MODEL_COMPACT,
            large_paths=LOCAL_MODEL_LARGE,
            immutable_store=False,
        ),
        "video_vae.lock.json": build_artifact_lock(
            stage1_target,
            compact_paths=stage1_compact,
            large_paths=stage1_large,
            immutable_store=False,
        ),
    }


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ProvenanceError("atomic no-replace publication requires renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"lock output already exists: {destination}")
        raise OSError(error, os.strerror(error), destination)


def _write_atomic(output: Path, payloads: dict[str, dict]) -> None:
    parent = output.parent.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"lock output already exists: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    try:
        for name, payload in sorted(payloads.items()):
            path = temporary / name
            path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _rename_noreplace(temporary, output)
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--teacher-root", required=True)
    parser.add_argument("--local-model-root", required=True)
    parser.add_argument("--stage1-target-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        output = Path(args.output_root)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"lock output already exists: {output}")
        payloads = prepare_payloads(args)
        if not args.dry_run:
            _write_atomic(output, payloads)
        for name, payload in sorted(payloads.items()):
            print(f"{name}={canonical_json(payload)}")
        print(f"LOCK_ROOT={output.resolve(strict=not args.dry_run)}")
        return 0
    except (OSError, ProvenanceError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
