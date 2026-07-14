"""Deterministic manifests and cache identities for progressive Cosmos Stage 2."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path


_CACHE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _normalise_records(records):
    out = []
    seen = set()
    for record in records:
        index = int(record["index"])
        task = str(record["task"]).strip()
        if index < 0:
            raise ValueError("record index must be non-negative")
        if not task:
            raise ValueError("record task must be non-empty")
        if index in seen:
            raise ValueError(f"duplicate record index {index}")
        seen.add(index)
        item = dict(record)
        item["index"] = index
        item["task"] = task
        out.append(item)
    if not out:
        raise ValueError("records must not be empty")
    return sorted(out, key=lambda item: (item["task"], item["index"]))


def _task_seed(task, protocol_seed):
    digest = hashlib.sha256(task.encode("utf-8")).digest()
    return int(protocol_seed) + int.from_bytes(digest[:8], "big")


def build_task_splits(records, *, selection_per_task, test_per_task, protocol_seed):
    """Split every task into disjoint train, selection, and final-test records."""
    selection_per_task = int(selection_per_task)
    test_per_task = int(test_per_task)
    if selection_per_task <= 0 or test_per_task <= 0:
        raise ValueError("selection_per_task and test_per_task must be positive")

    groups = defaultdict(list)
    for record in _normalise_records(records):
        groups[record["task"]].append(record)

    output = {"train": [], "selection": [], "test": []}
    required = selection_per_task + test_per_task + 1
    for task in sorted(groups):
        task_records = list(groups[task])
        if len(task_records) < required:
            raise ValueError(
                f"task {task!r} has {len(task_records)} records, needs at least {required}"
            )
        random.Random(_task_seed(task, protocol_seed)).shuffle(task_records)
        output["selection"].extend(task_records[:selection_per_task])
        output["test"].extend(
            task_records[selection_per_task:selection_per_task + test_per_task]
        )
        output["train"].extend(task_records[selection_per_task + test_per_task:])

    for split in output:
        output[split] = sorted(output[split], key=lambda item: item["index"])
    return output


def build_dataset_manifest(records, *, split, root_task, protocol_seed):
    """Return a SafeMultiLatentLeRobotDataset-compatible manifest with labels."""
    records = sorted(_normalise_records(records), key=lambda item: item["index"])
    root_task = str(root_task).strip()
    if not root_task:
        raise ValueError("root_task must be non-empty")
    return {
        "schema": "cosmos_progressive_stage2_manifest_v1",
        "split": str(split),
        "protocol_seed": int(protocol_seed),
        "tasks": [{"task": root_task, "indices": [item["index"] for item in records]}],
        "records": records,
    }


def aligned_teacher_path_indices(*, teacher_steps, student_steps):
    """Indices on an N-step teacher path aligned to each K-step student state."""
    teacher_steps = int(teacher_steps)
    student_steps = int(student_steps)
    if teacher_steps <= 0 or student_steps <= 0:
        raise ValueError("teacher_steps and student_steps must be positive")
    if teacher_steps % student_steps:
        raise ValueError("teacher_steps must be an integer multiple of student_steps")
    stride = teacher_steps // student_steps
    return list(range(0, teacher_steps + 1, stride))


def cosmos_latent_shape(config, *, batch_size):
    """Official Cosmos future-prediction latent shape for raw-policy rollouts."""
    return (
        int(batch_size),
        int(config.cosmos_latent_channels),
        int(config.cosmos_latent_frames),
        int(config.cosmos_latent_height),
        int(config.cosmos_latent_width),
    )


def manifest_digest(manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path(cache_root, *, sample_index, pair_id):
    component = _CACHE_COMPONENT.sub("_", str(pair_id)).strip("._")
    if not component:
        raise ValueError("pair_id must contain at least one safe character")
    return Path(cache_root) / f"sample_{int(sample_index):06d}__{component}.pt"
