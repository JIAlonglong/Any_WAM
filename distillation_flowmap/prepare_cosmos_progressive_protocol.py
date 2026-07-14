#!/usr/bin/env python3
"""Create deterministic train/selection/test manifests for Cosmos Stage 2."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "wan_va"))

from distillation.patches import SafeMultiLatentLeRobotDataset, install_flash_attn_stub
from distillation_flowmap.cosmos_progressive_protocol import (
    build_dataset_manifest,
    build_task_splits,
    manifest_digest,
)


install_flash_attn_stub()


def _write_json(path, payload, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing protocol artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pair_payload(pair_texts, protocol_seed):
    pairs = []
    for pair_index, text in enumerate(pair_texts):
        try:
            t_value, r_value = (float(value) for value in text.split(",", 1))
        except ValueError as exc:
            raise ValueError(f"Invalid pair {text!r}; expected t,r") from exc
        if not (0.0 <= r_value <= t_value <= 1000.0):
            raise ValueError(f"Invalid pair {text!r}; expected 0 <= r <= t <= 1000")
        pairs.append({
            "pair_id": f"t{int(t_value)}_r{int(r_value)}_i{pair_index}",
            "t": t_value,
            "r": r_value,
            "pair_seed": int(protocol_seed) + pair_index * 1009,
        })
    if not pairs:
        raise ValueError("At least one fixed t,r pair is required")
    return {"schema": "cosmos_progressive_eval_pairs_v1", "pairs": pairs}


def _dataset_records(dataset):
    records = []
    for index in range(len(dataset)):
        meta = dataset.get_sample_meta(index)
        task = str(meta.get("action_text") or "").strip()
        if not task:
            tasks = meta.get("tasks", [])
            task = str(tasks[0]).strip() if tasks else ""
        if not task:
            raise ValueError(f"Dataset sample {index} has no task label")
        records.append({
            "index": int(index),
            "task": task,
            "episode_index": int(meta.get("episode_index", -1)),
            "start_frame": int(meta.get("start_frame", -1)),
            "end_frame": int(meta.get("end_frame", -1)),
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="distillation_flowmap.config_libero_cosmos_policy_stage2_progressive")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-per-task", type=int, default=3)
    parser.add_argument("--test-per-task", type=int, default=5)
    parser.add_argument("--protocol-seed", type=int, default=20260714)
    parser.add_argument("--pairs", nargs="+", default=["1000,0"])
    parser.add_argument("--root-task", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = importlib.import_module(args.config).cfg
    cfg.rank = 0
    cfg.local_rank = 0
    cfg.world_size = 1
    cfg.dataset_path = str(Path(args.dataset_path).resolve())
    cfg.empty_emb_path = os.path.join(cfg.dataset_path, "empty_emb.pt")
    cfg.cache_dataset_in_memory = False
    cfg.return_raw_observation = False
    cfg.cosmos_policy_use_raw_inference = False
    cfg.dataset_sample_manifest = None

    dataset = SafeMultiLatentLeRobotDataset(config=cfg)
    records = _dataset_records(dataset)
    splits = build_task_splits(
        records,
        selection_per_task=args.selection_per_task,
        test_per_task=args.test_per_task,
        protocol_seed=args.protocol_seed,
    )
    root_task = args.root_task or Path(cfg.dataset_path).name.split("-", 1)[0]
    output_dir = Path(args.output_dir)
    manifests = {}
    for split_name, split_records in splits.items():
        manifest = build_dataset_manifest(
            split_records,
            split=split_name,
            root_task=root_task,
            protocol_seed=args.protocol_seed,
        )
        manifest["dataset_path"] = cfg.dataset_path
        path = output_dir / f"{split_name}_manifest.json"
        _write_json(path, manifest, args.overwrite)
        manifests[split_name] = {
            "path": str(path),
            "digest": manifest_digest(manifest),
            "num_records": len(split_records),
        }

    pairs = _pair_payload(args.pairs, args.protocol_seed)
    pairs_path = output_dir / "eval_pairs.json"
    _write_json(pairs_path, pairs, args.overwrite)
    protocol = {
        "schema": "cosmos_progressive_stage2_protocol_v1",
        "config": args.config,
        "dataset_path": cfg.dataset_path,
        "protocol_seed": int(args.protocol_seed),
        "selection_per_task": int(args.selection_per_task),
        "test_per_task": int(args.test_per_task),
        "num_tasks": len({record["task"] for record in records}),
        "manifests": manifests,
        "eval_pairs_path": str(pairs_path),
        "teacher_cache_dirs": {
            "selection": str(output_dir / "teacher_cache" / "selection"),
            "test": str(output_dir / "teacher_cache" / "test"),
        },
    }
    _write_json(output_dir / "protocol.json", protocol, args.overwrite)
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
