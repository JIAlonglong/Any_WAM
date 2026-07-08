"""Protocol helpers for RobotWin StepWAM mini-ablation runs."""

import json
from pathlib import Path


DEFAULT_EVAL_PAIRS = ("1000,0", "1000,500", "750,250")
PAIR_SEED_STRIDE = 1009


def _as_task_list(task_names):
    tasks = [str(task).strip() for task in task_names if str(task).strip()]
    if not tasks:
        raise ValueError("task_names must contain at least one task")
    return tasks


def parse_pair(pair):
    if isinstance(pair, str):
        left, right = pair.split(",", 1)
        return float(left), float(right)
    if len(pair) != 2:
        raise ValueError(f"Invalid eval pair {pair!r}; expected t,r")
    return float(pair[0]), float(pair[1])


def _format_time(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def build_eval_pairs(pairs=DEFAULT_EVAL_PAIRS, protocol_seed=0):
    out = []
    for idx, pair in enumerate(pairs):
        t_value, r_value = parse_pair(pair)
        out.append({
            "pair_id": f"t{_format_time(t_value)}_r{_format_time(r_value)}_i{idx}",
            "t": t_value,
            "r": r_value,
            "pair_seed": int(protocol_seed) + idx * PAIR_SEED_STRIDE,
        })
    if not out:
        raise ValueError("pairs must contain at least one t,r pair")
    return out


def build_index_split(
    task_names,
    train_samples_per_task=80,
    heldout_samples_per_task=20,
    protocol_seed=0,
):
    train_samples_per_task = int(train_samples_per_task)
    heldout_samples_per_task = int(heldout_samples_per_task)
    if train_samples_per_task <= 0:
        raise ValueError("train_samples_per_task must be positive")
    if heldout_samples_per_task <= 0:
        raise ValueError("heldout_samples_per_task must be positive")

    tasks = []
    for task in _as_task_list(task_names):
        train_end = train_samples_per_task
        heldout_end = train_samples_per_task + heldout_samples_per_task
        tasks.append({
            "task": task,
            "train_indices": list(range(0, train_end)),
            "heldout_indices": list(range(train_end, heldout_end)),
        })
    return {
        "schema": "robotwin_stepwam_mini_split_v1",
        "protocol_seed": int(protocol_seed),
        "train_samples_per_task": train_samples_per_task,
        "heldout_samples_per_task": heldout_samples_per_task,
        "tasks": tasks,
    }


def protocol_manifest_paths(root, task_preset, protocol_seed):
    manifest_dir = (
        Path(root)
        / "protocol"
        / "manifests"
        / f"{task_preset}_protocol_seed_{int(protocol_seed)}"
    )
    return {
        "protocol_manifest_dir": str(manifest_dir),
        "train_manifest_path": str(manifest_dir / "train_manifest.json"),
        "heldout_eval_manifest_path": str(manifest_dir / "heldout_eval_manifest.json"),
        "eval_pairs_path": str(manifest_dir / "eval_pairs.json"),
    }


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _split_manifest(split, split_data, task_preset):
    key = "train_indices" if split == "train" else "heldout_indices"
    return {
        "schema": "robotwin_stepwam_mini_manifest_v1",
        "task_preset": task_preset,
        "split": split,
        "protocol_seed": split_data["protocol_seed"],
        "samples_per_task": (
            split_data["train_samples_per_task"]
            if split == "train"
            else split_data["heldout_samples_per_task"]
        ),
        "tasks": [
            {
                "task": entry["task"],
                "indices": list(entry[key]),
            }
            for entry in split_data["tasks"]
        ],
    }


def write_protocol_manifests(
    root,
    task_preset,
    protocol_seed,
    task_names,
    train_samples_per_task=80,
    heldout_samples_per_task=20,
    pairs=DEFAULT_EVAL_PAIRS,
):
    split_data = build_index_split(
        task_names=task_names,
        train_samples_per_task=train_samples_per_task,
        heldout_samples_per_task=heldout_samples_per_task,
        protocol_seed=protocol_seed,
    )
    paths = protocol_manifest_paths(root, task_preset, protocol_seed)
    _write_json(paths["train_manifest_path"], _split_manifest("train", split_data, task_preset))
    _write_json(paths["heldout_eval_manifest_path"], _split_manifest("heldout", split_data, task_preset))
    _write_json(paths["eval_pairs_path"], {
        "schema": "robotwin_stepwam_mini_eval_pairs_v1",
        "task_preset": task_preset,
        "protocol_seed": int(protocol_seed),
        "pairs": build_eval_pairs(pairs=pairs, protocol_seed=protocol_seed),
    })
    return paths
