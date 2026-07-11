"""Protocol helpers for RobotWin StepWAM mini-ablation runs."""

import json
from pathlib import Path


DEFAULT_EVAL_PAIRS = ("1000,0", "1000,500", "750,250")
PAIR_SEED_STRIDE = 1009
SAMPLE_SEED_STRIDE = 1000003


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


def _repo_task_name(repo_id):
    name = Path(str(repo_id).rstrip("/")).name
    return name.split("-", 1)[0]


def _repo_matches_task(repo_id, task_name):
    basename = Path(str(repo_id).rstrip("/")).name.lower()
    canonical = basename.split("-", 1)[0]
    task_name = str(task_name).strip().lower()
    return bool(
        task_name
        and (task_name == basename or task_name == canonical or basename.startswith(f"{task_name}-"))
    )


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


def load_eval_pairs(path):
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    raw_pairs = payload.get("pairs", payload) if isinstance(payload, dict) else payload
    out = []
    for idx, pair in enumerate(raw_pairs):
        if isinstance(pair, dict):
            t_value, r_value = parse_pair([pair["t"], pair["r"]])
            pair_id = str(pair.get("pair_id") or f"t{_format_time(t_value)}_r{_format_time(r_value)}_i{idx}")
            pair_seed = int(pair.get("pair_seed", idx * PAIR_SEED_STRIDE))
        else:
            t_value, r_value = parse_pair(pair)
            pair_id = f"t{_format_time(t_value)}_r{_format_time(r_value)}_i{idx}"
            pair_seed = idx * PAIR_SEED_STRIDE
        out.append({
            "pair_id": pair_id,
            "t": t_value,
            "r": r_value,
            "pair_seed": pair_seed,
        })
    if not out:
        raise ValueError(f"{path} does not contain any eval pairs")
    return out


def eval_seed_for_pair(pair, batch_idx):
    return int(pair["pair_seed"]) + int(batch_idx) * SAMPLE_SEED_STRIDE


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


def dataset_records_for_manifest(dataset, manifest, manifest_is_compact=False):
    """Resolve manifest entries to global dataset indices with task identity."""
    if isinstance(manifest, (str, Path)):
        with Path(manifest).open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    task_entries = manifest.get("tasks", [])
    if not task_entries:
        raise ValueError("manifest must contain at least one task entry")
    datasets = list(getattr(dataset, "_datasets", []))
    acc_dset_num = getattr(dataset, "acc_dset_num", {})
    if not datasets:
        raise ValueError("dataset must expose _datasets for task-local manifest lookup")

    records = []
    for task_entry in task_entries:
        task_name = str(task_entry["task"])
        matches = [
            (dset_id, sub_dataset)
            for dset_id, sub_dataset in enumerate(datasets)
            if _repo_matches_task(getattr(sub_dataset, "repo_id", dset_id), task_name)
        ]
        if not matches:
            raise KeyError(f"No dataset shard matched manifest task {task_name!r}")
        if len(matches) > 1:
            matched = ", ".join(str(getattr(ds, "repo_id", dset_id)) for dset_id, ds in matches)
            raise ValueError(f"Manifest task {task_name!r} matched multiple dataset shards: {matched}")
        dset_id, sub_dataset = matches[0]
        offset = int(acc_dset_num.get(dset_id, 0))
        selected_indices = [int(idx) for idx in task_entry.get("indices", [])]
        if manifest_is_compact:
            if len(sub_dataset) != len(selected_indices):
                raise ValueError(
                    f"Compact manifest task {task_name!r} expected "
                    f"{len(selected_indices)} samples, found {len(sub_dataset)}"
                )
            selected_indices = list(range(len(sub_dataset)))
        for local_idx in selected_indices:
            local_idx = int(local_idx)
            if local_idx < 0 or local_idx >= len(sub_dataset):
                raise IndexError(
                    f"Manifest task {task_name!r} local index {local_idx} is out of range "
                    f"for dataset length {len(sub_dataset)}"
                )
            records.append({
                "global_index": offset + local_idx,
                "task": task_name,
            })
    if not records:
        raise ValueError("manifest did not select any dataset indices")
    return records


def dataset_indices_for_manifest(dataset, manifest, manifest_is_compact=False):
    return [
        record["global_index"]
        for record in dataset_records_for_manifest(
            dataset,
            manifest,
            manifest_is_compact=manifest_is_compact,
        )
    ]


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
