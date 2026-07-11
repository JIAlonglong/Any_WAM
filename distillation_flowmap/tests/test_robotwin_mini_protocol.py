import json
from pathlib import Path

from distillation_flowmap.ablation.robotwin_mini_protocol import (
    build_eval_pairs,
    build_index_split,
    dataset_indices_for_manifest,
    eval_seed_for_pair,
    load_eval_pairs,
    write_protocol_manifests,
)


def test_index_split_is_deterministic_and_disjoint():
    split_a = build_index_split(
        ["place_a2b_right", "open_microwave"],
        train_samples_per_task=4,
        heldout_samples_per_task=2,
        protocol_seed=0,
    )
    split_b = build_index_split(
        ["place_a2b_right", "open_microwave"],
        train_samples_per_task=4,
        heldout_samples_per_task=2,
        protocol_seed=0,
    )

    assert split_a == split_b
    for task_entry in split_a["tasks"]:
        train_ids = set(task_entry["train_indices"])
        heldout_ids = set(task_entry["heldout_indices"])
        assert train_ids
        assert heldout_ids
        assert train_ids.isdisjoint(heldout_ids)


def test_eval_pairs_are_seeded_but_independent_of_training_seed():
    pairs_a = build_eval_pairs(["1000,0", "1000,500"], protocol_seed=7)
    pairs_b = build_eval_pairs(["1000,0", "1000,500"], protocol_seed=7)

    assert pairs_a == pairs_b
    assert pairs_a == [
        {"pair_id": "t1000_r0_i0", "t": 1000.0, "r": 0.0, "pair_seed": 7},
        {"pair_id": "t1000_r500_i1", "t": 1000.0, "r": 500.0, "pair_seed": 1016},
    ]


def test_write_protocol_manifests_records_paths_and_splits(tmp_path):
    paths = write_protocol_manifests(
        root=tmp_path,
        task_preset="core4",
        protocol_seed=0,
        task_names=["place_a2b_right", "open_microwave"],
        train_samples_per_task=3,
        heldout_samples_per_task=2,
        pairs=["1000,0"],
    )

    assert Path(paths["train_manifest_path"]).exists()
    assert Path(paths["heldout_eval_manifest_path"]).exists()
    assert Path(paths["eval_pairs_path"]).exists()
    assert "core4_protocol_seed_0" in paths["protocol_manifest_dir"]

    train = json.loads(Path(paths["train_manifest_path"]).read_text())
    heldout = json.loads(Path(paths["heldout_eval_manifest_path"]).read_text())
    eval_pairs = json.loads(Path(paths["eval_pairs_path"]).read_text())

    assert train["split"] == "train"
    assert heldout["split"] == "heldout"
    assert eval_pairs["pairs"][0]["pair_id"] == "t1000_r0_i0"
    assert train["tasks"][0]["indices"] == [0, 1, 2]
    assert heldout["tasks"][0]["indices"] == [3, 4]


class _TinyTaskDataset:
    def __init__(self, repo_id, length):
        self.repo_id = repo_id
        self.new_metas = [{"local": i} for i in range(length)]

    def __len__(self):
        return len(self.new_metas)


class _TinyMultiDataset:
    def __init__(self):
        self._datasets = [
            _TinyTaskDataset("/data/place_a2b_right-aloha-agilex_randomized_500-1000", 6),
            _TinyTaskDataset("/data/open_microwave", 5),
        ]
        self.acc_dset_num = {0: 0, 1: 6}


def test_dataset_indices_for_manifest_maps_task_local_indices_to_global_indices():
    dataset = _TinyMultiDataset()
    manifest = {
        "split": "heldout",
        "tasks": [
            {"task": "place_a2b_right", "indices": [3, 4]},
            {"task": "open_microwave", "indices": [1]},
        ],
    }

    assert dataset_indices_for_manifest(dataset, manifest) == [3, 4, 7]


def test_dataset_indices_for_manifest_maps_filtered_manifest_to_compact_indices():
    dataset = _TinyMultiDataset()
    dataset._datasets[0].new_metas = dataset._datasets[0].new_metas[3:5]
    dataset._datasets[1].new_metas = dataset._datasets[1].new_metas[1:2]
    dataset.acc_dset_num = {0: 0, 1: 2}
    manifest = {
        "split": "heldout",
        "tasks": [
            {"task": "place_a2b_right", "indices": [3, 4]},
            {"task": "open_microwave", "indices": [1]},
        ],
    }

    assert dataset_indices_for_manifest(
        dataset, manifest, manifest_is_compact=True
    ) == [0, 1, 2]


def test_offline_eval_uses_manifest_first_dataset_loading_for_rollout_and_video():
    repo_root = Path(__file__).resolve().parents[1]
    for name in ("rollout_eval_stage2.py", "rollout_eval_video_stage2.py"):
        source = (repo_root / name).read_text(encoding="utf-8")
        assert "cfg.dataset_sample_manifest = str(args.eval_manifest)" in source
        assert 'cfg.dataset_task_filter = ",".join(' in source
        assert "cfg.offline_eval_manifest_is_compact = True" in source
        assert "manifest_is_compact=bool(" in source


def test_load_eval_pairs_preserves_pair_ids_and_uses_sample_stable_seeds(tmp_path):
    path = tmp_path / "eval_pairs.json"
    path.write_text(json.dumps({
        "pairs": [
            {"pair_id": "shortcut", "t": 1000, "r": 0, "pair_seed": 11},
        ],
    }))

    pair = load_eval_pairs(path)[0]

    assert pair == {"pair_id": "shortcut", "t": 1000.0, "r": 0.0, "pair_seed": 11}
    assert eval_seed_for_pair(pair, batch_idx=0) == 11
    assert eval_seed_for_pair(pair, batch_idx=1) != 11
