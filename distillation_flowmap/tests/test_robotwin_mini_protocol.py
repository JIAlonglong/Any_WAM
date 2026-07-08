import json
from pathlib import Path

from distillation_flowmap.ablation.robotwin_mini_protocol import (
    build_eval_pairs,
    build_index_split,
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
