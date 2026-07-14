from distillation_flowmap.cosmos_progressive_protocol import (
    aligned_teacher_path_indices,
    build_dataset_manifest,
    build_task_splits,
    cosmos_latent_shape,
)
from types import SimpleNamespace


def test_task_splits_are_disjoint_and_reproducible():
    records = [
        {"index": index, "task": task}
        for task in ("drawer", "stack")
        for index in (range(0, 10) if task == "drawer" else range(10, 20))
    ]

    first = build_task_splits(
        records,
        selection_per_task=2,
        test_per_task=3,
        protocol_seed=17,
    )
    second = build_task_splits(
        records,
        selection_per_task=2,
        test_per_task=3,
        protocol_seed=17,
    )

    assert first == second
    train = {record["index"] for record in first["train"]}
    selection = {record["index"] for record in first["selection"]}
    test = {record["index"] for record in first["test"]}
    assert len(train) == 10
    assert len(selection) == 4
    assert len(test) == 6
    assert not train & selection
    assert not train & test
    assert not selection & test


def test_dataset_manifest_preserves_record_task_labels():
    records = [
        {"index": 4, "task": "drawer", "episode_index": 2},
        {"index": 9, "task": "stack", "episode_index": 7},
    ]

    manifest = build_dataset_manifest(
        records,
        split="selection",
        root_task="libero",
        protocol_seed=123,
    )

    assert manifest["tasks"] == [{"task": "libero", "indices": [4, 9]}]
    assert manifest["records"] == records
    assert manifest["split"] == "selection"


def test_aligned_teacher_path_indices_require_integer_compression_ratio():
    assert aligned_teacher_path_indices(teacher_steps=8, student_steps=4) == [0, 2, 4, 6, 8]

    try:
        aligned_teacher_path_indices(teacher_steps=3, student_steps=2)
    except ValueError as exc:
        assert "integer multiple" in str(exc)
    else:
        raise AssertionError("expected invalid teacher/student ratio to fail")


def test_cosmos_latent_shape_uses_official_prediction_horizon_not_dataset_cache():
    config = SimpleNamespace(
        cosmos_latent_channels=16,
        cosmos_latent_frames=9,
        cosmos_latent_height=28,
        cosmos_latent_width=28,
    )

    assert cosmos_latent_shape(config, batch_size=2) == (2, 16, 9, 28, 28)
