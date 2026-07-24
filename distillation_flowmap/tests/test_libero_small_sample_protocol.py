import json

from distillation_flowmap.ablation.libero_small_sample_protocol import (
    build_libero_task0_protocol,
    write_libero_task0_manifests,
)


TASK_LANGUAGE = "put both the alphabet soup and the tomato sauce in the basket"


def test_task0_protocol_is_frozen_to_40_train_and_10_heldout():
    protocol = build_libero_task0_protocol()

    assert protocol["benchmark"] == "libero_10"
    assert protocol["task_index"] == 0
    assert protocol["task_language"] == TASK_LANGUAGE
    assert protocol["train_indices"] == list(range(40))
    assert protocol["heldout_indices"] == list(range(40, 50))


def test_manifests_target_the_single_libero_repository(tmp_path):
    paths = write_libero_task0_manifests(tmp_path)

    train = json.loads(paths["train"].read_text())
    heldout = json.loads(paths["heldout"].read_text())
    assert train["tasks"] == [{"task": "libero", "indices": list(range(40))}]
    assert heldout["tasks"] == [
        {"task": "libero", "indices": list(range(40, 50))}
    ]
    assert train["protocol"]["task_language"] == TASK_LANGUAGE
    assert heldout["protocol"]["task_language"] == TASK_LANGUAGE


def test_existing_manifest_must_match_frozen_protocol(tmp_path):
    paths = write_libero_task0_manifests(tmp_path)
    train = json.loads(paths["train"].read_text())
    train["tasks"][0]["indices"] = [0]
    paths["train"].write_text(json.dumps(train))

    try:
        write_libero_task0_manifests(tmp_path)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mismatched existing manifest was accepted")
