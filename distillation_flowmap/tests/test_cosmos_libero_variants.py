"""RED contracts for the declarative Cosmos LIBERO Task-7 experiment matrix.

Copy this file to ``distillation_flowmap/tests/test_cosmos_libero_variants.py``
before implementing the resolver.  It deliberately imports a module that does
not exist at Task-6 HEAD: the first run must fail at collection time.
"""

import json
from pathlib import Path

import pytest

from distillation_flowmap.cosmos_libero_variants import (
    APM_VARIANTS_PATH,
    VARIANT_NAMES,
    VariantError,
    canonical_variant_json,
    load_video_apm_variants,
    resolve_variant,
    resolve_variants,
)


PROGRESSIVE_ARMS = ("s1", "s2", "s4", "universal", "universal-video-action")
APM_ARMS = ("stage1_only", "anchor_only", "field_only", "apm")
ALL_ARMS = PROGRESSIVE_ARMS + APM_ARMS
PROGRESSIVE_ALLOWED_DIFFERENCES = {
    "name",
    "progressive_stage",
    "max_train_steps",
    "master_port",
    "output_dir",
    "rollout_step_pairs",
    "danceopd_rollout_steps",
    "video_velocity_weight",
    "opd_rollout_grad_mode",
    "opd_rollout_grad_steps",
    "opd_endpoint_focus_prob",
}


def _resolve(tmp_path: Path, name: str, **overrides):
    return resolve_variant(
        name,
        output_root=tmp_path / "outputs",
        run_tag="task7-red",
        **overrides,
    )


def _plain(record):
    """The public immutable record is intentionally mapping-like."""
    return json.loads(canonical_variant_json(record))


def test_variant_catalogue_has_exactly_the_nine_declared_experiments():
    assert tuple(VARIANT_NAMES) == ALL_ARMS


@pytest.mark.parametrize("name", ALL_ARMS)
def test_every_variant_resolves_to_an_immutable_canonical_record(tmp_path, name):
    record = _resolve(tmp_path, name)
    payload = _plain(record)

    assert payload["name"] == name
    assert payload["output_dir"] == str(
        (tmp_path / "outputs" / "task7-red" / name).resolve()
    )
    assert payload["save_interval"] == 1000
    assert payload["run_tag"] == "task7-red"
    assert payload["train_seed"] == 42
    assert payload["learning_rate"] == pytest.approx(2e-7)
    assert payload["opd_aux_weight"] == pytest.approx(0.10)
    assert payload["opd_aux_warmup_steps"] == 8
    assert payload["opd_aux_prob"] == pytest.approx(1.0)
    assert payload["opd_danceopd_query_alpha"] == pytest.approx(5.0)
    assert payload["opd_danceopd_query_beta"] == pytest.approx(2.0)
    assert payload["tensorboard_enabled"] is True
    assert payload["wandb_mode"] == "offline"
    assert payload["enable_wandb"] is False
    assert payload["hf_offline"] is True
    assert payload["transformers_offline"] is True
    assert payload["hf_hub_offline"] is True
    assert canonical_variant_json(record) == json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )

    with pytest.raises((TypeError, AttributeError)):
        record["name"] = "mutated"
    with pytest.raises((TypeError, AttributeError)):
        record["rollout_step_pairs"].append((9, 9))


def test_progressive_arms_differ_only_in_declared_budget_fields(tmp_path):
    records = {name: _plain(_resolve(tmp_path, name)) for name in PROGRESSIVE_ARMS}

    expected = {
        "s1": {
            "progressive_stage": "s1",
            "max_train_steps": 3000,
            "master_port": 29663,
            "rollout_step_pairs": [[4, 1]],
            "danceopd_rollout_steps": [1],
            "video_endpoint_weight": 1.0,
            "video_velocity_weight": 0.0,
            "opd_rollout_grad_mode": "last_step",
            "opd_rollout_grad_steps": 1,
            "opd_endpoint_focus_prob": 0.90,
        },
        "s2": {
            "progressive_stage": "s2",
            "max_train_steps": 3000,
            "master_port": 29662,
            "rollout_step_pairs": [[4, 2]],
            "danceopd_rollout_steps": [2],
            "video_endpoint_weight": 1.0,
            "video_velocity_weight": 1.0,
            "opd_rollout_grad_mode": "last_step",
            "opd_rollout_grad_steps": 1,
            "opd_endpoint_focus_prob": 0.85,
        },
        "s4": {
            "progressive_stage": "s4",
            "max_train_steps": 5000,
            "master_port": 29661,
            "rollout_step_pairs": [[8, 4]],
            "danceopd_rollout_steps": [4],
            "video_endpoint_weight": 1.0,
            "video_velocity_weight": 1.0,
            "opd_rollout_grad_mode": "suffix",
            "opd_rollout_grad_steps": 2,
            "opd_endpoint_focus_prob": 0.80,
        },
        "universal": {
            "progressive_stage": "universal",
            "max_train_steps": 5000,
            "master_port": 29664,
            "rollout_step_pairs": [[8, 1], [8, 2], [8, 4]],
            "danceopd_rollout_steps": [2, 4],
            "video_endpoint_weight": 1.0,
            "video_velocity_weight": 1.0,
            "opd_rollout_grad_mode": "last_step",
            "opd_rollout_grad_steps": 1,
            "opd_endpoint_focus_prob": 0.85,
        },
        "universal-video-action": {
            "progressive_stage": "universal",
            "max_train_steps": 5000,
            "master_port": 29666,
            "rollout_step_pairs": [[8, 1], [8, 2], [8, 4]],
            "danceopd_rollout_steps": [2, 4],
            "video_endpoint_weight": 1.0,
            "video_velocity_weight": 1.0,
            "opd_rollout_grad_mode": "last_step",
            "opd_rollout_grad_steps": 1,
            "opd_endpoint_focus_prob": 0.85,
        },
    }
    for name, subset in expected.items():
        assert {key: records[name][key] for key in subset} == subset
        assert records[name]["action_endpoint_weight"] == 1.0
        assert records[name]["action_opd_enabled"] is False
        assert records[name]["use_opd_aux"] is True

    _assert_only_declared_progressive_differences(records.values())


def _assert_only_declared_progressive_differences(records):
    normalised = []
    for record in records:
        copy = dict(record)
        for key in PROGRESSIVE_ALLOWED_DIFFERENCES:
            copy.pop(key)
        normalised.append(copy)
    assert normalised.count(normalised[0]) == len(normalised)


def test_progressive_isolation_check_detects_an_undeclared_arm_specific_mutation(
    tmp_path,
):
    records = [_plain(_resolve(tmp_path, name)) for name in PROGRESSIVE_ARMS]
    _assert_only_declared_progressive_differences(records)
    records[1]["undeclared_scientific_toggle"] = True
    with pytest.raises(AssertionError):
        _assert_only_declared_progressive_differences(records)


def test_apm_arms_differ_only_in_video_endpoint_and_compositional_coefficients(
    tmp_path,
):
    records = {name: _plain(_resolve(tmp_path, name)) for name in APM_ARMS}
    expected_coefficients = {
        "stage1_only": (0.0, 0.0),
        "anchor_only": (1.0, 0.0),
        "field_only": (0.0, 1.0),
        "apm": (1.0, 1.0),
    }
    normalised = []
    for name, (endpoint, velocity) in expected_coefficients.items():
        record = records[name]
        assert (record["video_endpoint_weight"], record["video_velocity_weight"]) == (
            endpoint,
            velocity,
        )
        assert record["progressive_stage"] == "universal"
        assert record["rollout_step_pairs"] == [[8, 1], [8, 2], [8, 4]]
        assert record["danceopd_rollout_steps"] == [2, 4]
        assert record["action_endpoint_weight"] == 0.0
        assert record["action_opd_enabled"] is False
        copy = dict(record)
        for key in (
            "name",
            "output_dir",
            "master_port",
            "video_endpoint_weight",
            "video_velocity_weight",
            "use_opd_aux",
            "opd_aux_standalone_step",
        ):
            copy.pop(key)
        normalised.append(copy)
    assert normalised.count(normalised[0]) == len(normalised)


def test_stage1_only_disables_opd_scheduling_instead_of_entering_zero_zero_opd(
    tmp_path,
):
    payload = _plain(_resolve(tmp_path, "stage1_only"))
    assert payload["use_opd_aux"] is False
    assert payload["opd_aux_standalone_step"] is False
    assert payload["video_endpoint_weight"] == payload["video_velocity_weight"] == 0.0


def test_apm_json_is_the_declarative_source_and_has_no_unknown_fields(tmp_path):
    payload = json.loads(Path(APM_VARIANTS_PATH).read_text(encoding="utf-8"))
    assert tuple(item["name"] for item in payload["variants"]) == APM_ARMS
    loaded = load_video_apm_variants(APM_VARIANTS_PATH)
    assert tuple(loaded) == APM_ARMS

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps({"variants": [{"name": "apm", "endpoint": 1, "velocity": 1, "typo": 1}]}),
        encoding="utf-8",
    )
    with pytest.raises(VariantError, match="unknown|typo"):
        load_video_apm_variants(invalid)


@pytest.mark.parametrize(
    ("name", "overrides", "error"),
    [
        ("unknown", {}, "unknown"),
        ("apm", {"steps": True}, "steps"),
        ("apm", {"steps": 0}, "steps"),
        ("apm", {"steps": -1}, "steps"),
        ("apm", {"steps": 1.5}, "steps"),
        ("apm", {"save_interval": False}, "save interval"),
        ("apm", {"save_interval": 0}, "save interval"),
        ("apm", {"master_port": True}, "port"),
        ("apm", {"master_port": 65536}, "port"),
        ("apm", {"master_port": 0}, "port"),
        ("apm", {"run_tag": ""}, "run tag"),
        ("apm", {"run_tag": "../escape"}, "run tag"),
        ("apm", {"run_tag": "a/b"}, "run tag"),
    ],
)
def test_resolver_rejects_invalid_overrides(tmp_path, name, overrides, error):
    with pytest.raises(VariantError, match=error):
        resolve_variant(name, output_root=tmp_path / "runs", **overrides)


def test_resolver_rejects_duplicate_output_port_and_manifest_records(tmp_path):
    with pytest.raises(VariantError, match="duplicate.*name"):
        resolve_variants(
            ("s4", "s4"), output_root=tmp_path / "outputs", run_tag="task7-red"
        )
    with pytest.raises(VariantError, match="duplicate.*port"):
        resolve_variants(
            ("s4", "universal"),
            output_root=tmp_path / "outputs",
            run_tag="task7-red",
            master_ports={"s4": 29661, "universal": 29661},
        )
    with pytest.raises(VariantError, match="duplicate.*output"):
        resolve_variants(
            ("s4", "universal"),
            output_root=tmp_path / "outputs",
            run_tag="task7-red",
            output_dirs={"s4": tmp_path / "same", "universal": tmp_path / "same"},
        )

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        json.dumps(
            {
                "variants": [
                    {"name": "apm", "endpoint": 1, "velocity": 1},
                    {"name": "apm", "endpoint": 0, "velocity": 0},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(VariantError, match="duplicate.*name"):
        load_video_apm_variants(duplicate)
