import ast
import json
from pathlib import Path

import pytest

import distillation_flowmap.cosmos_progressive_env_schema as env_schema
from distillation_flowmap.cosmos_libero_variants import (
    canonical_variant_json,
    resolve_variant,
)
from distillation_flowmap.cosmos_progressive_env_schema import (
    CANONICAL_ENV,
    CANONICAL_ENV_TO_JSON_FIELDS,
    CLEARED_LEGACY_ENV,
    CONFIG_CHAIN,
    DYNAMIC_ENV_SITES,
    PINNED_OPERATIONAL_ENV,
    EnvSchemaError,
    extract_config_env_reads,
    launcher_action_manifest,
    resolve_stage1_expected_step,
    validate_config_chain,
    validate_environment_contract,
    validate_launcher_environment,
)


ROOT = Path(__file__).resolve().parents[2]


def test_exact_six_layer_chain_and_every_literal_read_is_classified_once():
    validate_config_chain(ROOT)
    reads, dynamic = extract_config_env_reads(
        [ROOT / relative for relative in CONFIG_CHAIN]
    )
    actual = {read.name for read in reads}
    classified = CANONICAL_ENV | PINNED_OPERATIONAL_ENV | CLEARED_LEGACY_ENV
    assert actual == classified
    assert not (
        (CANONICAL_ENV & PINNED_OPERATIONAL_ENV)
        | (CANONICAL_ENV & CLEARED_LEGACY_ENV)
        | (PINNED_OPERATIONAL_ENV & CLEARED_LEGACY_ENV)
    )
    assert {site.fingerprint for site in dynamic} == set(DYNAMIC_ENV_SITES)
    validate_environment_contract(ROOT)


def test_new_parent_environment_read_is_a_schema_failure(tmp_path):
    copies = []
    for relative in CONFIG_CHAIN:
        source = ROOT / relative
        destination = tmp_path / Path(relative).name
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        copies.append(destination)
    copies[-1].write_text(
        copies[-1].read_text(encoding="utf-8")
        + '\ncfg.new_loss = float(os.environ.get("NEW_LOSS_WEIGHT", 1.0))\n',
        encoding="utf-8",
    )
    reads, _ = extract_config_env_reads(copies)
    actual = {read.name for read in reads}
    classified = CANONICAL_ENV | PINNED_OPERATIONAL_ENV | CLEARED_LEGACY_ENV
    assert actual - classified == {"NEW_LOSS_WEIGHT"}
    with pytest.raises(EnvSchemaError, match="NEW_LOSS_WEIGHT"):
        validate_environment_contract(ROOT, files=copies, check_chain=False)


def test_scanner_expands_helpers_and_literal_loops(tmp_path):
    source = tmp_path / "synthetic.py"
    source.write_text(
        """
import os as system
NAMES = ("LOOP_A", "LOOP_B")
def _env_float(name, default):
    return float(system.environ.get(name, default))
value = _env_float("HELPER_WEIGHT", 1.0)
for name in NAMES:
    print(system.environ[name])
""",
        encoding="utf-8",
    )
    reads, dynamic = extract_config_env_reads([source])
    assert {read.name for read in reads} == {"HELPER_WEIGHT", "LOOP_A", "LOOP_B"}
    assert dynamic == ()


def test_unreviewed_dynamic_environment_access_fails(tmp_path):
    source = tmp_path / "dynamic.py"
    source.write_text(
        """
import os
prefix = "LOSS_"
suffix = input()
value = os.environ.get(prefix + suffix)
""",
        encoding="utf-8",
    )
    reads, dynamic = extract_config_env_reads([source])
    assert not reads
    assert len(dynamic) == 1
    assert dynamic[0].fingerprint not in DYNAMIC_ENV_SITES


def test_every_canonical_read_has_an_identity_mapping():
    assert set(CANONICAL_ENV_TO_JSON_FIELDS) == CANONICAL_ENV
    assert all(CANONICAL_ENV_TO_JSON_FIELDS[name] for name in CANONICAL_ENV)
    provenance = {
        "dataset": {"root": "/dataset"},
        "teacher": {"root": "/teacher"},
        "video_vae": {"root": "/video-vae"},
        "local_model": {"root": "/local-model"},
        "cosmos_repo": {"root": "/cosmos"},
        "worker": {
            "python_realpath": "/python",
            "extra_pythonpath": ["/extra"],
        },
    }
    variant = json.loads(
        canonical_variant_json(
            resolve_variant(
                "apm",
                output_root="/output",
                provenance=provenance,
            )
        )
    )
    recovery = {
        "output_dir": "/output/default/apm",
        "output_root": "/output",
        "resume_from_path": "/stage1",
        "resume_online_from_target": True,
        "reset_resume_step": True,
        "resume_optimizer_state": False,
        "parent_stage1_path": "/stage1",
        "parent_stage1_contract_identity": "identity",
        "stage2_lineage": {},
        "student_base_model_path": "/stage1/target_student",
    }
    surface = {"variant": variant, "recovery": recovery}
    for name, paths in CANONICAL_ENV_TO_JSON_FIELDS.items():
        for path in paths:
            value = surface
            for component in path.split("."):
                value = value[component]


def test_launcher_runtime_validator_requires_set_and_unset_actions():
    expected_values = getattr(env_schema, "PINNED_ENV_EXPECTED_VALUES", {})
    environment = {
        name: "sealed" for name in CANONICAL_ENV
    }
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "SAVE_INTERVAL": "1000",
            "COSMOS_STAGE1_EXPECTED_STEP": "5000",
        }
    )
    for name, expected in expected_values.items():
        if expected[0] == "environment":
            environment[name] = environment[expected[1]]
        elif expected[0] == "stage1_expected_step":
            environment[name] = "5000"
        else:
            environment[name] = expected[1]
    validate_launcher_environment(environment)
    actions = launcher_action_manifest(environment)
    assert actions["USE_FSDP1"] == {
        "action": "set_pinned",
        "value": "1",
    }
    assert actions["WANDB_MODE"] == {
        "action": "set_pinned",
        "value": "offline",
    }
    assert actions["DISTILL_MODE"] == {"action": "unset_legacy"}
    missing = dict(environment)
    missing.pop(next(iter(CANONICAL_ENV)))
    with pytest.raises(EnvSchemaError, match="missing"):
        validate_launcher_environment(missing)
    hostile = dict(environment)
    hostile["USE_FSDP1"] = "0"
    with pytest.raises(EnvSchemaError, match="USE_FSDP1|pinned|expected"):
        validate_launcher_environment(hostile)
    leaked = dict(environment)
    leaked[next(iter(CLEARED_LEGACY_ENV))] = "hostile"
    with pytest.raises(EnvSchemaError, match="cleared|legacy"):
        validate_launcher_environment(leaked)


def test_stage1_expected_step_is_pinned_as_a_positive_decimal_contract():
    expected_values = getattr(env_schema, "PINNED_ENV_EXPECTED_VALUES", {})
    environment = {name: "sealed" for name in CANONICAL_ENV}
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "SAVE_INTERVAL": "1000",
            "COSMOS_STAGE1_EXPECTED_STEP": "3000",
        }
    )
    for name, expected in expected_values.items():
        if expected[0] == "environment":
            environment[name] = environment[expected[1]]
        elif expected[0] == "stage1_expected_step":
            environment[name] = "3000"
        else:
            environment[name] = expected[1]

    actions = launcher_action_manifest(environment)

    assert "COSMOS_STAGE1_EXPECTED_STEP" in PINNED_OPERATIONAL_ENV
    assert actions["COSMOS_STAGE1_EXPECTED_STEP"] == {
        "action": "set_pinned",
        "value": "3000",
    }
    validate_launcher_environment(environment, actions=actions)
    for invalid in ("0", "-1", "3e3"):
        invalid_environment = dict(environment)
        invalid_environment["COSMOS_STAGE1_EXPECTED_STEP"] = invalid
        with pytest.raises(EnvSchemaError, match="COSMOS_STAGE1_EXPECTED_STEP|positive"):
            launcher_action_manifest(invalid_environment)

    drifted_actions = dict(actions)
    drifted_actions["COSMOS_STAGE1_EXPECTED_STEP"] = {
        "action": "set_pinned",
        "value": "5000",
    }
    with pytest.raises(EnvSchemaError, match="action manifest"):
        validate_launcher_environment(environment, actions=drifted_actions)


def test_stage1_expected_step_resolver_defaults_and_rejects_non_decimal_input():
    assert resolve_stage1_expected_step(None) == 5000
    assert resolve_stage1_expected_step("3000") == 3000
    for invalid in ("0", "-1", "+3000", " 3000", "3000 ", "3e3"):
        with pytest.raises(EnvSchemaError, match="COSMOS_STAGE1_EXPECTED_STEP"):
            resolve_stage1_expected_step(invalid)


def test_newly_classified_pinned_read_requires_an_exact_action_spec(
    tmp_path, monkeypatch
):
    expected_values = getattr(env_schema, "PINNED_ENV_EXPECTED_VALUES", {})
    source = tmp_path / "synthetic.py"
    source.write_text(
        'import os\nvalue = os.environ.get("NEW_PINNED", "1")\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        env_schema,
        "PINNED_OPERATIONAL_ENV",
        PINNED_OPERATIONAL_ENV | {"NEW_PINNED"},
    )
    validate_environment_contract(
        ROOT,
        files=[ROOT / relative for relative in CONFIG_CHAIN] + [source],
        check_chain=False,
    )
    hostile = {
        name: "sealed" for name in CANONICAL_ENV
    }
    hostile.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "SAVE_INTERVAL": "1000",
            "NEW_PINNED": "1",
            "COSMOS_STAGE1_EXPECTED_STEP": "5000",
        }
    )
    for name, expected in expected_values.items():
        if expected[0] == "environment":
            hostile[name] = hostile[expected[1]]
        elif expected[0] == "stage1_expected_step":
            hostile[name] = "5000"
        else:
            hostile[name] = expected[1]
    with pytest.raises(EnvSchemaError, match="NEW_PINNED|action|spec"):
        launcher_action_manifest(hostile)

    monkeypatch.setattr(
        env_schema,
        "PINNED_ENV_EXPECTED_VALUES",
        {**expected_values, "NEW_PINNED": ("literal", "1")},
        raising=False,
    )
    assert launcher_action_manifest(hostile)["NEW_PINNED"] == {
        "action": "set_pinned",
        "value": "1",
    }
