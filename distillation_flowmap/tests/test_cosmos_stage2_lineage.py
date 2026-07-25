import json
import os
import shutil
from pathlib import Path

import pytest

from distillation_flowmap.cosmos_stage2_lineage import (
    ResolvedCosmosInferenceCheckpoint,
    ValidatedStage1Parent,
    resolve_cosmos_inference_checkpoint,
    validate_stage1_parent,
    validate_stage2_path_isolation,
    validate_stage2_resume,
)


RAW_CONTRACT = {
    "contract_version": 2,
    "training_contract_stage": "raw_stage1",
    "action_packing_schema": "downsample_survivor_v2",
    "action_downsample_factor": 4,
    "action_chunk_shape": [4, 4],
    "teacher_backend": "cosmos_policy",
}
STAGE2_CONTRACT = {
    "contract_version": 2,
    "training_contract_stage": "progressive_stage2",
    "action_packing_schema": "downsample_survivor_v2",
    "action_downsample_factor": 4,
    "action_chunk_shape": [4, 4],
    "deployment_timestep_start": 1000,
    "deployment_timestep_end": 0,
    "joint_student_steps": [1, 2, 4],
    "deployment_joint_rollout_interval": 4,
    "deployment_action_weight": 1.0,
    "raw_teacher_window_is_auxiliary": True,
}


def _transformer(root: Path, contract: dict[str, object], step: object) -> Path:
    root.mkdir(parents=True)
    payload = {**contract, "checkpoint_step": step, "model_hint": "test"}
    (root / "config.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (root / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    return root


def _stage1(tmp_path: Path, *, name: str = "corrected-stage1") -> Path:
    root = tmp_path / name
    for variant in ("online_student", "target_student"):
        _transformer(root / variant / "transformer", RAW_CONTRACT, 5000)
    return root


def _stage2(tmp_path: Path, *, step: int = 1000) -> tuple[Path, Path]:
    arm_root = tmp_path / "stage2-arm"
    checkpoint = arm_root / "checkpoints" / f"step_{step}"
    for variant in ("online_student", "target_student"):
        _transformer(
            checkpoint / variant / "transformer",
            STAGE2_CONTRACT,
            step,
        )
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "lr_scheduler.pt").write_bytes(b"scheduler")
    return arm_root, checkpoint


def _config(root: Path, variant: str = "online_student") -> Path:
    return root / variant / "transformer" / "config.json"


def _weights(root: Path, variant: str = "online_student") -> Path:
    return root / variant / "transformer" / "diffusion_pytorch_model.safetensors"


def _rewrite(path: Path, **updates: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _make_sharded(
    transformer: Path,
    names: list[str],
    *,
    create: bool = True,
) -> Path:
    (transformer / "diffusion_pytorch_model.safetensors").unlink()
    if create:
        for name in names:
            shard = transformer / name
            shard.parent.mkdir(parents=True, exist_ok=True)
            shard.write_bytes(b"shard")
    index = transformer / "diffusion_pytorch_model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": {f"tensor_{i}": name for i, name in enumerate(names)}})
        + "\n",
        encoding="utf-8",
    )
    return index


def test_valid_stage1_parent_is_independent_and_identity_is_deterministic(tmp_path):
    root = _stage1(tmp_path)

    first = validate_stage1_parent(root)
    second = validate_stage1_parent(root)

    assert isinstance(first, ValidatedStage1Parent)
    assert first == second
    assert first.canonical_path == str(root.resolve())
    assert len(first.contract_identity) == 64
    assert first.contract_identity != "0" * 64


def test_stage1_identity_changes_with_canonical_parent_path(tmp_path):
    original = _stage1(tmp_path / "original")
    copied = tmp_path / "copied" / original.name
    shutil.copytree(original, copied)

    assert (
        validate_stage1_parent(original).contract_identity
        != validate_stage1_parent(copied).contract_identity
    )


@pytest.mark.parametrize("variant", ["online_student", "target_student"])
def test_stage1_identity_changes_with_either_exact_config_bytes(tmp_path, variant):
    root = _stage1(tmp_path)
    before = validate_stage1_parent(root).contract_identity
    config = _config(root, variant)
    config.write_text(config.read_text(encoding="utf-8") + " \n", encoding="utf-8")

    assert validate_stage1_parent(root).contract_identity != before


def test_stage1_identity_changes_with_validated_contract_payload(tmp_path):
    root = _stage1(tmp_path)
    before = validate_stage1_parent(root).contract_identity
    for variant in ("online_student", "target_student"):
        _rewrite(_config(root, variant), model_hint="changed validated payload")

    assert validate_stage1_parent(root).contract_identity != before


def test_stage1_identity_does_not_hash_weight_bytes(tmp_path):
    root = _stage1(tmp_path)
    before = validate_stage1_parent(root).contract_identity
    _weights(root, "online_student").write_bytes(b"changed online weight bytes")
    _weights(root, "target_student").write_bytes(b"changed target weight bytes")

    assert validate_stage1_parent(root).contract_identity == before


@pytest.mark.parametrize(
    ("variant", "field", "value"),
    [
        ("online_student", "contract_version", None),
        ("target_student", "action_packing_schema", "wrong"),
        ("online_student", "action_downsample_factor", True),
        ("target_student", "action_chunk_shape", [4, 8]),
        ("online_student", "checkpoint_step", "5000"),
    ],
)
def test_stage1_rejects_missing_wrong_or_type_invalid_fields(
    tmp_path, variant, field, value
):
    root = _stage1(tmp_path)
    config = _config(root, variant)
    payload = json.loads(config.read_text(encoding="utf-8"))
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value
    config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=field):
        validate_stage1_parent(root)


def test_stage1_rejects_contract_or_checkpoint_mismatch_between_variants(tmp_path):
    root = _stage1(tmp_path)
    _rewrite(_config(root, "target_student"), checkpoint_step=4000)

    with pytest.raises(ValueError, match="checkpoint"):
        validate_stage1_parent(root, expected_step=4000)


def test_stage1_rejects_validated_payload_mismatch_between_variants(tmp_path):
    root = _stage1(tmp_path)
    _rewrite(_config(root, "target_student"), model_hint="different")

    with pytest.raises(ValueError, match="contract payloads"):
        validate_stage1_parent(root)


@pytest.mark.parametrize("component", ["online_student", "target_student"])
def test_stage1_rejects_online_or_target_symlink(tmp_path, component):
    real = _stage1(tmp_path / "real")
    root = _stage1(tmp_path)
    link = root / component
    for child in sorted(link.rglob("*"), reverse=True):
        child.unlink() if child.is_file() else child.rmdir()
    link.rmdir()
    link.symlink_to(real / component, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        validate_stage1_parent(root)


def test_stage1_rejects_same_inode_alias(tmp_path):
    root = _stage1(tmp_path)
    target_config = _config(root, "target_student")
    target_config.unlink()
    os.link(_config(root, "online_student"), target_config)

    with pytest.raises(ValueError, match="same inode"):
        validate_stage1_parent(root)


def test_stage1_rejects_contaminated_name_and_root_symlink(tmp_path):
    contaminated = _stage1(tmp_path, name="raw_stage1_5000")
    with pytest.raises(ValueError, match="contaminated"):
        validate_stage1_parent(contaminated)

    alias = tmp_path / "stage1-alias"
    alias.symlink_to(_stage1(tmp_path / "plain"), target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        validate_stage1_parent(alias)


def test_stage1_rejects_symlink_in_ancestor_path(tmp_path):
    real_parent = tmp_path / "real-parent"
    root = _stage1(real_parent)
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        validate_stage1_parent(alias_parent / root.name)


def test_stage1_lstat_walk_does_not_normalize_away_symlink_before_dotdot(tmp_path):
    root = _stage1(tmp_path)
    hop = tmp_path / "hop"
    hop.mkdir()
    holder = tmp_path / "holder"
    holder.mkdir()
    alias = holder / "alias"
    alias.symlink_to(hop, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        validate_stage1_parent(alias / ".." / root.name)


def test_stage1_rejects_cross_layout_hardlink_alias(tmp_path):
    root = _stage1(tmp_path)
    online_weight = _weights(root, "online_student")
    target_transformer = root / "target_student" / "transformer"
    _weights(root, "target_student").unlink()
    target_shard = target_transformer / "renamed-target-shard.safetensors"
    os.link(online_weight, target_shard)
    (target_transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"tensor": target_shard.name}})
    )

    with pytest.raises(ValueError, match="same inode"):
        validate_stage1_parent(root)


def test_stage1_rejects_cross_name_sharded_hardlink_alias(tmp_path):
    root = _stage1(tmp_path)
    online_transformer = root / "online_student" / "transformer"
    target_transformer = root / "target_student" / "transformer"
    _make_sharded(online_transformer, ["online-a.safetensors", "online-b.safetensors"])
    _make_sharded(target_transformer, ["target-a.safetensors", "target-b.safetensors"])
    (target_transformer / "target-b.safetensors").unlink()
    os.link(
        online_transformer / "online-a.safetensors",
        target_transformer / "target-b.safetensors",
    )

    with pytest.raises(ValueError, match="same inode"):
        validate_stage1_parent(root)


@pytest.mark.parametrize("variant", ["online_student", "target_student"])
def test_stage1_accepts_strict_sharded_weights(tmp_path, variant):
    root = _stage1(tmp_path)
    _make_sharded(
        root / variant / "transformer",
        [
            "diffusion_pytorch_model-00001-of-00002.safetensors",
            "diffusion_pytorch_model-00002-of-00002.safetensors",
        ],
    )

    validate_stage1_parent(root)


@pytest.mark.parametrize("name", ["nested/shard.safetensors", "/tmp/shard", ".."])
def test_stage1_rejects_nested_absolute_or_dot_shard_name(tmp_path, name):
    root = _stage1(tmp_path)
    _make_sharded(root / "online_student" / "transformer", [name], create=False)

    with pytest.raises(ValueError, match="shard name"):
        validate_stage1_parent(root)


def test_stage1_rejects_non_string_shard_name_cleanly(tmp_path):
    root = _stage1(tmp_path)
    transformer = root / "online_student" / "transformer"
    (transformer / "diffusion_pytorch_model.safetensors").unlink()
    index = transformer / "diffusion_pytorch_model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"tensor": ["not", "a", "name"]}}))

    with pytest.raises(ValueError, match="shard name"):
        validate_stage1_parent(root)


@pytest.mark.parametrize("kind", ["missing", "symlink"])
def test_stage1_rejects_missing_or_symlink_shard(tmp_path, kind):
    root = _stage1(tmp_path)
    transformer = root / "online_student" / "transformer"
    index = _make_sharded(transformer, ["shard.safetensors"], create=False)
    if kind == "symlink":
        outside = tmp_path / "outside.safetensors"
        outside.write_bytes(b"outside")
        (transformer / "shard.safetensors").symlink_to(outside)

    with pytest.raises((FileNotFoundError, ValueError), match=kind):
        validate_stage1_parent(root)
    assert index.exists()


def test_stage2_path_isolation_accepts_disjoint_output(tmp_path):
    stage1 = _stage1(tmp_path)
    output = tmp_path / "stage2" / "s4"

    validate_stage2_path_isolation(
        stage1_root=stage1,
        output_dir=output,
        resume_checkpoint=None,
    )


@pytest.mark.parametrize("relation", ["equal", "nested", "contains"])
def test_stage2_path_isolation_rejects_equal_or_nested_paths(tmp_path, relation):
    stage1 = _stage1(tmp_path)
    if relation == "equal":
        output = stage1
    elif relation == "nested":
        output = stage1 / "stage2"
    else:
        output = tmp_path

    with pytest.raises(ValueError, match="Stage-1"):
        validate_stage2_path_isolation(
            stage1_root=stage1,
            output_dir=output,
            resume_checkpoint=None,
        )


def test_stage2_path_isolation_rejects_output_and_parent_symlink_alias(tmp_path):
    stage1 = _stage1(tmp_path)
    direct = tmp_path / "direct"
    direct.symlink_to(stage1, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        validate_stage2_path_isolation(
            stage1_root=stage1,
            output_dir=direct,
            resume_checkpoint=None,
        )

    parent = tmp_path / "alias"
    parent.symlink_to(stage1, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        validate_stage2_path_isolation(
            stage1_root=stage1,
            output_dir=parent / "new-output",
            resume_checkpoint=None,
        )


def test_stage2_path_walk_resumes_after_missing_component_is_popped(tmp_path):
    stage1 = _stage1(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    alias_target = tmp_path / "alias-target"
    alias_target.mkdir()
    alias = base / "alias"
    alias.symlink_to(alias_target, target_is_directory=True)
    prospective = base / "missing" / ".." / "alias" / "new"

    with pytest.raises(ValueError, match="symlink"):
        validate_stage2_path_isolation(
            stage1_root=stage1,
            output_dir=prospective,
            resume_checkpoint=None,
        )


def test_stage2_path_isolation_requires_resume_in_own_output_arm(tmp_path):
    stage1 = _stage1(tmp_path)
    _, resume = _stage2(tmp_path)

    with pytest.raises(ValueError, match="output arm"):
        validate_stage2_path_isolation(
            stage1_root=stage1,
            output_dir=tmp_path / "different-arm",
            resume_checkpoint=resume,
        )


def test_valid_stage2_resume_returns_canonical_checkpoint(tmp_path):
    arm, checkpoint = _stage2(tmp_path)

    assert validate_stage2_resume(
        checkpoint, arm_root=arm, expected_step=1000
    ) == checkpoint.resolve()


def test_stage2_resume_rejects_checkpoint_outside_arm(tmp_path):
    arm, _ = _stage2(tmp_path)
    _, outside = _stage2(tmp_path / "outside")

    with pytest.raises(ValueError, match="checkpoints"):
        validate_stage2_resume(outside, arm_root=arm, expected_step=1000)


def test_stage2_resume_rejects_symlink_in_ancestor_path(tmp_path):
    arm, checkpoint = _stage2(tmp_path / "real")
    alias = tmp_path / "arm-alias"
    alias.symlink_to(arm, target_is_directory=True)
    aliased_checkpoint = alias / checkpoint.relative_to(arm)

    with pytest.raises(ValueError, match="symlink"):
        validate_stage2_resume(
            aliased_checkpoint,
            arm_root=arm,
            expected_step=1000,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_step", 999),
        ("training_contract_stage", "raw_stage1"),
        ("deployment_action_weight", 0.0),
    ],
)
def test_stage2_resume_rejects_wrong_step_stage_or_deployment_field(
    tmp_path, field, value
):
    arm, checkpoint = _stage2(tmp_path)
    _rewrite(_config(checkpoint), **{field: value})

    with pytest.raises(ValueError, match=field):
        validate_stage2_resume(checkpoint, arm_root=arm, expected_step=1000)


@pytest.mark.parametrize("state", ["optimizer.pt", "lr_scheduler.pt"])
def test_stage2_resume_rejects_missing_state(tmp_path, state):
    arm, checkpoint = _stage2(tmp_path)
    (checkpoint / state).unlink()

    with pytest.raises(FileNotFoundError, match=state):
        validate_stage2_resume(checkpoint, arm_root=arm, expected_step=1000)


@pytest.mark.parametrize(
    "controlled",
    [
        "config",
        "weight",
        "optimizer.pt",
        "lr_scheduler.pt",
    ],
)
def test_stage2_resume_rejects_symlinked_config_weight_or_state(
    tmp_path, controlled
):
    arm, checkpoint = _stage2(tmp_path)
    if controlled == "config":
        path = _config(checkpoint)
    elif controlled == "weight":
        path = _weights(checkpoint)
    else:
        path = checkpoint / controlled
    outside = tmp_path / f"outside-{path.name}"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        validate_stage2_resume(checkpoint, arm_root=arm, expected_step=1000)


def _lineaged_stage2(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    stage1 = _stage1(tmp_path)
    cosmos_base = tmp_path / "cosmos-policy-base"
    cosmos_base.mkdir()
    (cosmos_base / "config.json").write_text("{}\n", encoding="utf-8")
    for variant in ("online_student", "target_student"):
        _rewrite(
            _config(stage1, variant),
            student_base_model_path=str(cosmos_base.resolve()),
        )
    parent = validate_stage1_parent(stage1)
    arm, checkpoint = _stage2(tmp_path)
    for variant in ("online_student", "target_student"):
        _rewrite(
            _config(checkpoint, variant),
            teacher_backend="cosmos_policy",
            parent_stage1_path=parent.canonical_path,
            parent_stage1_contract_identity=parent.contract_identity,
            student_base_model_path=str(stage1 / "target_student"),
        )
    return stage1, arm, checkpoint, cosmos_base


def test_inference_resolver_accepts_only_explicit_validated_student_roles(tmp_path):
    stage1, _arm, checkpoint, cosmos_base = _lineaged_stage2(tmp_path)

    stage1_target = resolve_cosmos_inference_checkpoint(
        model_role="stage1_target",
        checkpoint_transformer=stage1 / "target_student" / "transformer",
    )
    stage2_online = resolve_cosmos_inference_checkpoint(
        model_role="stage2_online",
        checkpoint_transformer=checkpoint / "online_student" / "transformer",
    )
    stage2_target = resolve_cosmos_inference_checkpoint(
        model_role="stage2_target",
        checkpoint_transformer=checkpoint / "target_student" / "transformer",
    )

    assert isinstance(stage1_target, ResolvedCosmosInferenceCheckpoint)
    assert stage1_target.training_stage == "raw_stage1"
    assert stage1_target.transformer_path == str(
        (stage1 / "target_student" / "transformer").resolve()
    )
    assert stage2_online.training_stage == "progressive_stage2"
    assert stage2_online.transformer_path == str(
        (checkpoint / "online_student" / "transformer").resolve()
    )
    assert stage2_target.parent_stage1_path == str(stage1.resolve())
    assert stage2_target.cosmos_base_model_path == str(cosmos_base.resolve())


def test_inference_resolver_rejects_role_component_mismatch_and_teacher(tmp_path):
    stage1, _arm, checkpoint, _cosmos_base = _lineaged_stage2(tmp_path)

    with pytest.raises(ValueError, match="stage1_target"):
        resolve_cosmos_inference_checkpoint(
            model_role="stage1_target",
            checkpoint_transformer=stage1 / "online_student" / "transformer",
        )
    with pytest.raises(ValueError, match="stage2_target"):
        resolve_cosmos_inference_checkpoint(
            model_role="stage2_target",
            checkpoint_transformer=checkpoint / "online_student" / "transformer",
        )
    with pytest.raises(ValueError, match="official_teacher.*matched-K"):
        resolve_cosmos_inference_checkpoint(
            model_role="official_teacher",
            checkpoint_transformer=checkpoint / "target_student" / "transformer",
        )


def test_inference_resolver_rejects_missing_or_mismatched_base_model_metadata(tmp_path):
    stage1, _arm, checkpoint, _cosmos_base = _lineaged_stage2(tmp_path)
    for variant in ("online_student", "target_student"):
        _rewrite(_config(stage1, variant), student_base_model_path="/root/nas/unknown")

    with pytest.raises((FileNotFoundError, ValueError), match="student_base_model_path"):
        resolve_cosmos_inference_checkpoint(
            model_role="stage1_target",
            checkpoint_transformer=stage1 / "target_student" / "transformer",
        )

    # Rebuild a valid pair and prove Stage-2 cannot silently point at another parent.
    stage1, _arm, checkpoint, _cosmos_base = _lineaged_stage2(tmp_path / "second")
    for variant in ("online_student", "target_student"):
        _rewrite(_config(checkpoint, variant), student_base_model_path="/wrong/stage1")
    with pytest.raises(ValueError, match="student_base_model_path"):
        resolve_cosmos_inference_checkpoint(
            model_role="stage2_target",
            checkpoint_transformer=checkpoint / "target_student" / "transformer",
        )
