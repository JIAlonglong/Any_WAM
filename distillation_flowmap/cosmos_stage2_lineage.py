from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping

from distillation_flowmap.cosmos_training_contract import validate_contract_metadata
from distillation_flowmap.cosmos_hybrid_backend import (
    STUDENT_BACKEND,
    TEACHER_BACKEND,
    validate_cosmos_teacher_model_path,
    validate_wan_student_base_model_path,
)


_MONOLITHIC_WEIGHTS = "diffusion_pytorch_model.safetensors"
_SHARDED_INDEX = "diffusion_pytorch_model.safetensors.index.json"


@dataclass(frozen=True)
class ValidatedStage1Parent:
    canonical_path: str
    contract_identity: str
    checkpoint_step: int


@dataclass(frozen=True)
class ResolvedCosmosInferenceCheckpoint:
    """A student checkpoint whose role and ancestry have been verified."""

    model_role: str
    training_stage: str
    transformer_path: str
    checkpoint_path: str
    parent_stage1_path: str | None
    parent_stage1_expected_step: int | None
    student_backend: str
    teacher_backend: str
    wan_student_base_model_path: str
    cosmos_teacher_model_path: str
    checkpoint_contract_identity: str


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = path if path.is_absolute() else Path.cwd() / path
    anchor = Path(absolute.anchor)
    current = anchor
    unresolved: list[str] = []
    if stat.S_ISLNK(os.lstat(anchor).st_mode):
        raise ValueError(f"{label} filesystem anchor must not be a symlink: {anchor}")
    for part in absolute.parts[1:]:
        if part in ("", "."):
            continue
        if part == "..":
            if unresolved:
                unresolved.pop()
            elif current != anchor:
                current = current.parent
            continue
        if unresolved:
            unresolved.append(part)
            continue
        candidate = current / part
        try:
            component_stat = os.lstat(candidate)
        except FileNotFoundError:
            unresolved.append(part)
            continue
        if stat.S_ISLNK(component_stat.st_mode):
            raise ValueError(f"{label} path contains a symlink component: {candidate}")
        current = candidate


def _require_plain_directory(path: Path, *, label: str) -> None:
    _reject_symlink_components(path, label=label)
    if not path.exists():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if not path.is_dir():
        raise ValueError(f"{label} must be a directory: {path}")


def _require_plain_file(path: Path, *, label: str) -> None:
    _reject_symlink_components(path, label=label)
    if not path.exists():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")


def _read_json_object(path: Path, *, label: str) -> tuple[bytes, dict[str, object]]:
    _require_plain_file(path, label=label)
    exact = path.read_bytes()
    payload = json.loads(exact)
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return exact, payload


def _validate_step(
    payload: dict[str, object], *, expected_step: int, label: str
) -> None:
    if "checkpoint_step" not in payload:
        raise ValueError(f"{label} is missing checkpoint_step")
    actual = payload["checkpoint_step"]
    if type(actual) is not int or actual != expected_step:
        raise ValueError(
            f"{label} checkpoint_step must be exactly {expected_step!r} (int), "
            f"got {actual!r} ({type(actual).__name__})"
        )


def _contract_payload(payload: dict[str, object]) -> dict[str, object]:
    return {
        field: value
        for field, value in payload.items()
        if field != "checkpoint_step"
    }


def _validate_weights(transformer: Path, *, label: str) -> tuple[Path, ...]:
    monolithic = transformer / _MONOLITHIC_WEIGHTS
    if monolithic.is_symlink():
        raise ValueError(f"{label} weights must not be a symlink: {monolithic}")
    if monolithic.exists():
        _require_plain_file(monolithic, label=f"{label} weights")
        return (monolithic,)

    index = transformer / _SHARDED_INDEX
    _, payload = _read_json_object(index, label=f"{label} sharded weight index")
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index} must contain a nonempty weight_map")

    shards: list[Path] = []
    seen_names: set[str] = set()
    for raw_name in weight_map.values():
        if (
            not isinstance(raw_name, str)
            or raw_name in ("", ".", "..")
            or "/" in raw_name
            or "\\" in raw_name
            or os.path.isabs(raw_name)
            or Path(raw_name).name != raw_name
        ):
            raise ValueError(f"invalid shard name in {index}: {raw_name!r}")
        if raw_name in seen_names:
            continue
        seen_names.add(raw_name)
        shard = transformer / raw_name
        if shard.is_symlink():
            raise ValueError(f"symlink shard is forbidden: {shard}")
        if not shard.exists():
            raise FileNotFoundError(f"missing declared transformer shard: {shard}")
        if not shard.is_file():
            raise ValueError(f"declared transformer shard is not regular: {shard}")
        shards.append(shard)
    return (index, *sorted(shards))


def _inode_map(paths: tuple[Path, ...]) -> dict[tuple[int, int], Path]:
    result: dict[tuple[int, int], Path] = {}
    for path in paths:
        path_stat = path.stat()
        result[(path_stat.st_dev, path_stat.st_ino)] = path
    return result


def _canonical_json_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_stage1_parent(
    path: Path, *, expected_step: int = 5000
) -> ValidatedStage1Parent:
    path = Path(path)
    _require_plain_directory(path, label="Stage-1 root")
    canonical = path.resolve(strict=True)
    if path.name == "raw_stage1_5000" or canonical.name == "raw_stage1_5000":
        raise ValueError(f"known contaminated Stage-1 root is forbidden: {path}")
    if type(expected_step) is not int:
        raise TypeError("expected_step must be a plain integer")

    records: dict[str, tuple[Path, bytes, dict[str, object], dict[str, object]]] = {}
    controlled: dict[str, tuple[Path, ...]] = {}
    for variant in ("online_student", "target_student"):
        component = path / variant
        transformer = component / "transformer"
        _require_plain_directory(component, label=variant)
        _require_plain_directory(transformer, label=f"{variant}/transformer")
        config = transformer / "config.json"
        exact, payload = _read_json_object(config, label=f"{variant} config.json")
        validate_contract_metadata(payload, required_stage="raw_stage1")
        if payload.get("teacher_backend") != "cosmos_policy":
            raise ValueError(
                f"{variant} teacher_backend must be exactly 'cosmos_policy'"
            )
        if payload.get("student_backend") != STUDENT_BACKEND:
            raise ValueError(
                f"{variant} student_backend must be exactly {STUDENT_BACKEND!r}"
            )
        _validate_step(payload, expected_step=expected_step, label=variant)
        contract = _contract_payload(payload)
        weights = _validate_weights(transformer, label=variant)
        records[variant] = (config, exact, payload, contract)
        controlled[variant] = (component, transformer, config, *weights)

    online_inodes = _inode_map(controlled["online_student"])
    target_inodes = _inode_map(controlled["target_student"])
    shared_inodes = online_inodes.keys() & target_inodes.keys()
    if shared_inodes:
        shared = next(iter(shared_inodes))
        raise ValueError(
            "online_student and target_student must not share the same inode: "
            f"{online_inodes[shared]} and {target_inodes[shared]}"
        )

    online_payload = records["online_student"][2]
    target_payload = records["target_student"][2]
    if online_payload["checkpoint_step"] != target_payload["checkpoint_step"]:
        raise ValueError("online and target checkpoint steps must match")
    online_contract = records["online_student"][3]
    target_contract = records["target_student"][3]
    if online_contract != target_contract:
        raise ValueError("online and target contract payloads must match")

    identity_payload = {
        "canonical_stage1_path": str(canonical),
        "online_config_sha256": hashlib.sha256(
            records["online_student"][1]
        ).hexdigest(),
        "target_config_sha256": hashlib.sha256(
            records["target_student"][1]
        ).hexdigest(),
        "online_contract_sha256": _canonical_json_digest(online_contract),
        "target_contract_sha256": _canonical_json_digest(target_contract),
    }
    return ValidatedStage1Parent(
        canonical_path=str(canonical),
        contract_identity=_canonical_json_digest(identity_payload),
        checkpoint_step=expected_step,
    )


def validate_stage1_resume_hybrid_lineage(
    path: Path,
    *,
    expected_step: int,
    current_wan_student_base: str | Path,
    current_cosmos_teacher: str | Path,
) -> ValidatedStage1Parent:
    """Bind a Stage-1 resume checkpoint to the exact current hybrid roots."""

    parent = validate_stage1_parent(path, expected_step=expected_step)
    checkpoint_wan, checkpoint_teacher = _validated_hybrid_model_paths(parent)
    current_wan = str(
        validate_wan_student_base_model_path(current_wan_student_base)
    )
    current_teacher = str(
        validate_cosmos_teacher_model_path(current_cosmos_teacher)
    )
    if current_wan != checkpoint_wan:
        raise ValueError(
            "current Wan Student base does not match checkpoint provenance: "
            f"{current_wan} != {checkpoint_wan}"
        )
    if current_teacher != checkpoint_teacher:
        raise ValueError(
            "current Cosmos Teacher root does not match checkpoint provenance: "
            f"{current_teacher} != {checkpoint_teacher}"
        )
    return parent


def validate_stage2_resume(
    checkpoint: Path,
    *,
    arm_root: Path,
    expected_step: int,
    expected_parent: ValidatedStage1Parent | None = None,
) -> Path:
    checkpoint = Path(checkpoint)
    arm_root = Path(arm_root)
    if type(expected_step) is not int:
        raise TypeError("expected_step must be a plain integer")
    _require_plain_directory(checkpoint, label="Stage-2 checkpoint")
    canonical_checkpoint = checkpoint.resolve(strict=True)
    canonical_checkpoints_root = (arm_root.resolve(strict=False) / "checkpoints").resolve(
        strict=False
    )
    try:
        relative = canonical_checkpoint.relative_to(canonical_checkpoints_root)
    except ValueError as exc:
        raise ValueError(
            f"Stage-2 checkpoint must be below {canonical_checkpoints_root}"
        ) from exc
    if not relative.parts:
        raise ValueError(
            f"Stage-2 checkpoint must be exactly below {canonical_checkpoints_root}"
        )

    payloads: dict[str, dict[str, object]] = {}
    controlled: dict[str, tuple[Path, ...]] = {}
    for variant in ("online_student", "target_student"):
        component = checkpoint / variant
        transformer = component / "transformer"
        config = transformer / "config.json"
        _require_plain_directory(component, label=variant)
        _require_plain_directory(transformer, label=f"{variant}/transformer")
        _, payload = _read_json_object(
            config, label=f"{variant} config.json"
        )
        validate_contract_metadata(payload, required_stage="progressive_stage2")
        if payload.get("student_backend") != STUDENT_BACKEND:
            raise ValueError(
                f"{variant} student_backend must be exactly {STUDENT_BACKEND!r}"
            )
        if payload.get("teacher_backend") != TEACHER_BACKEND:
            raise ValueError(
                f"{variant} teacher_backend must be exactly {TEACHER_BACKEND!r}"
            )
        _validate_step(payload, expected_step=expected_step, label=variant)
        weights = _validate_weights(transformer, label=variant)
        payloads[variant] = payload
        controlled[variant] = (component, transformer, config, *weights)
        if expected_parent is not None:
            if payload.get("parent_stage1_path") != expected_parent.canonical_path:
                raise ValueError(
                    f"{variant} parent_stage1_path does not match validated parent"
                )
            if (
                payload.get("parent_stage1_contract_identity")
                != expected_parent.contract_identity
            ):
                raise ValueError(
                    f"{variant} parent_stage1_contract_identity does not match "
                    "validated parent"
                )
            _, expected_teacher = _validated_hybrid_model_paths(expected_parent)
            configured_teacher = payload.get("teacher_model_path")
            if not isinstance(configured_teacher, str) or not configured_teacher:
                raise ValueError(
                    f"{variant} teacher_model_path must identify the Stage-1 Teacher"
                )
            actual_teacher = str(
                validate_cosmos_teacher_model_path(configured_teacher)
            )
            if actual_teacher != expected_teacher:
                raise ValueError(
                    f"{variant} teacher_model_path does not match validated Stage-1 "
                    "Teacher"
                )
    sealed_parent_steps: dict[str, int | None] = {}
    for variant, payload in payloads.items():
        if "parent_stage1_expected_step" not in payload:
            sealed_parent_steps[variant] = None
            continue
        raw_step = payload["parent_stage1_expected_step"]
        if type(raw_step) is not int or raw_step <= 0:
            raise ValueError(
                f"{variant} parent_stage1_expected_step must be a positive integer"
            )
        sealed_parent_steps[variant] = raw_step
    present_parent_steps = [
        step for step in sealed_parent_steps.values() if step is not None
    ]
    if present_parent_steps:
        if len(present_parent_steps) != len(sealed_parent_steps):
            raise ValueError(
                "online and target parent_stage1_expected_step fields must match"
            )
        if len(set(present_parent_steps)) != 1:
            raise ValueError(
                "online and target parent_stage1_expected_step fields must match"
            )
        sealed_parent_step = present_parent_steps[0]
        if (
            expected_parent is not None
            and sealed_parent_step != expected_parent.checkpoint_step
        ):
            raise ValueError(
                "parent_stage1_expected_step must match the validated Stage-1 step"
            )
    elif expected_parent is not None and expected_parent.checkpoint_step != 5000:
        raise ValueError(
            "legacy checkpoint without parent_stage1_expected_step is compatible "
            "only with validated Stage-1 step 5000"
        )
    if _contract_payload(payloads["online_student"]) != _contract_payload(
        payloads["target_student"]
    ):
        raise ValueError("online and target Stage-2 contract payloads must match")
    shared = _inode_map(controlled["online_student"]).keys() & _inode_map(
        controlled["target_student"]
    ).keys()
    if shared:
        raise ValueError("online_student and target_student must be independent")
    _require_plain_file(checkpoint / "optimizer.pt", label="optimizer.pt")
    _require_plain_file(checkpoint / "lr_scheduler.pt", label="lr_scheduler.pt")
    return canonical_checkpoint


def _reject_path_symlinks(path: Path, *, label: str) -> None:
    _reject_symlink_components(path, label=label)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_stage2_path_isolation(
    *,
    stage1_root: Path,
    output_dir: Path,
    resume_checkpoint: Path | None,
) -> None:
    _reject_path_symlinks(Path(output_dir), label="Stage-2 output")
    stage1 = Path(stage1_root).resolve(strict=False)
    output = Path(output_dir).resolve(strict=False)
    if output == stage1 or _is_within(output, stage1) or _is_within(stage1, output):
        raise ValueError(
            f"Stage-2 output and Stage-1 root must be isolated: {output} vs {stage1}"
        )
    if resume_checkpoint is not None:
        resume = Path(resume_checkpoint).resolve(strict=False)
        if not _is_within(resume, output) or resume == output:
            raise ValueError(
                f"resume checkpoint must remain within its own output arm: {output}"
            )


def resolve_cosmos_inference_checkpoint(
    *, model_role: str, checkpoint_transformer: Path
) -> ResolvedCosmosInferenceCheckpoint:
    """Resolve a model role to an independently validated student checkpoint.

    This is deliberately called before constructing a model.  It makes the
    variant (Stage-1 target, Stage-2 online, Stage-2 target) explicit and
    rejects the official teacher until a real matched-K adapter exists.
    """

    allowed = {
        "stage1_target": ("raw_stage1", "target_student"),
        "stage2_online": ("progressive_stage2", "online_student"),
        "stage2_target": ("progressive_stage2", "target_student"),
    }
    if model_role == "official_teacher":
        raise ValueError(
            "official_teacher matched-K inference is unavailable: no verified "
            "Cosmos matched-K adapter is configured"
        )
    if model_role not in allowed:
        raise ValueError(f"unsupported Cosmos inference model_role: {model_role!r}")

    expected_stage, expected_variant = allowed[model_role]
    transformer = Path(checkpoint_transformer)
    _require_plain_directory(transformer, label="inference transformer")
    if transformer.name != "transformer" or transformer.parent.name != expected_variant:
        raise ValueError(
            f"{model_role} must reference its {expected_variant}/transformer directory"
        )
    canonical_transformer = transformer.resolve(strict=True)

    if expected_stage == "raw_stage1":
        stage1_root = transformer.parent.parent
        parent = validate_stage1_parent(stage1_root, expected_step=5000)
        wan_base, cosmos_teacher = _validated_hybrid_model_paths(parent)
        expected_transformer = (
            Path(parent.canonical_path) / "target_student" / "transformer"
        )
        if canonical_transformer != expected_transformer:
            raise ValueError("stage1_target does not match the validated Stage-1 target")
        return ResolvedCosmosInferenceCheckpoint(
            model_role=model_role,
            training_stage=expected_stage,
            transformer_path=str(canonical_transformer),
            checkpoint_path=parent.canonical_path,
            parent_stage1_path=None,
            parent_stage1_expected_step=None,
            student_backend=STUDENT_BACKEND,
            teacher_backend=TEACHER_BACKEND,
            wan_student_base_model_path=wan_base,
            cosmos_teacher_model_path=cosmos_teacher,
            checkpoint_contract_identity=parent.contract_identity,
        )

    checkpoint = transformer.parent.parent
    _, payload = _read_json_object(
        transformer / "config.json", label="inference transformer config.json"
    )
    step = payload.get("checkpoint_step")
    if type(step) is not int:
        raise ValueError("inference transformer checkpoint_step must be a plain integer")
    parent_path = payload.get("parent_stage1_path")
    if not isinstance(parent_path, str) or not parent_path:
        raise ValueError("Stage-2 inference checkpoint is missing parent_stage1_path")
    parent_step = payload.get("parent_stage1_expected_step", 5000)
    if type(parent_step) is not int or parent_step <= 0:
        raise ValueError(
            "Stage-2 inference checkpoint parent_stage1_expected_step must be "
            "a positive integer"
        )
    parent = validate_stage1_parent(Path(parent_path), expected_step=parent_step)
    wan_base, cosmos_teacher = _validated_hybrid_model_paths(parent)
    arm_root = checkpoint.parent.parent
    canonical_checkpoint = validate_stage2_resume(
        checkpoint,
        arm_root=arm_root,
        expected_step=step,
        expected_parent=parent,
    )
    checkpoint_configs: dict[str, dict[str, object]] = {}
    for variant in ("online_student", "target_student"):
        _, variant_payload = _read_json_object(
            canonical_checkpoint / variant / "transformer" / "config.json",
            label=f"{variant} inference config.json",
        )
        if variant_payload.get("teacher_backend") != "cosmos_policy":
            raise ValueError(
                f"{variant} teacher_backend must be exactly 'cosmos_policy'"
            )
        checkpoint_configs[variant] = variant_payload
        expected_stage1_student = Path(parent.canonical_path) / "target_student"
        configured_student = variant_payload.get("student_base_model_path")
        if (
            not isinstance(configured_student, str)
            or Path(configured_student).resolve(strict=False)
            != expected_stage1_student.resolve(strict=True)
        ):
            raise ValueError(
                f"{variant} student_base_model_path must exactly reference the "
                "validated Stage-1 target_student"
            )
    return ResolvedCosmosInferenceCheckpoint(
        model_role=model_role,
        training_stage=expected_stage,
        transformer_path=str(canonical_transformer),
        checkpoint_path=str(canonical_checkpoint),
        parent_stage1_path=parent.canonical_path,
        parent_stage1_expected_step=parent_step,
        student_backend=STUDENT_BACKEND,
        teacher_backend=TEACHER_BACKEND,
        wan_student_base_model_path=wan_base,
        cosmos_teacher_model_path=cosmos_teacher,
        checkpoint_contract_identity=_canonical_json_digest(
            {"checkpoint_path": str(canonical_checkpoint), "configs": checkpoint_configs}
        ),
    )


def _validated_hybrid_model_paths(
    parent: ValidatedStage1Parent,
) -> tuple[str, str]:
    """Read and independently validate the Student and Teacher roots."""

    _, payload = _read_json_object(
        Path(parent.canonical_path) / "target_student" / "transformer" / "config.json",
        label="Stage-1 target inference config.json",
    )
    if payload.get("student_backend") != STUDENT_BACKEND:
        raise ValueError(
            f"Stage-1 target student_backend must be exactly {STUDENT_BACKEND!r}"
        )
    if payload.get("teacher_backend") != TEACHER_BACKEND:
        raise ValueError(
            f"Stage-1 target teacher_backend must be exactly {TEACHER_BACKEND!r}"
        )
    student_path = payload.get("student_base_model_path")
    if not isinstance(student_path, str) or not student_path:
        raise ValueError(
            "Stage-1 target student_base_model_path must be a non-empty string"
        )
    teacher_path = payload.get("teacher_model_path")
    if not isinstance(teacher_path, str) or not teacher_path:
        raise ValueError(
            "Stage-1 target teacher_model_path must be a non-empty string"
        )
    wan_base = validate_wan_student_base_model_path(student_path)
    cosmos_teacher = validate_cosmos_teacher_model_path(teacher_path)
    return str(wan_base), str(cosmos_teacher)


def validated_stage1_hybrid_model_paths(
    parent: ValidatedStage1Parent,
) -> tuple[str, str]:
    """Public fail-closed accessor for verified Stage-1 hybrid roots."""

    return _validated_hybrid_model_paths(parent)


def stage2_inference_lineage_environment(
    resolved: ResolvedCosmosInferenceCheckpoint,
) -> dict[str, str]:
    """Derive the exact Stage-2 config environment from a verified checkpoint."""

    if resolved.training_stage != "progressive_stage2":
        raise ValueError(
            "Stage-2 inference lineage requires a progressive_stage2 checkpoint"
        )
    if not resolved.parent_stage1_path:
        raise ValueError("Stage-2 inference checkpoint is missing its Stage-1 parent")
    if type(resolved.parent_stage1_expected_step) is not int:
        raise ValueError(
            "Stage-2 inference checkpoint is missing its sealed Stage-1 step"
        )
    parent = validate_stage1_parent(
        Path(resolved.parent_stage1_path),
        expected_step=resolved.parent_stage1_expected_step,
    )
    expected_wan, expected_teacher = _validated_hybrid_model_paths(parent)
    if resolved.wan_student_base_model_path != expected_wan:
        raise ValueError("resolved Wan Student base no longer matches Stage-1")
    if resolved.cosmos_teacher_model_path != expected_teacher:
        raise ValueError("resolved Cosmos Teacher no longer matches Stage-1")
    lineage = json.dumps(
        {
            "parent_stage1_contract_identity": parent.contract_identity,
            "parent_stage1_path": parent.canonical_path,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "STUDENT_BASE_MODEL_PATH": str(
            Path(parent.canonical_path) / "target_student"
        ),
        "WAN_STUDENT_BASE_MODEL_PATH": expected_wan,
        "COSMOS_POLICY_PATH": expected_teacher,
        "RESUME_FROM_PATH": parent.canonical_path,
        "PARENT_STAGE1_PATH": parent.canonical_path,
        "PARENT_STAGE1_CONTRACT_IDENTITY": parent.contract_identity,
        "COSMOS_STAGE1_EXPECTED_STEP": str(resolved.parent_stage1_expected_step),
        "STAGE2_LINEAGE_JSON": lineage,
        "RESUME_ONLINE_FROM_TARGET": "1",
        "RESET_RESUME_STEP": "1",
        "RESUME_OPTIMIZER_STATE": "0",
    }


def bind_stage2_inference_runtime(
    resolved: ResolvedCosmosInferenceCheckpoint,
    *,
    environment: MutableMapping[str, str],
    configured_teacher_model_path: str | Path | None,
) -> dict[str, str]:
    """Fail closed on ambient/CLI lineage mismatches, then bind exact values."""

    expected = stage2_inference_lineage_environment(resolved)
    if configured_teacher_model_path is not None:
        configured_teacher = str(
            validate_cosmos_teacher_model_path(configured_teacher_model_path)
        )
        if configured_teacher != resolved.cosmos_teacher_model_path:
            raise ValueError(
                "configured Cosmos Teacher does not match checkpoint lineage"
            )
    for name, expected_value in expected.items():
        actual_value = environment.get(name)
        if actual_value not in (None, "", expected_value):
            raise ValueError(
                f"{name} does not match checkpoint lineage: "
                f"{actual_value!r} != {expected_value!r}"
            )
    environment.update(expected)
    return expected
