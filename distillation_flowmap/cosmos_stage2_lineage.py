from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from distillation_flowmap.cosmos_training_contract import validate_contract_metadata


_COMMON_CONTRACT_FIELDS = (
    "contract_version",
    "training_contract_stage",
    "action_packing_schema",
    "action_downsample_factor",
    "action_chunk_shape",
)
_STAGE2_CONTRACT_FIELDS = _COMMON_CONTRACT_FIELDS + (
    "deployment_timestep_start",
    "deployment_timestep_end",
    "joint_student_steps",
    "deployment_joint_rollout_interval",
    "deployment_action_weight",
    "raw_teacher_window_is_auxiliary",
)
_MONOLITHIC_WEIGHTS = "diffusion_pytorch_model.safetensors"
_SHARDED_INDEX = "diffusion_pytorch_model.safetensors.index.json"


@dataclass(frozen=True)
class ValidatedStage1Parent:
    canonical_path: str
    contract_identity: str


def _require_plain_directory(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    if not path.exists():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if not path.is_dir():
        raise ValueError(f"{label} must be a directory: {path}")


def _require_plain_file(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
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


def _contract_payload(
    payload: dict[str, object], *, stage: str
) -> dict[str, object]:
    fields = (
        _COMMON_CONTRACT_FIELDS if stage == "raw_stage1" else _STAGE2_CONTRACT_FIELDS
    )
    return {field: payload[field] for field in fields}


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


def _same_inode(first: Path, second: Path) -> bool:
    first_stat = first.stat()
    second_stat = second.stat()
    return (first_stat.st_dev, first_stat.st_ino) == (
        second_stat.st_dev,
        second_stat.st_ino,
    )


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
    controlled: dict[str, tuple[Path, Path, tuple[Path, ...]]] = {}
    for variant in ("online_student", "target_student"):
        component = path / variant
        transformer = component / "transformer"
        _require_plain_directory(component, label=variant)
        _require_plain_directory(transformer, label=f"{variant}/transformer")
        config = transformer / "config.json"
        exact, payload = _read_json_object(config, label=f"{variant} config.json")
        validate_contract_metadata(payload, required_stage="raw_stage1")
        _validate_step(payload, expected_step=expected_step, label=variant)
        contract = _contract_payload(payload, stage="raw_stage1")
        weights = _validate_weights(transformer, label=variant)
        records[variant] = (config, exact, payload, contract)
        controlled[variant] = (component, transformer, weights)

    online_component, online_transformer, online_weights = controlled[
        "online_student"
    ]
    target_component, target_transformer, target_weights = controlled[
        "target_student"
    ]
    inode_pairs = [
        (online_component, target_component),
        (online_transformer, target_transformer),
        (records["online_student"][0], records["target_student"][0]),
    ]
    if len(online_weights) == len(target_weights):
        inode_pairs.extend(zip(online_weights, target_weights))
    for online_path, target_path in inode_pairs:
        if _same_inode(online_path, target_path):
            raise ValueError(
                "online_student and target_student must not resolve to the same inode: "
                f"{online_path} and {target_path}"
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
    )


def validate_stage2_resume(
    checkpoint: Path, *, arm_root: Path, expected_step: int
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

    online = checkpoint / "online_student"
    transformer = online / "transformer"
    _require_plain_directory(online, label="online_student")
    _require_plain_directory(transformer, label="online_student/transformer")
    _, payload = _read_json_object(
        transformer / "config.json", label="online_student config.json"
    )
    validate_contract_metadata(payload, required_stage="progressive_stage2")
    _validate_step(payload, expected_step=expected_step, label="online_student")
    _validate_weights(transformer, label="online_student")
    _require_plain_file(checkpoint / "optimizer.pt", label="optimizer.pt")
    _require_plain_file(checkpoint / "lr_scheduler.pt", label="lr_scheduler.pt")
    return canonical_checkpoint


def _reject_path_symlinks(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    for component in reversed(absolute.parents):
        if component == component.parent:
            continue
        if component.is_symlink():
            raise ValueError(f"{label} parent must not be a symlink: {component}")
    if absolute.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {absolute}")


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
