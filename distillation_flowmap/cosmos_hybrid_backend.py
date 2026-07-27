from __future__ import annotations

import json
from pathlib import Path


STUDENT_BACKEND = "wan_flowmap"
TEACHER_BACKEND = "cosmos_policy"
WAN_TRANSFORMER_CLASS = "WanTransformer3DModel"
COSMOS_MODEL_TYPE = "cosmos-policy"

_COSMOS_POLICY_ASSETS = (
    "Cosmos-Policy-LIBERO-Predict2-2B.pt",
    "libero_dataset_statistics.json",
    "libero_t5_embeddings.pkl",
)


def _json_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def validate_wan_student_base_model_path(path: str | Path) -> Path:
    """Validate the base layout consumed by the Wan FlowMap Student loader."""

    root = Path(path).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(
            f"{STUDENT_BACKEND} student base must be a plain directory: {root}"
        )
    config_path = root / "transformer" / "config.json"
    if not config_path.is_file():
        raise ValueError(
            f"{STUDENT_BACKEND} requires transformer/config.json with "
            f"_class_name={WAN_TRANSFORMER_CLASS!r}; got {root}"
        )
    payload = _json_object(config_path, label="Wan student transformer config")
    actual = payload.get("_class_name")
    if actual != WAN_TRANSFORMER_CLASS:
        raise ValueError(
            f"{STUDENT_BACKEND} requires _class_name={WAN_TRANSFORMER_CLASS!r}, "
            f"got {actual!r} in {config_path}"
        )
    return root.resolve(strict=True)


def validate_cosmos_teacher_model_path(path: str | Path) -> Path:
    """Validate the official frozen Cosmos Policy Teacher root independently."""

    root = Path(path).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(
            f"{TEACHER_BACKEND} teacher root must be a plain directory: {root}"
        )
    config_path = root / "config.json"
    payload = _json_object(config_path, label="Cosmos Policy teacher config")
    actual = payload.get("model_type")
    if actual != COSMOS_MODEL_TYPE:
        raise ValueError(
            f"{TEACHER_BACKEND} requires model_type={COSMOS_MODEL_TYPE!r}, "
            f"got {actual!r} in {config_path}"
        )
    missing = [
        name
        for name in _COSMOS_POLICY_ASSETS
        if (root / name).is_symlink() or not (root / name).is_file()
    ]
    if missing:
        raise ValueError(
            f"{TEACHER_BACKEND} teacher root is missing official policy assets: "
            + ", ".join(missing)
        )
    return root.resolve(strict=True)
