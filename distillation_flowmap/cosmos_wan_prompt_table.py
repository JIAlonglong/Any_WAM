"""Build and validate the all-40 LIBERO prompt table for the Wan student.

Only the canonical task strings are inherited from the official Cosmos
artifact.  Embeddings are always produced by the explicitly selected Wan
tokenizer and text encoder.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch


FORMAT = "flash_wam.libero_wan_prompt_table.v1"
PROMPT_SHAPE = (1, 512, 4096)
DEFAULT_MANIFEST = Path(__file__).with_name("libero_40_task_manifest.json")
_CONFIG_FILES = (
    "model_index.json",
    "tokenizer/tokenizer_config.json",
    "text_encoder/config.json",
    "transformer/config.json",
)
_ARTIFACT_DIRS = ("text_encoder", "transformer")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_manifest_digest(tasks: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_json(list(tasks))).hexdigest()


def load_canonical_tasks(path: str | Path = DEFAULT_MANIFEST) -> tuple[str, ...]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format") != "flash_wam.libero_40_task_manifest.v1":
        raise ValueError(f"unsupported LIBERO task manifest format: {source}")
    tasks = payload.get("tasks")
    if (
        not isinstance(tasks, list)
        or len(tasks) != 40
        or any(not isinstance(task, str) or not task.strip() for task in tasks)
        or len(set(tasks)) != 40
    ):
        raise ValueError(
            f"canonical LIBERO task manifest must contain exactly 40 unique strings: {source}"
        )
    return tuple(tasks)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_wan_base(path: str | Path) -> dict[str, Any]:
    """Return a stable, cheap identity for the explicit Wan base model.

    Config contents are hashed.  Large weight files are represented by their
    relative names and sizes, which catches incomplete/mismatched layouts
    without hashing many gigabytes during every preflight.
    """

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Wan base model directory does not exist: {root}")

    configs = {}
    for relative in _CONFIG_FILES:
        source = root / relative
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(
                f"Wan base model is missing plain file {relative}: {source}"
            )
        configs[relative] = _sha256_file(source)

    weights = {}
    for relative_dir in _ARTIFACT_DIRS:
        directory = root / relative_dir
        files = sorted(
            source
            for source in directory.rglob("*")
            if source.is_file() and source.suffix in _WEIGHT_SUFFIXES
        )
        if not files:
            raise FileNotFoundError(
                f"Wan base model has no weights under {relative_dir}: {directory}"
            )
        for source in files:
            weights[str(source.relative_to(root))] = source.stat().st_size

    identity_payload = {"configs": configs, "weight_sizes": weights}
    return {
        "resolved_path": str(root),
        "identity_sha256": hashlib.sha256(
            _canonical_json(identity_payload)
        ).hexdigest(),
        **identity_payload,
    }


def build_prompt_embeddings(
    tasks: Sequence[str],
    *,
    tokenizer: Any,
    text_encoder: Any,
    device: str | torch.device,
    output_dtype: torch.dtype,
    prompt_cleaner: Callable[[str], str],
    batch_size: int = 4,
) -> dict[str, torch.Tensor]:
    """Encode task strings using the same padded Wan text contract as training."""

    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("tasks must be a non-empty sequence of unique strings")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    text_encoder.eval()
    result: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for start in range(0, len(tasks), batch_size):
            raw_prompts = list(tasks[start : start + batch_size])
            prompts = [prompt_cleaner(prompt) for prompt in raw_prompts]
            tokenized = tokenizer(
                prompts,
                padding="max_length",
                max_length=PROMPT_SHAPE[1],
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)
            encoded = text_encoder(input_ids, attention_mask).last_hidden_state
            expected = (len(raw_prompts), PROMPT_SHAPE[1], PROMPT_SHAPE[2])
            if tuple(encoded.shape) != expected:
                raise ValueError(
                    f"Wan text encoder must return shape {list(expected)}, "
                    f"got {tuple(encoded.shape)}"
                )
            encoded = encoded.to(dtype=output_dtype)
            encoded = encoded.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)
            if not torch.isfinite(encoded).all():
                raise ValueError("Wan text encoder produced non-finite prompt embeddings")
            for index, task in enumerate(raw_prompts):
                result[task] = encoded[index : index + 1].detach().cpu().clone()
    return result


def prompt_table_payload(
    embeddings: Mapping[str, torch.Tensor],
    *,
    tasks: Sequence[str],
    wan_base: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "embeddings": dict(embeddings),
        "metadata": {
            "task_count": len(tasks),
            "task_manifest_sha256": canonical_manifest_digest(tasks),
            "prompt_shape": list(PROMPT_SHAPE),
            "wan_base_identity": wan_base["identity_sha256"],
            "wan_base_resolved_path": wan_base["resolved_path"],
            "wan_base_configs": wan_base["configs"],
            "wan_base_weight_sizes": wan_base["weight_sizes"],
        },
    }


def save_prompt_table_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Publish a complete table atomically, never replacing an existing path."""

    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite prompt table: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"prompt table parent directory does not exist: {destination.parent}"
        )

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite prompt table: {destination}"
            ) from exc
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def validate_prompt_table(
    path: str | Path,
    *,
    tasks: Sequence[str],
    expected_wan_base_identity: str | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"prompt table must be a plain file: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("format") != FORMAT:
        raise ValueError(f"unsupported or missing prompt table format: {source}")
    embeddings = payload.get("embeddings")
    metadata = payload.get("metadata")
    if not isinstance(embeddings, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("prompt table must contain embeddings and metadata mappings")

    expected_tasks = set(tasks)
    actual_tasks = set(embeddings)
    missing = sorted(expected_tasks - actual_tasks)
    extra = sorted(actual_tasks - expected_tasks)
    if missing or extra:
        raise ValueError(
            f"prompt table task mismatch; missing={missing!r}, extra={extra!r}"
        )
    expected_digest = canonical_manifest_digest(tasks)
    if metadata.get("task_manifest_sha256") != expected_digest:
        raise ValueError("prompt table task manifest identity does not match")
    if (
        expected_wan_base_identity is not None
        and metadata.get("wan_base_identity") != expected_wan_base_identity
    ):
        raise ValueError("prompt table Wan base identity does not match")

    for task in tasks:
        embedding = embeddings[task]
        if not isinstance(embedding, torch.Tensor):
            raise TypeError(f"prompt embedding for {task!r} must be a torch Tensor")
        if tuple(embedding.shape) != PROMPT_SHAPE:
            raise ValueError(
                f"prompt embedding for {task!r} must have shape "
                f"{list(PROMPT_SHAPE)}, got {tuple(embedding.shape)}"
            )
        if not torch.is_floating_point(embedding) or not torch.isfinite(embedding).all():
            raise ValueError(f"prompt embedding for {task!r} must be finite floating point")
    return dict(metadata)
