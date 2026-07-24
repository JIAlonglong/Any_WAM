"""Bounded, read-only provenance identity for formal Cosmos LIBERO runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


ARTIFACT_LOCK_SCHEMA = "flashwam_artifact_lock_v1"
PROVENANCE_SCHEMA = "flashwam_cosmos_provenance_v1"
_SHA256_HEX_LENGTH = 64
_MAX_PACKAGE_METADATA_FILES = 4096
_MAX_PACKAGE_METADATA_BYTES = 64 << 20


class ProvenanceError(ValueError):
    """Raised when formal provenance cannot be established exactly."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in relative
    ):
        raise ProvenanceError(f"unsafe artifact-lock path: {relative!r}")
    path = root.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file():
        raise ProvenanceError(f"artifact-lock entry is not a plain file: {path}")
    try:
        path.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise ProvenanceError(f"artifact-lock entry escapes root: {relative}") from exc
    return path


def _canonical_root(path: str | Path, *, label: str) -> Path:
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ProvenanceError(f"{label} must be a plain directory: {root}")
    return root.resolve(strict=True)


def build_artifact_lock(
    root: str | Path,
    *,
    compact_paths: Iterable[str],
    large_paths: Iterable[str],
    immutable_store: bool,
) -> dict[str, Any]:
    """Prepare a lock payload explicitly; callers decide where to persist it."""

    if type(immutable_store) is not bool:
        raise ProvenanceError("immutable_store must be a plain boolean")
    canonical_root = _canonical_root(root, label="artifact root")
    compact = tuple(compact_paths)
    large = tuple(large_paths)
    if not compact or not large:
        raise ProvenanceError("artifact locks require compact and large entries")
    overlap = set(compact) & set(large)
    if overlap:
        raise ProvenanceError(f"artifact lock entry class overlap: {sorted(overlap)}")
    entries = []
    for verification, relatives in (("compact", compact), ("large", large)):
        for relative in relatives:
            path = _plain_file(canonical_root, relative)
            entries.append(
                {
                    "path": relative,
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "verification": verification,
                }
            )
    entries.sort(key=lambda item: item["path"])
    return {
        "schema": ARTIFACT_LOCK_SCHEMA,
        "root": str(canonical_root),
        "immutable_store": immutable_store,
        "entries": entries,
        "tree_digest": _sha256_bytes(canonical_json(entries).encode()),
    }


def _read_artifact_lock(path: str | Path) -> tuple[Path, dict[str, Any]]:
    lock_path = Path(path)
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ProvenanceError(f"artifact lock must be a plain file: {lock_path}")
    raw = lock_path.read_text(encoding="utf-8").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProvenanceError(f"artifact lock is not JSON: {lock_path}") from exc
    if not isinstance(payload, dict) or canonical_json(payload) != raw:
        raise ProvenanceError(f"artifact lock must be exact canonical JSON: {lock_path}")
    return lock_path.resolve(strict=True), payload


def verify_artifact_lock(
    root: str | Path,
    lock_path: str | Path,
    *,
    verify_large_artifact_digests: bool,
) -> dict[str, Any]:
    if type(verify_large_artifact_digests) is not bool:
        raise ProvenanceError(
            "verify_large_artifact_digests must be a plain boolean"
        )
    canonical_root = _canonical_root(root, label="artifact root")
    canonical_lock, payload = _read_artifact_lock(lock_path)
    required_fields = {
        "schema",
        "root",
        "immutable_store",
        "entries",
        "tree_digest",
    }
    if set(payload) != required_fields:
        raise ProvenanceError("artifact lock fields do not match schema")
    if payload["schema"] != ARTIFACT_LOCK_SCHEMA:
        raise ProvenanceError("unsupported artifact lock schema")
    if type(payload["immutable_store"]) is not bool:
        raise ProvenanceError("artifact lock immutable_store must be a plain boolean")
    if payload["root"] != str(canonical_root):
        raise ProvenanceError("artifact lock root does not match resolved root")
    entries = payload["entries"]
    if not isinstance(entries, list) or not entries:
        raise ProvenanceError("artifact lock entries must be a non-empty list")
    if entries != sorted(entries, key=lambda item: item.get("path", "")):
        raise ProvenanceError("artifact lock entries must be path sorted")
    if payload["tree_digest"] != _sha256_bytes(canonical_json(entries).encode()):
        raise ProvenanceError("artifact lock tree digest does not match entries")
    if not verify_large_artifact_digests:
        raise ProvenanceError(
            "large artifact digests require explicit verification; "
            "self-asserted immutable_store is not a trusted attestation"
        )

    compact_verified = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "sha256",
            "size_bytes",
            "verification",
        }:
            raise ProvenanceError("artifact lock entry fields do not match schema")
        relative = entry["path"]
        if not isinstance(relative, str) or relative in seen:
            raise ProvenanceError("artifact lock paths must be unique strings")
        seen.add(relative)
        if entry["verification"] not in {"compact", "large"}:
            raise ProvenanceError("unknown artifact lock verification class")
        if (
            type(entry["size_bytes"]) is not int
            or entry["size_bytes"] < 0
            or not isinstance(entry["sha256"], str)
            or len(entry["sha256"]) != _SHA256_HEX_LENGTH
        ):
            raise ProvenanceError("invalid artifact lock size or digest")
        path = _plain_file(canonical_root, relative)
        if path.stat().st_size != entry["size_bytes"]:
            raise ProvenanceError(f"artifact size differs from lock: {relative}")
        should_hash = (
            entry["verification"] == "compact"
            or verify_large_artifact_digests
        )
        if should_hash and _sha256_file(path) != entry["sha256"]:
            raise ProvenanceError(f"artifact digest differs from lock: {relative}")
        if entry["verification"] == "compact":
            compact_verified.append(entry)
    if not compact_verified:
        raise ProvenanceError("artifact lock has no compact verification entries")
    return {
        "root": str(canonical_root),
        "lock_path": str(canonical_lock),
        "lock_sha256": _sha256_file(canonical_lock),
        "tree_digest": payload["tree_digest"],
        "compact_verified_sha256": _sha256_bytes(
            canonical_json(compact_verified).encode()
        ),
        "immutable_store": payload["immutable_store"],
        "large_digests_verified": bool(verify_large_artifact_digests),
    }


def fingerprint_git_repository(
    path: str | Path, *, purpose: str
) -> dict[str, str]:
    root = _canonical_root(path, label=f"{purpose} repository")
    git_environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    git_environment.update(
        {
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    git_prefix = [
        "git",
        "-c",
        "core.refreshIndex=false",
        "-c",
        "core.fsmonitor=false",
        "-C",
        str(root),
    ]

    def run(*args: str, text: bool = True):
        try:
            return subprocess.run(
                [*git_prefix, *args],
                check=True,
                capture_output=True,
                text=text,
                env=git_environment,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ProvenanceError(f"unable to fingerprint {purpose} repository") from exc

    head = run("rev-parse", "HEAD").strip()
    if not head or len(head) != 40:
        raise ProvenanceError(f"{purpose} repository has no full Git HEAD")
    status = run("status", "--porcelain=v1", "--untracked-files=all", "-z", text=False)
    if status:
        raise ProvenanceError(
            f"{purpose} repository is dirty; formal provenance requires a clean tree"
        )
    return {"kind": "git-clean", "head": head, "root": str(root)}


_WORKER_PROBE = r"""
import json, os, platform, sys
try:
    import torch
    torch_info = {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "git_version": torch.version.git_version,
    }
except Exception as exc:
    raise SystemExit("torch probe failed: " + type(exc).__name__ + ": " + str(exc))
print(json.dumps({
    "executable": os.path.realpath(sys.executable),
    "python_version": sys.version,
    "implementation": platform.python_implementation(),
    "prefix": os.path.realpath(sys.prefix),
    "base_prefix": os.path.realpath(sys.base_prefix),
    "torch": torch_info,
}, sort_keys=True, separators=(",", ":")))
"""


def _worker_identity(
    python: str | Path,
    site_packages: str | Path,
    *,
    extra_pythonpath: Iterable[str | Path],
) -> dict[str, Any]:
    executable = Path(python)
    if executable.is_symlink():
        executable = executable.resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ProvenanceError(f"worker Python is not executable: {python}")
    site = _canonical_root(site_packages, label="worker site-packages")
    extra = []
    for value in extra_pythonpath:
        extra.append(str(_canonical_root(value, label="worker extra PYTHONPATH")))
    environment = os.environ.copy()
    if extra:
        environment["PYTHONPATH"] = os.pathsep.join(extra)
    try:
        output = subprocess.run(
            [str(executable), "-c", _WORKER_PROBE],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
        ).stdout.strip()
        probe = json.loads(output)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise ProvenanceError("Cosmos worker Python probe failed") from exc
    if not isinstance(probe, dict) or probe.get("executable") != str(executable.resolve()):
        raise ProvenanceError("Cosmos worker probe returned the wrong executable")

    metadata_entries = []
    total_bytes = 0
    for metadata in sorted(site.glob("*.dist-info/METADATA")):
        if metadata.is_symlink() or not metadata.is_file():
            raise ProvenanceError(f"invalid worker package metadata: {metadata}")
        total_bytes += metadata.stat().st_size
        if (
            len(metadata_entries) >= _MAX_PACKAGE_METADATA_FILES
            or total_bytes > _MAX_PACKAGE_METADATA_BYTES
        ):
            raise ProvenanceError("worker package metadata exceeds bounded probe limits")
        metadata_entries.append(
            {
                "path": metadata.relative_to(site).as_posix(),
                "size_bytes": metadata.stat().st_size,
                "sha256": _sha256_file(metadata),
            }
        )
    if not metadata_entries or not any(
        entry["path"].lower().startswith("torch-") for entry in metadata_entries
    ):
        raise ProvenanceError("worker package metadata must include torch")
    payload = {
        "probe": probe,
        "python_realpath": str(executable.resolve()),
        "site_packages": str(site),
        "extra_pythonpath": extra,
        "packages_sha256": _sha256_bytes(
            canonical_json(metadata_entries).encode()
        ),
    }
    payload["identity_sha256"] = _sha256_bytes(canonical_json(payload).encode())
    return payload


def _preflight_identity(python: str | Path) -> dict[str, Any]:
    requested = Path(python)
    if not requested.is_file() or not os.access(requested, os.X_OK):
        raise ProvenanceError(
            f"formal preflight Python is not executable: {requested}"
        )
    executable = requested.resolve(strict=True)
    running = Path(sys.executable).resolve(strict=True)
    if executable != running:
        raise ProvenanceError(
            "formal preflight Python does not match the running interpreter"
        )
    payload = {
        "python_realpath": str(executable),
        "python_sha256": _sha256_file(executable),
        "size_bytes": executable.stat().st_size,
        "python_version": sys.version,
        "implementation": platform.python_implementation(),
        "prefix": str(Path(sys.prefix).resolve(strict=True)),
        "base_prefix": str(Path(sys.base_prefix).resolve(strict=True)),
    }
    payload["identity_sha256"] = _sha256_bytes(canonical_json(payload).encode())
    return payload


def resolve_formal_provenance(
    *,
    dataset_root: str | Path,
    dataset_lock: str | Path,
    teacher_root: str | Path,
    teacher_lock: str | Path,
    video_vae_root: str | Path,
    video_vae_lock: str | Path,
    local_model_root: str | Path,
    local_model_lock: str | Path,
    flashwam_repo: str | Path,
    cosmos_repo: str | Path,
    preflight_python: str | Path,
    worker_python: str | Path,
    worker_site_packages: str | Path,
    extra_pythonpath: Iterable[str | Path],
    verify_large_artifact_digests: bool,
) -> dict[str, Any]:
    """Resolve formal provenance without writing or recursively walking artifacts."""

    payload = {
        "schema": PROVENANCE_SCHEMA,
        "dataset": verify_artifact_lock(
            dataset_root,
            dataset_lock,
            verify_large_artifact_digests=verify_large_artifact_digests,
        ),
        "teacher": verify_artifact_lock(
            teacher_root,
            teacher_lock,
            verify_large_artifact_digests=verify_large_artifact_digests,
        ),
        "video_vae": verify_artifact_lock(
            video_vae_root,
            video_vae_lock,
            verify_large_artifact_digests=verify_large_artifact_digests,
        ),
        "local_model": verify_artifact_lock(
            local_model_root,
            local_model_lock,
            verify_large_artifact_digests=verify_large_artifact_digests,
        ),
        "flashwam_repo": fingerprint_git_repository(
            flashwam_repo, purpose="Flash-WAM"
        ),
        "cosmos_repo": fingerprint_git_repository(
            cosmos_repo, purpose="Cosmos"
        ),
        "preflight": _preflight_identity(preflight_python),
        "worker": _worker_identity(
            worker_python,
            worker_site_packages,
            extra_pythonpath=extra_pythonpath,
        ),
    }
    payload["identity_sha256"] = _sha256_bytes(canonical_json(payload).encode())
    return payload
