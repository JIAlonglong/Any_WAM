#!/usr/bin/env python3
"""Plan and (only when explicitly requested) run isolated Cosmos mixed policies.

The legacy ``run_cosmos_progressive_stage2.py`` runner represents a chained
S4 -> S2 -> S1 experiment and deliberately remains untouched.  This runner
instead owns one independent policy root at a time: ``universe``, ``s2``, or
``s1``.  Each fresh policy starts from the common Stage-1 *online student*
checkpoint, uses the frozen full Cosmos objective template, and changes only
the mixed endpoint rollout distribution supplied by ``cosmos_mixed_step_policy``.

Without ``--run`` this module is a side-effect-free plan generator.  That is
intentional: generating a command must never start an eight-GPU job, make a
cache, or mutate a training root.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat as _stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

from distillation_flowmap.cosmos_mixed_step_policy import (
    get_mixed_step_policy_spec,
    parse_forced_indices,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "training_data" / "libero-long-lerobot"
DEFAULT_STAGE1_CHECKPOINT = (
    Path("/root/nas/junjie/jj/Any_WAM/distillation_flowmap")
    / "output_libero_cosmos_policy_stage1_cosmos_latent_cdiff_8gpu_20260706_cosmos_latent_s1s2_8gpu"
    / "checkpoints"
    / "step_5000"
)
DEFAULT_TEACHER_MODEL = Path(
    "/root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
)
DEFAULT_TORCHRUN = Path("/root/nas/junjie/conda_envs/any_wam/bin/torchrun")
DEFAULT_DEVICE_LIST = "0,1,2,3,4,5,6,7"
DEFAULT_WORLD_SIZE = 8
DEFAULT_MAX_TRAIN_STEPS = 5000
DEFAULT_SAVE_INTERVAL = 250
DEFAULT_MASTER_PORT = 29801
DEFAULT_TRAIN_SEED = 20260716
SUPPORTED_POLICY_NAMES = ("universe", "s2", "s1")

# This is forbidden, including a symlink alias or a descendant path.  The
# independent-policy runner must never touch the old staged S4 experiment.
LEGACY_PROGRESSIVE_ROOT = (
    REPO_ROOT
    / "distillation_flowmap"
    / "output_libero_cosmos_policy_stage2_progressive_20260714_full"
)

_METADATA_FILENAMES = (
    "protocol.json",
    "train_manifest.json",
    "selection_manifest.json",
    "test_manifest.json",
    "eval_pairs.json",
)
_BUDGET_SPECS = (
    ("s1", 4, 1, "t4"),
    ("s2", 4, 2, "t4"),
    ("s4", 8, 4, "t8"),
)
_PREFLIGHT_LABELS = ("s1", "s2", "s4")
_PREFLIGHT_FORCE_INDICES = (0, 1, 2, 0, 1, 2, 0, 1, 2)
_PREFLIGHT_MINIMUM_PER_LABEL = 3


@dataclass(frozen=True)
class _FreshPolicyRootIdentity:
    """Directory identity recorded immediately after a fresh-root claim."""

    root: Path
    st_dev: int
    st_ino: int


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def _resolve_path(value: str | Path) -> Path:
    """Resolve aliases without requiring that the final path already exists."""
    return _as_path(value).resolve(strict=False)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _as_str_env(values: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in values.items()}


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether either resolved path is contained by the other."""
    return _is_within(left, right) or _is_within(right, left)


def _capture_fresh_policy_root_identity(root: str | Path) -> _FreshPolicyRootIdentity:
    """Record a newly claimed root without following a replacement symlink."""
    root = _as_path(root)
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise RuntimeError(
            f"Cannot capture fresh policy root identity: {root}"
        ) from exc
    if _stat.S_ISLNK(metadata.st_mode) or not _stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(
            "Cannot capture fresh policy root identity because it is not a "
            f"non-symlink directory: {root}"
        )
    return _FreshPolicyRootIdentity(
        root=root,
        st_dev=int(metadata.st_dev),
        st_ino=int(metadata.st_ino),
    )


def _validate_fresh_policy_root_identity(
    identity: _FreshPolicyRootIdentity, *, phase: str
) -> Path:
    """Reject an identity/symlink change at a runner-owned phase boundary."""
    root = identity.root
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise RuntimeError(
            f"Fresh policy root identity changed before {phase}: root is missing: {root}"
        ) from exc
    if _stat.S_ISLNK(metadata.st_mode) or not _stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(
            "Fresh policy root identity changed before "
            f"{phase}: root is no longer a non-symlink directory: {root}"
        )
    if (int(metadata.st_dev), int(metadata.st_ino)) != (
        identity.st_dev,
        identity.st_ino,
    ):
        raise RuntimeError(
            "Fresh policy root identity changed before "
            f"{phase}: root={root}"
        )
    return root


def _reject_symlinked_owned_artifact_path(
    root: str | Path, artifact: str | Path, *, label: str
) -> None:
    """Reject a symlink in an owned artifact path before any write is attempted.

    Resolving the final path catches escapes outside ``root``.  This second
    check also rejects an in-root alias such as ``root/protocol -> root/other``:
    runner-owned destinations are deliberately canonical, direct children of
    the policy root rather than mutable symlink routing points.
    """
    resolved_root = _resolve_path(root)
    lexical_artifact = _as_path(artifact)
    try:
        relative_parts = lexical_artifact.relative_to(resolved_root).parts
    except ValueError as exc:
        raise ValueError(
            "Cosmos mixed-step training artifact is not a direct child of the "
            f"canonical policy root: artifact={label}, path={lexical_artifact}, "
            f"root={resolved_root}"
        ) from exc
    current = resolved_root
    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(
                "Cosmos mixed-step training artifact must not traverse a symlink: "
                f"artifact={label}, path={current}"
            )


def _reject_existing_output_artifact(path: str | Path, *, label: str) -> None:
    path = _as_path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(
            "Refusing to overwrite existing Cosmos mixed-step training artifact: "
            f"artifact={label}, path={path}"
        )


def _require_existing_directory_or_missing(path: str | Path, *, label: str) -> None:
    path = _as_path(path)
    if path.exists() and not path.is_dir():
        raise FileExistsError(
            "Cosmos mixed-step training artifact directory is not a directory: "
            f"artifact={label}, path={path}"
        )


def validate_common_stage1_checkpoint(stage1_checkpoint: str | Path) -> Path:
    """Require the reviewed common Stage-1 online-student source exactly.

    Fresh mixed policies deliberately do not accept a convenient alternate
    checkpoint.  Resume plans keep using their own policy checkpoint for the
    optimizer/model state, while their root provenance must still identify this
    same common Stage-1 source.
    """
    resolved_stage1 = _resolve_path(stage1_checkpoint)
    expected_stage1 = _resolve_path(DEFAULT_STAGE1_CHECKPOINT)
    if resolved_stage1 != expected_stage1:
        raise ValueError(
            "Cosmos mixed-step initial policies require the approved common "
            "Stage-1 online-student checkpoint: "
            f"expected={expected_stage1}, got={resolved_stage1}"
        )
    return resolved_stage1


def _require_supported_policy_name(policy_name: str) -> str:
    normalized_name = str(policy_name).strip().lower()
    if normalized_name not in SUPPORTED_POLICY_NAMES:
        raise ValueError(
            "Unsupported Cosmos mixed-step policy "
            f"{normalized_name!r}; expected one of {list(SUPPORTED_POLICY_NAMES)}"
        )
    return normalized_name


def parse_device_list(
    device_list: str | Sequence[int | str], *, world_size: int = DEFAULT_WORLD_SIZE
) -> tuple[str, ...]:
    """Parse and strictly validate the eight CUDA devices used by all ranks."""
    world_size = int(world_size)
    if world_size != DEFAULT_WORLD_SIZE:
        raise ValueError(
            "Cosmos mixed-step policies require exactly 8 ranks; "
            f"got world_size={world_size}"
        )
    tokens = (
        [part.strip() for part in str(device_list).split(",")]
        if isinstance(device_list, str)
        else [str(part).strip() for part in device_list]
    )
    if not tokens or any(not token for token in tokens):
        raise ValueError("CUDA device list must contain non-empty comma-separated devices")
    if any(not token.isdecimal() for token in tokens):
        raise ValueError("CUDA device list must contain numeric device ordinals")
    if len(tokens) != world_size:
        raise ValueError(
            f"Cosmos mixed-step policies require {world_size} devices; got {len(tokens)}"
        )
    if len(set(tokens)) != len(tokens):
        raise ValueError("CUDA device list entries must be unique")
    return tuple(tokens)


def validate_policy_root(
    root: str | Path,
    *,
    legacy_root: str | Path = LEGACY_PROGRESSIVE_ROOT,
    resume: bool = False,
) -> Path:
    """Reject unsafe roots without deleting or modifying any existing output.

    Plan construction remains side-effect-free and may inspect an existing
    root, but an explicit fresh ``--run`` claims it atomically and therefore
    rejects an existing empty or metadata-only root.  Resuming is an explicit
    mode and is validated separately against its own checkpoint and manifest
    ownership.
    """
    resolved_root = _resolve_path(root)
    resolved_legacy = _resolve_path(legacy_root)
    if _paths_overlap(resolved_root, resolved_legacy):
        raise ValueError(
            "Cosmos mixed-step policy root must not overlap the legacy progressive "
            f"S4 root: root={resolved_root}, legacy={resolved_legacy}"
        )
    manifest_path = resolved_root / "policy_manifest.json"
    if not resume and (manifest_path.exists() or manifest_path.is_symlink()):
        raise FileExistsError(
            "Refusing to start a fresh independent policy in a root with an "
            f"existing policy_manifest.json: {manifest_path}"
        )
    checkpoints_dir = resolved_root / "checkpoints"
    if not resume and checkpoints_dir.exists() and any(checkpoints_dir.iterdir()):
        raise FileExistsError(
            "Refusing to start an independent policy in a root with existing "
            f"checkpoints: {checkpoints_dir}. Pass an explicit valid resume mode instead."
        )
    return resolved_root


def validate_policy_source_isolation(
    root: str | Path,
    *,
    protocol_source_root: str | Path,
    stage1_checkpoint: str | Path | None,
    teacher_model_path: str | Path,
    dataset_path: str | Path,
) -> None:
    """Ensure output writes cannot land in immutable protocol/source artifacts."""
    resolved_root = _resolve_path(root)
    sources: tuple[tuple[str, Path], ...] = (
        ("protocol source root", _resolve_path(protocol_source_root)),
        ("teacher model path", _resolve_path(teacher_model_path)),
        ("dataset path", _resolve_path(dataset_path)),
    )
    if stage1_checkpoint is not None:
        sources = (
            ("Stage-1 source checkpoint", _resolve_path(stage1_checkpoint)),
            *sources,
        )
    for label, source in sources:
        if _paths_overlap(resolved_root, source):
            raise ValueError(
                "Cosmos mixed-step policy output root must not overlap "
                f"{label}: root={resolved_root}, source={source}"
            )


def validate_policy_eval_artifact_isolation(
    *,
    root: str | Path,
    checkpoint_dir: str | Path,
    protocol_source_root: str | Path,
    dataset_path: str | Path,
    teacher_model_path: str | Path,
    stage1_checkpoint: str | Path,
    selection_proxy_path: str | Path | None = None,
) -> dict[str, Path]:
    """Resolve every eval write target before allowing an eval plan to proceed.

    The policy root can be valid while a nested ``metrics`` entry (or an
    individual result/marker) is a symlink into one of the immutable inputs.
    Validate every concrete artifact path independently so plan construction
    and plan execution use the same no-write-into-source boundary.
    """
    root = _resolve_path(root)
    checkpoint_dir = _resolve_path(checkpoint_dir)
    metrics_dir = root / "metrics" / "selection" / checkpoint_dir.name
    expected_selection_proxy = metrics_dir / "selection_proxy.json"
    artifacts: list[tuple[str, str | Path]] = [
        ("metrics directory", metrics_dir),
        ("S1 output JSON", metrics_dir / "s1.json"),
        ("S1 output JSON temporary file", metrics_dir / "s1.json.tmp"),
        ("S2 output JSON", metrics_dir / "s2.json"),
        ("S2 output JSON temporary file", metrics_dir / "s2.json.tmp"),
        ("S4 output JSON", metrics_dir / "s4.json"),
        ("S4 output JSON temporary file", metrics_dir / "s4.json.tmp"),
        ("selection proxy", expected_selection_proxy),
        (
            "selection proxy temporary file",
            expected_selection_proxy.with_name(expected_selection_proxy.name + ".tmp"),
        ),
        ("evaluation completion marker", metrics_dir / "EVAL_COMPLETE"),
        ("evaluation completion marker temporary file", metrics_dir / "EVAL_COMPLETE.tmp"),
        ("evaluation failure marker", metrics_dir / "EVAL_FAILED"),
        ("evaluation failure marker temporary file", metrics_dir / "EVAL_FAILED.tmp"),
    ]
    if selection_proxy_path is not None:
        artifacts.append(("declared selection proxy", selection_proxy_path))
    sources = (
        ("protocol source root", _resolve_path(protocol_source_root)),
        ("dataset path", _resolve_path(dataset_path)),
        ("teacher model path", _resolve_path(teacher_model_path)),
        ("Stage-1 source checkpoint", _resolve_path(stage1_checkpoint)),
    )
    resolved_artifacts: dict[str, Path] = {}
    for label, artifact in artifacts:
        resolved_artifact = _resolve_path(artifact)
        if not _is_within(resolved_artifact, root):
            raise ValueError(
                "Cosmos mixed-step evaluation artifact must resolve under policy root: "
                f"artifact={label}, path={resolved_artifact}, root={root}"
            )
        for source_label, source in sources:
            if _paths_overlap(resolved_artifact, source):
                raise ValueError(
                    "Cosmos mixed-step evaluation artifact must not overlap immutable "
                    f"{source_label}: artifact={label}, path={resolved_artifact}, "
                    f"source={source}"
                )
        resolved_artifacts[label] = resolved_artifact
    return resolved_artifacts


def _reject_existing_eval_output(path: str | Path, *, label: str) -> None:
    path = _as_path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(
            "Refusing to overwrite existing Cosmos mixed-step evaluation output: "
            f"artifact={label}, path={path}"
        )


def validate_policy_eval_write_availability(
    plan: Mapping[str, Any],
    *,
    checkpoint_dir: str | Path,
    completed_budgets: Sequence[str] = (),
) -> None:
    """Reserve outputs before every evaluator subprocess.

    The evaluator writes JSON directly, so a post-evaluation proxy check is too
    late to prevent overwrites.  ``completed_budgets`` is the small set that a
    prior evaluator in this same invocation may have legitimately produced;
    those paths are still checked for symlink traversal but are not treated as
    collisions before the next budget runs.
    """
    try:
        root = _resolve_path(plan["root"])
        protocol_source_root = _resolve_path(plan["protocol_source_root"])
        dataset_path = _resolve_path(plan["dataset_path"])
        teacher_model_path = _resolve_path(plan["teacher_model_path"])
        stage1_checkpoint = _resolve_path(plan["stage1_checkpoint"])
        declared_selection_proxy = _resolve_path(plan["selection_proxy_path"])
    except (KeyError, TypeError, ValueError, OSError) as exc:
        _raise_invalid_eval_execution_plan(
            f"missing or invalid eval write metadata: {exc}"
        )
    completed = {str(budget) for budget in completed_budgets}
    expected_budgets = {budget for budget, *_unused in _BUDGET_SPECS}
    if not completed.issubset(expected_budgets):
        _raise_invalid_eval_execution_plan(
            f"unknown completed evaluation budget(s): {sorted(completed - expected_budgets)}"
        )
    checkpoint_dir = _resolve_path(checkpoint_dir)
    artifact_paths = validate_policy_eval_artifact_isolation(
        root=root,
        checkpoint_dir=checkpoint_dir,
        protocol_source_root=protocol_source_root,
        dataset_path=dataset_path,
        teacher_model_path=teacher_model_path,
        stage1_checkpoint=stage1_checkpoint,
        selection_proxy_path=declared_selection_proxy,
    )
    metrics_dir = root / "metrics" / "selection" / checkpoint_dir.name
    selection_proxy = metrics_dir / "selection_proxy.json"
    if declared_selection_proxy != artifact_paths["selection proxy"]:
        _raise_invalid_eval_execution_plan(
            "selection_proxy_path is not the owned checkpoint selection proxy"
        )
    owned_paths: list[tuple[str, Path]] = [
        ("metrics directory", metrics_dir),
        ("selection proxy", selection_proxy),
        ("selection proxy temporary file", selection_proxy.with_name(selection_proxy.name + ".tmp")),
        ("evaluation completion marker", metrics_dir / "EVAL_COMPLETE"),
        ("evaluation completion marker temporary file", metrics_dir / "EVAL_COMPLETE.tmp"),
        ("evaluation failure marker", metrics_dir / "EVAL_FAILED"),
        ("evaluation failure marker temporary file", metrics_dir / "EVAL_FAILED.tmp"),
    ]
    for budget, *_unused in _BUDGET_SPECS:
        output_json = metrics_dir / f"{budget}.json"
        owned_paths.extend(
            (
                (f"{budget} evaluation JSON", output_json),
                (f"{budget} evaluation JSON temporary file", output_json.with_name(output_json.name + ".tmp")),
            )
        )
    for label, path in owned_paths:
        _reject_symlinked_owned_artifact_path(root, path, label=f"evaluation {label}")
    _require_existing_directory_or_missing(metrics_dir, label="evaluation metrics directory")

    _reject_existing_eval_output(selection_proxy, label="selection proxy")
    _reject_existing_eval_output(
        selection_proxy.with_name(selection_proxy.name + ".tmp"),
        label="selection proxy temporary file",
    )
    for marker in ("EVAL_COMPLETE", "EVAL_COMPLETE.tmp", "EVAL_FAILED", "EVAL_FAILED.tmp"):
        _reject_existing_eval_output(metrics_dir / marker, label=f"evaluation marker {marker}")
    for budget, *_unused in _BUDGET_SPECS:
        output_json = metrics_dir / f"{budget}.json"
        _reject_existing_eval_output(
            output_json.with_name(output_json.name + ".tmp"),
            label=f"{budget} evaluation JSON temporary file",
        )
        if budget not in completed:
            _reject_existing_eval_output(output_json, label=f"{budget} evaluation JSON")
        elif output_json.exists() and not output_json.is_file():
            raise FileExistsError(
                "Completed Cosmos mixed-step evaluation output is not a regular file: "
                f"artifact={budget} evaluation JSON, path={output_json}"
            )


def validate_resume_policy_ownership(
    root: str | Path,
    policy_name: str,
    *,
    expected_source_checkpoint: str | Path | None = None,
) -> Path:
    """Require a root-level manifest that proves a resume owns this policy.

    Training resumes additionally pin their provenance to the reviewed common
    Stage-1 source.  Eval-only plans intentionally leave that optional so they
    can inspect already-created policy roots without changing their historical
    provenance contract.
    """
    root = _resolve_path(root)
    expected_policy = _require_supported_policy_name(policy_name)
    manifest_path = root / "policy_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Cannot resume mixed-step policy without root policy_manifest.json: "
            f"{manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Mixed-step resume policy_manifest is malformed: {manifest_path}"
        ) from exc
    try:
        actual_policy = str(payload["policy"]["name"]).strip().lower()
    except (KeyError, TypeError):
        raise ValueError(
            f"Mixed-step resume policy_manifest is malformed: {manifest_path}"
        ) from None
    if actual_policy != expected_policy:
        raise ValueError(
            "Mixed-step resume policy_manifest belongs to "
            f"{actual_policy!r}, not requested policy {expected_policy!r}: {manifest_path}"
        )
    if expected_source_checkpoint is not None:
        try:
            actual_source = _resolve_path(payload["source_checkpoint"])
        except (KeyError, TypeError, OSError) as exc:
            raise ValueError(
                "Mixed-step resume policy_manifest has no valid common Stage-1 "
                f"source provenance: {manifest_path}"
            ) from exc
        expected_source = validate_common_stage1_checkpoint(expected_source_checkpoint)
        if actual_source != expected_source:
            raise ValueError(
                "Mixed-step resume policy_manifest does not retain the approved "
                "common Stage-1 source provenance: "
                f"expected={expected_source}, got={actual_source}"
            )
    return manifest_path


def validate_preflight_selection_evidence(
    path: str | Path,
    *,
    minimum_per_label: int = _PREFLIGHT_MINIMUM_PER_LABEL,
) -> dict[str, int]:
    """Strictly prove that the short preflight sampled every forced endpoint.

    The worker calls this after the runner returns and before it can publish
    ``PREFLIGHT_COMPLETE``.  There is intentionally no best-effort fallback:
    malformed JSONL, a non-Universe record, an unforced selection, or an
    unknown/mismatched endpoint makes the worker fail and publish its failure
    marker through the existing exit trap.
    """
    minimum_per_label = _validate_positive_int(
        minimum_per_label, name="minimum_per_label"
    )
    if minimum_per_label != _PREFLIGHT_MINIMUM_PER_LABEL:
        raise ValueError(
            "Preflight evidence requires exactly three records for each fixed "
            "S1/S2/S4 endpoint"
        )
    path = _as_path(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(
            "Preflight rank-zero selection JSONL is missing or not a regular file: "
            f"{path}"
        )
    expected_pairs = {
        "s1": (0, 4, 1),
        "s2": (1, 4, 2),
        "s4": (2, 8, 4),
    }
    expected_labels = _PREFLIGHT_LABELS * _PREFLIGHT_MINIMUM_PER_LABEL
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(
                f"Preflight rank-zero selection JSONL contains a blank line at "
                f"{path}:{line_number}"
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed preflight rank-zero selection JSONL at {path}:{line_number}"
            ) from exc
        if not isinstance(record, Mapping):
            raise ValueError(
                f"Preflight selection record is not an object at {path}:{line_number}"
            )
        records.append(record)
    if len(records) != len(expected_labels):
        raise ValueError(
            "Preflight rank-zero selection JSONL must contain exactly nine "
            f"forced records, got {len(records)}: {path}"
        )

    counts = {label: 0 for label in _PREFLIGHT_LABELS}
    for selection_ordinal, (record, expected_label) in enumerate(
        zip(records, expected_labels)
    ):
        line_number = selection_ordinal + 1
        if record.get("policy_name") != "universe":
            raise ValueError(
                f"Preflight selection record is not Universe at {path}:{line_number}"
            )
        if record.get("forced") is not True:
            raise ValueError(
                f"Preflight selection record is not forced at {path}:{line_number}"
            )
        label = record.get("pair_label")
        if label not in expected_pairs:
            raise ValueError(
                f"Preflight selection record has an unsupported endpoint at "
                f"{path}:{line_number}: {label!r}"
            )
        if label != expected_label:
            raise ValueError(
                "Preflight rank-zero selection JSONL violates the fixed forced "
                f"endpoint sequence at {path}:{line_number}: expected={expected_label}, "
                f"got={label}"
            )
        try:
            actual_ordinal = int(record["selection_ordinal"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Preflight selection record has malformed selection ordinal at "
                f"{path}:{line_number}"
            ) from exc
        if actual_ordinal != selection_ordinal:
            raise ValueError(
                "Preflight rank-zero selection JSONL violates the fixed forced "
                f"selection ordinal sequence at {path}:{line_number}: "
                f"expected={selection_ordinal}, got={actual_ordinal}"
            )
        expected_index, expected_teacher, expected_student = expected_pairs[label]
        try:
            actual_index = int(record["pair_index"])
            actual_teacher = int(record["teacher_steps"])
            actual_student = int(record["student_steps"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Preflight selection record has malformed pair metadata at "
                f"{path}:{line_number}"
            ) from exc
        if (actual_index, actual_teacher, actual_student) != (
            expected_index,
            expected_teacher,
            expected_student,
        ):
            raise ValueError(
                f"Preflight selection record has mismatched pair metadata at "
                f"{path}:{line_number}"
            )
        counts[label] += 1
    if any(count != minimum_per_label for count in counts.values()):
        raise ValueError(
            "Preflight rank-zero selection JSONL violates the fixed forced "
            f"endpoint counts: {counts}"
        )
    return counts


def _validate_positive_int(value: int, *, name: str, allow_zero: bool = False) -> int:
    value = int(value)
    if value < 0 or (value == 0 and not allow_zero):
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparator}")
    return value


def _validate_master_port(master_port: int) -> int:
    master_port = int(master_port)
    if not 1 <= master_port <= 65535:
        raise ValueError("master_port must be in [1, 65535]")
    return master_port


def build_torchrun_command(
    *,
    torchrun: str | Path,
    world_size: int,
    master_port: int,
    teacher_model_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    resume_from_path: str | Path,
    gradient_accumulation_steps: int = 1,
) -> list[str]:
    """Build the exact eight-rank training argv without launching it."""
    world_size = int(world_size)
    if world_size != DEFAULT_WORLD_SIZE:
        raise ValueError(
            "Cosmos mixed-step policies require --nproc_per_node=8; "
            f"got {world_size}"
        )
    if int(gradient_accumulation_steps) != 1:
        raise ValueError(
            "Cosmos full standalone OPD requires gradient_accumulation_steps=1"
        )
    master_port = _validate_master_port(master_port)
    return [
        str(_as_path(torchrun)),
        f"--nproc_per_node={world_size}",
        f"--master_port={master_port}",
        "distillation_flowmap/train.py",
        "--teacher-model-path",
        str(_as_path(teacher_model_path)),
        "--dataset-path",
        str(_as_path(dataset_path)),
        "--output-dir",
        str(_as_path(output_dir)),
        "--resume-from-path",
        str(_as_path(resume_from_path)),
        "--gradient-accumulation-steps",
        "1",
    ]


def _policy_payload(policy_name: str) -> dict[str, Any]:
    spec = get_mixed_step_policy_spec(_require_supported_policy_name(policy_name))
    return {
        "name": spec.name,
        "rollout_step_pairs": [list(pair) for pair in spec.rollout_step_pairs],
        "weights": list(spec.weights),
    }


def _checkpoint_transformer_dir(checkpoint_dir: str | Path) -> Path:
    return _as_path(checkpoint_dir) / "online_student" / "transformer"


def _plan_target_step(
    *,
    current_step: int,
    chunk_size: int,
    max_train_steps: int,
    stop_after_step: int,
) -> tuple[int, int]:
    """Return expected checkpoint step and the exact STOP_AFTER_STEP value."""
    current_step = _validate_positive_int(
        current_step, name="current_step", allow_zero=True
    )
    chunk_size = _validate_positive_int(
        chunk_size, name="chunk_size", allow_zero=True
    )
    max_train_steps = _validate_positive_int(max_train_steps, name="max_train_steps")
    stop_after_step = _validate_positive_int(
        stop_after_step, name="stop_after_step", allow_zero=True
    )
    if current_step >= max_train_steps:
        raise ValueError("current_step must be smaller than max_train_steps")
    if stop_after_step:
        if not current_step < stop_after_step <= max_train_steps:
            raise ValueError(
                "stop_after_step must be greater than current_step and no larger "
                "than max_train_steps"
            )
        return stop_after_step, stop_after_step
    if chunk_size:
        target_step = min(current_step + chunk_size, max_train_steps)
        return target_step, target_step
    # Zero means the normal full-run semantics understood by train.py.
    return max_train_steps, 0


def _future_training_checkpoint_steps(
    *, current_step: int, target_step: int, save_interval: int
) -> tuple[int, ...]:
    """Return every checkpoint the trainer can write in this invocation.

    ``flowmap_trainer`` saves after each positive periodic boundary and always
    writes the requested final step.  A resume therefore has more write
    destinations than just its final checkpoint; reserve all of them before
    permitting the train subprocess to start.
    """
    current_step = _validate_positive_int(
        current_step, name="current_step", allow_zero=True
    )
    target_step = _validate_positive_int(target_step, name="target_step")
    save_interval = _validate_positive_int(save_interval, name="save_interval")
    if target_step <= current_step:
        raise ValueError("target_step must be greater than current_step")
    first_periodic_step = ((current_step // save_interval) + 1) * save_interval
    periodic_steps = range(first_periodic_step, target_step + 1, save_interval)
    return tuple(sorted({*periodic_steps, target_step}))


def build_multibudget_eval_plans(
    *,
    checkpoint_dir: str | Path,
    dataset_path: str | Path,
    selection_manifest: str | Path,
    eval_pairs: str | Path,
    shared_protocol_root: str | Path,
    output_dir: str | Path,
    teacher_model_path: str | Path,
    mixed_policy_name: str | None = None,
    python_executable: str | Path | None = None,
    eval_device_list: str = "0",
    eval_worker_device_list: str = "0",
) -> list[dict[str, Any]]:
    """Build isolated fixed-cache S1/S2/S4 evaluator child plans.

    The t4 and t8 cache directories intentionally differ because cache file
    names do not encode teacher-step count.  This helper only builds argv/env;
    it never launches evaluation or cache construction.
    """
    checkpoint_dir = _as_path(checkpoint_dir)
    dataset_path = _as_path(dataset_path)
    selection_manifest = _as_path(selection_manifest)
    eval_pairs = _as_path(eval_pairs)
    shared_protocol_root = _as_path(shared_protocol_root)
    output_dir = _as_path(output_dir)
    teacher_model_path = _as_path(teacher_model_path)
    python_executable = str(python_executable or sys.executable)
    eval_device_list = str(eval_device_list).strip()
    eval_worker_device_list = str(eval_worker_device_list).strip()
    if not eval_device_list or not eval_worker_device_list:
        raise ValueError("Evaluator CUDA device lists must be non-empty")

    try:
        checkpoint_step = int(checkpoint_dir.name.removeprefix("step_"))
    except ValueError:
        checkpoint_step = None
    checkpoint_label = (
        f"step_{checkpoint_step}" if checkpoint_step is not None else checkpoint_dir.name
    )
    metrics_dir = output_dir / "metrics" / "selection" / checkpoint_label
    base_env = _as_str_env(
        {
            "CONFIG_FILE": "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive",
            "COSMOS_PROGRESSIVE_STAGE": "s4",
            "CUDA_VISIBLE_DEVICES": eval_device_list,
            "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES": eval_worker_device_list,
            "CACHE_DATASET_IN_MEMORY": "0",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    if mixed_policy_name:
        base_env["COSMOS_MIXED_STEP_POLICY"] = str(mixed_policy_name)
    plans: list[dict[str, Any]] = []
    for budget, teacher_steps, student_steps, cache_name in _BUDGET_SPECS:
        cache_dir = shared_protocol_root / "teacher_cache" / "selection" / cache_name
        output_json = metrics_dir / f"{budget}.json"
        argv = [
            python_executable,
            "distillation_flowmap/eval_cosmos_progressive_stage2.py",
            "--checkpoint-transformer",
            str(_checkpoint_transformer_dir(checkpoint_dir)),
            "--config",
            "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive",
            "--teacher-model-path",
            str(teacher_model_path),
            "--dataset-path",
            str(dataset_path),
            "--manifest",
            str(selection_manifest),
            "--pairs",
            str(eval_pairs),
            "--cache-dir",
            str(cache_dir),
            "--student-steps",
            str(student_steps),
            "--teacher-steps",
            str(teacher_steps),
            "--output-json",
            str(output_json),
        ]
        plans.append(
            {
                "budget": budget,
                "teacher_steps": teacher_steps,
                "student_steps": student_steps,
                "checkpoint_dir": checkpoint_dir,
                "checkpoint_transformer": _checkpoint_transformer_dir(checkpoint_dir),
                "cache_dir": cache_dir,
                "output_json": output_json,
                "env": dict(base_env),
                "argv": argv,
            }
        )
    return plans


def _raise_invalid_eval_execution_plan(detail: str) -> None:
    raise ValueError(
        "Cosmos mixed-step eval execution plan violates approved evaluator-only "
        f"contract: {detail}"
    )


def validate_policy_eval_execution_contract(
    plan: Mapping[str, Any],
    *,
    checkpoint_dir: str | Path,
    protocol_metadata_root: str | Path | None = None,
) -> None:
    """Reject malformed public eval plans before any evaluator subprocess runs.

    ``execute_policy_eval_plan`` accepts a mapping to keep plan generation
    testable, so it must not trust a caller-provided ``eval_plans`` list.  This
    validates the exact approved S1/S2/S4 evaluator shape, fixed t4/t4/t8
    caches, output locations, and evaluator-only argv tail before execution.
    """
    try:
        policy_name = _require_supported_policy_name(plan["policy"]["name"])
        root = _resolve_path(plan["root"])
        protocol_root = _resolve_path(plan["protocol_source_root"])
        dataset_path = _resolve_path(plan["dataset_path"])
        teacher_model_path = _resolve_path(plan["teacher_model_path"])
        stage1_checkpoint = _resolve_path(plan["stage1_checkpoint"])
        selection_proxy_path = _resolve_path(plan["selection_proxy_path"])
        eval_plans = plan["eval_plans"]
    except (KeyError, TypeError, ValueError) as exc:
        _raise_invalid_eval_execution_plan(f"missing or invalid plan metadata: {exc}")
    if isinstance(eval_plans, (str, bytes)) or not isinstance(eval_plans, Sequence):
        _raise_invalid_eval_execution_plan("eval_plans must be a three-item sequence")
    if len(eval_plans) != len(_BUDGET_SPECS):
        _raise_invalid_eval_execution_plan(
            f"expected exactly {len(_BUDGET_SPECS)} eval plans, got {len(eval_plans)}"
        )
    checkpoint_dir = _resolve_path(checkpoint_dir)
    try:
        artifact_paths = validate_policy_eval_artifact_isolation(
            root=root,
            checkpoint_dir=checkpoint_dir,
            protocol_source_root=protocol_root,
            dataset_path=dataset_path,
            teacher_model_path=teacher_model_path,
            stage1_checkpoint=stage1_checkpoint,
            selection_proxy_path=selection_proxy_path,
        )
    except ValueError as exc:
        _raise_invalid_eval_execution_plan(str(exc))
    if protocol_metadata_root is None:
        metadata_root = protocol_root
    else:
        metadata_root = _resolve_path(protocol_metadata_root)
        expected_metadata_root = root / "protocol"
        if metadata_root != expected_metadata_root:
            _raise_invalid_eval_execution_plan(
                "training evaluation metadata must be the canonical root/protocol copy"
            )
    metrics_dir = root / "metrics" / "selection" / checkpoint_dir.name
    expected_selection_proxy_path = artifact_paths["selection proxy"]
    if selection_proxy_path != expected_selection_proxy_path:
        _raise_invalid_eval_execution_plan(
            "selection_proxy_path is not the owned checkpoint selection proxy"
        )
    trusted_python_executable = _resolve_path(sys.executable)
    for eval_plan, (budget, teacher_steps, student_steps, cache_name) in zip(
        eval_plans, _BUDGET_SPECS
    ):
        if not isinstance(eval_plan, Mapping):
            _raise_invalid_eval_execution_plan(f"{budget} plan is not a mapping")
        expected_cache_dir = protocol_root / "teacher_cache" / "selection" / cache_name
        expected_output_json = metrics_dir / f"{budget}.json"
        expected_transformer = _checkpoint_transformer_dir(checkpoint_dir)
        try:
            actual_budget = str(eval_plan["budget"])
            actual_teacher_steps = int(eval_plan["teacher_steps"])
            actual_student_steps = int(eval_plan["student_steps"])
            actual_checkpoint = _resolve_path(eval_plan["checkpoint_dir"])
            actual_transformer = _resolve_path(eval_plan["checkpoint_transformer"])
            actual_cache_dir = _resolve_path(eval_plan["cache_dir"])
            actual_output_json = _resolve_path(eval_plan["output_json"])
            env = eval_plan["env"]
            argv = eval_plan["argv"]
        except (KeyError, TypeError, ValueError) as exc:
            _raise_invalid_eval_execution_plan(f"{budget} plan is malformed: {exc}")
        if (
            actual_budget != budget
            or actual_teacher_steps != teacher_steps
            or actual_student_steps != student_steps
            or actual_checkpoint != checkpoint_dir
            or actual_transformer != _resolve_path(expected_transformer)
            or actual_cache_dir != _resolve_path(expected_cache_dir)
            or actual_output_json != _resolve_path(expected_output_json)
        ):
            _raise_invalid_eval_execution_plan(
                f"{budget} must use ({teacher_steps},{student_steps}) with {cache_name} cache"
            )
        if not isinstance(env, Mapping):
            _raise_invalid_eval_execution_plan(f"{budget} env is not a mapping")
        required_env = {
            "CONFIG_FILE": "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive",
            "COSMOS_PROGRESSIVE_STAGE": "s4",
            "COSMOS_MIXED_STEP_POLICY": policy_name,
            "CACHE_DATASET_IN_MEMORY": "0",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
        if any(str(env.get(key, "")) != value for key, value in required_env.items()):
            _raise_invalid_eval_execution_plan(f"{budget} env is not the approved proxy env")
        if not str(env.get("CUDA_VISIBLE_DEVICES", "")).strip() or not str(
            env.get("COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES", "")
        ).strip():
            _raise_invalid_eval_execution_plan(f"{budget} env has an empty evaluator device list")
        if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
            _raise_invalid_eval_execution_plan(f"{budget} argv is not a sequence")
        argv = list(argv)
        if (
            not argv
            or not isinstance(argv[0], str)
            or argv[0] != str(trusted_python_executable)
        ):
            _raise_invalid_eval_execution_plan(
                f"{budget} argv does not use the trusted intended Python executable"
            )
        expected_argv_tail = [
            "distillation_flowmap/eval_cosmos_progressive_stage2.py",
            "--checkpoint-transformer",
            str(expected_transformer),
            "--config",
            "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive",
            "--teacher-model-path",
            str(teacher_model_path),
            "--dataset-path",
            str(dataset_path),
            "--manifest",
            str(metadata_root / "selection_manifest.json"),
            "--pairs",
            str(metadata_root / "eval_pairs.json"),
            "--cache-dir",
            str(expected_cache_dir),
            "--student-steps",
            str(student_steps),
            "--teacher-steps",
            str(teacher_steps),
            "--output-json",
            str(expected_output_json),
        ]
        if argv[1:] != expected_argv_tail or "distillation_flowmap/train.py" in argv:
            _raise_invalid_eval_execution_plan(
                f"{budget} argv must be the approved evaluator-only command"
            )


def _raise_invalid_train_execution_plan(detail: str) -> None:
    raise ValueError(
        "Cosmos mixed-step training execution plan violates the owned-artifact "
        f"contract: {detail}"
    )


def _single_argv_option_value(
    argv: Sequence[Any], option: str, *, plan_kind: str
) -> str:
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
        _raise_invalid_train_execution_plan(f"{plan_kind} argv is not a sequence")
    values: list[str] = []
    for index, token in enumerate(argv):
        if token == option:
            if index + 1 >= len(argv):
                _raise_invalid_train_execution_plan(
                    f"{plan_kind} argv has no value after {option}"
                )
            values.append(str(argv[index + 1]))
    if len(values) != 1:
        _raise_invalid_train_execution_plan(
            f"{plan_kind} argv must contain exactly one {option}"
        )
    return values[0]


def _validate_training_artifact_containment(
    *,
    root: Path,
    sources: Sequence[tuple[str, Path]],
    artifacts: Sequence[tuple[str, Path]],
) -> dict[str, Path]:
    resolved_artifacts: dict[str, Path] = {}
    for label, artifact in artifacts:
        _reject_symlinked_owned_artifact_path(root, artifact, label=label)
        resolved_artifact = _resolve_path(artifact)
        if not _is_within(resolved_artifact, root):
            raise ValueError(
                "Cosmos mixed-step training artifact must resolve under policy root: "
                f"artifact={label}, path={resolved_artifact}, root={root}"
            )
        for source_label, source in sources:
            if _paths_overlap(resolved_artifact, source):
                raise ValueError(
                    "Cosmos mixed-step training artifact must not overlap immutable "
                    f"{source_label}: artifact={label}, path={resolved_artifact}, "
                    f"source={source}"
                )
        resolved_artifacts[label] = resolved_artifact
    return resolved_artifacts


def _validate_protocol_metadata_copy_contract(plan: Mapping[str, Any]) -> list[tuple[Path, Path]]:
    """Validate protocol copy sources/destinations before any mkdir or copy.

    This deliberately remains narrower than the full execution contract so the
    public copy helper is safe even when it is called directly in a dry test.
    In particular, it rejects a nested ``root/protocol`` symlink before the
    first destination parent is created.
    """
    try:
        root = _resolve_path(plan["root"])
        source_root = _resolve_path(plan["protocol_source_root"])
        destination_root = _as_path(plan["protocol_dir"])
    except (KeyError, TypeError, OSError) as exc:
        _raise_invalid_train_execution_plan(
            f"missing protocol copy metadata: {exc}"
        )
    expected_destination_root = root / "protocol"
    if _resolve_path(destination_root) != _resolve_path(expected_destination_root) or (
        _as_path(destination_root) != expected_destination_root
    ):
        _raise_invalid_train_execution_plan(
            "protocol_dir is not the canonical root/protocol destination"
        )
    pairs: list[tuple[Path, Path]] = []
    for filename in _METADATA_FILENAMES:
        source = source_root / filename
        destination = expected_destination_root / filename
        _validate_training_artifact_containment(
            root=root,
            sources=(("protocol source root", source_root),),
            artifacts=((f"protocol metadata {filename}", destination),),
        )
        if not source.is_file():
            raise FileNotFoundError(f"Missing immutable protocol metadata: {source}")
        if destination.exists() and not destination.is_file():
            raise FileExistsError(
                "Protocol metadata destination is not a regular file: "
                f"{destination}"
            )
        pairs.append((source, destination))
    return pairs


def validate_policy_train_execution_contract(plan: Mapping[str, Any]) -> dict[str, Path]:
    """Validate every runner-owned training write before build or execution.

    The train process, metadata copy, evaluator JSONs, and proxy writer all
    receive their destinations from a mutable plan mapping.  Resolve and pin
    every one to the canonical independent policy root before a write is even
    possible; then reject any existing output that would be replaced.  The
    sole exception is byte-checked, regular immutable protocol metadata, which
    is handled by :func:`copy_immutable_protocol_metadata`.
    """
    try:
        policy_name = _require_supported_policy_name(plan["policy"]["name"])
        root = _resolve_path(plan["root"])
        output_dir = _resolve_path(plan["output_dir"])
        protocol_source_root = _resolve_path(plan["protocol_source_root"])
        dataset_path = _resolve_path(plan["dataset_path"])
        teacher_model_path = _resolve_path(plan["teacher_model_path"])
        source_checkpoint = validate_common_stage1_checkpoint(plan["source_checkpoint"])
        stage1_checkpoint = _resolve_path(plan["stage1_checkpoint"])
        current_step = int(plan["current_step"])
        target_step = int(plan["target_step"])
        save_interval = _validate_positive_int(
            plan["save_interval"], name="save_interval"
        )
        protocol_dir = _as_path(plan["protocol_dir"])
        checkpoint_dir = _as_path(plan["checkpoint_dir"])
        selection_proxy_path = _as_path(plan["selection_proxy_path"])
        train_env = plan["train_env"]
        train_argv = plan["train_argv"]
    except (KeyError, TypeError, ValueError, OSError) as exc:
        _raise_invalid_train_execution_plan(f"missing or invalid plan metadata: {exc}")
    if current_step < 0 or target_step <= current_step:
        _raise_invalid_train_execution_plan(
            f"invalid current/target steps: current={current_step}, target={target_step}"
        )
    is_resume = current_step > 0
    future_checkpoint_steps = _future_training_checkpoint_steps(
        current_step=current_step,
        target_step=target_step,
        save_interval=save_interval,
    )
    if output_dir != root:
        _raise_invalid_train_execution_plan("output_dir is not the canonical policy root")
    if stage1_checkpoint != source_checkpoint:
        _raise_invalid_train_execution_plan(
            "stage1_checkpoint and source_checkpoint disagree about common provenance"
        )
    validate_policy_root(root, resume=is_resume)
    validate_policy_source_isolation(
        root,
        protocol_source_root=protocol_source_root,
        stage1_checkpoint=source_checkpoint,
        teacher_model_path=teacher_model_path,
        dataset_path=dataset_path,
    )

    expected_protocol_dir = root / "protocol"
    expected_metrics_dir = root / "metrics"
    expected_metrics_jsonl = expected_metrics_dir / "cosmos_mixed_step_opd.jsonl"
    expected_checkpoint_dir = root / "checkpoints" / f"step_{target_step}"
    expected_selection_dir = (
        root / "metrics" / "selection" / f"step_{target_step}"
    )
    expected_selection_proxy = expected_selection_dir / "selection_proxy.json"
    expected_eval_outputs = {
        budget: expected_selection_dir / f"{budget}.json" for budget in _PREFLIGHT_LABELS
    }
    expected_manifest = (
        root / "policy_manifest.json"
        if not is_resume
        else expected_protocol_dir / f"policy_manifest_resume_step_{current_step}.json"
    )
    expected_resume_path = (
        source_checkpoint
        if not is_resume
        else root / "checkpoints" / f"step_{current_step}"
    )
    if _as_path(protocol_dir) != expected_protocol_dir:
        _raise_invalid_train_execution_plan("protocol_dir is not canonical root/protocol")
    if _as_path(checkpoint_dir) != expected_checkpoint_dir:
        _raise_invalid_train_execution_plan(
            "checkpoint_dir is not the canonical target checkpoint"
        )
    if _as_path(selection_proxy_path) != expected_selection_proxy:
        _raise_invalid_train_execution_plan(
            "selection_proxy_path is not the canonical target selection proxy"
        )
    if is_resume:
        _reject_symlinked_owned_artifact_path(
            root, expected_resume_path, label="resume checkpoint"
        )
        owned_resume_path = validate_owned_policy_checkpoint(
            root=root, checkpoint_dir=expected_resume_path
        )
        if owned_resume_path != expected_resume_path:
            _raise_invalid_train_execution_plan(
                "resume checkpoint is not the expected historical policy checkpoint"
            )

    sources = (
        ("protocol source root", protocol_source_root),
        ("dataset path", dataset_path),
        ("teacher model path", teacher_model_path),
        ("Stage-1 source checkpoint", source_checkpoint),
        ("legacy progressive S4 root", _resolve_path(LEGACY_PROGRESSIVE_ROOT)),
    )
    artifacts: list[tuple[str, Path]] = [
        ("protocol directory", expected_protocol_dir),
        *[
            (f"protocol metadata {filename}", expected_protocol_dir / filename)
            for filename in _METADATA_FILENAMES
        ],
        ("metrics directory", expected_metrics_dir),
        ("rank-zero selection JSONL", expected_metrics_jsonl),
        *[
            (f"future checkpoint step_{step}", root / "checkpoints" / f"step_{step}")
            for step in future_checkpoint_steps
        ],
        *[
            (
                f"future checkpoint step_{step} transformer",
                _checkpoint_transformer_dir(root / "checkpoints" / f"step_{step}"),
            )
            for step in future_checkpoint_steps
        ],
        ("selection metrics directory", expected_selection_dir),
        ("selection proxy", expected_selection_proxy),
        ("policy manifest", expected_manifest),
        ("policy manifest temporary file", expected_manifest.with_name(expected_manifest.name + ".tmp")),
        ("selection proxy temporary file", expected_selection_proxy.with_name(expected_selection_proxy.name + ".tmp")),
    ]
    for budget, output_json in expected_eval_outputs.items():
        artifacts.extend(
            (
                (f"{budget} evaluation JSON", output_json),
                (
                    f"{budget} evaluation JSON temporary file",
                    output_json.with_name(output_json.name + ".tmp"),
                ),
            )
        )
    resolved_artifacts = _validate_training_artifact_containment(
        root=root,
        sources=sources,
        artifacts=artifacts,
    )
    for label, directory in (
        ("protocol directory", expected_protocol_dir),
        ("metrics directory", expected_metrics_dir),
        ("selection metrics directory", expected_selection_dir),
        ("checkpoints directory", expected_checkpoint_dir.parent),
    ):
        _require_existing_directory_or_missing(directory, label=label)
    for filename in _METADATA_FILENAMES:
        destination = expected_protocol_dir / filename
        if destination.exists() and not destination.is_file():
            raise FileExistsError(
                "Protocol metadata destination is not a regular file: "
                f"{destination}"
            )

    for step in future_checkpoint_steps:
        future_checkpoint_dir = root / "checkpoints" / f"step_{step}"
        _reject_existing_output_artifact(
            future_checkpoint_dir, label=f"future checkpoint step_{step}"
        )
        _reject_existing_output_artifact(
            _checkpoint_transformer_dir(future_checkpoint_dir),
            label=f"future checkpoint step_{step} transformer",
        )
    _reject_existing_output_artifact(expected_selection_proxy, label="selection proxy")
    _reject_existing_output_artifact(
        expected_selection_proxy.with_name(expected_selection_proxy.name + ".tmp"),
        label="selection proxy temporary file",
    )
    for budget, output_json in expected_eval_outputs.items():
        _reject_existing_output_artifact(output_json, label=f"{budget} evaluation JSON")
        _reject_existing_output_artifact(
            output_json.with_name(output_json.name + ".tmp"),
            label=f"{budget} evaluation JSON temporary file",
        )
    if is_resume:
        validate_resume_policy_ownership(
            root,
            policy_name,
            expected_source_checkpoint=source_checkpoint,
        )
        if expected_metrics_jsonl.exists() and not expected_metrics_jsonl.is_file():
            raise FileExistsError(
                "Resume Cosmos mixed-step metrics JSONL is not a regular append-only "
                f"file: {expected_metrics_jsonl}"
            )
    else:
        _reject_existing_output_artifact(
            expected_metrics_jsonl, label="rank-zero selection JSONL"
        )
    _reject_existing_output_artifact(expected_manifest, label="policy manifest")
    _reject_existing_output_artifact(
        expected_manifest.with_name(expected_manifest.name + ".tmp"),
        label="policy manifest temporary file",
    )
    if not is_resume and (root / "policy_manifest.json").exists():
        _raise_invalid_train_execution_plan("fresh root unexpectedly has policy_manifest.json")

    if not isinstance(train_env, Mapping):
        _raise_invalid_train_execution_plan("train_env is not a mapping")
    required_env = {
        "CONFIG_FILE": "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive",
        "COSMOS_PROGRESSIVE_STAGE": "s4",
        "COSMOS_MIXED_STEP_POLICY": policy_name,
        "COSMOS_MIXED_STEP_METRICS_PATH": str(expected_metrics_jsonl),
        "COSMOS_PROGRESSIVE_OUTPUT_ROOT": str(root),
        "OUTPUT_DIR": str(root),
        "RESUME_FROM_PATH": str(expected_resume_path),
        "RESUME_ONLINE_FROM_TARGET": "0",
        "SKIP_TARGET_STUDENT_FOR_COSMOS_LATENT": "1",
        "SAVE_INTERVAL": str(save_interval),
    }
    if any(str(train_env.get(key, "")) != value for key, value in required_env.items()):
        _raise_invalid_train_execution_plan(
            "train_env does not preserve the canonical owned output/source paths"
        )
    if _resolve_path(
        _single_argv_option_value(train_argv, "--output-dir", plan_kind="train")
    ) != root:
        _raise_invalid_train_execution_plan("train argv output-dir is not the policy root")
    if _resolve_path(
        _single_argv_option_value(train_argv, "--resume-from-path", plan_kind="train")
    ) != expected_resume_path:
        _raise_invalid_train_execution_plan(
            "train argv resume-from-path is not the approved source"
        )

    preflight_evidence = bool(plan.get("preflight_evidence", False))
    if preflight_evidence:
        if (
            policy_name != "universe"
            or current_step != 0
            or target_step != 9
            or int(plan.get("max_train_steps", -1)) != 9
            or int(plan.get("save_interval", -1)) != 9
            or int(plan.get("stop_after_step", -1)) != 9
            or tuple(int(index) for index in plan.get("forced_indices", ()))
            != _PREFLIGHT_FORCE_INDICES
            or str(train_env.get("OPD_AUX_WARMUP_STEPS", "")) != "0"
            or str(train_env.get("OPD_AUX_INTERVAL", "")) != "1"
        ):
            _raise_invalid_train_execution_plan(
                "preflight evidence mode must be the fixed 9-step forced Universe contract"
            )
    elif "OPD_AUX_WARMUP_STEPS" in train_env or "OPD_AUX_INTERVAL" in train_env:
        _raise_invalid_train_execution_plan(
            "full policy plans must keep the default auxiliary schedule"
        )

    try:
        validate_policy_eval_execution_contract(
            plan,
            checkpoint_dir=expected_checkpoint_dir,
            protocol_metadata_root=expected_protocol_dir,
        )
    except ValueError as exc:
        _raise_invalid_train_execution_plan(str(exc))
    return resolved_artifacts


def validate_owned_policy_checkpoint(
    *, root: str | Path, checkpoint_dir: str | Path
) -> Path:
    """Require an evaluation checkpoint to be a direct checkpoint of ``root``.

    Evaluation is deliberately unable to point a policy root at a checkpoint
    owned by a different experiment.  Resolving both sides also rejects a
    symlink inside ``root/checkpoints`` that escapes into another output tree.
    """
    resolved_root = _resolve_path(root)
    resolved_checkpoint = _resolve_path(checkpoint_dir)
    checkpoints_dir = resolved_root / "checkpoints"
    checkpoint_name = resolved_checkpoint.name
    checkpoint_suffix = checkpoint_name.removeprefix("step_")
    if (
        not checkpoint_name.startswith("step_")
        or not checkpoint_suffix
        or not checkpoint_suffix.isascii()
        or not checkpoint_suffix.isdecimal()
    ):
        raise ValueError(
            "Cosmos mixed-step evaluation checkpoint must be a direct "
            f"owned checkpoints/step_N directory: {resolved_checkpoint}"
        )
    if resolved_checkpoint.parent != checkpoints_dir:
        raise ValueError(
            "Cosmos mixed-step evaluation checkpoint must be a direct "
            f"owned checkpoints/step_N directory: {resolved_checkpoint}"
        )
    return resolved_checkpoint


def build_policy_eval_plan(
    *,
    policy_name: str,
    root: str | Path,
    checkpoint_dir: str | Path,
    dataset_path: str | Path,
    protocol_source_root: str | Path,
    stage1_checkpoint: str | Path = DEFAULT_STAGE1_CHECKPOINT,
    teacher_model_path: str | Path = DEFAULT_TEACHER_MODEL,
    legacy_root: str | Path = LEGACY_PROGRESSIVE_ROOT,
    eval_device_list: str = "0",
    eval_worker_device_list: str = "0",
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Build a target-free three-budget evaluation plan without any training.

    This is intentionally separate from :func:`build_policy_train_plan` so a
    launcher cannot accidentally retrain when the requested operation is an
    offline fixed-cache proxy evaluation.  The plan owns no train argv and
    only has the reviewed S1/S2/S4 evaluator child plans.
    """
    policy = _policy_payload(policy_name)
    root = validate_policy_root(root, legacy_root=legacy_root, resume=True)
    protocol_source_root = _resolve_path(protocol_source_root)
    dataset_path = _resolve_path(dataset_path)
    stage1_checkpoint = _resolve_path(stage1_checkpoint)
    teacher_model_path = _resolve_path(teacher_model_path)
    python_executable = _resolve_path(python_executable or sys.executable)
    validate_policy_source_isolation(
        root,
        protocol_source_root=protocol_source_root,
        stage1_checkpoint=stage1_checkpoint,
        teacher_model_path=teacher_model_path,
        dataset_path=dataset_path,
    )
    validate_resume_policy_ownership(root, policy["name"])
    checkpoint_dir = validate_owned_policy_checkpoint(
        root=root, checkpoint_dir=checkpoint_dir
    )
    artifact_paths = validate_policy_eval_artifact_isolation(
        root=root,
        checkpoint_dir=checkpoint_dir,
        protocol_source_root=protocol_source_root,
        dataset_path=dataset_path,
        teacher_model_path=teacher_model_path,
        stage1_checkpoint=stage1_checkpoint,
    )
    selection_manifest = protocol_source_root / "selection_manifest.json"
    eval_pairs = protocol_source_root / "eval_pairs.json"
    eval_plans = build_multibudget_eval_plans(
        checkpoint_dir=checkpoint_dir,
        dataset_path=dataset_path,
        selection_manifest=selection_manifest,
        eval_pairs=eval_pairs,
        shared_protocol_root=protocol_source_root,
        output_dir=root,
        teacher_model_path=teacher_model_path,
        mixed_policy_name=policy["name"],
        python_executable=python_executable,
        eval_device_list=eval_device_list,
        eval_worker_device_list=eval_worker_device_list,
    )
    return {
        "schema": "cosmos_mixed_step_policy_eval_plan_v1",
        "policy": policy,
        "root": root,
        "output_dir": root,
        "protocol_source_root": protocol_source_root,
        "dataset_path": dataset_path,
        "stage1_checkpoint": stage1_checkpoint,
        "teacher_model_path": teacher_model_path,
        "python_executable": python_executable,
        "checkpoint_dir": checkpoint_dir,
        "selection_proxy_path": artifact_paths["selection proxy"],
        "eval_plans": eval_plans,
    }


def build_policy_train_plan(
    *,
    policy_name: str,
    root: str | Path,
    dataset_path: str | Path,
    protocol_source_root: str | Path,
    stage1_checkpoint: str | Path = DEFAULT_STAGE1_CHECKPOINT,
    current_step: int = 0,
    chunk_size: int = 0,
    max_train_steps: int = DEFAULT_MAX_TRAIN_STEPS,
    save_interval: int = DEFAULT_SAVE_INTERVAL,
    stop_after_step: int = 0,
    master_port: int = DEFAULT_MASTER_PORT,
    train_seed: int = DEFAULT_TRAIN_SEED,
    torchrun: str | Path = DEFAULT_TORCHRUN,
    teacher_model_path: str | Path = DEFAULT_TEACHER_MODEL,
    device_list: str | Sequence[int | str] = DEFAULT_DEVICE_LIST,
    world_size: int = DEFAULT_WORLD_SIZE,
    force_sequence: str | Sequence[int | str] | None = None,
    preflight_evidence: bool = False,
    opd_aux_warmup_steps: int | None = None,
    opd_aux_interval: int | None = None,
    resume_from_path: str | Path | None = None,
    legacy_root: str | Path = LEGACY_PROGRESSIVE_ROOT,
    eval_device_list: str = "0",
    eval_worker_device_list: str = "0",
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Build an independent initial or resume policy plan without side effects."""
    policy = _policy_payload(policy_name)
    policy_spec = get_mixed_step_policy_spec(policy["name"])
    devices = parse_device_list(device_list, world_size=world_size)
    world_size = int(world_size)
    current_step = _validate_positive_int(
        current_step, name="current_step", allow_zero=True
    )
    max_train_steps = _validate_positive_int(max_train_steps, name="max_train_steps")
    save_interval = _validate_positive_int(save_interval, name="save_interval")
    target_step, effective_stop_after = _plan_target_step(
        current_step=current_step,
        chunk_size=chunk_size,
        max_train_steps=max_train_steps,
        stop_after_step=stop_after_step,
    )
    initial_plan = current_step == 0
    root = validate_policy_root(root, legacy_root=legacy_root, resume=not initial_plan)
    output_dir = root
    stage1_checkpoint = validate_common_stage1_checkpoint(stage1_checkpoint)
    protocol_source_root = _resolve_path(protocol_source_root)
    teacher_model_path = _resolve_path(teacher_model_path)
    dataset_path = _resolve_path(dataset_path)
    python_executable = _resolve_path(python_executable or sys.executable)
    validate_policy_source_isolation(
        output_dir,
        protocol_source_root=protocol_source_root,
        stage1_checkpoint=stage1_checkpoint,
        teacher_model_path=teacher_model_path,
        dataset_path=dataset_path,
    )

    if initial_plan:
        if resume_from_path is not None and _resolve_path(resume_from_path) != _resolve_path(
            stage1_checkpoint
        ):
            raise ValueError(
                "A fresh mixed-step policy must start from the common Stage-1 "
                "online student, not an arbitrary resume checkpoint"
            )
        train_resume_path = stage1_checkpoint
        reset_resume_step = "1"
        resume_optimizer_state = "0"
    else:
        validate_resume_policy_ownership(
            output_dir,
            policy["name"],
            expected_source_checkpoint=stage1_checkpoint,
        )
        own_checkpoint = output_dir / "checkpoints" / f"step_{current_step}"
        if resume_from_path is not None and _resolve_path(resume_from_path) != _resolve_path(
            own_checkpoint
        ):
            raise ValueError(
                "A mixed-step resume must use this policy's own online checkpoint: "
                f"{own_checkpoint}"
            )
        train_resume_path = own_checkpoint
        reset_resume_step = "0"
        resume_optimizer_state = "1"

    forced_sequence = "" if force_sequence is None else force_sequence
    forced_sequence_for_env = (
        str(forced_sequence).strip()
        if isinstance(forced_sequence, str)
        else ",".join(str(item).strip() for item in forced_sequence)
    )
    forced_indices = parse_forced_indices(forced_sequence_for_env, policy_spec)
    preflight_evidence = bool(preflight_evidence)
    if preflight_evidence:
        if (
            policy["name"] != "universe"
            or current_step != 0
            or target_step != 9
            or max_train_steps != 9
            or save_interval != 9
            or effective_stop_after != 9
            or tuple(forced_indices) != _PREFLIGHT_FORCE_INDICES
            or opd_aux_warmup_steps is None
            or opd_aux_interval is None
        ):
            raise ValueError(
                "Preflight evidence mode requires the fixed 9-step forced Universe "
                "schedule and explicit auxiliary overrides"
            )
        opd_aux_warmup_steps = _validate_positive_int(
            opd_aux_warmup_steps, name="opd_aux_warmup_steps", allow_zero=True
        )
        opd_aux_interval = _validate_positive_int(
            opd_aux_interval, name="opd_aux_interval"
        )
        if opd_aux_warmup_steps != 0 or opd_aux_interval != 1:
            raise ValueError(
                "Preflight evidence mode permits only OPD_AUX_WARMUP_STEPS=0 "
                "and OPD_AUX_INTERVAL=1"
            )
    elif opd_aux_warmup_steps is not None or opd_aux_interval is not None:
        raise ValueError(
            "Auxiliary schedule overrides are reserved for controlled preflight evidence"
        )
    protocol_dir = output_dir / "protocol"
    train_manifest = protocol_dir / "train_manifest.json"
    selection_manifest = protocol_dir / "selection_manifest.json"
    eval_pairs = protocol_dir / "eval_pairs.json"
    checkpoint_dir = output_dir / "checkpoints" / f"step_{target_step}"
    objective_settings = {
        "progressive_template_stage": "s4",
        "full_cosmos_objective": True,
        "target_free_online_student": True,
        "mixed_endpoint_sampling": True,
        "cosmos_mixed_step_policy": policy["name"],
        "rollout_step_pairs": policy["rollout_step_pairs"],
        "rollout_step_pair_weights": policy["weights"],
        "gradient_accumulation_steps": 1,
        "opd_aux_standalone_step": True,
        "opd_serial_student_cfg": True,
        "opd_spatial_crop_size": 28,
    }
    train_env = _as_str_env(
        {
            "CONFIG_FILE": "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive",
            # The old stage names choose full-objective hyperparameters.  The
            # actual S1/S2/S4 endpoint distribution comes only from the mixed
            # policy, never from a non-existent `universe` legacy stage.
            "COSMOS_PROGRESSIVE_STAGE": "s4",
            "COSMOS_MIXED_STEP_POLICY": policy["name"],
            "COSMOS_MIXED_STEP_SELECTOR_SEED": int(train_seed),
            "COSMOS_MIXED_STEP_FORCE_SEQUENCE": forced_sequence_for_env,
            "COSMOS_MIXED_STEP_METRICS_PATH": output_dir
            / "metrics"
            / "cosmos_mixed_step_opd.jsonl",
            "CUDA_VISIBLE_DEVICES": ",".join(devices),
            "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES": ",".join(devices),
            "CACHE_DATASET_IN_MEMORY": "0",
            "COSMOS_PROGRESSIVE_OUTPUT_ROOT": output_dir,
            "OUTPUT_DIR": output_dir,
            "RESUME_FROM_PATH": train_resume_path,
            "RESUME_ONLINE_FROM_TARGET": "0",
            "RESET_RESUME_STEP": reset_resume_step,
            "RESUME_OPTIMIZER_STATE": resume_optimizer_state,
            "SKIP_TARGET_STUDENT_FOR_COSMOS_LATENT": "1",
            "MAX_TRAIN_STEPS": max_train_steps,
            "SAVE_INTERVAL": save_interval,
            "STOP_AFTER_STEP": effective_stop_after,
            "DATASET_SAMPLE_MANIFEST": train_manifest,
            "TRAIN_SEED": int(train_seed),
            "ENABLE_WANDB": "0",
            "USE_FSDP1": "1",
            "GRADIENT_CHECKPOINTING": "1",
            "OPD_AUX_GRADIENT_CHECKPOINTING": "1",
            "OPD_SERIAL_STUDENT_CFG": "1",
            "OPD_AUX_STANDALONE_STEP": "1",
            "OPD_COSMOS_SPATIAL_CROP_SIZE": "28",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    if preflight_evidence:
        train_env["OPD_AUX_WARMUP_STEPS"] = str(opd_aux_warmup_steps)
        train_env["OPD_AUX_INTERVAL"] = str(opd_aux_interval)
    train_argv = build_torchrun_command(
        torchrun=torchrun,
        world_size=world_size,
        master_port=master_port,
        teacher_model_path=teacher_model_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        resume_from_path=train_resume_path,
    )
    eval_plans = build_multibudget_eval_plans(
        checkpoint_dir=checkpoint_dir,
        dataset_path=dataset_path,
        selection_manifest=selection_manifest,
        eval_pairs=eval_pairs,
        shared_protocol_root=protocol_source_root,
        output_dir=output_dir,
        teacher_model_path=teacher_model_path,
        mixed_policy_name=policy["name"],
        python_executable=python_executable,
        eval_device_list=eval_device_list,
        eval_worker_device_list=eval_worker_device_list,
    )
    plan = {
        "schema": "cosmos_mixed_step_policy_plan_v1",
        "policy": policy,
        "root": root,
        "output_dir": output_dir,
        "protocol_source_root": protocol_source_root,
        "protocol_dir": protocol_dir,
        "dataset_path": dataset_path,
        "teacher_model_path": teacher_model_path,
        "source_checkpoint": stage1_checkpoint,
        "stage1_checkpoint": stage1_checkpoint,
        "resume_from_path": train_resume_path,
        "current_step": current_step,
        "target_step": target_step,
        "max_train_steps": max_train_steps,
        "save_interval": save_interval,
        "stop_after_step": effective_stop_after,
        "checkpoint_dir": checkpoint_dir,
        "selection_proxy_path": output_dir
        / "metrics"
        / "selection"
        / f"step_{target_step}"
        / "selection_proxy.json",
        "train_seed": int(train_seed),
        "device_list": list(devices),
        "world_size": world_size,
        "master_port": _validate_master_port(master_port),
        "forced_sequence": forced_sequence_for_env,
        "forced_indices": list(forced_indices),
        "preflight_evidence": preflight_evidence,
        "opd_aux_warmup_steps": opd_aux_warmup_steps,
        "opd_aux_interval": opd_aux_interval,
        "objective_settings": objective_settings,
        "train_env": train_env,
        "train_argv": train_argv,
        "eval_plans": eval_plans,
    }
    validate_policy_train_execution_contract(plan)
    return plan


def build_policy_manifest_payload(
    plan: Mapping[str, Any],
    *,
    git_hash: str,
    dirty_diff_hash: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build the immutable, JSON-safe provenance payload written before launch."""
    timestamp = timestamp or _datetime.datetime.now(
        tz=_datetime.timezone.utc
    ).isoformat()
    visible_devices = [
        int(device) if str(device).isdecimal() else str(device)
        for device in plan["device_list"]
    ]
    return {
        "schema": "cosmos_mixed_step_policy_manifest_v1",
        "created_at": timestamp,
        "git_hash": str(git_hash),
        "dirty_diff_hash": str(dirty_diff_hash),
        "root": str(plan["root"]),
        "policy": _json_ready(plan["policy"]),
        "seed": int(plan["train_seed"]),
        "source_checkpoint": str(plan["source_checkpoint"]),
        "resume_from_path": str(plan["resume_from_path"]),
        "protocol_source_root": str(plan["protocol_source_root"]),
        "protocol_dir": str(plan["protocol_dir"]),
        "dataset_path": str(plan["dataset_path"]),
        "visible_devices": visible_devices,
        "world_size": int(plan["world_size"]),
        "master_port": int(plan["master_port"]),
        "current_step": int(plan["current_step"]),
        "target_step": int(plan["target_step"]),
        "max_train_steps": int(plan["max_train_steps"]),
        "save_interval": int(plan["save_interval"]),
        "stop_after_step": int(plan["stop_after_step"]),
        "forced_sequence": str(plan["forced_sequence"]),
        "forced_indices": list(plan["forced_indices"]),
        "objective_settings": _json_ready(plan["objective_settings"]),
        "train_environment": _json_ready(plan["train_env"]),
        "train_command": list(plan["train_argv"]),
        "evaluation_plans": _json_ready(plan["eval_plans"]),
        "selection_proxy_path": str(plan["selection_proxy_path"]),
        "note": "Fixed-cache proxy evaluation only; this manifest does not claim rollout success.",
    }


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _dirty_diff_hash() -> str:
    """Hash tracked diff plus untracked filenames, without adding/staging anything."""
    try:
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=REPO_ROOT
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=REPO_ROOT,
        )
    except Exception:
        return "unknown"
    return hashlib.sha256(diff + b"\0" + status).hexdigest()


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path


def write_policy_manifest(plan: Mapping[str, Any]) -> Path:
    """Write provenance before a launch; never stage, push, or start a job."""
    validate_policy_train_execution_contract(plan)
    root = _resolve_path(plan["root"])
    is_resume = int(plan["current_step"]) > 0
    path = (
        root / "policy_manifest.json"
        if not is_resume
        else root / "protocol" / f"policy_manifest_resume_step_{int(plan['current_step'])}.json"
    )
    payload = build_policy_manifest_payload(
        plan,
        git_hash=_git_hash(),
        dirty_diff_hash=_dirty_diff_hash(),
    )
    return _write_json(path, payload)


def claim_fresh_policy_root(plan: Mapping[str, Any]) -> Path:
    """Atomically claim the empty root of a fresh policy execution.

    Plan generation intentionally has no side effects.  At the explicit run
    boundary, however, a fresh policy must own a previously absent directory;
    this prevents a launcher race from turning an empty-looking child into a
    reused policy root between validation and metadata writes.
    """
    try:
        current_step = int(plan["current_step"])
        root = _as_path(plan["root"])
    except (KeyError, TypeError, ValueError) as exc:
        _raise_invalid_train_execution_plan(
            f"missing fresh policy root metadata: {exc}"
        )
    if current_step != 0:
        _raise_invalid_train_execution_plan(
            "only a fresh policy execution may claim a new policy root"
        )
    root = validate_policy_root(root, resume=False)
    try:
        root.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            "Refusing to reuse an existing fresh policy root; it must be absent "
            f"at execution start: {root}"
        ) from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Fresh policy root parent is missing; the launcher must provide an "
            f"existing ROOT_BASE: {root.parent}"
        ) from exc
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(
            f"Unable to claim a regular fresh policy root: {root}"
        )
    return root


def copy_immutable_protocol_metadata(plan: Mapping[str, Any]) -> list[Path]:
    """Copy only JSON protocol metadata; deliberately never copy teacher caches."""
    # Do this before the first mkdir/copy.  In particular a pre-created
    # root/protocol symlink must never redirect metadata into a source tree.
    metadata_pairs = _validate_protocol_metadata_copy_contract(plan)
    copied: list[Path] = []
    for source, destination in metadata_pairs:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise RuntimeError(
                    "Refusing to replace differing protocol metadata at "
                    f"{destination}"
                )
        else:
            shutil.copy2(source, destination)
            destination.chmod(destination.stat().st_mode & ~0o222)
        copied.append(destination)
    return copied


def _validate_checkpoint_transformer(checkpoint_dir: str | Path, *, role: str) -> None:
    config_path = _checkpoint_transformer_dir(checkpoint_dir) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {role} online student transformer: {config_path}")


def validate_target_free_online_student(checkpoint_dir: str | Path, *, role: str) -> None:
    """Require the deployable online student and reject a target-EMA checkpoint.

    The independent Cosmos mixed policies are explicitly target-free.  This
    small validation is shared by the full train completion path and the
    eval-only path so neither can silently report a target-student artifact as
    a deployable policy.
    """
    _validate_checkpoint_transformer(checkpoint_dir, role=role)
    target_student = _as_path(checkpoint_dir) / "target_student"
    if target_student.exists() or target_student.is_symlink():
        raise RuntimeError(
            f"Expected target-free {role} checkpoint, found target_student: {target_student}"
        )


def validate_eval_caches(
    eval_plans: Sequence[Mapping[str, Any]],
    *,
    protocol_metadata_root: str | Path | None = None,
) -> None:
    """Verify each planned cache has every fixed selection record/pair payload.

    Fresh training plans validate caches before they claim and populate their
    owned ``root/protocol`` copy, so they must read manifest metadata from the
    immutable source.  Eval-only and resumed paths keep validating the copied
    metadata referenced by their evaluator argv.
    """
    metadata_root = (
        _resolve_path(protocol_metadata_root)
        if protocol_metadata_root is not None
        else None
    )
    for plan in eval_plans:
        if metadata_root is None:
            manifest_path = _as_path(
                plan["argv"][plan["argv"].index("--manifest") + 1]
            )
            pairs_path = _as_path(plan["argv"][plan["argv"].index("--pairs") + 1])
        else:
            manifest_path = metadata_root / "selection_manifest.json"
            pairs_path = metadata_root / "eval_pairs.json"
        cache_dir = _as_path(plan["cache_dir"])
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            pairs = json.loads(pairs_path.read_text(encoding="utf-8")).get("pairs", [])
        except FileNotFoundError:
            raise
        if not pairs:
            raise ValueError(f"Evaluation pairs are empty: {pairs_path}")
        missing: list[str] = []
        for record in manifest.get("records", []):
            for pair in pairs:
                filename = f"sample_{int(record['index']):06d}__{pair['pair_id']}.pt"
                if not (cache_dir / filename).is_file():
                    missing.append(str(cache_dir / filename))
                    if len(missing) >= 5:
                        break
            if len(missing) >= 5:
                break
        if missing:
            raise FileNotFoundError(
                "Fixed evaluation cache is incomplete for "
                f"{plan['budget']}: " + ", ".join(missing)
            )


def _write_selection_proxy(plan: Mapping[str, Any]) -> Path:
    try:
        root = _resolve_path(plan["root"])
        checkpoint_dir = _resolve_path(plan["checkpoint_dir"])
        protocol_source_root = _resolve_path(plan["protocol_source_root"])
        dataset_path = _resolve_path(plan["dataset_path"])
        teacher_model_path = _resolve_path(plan["teacher_model_path"])
        stage1_checkpoint = _resolve_path(
            plan.get("source_checkpoint", plan["stage1_checkpoint"])
        )
        selection_proxy_path = _as_path(plan["selection_proxy_path"])
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError(
            "Cosmos mixed-step selection proxy has invalid ownership metadata: "
            f"{exc}"
        ) from exc
    expected_selection_proxy = (
        root / "metrics" / "selection" / checkpoint_dir.name / "selection_proxy.json"
    )
    if selection_proxy_path != expected_selection_proxy:
        raise ValueError(
            "Cosmos mixed-step selection proxy path is not owned by the checkpoint: "
            f"expected={expected_selection_proxy}, got={selection_proxy_path}"
        )
    validate_policy_eval_artifact_isolation(
        root=root,
        checkpoint_dir=checkpoint_dir,
        protocol_source_root=protocol_source_root,
        dataset_path=dataset_path,
        teacher_model_path=teacher_model_path,
        stage1_checkpoint=stage1_checkpoint,
        selection_proxy_path=selection_proxy_path,
    )
    _reject_symlinked_owned_artifact_path(
        root, expected_selection_proxy, label="selection proxy"
    )
    _reject_existing_output_artifact(
        expected_selection_proxy, label="selection proxy"
    )
    _reject_existing_output_artifact(
        expected_selection_proxy.with_name(expected_selection_proxy.name + ".tmp"),
        label="selection proxy temporary file",
    )
    results: dict[str, Any] = {}
    for eval_plan in plan["eval_plans"]:
        output_json = _as_path(eval_plan["output_json"])
        if not output_json.is_file():
            raise RuntimeError(
                f"Evaluator did not write expected {eval_plan['budget']} result: {output_json}"
            )
        results[str(eval_plan["budget"])] = {
            "student_steps": int(eval_plan["student_steps"]),
            "teacher_steps": int(eval_plan["teacher_steps"]),
            "cache_dir": str(eval_plan["cache_dir"]),
            "output_json": str(output_json),
            "metrics": json.loads(output_json.read_text(encoding="utf-8")),
        }
    return _write_json(
        expected_selection_proxy,
        {
            "schema": "cosmos_mixed_step_selection_proxy_v1",
            "checkpoint_dir": str(plan["checkpoint_dir"]),
            "budgets": results,
            "note": "Fixed-cache offline proxy only; not an execution-success claim.",
        },
    )


def execute_policy_plan(plan: Mapping[str, Any]) -> None:
    """Execute a plan only after an explicit ``--run`` CLI opt-in.

    This path has intentionally strict validation and does not build missing
    data/caches, overwrite checkpoints, delete outputs, or start a tmux session.
    """
    is_resume = int(plan["current_step"]) > 0
    validate_policy_root(plan["root"], resume=is_resume)
    validate_policy_source_isolation(
        plan["root"],
        protocol_source_root=plan["protocol_source_root"],
        stage1_checkpoint=plan["source_checkpoint"],
        teacher_model_path=plan["teacher_model_path"],
        dataset_path=plan["dataset_path"],
    )
    validate_common_stage1_checkpoint(plan["source_checkpoint"])
    if is_resume:
        validate_resume_policy_ownership(
            plan["root"],
            plan["policy"]["name"],
            expected_source_checkpoint=plan["source_checkpoint"],
        )
    validate_policy_train_execution_contract(plan)
    _validate_checkpoint_transformer(plan["resume_from_path"], role="resume")
    validate_eval_caches(
        plan["eval_plans"], protocol_metadata_root=plan["protocol_source_root"]
    )
    fresh_root_identity: _FreshPolicyRootIdentity | None = None
    if not is_resume:
        claimed_root = claim_fresh_policy_root(plan)
        fresh_root_identity = _capture_fresh_policy_root_identity(claimed_root)
    if fresh_root_identity is not None:
        _validate_fresh_policy_root_identity(
            fresh_root_identity, phase="protocol copy"
        )
    copy_immutable_protocol_metadata(plan)
    if fresh_root_identity is not None:
        _validate_fresh_policy_root_identity(
            fresh_root_identity, phase="policy manifest"
        )
    manifest_path = write_policy_manifest(plan)
    print(f"Wrote policy manifest: {manifest_path}", flush=True)
    train_env = {**os.environ, **dict(plan["train_env"])}
    if fresh_root_identity is not None:
        _validate_fresh_policy_root_identity(
            fresh_root_identity, phase="training subprocess"
        )
    subprocess.run(list(plan["train_argv"]), cwd=REPO_ROOT, env=train_env, check=True)
    if fresh_root_identity is not None:
        _validate_fresh_policy_root_identity(
            fresh_root_identity, phase="post-training validation"
        )
    validate_target_free_online_student(plan["checkpoint_dir"], role="trained")
    completed_budgets: tuple[str, ...] = ()
    for eval_plan in plan["eval_plans"]:
        if fresh_root_identity is not None:
            _validate_fresh_policy_root_identity(
                fresh_root_identity, phase=f"{eval_plan['budget']} evaluator"
            )
        validate_policy_eval_write_availability(
            plan,
            checkpoint_dir=plan["checkpoint_dir"],
            completed_budgets=completed_budgets,
        )
        eval_env = {**os.environ, **dict(eval_plan["env"])}
        subprocess.run(list(eval_plan["argv"]), cwd=REPO_ROOT, env=eval_env, check=True)
        completed_budgets = (*completed_budgets, str(eval_plan["budget"]))
    if fresh_root_identity is not None:
        _validate_fresh_policy_root_identity(
            fresh_root_identity, phase="selection proxy"
        )
    selection_proxy_path = _write_selection_proxy(plan)
    print(f"Wrote selection proxy: {selection_proxy_path}", flush=True)


def execute_policy_eval_plan(plan: Mapping[str, Any]) -> None:
    """Run only the three fixed-cache proxy evaluators from an owned checkpoint.

    This function deliberately has no train command, no protocol-copy side
    effect, and no Stage-1 resume behavior.  It is the only execution helper
    used by the launcher's ``eval`` command.
    """
    validate_policy_root(plan["root"], resume=True)
    validate_policy_source_isolation(
        plan["root"],
        protocol_source_root=plan["protocol_source_root"],
        stage1_checkpoint=plan["stage1_checkpoint"],
        teacher_model_path=plan["teacher_model_path"],
        dataset_path=plan["dataset_path"],
    )
    validate_resume_policy_ownership(plan["root"], plan["policy"]["name"])
    checkpoint_dir = validate_owned_policy_checkpoint(
        root=plan["root"], checkpoint_dir=plan["checkpoint_dir"]
    )
    validate_policy_eval_execution_contract(plan, checkpoint_dir=checkpoint_dir)
    validate_target_free_online_student(checkpoint_dir, role="evaluation")
    validate_eval_caches(plan["eval_plans"])
    completed_budgets: tuple[str, ...] = ()
    for eval_plan in plan["eval_plans"]:
        validate_policy_eval_write_availability(
            plan,
            checkpoint_dir=checkpoint_dir,
            completed_budgets=completed_budgets,
        )
        eval_env = {**os.environ, **dict(eval_plan["env"])}
        subprocess.run(list(eval_plan["argv"]), cwd=REPO_ROOT, env=eval_env, check=True)
        completed_budgets = (*completed_budgets, str(eval_plan["budget"]))
    selection_proxy_path = _write_selection_proxy(plan)
    print(f"Wrote selection proxy: {selection_proxy_path}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or explicitly run one isolated eight-GPU Cosmos mixed-step policy."
    )
    parser.add_argument("--policy", choices=SUPPORTED_POLICY_NAMES, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="New policy-specific root; no default is provided by design.",
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--protocol-source-root",
        type=Path,
        required=True,
        help="Read-only source holding protocol JSON and t4/t8 teacher caches.",
    )
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1_CHECKPOINT)
    parser.add_argument("--resume-from-path", type=Path, default=None)
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        default=None,
        help=(
            "Evaluate this owned policy checkpoint only. This selects the "
            "three-budget fixed-cache proxy path and never trains."
        ),
    )
    parser.add_argument("--current-step", type=int, default=0)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Optional bounded chunk; zero means full run unless stop-after-step is set.",
    )
    parser.add_argument("--max-train-steps", type=int, default=DEFAULT_MAX_TRAIN_STEPS)
    parser.add_argument("--save-interval", type=int, default=DEFAULT_SAVE_INTERVAL)
    parser.add_argument("--stop-after-step", type=int, default=0)
    parser.add_argument("--master-port", type=int, default=DEFAULT_MASTER_PORT)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--torchrun", type=Path, default=DEFAULT_TORCHRUN)
    parser.add_argument("--teacher-model-path", type=Path, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--device-list", default=DEFAULT_DEVICE_LIST)
    parser.add_argument("--world-size", type=int, default=DEFAULT_WORLD_SIZE)
    parser.add_argument(
        "--force-sequence",
        default="",
        help="Optional deterministic selector labels/indices, e.g. s1,s2,s4 for a preflight.",
    )
    parser.add_argument(
        "--preflight-evidence",
        action="store_true",
        help=(
            "Enable only the reviewed 9-step forced-Universe preflight contract; "
            "requires --opd-aux-warmup-steps 0 and --opd-aux-interval 1."
        ),
    )
    parser.add_argument("--opd-aux-warmup-steps", type=int, default=None)
    parser.add_argument("--opd-aux-interval", type=int, default=None)
    parser.add_argument("--eval-device-list", default="0")
    parser.add_argument("--eval-worker-device-list", default="0")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute after printing the plan. Without this flag nothing is launched or written.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.eval_checkpoint is not None:
        plan = build_policy_eval_plan(
            policy_name=args.policy,
            root=args.root,
            checkpoint_dir=args.eval_checkpoint,
            dataset_path=args.dataset_path,
            protocol_source_root=args.protocol_source_root,
            stage1_checkpoint=args.stage1_checkpoint,
            teacher_model_path=args.teacher_model_path,
            eval_device_list=args.eval_device_list,
            eval_worker_device_list=args.eval_worker_device_list,
        )
    else:
        plan = build_policy_train_plan(
            policy_name=args.policy,
            root=args.root,
            dataset_path=args.dataset_path,
            protocol_source_root=args.protocol_source_root,
            stage1_checkpoint=args.stage1_checkpoint,
            current_step=args.current_step,
            chunk_size=args.chunk_size,
            max_train_steps=args.max_train_steps,
            save_interval=args.save_interval,
            stop_after_step=args.stop_after_step,
            master_port=args.master_port,
            train_seed=args.train_seed,
            torchrun=args.torchrun,
            teacher_model_path=args.teacher_model_path,
            device_list=args.device_list,
            world_size=args.world_size,
            force_sequence=args.force_sequence,
            preflight_evidence=args.preflight_evidence,
            opd_aux_warmup_steps=args.opd_aux_warmup_steps,
            opd_aux_interval=args.opd_aux_interval,
            resume_from_path=args.resume_from_path,
            eval_device_list=args.eval_device_list,
            eval_worker_device_list=args.eval_worker_device_list,
        )
    print(json.dumps(_json_ready(plan), indent=2, sort_keys=True))
    if args.run:
        if args.eval_checkpoint is not None:
            execute_policy_eval_plan(plan)
        else:
            execute_policy_plan(plan)


if __name__ == "__main__":
    main()
