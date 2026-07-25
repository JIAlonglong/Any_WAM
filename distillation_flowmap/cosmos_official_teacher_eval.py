"""Strict matched-budget adapter for the official Cosmos Policy teacher."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import numpy as np

from distillation_flowmap.cosmos_training_contract import (
    normalize_cosmos_inference_request,
)


@dataclass(frozen=True)
class ResolvedCosmosOfficialTeacher:
    """Verified monolithic official policy root, never a student transformer."""

    model_role: str
    backend: str
    root_path: str
    weight_path: str
    config_path: str
    dataset_stats_path: str
    t5_embeddings_path: str
    contract_identity: str
    transformer_path: None = None


def _require_plain(path: Path, *, directory: bool, label: str) -> None:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} is missing: {path}") from None
    if stat.S_ISLNK(mode):
        raise ValueError(f"{label} must not be a symlink: {path}")
    if directory and not stat.S_ISDIR(mode):
        raise ValueError(f"{label} must be a directory: {path}")
    if not directory and not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular file: {path}")


def resolve_cosmos_official_teacher_root(
    path: str | Path,
) -> ResolvedCosmosOfficialTeacher:
    """Validate the uploaded official Cosmos Policy monolithic checkpoint."""

    root = Path(path)
    _require_plain(root, directory=True, label="official Cosmos teacher root")
    if root.name == "transformer" or root.parent.name in {
        "online_student",
        "target_student",
    }:
        raise ValueError(
            "official_teacher requires the monolithic Cosmos Policy root, "
            "not a student transformer directory"
        )
    canonical_root = root.resolve(strict=True)
    config_path = canonical_root / "config.json"
    stats_path = canonical_root / "libero_dataset_statistics.json"
    embeddings_path = canonical_root / "libero_t5_embeddings.pkl"
    for required, label in (
        (config_path, "official teacher config.json"),
        (stats_path, "official teacher dataset statistics"),
        (embeddings_path, "official teacher T5 embeddings"),
    ):
        _require_plain(required, directory=False, label=label)
    weights = sorted(canonical_root.glob("*.pt"))
    if len(weights) != 1:
        raise ValueError(
            "official Cosmos teacher root must contain exactly one monolithic .pt "
            f"checkpoint, found {len(weights)}"
        )
    _require_plain(weights[0], directory=False, label="official teacher checkpoint")

    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if not isinstance(config, dict):
        raise TypeError("official teacher config.json must contain a JSON object")
    if config.get("model_type") != "cosmos-policy":
        raise ValueError(
            "official teacher model_type must be exactly 'cosmos-policy'"
        )
    if config.get("architecture") != "diffusion-transformer":
        raise ValueError(
            "official teacher architecture must be exactly 'diffusion-transformer'"
        )
    diffusion = config.get("diffusion_config")
    if not isinstance(diffusion, dict) or diffusion.get("generation_mode") != "parallel":
        raise ValueError(
            "official teacher backend must use Cosmos Policy parallel generation"
        )
    actions = (
        config.get("output_spec", {}).get("actions")
        if isinstance(config.get("output_spec"), dict)
        else None
    )
    if (
        not isinstance(actions, dict)
        or actions.get("dim") != 7
        or actions.get("horizon") != 16
    ):
        raise ValueError(
            "official teacher action contract must be horizon=16 and dim=7"
        )

    weight_stat = weights[0].stat()
    identity = hashlib.sha256()
    identity.update(config_bytes)
    identity.update(weights[0].name.encode("utf-8"))
    identity.update(str(weight_stat.st_size).encode("ascii"))
    identity.update(stats_path.read_bytes())
    identity.update(embeddings_path.read_bytes())
    return ResolvedCosmosOfficialTeacher(
        model_role="official_teacher",
        backend="cosmos_policy",
        root_path=str(canonical_root),
        weight_path=str(weights[0]),
        config_path=str(config_path),
        dataset_stats_path=str(stats_path),
        t5_embeddings_path=str(embeddings_path),
        contract_identity=identity.hexdigest(),
    )


class OfficialTeacherMatchedBudgetAdapter:
    """Call official ``get_action`` and require worker-observed matched K."""

    _REQUIRED_PROOF_FIELDS = (
        "requested_video_steps",
        "requested_action_steps",
        "effective_video_steps",
        "effective_action_steps",
        "matched_budget_verified",
    )

    def __init__(
        self,
        *,
        teacher: Any,
        video_steps: int,
        action_steps: int,
        include_future: bool = True,
    ) -> None:
        request = normalize_cosmos_inference_request(
            model_role="official_teacher",
            video_steps=video_steps,
            action_steps=action_steps,
            student_steps=None,
        )
        self.teacher = teacher
        self.video_steps = request.video_steps
        self.action_steps = request.action_steps
        self.include_future = bool(include_future)
        if not self.include_future:
            raise ValueError(
                "official matched video/action evaluation requires future-video generation"
            )

    @staticmethod
    def _plain_scalar(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        return value

    def infer_raw(self, raw_batch: Mapping[str, Any]) -> dict[str, Any]:
        configured = getattr(self.teacher, "num_denoising_steps_action", None)
        if type(configured) is not int or configured != self.action_steps:
            raise RuntimeError(
                "official teacher is configured with action steps "
                f"{configured!r}, expected {self.action_steps}; refusing a silent fallback"
            )
        result = self.teacher.predict_raw_action_result(
            raw_batch,
            include_future=self.include_future,
            video_steps=self.video_steps,
            action_steps=self.action_steps,
        )
        if not isinstance(result, Mapping):
            raise TypeError("official teacher result must be a mapping")
        missing = [
            name for name in self._REQUIRED_PROOF_FIELDS if name not in result
        ]
        if missing:
            raise RuntimeError(
                f"official teacher matched-budget response is missing {missing}"
            )
        expected = self.action_steps
        proof = {
            name: self._plain_scalar(result[name])
            for name in self._REQUIRED_PROOF_FIELDS
        }
        if (
            type(proof["matched_budget_verified"]) is not bool
            or not proof["matched_budget_verified"]
            or any(
                type(proof[name]) is not int or proof[name] != expected
                for name in self._REQUIRED_PROOF_FIELDS[:-1]
            )
        ):
            raise RuntimeError(
                "official teacher did not return a verified matched budget "
                f"K={expected}: {proof}"
            )
        actions = np.asarray(result.get("actions"), dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[None]
        if actions.ndim != 3 or actions.shape[1:] != (16, 7):
            raise RuntimeError(
                "official teacher actions must have shape [B,16,7], "
                f"got {actions.shape}"
            )
        output = dict(result)
        output["actions"] = actions
        output.update(proof)
        output["model_role"] = "official_teacher"
        return output
