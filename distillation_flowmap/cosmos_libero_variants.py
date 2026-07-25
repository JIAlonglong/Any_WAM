"""Immutable experiment catalogue for aligned Cosmos LIBERO Stage-2 runs."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any


APM_VARIANTS_PATH = Path(__file__).with_name(
    "cosmos_libero_video_apm_variants.json"
)

_RUN_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_PROGRESSIVE = {
    "s1": {
        "progressive_stage": "s1",
        "max_train_steps": 3000,
        "master_port": 29663,
        "rollout_step_pairs": ((4, 1),),
        "danceopd_rollout_steps": (1,),
        "video_endpoint_weight": 1.0,
        "video_velocity_weight": 0.0,
        "action_endpoint_weight": 1.0,
        "opd_rollout_grad_mode": "last_step",
        "opd_rollout_grad_steps": 1,
        "opd_endpoint_focus_prob": 0.90,
    },
    "s2": {
        "progressive_stage": "s2",
        "max_train_steps": 3000,
        "master_port": 29662,
        "rollout_step_pairs": ((4, 2),),
        "danceopd_rollout_steps": (2,),
        "video_endpoint_weight": 1.0,
        "video_velocity_weight": 1.0,
        "action_endpoint_weight": 1.0,
        "opd_rollout_grad_mode": "last_step",
        "opd_rollout_grad_steps": 1,
        "opd_endpoint_focus_prob": 0.85,
    },
    "s4": {
        "progressive_stage": "s4",
        "max_train_steps": 5000,
        "master_port": 29661,
        "rollout_step_pairs": ((8, 4),),
        "danceopd_rollout_steps": (4,),
        "video_endpoint_weight": 1.0,
        "video_velocity_weight": 1.0,
        "action_endpoint_weight": 1.0,
        "opd_rollout_grad_mode": "suffix",
        "opd_rollout_grad_steps": 2,
        "opd_endpoint_focus_prob": 0.80,
    },
    "universal": {
        "progressive_stage": "universal",
        "max_train_steps": 5000,
        "master_port": 29664,
        "rollout_step_pairs": ((8, 1), (8, 2), (8, 4)),
        "danceopd_rollout_steps": (2, 4),
        "video_endpoint_weight": 1.0,
        "video_velocity_weight": 1.0,
        "action_endpoint_weight": 1.0,
        "opd_rollout_grad_mode": "last_step",
        "opd_rollout_grad_steps": 1,
        "opd_endpoint_focus_prob": 0.85,
    },
    "universal-video-action": {
        "progressive_stage": "universal",
        "max_train_steps": 5000,
        "master_port": 29666,
        "rollout_step_pairs": ((8, 1), (8, 2), (8, 4)),
        "danceopd_rollout_steps": (2, 4),
        "video_endpoint_weight": 1.0,
        "video_velocity_weight": 1.0,
        "action_endpoint_weight": 1.0,
        "opd_rollout_grad_mode": "last_step",
        "opd_rollout_grad_steps": 1,
        "opd_endpoint_focus_prob": 0.85,
    },
}
_APM_PORTS = {
    "stage1_only": 29667,
    "anchor_only": 29668,
    "field_only": 29669,
    "apm": 29670,
}
VARIANT_NAMES = (
    "s1",
    "s2",
    "s4",
    "universal",
    "universal-video-action",
    "stage1_only",
    "anchor_only",
    "field_only",
    "apm",
)


class VariantError(ValueError):
    """Raised when an experiment record cannot be normalized safely."""


class FrozenVariant(Mapping[str, Any]):
    """Small deeply immutable, mapping-compatible normalized record."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any]):
        self._values = MappingProxyType(
            {key: _freeze(value) for key, value in values.items()}
        )

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise VariantError(f"{label} must be a positive plain integer")
    return value


def _port(value: object) -> int:
    port = _positive_int(value, label="master port")
    if port > 65535:
        raise VariantError("master port must be in [1, 65535]")
    return port


def _run_tag(value: object) -> str:
    if not isinstance(value, str) or not _RUN_TAG_RE.fullmatch(value):
        raise VariantError(
            "run tag must be one path-safe component containing only "
            "letters, numbers, '.', '_' or '-'"
        )
    return value


def _coefficient(value: object, *, label: str) -> float:
    if type(value) not in (int, float) or value not in (0, 1, 0.0, 1.0):
        raise VariantError(f"{label} must be exactly 0 or 1")
    return float(value)


def _canonical_optional_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).resolve(strict=False))


def load_video_apm_variants(
    path: str | Path = APM_VARIANTS_PATH,
) -> Mapping[str, Mapping[str, object]]:
    """Load and strictly validate the four declarative video-only APM arms."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VariantError(f"unable to load APM variant catalogue: {source}") from exc
    if not isinstance(payload, dict):
        raise VariantError("APM catalogue must contain a JSON object")
    unknown_root = set(payload) - {"variants"}
    if unknown_root:
        raise VariantError(f"unknown APM catalogue fields: {sorted(unknown_root)}")
    variants = payload.get("variants")
    if not isinstance(variants, list):
        raise VariantError("APM catalogue variants must be a list")

    resolved: dict[str, Mapping[str, object]] = {}
    allowed = {"name", "endpoint", "velocity"}
    for index, item in enumerate(variants):
        if not isinstance(item, dict):
            raise VariantError(f"APM variant {index} must be an object")
        unknown = set(item) - allowed
        missing = allowed - set(item)
        if unknown:
            raise VariantError(f"unknown APM variant fields: {sorted(unknown)}")
        if missing:
            raise VariantError(f"missing APM variant fields: {sorted(missing)}")
        name = item["name"]
        if not isinstance(name, str) or name not in _APM_PORTS:
            raise VariantError(f"unknown APM variant name: {name!r}")
        if name in resolved:
            raise VariantError(f"duplicate APM variant name: {name}")
        resolved[name] = MappingProxyType(
            {
                "endpoint": _coefficient(
                    item["endpoint"], label=f"{name} endpoint"
                ),
                "velocity": _coefficient(
                    item["velocity"], label=f"{name} velocity"
                ),
            }
        )
    if tuple(resolved) != tuple(_APM_PORTS):
        raise VariantError(
            "APM catalogue must declare exactly stage1_only, anchor_only, "
            "field_only, apm in that order"
        )
    return MappingProxyType(resolved)


def _base_spec(name: str) -> dict[str, object]:
    if name in _PROGRESSIVE:
        return dict(_PROGRESSIVE[name])
    apm = load_video_apm_variants()
    if name in apm:
        return {
            "progressive_stage": "universal",
            "max_train_steps": 5000,
            "master_port": _APM_PORTS[name],
            "rollout_step_pairs": ((8, 1), (8, 2), (8, 4)),
            "danceopd_rollout_steps": (2, 4),
            "video_endpoint_weight": apm[name]["endpoint"],
            "video_velocity_weight": apm[name]["velocity"],
            "action_endpoint_weight": 0.0,
            "opd_rollout_grad_mode": "last_step",
            "opd_rollout_grad_steps": 1,
            "opd_endpoint_focus_prob": 0.85,
        }
    raise VariantError(f"unknown Cosmos LIBERO variant: {name!r}")


def resolve_variant(
    name: str,
    *,
    output_root: str | Path,
    run_tag: str = "default",
    steps: int | None = None,
    save_interval: int = 1000,
    master_port: int | None = None,
    output_dir: str | Path | None = None,
    dataset_path: str | Path | None = None,
    teacher_model_path: str | Path | None = None,
    cosmos_video_vae_model_path: str | Path | None = None,
    cosmos_policy_repo: str | Path | None = None,
    cosmos_policy_python: str | Path | None = None,
    cosmos_policy_extra_pythonpath: str | None = None,
    cosmos_policy_local_model_dir: str | Path | None = None,
    attention_mode: str = "flex",
    provenance: Mapping[str, Any] | None = None,
) -> FrozenVariant:
    """Resolve one arm without creating, modifying, or probing output paths."""

    spec = _base_spec(name)
    tag = _run_tag(run_tag)
    max_steps = (
        _positive_int(steps, label="steps")
        if steps is not None
        else int(spec["max_train_steps"])
    )
    interval = _positive_int(save_interval, label="save interval")
    port = _port(
        master_port if master_port is not None else spec["master_port"]
    )
    selected_output = (
        Path(output_dir)
        if output_dir is not None
        else Path(output_root) / tag / name
    ).resolve(strict=False)
    use_opd_aux = bool(
        spec["video_endpoint_weight"] or spec["video_velocity_weight"]
    )
    record = {
        "name": name,
        "run_tag": tag,
        "progressive_stage": spec["progressive_stage"],
        "max_train_steps": max_steps,
        "save_interval": interval,
        "master_port": port,
        "output_dir": str(selected_output),
        "rollout_step_pairs": spec["rollout_step_pairs"],
        "danceopd_rollout_steps": spec["danceopd_rollout_steps"],
        "video_endpoint_weight": spec["video_endpoint_weight"],
        "video_velocity_weight": spec["video_velocity_weight"],
        "action_endpoint_weight": spec["action_endpoint_weight"],
        "action_opd_enabled": name == "universal-video-action",
        "use_opd_aux": use_opd_aux,
        "opd_aux_standalone_step": use_opd_aux,
        "learning_rate": 2e-7,
        "opd_aux_weight": 0.10,
        "opd_aux_warmup_steps": 8,
        "opd_aux_prob": 1.0,
        "opd_rollout_grad_mode": spec["opd_rollout_grad_mode"],
        "opd_rollout_grad_steps": spec["opd_rollout_grad_steps"],
        "opd_endpoint_focus_prob": spec["opd_endpoint_focus_prob"],
        "opd_danceopd_query_alpha": 5.0,
        "opd_danceopd_query_beta": 2.0,
        "opd_aux_interval": 8,
        "opd_aux_phase": 2,
        "opd_action_rollout_grad_mode": spec["opd_rollout_grad_mode"],
        "opd_danceopd_verify_terminal_prior": True,
        "opd_danceopd_terminal_prior_tolerance": 2e-6,
        "opd_danceopd_terminal_prior_warn_factor": 0.5,
        "opd_cosmos_spatial_crop_size": 28,
        "opd_joint_action_rollout": True,
        "cosmos_use_teacher_action_anchor": True,
        "diffusion_ratio": 0.5,
        "consistency_ratio": 0.25,
        "flowmap_ratio": 0.25,
        "video_loss_weight": 1.0,
        "action_loss_weight": 1.0,
        "action_block_weight": 4.0,
        "beta1": 0.9,
        "beta2": 0.95,
        "ema_decay": 0.999,
        "ema_warmup_steps": 100,
        "drop_text_ratio": 0.1,
        "fuse_guidance_scale": 3.0,
        "cfg_min": 3.0,
        "cfg_max": 3.0,
        "max_grad_norm": 0.3,
        "warmup_steps": 100,
        "num_ddim_timesteps_action": 1,
        "cosmos_policy_use_raw_inference": True,
        "skip_target_student_for_cosmos_latent": True,
        "cosmos_latent_cdiff_loss_weight": 1.0,
        "cosmos_latent_endpoint_loss_weight": 0.0,
        "cosmos_latent_epsilon": 0.001,
        "cosmos_latent_t_min": 4.0 / 5.0,
        "cosmos_latent_t_max": 80.0 / 81.0,
        "cosmos_latent_channels": 16,
        "cosmos_latent_frames": 9,
        "cosmos_latent_height": 28,
        "cosmos_latent_width": 28,
        "cosmos_latent_center_velocity_mode": "symmetric_average",
        "cosmos_latent_target_mode": "hybrid_cdiff",
        "cosmos_latent_cdiff_interval": 4,
        "mechanism_diagnostics": True,
        "mechanism_diagnostic_interval": 100,
        "mechanism_diagnostic_seed": 42,
        "mechanism_diagnostic_r": 500,
        "mechanism_diagnostic_s": 250,
        "mechanism_diagnostic_teacher_steps": 8,
        "mechanism_cosmos_t_min": 4.0 / 5.0,
        "mechanism_cosmos_t_max": 80.0 / 81.0,
        "dataset_path": _canonical_optional_path(dataset_path),
        "dataset_sample_manifest": None,
        "teacher_model_path": _canonical_optional_path(teacher_model_path),
        "cosmos_video_vae_model_path": _canonical_optional_path(
            cosmos_video_vae_model_path
        ),
        "cosmos_policy_repo": _canonical_optional_path(cosmos_policy_repo),
        "cosmos_policy_python": _canonical_optional_path(cosmos_policy_python),
        "cosmos_policy_extra_pythonpath": cosmos_policy_extra_pythonpath,
        "cosmos_policy_local_model_dir": _canonical_optional_path(
            cosmos_policy_local_model_dir
        ),
        "cosmos_policy_config_name": "cosmos_predict2_2b_480p_libero__inference_only",
        "cosmos_policy_config_file": (
            "cosmos_predict2/_src/predict2/cosmos_policy/config/config.py"
        ),
        "cosmos_policy_num_denoising_steps_action": 5,
        "cosmos_policy_seed": 1,
        "cosmos_policy_primary_image_key": "observation.images.agentview_rgb",
        "cosmos_policy_wrist_image_key": (
            "observation.images.eye_in_hand_rgb"
        ),
        "cosmos_policy_inference_mode": "subprocess",
        "attention_mode": attention_mode,
        "provenance": _jsonable(provenance) if provenance is not None else None,
        "train_seed": 42,
        "tensorboard_enabled": True,
        "enable_wandb": False,
        "wandb_mode": "offline",
        "hf_offline": True,
        "transformers_offline": True,
        "hf_hub_offline": True,
    }
    return FrozenVariant(record)


def resolve_variants(
    names: Sequence[str],
    *,
    output_root: str | Path,
    run_tag: str = "default",
    master_ports: Mapping[str, int] | None = None,
    output_dirs: Mapping[str, str | Path] | None = None,
) -> tuple[FrozenVariant, ...]:
    """Resolve a set while rejecting ambiguous experiment identities."""

    if len(names) != len(set(names)):
        raise VariantError("duplicate variant name")
    ports = master_ports or {}
    outputs = output_dirs or {}
    unknown_overrides = (set(ports) | set(outputs)) - set(names)
    if unknown_overrides:
        raise VariantError(f"unknown variant override: {sorted(unknown_overrides)}")
    records = tuple(
        resolve_variant(
            name,
            output_root=output_root,
            run_tag=run_tag,
            master_port=ports.get(name),
            output_dir=outputs.get(name),
        )
        for name in names
    )
    resolved_ports = [record["master_port"] for record in records]
    if len(resolved_ports) != len(set(resolved_ports)):
        raise VariantError("duplicate master port")
    resolved_outputs = [record["output_dir"] for record in records]
    if len(resolved_outputs) != len(set(resolved_outputs)):
        raise VariantError("duplicate output directory")
    return records


def canonical_variant_json(record: Mapping[str, object]) -> str:
    """Return the stable identity serialized into manifests and checkpoints."""

    return json.dumps(
        _jsonable(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
