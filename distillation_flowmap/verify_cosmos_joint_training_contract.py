from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from distillation_flowmap.config_libero_cosmos_policy_stage2_progressive import (
    cfg as stage2_config,
)
from distillation_flowmap.cosmos_policy_adapter import cosmos_actions_to_flowmap_x0
from distillation_flowmap.cosmos_deployment_rollout import (
    deployment_endpoint_losses,
    deployment_joint_step_for_update,
    should_run_deployment_joint_rollout,
    should_run_raw_auxiliary,
)
from distillation_flowmap.cosmos_training_contract import (
    ACTION_PACKING_SCHEMA,
    contract_metadata,
    validate_contract_metadata,
)
from evaluation.libero.cosmos_progressive_s4_server import (
    ActionDecodingTemplate,
    decode_student_action,
)


VERIFIER_MODULE = "distillation_flowmap.verify_cosmos_joint_training_contract"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _distinguishable_actions(config) -> torch.Tensor:
    q01 = torch.as_tensor(config.norm_stat["q01"], dtype=torch.float32)[:7]
    q99 = torch.as_tensor(config.norm_stat["q99"], dtype=torch.float32)[:7]
    fractions = torch.linspace(0.1, 0.9, 16 * 7, dtype=torch.float32).reshape(
        16, 7
    )
    return (q01 + fractions * (q99 - q01)).unsqueeze(0)


def _production_action_round_trip(config) -> tuple[int, float]:
    actions = _distinguishable_actions(config)
    full = cosmos_actions_to_flowmap_x0(
        actions,
        target_shape=(1, len(config.inverse_used_action_channel_ids), 16, 4, 1),
        q01=config.norm_stat["q01"],
        q99=config.norm_stat["q99"],
        inverse_used_action_channel_ids=config.inverse_used_action_channel_ids,
        device=torch.device("cpu"),
        dtype=torch.float32,
        packing_schema=ACTION_PACKING_SCHEMA,
        downsample_factor=4,
    )
    compact = full[:, :, ::4]
    decoded = decode_student_action(
        compact,
        ActionDecodingTemplate.from_config(config),
    )
    expected = actions[0].numpy()
    if decoded.shape != expected.shape:
        raise RuntimeError(
            f"production action round trip returned {decoded.shape}, expected {expected.shape}"
        )
    max_abs_error = float(np.max(np.abs(decoded - expected)))
    if max_abs_error >= 1e-5:
        raise RuntimeError(
            f"production action round trip max_abs_error={max_abs_error} is not < 1e-5"
        )
    return int(decoded.shape[0]), max_abs_error


def _verify_deployment_contract(config) -> dict[str, object]:
    joint_student_step_cycle = [
        deployment_joint_step_for_update(index) for index in range(6)
    ]
    if joint_student_step_cycle != [1, 2, 4, 1, 2, 4]:
        raise RuntimeError(
            f"production joint student-step cycle is {joint_student_step_cycle!r}"
        )

    interval = getattr(config, "deployment_joint_rollout_interval")
    if type(interval) is not int or interval != 4:
        raise RuntimeError(f"deployment rollout interval must be int 4, got {interval!r}")
    deployment_schedule_steps = [
        step
        for step in range(8, 25)
        if should_run_deployment_joint_rollout(step, interval)
    ]
    if deployment_schedule_steps != [8, 12, 16, 20, 24]:
        raise RuntimeError(
            f"production deployment cadence is {deployment_schedule_steps!r}"
        )

    raw_warmup = getattr(config, "opd_aux_warmup_steps")
    raw_interval = getattr(config, "opd_aux_interval")
    raw_phase = getattr(config, "opd_aux_phase")
    if type(raw_warmup) is not int or raw_warmup != 8:
        raise RuntimeError(f"raw auxiliary warmup must be int 8, got {raw_warmup!r}")
    if type(raw_interval) is not int or raw_interval != 8:
        raise RuntimeError(f"raw auxiliary interval must be int 8, got {raw_interval!r}")
    if type(raw_phase) is not int or raw_phase != 2:
        raise RuntimeError(f"raw auxiliary phase must be int 2, got {raw_phase!r}")
    raw_auxiliary_schedule_steps = [
        step
        for step in range(8, 25)
        if should_run_raw_auxiliary(
            step,
            warmup=raw_warmup,
            interval=raw_interval,
            phase=raw_phase,
        )
    ]
    if raw_auxiliary_schedule_steps != [10, 18]:
        raise RuntimeError(
            f"production raw auxiliary cadence is {raw_auxiliary_schedule_steps!r}"
        )
    schedules_disjoint = not (
        set(deployment_schedule_steps) & set(raw_auxiliary_schedule_steps)
    )
    if not schedules_disjoint:
        raise RuntimeError("deployment and raw auxiliary schedules overlap")

    losses = deployment_endpoint_losses(
        torch.ones(1, 2),
        torch.zeros(1, 2),
        torch.full((1, 2), 2.0),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        action_weight=getattr(config, "deployment_action_weight"),
    )
    endpoint_losses = {
        name: float(loss.detach().cpu().item())
        for name, loss in losses.items()
    }
    expected_losses = {"video": 1.0, "action": 4.0, "total": 5.0}
    if endpoint_losses != expected_losses:
        raise RuntimeError(
            f"production endpoint losses are {endpoint_losses!r}, "
            f"expected {expected_losses!r}"
        )
    return {
        "joint_student_step_cycle": joint_student_step_cycle,
        "deployment_schedule_steps": deployment_schedule_steps,
        "raw_auxiliary_schedule_steps": raw_auxiliary_schedule_steps,
        "schedules_disjoint": schedules_disjoint,
        "endpoint_losses": endpoint_losses,
    }


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout.strip()


def _write_new_attestation(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite existing attestation {path}"
            ) from None
    finally:
        temp_path.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attest the executable Cosmos joint-training contract."
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    metadata = contract_metadata(stage2_config, stage="progressive_stage2")
    validate_contract_metadata(metadata, required_stage="progressive_stage2")
    num_actions, max_abs_error = _production_action_round_trip(stage2_config)
    deployment_execution = _verify_deployment_contract(stage2_config)
    payload = {
        **metadata,
        "git_commit": _git_commit(),
        "verifier_module": VERIFIER_MODULE,
        "action_round_trip": {
            "num_actions": num_actions,
            "max_abs_error": max_abs_error,
        },
        "deployment_execution": deployment_execution,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_new_attestation(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
