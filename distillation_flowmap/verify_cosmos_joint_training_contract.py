from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from distillation_flowmap.config_libero_cosmos_policy_stage2_progressive import (
    cfg as stage2_config,
)
from distillation_flowmap.cosmos_policy_adapter import cosmos_actions_to_flowmap_x0
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


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_new_attestation(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing attestation {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    created_temp = False
    try:
        with temp_path.open("x") as handle:
            created_temp = True
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing attestation {path}")
        os.replace(temp_path, path)
        created_temp = False
    finally:
        if created_temp:
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
    payload = {
        **metadata,
        "git_commit": _git_commit(),
        "verifier_module": VERIFIER_MODULE,
        "action_round_trip": {
            "num_actions": num_actions,
            "max_abs_error": max_abs_error,
        },
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_new_attestation(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
