#!/usr/bin/env python3
"""Plan or execute one four-GPU LIBERO APM small-sample arm."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

from distillation_flowmap.ablation.libero_small_sample_protocol import (
    build_libero_task0_protocol,
    write_libero_task0_manifests,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = Path(__file__).resolve().parent
VARIANTS_FILE = ABLATION_DIR / "libero_apm_lora_variants.json"


def _read_variants() -> dict:
    return json.loads(VARIANTS_FILE.read_text(encoding="utf-8"))


def _find_variant(name: str) -> dict:
    metadata = _read_variants()
    for variant in metadata["variants"]:
        if name in (variant["name"], variant["id"]):
            return variant
    known = ", ".join(item["name"] for item in metadata["variants"])
    raise ValueError(f"Unknown variant {name!r}; expected one of: {known}")


def _command(env: dict[str, str], argv: list[str]) -> str:
    prefix = " ".join(
        f"{key}={shlex.quote(str(value))}" for key, value in sorted(env.items())
    )
    return f"{prefix} {shlex.join(argv)}"


def _git_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def build_run_plan(args) -> dict:
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.save_interval <= 0:
        raise ValueError("save interval must be positive")
    if not 1 <= args.master_port <= 65535:
        raise ValueError("master port must be in [1, 65535]")
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",")]
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise ValueError("gpu ids must contain exactly four unique IDs")
    if any(not item.isdigit() for item in gpu_ids):
        raise ValueError("gpu ids must be non-negative integers")

    variant = _find_variant(args.variant)
    run_dir = Path(args.output_root) / variant["name"] / "seed_42"
    stage2_dir = run_dir / "stage2"
    checkpoint = stage2_dir / "checkpoints" / f"step_{args.steps}"
    protocol_root = Path(args.output_root) / "protocol"
    train_manifest = protocol_root / "train_manifest.json"
    heldout_manifest = protocol_root / "heldout_manifest.json"

    env = {
        "ACTION_LOSS_WEIGHT": "1.0",
        "ATTN_MODE": "torch",
        "CONFIG_FILE": "distillation_flowmap.config_libero_apm_lora_ablation",
        "CUDA_VISIBLE_DEVICES": args.gpu_ids,
        "DATASET_SAMPLE_MANIFEST": str(train_manifest),
        "EMPTY_EMB_PATH": str(args.empty_emb_path),
        "ENABLE_GRAD_BRANCH_DIAGNOSTICS": "1",
        "GRADIENT_CHECKPOINTING": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "LEARNING_RATE": "5e-6",
        "LORA_ALPHA": "64",
        "LORA_DROPOUT": "0.0",
        "LORA_RANK": "128",
        "MAX_TRAIN_STEPS": str(args.steps),
        "MECHANISM_DIAGNOSTICS": "1",
        "MECHANISM_DIAGNOSTIC_INTERVAL": "50",
        "OPD_AUX_ACTION": "0",
        "OPD_AUX_GRADIENT_CHECKPOINTING": "1",
        "OPD_AUX_INTERVAL": "4",
        "OPD_AUX_MODALITIES": "video",
        "OPD_AUX_PROB": "1.0",
        "OPD_DANCEOPD_ACTION_VELOCITY_WEIGHT": "0.0",
        "OPD_DANCEOPD_ENDPOINT_WEIGHT": variant["endpoint_weight"],
        "OPD_DANCEOPD_ROLLOUT_STEPS": "2,4",
        "OPD_DANCEOPD_VELOCITY_WEIGHT": variant["velocity_weight"],
        "OPD_JOINT_ACTION_ROLLOUT": "0",
        "OPD_QUERY_MODE": "danceopd",
        "OPD_ROLLOUT_STEP_PAIRS": "8,1;8,2;8,4",
        "OUTPUT_DIR": str(stage2_dir),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        ),
        "RESET_RESUME_STEP": "1",
        "RESUME_FROM_PATH": str(args.stage1_ckpt),
        "RESUME_ONLINE_FROM_TARGET": "1",
        "SAVE_INTERVAL": str(args.save_interval),
        "SKIP_TEACHER_COMPILE": "1",
        "TRAIN_SEED": "42",
        "TRANSFORMERS_OFFLINE": "1",
        "USE_FSDP1": "1",
        "USE_LORA": "1",
        "WANDB_MODE": os.environ.get("WANDB_MODE", "offline"),
    }
    argv = [
        str(args.torchrun),
        "--nproc_per_node=4",
        f"--master_port={args.master_port}",
        "distillation_flowmap/train.py",
        "--teacher-model-path",
        str(args.teacher_model_path),
        "--dataset-path",
        str(args.dataset_path),
        "--gradient-accumulation-steps",
        "1",
    ]
    protocol = build_libero_task0_protocol()
    return {
        "variant": variant,
        "run_dir": run_dir,
        "stage2_dir": stage2_dir,
        "checkpoint": checkpoint,
        "protocol_root": protocol_root,
        "train_manifest": train_manifest,
        "heldout_manifest": heldout_manifest,
        "protocol": protocol,
        "stage1_ckpt": Path(args.stage1_ckpt),
        "teacher_model_path": Path(args.teacher_model_path),
        "dataset_path": Path(args.dataset_path),
        "empty_emb_path": Path(args.empty_emb_path),
        "stage2": {"env": env, "argv": argv, "command": _command(env, argv)},
        "dry_run": bool(args.dry_run),
    }


def write_run_files(plan: dict) -> Path:
    if plan["run_dir"].exists():
        raise FileExistsError(f"Refusing to reuse run directory: {plan['run_dir']}")
    write_libero_task0_manifests(plan["protocol_root"])
    plan["run_dir"].mkdir(parents=True)
    manifest = {
        "git_hash": _git_hash(),
        "variant": plan["variant"]["name"],
        "variant_id": plan["variant"]["id"],
        "protocol": plan["protocol"],
        "train_manifest": str(plan["train_manifest"]),
        "heldout_manifest": str(plan["heldout_manifest"]),
        "stage1_ckpt": str(plan["stage1_ckpt"]),
        "teacher_model_path": str(plan["teacher_model_path"]),
        "dataset_path": str(plan["dataset_path"]),
        "empty_emb_path": str(plan["empty_emb_path"]),
        "stage2_dir": str(plan["stage2_dir"]),
        "checkpoint": str(plan["checkpoint"]),
        "environment": plan["stage2"]["env"],
        "argv": plan["stage2"]["argv"],
        "command": plan["stage2"]["command"],
    }
    path = plan["run_dir"] / "run_manifest.json"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _validate_real_inputs(plan: dict) -> None:
    for label in ("stage1_ckpt", "teacher_model_path", "dataset_path"):
        if not plan[label].is_dir():
            raise FileNotFoundError(f"Missing {label}: {plan[label]}")
    if not plan["empty_emb_path"].is_file():
        raise FileNotFoundError(
            f"Missing empty embedding: {plan['empty_emb_path']}"
        )
    if not Path(plan["stage2"]["argv"][0]).is_file():
        raise FileNotFoundError(f"Missing torchrun: {plan['stage2']['argv'][0]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--teacher-model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--empty-emb-path", type=Path, required=True)
    parser.add_argument("--stage1-ckpt", type=Path, required=True)
    parser.add_argument("--torchrun", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--master-port", type=int, default=29671)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_run_plan(args)
    print(f"VARIANT={plan['variant']['name']}")
    print(f"RUN_DIR={plan['run_dir']}")
    print(f"TRAIN_MANIFEST={plan['train_manifest']}")
    print(f"HELDOUT_MANIFEST={plan['heldout_manifest']}")
    print(f"COMMAND={plan['stage2']['command']}")
    if args.dry_run:
        return
    _validate_real_inputs(plan)
    manifest = write_run_files(plan)
    print(f"RUN_MANIFEST={manifest}")
    if args.execute:
        env = os.environ.copy()
        env.update(plan["stage2"]["env"])
        subprocess.run(
            plan["stage2"]["argv"],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
