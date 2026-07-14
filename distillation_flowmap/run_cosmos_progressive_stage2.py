#!/usr/bin/env python3
"""Run one checkpointed chunk of the Cosmos K=4 -> K=2 -> K=1 protocol."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "distillation_flowmap" / "output_libero_cosmos_policy_stage2_progressive_20260714_full"
DEFAULT_DATASET = REPO_ROOT / "training_data" / "libero-long-lerobot"
DEFAULT_STAGE1 = (
    Path("/root/nas/junjie/jj/Any_WAM/distillation_flowmap")
    / "output_libero_cosmos_policy_stage1_cosmos_latent_cdiff_8gpu_20260706_cosmos_latent_s1s2_8gpu"
    / "checkpoints" / "step_5000"
)
DEFAULT_TEACHER = Path(
    "/root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
)
DEFAULT_TORCHRUN = Path("/root/nas/junjie/conda_envs/any_wam/bin/torchrun")

STAGE_SPECS = {
    "s4": {"teacher_steps": 8, "student_steps": 4, "max_steps": 5000},
    "s2": {"teacher_steps": 4, "student_steps": 2, "max_steps": 3000},
    "s1": {"teacher_steps": 4, "student_steps": 1, "max_steps": 3000},
}


def _as_str_env(values):
    return {str(key): str(value) for key, value in values.items()}


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def build_stage_chunk_plan(
    *,
    stage,
    root,
    dataset_path,
    train_manifest,
    selection_manifest,
    eval_pairs,
    selection_cache_dir,
    stage1_checkpoint,
    current_step,
    chunk_size,
    master_port,
    train_seed,
    torchrun,
    resume_from_path=None,
    teacher_model_path=DEFAULT_TEACHER,
    gradient_accumulation_steps=1,
):
    """Build one resumable train/eval chunk without launching it."""
    stage = str(stage).lower()
    if stage not in STAGE_SPECS:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {sorted(STAGE_SPECS)}")
    spec = dict(STAGE_SPECS[stage])
    root = Path(root)
    output_dir = root / stage
    current_step = int(current_step)
    chunk_size = int(chunk_size)
    gradient_accumulation_steps = int(gradient_accumulation_steps)
    if current_step < 0 or current_step >= spec["max_steps"]:
        raise ValueError(f"current_step must be in [0, {spec['max_steps'] - 1}]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if gradient_accumulation_steps != 1:
        raise ValueError(
            "Cosmos progressive standalone OPD requires "
            "gradient_accumulation_steps=1"
        )
    target_step = min(current_step + chunk_size, spec["max_steps"])

    if current_step == 0:
        if stage == "s4":
            initial_checkpoint = Path(stage1_checkpoint)
        else:
            if resume_from_path is None:
                raise ValueError(
                    f"{stage} requires an explicit selected predecessor checkpoint"
                )
            initial_checkpoint = Path(resume_from_path)
        reset_resume_step = "1"
        resume_optimizer_state = "0"
    else:
        initial_checkpoint = output_dir / "checkpoints" / f"step_{current_step}"
        reset_resume_step = "0"
        resume_optimizer_state = "1"

    common_env = {
        "CONFIG_FILE": "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive",
        "COSMOS_PROGRESSIVE_STAGE": stage,
        "COSMOS_PROGRESSIVE_OUTPUT_ROOT": root,
        "OUTPUT_DIR": output_dir,
        "MAX_TRAIN_STEPS": spec["max_steps"],
        "STOP_AFTER_STEP": target_step,
        "SAVE_INTERVAL": chunk_size,
        "DATASET_SAMPLE_MANIFEST": Path(train_manifest),
        "RESUME_FROM_PATH": initial_checkpoint,
        "RESUME_ONLINE_FROM_TARGET": "0",
        "RESET_RESUME_STEP": reset_resume_step,
        "RESUME_OPTIMIZER_STATE": resume_optimizer_state,
        "TRAIN_SEED": train_seed,
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
    train_env = _as_str_env({
        **common_env,
        "CUDA_VISIBLE_DEVICES": "6,7",
        "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES": "6,7",
    })
    train_argv = [
        str(torchrun),
        "--nproc_per_node=2",
        f"--master_port={int(master_port)}",
        "distillation_flowmap/train.py",
        "--teacher-model-path", str(teacher_model_path),
        "--dataset-path", str(dataset_path),
        "--output-dir", str(output_dir),
        "--resume-from-path", str(initial_checkpoint),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
    ]

    checkpoint_dir = output_dir / "checkpoints" / f"step_{target_step}"
    metrics_path = output_dir / "metrics" / f"selection_step_{target_step}.json"
    eval_env = _as_str_env({
        **common_env,
        "CUDA_VISIBLE_DEVICES": "6",
        "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES": "7",
    })
    eval_argv = [
        sys.executable,
        "distillation_flowmap/eval_cosmos_progressive_stage2.py",
        "--checkpoint-transformer", str(checkpoint_dir / "online_student" / "transformer"),
        "--dataset-path", str(dataset_path),
        "--manifest", str(selection_manifest),
        "--pairs", str(eval_pairs),
        "--cache-dir", str(selection_cache_dir),
        "--student-steps", str(spec["student_steps"]),
        "--teacher-steps", str(spec["teacher_steps"]),
        "--output-json", str(metrics_path),
    ]
    return {
        "schema": "cosmos_progressive_stage2_chunk_v1",
        "git_hash": _git_hash(),
        "stage": stage,
        "spec": spec,
        "root": root,
        "output_dir": output_dir,
        "current_step": current_step,
        "target_step": target_step,
        "checkpoint_dir": checkpoint_dir,
        "metrics_path": metrics_path,
        "train_manifest": Path(train_manifest),
        "selection_manifest": Path(selection_manifest),
        "eval_pairs": Path(eval_pairs),
        "selection_cache_dir": Path(selection_cache_dir),
        "train_env": train_env,
        "train_argv": train_argv,
        "eval_env": eval_env,
        "eval_argv": eval_argv,
    }


def _validate_fixed_cache(plan):
    manifest = json.loads(plan["selection_manifest"].read_text(encoding="utf-8"))
    pairs = json.loads(plan["eval_pairs"].read_text(encoding="utf-8")).get("pairs", [])
    cache_dir = plan["selection_cache_dir"]
    missing = []
    for record in manifest.get("records", []):
        for pair in pairs:
            name = f"sample_{int(record['index']):06d}__{pair['pair_id']}.pt"
            if not (cache_dir / name).is_file():
                missing.append(name)
                if len(missing) >= 5:
                    break
        if len(missing) >= 5:
            break
    if missing:
        raise FileNotFoundError(
            "Selection teacher cache is incomplete; missing " + ", ".join(missing)
        )


def _write_chunk_manifest(plan):
    manifest_path = plan["output_dir"] / "protocol" / f"chunk_{plan['target_step']}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(_json_ready(plan), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def execute_stage_chunk(plan):
    """Validate fixed inputs, train exactly one chunk, then run held-out selection."""
    _validate_fixed_cache(plan)
    resume_config = Path(plan["train_env"]["RESUME_FROM_PATH"]) / "online_student" / "transformer" / "config.json"
    if not resume_config.is_file():
        raise FileNotFoundError(f"Missing resume checkpoint: {resume_config}")
    _write_chunk_manifest(plan)
    train_env = {**os.environ, **plan["train_env"]}
    subprocess.run(plan["train_argv"], cwd=REPO_ROOT, env=train_env, check=True)
    checkpoint_config = plan["checkpoint_dir"] / "online_student" / "transformer" / "config.json"
    if not checkpoint_config.is_file():
        raise RuntimeError(f"Training ended without expected checkpoint: {checkpoint_config}")
    eval_env = {**os.environ, **plan["eval_env"]}
    subprocess.run(plan["eval_argv"], cwd=REPO_ROOT, env=eval_env, check=True)
    if not plan["metrics_path"].is_file():
        raise RuntimeError(f"Held-out evaluation did not write metrics: {plan['metrics_path']}")
    history_path = plan["output_dir"] / "metrics" / "selection_history.jsonl"
    with history_path.open("a", encoding="utf-8") as f:
        f.write(plan["metrics_path"].read_text(encoding="utf-8").strip() + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGE_SPECS), required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--selection-manifest", type=Path, default=None)
    parser.add_argument("--eval-pairs", type=Path, default=None)
    parser.add_argument("--selection-cache-dir", type=Path, default=None)
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument(
        "--resume-from-path", type=Path, default=None,
        help="Selected preceding-stage checkpoint; required when first starting S2 or S1.",
    )
    parser.add_argument("--current-step", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--master-port", type=int, default=29761)
    parser.add_argument("--train-seed", type=int, default=20260714)
    parser.add_argument("--teacher-model-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--torchrun", type=Path, default=DEFAULT_TORCHRUN)
    parser.add_argument("--run", action="store_true", help="Execute; otherwise print the plan only.")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol_dir = args.root / "protocol"
    plan = build_stage_chunk_plan(
        stage=args.stage,
        root=args.root,
        dataset_path=args.dataset_path,
        train_manifest=args.train_manifest or protocol_dir / "train_manifest.json",
        selection_manifest=args.selection_manifest or protocol_dir / "selection_manifest.json",
        eval_pairs=args.eval_pairs or protocol_dir / "eval_pairs.json",
        selection_cache_dir=args.selection_cache_dir or protocol_dir / "teacher_cache" / "selection",
        stage1_checkpoint=args.stage1_checkpoint,
        resume_from_path=args.resume_from_path,
        current_step=args.current_step,
        chunk_size=args.chunk_size,
        master_port=args.master_port,
        train_seed=args.train_seed,
        torchrun=args.torchrun,
        teacher_model_path=args.teacher_model_path,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    print(json.dumps(_json_ready(plan), indent=2, sort_keys=True))
    if args.run:
        execute_stage_chunk(plan)


if __name__ == "__main__":
    main()
