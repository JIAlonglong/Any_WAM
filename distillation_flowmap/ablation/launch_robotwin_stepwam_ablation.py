#!/usr/bin/env python3
"""Launch or dry-run one RobotWin StepWAM ablation training job."""

import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = REPO_ROOT / "distillation_flowmap" / "output_robotwin_stepwam_ablation"
SAFE_SHELL_VALUE = re.compile(r"^[A-Za-z0-9_./,:=+-]+$")


def first_existing(candidates):
    for path in candidates:
        path = Path(path)
        if path.exists():
            return path
    return Path(candidates[0])


DEFAULT_TEACHER = first_existing([
    REPO_ROOT / "checkpoints" / "lingbot-va-posttrain-robotwin",
    REPO_ROOT / "checkpoints" / "base",
])
DEFAULT_DATASET = first_existing([
    REPO_ROOT / "training_data" / "lerobot_robotwin_eef_aug_500",
    Path("/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500"),
])
DEFAULT_EMPTY_EMB = first_existing([
    DEFAULT_DATASET / "empty_emb.pt",
    DEFAULT_DATASET.parent / "empty_emb.pt",
    Path("/root/nas/junjie/data/robotwin-clean-and-aug-lerobot/empty_emb.pt"),
])
DEFAULT_TORCHRUN = first_existing([
    Path("/root/nas/junjie/conda_envs/any_wam/bin/torchrun"),
    "torchrun",
])


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata():
    variants = read_json(ABLATION_DIR / "robotwin_stepwam_variants.json")
    tasks = read_json(ABLATION_DIR / "robotwin_stepwam_tasks.json")
    return variants, tasks


def find_variant(variants, name_or_id):
    for variant in variants["variants"]:
        if variant["name"] == name_or_id or variant["id"] == name_or_id:
            return variant
    known = ", ".join(v["name"] for v in variants["variants"])
    raise ValueError(f"Unknown variant {name_or_id!r}; known variants: {known}")


def shell_value(value):
    text = str(value)
    if SAFE_SHELL_VALUE.match(text):
        return text
    return shlex.quote(text)


def env_command(env, argv):
    env_part = " ".join(f"{key}={shell_value(value)}" for key, value in sorted(env.items()))
    return f"{env_part} {shlex.join([str(v) for v in argv])}"


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def build_stage_command(
    *,
    stage_name,
    config_module,
    stage_env,
    output_dir,
    max_steps,
    teacher_model_path,
    dataset_path,
    gradient_accumulation_steps,
    ngpu,
    master_port,
    torchrun_path,
    resume_from_path=None,
):
    env = {
        **stage_env,
        "CONFIG_FILE": config_module,
        "OUTPUT_DIR": str(output_dir),
        "MAX_TRAIN_STEPS": str(max_steps),
        "WANDB_MODE": os.environ.get("WANDB_MODE", "offline"),
    }
    if resume_from_path is not None:
        env["RESUME_FROM_PATH"] = str(resume_from_path)
        env["RESUME_ONLINE_FROM_TARGET"] = os.environ.get("RESUME_ONLINE_FROM_TARGET", "1")

    argv = [
        str(torchrun_path),
        f"--nproc_per_node={ngpu}",
        f"--master_port={master_port}",
        "distillation_flowmap/train.py",
        "--teacher-model-path",
        str(teacher_model_path),
        "--dataset-path",
        str(dataset_path),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
    ]
    return {
        "stage": stage_name,
        "env": env,
        "argv": argv,
        "command": env_command(env, argv),
    }


def build_run_plan(args):
    metadata, tasks = load_metadata()
    defaults = metadata["defaults"]
    variant = find_variant(metadata, args.variant)
    root = Path(args.root)
    run_dir = root / variant["name"] / f"seed_{args.seed}"
    stage1_dir = run_dir / "stage1"
    stage2_dir = run_dir / "stage2"
    stage1_steps = args.stage1_steps or int(defaults["stage1_steps"])
    stage2_steps = args.stage2_steps or int(defaults["stage2_steps"])
    stage1_ckpt = stage1_dir / "checkpoints" / f"step_{stage1_steps}"
    stage2_ckpt = stage2_dir / "checkpoints" / f"step_{stage2_steps}"

    common_env = {}
    if args.empty_emb_path is not None:
        common_env["EMPTY_EMB_PATH"] = str(args.empty_emb_path)

    stage1 = build_stage_command(
        stage_name="stage1",
        config_module=defaults["stage1_config"],
        stage_env={**common_env, **variant.get("stage1_env", {})},
        output_dir=stage1_dir,
        max_steps=stage1_steps,
        teacher_model_path=args.teacher_model_path,
        dataset_path=args.dataset_path,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ngpu=args.ngpu,
        master_port=args.master_port,
        torchrun_path=args.torchrun,
    )
    stage2 = build_stage_command(
        stage_name="stage2",
        config_module=defaults["stage2_config"],
        stage_env={**common_env, **variant.get("stage2_env", {})},
        output_dir=stage2_dir,
        max_steps=stage2_steps,
        teacher_model_path=args.teacher_model_path,
        dataset_path=args.dataset_path,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ngpu=args.ngpu,
        master_port=args.master_port + 1,
        torchrun_path=args.torchrun,
        resume_from_path=stage1_ckpt,
    )
    manifest = {
        "git_hash": git_hash(),
        "variant": variant["name"],
        "variant_id": variant["id"],
        "seed": args.seed,
        "teacher_model_path": str(args.teacher_model_path),
        "dataset_path": str(args.dataset_path),
        "empty_emb_path": str(args.empty_emb_path) if args.empty_emb_path else None,
        "run_dir": str(run_dir),
        "stage1_ckpt": str(stage1_ckpt),
        "stage2_ckpt": str(stage2_ckpt),
        "task_list": tasks,
        "stage1_env": stage1["env"],
        "stage2_env": stage2["env"],
        "commands": {
            "stage1": stage1["command"],
            "stage2": stage2["command"],
        },
    }
    return {
        "variant": variant,
        "run_dir": run_dir,
        "stage1": stage1,
        "stage2": stage2,
        "manifest": manifest,
    }


def write_manifest(run_dir, manifest):
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest_path


def run_command(command):
    subprocess.run(command, cwd=REPO_ROOT, shell=True, check=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--teacher-model-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--empty-emb-path", type=Path, default=DEFAULT_EMPTY_EMB)
    parser.add_argument("--torchrun", type=Path, default=Path(os.environ.get("TORCHRUN", DEFAULT_TORCHRUN)))
    parser.add_argument("--stage1-steps", type=int, default=None)
    parser.add_argument("--stage2-steps", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=29620)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("both", "stage1", "stage2"),
        default="both",
        help="Run both stages or only one stage.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plan = build_run_plan(args)
    print(f"RUN_DIR={plan['run_dir']}")
    print(f"STAGE1: {plan['stage1']['command']}")
    print(f"STAGE2: {plan['stage2']['command']}")

    if args.dry_run:
        print(json.dumps(plan["manifest"], indent=2, sort_keys=True))
        return

    manifest_path = write_manifest(plan["run_dir"], plan["manifest"])
    print(f"MANIFEST={manifest_path}")
    if args.stage in ("both", "stage1"):
        run_command(plan["stage1"]["command"])
    if args.stage in ("both", "stage2"):
        run_command(plan["stage2"]["command"])


if __name__ == "__main__":
    main()
