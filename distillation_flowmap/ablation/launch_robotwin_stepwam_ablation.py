#!/usr/bin/env python3
"""Launch or dry-run one RobotWin StepWAM ablation training job."""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from distillation_flowmap.ablation.robotwin_mini_protocol import (
    DEFAULT_EVAL_PAIRS,
    protocol_manifest_paths,
    write_protocol_manifests,
)


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


def split_csv(value):
    if not value:
        return []
    return [v.strip() for v in str(value).replace(";", ",").split(",") if v.strip()]


def representative_task_filter(tasks):
    splits = tasks.get("splits", {})
    ordered = []
    for split_name in ("easy", "hard"):
        ordered.extend(splits.get(split_name, []))
    for split_tasks in splits.values():
        ordered.extend(split_tasks)

    deduped = []
    seen = set()
    for task in ordered:
        if task not in seen:
            deduped.append(task)
            seen.add(task)
    return deduped


def task_preset_filter(tasks, preset):
    preset = str(preset)
    if preset in ("representative", "default"):
        return representative_task_filter(tasks)
    presets = tasks.get("presets", {})
    if preset not in presets:
        known = ", ".join(sorted(["representative", *presets.keys()]))
        raise ValueError(f"Unknown task preset {preset!r}; known presets: {known}")
    return list(presets[preset])


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
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
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
    if args.task_filter and args.task_preset:
        raise ValueError("--task-filter and --task-preset are mutually exclusive")
    if args.all_dataset_tasks and args.task_preset:
        raise ValueError("--all-dataset-tasks and --task-preset are mutually exclusive")

    if args.all_dataset_tasks:
        selected_task_preset = "all"
    elif args.task_filter:
        selected_task_preset = "custom"
    else:
        selected_task_preset = args.task_preset or "representative"

    stage1_run_dir = (
        root / "shared_stage1" / selected_task_preset / f"seed_{args.seed}"
        if args.use_shared_stage1
        else run_dir
    )
    stage1_dir = stage1_run_dir / "stage1"
    stage2_dir = run_dir / "stage2"
    stage1_steps = args.stage1_steps or int(defaults["stage1_steps"])
    stage2_steps = args.stage2_steps or int(defaults["stage2_steps"])
    derived_stage1_ckpt = stage1_dir / "checkpoints" / f"step_{stage1_steps}"
    stage1_ckpt = args.stage1_ckpt or derived_stage1_ckpt
    stage2_ckpt = stage2_dir / "checkpoints" / f"step_{stage2_steps}"

    if args.all_dataset_tasks:
        task_filter = []
    elif args.task_filter:
        task_filter = split_csv(args.task_filter)
    else:
        task_filter = task_preset_filter(tasks, selected_task_preset)

    protocol_paths = protocol_manifest_paths(root, selected_task_preset, args.protocol_seed)
    common_env = {}
    if args.empty_emb_path is not None:
        common_env["EMPTY_EMB_PATH"] = str(args.empty_emb_path)
    if task_filter:
        common_env["DATASET_TASK_FILTER"] = ",".join(task_filter)
        common_env["DATASET_SAMPLE_MANIFEST"] = protocol_paths["train_manifest_path"]
    if args.max_episodes_per_task is not None:
        common_env["DATASET_MAX_EPISODES_PER_TASK"] = str(args.max_episodes_per_task)
    if args.max_samples_per_task is not None:
        common_env["DATASET_MAX_SAMPLES_PER_TASK"] = str(args.max_samples_per_task)

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
        stage_env={
            **common_env,
            "ENABLE_GRAD_BRANCH_DIAGNOSTICS": "1",
            **variant.get("stage2_env", {}),
        },
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
        "task_preset": selected_task_preset,
        "teacher_model_path": str(args.teacher_model_path),
        "dataset_path": str(args.dataset_path),
        "empty_emb_path": str(args.empty_emb_path) if args.empty_emb_path else None,
        "run_dir": str(run_dir),
        "stage1_ckpt": str(stage1_ckpt),
        "shared_stage1_ckpt": (
            str(stage1_ckpt)
            if args.use_shared_stage1 or args.stage1_ckpt is not None
            else None
        ),
        "stage2_ckpt": str(stage2_ckpt),
        "task_list": tasks,
        "selected_task_filter": task_filter,
        "dataset_max_episodes_per_task": args.max_episodes_per_task,
        "dataset_max_samples_per_task": args.max_samples_per_task,
        "protocol_seed": args.protocol_seed,
        "train_samples_per_task": args.train_samples_per_task,
        "heldout_samples_per_task": args.heldout_samples_per_task,
        "eval_pairs": args.eval_pairs,
        **protocol_paths,
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
    parser.add_argument(
        "--stage1-ckpt",
        type=Path,
        default=None,
        help="Explicit Stage1 checkpoint for Stage2; overrides the derived path.",
    )
    parser.add_argument(
        "--task-filter",
        default=None,
        help="Comma-separated RobotWin task names. Defaults to the metadata Easy+Hard subset.",
    )
    parser.add_argument(
        "--task-preset",
        default=None,
        help="Named task preset from robotwin_stepwam_tasks.json, e.g. core4 or core6.",
    )
    parser.add_argument(
        "--all-dataset-tasks",
        action="store_true",
        help="Disable the default representative task filter and train on every dataset task.",
    )
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--max-samples-per-task", type=int, default=None)
    parser.add_argument("--protocol-seed", type=int, default=0)
    parser.add_argument("--train-samples-per-task", type=int, default=80)
    parser.add_argument("--heldout-samples-per-task", type=int, default=20)
    parser.add_argument("--eval-pairs", nargs="+", default=list(DEFAULT_EVAL_PAIRS))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=29620)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--use-shared-stage1",
        action="store_true",
        help="Store/read Stage 1 under <root>/shared_stage1/<task_preset>/seed_<seed>/.",
    )
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

    if (
        (args.use_shared_stage1 or args.stage1_ckpt is not None)
        and args.stage == "stage2"
        and not Path(plan["manifest"]["stage1_ckpt"]).exists()
    ):
        raise FileNotFoundError(
            "Shared Stage 1 checkpoint is required for --stage stage2: "
            f"{plan['manifest']['stage1_ckpt']}"
        )

    if plan["manifest"]["selected_task_filter"]:
        write_protocol_manifests(
            root=args.root,
            task_preset=plan["manifest"]["task_preset"],
            protocol_seed=args.protocol_seed,
            task_names=plan["manifest"]["selected_task_filter"],
            train_samples_per_task=args.train_samples_per_task,
            heldout_samples_per_task=args.heldout_samples_per_task,
            pairs=args.eval_pairs,
        )

    manifest_path = write_manifest(plan["run_dir"], plan["manifest"])
    print(f"MANIFEST={manifest_path}")
    if args.stage in ("both", "stage1"):
        run_command(plan["stage1"]["command"])
    if args.stage in ("both", "stage2"):
        run_command(plan["stage2"]["command"])


if __name__ == "__main__":
    main()
