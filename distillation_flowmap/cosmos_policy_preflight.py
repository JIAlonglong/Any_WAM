#!/usr/bin/env python3
"""Preflight checks for LIBERO Cosmos Policy raw distillation."""

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
WAN_VA_ROOT = PROJECT_ROOT / "wan_va"
if str(WAN_VA_ROOT) not in sys.path:
    sys.path.append(str(WAN_VA_ROOT))

from distillation_flowmap.cosmos_policy_adapter import (
    CosmosPolicyActionTeacher,
    resolve_cosmos_policy_assets,
)


DEFAULT_TEACHER = "/root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
DEFAULT_STUDENT = "/root/nas/junjie/jj/Any_WAM/checkpoints/lingbot-va-posttrain-libero"
DEFAULT_DATASET = str(PROJECT_ROOT / "training_data/libero-long-lerobot")
DEFAULT_COSMOS_REPO = "/root/nas/junjie/cosmos_predict2_5/repos/cosmos-predict2.5"
DEFAULT_COSMOS_PYTHON = "/root/nas/junjie/cosmos_predict2_5/envs/predict2_py310/bin/python"
DEFAULT_EXTRA_PYTHONPATH = "/root/nas/junjie/conda_envs/any_wam/lib/python3.10/site-packages"
DEFAULT_LOCAL_MODEL = "/root/nas/junjie/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World"
DEFAULT_TOKENIZER_SOURCE = "/root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth"


def ok(message):
    print(f"OK {message}")


def require_path(path, kind, label):
    path = Path(path).expanduser()
    exists = path.is_file() if kind == "file" else path.is_dir()
    if not exists:
        raise FileNotFoundError(f"{label} not found: {path}")
    ok(f"{label}: {path}")
    return path


def check_student_base(path):
    root = require_path(path, "dir", "student base")
    config = root / "config.json" if root.name == "transformer" else root / "transformer/config.json"
    if not config.is_file():
        raise FileNotFoundError(f"student base missing transformer/config.json under {root}")
    ok(f"student base transformer config: {config}")


def check_tokenizer(local_model_dir, tokenizer_source, fix):
    tokenizer = Path(local_model_dir).expanduser() / "tokenizer/tokenizer.pth"
    if tokenizer.is_file():
        ok(f"local tokenizer: {tokenizer}")
        return
    source = Path(tokenizer_source).expanduser()
    if not source.is_file():
        raise FileNotFoundError(
            f"local tokenizer missing at {tokenizer}; source tokenizer also missing: {source}"
        )
    if not fix:
        raise FileNotFoundError(
            f"local tokenizer missing at {tokenizer}; rerun with --fix-local-tokenizer "
            f"to symlink {source}"
        )
    tokenizer.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source, tokenizer)
    ok(f"created local tokenizer symlink: {tokenizer} -> {source}")


def check_cosmos_python(args):
    require_path(args.cosmos_python, "file", "Cosmos python")
    require_path(args.cosmos_repo, "dir", "Cosmos repo")
    config_file = Path(args.cosmos_config_file)
    config_path = config_file if config_file.is_absolute() else Path(args.cosmos_repo) / config_file
    require_path(config_path, "file", "Cosmos config file")

    code = (
        "import os, sys; "
        f"sys.path.insert(0, {args.cosmos_repo!r}); "
        f"extra = {args.extra_pythonpath!r}; "
        "[sys.path.append(p) for p in extra.split(os.pathsep) if p and p not in sys.path]; "
        "import h5py; "
        "import cosmos_predict2; "
        "from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot import cosmos_utils; "
        "print('cosmos import ok')"
    )
    env = os.environ.copy()
    pythonpath = [args.cosmos_repo]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if args.local_model_dir:
        env["COSMOS_PREDICT25_LOCAL_MODEL_DIR"] = args.local_model_dir
    result = subprocess.run(
        [args.cosmos_python, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Cosmos python import check failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    ok("Cosmos python can import h5py and official cosmos_utils")


def load_dataset_sample(args):
    os.environ["COSMOS_POLICY_PATH"] = args.teacher_model_path
    os.environ["STUDENT_BASE_MODEL_PATH"] = args.student_base_model_path
    os.environ["DATASET_PATH"] = args.dataset_path
    os.environ["COSMOS_POLICY_USE_RAW_INFERENCE"] = "1"
    os.environ["COSMOS_POLICY_INFERENCE_MODE"] = args.inference_mode
    os.environ["COSMOS_POLICY_PYTHON"] = args.cosmos_python
    os.environ["COSMOS_PREDICT2_REPO"] = args.cosmos_repo
    os.environ["COSMOS_POLICY_EXTRA_PYTHONPATH"] = args.extra_pythonpath
    os.environ["COSMOS_PREDICT25_LOCAL_MODEL_DIR"] = args.local_model_dir
    os.environ["COSMOS_POLICY_CONFIG_NAME"] = args.cosmos_config_name
    os.environ["COSMOS_POLICY_CONFIG_FILE"] = args.cosmos_config_file
    os.environ["COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION"] = str(args.num_denoising_steps_action)

    config_mod = importlib.import_module("distillation_flowmap.config_libero_cosmos_policy_stage1")
    cfg = config_mod.cfg
    cfg.dataset_path = args.dataset_path
    cfg.teacher_model_path = args.teacher_model_path
    cfg.student_base_model_path = args.student_base_model_path
    cfg.return_raw_observation = True
    cfg.cosmos_policy_use_raw_inference = True
    cfg.cache_dataset_in_memory = False
    cfg.load_worker = 0
    cfg.batch_size = 1

    from distillation.patches import SafeMultiLatentLeRobotDataset, install_flash_attn_stub

    install_flash_attn_stub()

    dataset = SafeMultiLatentLeRobotDataset(config=cfg)
    sample = dataset[0]
    for key in ("raw_primary_image", "raw_wrist_image", "raw_proprio", "raw_task", "actions"):
        if key not in sample:
            raise KeyError(f"dataset sample missing {key}")
    ok(
        "dataset raw sample: "
        f"primary={tuple(sample['raw_primary_image'].shape)} "
        f"wrist={tuple(sample['raw_wrist_image'].shape)} "
        f"proprio={tuple(sample['raw_proprio'].shape)} "
        f"actions={tuple(sample['actions'].shape)} "
        f"task={sample['raw_task']!r}"
    )
    return cfg, sample


def check_worker(args, cfg, sample):
    from torch.utils.data._utils.collate import default_collate

    batch = default_collate([sample])
    teacher = CosmosPolicyActionTeacher(args.teacher_model_path, dtype=torch.float32, config=cfg)
    x0 = teacher.action_target_x0({"latent": batch["actions"]}, raw_batch=batch)
    if x0 is None:
        raise RuntimeError("raw worker returned None; cosmos_policy_use_raw_inference is not active")
    ok(f"raw worker x0: shape={tuple(x0.shape)} dtype={x0.dtype}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model-path", default=os.environ.get("COSMOS_POLICY_PATH", DEFAULT_TEACHER))
    parser.add_argument("--student-base-model-path", default=os.environ.get("STUDENT_BASE_MODEL_PATH", DEFAULT_STUDENT))
    parser.add_argument("--dataset-path", default=os.environ.get("DATASET_PATH", DEFAULT_DATASET))
    parser.add_argument("--cosmos-repo", default=os.environ.get("COSMOS_PREDICT2_REPO", DEFAULT_COSMOS_REPO))
    parser.add_argument("--cosmos-python", default=os.environ.get("COSMOS_POLICY_PYTHON", DEFAULT_COSMOS_PYTHON))
    parser.add_argument("--extra-pythonpath", default=os.environ.get("COSMOS_POLICY_EXTRA_PYTHONPATH", DEFAULT_EXTRA_PYTHONPATH))
    parser.add_argument("--local-model-dir", default=os.environ.get("COSMOS_PREDICT25_LOCAL_MODEL_DIR", DEFAULT_LOCAL_MODEL))
    parser.add_argument("--tokenizer-source", default=os.environ.get("COSMOS_PREDICT25_TOKENIZER_SOURCE", DEFAULT_TOKENIZER_SOURCE))
    parser.add_argument(
        "--cosmos-config-name",
        default=os.environ.get("COSMOS_POLICY_CONFIG_NAME", "cosmos_predict2_2b_480p_libero__inference_only"),
    )
    parser.add_argument(
        "--cosmos-config-file",
        default=os.environ.get(
            "COSMOS_POLICY_CONFIG_FILE",
            "cosmos_predict2/_src/predict2/cosmos_policy/config/config.py",
        ),
    )
    parser.add_argument("--inference-mode", default=os.environ.get("COSMOS_POLICY_INFERENCE_MODE", "subprocess"))
    parser.add_argument(
        "--num-denoising-steps-action",
        type=int,
        default=int(os.environ.get("COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION", 1)),
    )
    parser.add_argument("--fix-local-tokenizer", action="store_true")
    parser.add_argument("--check-dataset", action="store_true")
    parser.add_argument("--check-worker", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    assets = resolve_cosmos_policy_assets(args.teacher_model_path)
    ok(f"Cosmos Policy checkpoint: {assets['weight_path']}")
    ok(f"Cosmos Policy stats: {assets['dataset_stats_path']}")
    ok(f"Cosmos Policy text embeddings: {assets['t5_embeddings_path']}")
    check_student_base(args.student_base_model_path)
    require_path(args.dataset_path, "dir", "dataset")
    require_path(args.local_model_dir, "dir", "local model dir")
    check_tokenizer(args.local_model_dir, args.tokenizer_source, args.fix_local_tokenizer)
    check_cosmos_python(args)

    cfg = sample = None
    if args.check_dataset or args.check_worker:
        cfg, sample = load_dataset_sample(args)
    if args.check_worker:
        check_worker(args, cfg, sample)
    ok("preflight complete")


if __name__ == "__main__":
    main()
