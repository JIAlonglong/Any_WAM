"""Single-GPU post-hoc Figure 4 sweep for LingBot-VA/LIBERO checkpoints."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping

import torch

from distillation_flowmap.mechanism_checkpoint_sweep import (
    CheckpointSpec,
    discover_target_checkpoints,
    paper_record,
    plot_figure4,
    write_records_atomic,
)
from distillation_flowmap.mechanism_diagnostics import (
    build_diagnostic_probe_noise,
    means_from_reduced_stats,
)


LOGGER = logging.getLogger("libero-figure4-mechanism")
_CONTINUOUS_FACTOR = 1
_DEFAULT_CONFIG = (
    "distillation_flowmap.config_libero_fullfinetune_stage2_video_only_opd"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep target-student checkpoints for LIBERO Figure 4"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--teacher-model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--student-role",
        choices=("target_student",),
        default="target_student",
    )
    parser.add_argument("--steps", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--r", type=int, default=500)
    parser.add_argument("--s", type=int, default=250)
    parser.add_argument("--teacher-steps", type=int, default=8)
    parser.add_argument("--gpu-id", default="0")
    return parser


def select_checkpoints(
    checkpoints: Iterable[CheckpointSpec],
    requested_steps: Iterable[int] | None,
) -> list[CheckpointSpec]:
    ordered = sorted(checkpoints, key=lambda item: item.step)
    if requested_steps is None:
        return ordered
    requested = {int(step) for step in requested_steps}
    found = {item.step for item in ordered}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"requested checkpoint steps not found: {missing}")
    return [item for item in ordered if item.step in requested]


def _load_checkpoint_config(checkpoint: CheckpointSpec) -> dict:
    config_path = checkpoint.transformer / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_factor1_checkpoint(checkpoint: CheckpointSpec) -> dict:
    config = _load_checkpoint_config(checkpoint)
    factor = int(config.get("action_downsample_factor", -1))
    if factor != _CONTINUOUS_FACTOR:
        raise ValueError(
            f"{checkpoint.transformer} must declare "
            f"action_downsample_factor=1, found {factor}"
        )
    metadata_step = int(config.get("checkpoint_step", checkpoint.step))
    if metadata_step != checkpoint.step:
        raise ValueError(
            f"checkpoint metadata step {metadata_step} does not match "
            f"directory step {checkpoint.step}"
        )
    return config


def configure_eval_config(
    base_config,
    *,
    checkpoint: CheckpointSpec,
    output_dir: Path,
    teacher_model_path: Path,
    dataset_path: Path,
    seed: int,
    r: int,
    s: int,
    teacher_steps: int,
):
    cfg = copy.deepcopy(base_config)
    cfg.rank = 0
    cfg.local_rank = 0
    cfg.world_size = 1
    cfg.output_dir = str(output_dir)
    cfg.teacher_model_path = str(teacher_model_path)
    cfg.dataset_path = str(dataset_path)
    cfg.empty_emb_path = str(dataset_path / "empty_emb.pt")
    cfg.resume_from_path = str(checkpoint.root)
    cfg.resume_from_step = None
    cfg.resume_online_from_target = True
    cfg.reset_resume_step = False
    cfg.resume_optimizer_state = False
    cfg.offline_eval_skip_target_student = True
    cfg.offline_eval_use_fsdp_teacher = False
    cfg.offline_eval_skip_teacher = False
    cfg.enable_wandb = False
    cfg.use_torch_compile = False
    cfg.gradient_checkpointing = False
    cfg.opd_aux_gradient_checkpointing = False
    cfg.skip_teacher_compile = True
    cfg.load_worker = 0
    cfg.batch_size = 1
    cfg.action_downsample_factor = _CONTINUOUS_FACTOR
    cfg.mechanism_diagnostics = True
    cfg.mechanism_diagnostic_seed = int(seed)
    cfg.mechanism_diagnostic_r = int(r)
    cfg.mechanism_diagnostic_s = int(s)
    cfg.mechanism_diagnostic_teacher_steps = int(teacher_steps)
    cfg.mechanism_diagnostic_num_batches = 1
    return cfg


def probe_metadata(
    batch: Mapping,
    *,
    dataset_index: int,
    seed: int,
    r: int,
    s: int,
    teacher_steps: int,
) -> dict:
    tensor_shapes = {}
    tensor_dtypes = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor_shapes[key] = list(value.shape)
            tensor_dtypes[key] = str(value.dtype)
    scalar_context = {}
    for key, value in batch.items():
        if isinstance(value, (str, int, float, bool)):
            scalar_context[key] = value
    return {
        "dataset_index": int(dataset_index),
        "batch_index": int(batch.get("_mechanism_diagnostic_batch_index", 0)),
        "seed": int(seed),
        "r": int(r),
        "s": int(s),
        "teacher_integration_steps": int(teacher_steps),
        "batch_size": int(batch["latents"].shape[0]),
        "tensor_shapes": tensor_shapes,
        "tensor_dtypes": tensor_dtypes,
        "scalar_context": scalar_context,
        "action_mask_rule": "broadcast valid action tokens only",
        "checkpoint_step_enters_probe_seed": False,
    }


def build_manifest(
    *,
    git_commit: str,
    teacher_checkpoint: Path,
    checkpoints: Iterable[CheckpointSpec],
    probe: Mapping,
    dtype: str,
    gpu: str,
    peak_memory_bytes: int,
    runtime_seconds: float,
) -> dict:
    checkpoints = sorted(checkpoints, key=lambda item: item.step)
    return {
        "git_commit": git_commit,
        "teacher_checkpoint": str(teacher_checkpoint),
        "student_role": "target_student",
        "student_checkpoints": [
            {
                "step": item.step,
                "root": str(item.root),
                "transformer": str(item.transformer),
            }
            for item in checkpoints
        ],
        "checkpoint_steps": [item.step for item in checkpoints],
        "probe_bank": dict(probe),
        "dtype": dtype,
        "gpu": gpu,
        "peak_memory_bytes": int(peak_memory_bytes),
        "runtime_seconds": float(runtime_seconds),
        "metric_reduction": {
            "g_anchor_l2": "per-sample squared L2 mean",
            "g_anchor_mse": "valid video latent element MSE",
            "g_comp_l2": "per-sample squared L2 mean",
            "g_comp_mse": "valid video latent element MSE",
            "action_errors": "masked valid-action-token MSE",
        },
        "action_mask_rule": "broadcast valid action tokens only",
        "action_contexts": {
            "e_student": {
                "state": "(z_r.video, z_r.action)",
                "trajectory_synchronized": True,
            },
            "e_video": {
                "state": "(y_r.video, z_r.action)",
                "trajectory_synchronized": False,
                "description": (
                    "counterfactual teacher-video swap with student action "
                    "state fixed; not a causal oracle"
                ),
            },
            "e_joint": {
                "state": "(y_r.video, y_r.action)",
                "trajectory_synchronized": True,
            },
        },
        "epsilon": 1e-8,
    }


def render_reproduce_script(
    *,
    python: Path,
    repo_root: Path,
    run_root: Path,
    teacher_model_path: Path,
    dataset_path: Path,
    output_dir: Path,
    seed: int,
    r: int,
    s: int,
    teacher_steps: int,
    gpu_id: str,
) -> str:
    quote = lambda value: shlex.quote(str(value))
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {quote(repo_root)}",
            (
                f"CUDA_VISIBLE_DEVICES={quote(gpu_id)} "
                f"PYTHON={quote(python)} "
                "bash distillation_flowmap/run_libero_figure4_mechanism_1gpu.sh "
                f"--run-root {quote(run_root)} "
                f"--teacher-model-path {quote(teacher_model_path)} "
                f"--dataset-path {quote(dataset_path)} "
                f"--output-dir {quote(output_dir)} "
                "--student-role target_student "
                f"--seed {int(seed)} --r {int(r)} --s {int(s)} "
                f"--teacher-steps {int(teacher_steps)}"
            ),
            "",
        ]
    )


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_existing_records(output_dir: Path) -> list[dict]:
    path = output_dir / "mechanism_metrics.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def _configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", mode="a")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def _install_repo_imports(repo_root: Path) -> None:
    for path in (repo_root, repo_root / "wan_va", repo_root / "distillation_flowmap"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    from distillation.patches import install_flash_attn_stub

    install_flash_attn_stub()


def _dispose_training_only_state(trainer) -> None:
    if getattr(trainer, "tb_writer", None) is not None:
        trainer.tb_writer.close()
        trainer.tb_writer = None
    for name in (
        "optimizer",
        "lr_scheduler",
        "target_student",
        "discriminator",
        "discriminator_optimizer",
    ):
        if hasattr(trainer, name):
            try:
                delattr(trainer, name)
            except AttributeError:
                setattr(trainer, name, None)


def _dispose_student(trainer) -> None:
    for name in ("student", "_student_nofsdp"):
        model = getattr(trainer, name, None)
        if model is not None:
            setattr(trainer, name, None)
            del model
    trainer._nofsdp_synced = False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_flowmap_target_student(trainer, checkpoint: CheckpointSpec):
    validate_factor1_checkpoint(checkpoint)
    from modules.utils import load_transformer
    from model_flowmap import patch_model_forward, setup_flowmap_model
    from safetensors import safe_open

    LOGGER.info("Loading target student step=%d from %s", checkpoint.step, checkpoint.transformer)
    student = load_transformer(
        str(checkpoint.transformer),
        torch_dtype=torch.float32,
        torch_device="cpu",
        attn_mode=trainer.attn_mode,
    )
    student = student.to(trainer.dtype)
    student = setup_flowmap_model(
        student,
        gate_value=trainer.config.gate_value,
        deltatime_type=trainer.config.deltatime_type,
    )
    student = patch_model_forward(student)
    weights_path = checkpoint.transformer / "diffusion_pytorch_model.safetensors"
    delta_state = {}
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if ".delta_embedder." in key:
                delta_state[key] = handle.get_tensor(key)
    if not delta_state:
        raise RuntimeError(f"no FlowMap delta weights found in {weights_path}")
    _, unexpected = student.load_state_dict(delta_state, strict=False)
    if unexpected:
        raise RuntimeError(
            f"unexpected FlowMap delta keys for step {checkpoint.step}: {unexpected}"
        )
    student._flowmap_gradient_checkpointing = False
    student._flowmap_force_gradient_checkpointing = False
    student = student.to(device=trainer.device, dtype=trainer.dtype)
    student.requires_grad_(False)
    student.eval()
    trainer.student = student
    trainer._student_nofsdp = None
    trainer._nofsdp_synced = False
    return student


def _cpu_clone_batch(batch: Mapping) -> dict:
    cloned = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _device_batch(batch: Mapping, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=False)
        if isinstance(value, torch.Tensor)
        else copy.deepcopy(value)
        for key, value in batch.items()
    }


def _prepare_probe(trainer, *, seed: int, r: int, s: int, teacher_steps: int):
    batches = trainer._get_mechanism_diagnostic_batches()
    if len(batches) != 1:
        raise RuntimeError(f"expected one diagnostic batch, found {len(batches)}")
    batch = _cpu_clone_batch(batches[0])
    factor = int(trainer.config.action_downsample_factor)
    action_clean = batch["actions"][:, :, ::factor]
    probe = build_diagnostic_probe_noise(
        batch["latents"],
        action_clean,
        seed=seed,
        batch_index=int(batch.get("_mechanism_diagnostic_batch_index", 0)),
        rank=0,
    )
    dataset_index = int(seed) % len(trainer.train_loader.dataset)
    metadata = probe_metadata(
        batch,
        dataset_index=dataset_index,
        seed=seed,
        r=r,
        s=s,
        teacher_steps=teacher_steps,
    )
    return batch, probe, metadata


def _evaluate_checkpoint(trainer, batch: Mapping, probe: Mapping[str, torch.Tensor]):
    device_batch = _device_batch(batch, trainer.device)
    device_batch["_mechanism_probe_video_noise"] = probe["video_noise"].to(
        trainer.device
    )
    device_batch["_mechanism_probe_action_noise"] = probe["action_noise"].to(
        trainer.device
    )
    device_batch["_mechanism_diagnostic_context"] = (
        trainer._prepare_mechanism_diagnostic_context(device_batch)
    )
    with torch.inference_mode():
        stats = trainer._compute_mechanism_diagnostic_stats(
            device_batch, diagnostic_index=0
        )
    detached = {name: value.detach().float() for name, value in stats.items()}
    valid_flag = float(detached["diagnostic_valid"].item())
    count_keys = sorted(
        name for name in detached if name.endswith("_count")
    )
    counts = {float(detached[name].item()) for name in count_keys}
    if valid_flag != 1.0 or len(counts) != 1:
        raise RuntimeError(
            f"non-finite or inconsistent diagnostic stats: "
            f"valid={valid_flag}, counts={sorted(counts)}"
        )
    valid_count = int(next(iter(counts)))
    metrics = means_from_reduced_stats(detached)
    del device_batch
    return metrics, valid_count


def _write_runtime_artifacts(
    *,
    output_dir: Path,
    records: list[dict],
    repo_root: Path,
    args,
    checkpoints: list[CheckpointSpec],
    probe_meta: Mapping,
    dtype: str,
    gpu: str,
    started_at: float,
) -> None:
    write_records_atomic(records, output_dir)
    plot_figure4(records, output_dir)
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    manifest = build_manifest(
        git_commit=_git_commit(repo_root),
        teacher_checkpoint=args.teacher_model_path,
        checkpoints=checkpoints,
        probe=probe_meta,
        dtype=dtype,
        gpu=gpu,
        peak_memory_bytes=peak,
        runtime_seconds=time.perf_counter() - started_at,
    )
    manifest["evaluated_steps"] = sorted(int(row["step"]) for row in records)
    _atomic_json(output_dir / "manifest.json", manifest)
    script = render_reproduce_script(
        python=Path(sys.executable),
        repo_root=repo_root,
        run_root=args.run_root,
        teacher_model_path=args.teacher_model_path,
        dataset_path=args.dataset_path,
        output_dir=output_dir,
        seed=args.seed,
        r=args.r,
        s=args.s,
        teacher_steps=args.teacher_steps,
        gpu_id=args.gpu_id,
    )
    reproduce = output_dir / "reproduce.sh"
    reproduce.write_text(script, encoding="utf-8")
    reproduce.chmod(0o755)


def run(args) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    args.run_root = args.run_root.resolve()
    args.teacher_model_path = args.teacher_model_path.resolve()
    args.dataset_path = args.dataset_path.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.run_root / "outputs" / "mechanism_diagnostics"
    )
    _configure_logging(output_dir)
    discovered = discover_target_checkpoints(args.run_root)
    checkpoints = select_checkpoints(discovered, args.steps)
    if not checkpoints:
        raise RuntimeError(f"no target-student checkpoints found under {args.run_root}")
    for checkpoint in checkpoints:
        validate_factor1_checkpoint(checkpoint)
    if not (1000 > args.r > args.s > 0):
        raise ValueError("diagnostic levels must satisfy 1000 > r > s > 0")

    _install_repo_imports(repo_root)
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()

    import importlib
    from distributed.util import init_distributed

    init_distributed(1, 0, 0)
    base_config = importlib.import_module(args.config).cfg
    cfg = configure_eval_config(
        base_config,
        checkpoint=checkpoints[0],
        output_dir=output_dir,
        teacher_model_path=args.teacher_model_path,
        dataset_path=args.dataset_path,
        seed=args.seed,
        r=args.r,
        s=args.s,
        teacher_steps=args.teacher_steps,
    )
    import distillation_flowmap.flowmap_trainer as trainer_module

    trainer_module.HAS_WANDB = False
    trainer_module.HAS_TENSORBOARD = False
    trainer = trainer_module.FlowMapDistiller(cfg)
    _dispose_training_only_state(trainer)
    _dispose_student(trainer)
    batch, probe, probe_meta = _prepare_probe(
        trainer,
        seed=args.seed,
        r=args.r,
        s=args.s,
        teacher_steps=args.teacher_steps,
    )
    torch.save(
        {
            "video_noise": probe["video_noise"],
            "action_noise": probe["action_noise"],
            "dataset_index": probe_meta["dataset_index"],
            "seed": args.seed,
            "r": args.r,
            "s": args.s,
        },
        output_dir / "probe_bank.pt",
    )
    _atomic_json(output_dir / "probe_bank_metadata.json", probe_meta)

    existing = {
        int(row["step"]): row for row in _load_existing_records(output_dir)
    }
    gpu = torch.cuda.get_device_name(0)
    for checkpoint in checkpoints:
        _dispose_student(trainer)
        _load_flowmap_target_student(trainer, checkpoint)
        metrics, valid_count = _evaluate_checkpoint(trainer, batch, probe)
        existing[checkpoint.step] = paper_record(
            step=checkpoint.step,
            checkpoint=checkpoint.transformer,
            metrics=metrics,
            valid_sample_count=valid_count,
        )
        ordered = [existing[step] for step in sorted(existing)]
        _write_runtime_artifacts(
            output_dir=output_dir,
            records=ordered,
            repo_root=repo_root,
            args=args,
            checkpoints=checkpoints,
            probe_meta=probe_meta,
            dtype=str(trainer.dtype),
            gpu=gpu,
            started_at=started_at,
        )
        LOGGER.info(
            "Completed step=%d valid=%d g_anchor=%.6g g_comp=%.6g "
            "e_student=%.6g e_video=%.6g e_joint=%.6g",
            checkpoint.step,
            valid_count,
            existing[checkpoint.step]["g_anchor_l2"],
            existing[checkpoint.step]["g_comp_l2"],
            existing[checkpoint.step]["e_student"],
            existing[checkpoint.step]["e_video"],
            existing[checkpoint.step]["e_joint"],
        )
    _dispose_student(trainer)
    LOGGER.info(
        "Sweep complete: steps=%s runtime=%.1fs peak_memory=%.2fGiB output=%s",
        [item.step for item in checkpoints],
        time.perf_counter() - started_at,
        torch.cuda.max_memory_allocated() / (1024**3),
        output_dir,
    )


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
