#!/usr/bin/env python3
"""Generate multiple videos with different seeds for Stage 2 (FlowMap student) and Base (teacher).

The underlying VideoGenerator.generate() deletes self.transformer and self.text_encoder
after each run, so we recreate a fresh VideoGenerator instance per seed.

Usage:
  # Stage 2 (student, 8 steps)
  CUDA_VISIBLE_DEVICES=3 python generate_multi_videos.py \
      --mode student \
      --checkpoint-path /root/nas/junjie/jj/Any_WAM/distillation_flowmap/output_libero_onpolicy/checkpoints/step_1700/online_student/transformer \
      --num-steps 8 --num-frames 32 --cfg-scale 2.0 \
      --num-samples 6 --start-seed 42 \
      --output-dir ./train_out/multi_videos/stage2

  # Base (teacher, 20 steps)
  CUDA_VISIBLE_DEVICES=3 python generate_multi_videos.py \
      --mode teacher \
      --base-model-path /root/nas/junjie/jj/Any_WAM/checkpoints/lingbot-va-posttrain-libero \
      --num-steps 20 --num-frames 32 --cfg-scale 2.0 \
      --num-samples 6 --start-seed 100 \
      --output-dir ./train_out/multi_videos/base
"""
import argparse
import os
import socket
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "wan_va"))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "distillation_flowmap"))

import numpy as np
import torch

# --- Script-local imports (require sys.path setup above) ---
from generate_video_lingbot import VideoGenerator, build_libero_config
from distributed.util import init_distributed
from utils import init_logger, logger


def _init_distributed():
    """Initialize single-GPU distributed once per process."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    port = int(os.environ["MASTER_PORT"])
    for offset in range(100):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("localhost", port + offset))
        sock.close()
        if result != 0:
            port = port + offset
            break
    os.environ["MASTER_PORT"] = str(port)

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    init_logger()


def generate_multi(
    mode,
    base_model_path,
    checkpoint_path,
    example_dir,
    num_steps,
    num_chunks,
    output_dir,
    num_samples,
    start_seed,
):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_samples):
        seed = start_seed + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        output_path = os.path.join(output_dir, f"{mode}_seed{seed:03d}.mp4")

        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{num_samples}] Generating {mode} seed={seed} -> {output_path}")
        logger.info(f"{'='*60}")

        config = build_libero_config(
            base_model_path=base_model_path,
            example_dir=example_dir,
            num_chunks=num_chunks,
        )
        config.save_root = os.path.dirname(output_path) or "."

        if mode == "teacher":
            config.num_inference_steps = num_steps

        gen = VideoGenerator(
            config=config,
            mode=mode,
            checkpoint_path=checkpoint_path if mode == "student" else None,
            num_steps=num_steps,
        )
        gen.generate(output_path=output_path)

        # Free references so GC can release CUDA memory before next load
        del gen

        fsize = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        logger.info(f"  Saved: {output_path} ({fsize / 1024:.0f} KB)")

    logger.info(f"\nAll {num_samples} videos saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-sample video generation for Stage 2 / Base"
    )
    parser.add_argument(
        "--mode", choices=["teacher", "student"], default="student",
        help="Teacher: 20-step | Student: 8-step FlowMap+LoRA",
    )
    parser.add_argument(
        "--base-model-path", type=str,
        default="/root/nas/junjie/jj/Any_WAM/checkpoints/lingbot-va-posttrain-libero",
    )
    parser.add_argument(
        "--checkpoint-path", type=str,
        default="/root/nas/junjie/jj/Any_WAM/distillation_flowmap/output_libero_onpolicy/checkpoints/step_1700/online_student/transformer",
    )
    parser.add_argument(
        "--example-dir", type=str,
        default="/root/nas/junjie/jj/Any_WAM/lingbot-va/example/libero",
    )
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=str, default="./train_out/multi_videos",
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=2.0,
        help="CFG guidance scale (default: 2.0, lower = less distortion).",
    )
    args = parser.parse_args()

    if args.num_steps is None:
        args.num_steps = 8 if args.mode == "student" else 20

    _init_distributed()

    generate_multi(
        mode=args.mode,
        base_model_path=args.base_model_path,
        checkpoint_path=args.checkpoint_path,
        example_dir=args.example_dir,
        num_steps=args.num_steps,
        num_chunks=args.num_chunks,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        start_seed=args.start_seed,
    )


if __name__ == "__main__":
    main()
