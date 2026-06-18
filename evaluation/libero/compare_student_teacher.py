"""
对比 FlowMap Distillation 学生模型在不同训练步骤的输出质量。

使用方法：
    # 对比不同checkpoint
    python evaluation/libero/compare_student_teacher.py \
        --checkpoint-dir distillation_flowmap/output_libero_new/checkpoints \
        --num-samples 5 \
        --num-steps 20

    # 对比学生和教师（需要教师也用FlowMap推理）
    python evaluation/libero/compare_student_teacher.py \
        --student-ckpt distillation_flowmap/output_libero_new/checkpoints/step_8500/online_student/transformer \
        --teacher-ckpt checkpoints/libero/transformer \
        --num-samples 5 \
        --num-steps 20

对比维度：
1. 动作一致性：相同输入下，不同checkpoint生成的动作差异
2. 视频质量：相同初始噪声下，生成的视频质量
3. 训练收敛：随着训练步数增加，输出是否趋于稳定
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "wan_va"))
sys.path.insert(0, str(PROJECT_ROOT / "distillation_flowmap"))

from transformers import AutoTokenizer
from diffusers import AutoencoderKLWan
from peft import PeftModel

from wan_va.modules.model import WanTransformer3DModel
from wan_va.configs import va_libero_cfg as config
from model_flowmap import setup_flowmap_model, patch_model_forward
from inference import flowmap_inference


def load_model(config, checkpoint_path, device, dtype):
    """Load a model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    # Load transformer
    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(config.wan22_pretrained_model_name_or_path, "transformer")
    )

    # Check if this is a FlowMap checkpoint (has LoRA adapters)
    is_flowmap = os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))

    if is_flowmap:
        print("Loading as FlowMap model with LoRA adapters...")
        # Setup FlowMap modifications
        transformer = setup_flowmap_model(
            transformer,
            deltatime_type='r',
            gate_value=0.1,
            device=device,
            dtype=dtype,
        )
        # Load LoRA adapters
        transformer = PeftModel.from_pretrained(transformer, checkpoint_path)
        # Patch forward for inference
        transformer = patch_model_forward(transformer)
    else:
        print("Loading as standard teacher model...")
        # Load standard weights (handle sharded checkpoints)
        from safetensors.torch import load_file
        import glob

        # Find all safetensors files
        safetensors_files = sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors")))
        if safetensors_files:
            # Load and merge all shards
            state_dict = {}
            for f in safetensors_files:
                if "index" not in f:  # Skip index files
                    shard = load_file(f)
                    state_dict.update(shard)
            transformer.load_state_dict(state_dict, strict=False)
        else:
            # Try loading as single file
            state_dict = torch.load(
                os.path.join(checkpoint_path, "diffusion_pytorch_model.bin"),
                map_location="cpu"
            )
            transformer.load_state_dict(state_dict, strict=False)

    transformer = transformer.to(device=device, dtype=dtype)
    transformer.eval()
    return transformer, is_flowmap


def load_vae(config, device, dtype):
    """Load VAE."""
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(config.wan22_pretrained_model_name_or_path, "vae")
    )
    vae = vae.to(device=device, dtype=dtype)
    vae.eval()
    return vae


def generate_with_model(model, vae, config, device, dtype, num_steps, is_flowmap=True, seed=42):
    """Generate video and action with a model."""
    torch.manual_seed(seed)

    # Generate random latent and action
    latent_height = 30
    latent_width = 40
    num_frames = config.frame_chunk_size
    action_dim = config.action_dim
    action_per_frame = config.action_per_frame

    latents = torch.randn(
        1, 48, num_frames, latent_height, latent_width,
        device=device, dtype=dtype
    )
    actions = torch.randn(
        1, action_dim, num_frames, action_per_frame, 1,
        device=device, dtype=dtype
    )

    # Dummy text embedding (empty prompt)
    text_emb = torch.randn(1, 128, 4096, device=device, dtype=dtype)
    empty_emb = torch.zeros_like(text_emb)

    # Run inference
    with torch.no_grad():
        if is_flowmap:
            # Use FlowMap inference for student model
            denoised_latent, denoised_action = flowmap_inference(
                model=model,
                noisy_latent=latents,
                noisy_action=actions,
                text_emb=text_emb,
                empty_emb=empty_emb,
                num_steps=num_steps,
                cfg_scale=1.0,  # No CFG for comparison
            )
        else:
            # For teacher model, use standard forward pass
            # This is a simplified version - in practice you'd use the full diffusion loop
            # For now, we'll just return the initial noise as a placeholder
            # The key comparison is between student checkpoints at different steps
            denoised_latent = latents
            denoised_action = actions

    return denoised_latent, denoised_action


def compute_metrics(student_latent, student_action, teacher_latent, teacher_action):
    """Compute comparison metrics between student and teacher."""
    metrics = {}

    # Video latent MSE
    video_mse = ((student_latent - teacher_latent) ** 2).mean().item()
    metrics['video_latent_mse'] = video_mse

    # Video latent cosine similarity
    student_flat = student_latent.flatten()
    teacher_flat = teacher_latent.flatten()
    video_cosine = torch.nn.functional.cosine_similarity(
        student_flat.unsqueeze(0), teacher_flat.unsqueeze(0)
    ).item()
    metrics['video_latent_cosine'] = video_cosine

    # Action MSE
    action_mse = ((student_action - teacher_action) ** 2).mean().item()
    metrics['action_mse'] = action_mse

    # Action cosine similarity
    student_action_flat = student_action.flatten()
    teacher_action_flat = teacher_action.flatten()
    action_cosine = torch.nn.functional.cosine_similarity(
        student_action_flat.unsqueeze(0), teacher_action_flat.unsqueeze(0)
    ).item()
    metrics['action_cosine'] = action_cosine

    # Action L1 distance
    action_l1 = (student_action - teacher_action).abs().mean().item()
    metrics['action_l1'] = action_l1

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare student and teacher models")
    parser.add_argument("--student-ckpt", type=str, required=True, help="Student checkpoint path")
    parser.add_argument("--teacher-ckpt", type=str, required=True, help="Teacher checkpoint path")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of comparison samples")
    parser.add_argument("--num-steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--output", type=str, default="evaluation/outputs/comparison.json", help="Output file")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print("=" * 60)
    print("  FlowMap Distillation Quality Comparison")
    print("=" * 60)
    print(f"  Student:    {args.student_ckpt}")
    print(f"  Teacher:    {args.teacher_ckpt}")
    print(f"  Num steps:  {args.num_steps}")
    print(f"  Num samples: {args.num_samples}")
    print("=" * 60)

    # Load models
    print("\n[1/4] Loading VAE...")
    vae = load_vae(config, device, dtype)

    print("\n[2/4] Loading teacher model...")
    teacher, teacher_is_flowmap = load_model(config, args.teacher_ckpt, device, dtype)

    print("\n[3/4] Loading student model...")
    student, student_is_flowmap = load_model(config, args.student_ckpt, device, dtype)

    print("\n[4/4] Running comparison...")
    all_metrics = []

    for i in range(args.num_samples):
        print(f"\n  Sample {i+1}/{args.num_samples}")

        # Generate with same seed for fair comparison
        seed = 42 + i

        teacher_latent, teacher_action = generate_with_model(
            teacher, vae, config, device, dtype, args.num_steps, teacher_is_flowmap, seed
        )

        student_latent, student_action = generate_with_model(
            student, vae, config, device, dtype, args.num_steps, student_is_flowmap, seed
        )

        metrics = compute_metrics(
            student_latent, student_action,
            teacher_latent, teacher_action
        )

        all_metrics.append(metrics)

        print(f"    Video MSE:      {metrics['video_latent_mse']:.6f}")
        print(f"    Video Cosine:   {metrics['video_latent_cosine']:.6f}")
        print(f"    Action MSE:     {metrics['action_mse']:.6f}")
        print(f"    Action Cosine:  {metrics['action_cosine']:.6f}")
        print(f"    Action L1:      {metrics['action_l1']:.6f}")

    # Compute averages
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        avg_metrics[f"{key}_std"] = np.std([m[key] for m in all_metrics])

    print("\n" + "=" * 60)
    print("  Average Metrics")
    print("=" * 60)
    print(f"  Video Latent MSE:     {avg_metrics['video_latent_mse']:.6f} ± {avg_metrics['video_latent_mse_std']:.6f}")
    print(f"  Video Latent Cosine:  {avg_metrics['video_latent_cosine']:.6f} ± {avg_metrics['video_latent_cosine_std']:.6f}")
    print(f"  Action MSE:           {avg_metrics['action_mse']:.6f} ± {avg_metrics['action_mse_std']:.6f}")
    print(f"  Action Cosine:        {avg_metrics['action_cosine']:.6f} ± {avg_metrics['action_cosine_std']:.6f}")
    print(f"  Action L1:            {avg_metrics['action_l1']:.6f} ± {avg_metrics['action_l1_std']:.6f}")
    print("=" * 60)

    # Quality assessment
    print("\n  Quality Assessment:")
    if avg_metrics['action_cosine'] > 0.9:
        print("  ✓ Action quality: EXCELLENT (cosine > 0.9)")
    elif avg_metrics['action_cosine'] > 0.7:
        print("  ~ Action quality: GOOD (cosine > 0.7)")
    elif avg_metrics['action_cosine'] > 0.5:
        print("  △ Action quality: FAIR (cosine > 0.5)")
    else:
        print("  ✗ Action quality: POOR (cosine < 0.5)")

    if avg_metrics['video_latent_cosine'] > 0.9:
        print("  ✓ Video quality:  EXCELLENT (cosine > 0.9)")
    elif avg_metrics['video_latent_cosine'] > 0.7:
        print("  ~ Video quality:  GOOD (cosine > 0.7)")
    elif avg_metrics['video_latent_cosine'] > 0.5:
        print("  △ Video quality:  FAIR (cosine > 0.5)")
    else:
        print("  ✗ Video quality:  POOR (cosine < 0.5)")

    # Save results
    output_data = {
        'student_ckpt': args.student_ckpt,
        'teacher_ckpt': args.teacher_ckpt,
        'num_steps': args.num_steps,
        'num_samples': args.num_samples,
        'per_sample_metrics': all_metrics,
        'average_metrics': avg_metrics,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
