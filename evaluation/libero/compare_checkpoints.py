"""
对比 FlowMap Distillation 不同训练步骤的checkpoint质量。

使用方法：
    python evaluation/libero/compare_checkpoints.py \
        --checkpoint-dir distillation_flowmap/output_libero_new/checkpoints \
        --steps 1000 3000 5000 7000 8500 \
        --num-samples 3 \
        --num-steps 20

评估指标：
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

from wan_va.modules.model import WanTransformer3DModel
from wan_va.configs import va_libero_cfg as config
from model_flowmap import setup_flowmap_model, patch_model_forward
from inference import flowmap_inference


def load_student_model(config, checkpoint_path, device, dtype):
    """Load a student model from checkpoint."""
    print(f"Loading student model from: {checkpoint_path}")

    # Check if it's a LoRA checkpoint or full model
    is_lora = os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))

    if is_lora:
        # Load base transformer and apply LoRA
        transformer = WanTransformer3DModel.from_pretrained(
            os.path.join(config.wan22_pretrained_model_name_or_path, "transformer")
        )
        transformer = setup_flowmap_model(transformer, deltatime_type='r', gate_value=0.1)
        from peft import PeftModel
        transformer = PeftModel.from_pretrained(transformer, checkpoint_path)
        transformer = patch_model_forward(transformer)
    else:
        # Load full model checkpoint
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(checkpoint_path, "diffusion_pytorch_model.safetensors"))

        # Check if the checkpoint has FlowMap keys
        has_flowmap = any('delta_embedder' in k for k in state_dict.keys())

        transformer = WanTransformer3DModel.from_pretrained(
            os.path.join(config.wan22_pretrained_model_name_or_path, "transformer")
        )

        if has_flowmap:
            transformer = setup_flowmap_model(transformer, deltatime_type='r', gate_value=0.1)
            transformer = patch_model_forward(transformer)

        # Load the state dict
        transformer.load_state_dict(state_dict, strict=False)

    transformer = transformer.to(device=device, dtype=dtype)
    transformer.eval()
    return transformer


def generate_with_model(model, config, device, dtype, num_steps, seed=42):
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
        denoised_latent, denoised_action = flowmap_inference(
            model=model,
            noisy_latent=latents,
            noisy_action=actions,
            text_emb=text_emb,
            empty_emb=empty_emb,
            num_steps=num_steps,
            cfg_scale=1.0,  # No CFG for comparison
        )

    return denoised_latent, denoised_action


def compute_metrics(output1_latent, output1_action, output2_latent, output2_action):
    """Compute comparison metrics between two outputs."""
    metrics = {}

    # Video latent MSE
    video_mse = ((output1_latent - output2_latent) ** 2).mean().item()
    metrics['video_latent_mse'] = video_mse

    # Video latent cosine similarity
    flat1 = output1_latent.flatten()
    flat2 = output2_latent.flatten()
    video_cosine = torch.nn.functional.cosine_similarity(
        flat1.unsqueeze(0), flat2.unsqueeze(0)
    ).item()
    metrics['video_latent_cosine'] = video_cosine

    # Action MSE
    action_mse = ((output1_action - output2_action) ** 2).mean().item()
    metrics['action_mse'] = action_mse

    # Action cosine similarity
    action_flat1 = output1_action.flatten()
    action_flat2 = output2_action.flatten()
    action_cosine = torch.nn.functional.cosine_similarity(
        action_flat1.unsqueeze(0), action_flat2.unsqueeze(0)
    ).item()
    metrics['action_cosine'] = action_cosine

    # Action L1 distance
    action_l1 = (output1_action - output2_action).abs().mean().item()
    metrics['action_l1'] = action_l1

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare checkpoints at different training steps")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--steps", type=int, nargs="+", default=[1000, 3000, 5000, 7000, 8500],
                        help="Training steps to compare")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of comparison samples")
    parser.add_argument("--num-steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--output", type=str, default="evaluation/outputs/checkpoint_comparison.json",
                        help="Output file")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16  # Use bfloat16 to match model weights

    print("=" * 60)
    print("  FlowMap Distillation Checkpoint Comparison")
    print("=" * 60)
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Steps to compare: {args.steps}")
    print(f"  Num steps (inference): {args.num_steps}")
    print(f"  Num samples: {args.num_samples}")
    print("=" * 60)

    # Find checkpoint paths
    checkpoint_paths = {}
    for step in args.steps:
        # Try different checkpoint naming conventions
        possible_paths = [
            os.path.join(args.checkpoint_dir, f"step_{step}", "online_student", "transformer"),
            os.path.join(args.checkpoint_dir, f"step_{step}", "target_student", "transformer"),
            os.path.join(args.checkpoint_dir, f"step_{step}", "transformer"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_paths[step] = path
                break

    if not checkpoint_paths:
        print("ERROR: No checkpoints found!")
        print(f"  Looking in: {args.checkpoint_dir}")
        print(f"  Steps: {args.steps}")
        return

    print(f"\nFound checkpoints:")
    for step, path in sorted(checkpoint_paths.items()):
        print(f"  step_{step}: {path}")

    # Generate outputs for each checkpoint (load one at a time to save memory)
    print("\n[1/2] Generating outputs for each checkpoint...")
    all_outputs = {}

    for step, path in checkpoint_paths.items():
        print(f"\n  Processing step_{step}...")
        try:
            # Load model
            model = load_student_model(config, path, device, dtype)

            # Generate outputs
            outputs = []
            for i in range(args.num_samples):
                seed = 42 + i
                latent, action = generate_with_model(
                    model, config, device, dtype, args.num_steps, seed
                )
                outputs.append((latent.cpu(), action.cpu()))

            all_outputs[step] = outputs

            # Free memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  WARNING: Failed to process step_{step}: {e}")

    if len(all_outputs) < 2:
        print("ERROR: Need at least 2 checkpoints to compare!")
        return

    # Compare checkpoints
    print("\n[2/2] Comparing checkpoints...")
    all_comparisons = {}

    steps = sorted(all_outputs.keys())
    for i in range(len(steps)):
        for j in range(i + 1, len(steps)):
            step1, step2 = steps[i], steps[j]
            print(f"\n  Comparing step_{step1} vs step_{step2}:")

            sample_metrics = []
            for k in range(args.num_samples):
                metrics = compute_metrics(
                    all_outputs[step1][k][0], all_outputs[step1][k][1],
                    all_outputs[step2][k][0], all_outputs[step2][k][1]
                )
                sample_metrics.append(metrics)

            # Average metrics
            avg_metrics = {}
            for key in sample_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in sample_metrics])

            comparison_key = f"step_{step1}_vs_step_{step2}"
            all_comparisons[comparison_key] = avg_metrics

            print(f"    Video MSE:     {avg_metrics['video_latent_mse']:.6f}")
            print(f"    Video Cosine:  {avg_metrics['video_latent_cosine']:.6f}")
            print(f"    Action MSE:    {avg_metrics['action_mse']:.6f}")
            print(f"    Action Cosine: {avg_metrics['action_cosine']:.6f}")
            print(f"    Action L1:     {avg_metrics['action_l1']:.6f}")

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    # Check convergence (compare consecutive steps)
    print("\n  Convergence Analysis (consecutive steps):")
    for i in range(len(steps) - 1):
        step1, step2 = steps[i], steps[i + 1]
        key = f"step_{step1}_vs_step_{step2}"
        if key in all_comparisons:
            cosine = all_comparisons[key]['action_cosine']
            if cosine > 0.95:
                status = "✓ CONVERGED"
            elif cosine > 0.8:
                status = "~ CONVERGING"
            else:
                status = "△ NOT CONVERGED"
            print(f"    step_{step1} -> step_{step2}: action_cosine={cosine:.4f} {status}")

    # Save results
    output_data = {
        'checkpoint_dir': args.checkpoint_dir,
        'steps': steps,
        'num_steps': args.num_steps,
        'num_samples': args.num_samples,
        'comparisons': all_comparisons,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
