"""
对比 student 和 teacher 的动作预测精度。

用法：
  python evaluation/libero/compare_actions.py \
      --teacher-dir visualization/teacher/ \
      --student-dir visualization/student/ \
      --output-file results/action_comparison.json

服务器启动时加 --save_root 会保存 .pt 文件（actions, latents 等）。
"""

import argparse
import glob
import json
import os

import numpy as np
import torch


def load_actions(directory, prefix="actions"):
    """加载目录下所有 actions_*.pt 文件（支持嵌套子目录）。"""
    pattern = os.path.join(directory, "**", f"{prefix}_*.pt")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No {prefix}_*.pt files found in {directory}")
    all_actions = []
    for f in files:
        data = torch.load(f, map_location="cpu")
        if isinstance(data, dict):
            data = data.get("actions", data.get("action", list(data.values())[0]))
        if isinstance(data, torch.Tensor):
            all_actions.append(data.float())
        elif isinstance(data, list):
            all_actions.append(torch.stack(data).float())
    return torch.cat(all_actions, dim=0), files


def compute_metrics(teacher_actions, student_actions):
    """计算对比指标。"""
    # 对齐长度
    min_len = min(teacher_actions.shape[0], student_actions.shape[0])
    t = teacher_actions[:min_len].float()
    s = student_actions[:min_len].float()

    # 展平到 (N, D)
    if t.dim() > 2:
        t_flat = t.reshape(min_len, -1)
        s_flat = s.reshape(min_len, -1)
    else:
        t_flat, s_flat = t, s

    # MSE
    mse = (t_flat - s_flat).pow(2).mean().item()

    # 余弦相似度（逐样本）
    t_norm = t_flat / (t_flat.norm(dim=-1, keepdim=True) + 1e-8)
    s_norm = s_flat / (s_flat.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = (t_norm * s_norm).sum(dim=-1).mean().item()

    # L1
    l1 = (t_flat - s_flat).abs().mean().item()

    # 逐维度 MSE
    per_dim_mse = (t_flat - s_flat).pow(2).mean(dim=0).tolist()

    # 最大绝对误差
    max_err = (t_flat - s_flat).abs().max().item()

    return {
        "num_samples": min_len,
        "mse": mse,
        "l1": l1,
        "cosine_similarity": cos_sim,
        "max_abs_error": max_err,
        "per_dim_mse": per_dim_mse,
        "teacher_action_shape": list(t.shape),
        "student_action_shape": list(s.shape),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare student vs teacher actions")
    parser.add_argument("--teacher-dir", required=True, help="Directory with teacher .pt files")
    parser.add_argument("--student-dir", required=True, help="Directory with student .pt files")
    parser.add_argument("--output-file", default="results/action_comparison.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    print(f"Loading teacher actions from {args.teacher_dir}")
    teacher_actions, teacher_files = load_actions(args.teacher_dir)
    print(f"  Found {len(teacher_files)} files, shape: {teacher_actions.shape}")

    print(f"Loading student actions from {args.student_dir}")
    student_actions, student_files = load_actions(args.student_dir)
    print(f"  Found {len(student_files)} files, shape: {student_actions.shape}")

    metrics = compute_metrics(teacher_actions, student_actions)

    print("\n=== Action Comparison Results ===")
    print(f"  Samples:         {metrics['num_samples']}")
    print(f"  MSE:             {metrics['mse']:.6f}")
    print(f"  L1:              {metrics['l1']:.6f}")
    print(f"  Cosine Similarity: {metrics['cosine_similarity']:.4f}")
    print(f"  Max Abs Error:   {metrics['max_abs_error']:.6f}")

    with open(args.output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
