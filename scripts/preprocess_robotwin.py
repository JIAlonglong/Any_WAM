#!/usr/bin/env python3
"""
RobotWin 数据预处理脚本

将原始 RobotWin 视频数据编码为 Flash-WAM 训练所需的 latent 格式。

原始数据结构：
    {data_root}/
    ├── adjust_bottle/
    │   ├── aloha-agilex_clean_50/
    │   │   ├── videos/0.mp4, 1.mp4, ...     # 视频文件
    │   │   ├── metas/0.txt, 1.txt, ...      # 任务描述文本（每行一个描述）
    │   │   └── qpos/0.pt, 1.pt, ...         # 机器人关节位置 (actions)
    │   └── aloha-agilex_randomized_500/
    │       └── ...
    └── ... (约 50 个任务)

目标数据格式（LatentLeRobotDataset 期望）：
    {output_dir}/
    ├── meta/
    │   └── info.json                         # LeRobot 数据集元信息
    ├── latents/
    │   └── chunk-000/
    │       └── observation.images.cam_high/
    │           └── episode_{ep_id:06d}_{start}_{end}.pth
    │               包含: {
    │                   'latent': tensor,           # VAE 编码后的 latent
    │                   'latent_num_frames': int,    # 帧数
    │                   'latent_height': int,        # 高度
    │                   'latent_width': int,         # 宽度
    │                   'text_emb': tensor,          # 文本嵌入
    │                   'frame_ids': tensor,         # 帧 ID 映射
    │               }
    └── data/
        └── chunk-000/
            └── episode_{ep_id:06d}.parquet  # 动作数据

用法：
    python scripts/preprocess_robotwin.py \
        --data-root /root/intern/jialongliu/projects/Motus/data/robotwin2/robotwin_dataset \
        --checkpoint-dir checkpoints/base \
        --output-dir training_data/lerobot_robotwin \
        --camera-key observation.images.cam_high \
        --num-workers 4
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# ============================================================
# 视频解码工具
# ============================================================

def decode_video_frames(video_path):
    """
    使用 cv2 解码视频文件为 numpy 帧数组（比 ffmpeg subprocess 快 3-5 倍）。

    参数:
        video_path: 视频文件路径

    返回:
        frames: numpy 数组，形状 (T, H, W, 3)，dtype=uint8，RGB 格式
        fps: 视频帧率
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # cv2 读取的是 BGR，转换为 RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"视频无帧: {video_path}")

    return np.stack(frames), fps


# ============================================================
# qpos（动作数据）加载工具
# ============================================================

def load_qpos(qpos_path):
    """
    加载 qpos 文件中的机器人关节位置数据。

    参数:
        qpos_path: qpos .pt 文件路径

    返回:
        qpos: numpy 数组，形状 (T, action_dim)

    实现原理：
        - qpos 文件是 torch.save 保存的 zip 格式
        - 内部包含 data/0 文件，存储 float32 张量的原始数据
        - 直接从 zip 中读取 bytes 并解析为 float32 数组
    """
    with zipfile.ZipFile(qpos_path, "r") as z:
        # 读取原始二进制数据
        data_bytes = z.read("0/data/0")

        # 解析为 float32 数组
        num_values = len(data_bytes) // 4
        values = struct.unpack(f"<{num_values}f", data_bytes)

        # 根据数据量推断形状：action_dim = 14（双臂各 7 自由度）
        action_dim = 14
        num_frames = num_values // action_dim
        qpos = np.array(values[:num_frames * action_dim], dtype=np.float32)
        qpos = qpos.reshape(num_frames, action_dim)

    return qpos


# ============================================================
# meta 文本加载工具
# ============================================================

def load_meta_text(meta_path):
    """
    加载 meta 文本文件中的任务描述。

    参数:
        meta_path: meta .txt 文件路径

    返回:
        task_descriptions: 字符串列表，每个元素是一行任务描述

    实现原理：
        - meta 文件每行包含一个任务描述
        - 通常一个 episode 的 meta 文件包含多行（不同描述变体）
        - 我们取第一行作为该 episode 的任务描述
    """
    with open(meta_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"meta 文件为空: {meta_path}")

    return lines


# ============================================================
# VAE 编码工具
# ============================================================

def encode_video_to_latent(vae, frames, device):
    """
    使用 VAE 将视频帧编码为 latent 表示。

    参数:
        vae: AutoencoderKLWan 模型
        frames: numpy 数组，形状 (T, H, W, 3)，uint8，RGB
        device: torch 设备

    返回:
        latent: torch 张量，形状 (C*f*h*w,)，已展平
        latent_num_frames: latent 时间维度大小
        latent_height: latent 高度维度大小
        latent_width: latent 宽度维度大小

    实现原理：
        - 将 uint8 帧归一化到 [-1, 1]
        - 转换为 (B, C, T, H, W) 格式输入 VAE
        - 使用 bf16 编码以节省显存
        - VAE 输出 latent 的形状为 (B, C, f, h, w)
    """
    num_frames = frames.shape[0]
    height = frames.shape[1]
    width = frames.shape[2]

    # 归一化到 [-1, 1]：(x / 255.0) * 2 - 1
    video_tensor = torch.from_numpy(frames).float() / 255.0 * 2.0 - 1.0

    # 转换为 (B, C, T, H, W) 格式
    video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
    video_tensor = video_tensor.unsqueeze(0)  # (B, T, C, H, W)
    video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

    # 使用 bf16 编码以节省显存
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # VAE 编码
        latent_dist = vae.encode(video_tensor.to(device))
        latent = latent_dist.latent_dist.sample()
        latent = latent.float()  # 转回 float32 存储

    # 记录 latent 的空间维度
    # latent 形状: (B, C, f, h, w)
    latent_num_frames = latent.shape[2]
    latent_height = latent.shape[3]
    latent_width = latent.shape[4]

    # 展平为 (C*f*h*w,) 以便存储
    latent = latent.squeeze(0).flatten()

    return latent, latent_num_frames, latent_height, latent_width


# ============================================================
# 文本编码工具
# ============================================================

def encode_text(text_encoder, tokenizer, text, device, max_length=512):
    """
    使用 Text Encoder 将文本编码为嵌入向量。

    参数:
        text_encoder: UMT5EncoderModel 模型
        tokenizer: T5TokenizerFast 分词器
        text: 输入文本字符串
        device: torch 设备
        max_length: 最大 token 长度

    返回:
        text_emb: torch 张量，文本嵌入

    实现原理：
        - 使用 T5TokenizerFast 将文本分词
        - 使用 UMT5EncoderModel 编码为嵌入
        - 使用 bf16 编码以节省显存
    """
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # 分词
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # 编码
        outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = outputs.last_hidden_state.float()  # 转回 float32 存储

    return text_emb.cpu()


# ============================================================
# 数据扫描与 episode 枚举
# ============================================================

def scan_dataset(data_root, max_episodes=None):
    """
    扫描原始数据目录，枚举所有 episode。

    参数:
        data_root: 原始数据根目录
        max_episodes: 每个任务子目录的最大 episode 数（None = 全部）

    返回:
        episodes: 列表，每个元素是一个字典：
            {
                'task_name': str,          # 任务名（如 adjust_bottle）
                'subset_name': str,        # 子集名（如 aloha-agilex_clean_50）
                'episode_id': int,         # episode 编号（全局唯一）
                'video_path': Path,        # 视频文件路径
                'meta_path': Path,         # meta 文本路径
                'qpos_path': Path,         # qpos 文件路径
                'task_description': str,   # 任务描述文本
            }

    实现原理：
        - 遍历 data_root 下的所有任务目录
        - 每个任务目录下有 aloha-agilex_clean_50 和 aloha-agilex_randomized_500 子目录
        - 每个子目录下有 videos/, metas/, qpos/ 三个文件夹
        - 通过匹配文件名（数字编号）来关联 video、meta、qpos
    """
    data_root = Path(data_root)
    episodes = []
    global_episode_id = 0

    # 遍历所有任务目录
    task_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and d.name not in ("dataset", "dump_bin_bigbin", "README.md")
    ])

    for task_dir in tqdm(task_dirs, desc="扫描任务目录"):
        task_name = task_dir.name

        # 遍历每个任务下的子集目录（clean_50, randomized_500）
        subset_dirs = sorted([
            d for d in task_dir.iterdir()
            if d.is_dir()
        ])

        for subset_dir in subset_dirs:
            subset_name = subset_dir.name
            video_dir = subset_dir / "videos"
            meta_dir = subset_dir / "metas"
            qpos_dir = subset_dir / "qpos"

            # 检查必要的目录是否存在
            if not video_dir.exists() or not meta_dir.exists() or not qpos_dir.exists():
                print(f"警告: 跳过不完整的子集 {subset_dir}（缺少 videos/metas/qpos 目录）")
                continue

            # 获取所有视频文件并按编号排序
            video_files = sorted(
                video_dir.glob("*.mp4"),
                key=lambda x: int(x.stem),
            )

            # 限制 episode 数量
            if max_episodes is not None:
                video_files = video_files[:max_episodes]

            for video_path in video_files:
                ep_idx = int(video_path.stem)
                meta_path = meta_dir / f"{ep_idx}.txt"
                qpos_path = qpos_dir / f"{ep_idx}.pt"

                # 检查配套文件是否存在
                if not meta_path.exists():
                    print(f"警告: 跳过 {video_path}（缺少 meta 文件 {meta_path}）")
                    continue
                if not qpos_path.exists():
                    print(f"警告: 跳过 {video_path}（缺少 qpos 文件 {qpos_path}）")
                    continue

                # 加载任务描述（取第一行）
                try:
                    task_descriptions = load_meta_text(meta_path)
                    task_description = task_descriptions[0]
                except Exception as e:
                    print(f"警告: 跳过 {video_path}（读取 meta 失败: {e}）")
                    continue

                episodes.append({
                    "task_name": task_name,
                    "subset_name": subset_name,
                    "episode_id": global_episode_id,
                    "video_path": video_path,
                    "meta_path": meta_path,
                    "qpos_path": qpos_path,
                    "task_description": task_description,
                })
                global_episode_id += 1

    return episodes


# ============================================================
# 单个 episode 处理
# ============================================================

def process_episode(
    episode_info,
    vae,
    text_encoder,
    tokenizer,
    output_dir,
    camera_key,
    device,
):
    """
    处理单个 episode：解码视频、VAE 编码、文本编码、保存数据。

    参数:
        episode_info: episode 信息字典（来自 scan_dataset）
        vae: VAE 模型
        text_encoder: Text Encoder 模型
        tokenizer: 分词器
        output_dir: 输出目录
        camera_key: 相机视角键名
        device: torch 设备

    返回:
        episode_meta: episode 元信息字典（用于生成 info.json 和 parquet）
        成功返回 None 表示跳过

    实现原理：
        1. 解码视频帧
        2. VAE 编码为 latent
        3. Text Encoder 编码任务描述
        4. 加载 qpos 作为 action
        5. 保存 .pth 文件（latent）和 .parquet 文件（action）
    """
    ep_id = episode_info["episode_id"]
    video_path = episode_info["video_path"]
    qpos_path = episode_info["qpos_path"]
    task_description = episode_info["task_description"]

    # ---- 步骤 1: 解码视频帧 ----
    try:
        frames, fps = decode_video_frames(video_path)
    except Exception as e:
        print(f"警告: 跳过 episode {ep_id}（视频解码失败: {e}）")
        return None

    num_frames = frames.shape[0]

    # ---- 步骤 1.5: 如果分辨率不是 480x640，使用 cv2 resize 到目标分辨率 ----
    target_h, target_w = 480, 640
    if frames.shape[1] != target_h or frames.shape[2] != target_w:
        import cv2
        resized_frames = []
        for i in range(num_frames):
            resized = cv2.resize(frames[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized)
        frames = np.stack(resized_frames)  # (T, 480, 640, 3)

    # ---- 步骤 2: VAE 编码 ----
    try:
        latent, latent_num_frames, latent_height, latent_width = encode_video_to_latent(
            vae, frames, device
        )
    except Exception as e:
        print(f"警告: 跳过 episode {ep_id}（VAE 编码失败: {e}）")
        return None

    # ---- 步骤 3: Text Encoder 编码 ----
    try:
        text_emb = encode_text(text_encoder, tokenizer, task_description, device)
    except Exception as e:
        print(f"警告: 跳过 episode {ep_id}（文本编码失败: {e}）")
        return None

    # ---- 步骤 4: 加载 qpos（动作数据） ----
    try:
        qpos = load_qpos(qpos_path)
    except Exception as e:
        print(f"警告: 跳过 episode {ep_id}（加载 qpos 失败: {e}）")
        return None

    # ---- 步骤 5: 构建 frame_ids ----
    # frame_ids 映射 latent 帧到原始视频帧
    # latent 的时间维度经过 VAE 的时间下采样，每个 latent 帧对应多个原始帧
    frame_ids = torch.arange(num_frames)

    # ---- 步骤 6: 保存 latent .pth 文件 ----
    # 文件名格式: episode_{ep_id:06d}_{start_frame}_{end_frame}.pth
    # 对于整个 episode: start=0, end=num_frames
    start_frame = 0
    end_frame = num_frames

    latent_dir = output_dir / "latents" / "chunk-000" / camera_key
    latent_dir.mkdir(parents=True, exist_ok=True)
    latent_file = latent_dir / f"episode_{ep_id:06d}_{start_frame}_{end_frame}.pth"

    latent_data = {
        "latent": latent,
        "latent_num_frames": latent_num_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "text_emb": text_emb,
        "frame_ids": frame_ids,
    }
    torch.save(latent_data, latent_file)

    # ---- 步骤 7: 保存 action .parquet 文件 ----
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        data_dir = output_dir / "data" / "chunk-000"
        data_dir.mkdir(parents=True, exist_ok=True)
        parquet_file = data_dir / f"episode_{ep_id:06d}.parquet"

        # 将 qpos 转换为 pyarrow Table
        # qpos 形状: (T, 14)，列名: action_0, action_1, ..., action_13
        num_action_dims = qpos.shape[1]
        columns = {f"action_{i}": qpos[:, i] for i in range(num_action_dims)}
        columns["episode_index"] = np.full(num_frames, ep_id, dtype=np.int64)
        columns["frame_index"] = np.arange(num_frames, dtype=np.int64)
        columns["task_index"] = np.zeros(num_frames, dtype=np.int64)
        columns["task"] = [task_description] * num_frames

        table = pa.table(columns)
        pq.write_table(table, parquet_file)
    except ImportError:
        # 如果 pyarrow 不可用，使用 pandas 保存
        try:
            import pandas as pd

            data_dir = output_dir / "data" / "chunk-000"
            data_dir.mkdir(parents=True, exist_ok=True)
            parquet_file = data_dir / f"episode_{ep_id:06d}.parquet"

            df_data = {f"action_{i}": qpos[:, i] for i in range(qpos.shape[1])}
            df_data["episode_index"] = np.full(num_frames, ep_id, dtype=np.int64)
            df_data["frame_index"] = np.arange(num_frames, dtype=np.int64)
            df_data["task_index"] = np.zeros(num_frames, dtype=np.int64)
            df_data["task"] = [task_description] * num_frames

            df = pd.DataFrame(df_data)
            df.to_parquet(parquet_file, index=False)
        except Exception as e:
            print(f"警告: 跳过 episode {ep_id}（保存 parquet 失败: {e}）")
            return None
    except Exception as e:
        print(f"警告: 跳过 episode {ep_id}（保存 parquet 失败: {e}）")
        return None

    # 返回 episode 元信息
    return {
        "episode_index": ep_id,
        "tasks": task_description,
        "num_frames": num_frames,
        "fps": fps,
        "task_name": episode_info["task_name"],
        "subset_name": episode_info["subset_name"],
        "video_path": str(video_path),
        "latent_file": str(latent_file),
        "parquet_file": str(parquet_file),
        "action_config": [
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
            }
        ],
    }


# ============================================================
# 生成 info.json
# ============================================================

def generate_info_json(output_dir, episodes_meta, camera_key):
    """
    生成 LeRobot 数据集的 meta/info.json 文件。

    参数:
        output_dir: 输出目录
        episodes_meta: episode 元信息列表
        camera_key: 相机视角键名

    实现原理：
        - 按照 LeRobot v2.1 格式生成 info.json
        - 包含数据集的基本信息、特征定义、episode 列表等
    """
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 统计总帧数
    total_frames = sum(ep["num_frames"] for ep in episodes_meta)

    # 构建 episodes 字典
    episodes_dict = {}
    for ep in episodes_meta:
        ep_idx = ep["episode_index"]
        episodes_dict[str(ep_idx)] = {
            "episode_index": ep_idx,
            "tasks": ep["tasks"],
            "action_config": ep["action_config"],
        }

    # 构建 info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "aloha",
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "total_tasks": len(episodes_meta),
        "total_videos": len(episodes_meta),
        "total_chunks": 1,
        "chunks_size": len(episodes_meta),
        "fps": 30,
        "splits": {
            "train": f"0:{len(episodes_meta)}",
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [14],
                "names": None,
            },
            "observation.images.cam_high": {
                "dtype": "video",
                "shape": [360, 320, 3],
                "names": ["height", "width, channels"],
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "task": {
                "dtype": "string",
                "shape": [1],
                "names": None,
            },
        },
        "episodes": episodes_dict,
    }

    info_path = meta_dir / "info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"已生成 info.json: {info_path}")


# ============================================================
# 生成 empty_emb.pt（空文本嵌入）
# ============================================================

def generate_empty_emb(text_encoder, tokenizer, output_dir, device):
    """
    生成空文本嵌入文件（用于 CFG 无条件推理）。

    参数:
        text_encoder: Text Encoder 模型
        tokenizer: 分词器
        output_dir: 输出目录
        device: torch 设备

    实现原理：
        - 编码空字符串 "" 得到空文本嵌入
        - 训练时以一定概率用空嵌入替换真实嵌入（CFG dropout）
        - 推理时用空嵌入作为无条件输入
    """
    empty_emb = encode_text(text_encoder, tokenizer, "", device)
    empty_emb_path = output_dir / "empty_emb.pt"
    torch.save(empty_emb, empty_emb_path)
    print(f"已生成 empty_emb.pt: {empty_emb_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="将 RobotWin 视频数据预处理为 Flash-WAM 训练所需的 latent 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/root/intern/jialongliu/projects/Motus/data/robotwin2/robotwin_dataset",
        help="原始数据根目录",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/base",
        help="模型检查点目录（包含 vae/, text_encoder/, tokenizer/ 子目录）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_data/lerobot_robotwin",
        help="输出目录",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default="observation.images.cam_high",
        help="相机视角键名",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="并行 worker 数（当前为串行处理，此参数预留）",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="每个任务子目录的最大 episode 数（默认全部处理）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="GPU 设备",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过已存在的 latent 文件",
    )

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 设置 GPU
    if device.type == "cuda":
        torch.cuda.set_device(device)

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    data_root = Path(args.data_root)

    # 检查路径
    if not data_root.exists():
        print(f"错误: 数据根目录不存在: {data_root}")
        return
    if not checkpoint_dir.exists():
        print(f"错误: 检查点目录不存在: {checkpoint_dir}")
        return

    # ---- 加载模型 ----
    print("=" * 60)
    print("加载模型...")

    # 安装 flash_attn 存根（防止导入失败）
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from distillation.patches import install_flash_attn_stub
    install_flash_attn_stub()

    # 加载 VAE
    print("加载 VAE...")
    from wan_va.modules.utils import load_vae, load_text_encoder, load_tokenizer
    vae = load_vae(
        vae_path=str(checkpoint_dir / "vae"),
        torch_dtype=torch.bfloat16,
        torch_device=device,
    )
    vae.eval()
    print("VAE 加载完成")

    # 加载 Text Encoder
    print("加载 Text Encoder...")
    text_encoder = load_text_encoder(
        text_encoder_path=str(checkpoint_dir / "text_encoder"),
        torch_dtype=torch.bfloat16,
        torch_device=device,
    )
    text_encoder.eval()
    print("Text Encoder 加载完成")

    # 加载 Tokenizer
    print("加载 Tokenizer...")
    tokenizer = load_tokenizer(
        tokenizer_path=str(checkpoint_dir / "tokenizer"),
    )
    print("Tokenizer 加载完成")

    # ---- 扫描数据集 ----
    print("=" * 60)
    print("扫描数据集...")
    episodes = scan_dataset(data_root, max_episodes=args.max_episodes)
    print(f"共发现 {len(episodes)} 个 episode")

    if len(episodes) == 0:
        print("错误: 未发现任何 episode")
        return

    # ---- 创建输出目录 ----
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(exist_ok=True)
    (output_dir / "latents" / "chunk-000" / args.camera_key).mkdir(parents=True, exist_ok=True)
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # ---- 生成 empty_emb.pt ----
    print("=" * 60)
    print("生成 empty_emb.pt...")
    generate_empty_emb(text_encoder, tokenizer, output_dir, device)

    # ---- 处理所有 episode ----
    print("=" * 60)
    print("开始处理 episode...")
    episodes_meta = []
    skipped = 0

    for episode_info in tqdm(episodes, desc="处理 episode"):
        ep_id = episode_info["episode_id"]

        # 检查是否跳过已存在的文件
        if args.skip_existing:
            latent_dir = output_dir / "latents" / "chunk-000" / args.camera_key
            # 检查是否有以 episode_id 开头的文件
            existing_files = list(latent_dir.glob(f"episode_{ep_id:06d}_*.pth"))
            if existing_files:
                skipped += 1
                continue

        # 设置当前 episode 的 GPU
        with torch.cuda.device(device):
            ep_meta = process_episode(
                episode_info=episode_info,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                output_dir=output_dir,
                camera_key=args.camera_key,
                device=device,
            )

        if ep_meta is not None:
            episodes_meta.append(ep_meta)

    print(f"\n处理完成: 成功 {len(episodes_meta)} 个, 跳过 {skipped} 个")

    # ---- 生成 info.json ----
    print("=" * 60)
    print("生成 info.json...")
    generate_info_json(output_dir, episodes_meta, args.camera_key)

    # ---- 完成 ----
    print("=" * 60)
    print("预处理完成！")
    print(f"输出目录: {output_dir}")
    print(f"  - meta/info.json: 数据集元信息")
    print(f"  - latents/chunk-000/{args.camera_key}/: latent 文件")
    print(f"  - data/chunk-000/: action parquet 文件")
    print(f"  - empty_emb.pt: 空文本嵌入")
    print(f"共处理 {len(episodes_meta)} 个 episode")


if __name__ == "__main__":
    main()
