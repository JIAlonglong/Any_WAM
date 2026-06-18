#!/usr/bin/env python3
"""
FlowMap 蒸馏模型视频生成脚本。

直接使用 VA_Server 进行推理，无需 websocket 服务器。
生成 LIBERO 任务的视频并保存到指定目录。

用法：
  conda run -n flashwam python evaluation/libero/generate_videos_flowmap.py \
      --checkpoint-path distillation_flowmap/output_libero_new/checkpoints/step_8500/online_student/transformer \
      --num-steps 2 \
      --output-dir evaluation/outputs/flowmap_videos \
      --num-tasks 3 \
      --num-episodes 2
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# 添加路径（使用绝对路径确保正确性）
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
wan_va_path = os.path.join(_project_root, "wan_va")
distillation_path = os.path.join(_project_root, "distillation_flowmap")

if wan_va_path not in sys.path:
    sys.path.insert(0, wan_va_path)
if distillation_path not in sys.path:
    sys.path.insert(0, distillation_path)

from wan_va.configs import VA_CONFIGS
from wan_va.distributed.fsdp import shard_model
from wan_va.distributed.util import _configure_model, init_distributed
from wan_va.modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from wan_va.utils import get_mesh_id, init_logger, logger

from model_flowmap import patch_model_forward, setup_flowmap_model
from inference import create_inference_kwargs, flowmap_inference


class VideoGenerator:
    """视频生成器：使用 FlowMap 蒸馏模型生成 LIBERO 任务视频。"""

    def __init__(self, config, checkpoint_path, num_steps=2, device="cuda"):
        self.config = config
        self.device = torch.device(device)
        self.dtype = config.param_dtype
        self.num_steps = num_steps

        # 加载基础模型组件
        base_model_path = config.wan22_pretrained_model_name_or_path

        logger.info("Loading VAE...")
        self.vae = load_vae(
            os.path.join(base_model_path, "vae"),
            torch_dtype=self.dtype,
            torch_device=self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)

        logger.info("Loading tokenizer...")
        self.tokenizer = load_tokenizer(
            os.path.join(base_model_path, "tokenizer")
        )

        logger.info("Loading text encoder...")
        self.text_encoder = load_text_encoder(
            os.path.join(base_model_path, "text_encoder"),
            torch_dtype=self.dtype,
            torch_device=self.device,
        )

        logger.info("Loading transformer...")
        self.transformer = load_transformer(
            os.path.join(base_model_path, "transformer"),
            torch_dtype=self.dtype,
            torch_device=self.device,
            attn_mode="torch",
        )

        # 应用 FlowMap 修改
        if checkpoint_path is not None:
            logger.info("Applying FlowMap modifications...")
            ckpt_config_path = os.path.join(checkpoint_path, "config.json")
            ckpt_config = {}
            if os.path.exists(ckpt_config_path):
                with open(ckpt_config_path, "r") as f:
                    ckpt_config = json.load(f)

            gate_value = ckpt_config.get("gate_value", 0.0)
            deltatime_type = ckpt_config.get("deltatime_type", "r")

            self.transformer = setup_flowmap_model(
                self.transformer,
                gate_value=gate_value,
                deltatime_type=deltatime_type,
            )
            self.transformer = patch_model_forward(self.transformer)

            # 加载 LoRA 权重
            use_lora = ckpt_config.get("use_lora", False)
            if use_lora:
                logger.info("Loading LoRA adapters...")
                from peft import LoraConfig, get_peft_model

                lora_config = LoraConfig(
                    r=ckpt_config.get("lora_rank", 256),
                    lora_alpha=ckpt_config.get("lora_alpha", 128),
                    target_modules=ckpt_config.get(
                        "lora_target_modules",
                        [
                            "to_q", "to_k", "to_v", "to_out.0",
                            "ffn.net.0.proj", "ffn.net.2",
                            "time_proj",
                            "delta_embedder.linear_1",
                            "delta_embedder.linear_2",
                        ],
                    ),
                    lora_dropout=ckpt_config.get("lora_dropout", 0.0),
                    bias="none",
                )
                self.transformer = get_peft_model(
                    self.transformer, lora_config, adapter_name="default"
                )

                # 加载适配器权重
                adapter_weights_path = os.path.join(
                    checkpoint_path, "diffusion_pytorch_model.safetensors"
                )
                if os.path.exists(adapter_weights_path):
                    from safetensors.torch import load_file
                    adapter_state = load_file(adapter_weights_path)
                    model_keys = set(self.transformer.state_dict().keys())
                    loadable = {
                        k: v for k, v in adapter_state.items()
                        if k in model_keys
                    }
                    missing, unexpected = self.transformer.load_state_dict(
                        loadable, strict=False
                    )
                    logger.info(
                        f"LoRA weights loaded: {len(loadable)} keys, "
                        f"{len(missing)} missing, {len(unexpected)} unexpected"
                    )
                self.transformer.to(self.dtype)

        # 配置模型
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_model,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=True,
        )

        # 构建推理参数
        from easydict import EasyDict
        _inf_cfg = EasyDict({
            "num_train_timesteps": ckpt_config.get("num_train_timesteps", 1000),
            "snr_shift": ckpt_config.get("snr_shift", config.snr_shift),
            "action_snr_shift": ckpt_config.get("action_snr_shift", config.action_snr_shift),
        })
        self.flowmap_kwargs = create_inference_kwargs(_inf_cfg)

        # 环境配置
        self.height = config.height
        self.width = config.width
        self.action_dim = config.action_dim
        self.action_per_frame = config.action_per_frame
        self.obs_cam_keys = config.obs_cam_keys

        # 动作归一化参数
        self.actions_q01 = torch.tensor(
            config.norm_stat["q01"], dtype=torch.float32
        ).reshape(-1, 1, 1).to(self.device)
        self.actions_q99 = torch.tensor(
            config.norm_stat["q99"], dtype=torch.float32
        ).reshape(-1, 1, 1).to(self.device)
        self.action_norm_method = config.action_norm_method

        # 动作掩码
        self.action_mask = torch.zeros([config.action_dim]).bool()
        self.action_mask[config.used_action_channel_ids] = True

        logger.info("VideoGenerator initialized successfully.")

    def _get_t5_prompt_embeds(self, prompt, max_sequence_length=512):
        """获取 T5 文本嵌入。"""
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(
            text_input_ids.to(text_encoder_device),
            mask.to(text_encoder_device),
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat(
                    [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]
                )
                for u in prompt_embeds
            ],
            dim=0,
        )

        _, seq_len, _ = prompt_embeds.shape
        return prompt_embeds.to(self.device)

    def encode_prompt(self, prompt):
        """编码文本提示。"""
        prompt_embeds = self._get_t5_prompt_embeds(prompt)
        empty_embeds = self._get_t5_prompt_embeds("")
        return prompt_embeds, empty_embeds

    def normalize_latents(self, latents, latents_mean, latents_std):
        """归一化 latent。"""
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action):
        """预处理动作。"""
        action_model_input = torch.from_numpy(action).to(self.device)
        CA, FA, HA = action_model_input.shape
        action_model_input_paded = torch.nn.functional.pad(
            action_model_input, [0, 0, 0, 0, 0, 1], mode="constant", value=0
        )
        action_model_input = action_model_input_paded[
            self.config.inverse_used_action_channel_ids
        ]

        if self.action_norm_method == "quantiles":
            action_model_input = (
                (action_model_input - self.actions_q01)
                / (self.actions_q99 - self.actions_q01 + 1e-6)
                * 2.0
                - 1.0
            )
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        """后处理动作。"""
        action = action.cpu()  # B, C, F, H, W
        action = action[0, ..., 0]  # C, F, H
        if self.action_norm_method == "quantiles":
            # 确保所有张量在同一设备上
            actions_q01 = self.actions_q01.cpu()
            actions_q99 = self.actions_q99.cpu()
            action = (
                (action + 1)
                / 2
                * (actions_q99 - actions_q01 + 1e-6)
                + actions_q01
            )
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.config.used_action_channel_ids]

    def _encode_obs(self, obs):
        """编码观测图像。"""
        images = obs["obs"]
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None

        videos = []
        for k_i, k in enumerate(self.obs_cam_keys):
            height_i, width_i = self.height, self.width

            history_video_k = torch.from_numpy(
                np.stack([each[k] for each in images])
            ).float().permute(3, 0, 1, 2)
            history_video_k = torch.nn.functional.interpolate(
                history_video_k,
                size=(height_i, width_i),
                mode="bilinear",
                align_corners=False,
            ).unsqueeze(0)
            videos.append(history_video_k)

        videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
        vae_device = next(self.streaming_vae.vae.parameters()).device
        videos_chunk = videos.to(vae_device).to(self.dtype)
        enc_out = self.streaming_vae.encode_chunk(videos_chunk)

        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent.to(self.device)

    def generate_video(self, obs, prompt):
        """生成视频和动作。"""
        # 编码文本
        text_emb, empty_emb = self.encode_prompt(prompt)

        # 编码观测
        init_latent = self._encode_obs(obs)

        # 采样初始噪声
        frame_chunk_size = self.config.frame_chunk_size
        latent_height = self.height // 16
        latent_width = self.width // 16 * len(self.obs_cam_keys)

        latents = torch.randn(
            1, 48, frame_chunk_size, latent_height, latent_width,
            device=self.device, dtype=self.dtype,
        )
        actions = torch.randn(
            1, self.action_dim, frame_chunk_size, self.action_per_frame, 1,
            device=self.device, dtype=self.dtype,
        )

        # 运行 FlowMap 推理
        with torch.no_grad():
            denoised_latent, denoised_action = flowmap_inference(
                model=self.transformer,
                noisy_latent=latents,
                noisy_action=actions,
                text_emb=text_emb,
                empty_emb=empty_emb,
                num_steps=self.num_steps,
                cfg_scale=self.config.guidance_scale,
                **self.flowmap_kwargs,
            )

            # 恢复条件帧
            denoised_latent[:, :, 0:1] = init_latent[:, :, 0:1].to(self.dtype)
            denoised_action[:, :, 0:1] = 0.0

        denoised_action[:, ~self.action_mask] *= 0

        # 后处理动作
        actions_np = self.postprocess_action(denoised_action)

        return actions_np, denoised_latent


def save_video(frames, save_path, fps=15):
    """保存视频。"""
    import imageio
    imageio.mimsave(save_path, frames, fps=fps)
    logger.info(f"Video saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="FlowMap 蒸馏模型视频生成")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the distillation checkpoint directory",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2,
        help="Number of FlowMap inference steps (2-50, default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/outputs/flowmap_videos",
        help="Output directory for videos",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=3,
        help="Number of tasks to generate",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=2,
        help="Number of episodes per task",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    args = parser.parse_args()

    # 初始化分布式环境（单 GPU 模式）
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    init_distributed(1, 0, 0)

    # 加载配置
    config = VA_CONFIGS["libero"]

    # 创建视频生成器
    generator = VideoGenerator(
        config=config,
        checkpoint_path=args.checkpoint_path,
        num_steps=args.num_steps,
        device=args.device,
    )

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成视频
    logger.info(f"Generating videos for {args.num_tasks} tasks, {args.num_episodes} episodes each")

    # 注意：这里需要 LIBERO 环境来获取任务和初始状态
    # 如果没有 LIBERO 环境，可以使用模拟数据
    # 尝试导入 LIBERO
    try:
        from libero.libero import benchmark
        has_libero_benchmark = True
    except ImportError:
        has_libero_benchmark = False
        logger.warning("LIBERO benchmark not available, using task descriptions")

    try:
        from libero.libero.envs import OffScreenRenderEnv
        has_libero_env = True
    except ImportError:
        has_libero_env = False
        logger.warning("LIBERO environment not available, using simulated observations")

    # LIBERO 任务描述
    libero_task_descriptions = [
        "put both the alphabet soup and the tomato sauce in the basket",
        "put both the cream cheese box and the butter in the basket",
        "turn on the stove and put the moka pot on it",
        "put the white mug on the left plate and the yellow mug on the right plate",
        "put both the alphabet soup and the cream cheese box in the basket",
        "put both the butter and the tomato sauce in the basket",
        "put the moka pot on the stove and turn it on",
        "put the yellow mug on the left plate and the white mug on the right plate",
        "put the alphabet soup in the basket",
        "put the tomato sauce in the basket",
    ]

    # 获取任务信息
    if has_libero_benchmark:
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict["libero_10"]()
        num_tasks = min(args.num_tasks, benchmark_instance.get_num_tasks())
    else:
        num_tasks = min(args.num_tasks, len(libero_task_descriptions))

    for task_idx in range(num_tasks):
        # 获取任务描述
        if has_libero_benchmark:
            task = benchmark_instance.get_task(task_idx)
            prompt = task.language
        else:
            prompt = libero_task_descriptions[task_idx]

        logger.info(f"Task {task_idx}: {prompt}")

        # 获取初始状态
        if has_libero_benchmark and has_libero_env:
            env_args = {
                "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
                "camera_heights": 128,
                "camera_widths": 128,
            }
            init_states = benchmark_instance.get_task_init_states(task_idx)

        for episode_idx in range(args.num_episodes):
            logger.info(f"  Episode {episode_idx}")

            # 获取观测数据
            if has_libero_benchmark and has_libero_env:
                try:
                    env = OffScreenRenderEnv(**env_args)
                    env.reset()
                    env.set_init_state(init_states[episode_idx % init_states.shape[0]])
                    obs, _, _, _ = env.step([0.] * 7)
                    obs_data = {
                        "observation.images.agentview_rgb": np.ascontiguousarray(obs["agentview_image"][::-1]),
                        "observation.images.eye_in_hand_rgb": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]),
                    }
                    env.close()
                except Exception as e:
                    logger.warning(f"Failed to get observation from env: {e}")
                    obs_data = {
                        "observation.images.agentview_rgb": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
                        "observation.images.eye_in_hand_rgb": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
                    }
            else:
                # 使用模拟观测
                obs_data = {
                    "observation.images.agentview_rgb": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
                    "observation.images.eye_in_hand_rgb": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
                }

            # 生成视频（需要提供多帧观测，VAE 期望序列输入）
            try:
                # 创建多帧观测（重复同一帧以满足 VAE 输入要求）
                # VAE 期望至少 4 帧的序列
                num_obs_frames = 4
                obs_list = [obs_data] * num_obs_frames

                actions, latents = generator.generate_video(
                    obs={"obs": obs_list},
                    prompt=prompt,
                )

                # 保存结果
                task_dir = output_dir / f"{task_idx}_{prompt.replace(' ', '_')}"
                task_dir.mkdir(exist_ok=True)

                # 保存动作
                np.save(task_dir / f"actions_{episode_idx}.npy", actions)

                # 保存 latent
                torch.save(latents.cpu(), task_dir / f"latents_{episode_idx}.pt")

                logger.info(f"    Actions shape: {actions.shape}")
                logger.info(f"    Saved to: {task_dir}")
            except Exception as e:
                logger.error(f"    Failed to generate video: {e}")
                import traceback
                traceback.print_exc()

    logger.info("Video generation complete!")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    init_logger()
    main()
