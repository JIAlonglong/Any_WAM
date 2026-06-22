# Copyright 2024-2025 The Flash-WAM Team Authors. All rights reserved.
"""
Flash-WAM FlowMap Distillation Inference Server.

This is the inference server entry point for Flash-WAM distillation models.
It loads a FlowMap-distilled transformer, applies LoRA adapters if present,
and runs flowmap_inference() for fast 2-50 step denoising.

Supported environments:
  - LIBERO:  128x128, 2 cameras, 7-dim action, action_per_frame=4
  - RobotWin: 256x320, 3 cameras, 16-dim action, action_per_frame=16

Usage:
  torchrun --nproc_per_node=1 wan_va/wan_va_server.py \
      --config-name robotwin \
      --checkpoint-path /path/to/distillation/checkpoint/step_N/target_student/transformer \
      --num-steps 2 \
      --port 29536
"""

import argparse
import json
import os
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict

# ---------------------------------------------------------------------------
# Path setup: add wan_va and distillation_flowmap to Python path
# ---------------------------------------------------------------------------
_server_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_server_dir)

sys.path.insert(0, _server_dir)
sys.path.insert(0, os.path.join(_project_root, "distillation_flowmap"))

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
from wan_va.utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)

from model_flowmap import patch_model_forward, setup_flowmap_model
from inference import create_inference_kwargs, flowmap_inference
from einops import rearrange
from tqdm import tqdm


class VA_Server:
    """Flash-WAM FlowMap distillation inference server.

    Loads the base transformer, applies FlowMap modifications (dual timestep
    embeddings + patched forward), loads LoRA adapters if the checkpoint was
    trained with LoRA, and serves action predictions over websocket.
    """

    def __init__(self, job_config):
        self.cache_name = "pos"
        self.job_config = job_config
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(job_config, "enable_offload", False)
        self.num_steps = getattr(job_config, "num_steps", 2)
        self.action_num_steps = getattr(job_config, "action_num_steps", None)

        # ------------------------------------------------------------------
        # 1. Load pretrained components (VAE, text encoder, tokenizer)
        # ------------------------------------------------------------------
        base_model_path = job_config.wan22_pretrained_model_name_or_path

        self.vae = load_vae(
            os.path.join(base_model_path, "vae"),
            torch_dtype=self.dtype,
            torch_device="cpu" if self.enable_offload else self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)

        self.tokenizer = load_tokenizer(
            os.path.join(base_model_path, "tokenizer")
        )

        self.text_encoder = load_text_encoder(
            os.path.join(base_model_path, "text_encoder"),
            torch_dtype=self.dtype,
            torch_device="cpu" if self.enable_offload else self.device,
        )

        # ------------------------------------------------------------------
        # 2. Load transformer from pretrained base model
        # ------------------------------------------------------------------
        logger.info("Loading base transformer ...")
        self.transformer = load_transformer(
            os.path.join(base_model_path, "transformer"),
            torch_dtype=self.dtype,
            torch_device=self.device,
            attn_mode="torch",
        )

        # ------------------------------------------------------------------
        # 3. Apply FlowMap modifications (dual timestep embedder + patched fwd)
        # ------------------------------------------------------------------
        checkpoint_path = getattr(job_config, "checkpoint_path", None)
        if checkpoint_path is not None:
            logger.info("Applying FlowMap modifications ...")
            # Read checkpoint config to get gate_value and deltatime_type
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

            # ------------------------------------------------------------------
            # 4. Load LoRA adapters if checkpoint was trained with LoRA
            # ------------------------------------------------------------------
            use_lora = ckpt_config.get("use_lora", False)
            if use_lora:
                logger.info("Loading LoRA adapters ...")
                try:
                    from peft import LoraConfig, get_peft_model
                except ImportError:
                    raise ImportError(
                        "LoRA checkpoint requires the 'peft' library. "
                        "Install with: pip install peft"
                    )

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

                # Load adapter weights
                adapter_weights_path = os.path.join(
                    checkpoint_path, "diffusion_pytorch_model.safetensors"
                )
                if os.path.exists(adapter_weights_path):
                    from safetensors.torch import load_file
                    adapter_state = load_file(adapter_weights_path)
                    # Filter to only adapter-relevant keys
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
                else:
                    logger.warning(
                        f"LoRA weights not found at {adapter_weights_path}, "
                        "using randomly initialized LoRA"
                    )
                # Ensure uniform dtype for FSDP (LoRA adapters default to float32)
                self.transformer.to(self.dtype)
            else:
                # Full model checkpoint (not LoRA)
                full_weights_path = os.path.join(
                    checkpoint_path, "diffusion_pytorch_model.safetensors"
                )
                if os.path.exists(full_weights_path):
                    from safetensors.torch import load_file
                    full_state = load_file(full_weights_path)
                    missing, unexpected = self.transformer.load_state_dict(
                        full_state, strict=False
                    )
                    logger.info(
                        f"Full model weights loaded: {len(full_state)} keys, "
                        f"{len(missing)} missing, {len(unexpected)} unexpected"
                    )
                else:
                    logger.warning(
                        f"Checkpoint weights not found at {full_weights_path}"
                    )
        else:
            logger.warning(
                "No checkpoint path provided; running with base pretrained weights only."
            )

        # ------------------------------------------------------------------
        # 5. Configure model (FSDP sharding / dtype / eval mode)
        # ------------------------------------------------------------------
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_model,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=True,
        )

        # ------------------------------------------------------------------
        # 6. Build FlowMap inference kwargs from distillation config
        # ------------------------------------------------------------------
        # Use distillation config defaults; checkpoint config may override
        if checkpoint_path is not None and os.path.exists(ckpt_config_path):
            # Create a minimal config-like object for create_inference_kwargs
            _inf_cfg = EasyDict({
                "num_train_timesteps": ckpt_config.get(
                    "num_train_timesteps", 1000
                ),
                "snr_shift": ckpt_config.get(
                    "snr_shift", job_config.snr_shift
                ),
                "action_snr_shift": ckpt_config.get(
                    "action_snr_shift", job_config.action_snr_shift
                ),
            })
        else:
            _inf_cfg = EasyDict({
                "num_train_timesteps": 1000,
                "snr_shift": job_config.snr_shift,
                "action_snr_shift": job_config.action_snr_shift,
            })
        self.flowmap_kwargs = create_inference_kwargs(_inf_cfg)

        # ------------------------------------------------------------------
        # 6b. Initialize FlowMatchSchedulers for standard denoising
        # ------------------------------------------------------------------
        self.scheduler = FlowMatchScheduler(
            shift=job_config.snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.action_scheduler = FlowMatchScheduler(
            shift=job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        # ------------------------------------------------------------------
        # 7. Environment-specific setup (robotwin_tshape needs dual VAE)
        # ------------------------------------------------------------------
        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == "robotwin_tshape":
            vae_half = load_vae(
                os.path.join(base_model_path, "vae"),
                torch_dtype=self.dtype,
                torch_device="cpu" if self.enable_offload else self.device,
            )
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)

        logger.info("VA_Server initialized successfully.")

    # ==================================================================
    # Text encoding (adapted from lingbot-va)
    # ==================================================================

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        device = device or self.device
        dtype = dtype or self.dtype

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
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
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
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_videos_per_prompt, seq_len, -1
        )
        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str
            ) else negative_prompt
            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    # ==================================================================
    # Latent normalization
    # ==================================================================

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    # ==================================================================
    # Action pre/post processing
    # ==================================================================

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(action)
        CA, FA, HA = action_model_input.shape
        action_model_input_paded = F.pad(
            action_model_input, [0, 0, 0, 0, 0, 1], mode="constant", value=0
        )
        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids
        ]

        if self.action_norm_method == "quantiles":
            action_model_input = (
                (action_model_input - self.actions_q01)
                / (self.actions_q99 - self.actions_q01 + 1e-6)
                * 2.0
                - 1.0
            )
        else:
            raise NotImplementedError(
                f"Unsupported action_norm_method: {self.action_norm_method}"
            )
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        action = action.cpu()  # B, C, F, H, W
        action = action[0, ..., 0]  # C, F, H
        if self.action_norm_method == "quantiles":
            action = (
                (action + 1)
                / 2
                * (self.actions_q99 - self.actions_q01 + 1e-6)
                + self.actions_q01
            )
        else:
            raise NotImplementedError(
                f"Unsupported action_norm_method: {self.action_norm_method}"
            )
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.job_config.used_action_channel_ids]

    # ==================================================================
    # Observation encoding
    # ==================================================================

    def _encode_obs(self, obs):
        images = obs["obs"]
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None

        videos = []
        for k_i, k in enumerate(self.job_config.obs_cam_keys):
            if self.env_type == "robotwin_tshape":
                if k_i == 0:
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            history_video_k = torch.from_numpy(
                np.stack([each[k] for each in images])
            ).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(
                history_video_k,
                size=(height_i, width_i),
                mode="bilinear",
                align_corners=False,
            ).unsqueeze(0)
            videos.append(history_video_k)

        if self.env_type == "robotwin_tshape":
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:], dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            enc_out_high = self.streaming_vae.encode_chunk(
                videos_high.to(vae_device).to(self.dtype)
            )
            enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(vae_device).to(self.dtype)
            )
            enc_out = torch.cat(
                [
                    torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                    enc_out_high,
                ],
                dim=-2,
            )
        else:
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

    # ==================================================================
    # Reset
    # ==================================================================

    def _reset(self, prompt=None):
        logger.info("Reset.")
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        self.frame_st_id = 0
        self.init_latent = None

        # Clear caches
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if self.env_type == "robotwin_tshape":
            self.latent_height = ((self.height // 16) * 3) // 2
            self.latent_width = self.width // 16
            self.streaming_vae_half.clear_cache()
        else:
            self.latent_height = self.height // 16
            self.latent_width = (
                self.width // 16 * len(self.job_config.obs_cam_keys)
            )

        patch_size = self.job_config.patch_size
        latent_token_per_chunk = (
            self.job_config.frame_chunk_size
            * self.latent_height
            * self.latent_width
        ) // (patch_size[0] * patch_size[1] * patch_size[2])
        action_token_per_chunk = (
            self.job_config.frame_chunk_size * self.action_per_frame
        )
        self.transformer.create_empty_cache(
            self.cache_name,
            self.job_config.attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            dtype=self.dtype,
            device=self.device,
            batch_size=2 if self.use_cfg else 1,
        )

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(
            self.job_config.norm_stat["q01"], dtype=torch.float32
        ).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(
            self.job_config.norm_stat["q99"], dtype=torch.float32
        ).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        # Encode prompt
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.job_config.guidance_scale > 1,
                num_videos_per_prompt=1,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )

        self.exp_name = (
            f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        )
        self.exp_save_root = os.path.join(
            getattr(self.job_config, "save_root", "./train_out"),
            "real",
            self.exp_name,
        )
        os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    # ==================================================================
    # Compute KV cache (streaming observation encoding)
    # ==================================================================

    def _compute_kv_cache(self, obs):
        self.transformer.clear_pred_cache(self.cache_name)
        save_async(
            obs["obs"],
            os.path.join(self.exp_save_root, f"obs_data_{self.frame_st_id}.pt"),
        )
        latent_model_input = self._encode_obs(obs)
        if self.frame_st_id == 0:
            latent_model_input = (
                torch.cat([self.init_latent, latent_model_input], dim=2)
                if latent_model_input is not None
                else self.init_latent
            )

        action_model_input = self.preprocess_action(obs["state"])
        action_model_input = action_model_input.to(latent_model_input)
        logger.info(
            f"get KV cache obs: {latent_model_input.shape} "
            f"{action_model_input.shape}"
        )

        # Use standard forward (not flowmap) for KV cache computation
        patch_size = self.job_config.patch_size
        input_dict = self._prepare_input(
            latent_model_input, action_model_input, frame_st_id=self.frame_st_id
        )

        with torch.no_grad():
            self.transformer(
                self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=False,
            )
            self.transformer(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=True,
            )

        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]

    # ==================================================================
    # Input preparation helpers
    # ==================================================================

    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict["noisy_latents"] = input_dict["noisy_latents"].repeat(
                2, 1, 1, 1, 1
            )
            input_dict["text_emb"] = torch.cat(
                [
                    self.prompt_embeds.to(self.dtype).clone(),
                    self.negative_prompt_embeds.to(self.dtype).clone(),
                ],
                dim=0,
            )
            input_dict["grid_id"] = input_dict["grid_id"][None].repeat(2, 1, 1)
            input_dict["timesteps"] = input_dict["timesteps"][None].repeat(2, 1)
        else:
            input_dict["grid_id"] = input_dict["grid_id"][None]
            input_dict["timesteps"] = input_dict["timesteps"][None]
        return input_dict

    def _prepare_input(
        self,
        latent_model_input,
        action_model_input,
        latent_t=0,
        action_t=0,
        latent_cond=None,
        action_cond=None,
        frame_st_id=0,
        patch_size=(1, 2, 2),
    ):
        """Prepare input_dict for the transformer (non-flowmap forward)."""
        input_dict = dict()

        if latent_model_input is not None:
            input_dict["latent_res_lst"] = {
                "noisy_latents": latent_model_input,
                "timesteps": torch.ones(
                    [latent_model_input.shape[2]],
                    dtype=torch.float32,
                    device=self.device,
                )
                * latent_t,
                "grid_id": get_mesh_id(
                    latent_model_input.shape[-3] // patch_size[0],
                    latent_model_input.shape[-2] // patch_size[1],
                    latent_model_input.shape[-1] // patch_size[2],
                    0,
                    1,
                    frame_st_id,
                ).to(self.device),
                "text_emb": self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict["latent_res_lst"]["noisy_latents"][:, :, 0:1] = (
                    latent_cond[:, :, 0:1]
                )
                input_dict["latent_res_lst"]["timesteps"][0:1] *= 0

        if action_model_input is not None:
            input_dict["action_res_lst"] = {
                "noisy_latents": action_model_input,
                "timesteps": torch.ones(
                    [action_model_input.shape[2]],
                    dtype=torch.float32,
                    device=self.device,
                )
                * action_t,
                "grid_id": get_mesh_id(
                    action_model_input.shape[-3],
                    action_model_input.shape[-2],
                    action_model_input.shape[-1],
                    1,
                    1,
                    frame_st_id,
                    action=True,
                ).to(self.device),
                "text_emb": self.prompt_embeds.to(self.dtype).clone(),
            }
            if action_cond is not None:
                input_dict["action_res_lst"]["noisy_latents"][:, :, 0:1] = (
                    action_cond[:, :, 0:1]
                )
                input_dict["action_res_lst"]["timesteps"][0:1] *= 0
            input_dict["action_res_lst"]["noisy_latents"][
                :, ~self.action_mask
            ] *= 0

        return input_dict

    # ==================================================================
    # FlowMap inference (the core denoising loop)
    # ==================================================================

    def _infer(self, obs, frame_st_id=0):
        """
        Standard denoising-based inference with streaming KV cache.
        Follows the original LingBot-VA server pattern: video denoising loop
        followed by action denoising loop, using update_cache to maintain
        temporal context across chunks.
        """
        frame_chunk_size = self.job_config.frame_chunk_size
        action_dim = self.job_config.action_dim

        if frame_st_id == 0:
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent

        latents = torch.randn(
            1, 48, frame_chunk_size, self.latent_height, self.latent_width,
            device=self.device, dtype=self.dtype,
        )
        actions = torch.randn(
            1, action_dim, frame_chunk_size, self.action_per_frame, 1,
            device=self.device, dtype=self.dtype,
        )

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)
        if video_step != -1:
            timesteps = timesteps[:video_step]

        action_timesteps = F.pad(action_timesteps, (0, 1), mode='constant', value=0)

        patch_size = self.job_config.patch_size

        with torch.no_grad():
            # --- Video Generation Loop ---
            for i, t in enumerate(tqdm(timesteps)):
                last_step = i == len(timesteps) - 1
                latent_cond = self.init_latent[:, :, 0:1].to(self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_input(
                    latents, None, t, t, latent_cond, None, frame_st_id=frame_st_id,
                )

                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False,
                )

                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        patch_size, video_noise_pred, frame_chunk_size,
                        self.latent_height, self.latent_width,
                        batch_size=2 if self.use_cfg else 1,
                    )
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (
                            video_noise_pred[:1] - video_noise_pred[1:]
                        )
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(
                        video_noise_pred, t, latents, return_dict=False,
                    )

                latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

            # --- Action Generation Loop ---
            for i, t in enumerate(tqdm(action_timesteps)):
                last_step = i == len(action_timesteps) - 1
                action_cond = torch.zeros(
                    [1, action_dim, 1, self.action_per_frame, 1],
                    device=self.device, dtype=self.dtype,
                ) if frame_st_id == 0 else None

                input_dict = self._prepare_input(
                    None, actions, t, t, None, action_cond, frame_st_id=frame_st_id,
                )
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True,
                )

                if not last_step:
                    action_noise_pred = rearrange(
                        action_noise_pred, 'b (f n) c -> b c f n 1', f=frame_chunk_size,
                    )
                    if self.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (
                            action_noise_pred[:1] - action_noise_pred[1:]
                        )
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(
                        action_noise_pred, t, actions, return_dict=False,
                    )

                actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

        actions[:, ~self.action_mask] *= 0

        save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        actions_np = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        return actions_np, latents

    # ==================================================================
    # Public inference interface
    # ==================================================================

    @torch.no_grad()
    def infer(self, obs):
        reset = obs.get("reset", False)
        prompt = obs.get("prompt", None)
        compute_kv_cache = obs.get("compute_kv_cache", False)

        if reset:
            logger.info("******************* Reset server ******************")
            self._reset(prompt=prompt)
            return dict()
        elif compute_kv_cache:
            logger.info("################# Compute KV Cache #################")
            self._compute_kv_cache(obs)
            return dict()
        else:
            logger.info("################# Infer One Chunk #################")
            action, _ = self._infer(obs, frame_st_id=self.frame_st_id)
            return dict(action=action)


# =========================================================================
# Entry point
# =========================================================================


def run(args):
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port

    # Apply CLI overrides
    if args.checkpoint_path is not None:
        config.checkpoint_path = args.checkpoint_path
    if args.num_steps is not None:
        config.num_steps = args.num_steps
    if args.action_num_steps is not None:
        config.action_num_steps = args.action_num_steps
    if args.save_root is not None:
        config.save_root = args.save_root

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    logger.info(f"Config: {args.config_name}")
    logger.info(f"Checkpoint: {config.checkpoint_path}")
    logger.info(f"Num steps: {config.num_steps}")
    logger.info(f"Port: {port}")

    model = VA_Server(config)
    run_async_server_mode(model, local_rank, config.host, port)


def main():
    parser = argparse.ArgumentParser(
        description="Flash-WAM FlowMap Distillation Inference Server"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default="robotwin",
        choices=["libero", "robotwin"],
        help="Environment config name (libero or robotwin).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (overrides config default).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help=(
            "Path to the distillation checkpoint directory "
            "(e.g., .../step_1000/target_student/transformer). "
            "Must contain config.json and diffusion_pytorch_model.safetensors."
        ),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2,
        help="Number of FlowMap inference steps for video (2-50, default: 2).",
    )
    parser.add_argument(
        "--action-num-steps",
        type=int,
        default=None,
        help="Number of FlowMap inference steps for action (default: same as --num-steps).",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving inference outputs.",
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finished all process!")


if __name__ == "__main__":
    init_logger()
    main()
