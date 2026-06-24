#!/usr/bin/env python3
"""
Self-contained video generation script for LingBot-VA / Flash-WAM.

Supports two modes:
  - Teacher: 20-step original FlowMatchScheduler denoising (LingBot-VA)
  - Student:  8-step FlowMap inference with LoRA (Flash-WAM)

Usage:
  # Student (8 steps, LoRA) -- default
  CUDA_VISIBLE_DEVICES=0 python generate_video_lingbot.py --mode student

  # Teacher (20 steps, no LoRA)
  CUDA_VISIBLE_DEVICES=0 python generate_video_lingbot.py --mode teacher
"""

import argparse
import os
import sys
import json
import time

# =========================================================================
# Path setup -- must happen before any project imports
# =========================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add lingbot-va (original server) paths first for Configs/Modules/Utils
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "wan_va"))

# Add distillation_flowmap for FlowMap inference (Student mode)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "distillation_flowmap"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

# LingBot-VA imports (from lingbot-va/wan_va/ on sys.path)
# from configs import VA_CONFIGS  # not used, config built manually
from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    save_async,
)

# Flash-WAM imports (from distillation_flowmap/ on sys.path, for Student mode)
from model_flowmap import patch_model_forward, setup_flowmap_model
from inference import create_inference_kwargs, flowmap_inference


# =========================================================================
# LIBERO config (matches va_libero_i2va_cfg)
# =========================================================================
def build_libero_config(base_model_path, example_dir, num_chunks=10, cfg_scale=2.0):
    """Build a LIBERO I2VA config matching the original va_libero_i2va_cfg."""
    from easydict import EasyDict
    cfg = EasyDict(__name__="Config: VA libero i2va (generated)")

    # Base model path
    cfg.wan22_pretrained_model_name_or_path = base_model_path

    # LIBERO-specific (matching va_libero_cfg.py)
    cfg.attn_window = 30
    cfg.frame_chunk_size = 4
    cfg.env_type = "none"
    cfg.height = 128
    cfg.width = 128
    cfg.action_dim = 30
    cfg.action_per_frame = 4
    cfg.obs_cam_keys = [
        "observation.images.agentview_rgb",
        "observation.images.eye_in_hand_rgb",
    ]
    cfg.guidance_scale = cfg_scale
    cfg.action_guidance_scale = 1
    cfg.num_inference_steps = 20
    cfg.video_exec_step = -1
    cfg.action_num_inference_steps = 50
    cfg.snr_shift = 5.0
    cfg.action_snr_shift = 0.05

    # Action config
    cfg.used_action_channel_ids = list(range(0, 7))
    inverse_ids = [len(cfg.used_action_channel_ids)] * cfg.action_dim
    for i, j in enumerate(cfg.used_action_channel_ids):
        inverse_ids[j] = i
    cfg.inverse_used_action_channel_ids = inverse_ids
    cfg.action_norm_method = "quantiles"
    cfg.norm_stat = {
        "q01": [
            -0.6589285731315613, -0.84375, -0.9375,
            -0.12107142806053162, -0.15964286029338837,
            -0.26571428775787354, -1.0
        ] + [0.0] * 23,
        "q99": [
            0.8999999761581421, 0.8544642925262451, 0.9375,
            0.17142857611179352, 0.1842857152223587,
            0.34392857551574707, 1.0
        ] + [0.0] * 23,
    }

    # Shared config
    cfg.host = "0.0.0.0"
    cfg.port = 29536
    cfg.param_dtype = torch.bfloat16
    cfg.save_root = "./train_out"
    cfg.patch_size = (1, 2, 2)
    cfg.enable_offload = False
    cfg.infer_mode = "i2va"

    # I2VA overrides
    cfg.input_img_path = example_dir
    cfg.num_chunks_to_infer = num_chunks
    cfg.prompt = "put both the alphabet soup and the tomato sauce in the basket"

    return cfg


# =========================================================================
# RobotWin config (matches va_robotwin_cfg)
# =========================================================================
def build_robotwin_config(base_model_path, example_dir, num_chunks=10, cfg_scale=2.0):
    from easydict import EasyDict
    cfg = EasyDict(__name__="Config: VA robotwin (generated)")
    cfg.wan22_pretrained_model_name_or_path = base_model_path
    cfg.attn_window = 72
    cfg.frame_chunk_size = 2
    cfg.env_type = "robotwin_tshape"
    cfg.height = 256
    cfg.width = 320
    cfg.action_dim = 30
    cfg.action_per_frame = 16
    cfg.obs_cam_keys = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
    cfg.guidance_scale = cfg_scale
    cfg.action_guidance_scale = 1
    cfg.num_inference_steps = 25
    cfg.video_exec_step = -1
    cfg.action_num_inference_steps = 50
    cfg.snr_shift = 5.0
    cfg.action_snr_shift = 1.0
    cfg.used_action_channel_ids = list(range(0, 7)) + list(range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
    inv = [len(cfg.used_action_channel_ids)] * cfg.action_dim
    for i, j in enumerate(cfg.used_action_channel_ids):
        inv[j] = i
    cfg.inverse_used_action_channel_ids = inv
    cfg.action_norm_method = "quantiles"
    cfg.norm_stat = {
        "q01": [-0.061727, -3.67e-05, -0.087835, -1, -1, -1, -1, -0.35471, -1.31e-06, -0.11975, -1, -1, -1, -1] + [0.] * 16,
        "q99": [0.34626, 0.39967, 0.14746, 1, 1, 1, 1, 0.03420, 0.39143, 0.17923, 1, 1, 1, 1] + [0.] * 14 + [1.0, 1.0],
    }
    cfg.host = "0.0.0.0"; cfg.port = 29536
    cfg.param_dtype = torch.bfloat16; cfg.save_root = "./train_out"
    cfg.patch_size = (1, 2, 2); cfg.enable_offload = False
    cfg.infer_mode = "i2va"
    cfg.input_img_path = example_dir
    cfg.num_chunks_to_infer = num_chunks
    cfg.prompt = "put the pot on the stove"
    return cfg


# =========================================================================
# VideoGenerator class
# =========================================================================
class VideoGenerator:
    """Generates videos from initial observations using LingBot-VA models.

    Supports two modes:
      - Teacher: Original FlowMatchScheduler-based denoising (20 steps)
      - Student: FlowMap-based fast denoising with LoRA (8 steps)
    """

    def __init__(self, config, mode="teacher", checkpoint_path=None, num_steps=None):
        self.config = config
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.cache_name = "pos"
        self.dtype = config.param_dtype
        self.device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        self.enable_offload = getattr(config, "enable_offload", False)
        self.env_type = getattr(config, "env_type", "none")

        logger.info(f"Mode: {mode}")
        logger.info(f"Device: {self.device}, dtype: {self.dtype}")

        # ---- 1. Load VAE ----
        self.vae = load_vae(
            os.path.join(config.wan22_pretrained_model_name_or_path, "vae"),
            torch_dtype=self.dtype,
            torch_device="cpu" if self.enable_offload else self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        self.streaming_vae_half = None
        if self.env_type == "robotwin_tshape":
            vae_half = load_vae(os.path.join(config.wan22_pretrained_model_name_or_path, "vae"), torch_dtype=self.dtype, torch_device="cpu" if self.enable_offload else self.device)
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)
        logger.info("VAE loaded.")

        # ---- 2. Load tokenizer & text encoder ----
        self.tokenizer = load_tokenizer(
            os.path.join(config.wan22_pretrained_model_name_or_path, "tokenizer")
        )
        self.text_encoder = load_text_encoder(
            os.path.join(config.wan22_pretrained_model_name_or_path, "text_encoder"),
            torch_dtype=self.dtype,
            torch_device="cpu" if self.enable_offload else self.device,
        )
        logger.info("Text encoder loaded.")

        # ---- 3. Load transformer ----
        self._load_transformer(config, mode, checkpoint_path, num_steps)
        logger.info("Transformer loaded.")

        # ---- 4. Schedulers ----
        self.scheduler = FlowMatchScheduler(
            shift=config.snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.action_scheduler = FlowMatchScheduler(
            shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        # ---- 5. VideoProcessor for decoding ----
        self.video_processor = VideoProcessor(vae_scale_factor=1)

    def _load_transformer(self, config, mode, checkpoint_path, num_steps):
        """Load transformer with mode-specific logic."""
        base_path = config.wan22_pretrained_model_name_or_path

        self.transformer = load_transformer(
            os.path.join(base_path, "transformer"),
            torch_dtype=self.dtype,
            torch_device=self.device,
            attn_mode="torch",
        )
        logger.info("Base transformer loaded.")

        if mode == "student":
            assert checkpoint_path is not None, "Student mode requires --checkpoint-path"
            self._setup_flowmap_and_lora(checkpoint_path, num_steps)
            self.num_steps = num_steps or 8
        else:
            self.num_steps = num_steps or config.num_inference_steps  # 20

        # FSDP sharding (world_size=1 -> effectively identity)
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_model,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=True,
        )
        logger.info(f"Model configured (mode={mode}, num_steps={self.num_steps}).")

    def _setup_flowmap_and_lora(self, checkpoint_path, num_steps):
        """Apply FlowMap modifications and load LoRA weights."""
        logger.info("Applying FlowMap modifications ...")

        # Read checkpoint config
        ckpt_config_path = os.path.join(checkpoint_path, "config.json")
        with open(ckpt_config_path, "r") as f:
            ckpt_config = json.load(f)

        gate_value = ckpt_config.get("gate_value", 0.0)
        deltatime_type = ckpt_config.get("deltatime_type", "r")

        # Apply FlowMap modifications
        self.transformer = setup_flowmap_model(
            self.transformer,
            gate_value=gate_value,
            deltatime_type=deltatime_type,
        )
        self.transformer = patch_model_forward(self.transformer)

        # Load LoRA if checkpoint uses it
        use_lora = ckpt_config.get("use_lora", False)
        if use_lora:
            logger.info("Loading LoRA adapters ...")
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=ckpt_config.get("lora_rank", 128),
                lora_alpha=ckpt_config.get("lora_alpha", 64),
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
            from safetensors.torch import load_file
            adapter_state = load_file(adapter_weights_path)
            model_keys = set(self.transformer.state_dict().keys())
            loadable = {k: v for k, v in adapter_state.items() if k in model_keys}
            missing, unexpected = self.transformer.load_state_dict(loadable, strict=False)
            logger.info(
                f"LoRA weights: {len(loadable)} keys loaded, "
                f"{len(missing)} missing, {len(unexpected)} unexpected"
            )
            self.transformer.to(self.dtype)
        else:
            # Full model checkpoint (no LoRA)
            full_weights_path = os.path.join(
                checkpoint_path, "diffusion_pytorch_model.safetensors"
            )
            from safetensors.torch import load_file
            full_state = load_file(full_weights_path)
            missing, unexpected = self.transformer.load_state_dict(full_state, strict=False)
            logger.info(
                f"Full weights: {len(full_state)} keys loaded, "
                f"{len(missing)} missing, {len(unexpected)} unexpected"
            )

        # Build flowmap kwargs
        from easydict import EasyDict
        _inf_cfg = EasyDict({
            "num_train_timesteps": ckpt_config.get("num_train_timesteps", 1000),
            "snr_shift": ckpt_config.get("snr_shift", self.config.snr_shift),
            "action_snr_shift": ckpt_config.get("action_snr_shift", self.config.action_snr_shift),
        })
        self.flowmap_kwargs = create_inference_kwargs(_inf_cfg)
        logger.info("FlowMap setup complete.")

    # ==================================================================
    # Text encoding (from original server)
    # ==================================================================
    def _get_t5_prompt_embeds(self, prompt, num_videos_per_prompt=1, max_sequence_length=512):
        device = self.device
        dtype = self.dtype

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
        prompt_embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ], dim=0)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        return prompt_embeds.to(device)

    def encode_prompt(self, prompt, negative_prompt=None, do_classifier_free_guidance=True,
                      max_sequence_length=226):
        device = self.device
        dtype = self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt, max_sequence_length=max_sequence_length,
        )

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt, max_sequence_length=max_sequence_length,
            )

        return prompt_embeds, negative_prompt_embeds

    # ==================================================================
    # Latent normalization
    # ==================================================================
    def normalize_latents(self, latents, latents_mean, latents_std):
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    # ==================================================================
    # Observation encoding (from original server)
    # ==================================================================
    def _encode_obs(self, obs):
        images = obs["obs"]
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None

        videos = []
        for k_i, k in enumerate(self.config.obs_cam_keys):
            if self.env_type == "robotwin_tshape":
                height_i, width_i = (self.height, self.width) if k_i == 0 else (self.height // 2, self.width // 2)
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
            v_high = videos[0] / 255.0 * 2.0 - 1.0
            v_lr = torch.cat(videos[1:], dim=0) / 255.0 * 2.0 - 1.0
            vd = next(self.streaming_vae.vae.parameters()).device
            eh = self.streaming_vae.encode_chunk(v_high.to(vd).to(self.dtype))
            el = self.streaming_vae_half.encode_chunk(v_lr.to(vd).to(self.dtype))
            enc_out = torch.cat([torch.cat(el.split(1, dim=0), dim=-1), eh], dim=-2)
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
    # Load initial observations from PNG files
    # ==================================================================
    def load_init_obs(self):
        imf_dict = {
            v: np.array(Image.open(os.path.join(self.config.input_img_path, f"{v}.png")).convert("RGB"))
            for v in self.config.obs_cam_keys
        }
        init_obs = {"obs": [imf_dict]}
        return init_obs

    # ==================================================================
    # Input preparation (from original server)
    # ==================================================================
    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict["noisy_latents"] = input_dict["noisy_latents"].repeat(2, 1, 1, 1, 1)
            input_dict["text_emb"] = torch.cat([
                self.prompt_embeds.to(self.dtype).clone(),
                self.negative_prompt_embeds.to(self.dtype).clone(),
            ], dim=0)
            input_dict["grid_id"] = input_dict["grid_id"][None].repeat(2, 1, 1)
            input_dict["timesteps"] = input_dict["timesteps"][None].repeat(2, 1)
        else:
            input_dict["grid_id"] = input_dict["grid_id"][None]
            input_dict["timesteps"] = input_dict["timesteps"][None]
        return input_dict

    def _prepare_latent_input(self, latent_model_input, action_model_input,
                               latent_t=0, action_t=0, latent_cond=None, action_cond=None,
                               frame_st_id=0, patch_size=(1, 2, 2)):
        input_dict = dict()

        if latent_model_input is not None:
            input_dict["latent_res_lst"] = {
                "noisy_latents": latent_model_input,
                "timesteps": torch.ones(
                    [latent_model_input.shape[2]], dtype=torch.float32, device=self.device
                ) * latent_t,
                "grid_id": get_mesh_id(
                    latent_model_input.shape[-3] // patch_size[0],
                    latent_model_input.shape[-2] // patch_size[1],
                    latent_model_input.shape[-1] // patch_size[2],
                    0, 1, frame_st_id,
                ).to(self.device),
                "text_emb": self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict["latent_res_lst"]["noisy_latents"][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict["latent_res_lst"]["timesteps"][0:1] *= 0

        if action_model_input is not None:
            input_dict["action_res_lst"] = {
                "noisy_latents": action_model_input,
                "timesteps": torch.ones(
                    [action_model_input.shape[2]], dtype=torch.float32, device=self.device
                ) * action_t,
                "grid_id": get_mesh_id(
                    action_model_input.shape[-3],
                    action_model_input.shape[-2],
                    action_model_input.shape[-1],
                    1, 1, frame_st_id, action=True,
                ).to(self.device),
                "text_emb": self.prompt_embeds.to(self.dtype).clone(),
            }
            if action_cond is not None:
                input_dict["action_res_lst"]["noisy_latents"][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict["action_res_lst"]["timesteps"][0:1] *= 0
            input_dict["action_res_lst"]["noisy_latents"][:, ~self.action_mask] *= 0

        return input_dict

    # ==================================================================
    # Action postprocessing
    # ==================================================================
    def postprocess_action(self, action):
        action = action.cpu()
        action = action[0, ..., 0]  # C, F, H
        if self.action_norm_method == "quantiles":
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 + 1e-6) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.config.used_action_channel_ids]

    # ==================================================================
    # Reset (from original server)
    # ==================================================================
    def _reset(self, prompt=None):
        logger.info("Reset.")
        self.use_cfg = (self.config.guidance_scale > 1) or (self.config.action_guidance_scale > 1)
        self.frame_st_id = 0
        self.init_latent = None

        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

        self.action_per_frame = self.config.action_per_frame
        self.height, self.width = self.config.height, self.config.width
        if self.env_type == "robotwin_tshape":
            self.latent_height = ((self.height // 16) * 3) // 2
            self.latent_width = self.width // 16
            if self.streaming_vae_half is not None:
                self.streaming_vae_half.clear_cache()
        else:
            self.latent_height = self.height // 16
            self.latent_width = self.width // 16 * len(self.config.obs_cam_keys)

        patch_size = self.config.patch_size
        latent_token_per_chunk = (
            self.config.frame_chunk_size * self.latent_height * self.latent_width
        ) // (patch_size[0] * patch_size[1] * patch_size[2])
        action_token_per_chunk = self.config.frame_chunk_size * self.action_per_frame
        # Standard denoising always uses _repeat_input_for_cfg → B=2 when CFG
        _cache_bs = 2 if self.use_cfg else 1
        self.transformer.create_empty_cache(
            self.cache_name, self.config.attn_window,
            latent_token_per_chunk, action_token_per_chunk,
            dtype=self.dtype, device=self.device,
            batch_size=_cache_bs,
        )

        self.action_mask = torch.zeros([self.config.action_dim]).bool()
        self.action_mask[self.config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(self.config.norm_stat["q01"], dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.config.norm_stat["q99"], dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.config.action_norm_method

        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.config.guidance_scale > 1,
                max_sequence_length=512,
            )

        self.exp_save_root = os.path.join(self.config.save_root, "i2va_demo")
        os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    # ==================================================================
    # Inference -- Teacher mode (original FlowMatchScheduler)
    # ==================================================================
    @torch.no_grad()
    def _infer_teacher(self, obs, frame_st_id=0):
        frame_chunk_size = self.config.frame_chunk_size

        if frame_st_id == 0:
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent

        latents = torch.randn(
            1, 48, frame_chunk_size, self.latent_height, self.latent_width,
            device=self.device, dtype=self.dtype,
        )
        actions = torch.randn(
            1, self.config.action_dim, frame_chunk_size, self.action_per_frame, 1,
            device=self.device, dtype=self.dtype,
        )

        video_inference_step = self.config.num_inference_steps
        action_inference_step = self.config.action_num_inference_steps
        video_step = self.config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode="constant", value=0)
        if video_step != -1:
            timesteps = timesteps[:video_step]
        action_timesteps = F.pad(action_timesteps, (0, 1), mode="constant", value=0)

        init_latent = self.init_latent

        # --- Video denoising loop ---
        for i, t in enumerate(tqdm(timesteps, desc=f"Video (chunk {frame_st_id})")):
            last_step = i == len(timesteps) - 1
            latent_cond = init_latent[:, :, 0:1].to(self.dtype) if frame_st_id == 0 else None
            input_dict = self._prepare_latent_input(
                latents, None, t, t, latent_cond, None, frame_st_id=frame_st_id,
            )

            video_noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=False,
            )

            if not last_step or video_step != -1:
                video_noise_pred = data_seq_to_patch(
                    self.config.patch_size, video_noise_pred,
                    frame_chunk_size, self.latent_height, self.latent_width,
                    batch_size=2 if self.use_cfg else 1,
                )
                if self.config.guidance_scale > 1:
                    video_noise_pred = (
                        video_noise_pred[1:]
                        + self.config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                    )
                else:
                    video_noise_pred = video_noise_pred[:1]
                latents = self.scheduler.step(video_noise_pred, t, latents, return_dict=False)

            latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

        # --- Action denoising loop ---
        for i, t in enumerate(tqdm(action_timesteps, desc=f"Action (chunk {frame_st_id})")):
            last_step = i == len(action_timesteps) - 1
            action_cond = torch.zeros(
                [1, self.config.action_dim, 1, self.action_per_frame, 1],
                device=self.device, dtype=self.dtype,
            ) if frame_st_id == 0 else None

            input_dict = self._prepare_latent_input(
                None, actions, t, t, None, action_cond, frame_st_id=frame_st_id,
            )
            action_noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=True,
            )

            if not last_step:
                action_noise_pred = rearrange(
                    action_noise_pred, "b (f n) c -> b c f n 1", f=frame_chunk_size
                )
                if self.config.action_guidance_scale > 1:
                    action_noise_pred = (
                        action_noise_pred[1:]
                        + self.config.action_guidance_scale
                        * (action_noise_pred[:1] - action_noise_pred[1:])
                    )
                else:
                    action_noise_pred = action_noise_pred[:1]
                actions = self.action_scheduler.step(action_noise_pred, t, actions, return_dict=False)

            actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

        actions[:, ~self.action_mask] *= 0
        actions_np = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        return actions_np, latents

    # ==================================================================
    # Inference -- Student mode (standard denoising + delta_embedder + KV cache)
    # ==================================================================
    # Standard denoising loop with FlowMatchScheduler (same structure as
    # teacher), but r_timestep is passed to the model so delta_embedder
    # contributes to time embedding.  KV cache maintained across chunks.
    @torch.no_grad()
    def _infer_student(self, obs, frame_st_id=0):
        frame_chunk_size = self.config.frame_chunk_size

        if frame_st_id == 0:
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent

        latents = torch.randn(
            1, 48, frame_chunk_size, self.latent_height, self.latent_width,
            device=self.device, dtype=self.dtype,
        )
        actions = torch.randn(
            1, self.config.action_dim, frame_chunk_size, self.action_per_frame, 1,
            device=self.device, dtype=self.dtype,
        )

        video_inference_step = self.num_steps
        action_inference_step = self.config.action_num_inference_steps
        video_step = self.config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode="constant", value=0)
        if video_step != -1:
            timesteps = timesteps[:video_step]
        action_timesteps = F.pad(action_timesteps, (0, 1), mode="constant", value=0)

        init_latent = self.init_latent

        # --- Video denoising loop (8 steps, with delta_embedder) ---
        for i, t in enumerate(tqdm(timesteps, desc=f"Video (chunk {frame_st_id})")):
            last_step = i == len(timesteps) - 1
            latent_cond = init_latent[:, :, 0:1].to(self.dtype) if frame_st_id == 0 else None

            # r_timestep: target noise level for FlowMap delta_embedder
            # Batch must match CFG (2x when guidance_scale > 1)
            r_t = timesteps[i + 1] if not last_step else 0.0
            _b = 2 if self.use_cfg else 1
            r_tensor = torch.full((_b, frame_chunk_size), r_t, device=self.device, dtype=self.dtype)
            if frame_st_id == 0:
                r_tensor[:, 0:1] = 0.0  # first frame stays clean

            input_dict = self._prepare_latent_input(
                latents, None, t, t, latent_cond, None, frame_st_id=frame_st_id,
            )

            video_noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=False,
                r_timestep=r_tensor,
            )

            if not last_step or video_step != -1:
                video_noise_pred = data_seq_to_patch(
                    self.config.patch_size, video_noise_pred,
                    frame_chunk_size, self.latent_height, self.latent_width,
                    batch_size=2 if self.use_cfg else 1,
                )
                if self.config.guidance_scale > 1:
                    video_noise_pred = (
                        video_noise_pred[1:]
                        + self.config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                    )
                else:
                    video_noise_pred = video_noise_pred[:1]
                latents = self.scheduler.step(video_noise_pred, t, latents, return_dict=False)

            latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

        # --- Action denoising loop (50 steps, no delta_embedder for action) ---
        for i, t in enumerate(tqdm(action_timesteps, desc=f"Action (chunk {frame_st_id})")):
            last_step = i == len(action_timesteps) - 1
            action_cond = torch.zeros(
                [1, self.config.action_dim, 1, self.action_per_frame, 1],
                device=self.device, dtype=self.dtype,
            ) if frame_st_id == 0 else None

            input_dict = self._prepare_latent_input(
                None, actions, t, t, None, action_cond, frame_st_id=frame_st_id,
            )
            action_noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=True,
            )

            if not last_step:
                action_noise_pred = rearrange(
                    action_noise_pred, "b (f n) c -> b c f n 1", f=frame_chunk_size
                )
                if self.config.action_guidance_scale > 1:
                    action_noise_pred = (
                        action_noise_pred[1:]
                        + self.config.action_guidance_scale
                        * (action_noise_pred[:1] - action_noise_pred[1:])
                    )
                else:
                    action_noise_pred = action_noise_pred[:1]
                actions = self.action_scheduler.step(action_noise_pred, t, actions, return_dict=False)

            actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

        actions[:, ~self.action_mask] *= 0
        actions_np = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        return actions_np, latents

    # ==================================================================
    # VAE Decode (from original server)
    # ==================================================================
    def decode_one_video(self, latents, output_type="np"):
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)

        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video

    # ==================================================================
    # Main generate pipeline
    # ==================================================================
    @torch.no_grad()
    def generate(self, output_path="demo.mp4"):
        logger.info(f"=== Starting {self.mode} video generation ===")
        self._reset(self.config.prompt)
        init_obs = self.load_init_obs()

        pred_latent_lst = []
        pred_action_lst = []

        infer_fn = self._infer_student if self.mode == "student" else self._infer_teacher

        for chunk_id in range(self.config.num_chunks_to_infer):
            logger.info(f"Chunk {chunk_id + 1}/{self.config.num_chunks_to_infer}")
            frame_st_id = chunk_id * self.config.frame_chunk_size
            actions, latents = infer_fn(init_obs, frame_st_id=frame_st_id)
            actions_t = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_action_lst.append(actions_t)

        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)

        logger.info(f"Concatenated latents: {pred_latent.shape}")
        logger.info(f"Concatenated actions: {pred_action.shape}")

        # Clean up
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half is not None:
            self.streaming_vae_half.clear_cache()
        del self.transformer
        del self.text_encoder
        torch.cuda.empty_cache()

        # Decode
        logger.info("Decoding video with VAE...")
        decoded_video = self.decode_one_video(pred_latent, "np")[0]
        logger.info(f"Decoded video shape: {decoded_video.shape}")

        # Export
        export_to_video(decoded_video, output_path, fps=10)
        logger.info(f"Video saved to: {output_path}")


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="LingBot-VA / Flash-WAM Video Generator")
    parser.add_argument(
        "--mode", type=str, choices=["teacher", "student"], default="student",
        help="Teacher: 20-step original denoising | Student: 8-step FlowMap+LoRA",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Override the prompt for video generation",
    )
    parser.add_argument(
        "--base-model-path", type=str,
        default=None,
        help="Path to base model (auto-selected by --env if not set).",
    )
    parser.add_argument(
        "--checkpoint-path", type=str,
        default="/root/nas/junjie/jj/Any_WAM/distillation_flowmap/output_libero_onpolicy/checkpoints/step_1700/online_student/transformer",
        help="Student checkpoint path (only used in student mode)",
    )
    parser.add_argument(
        "--num-steps", type=int, default=None,
        help="Override inference steps (teacher=20, student=8 by default)",
    )
    parser.add_argument(
        "--example-dir", type=str,
        default=None,  # auto-selected by --env
        help="Directory containing observation PNG files",
    )
    parser.add_argument(
        "--num-chunks", type=int, default=10,
        help="Number of chunks to infer (each chunk = 4 frames)",
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=2.0,
        help="CFG guidance scale (default: 2.0, lower = less distortion, higher = more prompt alignment)",
    )
    parser.add_argument(
        "--env", type=str, default="libero", choices=["libero", "robotwin"],
        help="Environment: libero or robotwin.",
    )
    parser.add_argument(
        "--output", type=str, default="./train_out/demo.mp4",
        help="Output video path",
    )

    args = parser.parse_args()

    # Init distributed (single GPU)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    # Free the port if already in use by trying different ports
    import socket
    port = int(os.environ["MASTER_PORT"])
    for offset in range(100):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port + offset))
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

    # Build config
    env_type = getattr(args, 'env', 'libero')
    if args.base_model_path is None:
        args.base_model_path = "/root/nas/junjie/jj/Any_WAM/checkpoints/lingbot-va-posttrain-robotwin" if env_type == "robotwin" else "/root/nas/junjie/jj/Any_WAM/checkpoints/lingbot-va-posttrain-libero"
    build_fn = build_robotwin_config if env_type == 'robotwin' else build_libero_config
    if args.example_dir is None:
        args.example_dir = "/root/nas/junjie/jj/Any_WAM/lingbot-va/example/robotwin" if env_type == "robotwin" else "/root/nas/junjie/jj/Any_WAM/lingbot-va/example/libero"
    config = build_fn(
        base_model_path=args.base_model_path,
        example_dir=args.example_dir,
        num_chunks=args.num_chunks,
        cfg_scale=args.cfg_scale,
    )
    config.save_root = os.path.dirname(args.output) or "."
    if args.prompt is not None:
        config.prompt = args.prompt

    # Set num steps
    if args.mode == "teacher":
        if args.num_steps is not None:
            config.num_inference_steps = args.num_steps
    else:
        num_steps = args.num_steps or (25 if env_type == "robotwin" else 8)

    # Override checkpoint for student
    checkpoint_path = args.checkpoint_path if args.mode == "student" else None

    logger.info(f"Base model: {config.wan22_pretrained_model_name_or_path}")
    logger.info(f"Example dir: {config.input_img_path}")
    logger.info(f"Output: {args.output}")

    # Create generator and run
    gen = VideoGenerator(
        config=config,
        mode=args.mode,
        checkpoint_path=checkpoint_path,
        num_steps=args.num_steps,
    )
    gen.generate(output_path=args.output)
    logger.info("Done!")


if __name__ == "__main__":
    main()
