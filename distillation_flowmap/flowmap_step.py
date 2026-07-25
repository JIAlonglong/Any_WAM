"""
Flow Map 蒸馏的训练步实现（FlowMapStepMixin）。

这个文件实现了 AnyFlow 风格的流映射蒸馏核心逻辑：
  1. 混合时间步采样：50% 扩散目标 + 25% 一致性目标 + 25% 流映射目标
  2. 教师模型通过 CFG 前向生成 v-prediction（伪 ground truth）
  3. 中心差分法计算 dF/dt（flow map 的数值梯度）
  4. 训练目标 = v_pred - (t-r) * dF/dt
  5. 学生模型在 r 处预测，与目标计算 MSE 损失

与原始 StepMixin（LCM 蒸馏）的核心区别：
  - 原始 LCM：r = t + k（固定跳步），目标来自 EMA 学生
  - FlowMap：r 从 [0, t] 中随机采样，目标来自教师 v-prediction + 中心差分修正
  - FlowMap 支持三种模式混合：扩散（r=t）、一致性（r=0）、流映射（r<t）

数学推导（Flow Map 目标）：
  教师在 t 处的 v-prediction: v_pred = F(x_t, t)
  flow map 的梯度: dF/dt ≈ [F(x_{t+ε}, t+ε) - F(x_{t-ε}, t-ε)] / (2ε)
  训练目标: target = v_pred - (t - r) * dF/dt
  当 r=t 时: target = v_pred（标准扩散目标）
  当 r=0 时: target = v_pred - t * dF/dt（一致性目标）
  当 r<t 时: target = v_pred - (t-r) * dF/dt（流映射目标）

参考实现：
  - AnyFlow: far/trainers/trainer_far_wan_anyflow_pretrain.py
  - Flash-WAM LCM: distillation/step.py
"""

import contextlib
import math
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
import contextlib


def _to_regular_tensor(t):
    """Convert DTensor to regular Tensor, preserving gradient flow via autograd."""
    if hasattr(t, 'to_local'):
        return t.to_local()  # autograd-aware conversion
    return t


def _downsample_action_grid_id(grid_id, action_latents, factor):
    """Downsample action RoPE grid IDs along the action frame dimension.

    action grid_id layout is [B, 4, F * N], matching actions [B, C, F, N, 1].
    A plain grid_id[:, :, ::factor] samples one token per frame; this keeps all
    N action tokens for every selected frame.
    """
    if grid_id is None:
        return None
    if factor == 1:
        return grid_id
    B, _, F_action, N_action, W_action = action_latents.shape
    if W_action != 1:
        raise ValueError(f"Expected action width 1, got {W_action}")
    return grid_id.reshape(B, 4, F_action, N_action, W_action)[:, :, ::factor].reshape(B, 4, -1)


def _same_state_velocity_loss(
    student_v,
    teacher_v,
    sample_weight,
    transition_loss_type="huber",
    transition_huber_c=1e-3,
):
    """Weighted per-sample velocity loss at the student-induced endpoint."""
    diff = student_v.float() - teacher_v.detach().float()
    if transition_loss_type == "huber":
        abs_diff = diff.abs()
        per_token = torch.where(
            abs_diff < transition_huber_c,
            0.5 * diff ** 2,
            transition_huber_c * (abs_diff - 0.5 * transition_huber_c),
        )
    else:
        per_token = diff ** 2
    per_sample = per_token.mean(dim=[1, 2, 3, 4])
    return (per_sample * sample_weight.to(per_sample.device)).mean()


def _synchronized_nonfinite_decision(*, loss, local_flags, device):
    """Agree on named loss failures before any rank enters backward."""
    flags = {
        origin: bool(local_flags.get(origin, False))
        for origin in NONFINITE_ORIGINS
    }
    if not bool(torch.isfinite(loss.detach()).all().item()) and not any(
        flags.values()
    ):
        flags["teacher_or_gt"] = True
    synchronized = reduce_nonfinite_origins(flags, device=device)
    return not any(synchronized.values()), synchronized
from einops import rearrange

from utils import data_seq_to_patch, logger
from distillation.consistency import scalings_for_boundary_conditions
from distillation_flowmap.cosmos_future_aux import select_official_future_camera_images
from distillation_flowmap.cosmos_policy_adapter import (
    compute_masked_action_stats,
    cosmos_actions_to_flowmap_x0,
    cosmos_latent_micro_step_index,
    cosmos_latent_should_request_cdiff,
)
from distillation_flowmap.cosmos_training_contract import (
    pack_actions_for_downsample,
)
from distillation_flowmap.kto_reweighting import (
    compute_normalized_focal_weights,
    piecewise_linear_scale,
)
from distillation_flowmap.opd_loss_composition import (
    compose_explicit_hybrid_opd,
)
from distillation_flowmap.opd_rollout_grad import (
    SUPPORTED_ROLLOUT_GRAD_MODES,
    rollout_step_requires_grad,
)
from distillation_flowmap.distributed_safety import (
    NONFINITE_ORIGINS,
    all_ranks_finite,
    reduce_nonfinite_origins,
)
from distillation_flowmap.danceopd_query import (
    aligned_anchor_mse,
    build_shifted_terminal_path,
    denoised_endpoint_mse,
    direct_velocity_mse,
    masked_video_velocity_mse,
    sample_nonterminal_semantic_query_indices,
    sample_endpoint_sigmas,
    sample_low_noise_query_indices,
    sample_semantic_query_indices,
    sample_uniform_rollout_step_pair,
    select_per_sample_trajectory_state,
)
from distillation_flowmap.cosmos_progressive_opd import (
    apply_full_endpoint_focus,
    build_cosmos_teacher_window_path,
    broadcast_joint_action_timesteps,
    center_spatial_crop_slices,
    compose_cosmos_endpoint_loss,
    constrain_cosmos_teacher_timestep_pair,
    rollout_velocity_field,
)
from distillation_flowmap.cosmos_deployment_rollout import (
    deployment_endpoint_losses,
)
from distillation_flowmap.numerical_contracts import (
    validate_terminal_prior_sources,
)
from distillation_flowmap.mechanism_diagnostics import (
    compute_mechanism_metric_samples,
    pack_finite_metric_stats,
)


def _validate_terminal_prior_pair(
    *,
    config,
    video_actual,
    video_expected,
    video_source_dtype,
    action_actual,
    action_expected,
    action_source_dtype,
    reference,
    error_context,
):
    """Execute the shared video/action terminal-prior production boundary."""
    validation = validate_terminal_prior_sources(
        {
            "video": (video_actual, video_expected, video_source_dtype),
            "action": (action_actual, action_expected, action_source_dtype),
        },
        config=config,
        warning_callback=(
            logger.warning if getattr(config, "rank", 0) == 0 else None
        ),
        error_context=error_context,
    )
    return validation, validation.diagnostic_tensors(reference)


class FlowMapStepMixin:
    """
    Flow Map 蒸馏的训练步 Mixin 类。

    提供以下核心方法：
      - sample_timestep_mixed: 混合时间步采样（扩散/一致性/流映射）
      - compute_central_difference: 中心差分法计算 flow map 梯度
      - _extract_video_v / _extract_action_v: 从模型输出提取 5D 张量
      - _consistency_function: 一致性函数（边界条件缩放）
      - _train_step: 完整的 Flow Map 训练步

    使用方式：
      class FlowMapDistiller(DataMixin, FlowMapStepMixin):
          ...
    """


    @property
    def _teacher_model(self):
        """Return the default frozen teacher for backward compatibility."""
        teacher = getattr(self, '_teacher_nofsdp', None)
        if teacher is not None:
            return teacher
        teacher = getattr(self, 'teacher', None)
        if teacher is None:
            raise RuntimeError("No teacher model is available for this FlowMapDistiller.")
        return teacher

    @property
    def _action_teacher_model(self):
        """Return the teacher that supplies action targets."""
        teacher = getattr(self, '_teacher_nofsdp', None)
        if teacher is not None:
            return teacher
        teacher = getattr(self, 'teacher', None)
        if teacher is None:
            raise RuntimeError("No action teacher model is available for this FlowMapDistiller.")
        return teacher

    @property
    def _video_teacher_model(self):
        """Return the teacher that supplies WanVA latent video targets."""
        video_teacher = getattr(self, '_video_teacher_nofsdp', None)
        if video_teacher is not None:
            return video_teacher
        return self._teacher_model

    def _cosmos_action_x0_target(self, action_dict, raw_batch, downsample_factor):
        """Return downsampled raw Cosmos action x0 when available."""
        if not getattr(self, 'is_cosmos_policy_teacher', False):
            return None
        teacher = self._action_teacher_model
        if not hasattr(teacher, 'action_target_x0'):
            return None
        if not getattr(teacher, 'raw_inference_enabled', False):
            return None
        target = teacher.action_target_x0(action_dict, raw_batch=raw_batch)
        if target is None:
            return None
        if target.ndim != 5:
            raise ValueError(
                f"Cosmos action x0 target must be 5D [B,C,F,N,1], got {tuple(target.shape)}"
            )
        if downsample_factor != 1:
            target = target[:, :, ::downsample_factor]
        return target.contiguous()

    def _cosmos_future_prediction_items(self, future_predictions, batch_size):
        """Return one official Cosmos future-prediction mapping per batch item."""
        if future_predictions is None:
            raise RuntimeError(
                "cfg.cosmos_video_target=True requires Cosmos raw inference to "
                "return future_image_predictions."
            )
        if isinstance(future_predictions, (list, tuple)):
            if len(future_predictions) != batch_size:
                raise ValueError(
                    "Cosmos future prediction batch size mismatch: "
                    f"got {len(future_predictions)}, expected {batch_size}"
                )
            return list(future_predictions)
        if not isinstance(future_predictions, dict):
            raise TypeError(
                "future_image_predictions must be a mapping or batch list of mappings, "
                f"got {type(future_predictions)!r}"
            )
        if batch_size == 1:
            return [future_predictions]

        items = []
        for batch_idx in range(batch_size):
            item = {}
            for key, value in future_predictions.items():
                if torch.is_tensor(value):
                    if value.ndim >= 4 and value.shape[0] == batch_size:
                        item[key] = value[batch_idx]
                    else:
                        item[key] = value
                elif hasattr(value, "shape"):
                    if len(value.shape) >= 4 and value.shape[0] == batch_size:
                        item[key] = value[batch_idx]
                    else:
                        item[key] = value
                elif isinstance(value, (list, tuple)) and len(value) == batch_size:
                    item[key] = value[batch_idx]
                else:
                    item[key] = value
            items.append(item)
        return items

    def _future_image_to_video_tensor(self, image):
        """Convert one official Cosmos future image/video to [1,3,T,H,W]."""
        tensor = image if torch.is_tensor(image) else torch.as_tensor(image)
        tensor = tensor.detach().to(torch.float32)

        if tensor.ndim == 5:
            if tensor.shape[0] != 1:
                raise ValueError(
                    "Per-sample future image tensor must have batch size 1 when 5D, "
                    f"got shape={tuple(tensor.shape)}"
                )
            tensor = tensor[0]

        if tensor.ndim == 3:
            if tensor.shape[-1] in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1).unsqueeze(1)
            elif tensor.shape[0] in (1, 3, 4):
                tensor = tensor.unsqueeze(1)
            else:
                raise ValueError(f"Cannot infer future image layout from shape={tuple(tensor.shape)}")
        elif tensor.ndim == 4:
            if tensor.shape[-1] in (1, 3, 4):        # [T,H,W,C]
                tensor = tensor.permute(3, 0, 1, 2)
            elif tensor.shape[1] in (1, 3, 4):       # [T,C,H,W]
                tensor = tensor.permute(1, 0, 2, 3)
            elif tensor.shape[0] in (1, 3, 4):       # [C,T,H,W]
                tensor = tensor
            else:
                raise ValueError(f"Cannot infer future video layout from shape={tuple(tensor.shape)}")
        else:
            raise ValueError(
                f"Expected future image/video tensor with 3-5 dims, got shape={tuple(tensor.shape)}"
            )

        if tensor.shape[0] == 1:
            tensor = tensor.expand(3, -1, -1, -1)
        elif tensor.shape[0] == 4:
            tensor = tensor[:3]
        elif tensor.shape[0] != 3:
            raise ValueError(f"Expected 1/3/4 image channels, got shape={tuple(tensor.shape)}")

        if tensor.numel() > 0:
            if tensor.min() < -0.1:
                tensor = (tensor + 1.0) * 127.5
            elif tensor.max() <= 2.0:
                tensor = tensor * 255.0
        tensor = tensor.clamp(0.0, 255.0).unsqueeze(0)  # [1,C,T,H,W]

        target_hw = (
            int(getattr(self.config, "height", tensor.shape[-2])),
            int(getattr(self.config, "width", tensor.shape[-1])),
        )
        if tensor.shape[-2:] != target_hw:
            bsz, channels, frames, height, width = tensor.shape
            frame_tensor = tensor.permute(0, 2, 1, 3, 4).reshape(
                bsz * frames, channels, height, width)
            frame_tensor = F.interpolate(
                frame_tensor, size=target_hw, mode="bilinear", align_corners=False)
            tensor = frame_tensor.reshape(
                bsz, frames, channels, target_hw[0], target_hw[1]
            ).permute(0, 2, 1, 3, 4).contiguous()
        return tensor

    def _pad_future_videos_to_common_time(self, videos):
        max_frames = max(video.shape[2] for video in videos)
        out = []
        for video in videos:
            if video.shape[2] < max_frames:
                pad = video[:, :, -1:].expand(
                    -1, -1, max_frames - video.shape[2], -1, -1)
                video = torch.cat([video, pad], dim=2)
            elif video.shape[2] > max_frames:
                video = video[:, :, :max_frames]
            out.append(video)
        return out

    @torch.no_grad()
    def _encode_cosmos_future_video_latents(self, future_predictions, ref_shape):
        """Encode official Cosmos future images into WanVA latent x0 layout."""
        if getattr(self, "cosmos_video_streaming_vae", None) is None:
            raise RuntimeError(
                "cfg.cosmos_video_target=True requires FlowMapDistiller to load "
                "self.cosmos_video_streaming_vae."
            )
        if str(getattr(self.config, "env_type", "")).lower() == "robotwin_tshape":
            raise NotImplementedError(
                "Cosmos future video targets for robotwin_tshape need the "
                "RobotWin half-resolution VAE layout; LIBERO is supported."
            )

        batch_size = int(ref_shape[0])
        cam_keys = list(getattr(self.config, "obs_cam_keys", []))
        if not cam_keys:
            raise ValueError("cfg.obs_cam_keys is required for Cosmos future video targets.")

        prediction_items = self._cosmos_future_prediction_items(
            future_predictions, batch_size)
        videos = []
        for prediction in prediction_items:
            if not isinstance(prediction, dict):
                raise TypeError(
                    "Each Cosmos future prediction must be a mapping, "
                    f"got {type(prediction)!r}"
                )
            for image in select_official_future_camera_images(prediction, cam_keys):
                videos.append(self._future_image_to_video_tensor(image))

        videos = self._pad_future_videos_to_common_time(videos)
        videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0

        vae_device = next(self.cosmos_video_vae.parameters()).device
        self.cosmos_video_streaming_vae.clear_cache()
        enc_out = self.cosmos_video_streaming_vae.encode_chunk(
            videos.to(device=vae_device, dtype=self.dtype)
        )
        mu, _ = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(
            self.cosmos_video_vae.config.latents_mean,
            device=mu.device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(
            self.cosmos_video_vae.config.latents_std,
            device=mu.device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1, 1)
        mu_norm = ((mu.float() - latents_mean) / latents_std).to(mu.dtype)

        num_cams = len(cam_keys)
        if mu_norm.shape[0] != batch_size * num_cams:
            raise ValueError(
                "Encoded Cosmos future latent batch mismatch: "
                f"got {mu_norm.shape[0]}, expected {batch_size * num_cams}"
            )
        mu_norm = mu_norm.reshape(batch_size, num_cams, *mu_norm.shape[1:])
        per_sample = [
            torch.cat([mu_norm[b, cam_idx] for cam_idx in range(num_cams)], dim=-1)
            for b in range(batch_size)
        ]
        latents = torch.stack(per_sample, dim=0)

        target_channels, target_frames, target_height, target_width = map(int, ref_shape[1:])
        if latents.shape[1] != target_channels:
            raise ValueError(
                f"Cosmos future latent channel mismatch: got {latents.shape[1]}, "
                f"expected {target_channels}"
            )
        if latents.shape[3] != target_height or latents.shape[4] != target_width:
            raise ValueError(
                "Cosmos future latent spatial mismatch: "
                f"got {tuple(latents.shape[3:])}, expected {(target_height, target_width)}"
            )
        if latents.shape[2] < target_frames:
            pad = latents[:, :, -1:].expand(
                -1, -1, target_frames - latents.shape[2], -1, -1)
            latents = torch.cat([latents, pad], dim=2)
        elif latents.shape[2] > target_frames:
            latents = latents[:, :, :target_frames]
        return latents.contiguous()

    # ==================================================================
    # 混合时间步采样：扩散目标 + 一致性目标 + 流映射目标
    # ==================================================================
    def _sample_adjacent_grid_timesteps(self, batch_size, num_frames, dtype, device):
        """Sample only neighboring edges on a fixed denoising grid."""
        grid = torch.as_tensor(
            getattr(self.config, "flowmap_adjacent_grid", [1000, 750, 500, 250, 0]),
            dtype=dtype,
            device=device,
        )
        if grid.ndim != 1 or grid.numel() < 2:
            raise ValueError("flowmap_adjacent_grid must contain at least two timesteps.")
        if not torch.all(grid[:-1] > grid[1:]):
            raise ValueError(
                "flowmap_adjacent_grid must be strictly descending, "
                f"got {grid.detach().cpu().tolist()}"
            )

        edge_idx = torch.randint(0, grid.numel() - 1, (batch_size,), device=device)
        t = grid[edge_idx].unsqueeze(1).expand(-1, num_frames)
        r = grid[edge_idx + 1].unsqueeze(1).expand(-1, num_frames)
        is_diffusion = torch.zeros(batch_size, dtype=torch.bool, device=device)
        return t, r, is_diffusion

    def sample_timestep_mixed(
        self, batch_size, num_frames, dtype, device, scheduler=None, pair_mode=None,
    ):
        """
        混合时间步采样，实现三种目标的随机切换。

        采样逻辑（参考 AnyFlow 的 sample_timestep）：
          1. 采样两个随机数 t_1, t_2 ∈ [0, 1]
          2. t = max(t_1, t_2), r = min(t_1, t_2)，确保 t >= r
          3. 根据配置比例决定 r 的取值：
             - 50% 概率：r = t（扩散模式，标准 FlowMatch 去噪）
             - 25% 概率：r = 0（一致性模式，LCM 端点映射）
             - 25% 概率：保持 r < t（流映射模式，AnyFlow 核心）

        参数:
            batch_size: 批次大小
            num_frames: 时间维度 T（帧数）
            dtype:      数据类型（通常为 torch.float32）
            device:     设备（GPU）
            scheduler:  使用哪个 FlowMatchScheduler 应用 SNR shift；
                        None 时使用视频 scheduler。

        返回:
            t:           主时间步 [batch_size, num_frames]，已应用 SNR shift 并缩放到原始时间步（0~num_train_timesteps）
            r:           参考时间步 [batch_size, num_frames]，已应用 SNR shift 并缩放到原始时间步（0~t）
            is_diffusion: 布尔掩码 [batch_size]，标记哪些样本是扩散目标

        注意：
          - 采样时 t 和 r 是归一化 sigma（0~1），per-sample [B]
          - 采样后扩展为 per-frame [B, T]，与 AnyFlow 参考实现一致
          - 返回前会应用 SNR shift（与 FlowMatchScheduler 一致）并乘以 num_train_timesteps
          - 返回值已是原始时间步，无需再乘以 num_train_timesteps
        """
        pair_mode = str(pair_mode or getattr(
            self.config, "flowmap_pair_mode", "arbitrary")).lower()
        if pair_mode in ("adjacent_grid", "adjacent", "local"):
            return self._sample_adjacent_grid_timesteps(
                batch_size, num_frames, dtype, device)
        if pair_mode not in ("arbitrary", "mixed", "anyflow", "flowmap"):
            raise ValueError(
                f"Unsupported pair_mode={pair_mode!r}; expected arbitrary or adjacent_grid."
            )

        # 步骤 1：采样两个均匀随机数（per-sample）
        t_1 = torch.rand(batch_size, dtype=dtype, device=device)
        t_2 = torch.rand(batch_size, dtype=dtype, device=device)

        # 步骤 2：确保 t >= r
        t = torch.maximum(t_1, t_2)
        r = torch.minimum(t_1, t_2)

        # 步骤 3：按样本随机选择目标模式
        total_ratio = self.diffusion_ratio + self.consistency_ratio + self.flowmap_ratio
        if total_ratio <= 0:
            raise ValueError("diffusion_ratio + consistency_ratio + flowmap_ratio must be positive.")
        diffusion_ratio = self.diffusion_ratio / total_ratio
        consistency_ratio = self.consistency_ratio / total_ratio
        mode_rand = torch.rand(batch_size, dtype=dtype, device=device)

        is_diffusion = mode_rand < diffusion_ratio
        is_consistency = (
            (mode_rand >= diffusion_ratio) &
            (mode_rand < diffusion_ratio + consistency_ratio)
        )

        # 扩散：r=t；一致性：r=0；剩余保持 r<t 的 flowmap 模式
        r = torch.where(is_diffusion, t, r)
        r = torch.where(is_consistency, torch.zeros_like(r), r)

        # 步骤 4：扩展为 per-frame [B, T]（与 AnyFlow 参考实现一致）
        t = t.unsqueeze(1).expand(-1, num_frames)
        r = r.unsqueeze(1).expand(-1, num_frames)

        # 应用对应分支的 SNR shift 并转换为原始时间步。
        # LIBERO action uses action_snr_shift=0.05, not the video shift=5.0.
        scheduler = scheduler or self.train_scheduler_latent
        t = scheduler.apply_shift(t) * self.config.num_train_timesteps
        r = scheduler.apply_shift(r) * self.config.num_train_timesteps
        return t, r, is_diffusion

    def sample_cosmos_latent_timestep_mixed(
        self, batch_size, num_frames, dtype, device,
    ):
        """Sample FlowMap t/r for Cosmos latent teacher queries.

        Cosmos Policy's EDM teacher is reliable in sigma=[4,80], which maps to
        FlowUniPC t=sigma/(1+sigma). We keep student timestep embeddings on the
        existing 0..num_train_timesteps scale, but central-diff math uses the
        normalized t/r values returned here.
        """
        t_min = float(getattr(self.config, "cosmos_latent_t_min", 4.0 / 5.0))
        t_max = float(getattr(self.config, "cosmos_latent_t_max", 80.0 / 81.0))
        if not (0.0 <= t_min < t_max < 1.0):
            raise ValueError(
                "Cosmos latent t range must satisfy 0 <= t_min < t_max < 1, "
                f"got t_min={t_min}, t_max={t_max}"
            )

        t_1 = torch.rand(batch_size, dtype=dtype, device=device)
        t_2 = torch.rand(batch_size, dtype=dtype, device=device)
        hi = torch.maximum(t_1, t_2)
        lo = torch.minimum(t_1, t_2)
        t_norm = t_min + (t_max - t_min) * hi
        ratio = torch.where(hi > 1e-6, lo / hi.clamp(min=1e-6), torch.zeros_like(lo))
        r_norm = t_norm * ratio

        total_ratio = self.diffusion_ratio + self.consistency_ratio + self.flowmap_ratio
        if total_ratio <= 0:
            raise ValueError("diffusion_ratio + consistency_ratio + flowmap_ratio must be positive.")
        diffusion_ratio = self.diffusion_ratio / total_ratio
        consistency_ratio = self.consistency_ratio / total_ratio
        mode_rand = torch.rand(batch_size, dtype=dtype, device=device)

        is_diffusion = mode_rand < diffusion_ratio
        is_consistency = (
            (mode_rand >= diffusion_ratio) &
            (mode_rand < diffusion_ratio + consistency_ratio)
        )
        r_norm = torch.where(is_diffusion, t_norm, r_norm)
        r_norm = torch.where(is_consistency, torch.zeros_like(r_norm), r_norm)

        t_norm = t_norm.unsqueeze(1).expand(-1, num_frames)
        r_norm = r_norm.unsqueeze(1).expand(-1, num_frames)
        return (
            t_norm * self.config.num_train_timesteps,
            r_norm * self.config.num_train_timesteps,
            t_norm,
            r_norm,
            is_diffusion,
        )

    def _apply_opd_low_noise_query_bias(self, query_t, query_r):
        """Bias OPD query states toward low-noise timesteps, following DanceOPD."""
        bias = str(getattr(self.config, "opd_query_bias", "none")).lower()
        if bias in ("", "none", "uniform", "off", "false"):
            return query_r
        if bias not in ("low_t", "low_noise", "danceopd"):
            raise ValueError(f"Unsupported opd_query_bias={bias!r}")

        ratio = max(0.0, min(1.0, float(getattr(self.config, "opd_query_bias_ratio", 1.0))))
        if ratio <= 0:
            return query_r

        alpha = float(getattr(self.config, "opd_low_noise_alpha", 5.0))
        beta = float(getattr(self.config, "opd_low_noise_beta", 2.0))
        max_sigma = max(0.0, min(1.0, float(getattr(self.config, "opd_low_noise_max_sigma", 0.25))))
        B = query_r.shape[0]
        device = query_r.device
        dtype = query_r.dtype
        select = torch.rand(B, device=device, dtype=dtype) < ratio
        if not select.any():
            return query_r

        dist_beta = torch.distributions.Beta(
            torch.tensor(alpha, device=device, dtype=dtype),
            torch.tensor(beta, device=device, dtype=dtype),
        )
        low_sigma = (1.0 - dist_beta.sample((B,))) * max_sigma
        low_r = low_sigma.unsqueeze(1).expand_as(query_r) * self.config.num_train_timesteps
        low_r = torch.minimum(low_r, query_t)
        select = select.unsqueeze(1).expand_as(query_r)
        return torch.where(select, low_r, query_r)

    def _build_timestep_path(self, timesteps, target_r, num_steps):
        """Linearly interpolate per-sample timestep grids from t to r."""
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        alpha = torch.linspace(
            0.0, 1.0, num_steps + 1, device=timesteps.device, dtype=timesteps.dtype
        )
        delta = target_r - timesteps
        return timesteps.unsqueeze(0) + delta.unsqueeze(0) * alpha.view(-1, 1, 1)

    def _timestep_to_sigma_5d(self, timestep):
        """Convert timestep tensor [B, F] to sigma tensor [B, 1, F, 1, 1]."""
        return timestep[:, None, :, None, None] / self.config.num_train_timesteps

    # ==================================================================
    # 中心差分法计算 flow map 梯度 dF/dt
    # ==================================================================
    # NOTE: kept for reference; not currently used
    @torch.no_grad()
    def compute_central_difference(
        self, video_v_cfg_func, noisy_latents, latents, noise, t, r, eps, mask=None
    ):
        """
        中心差分法计算 flow map 的数值梯度 dF/dt。

        数学原理：
          dF/dt ≈ [F(x_{t+ε}, t+ε) - F(x_{t-ε}, t-ε)] / (2ε)

          其中 F(x_t, t) 是教师模型的 CFG v-prediction。

        参数:
            video_v_cfg_func: 教师 CFG 前向函数，接受 (noisy_latents, timesteps) 返回 v-prediction
            noisy_latents:    当前噪声样本 x_t [B, C, F, H, W]
            latents:          干净样本 x_0 [B, C, F, H, W]
            noise:            随机噪声 ε [B, C, F, H, W]
            t:                主时间步 [B]（原始时间步，0~1000）
            r:                参考时间步 [B]（原始时间步，0~1000）
            eps:              扰动步长（标量，通常为 1.0）
            mask:             可选的 batch 粒度掩码 [B]，非零样本参与差分计算

        返回:
            dF_dt: flow map 梯度 [B, C, F, H, W]

        实现细节：
          - v_pred = noise - latents 是 FlowMatch 的训练目标方向
          - x_{t+ε} = x_t + v_pred * (ε / num_train_timesteps)
          - x_{t-ε} = x_t - v_pred * (ε / num_train_timesteps)
          - 注意：时间步是原始值（0~1000），除以 num_train_timesteps 得到归一化值
        """
        # 计算 FlowMatch 训练目标方向：v = noise - x_0
        # 这是从 x_0 到 noise 的方向，即"加噪方向"
        v_pred = noise - latents

        # t_plus_epsilon: 在 t+ε 处的前向
        # x_{t+ε} = x_t + v_pred * (ε / num_train_timesteps)
        # 注意：这里用 v_pred（加噪方向）来推进噪声样本
        # clamp 防止时间步越界
        t_plus = (t + eps).clamp(max=self.config.num_train_timesteps)
        noisy_latents_plus = noisy_latents + v_pred * (eps / self.config.num_train_timesteps)
        noise_pred_plus = video_v_cfg_func(noisy_latents_plus, t_plus)

        # t_minus_epsilon: 在 t-ε 处的前向
        # x_{t-ε} = x_t - v_pred * (ε / num_train_timesteps)
        # clamp 防止时间步越界
        t_minus = (t - eps).clamp(min=0)
        noisy_latents_minus = noisy_latents - v_pred * (eps / self.config.num_train_timesteps)
        noise_pred_minus = video_v_cfg_func(noisy_latents_minus, t_minus)

        # 中心差分：dF/dt = (v_{t+ε} - v_{t-ε}) / (2ε)
        dF_dt = (noise_pred_plus - noise_pred_minus) / (2 * eps)

        # 如果提供了 batch 粒度掩码，对不参与的样本置零
        if mask is not None:
            dF_dt = dF_dt * mask.float().view(-1, 1, 1, 1, 1)
        return dF_dt

    # ==================================================================
    @torch.no_grad()
    def compute_central_difference_merged(
        self, input_dict, empty_emb, cfg_scale, noisy_latents, latents, noise, t, r, eps,
        ref_shape, mask=None
    ):
        """
        合并版中心差分：将 t+ε 和 t-ε 两次前向合并为一次 4B 前向。

        优化原理：
          原始实现需要 2 次教师前向（每次 2B），共 4B 计算。
          本方法将 t+ε 和 t-ε 的 cond+uncond 合并为一次 4B 前向，
          减少 1 次教师前向（约 33% 的教师计算）。

        参数:
            input_dict:      标准 input_dict（含 latent_dict、action_dict 等）
            empty_emb:       空文本嵌入 [1, seq_len, dim]
            cfg_scale:       CFG 引导强度
            noisy_latents:   当前噪声样本 x_t [B, C, F, H, W]
            latents:         干净样本 x_0 [B, C, F, H, W]
            noise:           随机噪声 ε [B, C, F, H, W]
            t:               主时间步 [B, T]（原始时间步，0~1000）
            r:               参考时间步 [B, T]（原始时间步，0~1000）
            eps:             扰动步长（标量，通常为 1.0）
            ref_shape:       参考形状 [B, C, F, H, W]，用于提取视频张量
            mask:            可选的 batch 粒度掩码 [B]，非零样本参与差分计算

        返回:
            dF_dt: flow map 梯度 [B, C, F, H, W]
        """
        B = noisy_latents.shape[0]

        # 计算 FlowMatch 训练目标方向：v = noise - x_0
        v_pred = noise - latents

        # 计算 t+ε 和 t-ε 的噪声样本和时间步
        t_plus = (t + eps).clamp(max=self.config.num_train_timesteps)
        noisy_latents_plus = noisy_latents + v_pred * (eps / self.config.num_train_timesteps)

        t_minus = (t - eps).clamp(min=0)
        noisy_latents_minus = noisy_latents - v_pred * (eps / self.config.num_train_timesteps)

        # 构建两次 2B 前向的 input_dict（t+eps 和 t-eps 各一次 2B CFG 前向）
        # 注意：不能用 4B 一次前向，因为 4B 的序列太长会 OOM。
        # 同时 latent_dict 和 action_dict 的 batch 维必须一致，
        # 因为 init_mask 使用 latent_shape[0] 作为 B 同时构建 latent 和 action 的 seq_id。
        ld = input_dict['latent_dict']
        ad = input_dict['action_dict']

        def _build_2b_input(noisy_latents_2b, latents_2b, timesteps_2b):
            """构建一次 2B CFG 前向的 input_dict（cond + uncond）。"""
            text_emb = ld['text_emb']
            empty_expanded = empty_emb.expand(B, -1, -1)
            doubled = {
                'latent_dict': {
                    'noisy_latents': torch.cat([noisy_latents_2b, noisy_latents_2b], dim=0),
                    'latent':        torch.cat([latents_2b, latents_2b], dim=0),
                    'timesteps':     torch.cat([timesteps_2b, timesteps_2b], dim=0),
                    'cond_timesteps':torch.cat([ld['cond_timesteps'], ld['cond_timesteps']], dim=0),
                    'text_emb':      torch.cat([text_emb, empty_expanded], dim=0),
                },
                'action_dict': {
                    'noisy_latents': torch.cat([ad['noisy_latents'], ad['noisy_latents']], dim=0),
                    'latent':        torch.cat([ad['latent'], ad['latent']], dim=0),
                    'timesteps':     torch.cat([ad['timesteps'], ad['timesteps']], dim=0),
                    'cond_timesteps':torch.cat([ad['cond_timesteps'], ad['cond_timesteps']], dim=0),
                    'text_emb':      torch.cat([ad['text_emb'], empty_expanded], dim=0),
                },
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
            if 'grid_id' in ld and ld['grid_id'] is not None:
                doubled['latent_dict']['grid_id'] = torch.cat([ld['grid_id'], ld['grid_id']], dim=0)
            if 'grid_id' in ad and ad['grid_id'] is not None:
                doubled['action_dict']['grid_id'] = torch.cat([ad['grid_id'], ad['grid_id']], dim=0)
            return doubled

        # 保存/恢复 FlexAttnFunc 的 mask（避免污染学生模型的 mask）
        from modules.model import FlexAttnFunc
        saved_attn_mask = FlexAttnFunc.attention_mask
        saved_cross_mask = FlexAttnFunc.cross_attention_mask
        try:
            FlexAttnFunc.attention_mask = None
            FlexAttnFunc.cross_attention_mask = None

            # 前向 1: t+eps, cond+uncond (2B)
            input_plus = _build_2b_input(noisy_latents_plus, latents, t_plus)
            v_plus_all, _ = self._video_teacher_model(input_plus, train_mode=True)

            # 恢复 mask（teacher forward 会更新 mask，需要在下次前向前重置）
            FlexAttnFunc.attention_mask = None
            FlexAttnFunc.cross_attention_mask = None

            # 前向 2: t-eps, cond+uncond (2B)
            input_minus = _build_2b_input(noisy_latents_minus, latents, t_minus)
            v_minus_all, _ = self._video_teacher_model(input_minus, train_mode=True)
        finally:
            FlexAttnFunc.attention_mask = saved_attn_mask
            FlexAttnFunc.cross_attention_mask = saved_cross_mask

        # 拆分：前半 = 有条件，后半 = 无条件
        v_cond_plus, v_uncond_plus = v_plus_all.chunk(2, dim=0)
        v_cond_minus, v_uncond_minus = v_minus_all.chunk(2, dim=0)

        # CFG 组合
        v_cfg_plus = v_uncond_plus + cfg_scale * (v_cond_plus - v_uncond_plus)
        v_cfg_minus = v_uncond_minus + cfg_scale * (v_cond_minus - v_uncond_minus)

        # 提取 5D 视频张量
        video_v_plus = self._extract_video_v(v_cfg_plus, ref_shape, B)
        video_v_minus = self._extract_video_v(v_cfg_minus, ref_shape, B)

        # 中心差分：dF/dt = (v_{t+ε} - v_{t-ε}) / (2ε)
        dF_dt = (video_v_plus - video_v_minus) / (2 * eps)

        # 如果提供了 batch 粒度掩码，对不参与的样本置零
        if mask is not None:
            dF_dt = dF_dt * mask.float().view(-1, 1, 1, 1, 1)
        return dF_dt

    # ==================================================================
    # 合并 CFG 前向 + 中心差分为 2 次 3B 前向（替代 3 次 2B 前向）
    # ==================================================================
    @torch.no_grad()
    def _merged_cfg_central_diff(
        self, input_dict, empty_emb, cfg_scale,
        noisy_latents_plus, noisy_latents_minus,
        t_plus, t_minus, latents,
        action_noisy_plus=None, action_noisy_minus=None,
        action_t_plus=None, action_t_minus=None
    ):
        """
        合并 CFG 前向与中心差分为 2 次 3B 前向（替代 3 次 2B 前向）。

        非扩散样本需要：
          原方案: 1×2B(CFG at t) + 1×2B(中心差分 at t+ε) + 1×2B(中心差分 at t-ε) = 3 次
          新方案: 1×3B(cond at t,t+ε,t-ε) + 1×3B(uncond at t,t+ε,t-ε) = 2 次

        节省 1 次教师前向 + 1 次 FSDP all-gather。

        返回:
            v_cfg_t:        CFG v-prediction at t [L, C]
            action_cond_t:  有条件 action v-prediction at t [L_act, C]
            v_cfg_plus:     CFG v-prediction at t+ε [L, C]
            v_cfg_minus:    CFG v-prediction at t-ε [L, C]
        """
        B = input_dict['latent_dict']['noisy_latents'].shape[0]
        ld = input_dict['latent_dict']
        ad = input_dict['action_dict']
        empty_expanded = empty_emb.expand(B, -1, -1)

        if bool(getattr(self.config, 'memory_safe_teacher_forward', True)):
            def _build_b_input(noisy_latents_b, timesteps_b, action_noisy_b=None, action_t_b=None):
                action_noisy_b = ad['noisy_latents'] if action_noisy_b is None else action_noisy_b
                action_t_b = ad['timesteps'] if action_t_b is None else action_t_b
                out = {
                    'latent_dict': {
                        **ld,
                        'noisy_latents': noisy_latents_b,
                        'latent': latents,
                        'timesteps': timesteps_b,
                    },
                    'action_dict': {
                        **ad,
                        'noisy_latents': action_noisy_b,
                        'timesteps': action_t_b,
                    },
                    'chunk_size': input_dict['chunk_size'],
                    'window_size': input_dict['window_size'],
                }
                return out

            v_cond_t, v_uncond_t, action_cond_t = self._batched_cfg_forward(
                _build_b_input(ld['noisy_latents'], ld['timesteps']), empty_emb)
            v_cond_plus, v_uncond_plus, action_cond_plus = self._batched_cfg_forward(
                _build_b_input(noisy_latents_plus, t_plus, action_noisy_plus, action_t_plus), empty_emb)
            v_cond_minus, v_uncond_minus, action_cond_minus = self._batched_cfg_forward(
                _build_b_input(noisy_latents_minus, t_minus, action_noisy_minus, action_t_minus), empty_emb)

            v_cfg_t = v_uncond_t + cfg_scale * (v_cond_t - v_uncond_t)
            v_cfg_plus = v_uncond_plus + cfg_scale * (v_cond_plus - v_uncond_plus)
            v_cfg_minus = v_uncond_minus + cfg_scale * (v_cond_minus - v_uncond_minus)
            return v_cfg_t, action_cond_t, v_cfg_plus, v_cfg_minus, action_cond_plus, action_cond_minus

        def _build_3b_input(text_emb_3b):
            """构建 3B input_dict: (t, t+ε, t-ε) 三个时间步。"""
            if action_noisy_plus is None:
                action_noisy_3b = torch.cat([ad['noisy_latents']] * 3, dim=0)
                action_t_3b = torch.cat([ad['timesteps']] * 3, dim=0)
            else:
                action_noisy_3b = torch.cat([
                    ad['noisy_latents'], action_noisy_plus, action_noisy_minus], dim=0)
                action_t_3b = torch.cat([ad['timesteps'], action_t_plus, action_t_minus], dim=0)

            tri = {
                'latent_dict': {
                    'noisy_latents': torch.cat([
                        ld['noisy_latents'], noisy_latents_plus, noisy_latents_minus], dim=0),
                    'latent': torch.cat([latents, latents, latents], dim=0),
                    'timesteps': torch.cat([ld['timesteps'], t_plus, t_minus], dim=0),
                    'cond_timesteps': torch.cat([ld['cond_timesteps']] * 3, dim=0),
                    'text_emb': text_emb_3b,
                },
                'action_dict': {
                    'noisy_latents': action_noisy_3b,
                    'latent':        torch.cat([ad['latent']] * 3, dim=0),
                    'timesteps':     action_t_3b,
                    'cond_timesteps':torch.cat([ad['cond_timesteps']] * 3, dim=0),
                    'text_emb':      torch.cat([ad['text_emb']] * 3, dim=0),
                },
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
            if 'grid_id' in ld and ld['grid_id'] is not None:
                tri['latent_dict']['grid_id'] = torch.cat([ld['grid_id']] * 3, dim=0)
            if 'grid_id' in ad and ad['grid_id'] is not None:
                tri['action_dict']['grid_id'] = torch.cat([ad['grid_id']] * 3, dim=0)
            return tri

        # 保存/恢复 FlexAttnFunc 的 mask
        from modules.model import FlexAttnFunc
        saved_attn_mask = FlexAttnFunc.attention_mask
        saved_cross_mask = FlexAttnFunc.cross_attention_mask
        try:
            FlexAttnFunc.attention_mask = None
            FlexAttnFunc.cross_attention_mask = None

            # 前向 1 (3B): cond at (t, t+ε, t-ε)
            cond_input = _build_3b_input(torch.cat([ld['text_emb']] * 3, dim=0))
            v_cond_all, action_cond_all = self._video_teacher_model(cond_input, train_mode=True)

            # 重置 mask（第二次前向需要重新创建）
            FlexAttnFunc.attention_mask = None
            FlexAttnFunc.cross_attention_mask = None

            # 前向 2 (3B): uncond at (t, t+ε, t-ε)
            uncond_input = _build_3b_input(torch.cat([empty_expanded] * 3, dim=0))
            v_uncond_all, _ = self._video_teacher_model(uncond_input, train_mode=True)
        finally:
            FlexAttnFunc.attention_mask = saved_attn_mask
            FlexAttnFunc.cross_attention_mask = saved_cross_mask

        # 拆分 3B → 3 个 B: (t, t+ε, t-ε)
        v_cond_t, v_cond_plus, v_cond_minus = v_cond_all.chunk(3, dim=0)
        v_uncond_t, v_uncond_plus, v_uncond_minus = v_uncond_all.chunk(3, dim=0)
        action_cond_t, action_cond_plus, action_cond_minus = action_cond_all.chunk(3, dim=0)

        # CFG 组合
        v_cfg_t = v_uncond_t + cfg_scale * (v_cond_t - v_uncond_t)
        v_cfg_plus = v_uncond_plus + cfg_scale * (v_cond_plus - v_uncond_plus)
        v_cfg_minus = v_uncond_minus + cfg_scale * (v_cond_minus - v_uncond_minus)

        return v_cfg_t, action_cond_t, v_cfg_plus, v_cfg_minus, action_cond_plus, action_cond_minus

    # ==================================================================
    # 全合并：CFG + 中心差分 + action 中心差分 → 2 次 3B teacher forward
    # ==================================================================
    @torch.no_grad()
    def _merged_cfg_central_diff_unified(
        self, input_dict, empty_emb, cfg_scale,
        noisy_latents_plus, noisy_latents_minus,
        t_plus, t_minus, latents,
        action_noisy_plus=None, action_noisy_minus=None,
        action_t_plus=None, action_t_minus=None,
    ):
        """
        将 CFG + 中心差分 + action 中心差分合并为 2 次 3B forward。

        对比:
          memory_safe=True:  3×2B(CFG) + 2B(action_cd) = 8 teacher passes
          本函数:            2×3B(cond+uncond 合并)     = 2 teacher passes

        3B 布局: [t, t+ε, t-ε]，cond 和 uncond 通过 text_emb 区分。

        返回:
            v_cfg_t, action_cond_t, v_cfg_plus, v_cfg_minus,
            action_cond_plus, action_cond_minus
        """
        B = input_dict['latent_dict']['noisy_latents'].shape[0]
        ld = input_dict['latent_dict']
        ad = input_dict['action_dict']
        empty_expanded = empty_emb.expand(B, -1, -1)

        def _build_3b_input(text_emb_3b):
            if action_noisy_plus is None:
                action_noisy_3b = torch.cat([ad['noisy_latents']] * 3, dim=0)
                action_t_3b = torch.cat([ad['timesteps']] * 3, dim=0)
            else:
                action_noisy_3b = torch.cat([
                    ad['noisy_latents'], action_noisy_plus, action_noisy_minus], dim=0)
                action_t_3b = torch.cat([ad['timesteps'], action_t_plus, action_t_minus], dim=0)

            tri = {
                'latent_dict': {
                    'noisy_latents': torch.cat([
                        ld['noisy_latents'], noisy_latents_plus, noisy_latents_minus], dim=0),
                    'latent': torch.cat([latents, latents, latents], dim=0),
                    'timesteps': torch.cat([ld['timesteps'], t_plus, t_minus], dim=0),
                    'cond_timesteps': torch.cat([ld['cond_timesteps']] * 3, dim=0),
                    'text_emb': text_emb_3b,
                },
                'action_dict': {
                    'noisy_latents': action_noisy_3b,
                    'latent':        torch.cat([ad['latent']] * 3, dim=0),
                    'timesteps':     action_t_3b,
                    'cond_timesteps':torch.cat([ad['cond_timesteps']] * 3, dim=0),
                    'text_emb':      torch.cat([ad['text_emb']] * 3, dim=0),
                },
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
            if 'grid_id' in ld and ld['grid_id'] is not None:
                tri['latent_dict']['grid_id'] = torch.cat([ld['grid_id']] * 3, dim=0)
            if 'grid_id' in ad and ad['grid_id'] is not None:
                tri['action_dict']['grid_id'] = torch.cat([ad['grid_id']] * 3, dim=0)
            if 'actions_mask' in ad and ad['actions_mask'] is not None:
                tri['action_dict']['actions_mask'] = torch.cat([ad['actions_mask']] * 3, dim=0)
            return tri

        from modules.model import FlexAttnFunc
        saved_attn_mask = FlexAttnFunc.attention_mask
        saved_cross_mask = FlexAttnFunc.cross_attention_mask
        try:
            FlexAttnFunc.attention_mask = None
            FlexAttnFunc.cross_attention_mask = None

            # 前向 1 (3B): cond at (t, t+ε, t-ε)
            cond_input = _build_3b_input(torch.cat([ld['text_emb']] * 3, dim=0))
            v_cond_all, action_cond_all = self._video_teacher_model(cond_input, train_mode=True)

            FlexAttnFunc.attention_mask = None
            FlexAttnFunc.cross_attention_mask = None

            # 前向 2 (3B): uncond at (t, t+ε, t-ε)
            uncond_input = _build_3b_input(torch.cat([empty_expanded] * 3, dim=0))
            v_uncond_all, _ = self._video_teacher_model(uncond_input, train_mode=True)
        finally:
            FlexAttnFunc.attention_mask = saved_attn_mask
            FlexAttnFunc.cross_attention_mask = saved_cross_mask

        # 拆分 3B → 3 个 B: (t, t+ε, t-ε)
        v_ct, v_cplus, v_cminus = v_cond_all.chunk(3, dim=0)
        v_ut, v_uplus, v_uminus = v_uncond_all.chunk(3, dim=0)
        if action_cond_all is not None:
            a_ct, a_cplus, a_cminus = action_cond_all.chunk(3, dim=0)
        else:
            a_ct = a_cplus = a_cminus = None

        # CFG 组合
        v_cfg_t     = v_ut     + cfg_scale * (v_ct     - v_ut)
        v_cfg_plus  = v_uplus  + cfg_scale * (v_cplus  - v_uplus)
        v_cfg_minus = v_uminus + cfg_scale * (v_cminus - v_uminus)

        return v_cfg_t, a_ct, v_cfg_plus, v_cfg_minus, a_cplus, a_cminus

    # ==================================================================
    # 从模型输出中提取视频 v-prediction → [B, C, F, H, W]
    # ==================================================================
    def _extract_video_v(self, video_pred, ref_shape, batch_size):
        """
        将模型输出的扁平序列转换为 5D 视频张量。

        参数:
            video_pred: 模型输出的扁平化视频 token [B, seq_len, C]
            ref_shape:  参考形状 [B, C, F, H, W]，用于确定时空维度
            batch_size: 批次大小

        返回:
            5D 视频张量 [B, C, F, H, W]，可以直接用于计算损失
        """
        return data_seq_to_patch(
            self.patch_size, video_pred,
            ref_shape[-3], ref_shape[-2], ref_shape[-1],
            batch_size=batch_size,
        )

    # ==================================================================
    # 从模型输出中提取动作 v-prediction → [B, C, F, N, 1]
    # ==================================================================
    def _extract_action_v(self, action_pred, num_frames):
        """
        将模型输出的动作 token 重排为 5D 动作张量。

        参数:
            action_pred: 模型输出的扁平化动作 token [B, F*N, C]
            num_frames:  帧数 F

        返回:
            5D 动作张量 [B, C, F, N, 1]，N 是每帧的动作步数
        """
        seq_len = action_pred.shape[1]
        if seq_len % num_frames != 0:
            action_per_frame = int(getattr(self.config, 'action_per_frame', 0) or 0)
            expected_len = num_frames * action_per_frame
            if action_per_frame > 0 and seq_len >= expected_len:
                action_pred = action_pred[:, :expected_len]
            else:
                raise ValueError(
                    f"Cannot reshape action prediction with seq_len={seq_len}, "
                    f"num_frames={num_frames}, action_per_frame={action_per_frame}"
                )
        return rearrange(action_pred, 'b (f n) c -> b c f n 1', f=num_frames)

    # ==================================================================
    # 一致性函数：f(x_t, t) = c_skip * x_t + c_out * pred_x0
    # ==================================================================
    def _consistency_function(self, v_pred, noisy_latent, sigma, sigma_data=None):
        """
        一致性函数：将 v-prediction 转换为一致性预测。

        核心公式：
          pred_x0 = x_t - σ·v            （FlowMatch 反演：从噪声样本预测干净样本）
          f(x_t, σ) = c_skip·x_t + c_out·pred_x0  （一致性函数）

        参数:
            v_pred:       模型的 v-prediction（噪声方向预测）
            noisy_latent: 加噪后的潜在表示 x_t
            sigma:        当前的噪声水平 σ（归一化值 0~1）
            sigma_data:   数据噪声水平（默认使用配置中的值）

        返回:
            一致性预测结果，形状与 noisy_latent 相同
        """
        if sigma_data is None:
            sigma_data = self.config.sigma_data
        # 将 sigma 扩展为 5D 以匹配 latent 的形状 [B, C, F, H, W]
        # sigma 形状为 [B, T]，扩展为 [B, 1, T, 1, 1]
        sigma_5d = sigma[:, None, :, None, None].to(v_pred.dtype).to(v_pred.device)
        # 计算边界条件缩放系数
        c_skip, c_out = scalings_for_boundary_conditions(
            sigma_5d, sigma_data=sigma_data)
        # FlowMatch 反演：从 v-prediction 推断干净样本
        pred_x0 = noisy_latent - sigma_5d * v_pred
        # 一致性函数：加权组合
        return c_skip * noisy_latent + c_out * pred_x0

    def _teacher_forward_preserve_mask(self, input_dict):
        """Run a teacher forward without leaking FlexAttention mask state."""
        from modules.model import FlexAttnFunc
        saved_attn_mask = FlexAttnFunc.attention_mask
        saved_cross_mask = FlexAttnFunc.cross_attention_mask
        try:
            FlexAttnFunc.attention_mask = None
            FlexAttnFunc.cross_attention_mask = None
            return self._video_teacher_model(input_dict, train_mode=True)
        finally:
            FlexAttnFunc.attention_mask = saved_attn_mask
            FlexAttnFunc.cross_attention_mask = saved_cross_mask

    # ==================================================================
    # 批量 CFG 前向：有条件 + 无条件合并为一次 forward
    # ==================================================================
    def _batched_cfg_forward(self, input_dict, empty_emb):
        """
        将有条件和无条件输入沿 batch 维拼接，一次 teacher forward 完成 CFG。

        原始实现需要两次 teacher forward（cond + uncond），本方法将其合并为一次，
        将 teacher forward 次数从 2 降到 1，结果完全等价。

        参数:
            input_dict:  标准 input_dict（含 latent_dict、action_dict 等）
            empty_emb:   空文本嵌入 [1, seq_len, dim]

        返回:
            v_cond:      有条件 v-prediction [B, L, C]
            v_uncond:    无条件 v-prediction [B, L, C]
            action_cond: 有条件 action prediction [B, L, action_dim]（如果有的话）
        """
        B = input_dict['latent_dict']['noisy_latents'].shape[0]

        # 构建 2B batch 的 input_dict：前半 cond，后半 uncond
        def _cat(a, b):
            """沿 batch 维拼接两个张量。"""
            return torch.cat([a, b], dim=0)

        # 文本嵌入：前半用真实文本，后半用空嵌入
        text_emb = input_dict['latent_dict']['text_emb']
        empty_expanded = empty_emb.expand(B, -1, -1)

        ld = input_dict['latent_dict']
        ad = input_dict['action_dict']

        if bool(getattr(self.config, 'memory_safe_teacher_forward', True)):
            cond_input = {
                'latent_dict': {**ld},
                'action_dict': {**ad},
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
            uncond_latent_dict = {**ld, 'text_emb': empty_expanded}
            uncond_action_dict = {**ad, 'text_emb': empty_expanded}
            uncond_input = {
                'latent_dict': uncond_latent_dict,
                'action_dict': uncond_action_dict,
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
            v_cond, action_cond = self._teacher_forward_preserve_mask(cond_input)
            v_uncond, _ = self._teacher_forward_preserve_mask(uncond_input)
            return v_cond, v_uncond, action_cond

        # latent_dict：所有 batch 维张量翻倍，text_emb 特殊处理
        doubled_latent_dict = {
            'noisy_latents': _cat(ld['noisy_latents'], ld['noisy_latents']),
            'latent':        _cat(ld['latent'], ld['latent']),
            'timesteps':     _cat(ld['timesteps'], ld['timesteps']),
            'cond_timesteps':_cat(ld['cond_timesteps'], ld['cond_timesteps']),
            'text_emb':      _cat(text_emb, empty_expanded),
        }
        if 'grid_id' in ld and ld['grid_id'] is not None:
            doubled_latent_dict['grid_id'] = _cat(ld['grid_id'], ld['grid_id'])

        # action_dict：同理
        doubled_action_dict = {
            'noisy_latents': _cat(ad['noisy_latents'], ad['noisy_latents']),
            'latent':        _cat(ad['latent'], ad['latent']),
            'timesteps':     _cat(ad['timesteps'], ad['timesteps']),
            'cond_timesteps':_cat(ad['cond_timesteps'], ad['cond_timesteps']),
            'text_emb':      _cat(ad['text_emb'], empty_expanded),
        }
        if 'grid_id' in ad and ad['grid_id'] is not None:
            doubled_action_dict['grid_id'] = _cat(ad['grid_id'], ad['grid_id'])
        if 'actions_mask' in ad and ad['actions_mask'] is not None:
            doubled_action_dict['actions_mask'] = _cat(ad['actions_mask'], ad['actions_mask'])

        doubled_input = {
            'latent_dict': doubled_latent_dict,
            'action_dict': doubled_action_dict,
            'chunk_size': input_dict['chunk_size'],
            'window_size': input_dict['window_size'],
        }

        # 一次 forward 处理 2B batch
        # 注意：teacher forward 会更新 FlexAttnFunc 的类级别 block_mask（基于 2B batch），
        # 这会导致后续 student forward（B batch）block_mask 大小不匹配。
        # 因此在 teacher forward 前后保存/恢复 mask。
        #
        # We explicitly set masks to None before the teacher forward so that the
        # teacher's init_mask always creates fresh 2B masks (never reuses stale state).
        # After the teacher forward, we restore the saved masks.
        from modules.model import FlexAttnFunc
        saved_attn_mask = FlexAttnFunc.attention_mask
        saved_cross_mask = FlexAttnFunc.cross_attention_mask
        try:
            FlexAttnFunc.attention_mask = None
            FlexAttnFunc.cross_attention_mask = None
            v_all, action_all = self._video_teacher_model(doubled_input, train_mode=True)
        finally:
            FlexAttnFunc.attention_mask = saved_attn_mask
            FlexAttnFunc.cross_attention_mask = saved_cross_mask

        # 拆分：前半 = 有条件，后半 = 无条件

        v_cond, v_uncond = v_all.chunk(2, dim=0)
        action_cond, _ = action_all.chunk(2, dim=0) if action_all is not None else (None, None)

        return v_cond, v_uncond, action_cond

    @torch.no_grad()
    def _compute_action_central_difference_merged(
        self, input_dict, action_noisy_plus, action_noisy_minus,
        action_t_plus, action_t_minus, eps
    ):
        """Compute action-side dF/dt with video inputs fixed.

        The video branch stays at the current training sample/time. Only
        action_dict.noisy_latents and action_dict.timesteps are evaluated at
        t+eps and t-eps along the same action noising path.
        """
        ld = input_dict["latent_dict"]
        ad = input_dict["action_dict"]

        def _repeat2(x):
            return torch.cat([x, x], dim=0)

        action_dict = {
            "noisy_latents": torch.cat([action_noisy_plus, action_noisy_minus], dim=0),
            "latent": _repeat2(ad["latent"]),
            "timesteps": torch.cat([action_t_plus, action_t_minus], dim=0),
            "cond_timesteps": _repeat2(ad["cond_timesteps"]),
            "text_emb": _repeat2(ad["text_emb"]),
        }
        if "grid_id" in ad and ad["grid_id"] is not None:
            action_dict["grid_id"] = _repeat2(ad["grid_id"])
        if "actions_mask" in ad:
            action_dict["actions_mask"] = _repeat2(ad["actions_mask"])
        if "targets" in ad:
            action_dict["targets"] = _repeat2(ad["targets"])

        doubled_input = {
            "latent_dict": {
                "noisy_latents": _repeat2(ld["noisy_latents"]),
                "latent": _repeat2(ld["latent"]),
                "timesteps": _repeat2(ld["timesteps"]),
                "cond_timesteps": _repeat2(ld["cond_timesteps"]),
                "text_emb": _repeat2(ld["text_emb"]),
            },
            "action_dict": action_dict,
            "chunk_size": input_dict["chunk_size"],
            "window_size": input_dict["window_size"],
        }
        if "grid_id" in ld and ld["grid_id"] is not None:
            doubled_input["latent_dict"]["grid_id"] = _repeat2(ld["grid_id"])
        if "targets" in ld:
            doubled_input["latent_dict"]["targets"] = _repeat2(ld["targets"])

        from modules.model import FlexAttnFunc
        saved_attn_mask = FlexAttnFunc.attention_mask
        saved_cross_mask = FlexAttnFunc.cross_attention_mask
        try:
            FlexAttnFunc.attention_mask = None
            FlexAttnFunc.cross_attention_mask = None
            _, action_all = self._action_teacher_model(doubled_input, train_mode=True)
        finally:
            FlexAttnFunc.attention_mask = saved_attn_mask
            FlexAttnFunc.cross_attention_mask = saved_cross_mask

        if action_all is None:
            raise RuntimeError("Teacher did not return action output for action FlowMap central difference.")
        action_plus, action_minus = action_all.chunk(2, dim=0)
        return (action_plus - action_minus) / (2 * eps)

    # ==================================================================
    # Student Batched CFG Forward
    # ==================================================================
    def _student_cfg_forward(self, model, input_dict, empty_emb, cfg_scale,
                              B, ref_shape, r_timestep, action_r_timestep,
                              force_cfg=False, return_action=False):
        """Batched CFG forward for student — cond+uncond in single 2B pass.

        Like _batched_cfg_forward (teacher), but for student model (or nofsdp copy).
        Caller is responsible for temporarily unwrapping torch.compile from
        model.blocks if needed (Phase 2), or passing an uncompiled model (Phase 1).

        Args:
            model:               Student model (or _student_nofsdp) to call
            input_dict:          Standard input_dict with B=1 batch
            empty_emb:           Empty text embedding [1, seq_len, dim]
            cfg_scale:           CFG guidance strength
            B:                   Original batch size
            ref_shape:           Reference shape for _extract_video_v
            r_timestep:          Reference timestep [B, F]
            action_r_timestep:   Action reference timestep [B, F_ds]

        Returns:
            v_cfg:  CFG-combined v-prediction in 5D format [B, C, F, H, W]
            action: Optional action head output from the conditional student pass
                    when return_action=True.
        """
        if cfg_scale <= 1.0 and not force_cfg:
            from modules.model import FlexAttnFunc
            saved_attn = FlexAttnFunc.attention_mask
            saved_cross = FlexAttnFunc.cross_attention_mask
            try:
                FlexAttnFunc.attention_mask = None
                FlexAttnFunc.cross_attention_mask = None
                v, action_seq = model(input_dict, train_mode=True,
                                      r_timestep=r_timestep,
                                      action_r_timestep=action_r_timestep)
            finally:
                FlexAttnFunc.attention_mask = saved_attn
                FlexAttnFunc.cross_attention_mask = saved_cross
            v_cfg = self._extract_video_v(v, ref_shape, B)
            return (v_cfg, action_seq) if return_action else v_cfg

        from modules.model import FlexAttnFunc

        def _cat(a, b):
            return torch.cat([a, b], dim=0)

        text_emb = input_dict['latent_dict']['text_emb']
        empty_expanded = empty_emb.expand(B, -1, -1)

        ld = input_dict['latent_dict']
        ad = input_dict['action_dict']
        serial_student_cfg = bool(getattr(
            self.config, 'opd_serial_student_cfg', False))

        if serial_student_cfg:
            def _run_single_cfg(single_input, single_r, single_ar):
                saved_attn = FlexAttnFunc.attention_mask
                saved_cross = FlexAttnFunc.cross_attention_mask
                try:
                    FlexAttnFunc.attention_mask = None
                    FlexAttnFunc.cross_attention_mask = None
                    v, action_seq = model(single_input, train_mode=True,
                                          r_timestep=single_r,
                                          action_r_timestep=single_ar)
                finally:
                    FlexAttnFunc.attention_mask = saved_attn
                    FlexAttnFunc.cross_attention_mask = saved_cross
                return self._extract_video_v(v, ref_shape, B), action_seq

            cond_input = {
                'latent_dict': ld,
                'action_dict': ad,
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
            uncond_input = {
                'latent_dict': {**ld, 'text_emb': empty_expanded},
                'action_dict': {**ad, 'text_emb': empty_expanded},
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
            v_cond_5d, action_cond = _run_single_cfg(cond_input, r_timestep, action_r_timestep)
            v_uncond_5d, _ = _run_single_cfg(uncond_input, r_timestep, action_r_timestep)
            v_cfg = v_uncond_5d + cfg_scale * (v_cond_5d - v_uncond_5d)
            return (v_cfg, action_cond) if return_action else v_cfg

        doubled_latent_dict = {
            'noisy_latents':  _cat(ld['noisy_latents'], ld['noisy_latents']),
            'latent':         _cat(ld['latent'], ld['latent']),
            'timesteps':      _cat(ld['timesteps'], ld['timesteps']),
            'cond_timesteps': _cat(ld['cond_timesteps'], ld['cond_timesteps']),
            'text_emb':       _cat(text_emb, empty_expanded),
        }
        if 'grid_id' in ld and ld['grid_id'] is not None:
            doubled_latent_dict['grid_id'] = _cat(ld['grid_id'], ld['grid_id'])
        if 'targets' in ld:
            doubled_latent_dict['targets'] = _cat(ld['targets'], ld['targets'])

        doubled_action_dict = {
            'noisy_latents':  _cat(ad['noisy_latents'], ad['noisy_latents']),
            'latent':         _cat(ad['latent'], ad['latent']),
            'timesteps':      _cat(ad['timesteps'], ad['timesteps']),
            'cond_timesteps': _cat(ad['cond_timesteps'], ad['cond_timesteps']),
            'text_emb':       _cat(ad['text_emb'], empty_expanded),
        }
        if 'grid_id' in ad and ad['grid_id'] is not None:
            doubled_action_dict['grid_id'] = _cat(ad['grid_id'], ad['grid_id'])
        if 'targets' in ad:
            doubled_action_dict['targets'] = _cat(ad['targets'], ad['targets'])
        if 'actions_mask' in ad and ad['actions_mask'] is not None:
            doubled_action_dict['actions_mask'] = _cat(ad['actions_mask'], ad['actions_mask'])

        doubled_input = {
            'latent_dict': doubled_latent_dict,
            'action_dict': doubled_action_dict,
            'chunk_size': input_dict['chunk_size'],
            'window_size': input_dict['window_size'],
        }

        doubled_r = _cat(r_timestep, r_timestep)
        doubled_ar = _cat(action_r_timestep, action_r_timestep)

        # Save / clear / restore mask state — same pattern as teacher
        saved_attn = FlexAttnFunc.attention_mask
        saved_cross = FlexAttnFunc.cross_attention_mask
        try:
            FlexAttnFunc.attention_mask = None
            FlexAttnFunc.cross_attention_mask = None
            v_all, action_all = model(doubled_input, train_mode=True,
                                      r_timestep=doubled_r,
                                      action_r_timestep=doubled_ar)
        finally:
            FlexAttnFunc.attention_mask = saved_attn
            FlexAttnFunc.cross_attention_mask = saved_cross

        v_cond, v_uncond = v_all.chunk(2, dim=0)
        v_cond_5d = self._extract_video_v(v_cond, ref_shape, B)
        v_uncond_5d = self._extract_video_v(v_uncond, ref_shape, B)
        v_cfg = v_uncond_5d + cfg_scale * (v_cond_5d - v_uncond_5d)
        if return_action:
            action_cond = action_all.chunk(2, dim=0)[0] if action_all is not None else None
            return v_cfg, action_cond
        return v_cfg


    # ==================================================================
    # 时间步加权策略
    # ==================================================================
    def _get_timestep_weight(self, t, weight_type):
        """
        根据 weight_type 返回时间步权重。

        参数:
            t:           原始时间步 [B]（0~num_train_timesteps）
            weight_type: 权重类型字符串，支持 'beta08'、'gaussian'、'uniform'

        返回:
            权重张量 [B]
        """
        if weight_type == 'beta08':
            # Beta 分布风格：t^1.0 * (1-t)^0.5，偏向高噪声时间步
            t_norm = t / self.config.num_train_timesteps
            return t_norm ** 1.0 * (1 - t_norm) ** 0.5
        elif weight_type == 'gaussian':
            # 高斯权重：以 t=0.5 为中心，偏向中间时间步
            t_norm = t / self.config.num_train_timesteps
            return torch.exp(-2 * ((t_norm - 0.5) ** 2))
        else:
            # 均匀权重（默认）
            return torch.ones_like(t)

    @torch.no_grad()
    def _cosmos_video_cdiff_target(
        self,
        input_dict,
        batch,
        video_noisy_latents,
        video_noise,
        video_t,
        video_r,
        ref_shape,
        batch_size,
        cfg_scale,
    ):
        """Return WanVA/LingBotVA central-diff FlowMap target for Cosmos x0 anchors."""
        if getattr(self, "_video_teacher_nofsdp", None) is None:
            raise RuntimeError(
                "Cosmos video central-diff requires a WanVA video teacher. "
                "Set cfg.cosmos_video_cdiff_teacher_model_path."
            )

        v_pred = video_noise - batch["latents"]
        t_plus = (video_t + self.epsilon).clamp(
            max=self.config.num_train_timesteps)
        t_minus = (video_t - self.epsilon).clamp(min=0)
        noisy_latents_plus = video_noisy_latents + v_pred * (
            self.epsilon / self.config.num_train_timesteps)
        noisy_latents_minus = video_noisy_latents - v_pred * (
            self.epsilon / self.config.num_train_timesteps)

        if bool(getattr(self.config, "use_central_diff", True)):
            (
                video_v_cfg_seq,
                _action_cond_t,
                v_plus_seq,
                v_minus_seq,
                _action_cond_plus,
                _action_cond_minus,
            ) = self._merged_cfg_central_diff_unified(
                input_dict,
                self.empty_emb,
                cfg_scale,
                noisy_latents_plus,
                noisy_latents_minus,
                t_plus,
                t_minus,
                batch["latents"],
            )
            video_v_cfg_5d = self._extract_video_v(
                video_v_cfg_seq, ref_shape, batch_size)
            video_v_plus_5d = self._extract_video_v(
                v_plus_seq, ref_shape, batch_size)
            video_v_minus_5d = self._extract_video_v(
                v_minus_seq, ref_shape, batch_size)
            dF_dt = (video_v_plus_5d - video_v_minus_5d) / (2 * self.epsilon)
        else:
            video_v_cond, video_v_uncond, _ = self._batched_cfg_forward(
                input_dict, self.empty_emb)
            video_v_cfg = video_v_uncond + cfg_scale * (
                video_v_cond - video_v_uncond)
            video_v_cfg_5d = self._extract_video_v(
                video_v_cfg, ref_shape, batch_size)
            dF_dt = torch.zeros_like(video_v_cfg_5d)

        t_5d = video_t[:, None, :, None, None].to(video_v_cfg_5d)
        r_5d = video_r[:, None, :, None, None].to(video_v_cfg_5d)
        return (video_v_cfg_5d - (t_5d - r_5d) * dF_dt).detach()

    @torch.no_grad()
    def _cosmos_wanva_stage1_targets(
        self,
        input_dict,
        batch,
        video_noisy_latents,
        video_noise,
        video_t,
        video_r,
        action_noisy_latents,
        action_noise,
        action_t,
        action_r,
        ref_shape,
        batch_size,
        cfg_scale,
    ):
        """Return LingBotVA-style WanVA video/action FlowMap targets.

        Cosmos still supplies future-image endpoint anchors and raw action
        metrics. The primary vector-field targets come from the frozen
        WanVA/LingBotVA teacher so this path matches the regular Stage 1
        FlowMap objective.
        """
        if getattr(self, "_video_teacher_nofsdp", None) is None:
            raise RuntimeError(
                "Cosmos LingBotVA-aligned Stage 1 requires a WanVA video/action "
                "teacher. Set cfg.cosmos_video_cdiff_teacher_model_path."
            )

        use_central_diff = bool(getattr(self.config, "use_central_diff", True))
        action_ds = getattr(self.config, "action_downsample_factor", 4)
        num_frames = ref_shape[2]

        v_pred = video_noise - batch["latents"]
        t_plus = (video_t + self.epsilon).clamp(max=self.config.num_train_timesteps)
        noisy_latents_plus = (
            video_noisy_latents
            + v_pred * (self.epsilon / self.config.num_train_timesteps)
        )
        t_minus = (video_t - self.epsilon).clamp(min=0)
        noisy_latents_minus = (
            video_noisy_latents
            - v_pred * (self.epsilon / self.config.num_train_timesteps)
        )

        action_noisy_plus = action_noisy_minus = None
        action_t_plus = action_t_minus = None
        action_has_flowmap = False
        action_eps = float(getattr(self.config, "action_epsilon", self.epsilon))
        if (
            use_central_diff
            and bool(getattr(self.config, "action_use_flowmap", False))
        ):
            flowmap_token_mask = (
                ((action_t[:, ::action_ds] - action_r[:, ::action_ds]).abs() > 1e-3)
                & (action_r[:, ::action_ds] > 1e-3)
            )
            action_has_flowmap = bool(flowmap_token_mask.any().item())
            if action_has_flowmap:
                action_t_center = action_t.clamp(
                    min=action_eps,
                    max=self.config.num_train_timesteps - action_eps,
                )
                action_t_plus = action_t_center + action_eps
                action_t_minus = action_t_center - action_eps
                action_noisy_plus = self.train_scheduler_action.add_noise(
                    batch["actions"], action_noise, action_t_plus, t_dim=2
                )
                action_noisy_minus = self.train_scheduler_action.add_noise(
                    batch["actions"], action_noise, action_t_minus, t_dim=2
                )

        if use_central_diff:
            (
                video_v_cfg_seq,
                action_v_cond_seq,
                video_v_plus_seq,
                video_v_minus_seq,
                action_plus_seq,
                action_minus_seq,
            ) = self._merged_cfg_central_diff_unified(
                input_dict,
                self.empty_emb,
                cfg_scale,
                noisy_latents_plus,
                noisy_latents_minus,
                t_plus,
                t_minus,
                batch["latents"],
                action_noisy_plus=action_noisy_plus,
                action_noisy_minus=action_noisy_minus,
                action_t_plus=action_t_plus,
                action_t_minus=action_t_minus,
            )
            video_v_cfg_5d = self._extract_video_v(
                video_v_cfg_seq, ref_shape, batch_size)
            video_v_plus_5d = self._extract_video_v(
                video_v_plus_seq, ref_shape, batch_size)
            video_v_minus_5d = self._extract_video_v(
                video_v_minus_seq, ref_shape, batch_size)
            video_dF_dt = (video_v_plus_5d - video_v_minus_5d) / (2 * self.epsilon)

            action_dF_dt_5d = None
            if action_has_flowmap and action_plus_seq is not None and action_minus_seq is not None:
                action_plus_5d = self._extract_action_v(action_plus_seq, num_frames)
                action_minus_5d = self._extract_action_v(action_minus_seq, num_frames)
                action_dF_dt_5d = (action_plus_5d - action_minus_5d) / (2 * action_eps)
        else:
            video_v_cond, video_v_uncond, action_v_cond_seq = self._batched_cfg_forward(
                input_dict, self.empty_emb)
            video_v_cfg = video_v_uncond + cfg_scale * (
                video_v_cond - video_v_uncond)
            video_v_cfg_5d = self._extract_video_v(
                video_v_cfg, ref_shape, batch_size)
            video_dF_dt = torch.zeros_like(video_v_cfg_5d)
            action_dF_dt_5d = None

        if action_v_cond_seq is None:
            raise RuntimeError(
                "WanVA teacher did not return action output for LingBotVA-aligned "
                "Cosmos Stage 1."
            )
        action_v_5d = self._extract_action_v(action_v_cond_seq, num_frames)

        t_5d = video_t[:, None, :, None, None].to(video_v_cfg_5d)
        r_5d = video_r[:, None, :, None, None].to(video_v_cfg_5d)
        video_target = video_v_cfg_5d - (t_5d - r_5d) * video_dF_dt

        action_t_sigma = action_t / self.config.num_train_timesteps
        action_r_sigma = action_r / self.config.num_train_timesteps
        sigma_s_a = action_t_sigma[:, None, :, None, None].to(action_v_5d)
        sigma_r_a = action_r_sigma[:, None, :, None, None].to(action_v_5d)
        x_prev_action = action_noisy_latents + action_v_5d * (sigma_r_a - sigma_s_a)

        return {
            "video_target": video_target.detach(),
            "video_v_cfg": video_v_cfg_5d.detach(),
            "action_v": action_v_5d.detach(),
            "action_dF_dt": (
                action_dF_dt_5d.detach() if action_dF_dt_5d is not None else None
            ),
            "x_prev_action": x_prev_action.detach(),
        }

    # ==================================================================
    # 核心训练步：Flow Map 蒸馏
    # ==================================================================
    def _cosmos_policy_train_step(self, batch, batch_idx):
        """Stage step for Cosmos Policy checkpoints.

        The default path remains action-only. When cfg.cosmos_video_target=True,
        official Cosmos future_image_predictions are encoded through the WanVA
        VAE and used as the video x0 target, while official Cosmos actions are
        used as the action x0 target.
        """
        batch = self.convert_input_format(batch)
        profile_step = bool(getattr(self.config, "cosmos_train_step_profile", False))
        profile_times = {}
        profile_last = None

        def _profile_mark(name):
            nonlocal profile_last
            if not profile_step:
                return
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            now = time.perf_counter()
            if profile_last is not None:
                profile_times[name] = now - profile_last
            profile_last = now

        _profile_mark("start")

        teacher = self._action_teacher_model
        teacher_action_x0_full = None
        cosmos_latent_teacher_result = None
        cosmos_latent_cdiff_active = False
        cosmos_latent_target_step = None
        latent_target_mode = None
        video_t = video_r = video_t_norm = video_r_norm = None
        video_noise = None
        if getattr(self.config, "cosmos_latent_target", False):
            if not hasattr(teacher, "predict_raw_latent_target"):
                raise RuntimeError(
                    "cfg.cosmos_latent_target=True requires Cosmos Policy raw "
                    "inference with predict_raw_latent_target()."
                )
            if not getattr(teacher, "raw_inference_enabled", False):
                raise RuntimeError(
                    "cfg.cosmos_latent_target=True requires "
                    "cfg.cosmos_policy_use_raw_inference=True."
                )
            B_raw = int(batch["actions"].shape[0])
            latent_shape = (
                B_raw,
                int(getattr(self.config, "cosmos_latent_channels", 16)),
                int(getattr(self.config, "cosmos_latent_frames", 9)),
                int(getattr(self.config, "cosmos_latent_height", 28)),
                int(getattr(self.config, "cosmos_latent_width", 28)),
            )
            video_t, video_r, video_t_norm, video_r_norm, _ = (
                self.sample_cosmos_latent_timestep_mixed(
                    B_raw, latent_shape[2], dtype=torch.float32, device=self.device,
                )
            )
            video_noise = torch.randn(
                latent_shape, device=self.device, dtype=batch["actions"].dtype)
            latent_target_mode = getattr(
                self.config, "cosmos_latent_target_mode", "cdiff")
            latent_cdiff_interval = int(getattr(
                self.config, "cosmos_latent_cdiff_interval", 1))
            cosmos_latent_target_step = cosmos_latent_micro_step_index(
                global_step=getattr(self, "step", 0),
                batch_idx=batch_idx,
                gradient_accumulation_steps=getattr(
                    self, "gradient_accumulation_steps", 1),
            )
            cosmos_latent_cdiff_active = cosmos_latent_should_request_cdiff(
                latent_target_mode,
                step=cosmos_latent_target_step,
                interval=latent_cdiff_interval,
            )
            with torch.no_grad():
                cosmos_latent_teacher_result = teacher.predict_raw_latent_target(
                    batch,
                    noise=video_noise,
                    t=video_t_norm,
                    r=video_r_norm,
                    epsilon=float(getattr(self.config, "cosmos_latent_epsilon", 0.001)),
                    include_cdiff=cosmos_latent_cdiff_active,
                )
                teacher_action_x0_full = cosmos_actions_to_flowmap_x0(
                    cosmos_latent_teacher_result["actions"],
                    target_shape=tuple(batch["actions"].shape),
                    q01=self.config.norm_stat["q01"],
                    q99=self.config.norm_stat["q99"],
                    inverse_used_action_channel_ids=self.config.inverse_used_action_channel_ids,
                    device=batch["actions"].device,
                    dtype=batch["actions"].dtype,
                    packing_schema=self.config.action_packing_schema,
                    downsample_factor=self.config.action_downsample_factor,
                )
                batch["latents"] = cosmos_latent_teacher_result[
                    "cosmos_latent_x0"
                ].to(device=batch["actions"].device, dtype=batch["actions"].dtype)
                if bool(getattr(self.config, "cosmos_use_teacher_action_anchor", False)):
                    batch["actions"] = teacher_action_x0_full
        elif getattr(self, "cosmos_video_target", False):
            if not hasattr(teacher, "predict_raw_action_result"):
                raise RuntimeError(
                    "cfg.cosmos_video_target=True requires Cosmos Policy raw "
                    "inference with predict_raw_action_result()."
                )
            if not getattr(teacher, "raw_inference_enabled", False):
                raise RuntimeError(
                    "cfg.cosmos_video_target=True requires "
                    "cfg.cosmos_policy_use_raw_inference=True."
                )
            with torch.no_grad():
                raw_teacher_result = teacher.predict_raw_action_result(
                    batch, include_future=True)
                teacher_action_x0_full = cosmos_actions_to_flowmap_x0(
                    raw_teacher_result["actions"],
                    target_shape=tuple(batch["actions"].shape),
                    q01=self.config.norm_stat["q01"],
                    q99=self.config.norm_stat["q99"],
                    inverse_used_action_channel_ids=self.config.inverse_used_action_channel_ids,
                    device=batch["actions"].device,
                    dtype=batch["actions"].dtype,
                    packing_schema=self.config.action_packing_schema,
                    downsample_factor=self.config.action_downsample_factor,
                )
                batch["latents"] = self._encode_cosmos_future_video_latents(
                    raw_teacher_result.get("future_image_predictions"),
                    ref_shape=tuple(batch["latents"].shape),
                ).to(device=batch["latents"].device, dtype=batch["latents"].dtype)
        _profile_mark("teacher")

        B = batch['latents'].shape[0]
        ref_shape = batch['latents'].shape
        num_frames = ref_shape[2]
        action_frames = batch['actions'].shape[2]
        actions_mask = batch.get('actions_mask')
        input_dict = self._prepare_base_dict(batch)

        if video_t is None or video_r is None:
            video_t, video_r, _ = self.sample_timestep_mixed(
                B, num_frames, dtype=torch.float32, device=self.device,
            )
        action_t, action_r, _ = self.sample_timestep_mixed(
            B, action_frames, dtype=torch.float32, device=self.device,
            scheduler=self.train_scheduler_action,
        )
        action_r_sigma = action_r / self.config.num_train_timesteps
        cfg_scale = self.config.cfg_min + torch.rand(1).item() * (
            self.config.cfg_max - self.config.cfg_min)

        if video_noise is None:
            video_noise = torch.randn_like(batch['latents'])
            video_noisy_latents = self.train_scheduler_latent.add_noise(
                batch['latents'], video_noise, video_t, t_dim=2
            )
        else:
            sigma_t = video_t_norm[:, None, :, None, None].to(batch['latents'])
            video_noisy_latents = (
                (1.0 - sigma_t) * batch['latents'] + sigma_t * video_noise
            )
        video_v_target = self.train_scheduler_latent.training_target(
            batch['latents'], video_noise, video_t
        )
        input_dict['latent_dict']['noisy_latents'] = video_noisy_latents
        input_dict['latent_dict']['timesteps'] = video_t
        input_dict['latent_dict']['targets'] = video_v_target

        action_noise = torch.randn_like(batch['actions'])
        action_noisy_latents = self.train_scheduler_action.add_noise(
            batch['actions'], action_noise, action_t, t_dim=2
        )
        action_v_target = self.train_scheduler_action.training_target(
            batch['actions'], action_noise, action_t
        )
        input_dict['action_dict']['noisy_latents'] = action_noisy_latents
        input_dict['action_dict']['timesteps'] = action_t
        input_dict['action_dict']['targets'] = action_v_target

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        if hasattr(self.student, 'set_requires_gradient_sync'):
            self.student.set_requires_gradient_sync(should_sync)

        action_ds = getattr(self.config, 'action_downsample_factor', 4)
        action_noisy_ds = action_noisy_latents[:, :, ::action_ds]
        actions_gt_ds = batch['actions'][:, :, ::action_ds]
        action_r_ds = action_r[:, ::action_ds]

        student_input = {
            'latent_dict': {
                **input_dict['latent_dict'],
                'timesteps': video_t,
            },
            'action_dict': {
                'noisy_latents': action_noisy_ds,
                'latent': actions_gt_ds,
                'timesteps': action_t[:, ::action_ds],
                'cond_timesteps': input_dict['action_dict']['cond_timesteps'][:, ::action_ds],
                'text_emb': input_dict['action_dict']['text_emb'],
            },
            'chunk_size': input_dict['chunk_size'],
            'window_size': input_dict['window_size'],
        }
        if 'grid_id' in input_dict['action_dict'] and input_dict['action_dict']['grid_id'] is not None:
            student_input['action_dict']['grid_id'] = _downsample_action_grid_id(
                input_dict['action_dict']['grid_id'], batch['actions'], action_ds)
        if 'actions_mask' in input_dict['action_dict'] and input_dict['action_dict']['actions_mask'] is not None:
            student_input['action_dict']['actions_mask'] = input_dict['action_dict']['actions_mask'][:, :, ::action_ds]

        from modules.model import FlexAttnFunc
        ld = student_input['latent_dict']
        ad = student_input['action_dict']
        total_length = (
            ld['noisy_latents'].flatten(0, 1).shape[0] * 2 +
            ad['noisy_latents'].flatten(0, 1).shape[0] * 2
        )
        padded_length = (128 - total_length % 128) % 128
        FlexAttnFunc.init_mask(
            ld['noisy_latents'].shape,
            ad['noisy_latents'].shape,
            padded_length,
            student_input['chunk_size'],
            window_size=student_input['window_size'],
            patch_size=self.patch_size,
            device=self.device,
        )

        student_video_v_seq, student_action_v_seq = self.student(
            student_input, train_mode=True,
            r_timestep=video_r,
            action_r_timestep=action_r_ds,
        )
        _profile_mark("student_forward")
        if self.distill_video:
            student_video_v = self._extract_video_v(student_video_v_seq, ref_shape, B)
        student_action_v = self._extract_action_v(
            student_action_v_seq, student_input['action_dict']['noisy_latents'].shape[2])

        if actions_mask is None:
            mask = torch.ones_like(actions_gt_ds[:, :1]).float()
        else:
            mask = actions_mask[:, :, ::action_ds].float()
        action_denom = (mask.sum() * student_action_v.shape[1]).clamp(min=1)
        zero_metric = torch.tensor(0.0, device=self.device)
        raw_teacher_enabled = zero_metric
        raw_teacher_gt_mse = zero_metric
        raw_teacher_gt_l1 = zero_metric
        raw_teacher_abs_mean = zero_metric
        raw_gt_abs_mean = zero_metric
        cosmos_video_endpoint_loss = zero_metric
        cosmos_video_cdiff_loss = zero_metric
        lingbotva_stage1_targets = None
        if bool(getattr(self.config, "cosmos_action_lingbotva_stage1", False)):
            lingbotva_stage1_targets = self._cosmos_wanva_stage1_targets(
                input_dict=input_dict,
                batch=batch,
                video_noisy_latents=video_noisy_latents,
                video_noise=video_noise,
                video_t=video_t,
                video_r=video_r,
                action_noisy_latents=action_noisy_latents,
                action_noise=action_noise,
                action_t=action_t,
                action_r=action_r,
                ref_shape=ref_shape,
                batch_size=B,
                cfg_scale=cfg_scale,
            )

        sigma_r = action_r_sigma[:, None, ::action_ds, None, None].to(student_action_v)
        student_action_pred = action_noisy_ds - sigma_r * student_action_v

        if not hasattr(teacher, 'action_target_tokens'):
            raise RuntimeError("Cosmos Policy backend requires a teacher with action_target_tokens().")
        with torch.no_grad():
            teacher_action_x0 = teacher_action_x0_full
            if teacher_action_x0 is None and hasattr(teacher, 'action_target_x0') and getattr(teacher, 'raw_inference_enabled', False):
                teacher_action_x0 = teacher.action_target_x0(input_dict['action_dict'], raw_batch=batch)
            if teacher_action_x0 is not None:
                teacher_action_pred = teacher_action_x0[:, :, ::action_ds]
            else:
                teacher_action_v_seq = teacher.action_target_tokens(input_dict['action_dict'])
                teacher_action_v = self._extract_action_v(teacher_action_v_seq, action_frames)
                teacher_action_pred = action_noisy_ds - sigma_r * teacher_action_v[:, :, ::action_ds]
        if teacher_action_x0 is not None:
            raw_stats = compute_masked_action_stats(teacher_action_pred, actions_gt_ds, mask)
            raw_teacher_enabled = torch.tensor(1.0, device=self.device)
            raw_teacher_gt_mse = raw_stats["mse"]
            raw_teacher_gt_l1 = raw_stats["l1"]
            raw_teacher_abs_mean = raw_stats["teacher_abs_mean"]
            raw_gt_abs_mean = raw_stats["target_abs_mean"]

        if lingbotva_stage1_targets is not None:
            action_v_5d = lingbotva_stage1_targets["action_v"]
            action_dF_dt_5d = lingbotva_stage1_targets["action_dF_dt"]
            use_action_distill = getattr(self.config, "use_action_distill", True)
            target_action_pred = None

            if self.distill_action and use_action_distill:
                video_sigma_s = self._timestep_to_sigma_5d(video_t)
                video_sigma_r = self._timestep_to_sigma_5d(video_r)
                x_prev_video = (
                    video_noisy_latents
                    + lingbotva_stage1_targets["video_v_cfg"] * (
                        video_sigma_r - video_sigma_s)
                )
                target_input = {
                    "latent_dict": {
                        **input_dict["latent_dict"],
                        "noisy_latents": x_prev_video.detach(),
                        "timesteps": video_r,
                    },
                    "action_dict": {
                        "noisy_latents": lingbotva_stage1_targets[
                            "x_prev_action"
                        ][:, :, ::action_ds],
                        "latent": actions_gt_ds,
                        "timesteps": action_r[:, ::action_ds],
                        "cond_timesteps": input_dict["action_dict"][
                            "cond_timesteps"
                        ][:, ::action_ds],
                        "text_emb": input_dict["action_dict"]["text_emb"],
                    },
                    "chunk_size": input_dict["chunk_size"],
                    "window_size": input_dict["window_size"],
                }
                if (
                    "grid_id" in input_dict["action_dict"]
                    and input_dict["action_dict"]["grid_id"] is not None
                ):
                    target_input["action_dict"]["grid_id"] = _downsample_action_grid_id(
                        input_dict["action_dict"]["grid_id"],
                        batch["actions"],
                        action_ds,
                    )
                if (
                    "actions_mask" in input_dict["action_dict"]
                    and input_dict["action_dict"]["actions_mask"] is not None
                ):
                    target_input["action_dict"]["actions_mask"] = input_dict[
                        "action_dict"
                    ]["actions_mask"][:, :, ::action_ds]

                with torch.no_grad():
                    _, target_action_v_seq = self.target_student(
                        target_input,
                        train_mode=True,
                        r_timestep=video_r,
                        action_r_timestep=action_r[:, ::action_ds],
                    )
                    target_action_v = self._extract_action_v(
                        target_action_v_seq,
                        target_input["action_dict"]["noisy_latents"].shape[2],
                    )
                    target_action_pred = (
                        lingbotva_stage1_targets["x_prev_action"][:, :, ::action_ds]
                        - sigma_r.to(target_action_v) * target_action_v
                    )

            if use_action_distill and target_action_pred is not None:
                action_target_pred = target_action_pred
            elif self.action_distill_mode == "x0":
                action_target_pred = actions_gt_ds
            else:
                student_action_pred = self._consistency_function(
                    student_action_v, action_noisy_ds, action_r_sigma[:, ::action_ds]
                )
                action_target_pred = action_v_5d[:, :, ::action_ds]

            action_use_flowmap = getattr(self.config, "action_use_flowmap", False)
            flowmap_token_mask = torch.zeros_like(
                action_r[:, ::action_ds], dtype=torch.bool)
            if action_use_flowmap:
                flowmap_token_mask = (
                    ((action_t[:, ::action_ds] - action_r[:, ::action_ds]).abs() > 1e-3)
                    & (action_r[:, ::action_ds] > 1e-3)
                )

            if (
                action_use_flowmap
                and flowmap_token_mask.any()
                and action_dF_dt_5d is not None
            ):
                action_t_5d = action_t[:, None, ::action_ds, None, None].to(action_v_5d)
                action_r_5d = action_r[:, None, ::action_ds, None, None].to(action_v_5d)
                action_flowmap_v_target = action_v_5d[:, :, ::action_ds] - (
                    action_t_5d - action_r_5d
                ) * action_dF_dt_5d[:, :, ::action_ds]

                if self.action_distill_mode == "x0":
                    flowmap_action_target_pred = (
                        action_noisy_ds - sigma_r.to(action_flowmap_v_target)
                        * action_flowmap_v_target
                    )
                else:
                    flowmap_action_target_pred = action_flowmap_v_target

                action_target_pred = torch.where(
                    flowmap_token_mask[:, None, :, None, None],
                    flowmap_action_target_pred,
                    action_target_pred,
                )
        else:
            action_target_pred = teacher_action_pred

        action_diff = (student_action_pred.float() - action_target_pred.detach().float()) * mask
        action_loss = (action_diff ** 2).sum() / action_denom

        gt_regression_loss = torch.tensor(0.0, device=self.device)
        if getattr(self.config, 'use_gt_regression', True) and self.gt_regression_weight > 0:
            gt_diff = (student_action_pred.float() - actions_gt_ds.float()) * mask
            gt_regression_loss = (gt_diff ** 2).sum() / action_denom

        action_local_fm_loss = torch.tensor(0.0, device=self.device)
        action_aware_loss = torch.tensor(0.0, device=self.device)
        if self.action_aware:
            action_targets = action_v_target[:, :, ::action_ds]
            aa_diff = (student_action_v.float() - action_targets.float().detach()) * mask
            action_local_fm_loss = (aa_diff ** 2).sum() / action_denom
            action_aware_loss = action_local_fm_loss

        video_loss = torch.tensor(0.0, device=self.device)
        if self.distill_video:
            if getattr(self.config, "cosmos_latent_target", False):
                if cosmos_latent_teacher_result is None:
                    raise RuntimeError("Missing Cosmos latent teacher result.")
                if "cosmos_latent_cdiff_target" in cosmos_latent_teacher_result:
                    latent_target = cosmos_latent_teacher_result[
                        "cosmos_latent_cdiff_target"
                    ].to(device=student_video_v.device, dtype=student_video_v.dtype)
                else:
                    latent_target = video_v_target.to(
                        device=student_video_v.device, dtype=student_video_v.dtype)
                cdiff_diff = student_video_v.float() - latent_target.float()
                loss_type = getattr(self.config, "loss_type", "l2")
                if loss_type == "huber":
                    huber_c = float(getattr(self.config, "huber_c", 0.001))
                    abs_diff = cdiff_diff.abs()
                    token_loss = torch.where(
                        abs_diff < huber_c,
                        0.5 * cdiff_diff ** 2,
                        huber_c * (abs_diff - 0.5 * huber_c),
                    ).mean(dim=1)
                else:
                    token_loss = (cdiff_diff ** 2).mean(dim=1)
                per_sample_loss = token_loss.flatten(1).mean(dim=1)
                weight_type = getattr(self.config, "weight_type", "uniform")
                weight = self._get_timestep_weight(
                    video_t.mean(dim=-1), weight_type).to(self.device)
                cosmos_video_cdiff_loss = (per_sample_loss * weight).mean()
                cdiff_loss_weight = float(
                    getattr(self.config, "cosmos_latent_cdiff_loss_weight", 1.0)
                )
                endpoint_loss_weight = float(
                    getattr(self.config, "cosmos_latent_endpoint_loss_weight", 0.0)
                )
                video_loss = cdiff_loss_weight * cosmos_video_cdiff_loss
                if endpoint_loss_weight > 0:
                    video_sigma_r = (
                        video_r / self.config.num_train_timesteps
                    )[:, None, :, None, None].to(student_video_v)
                    student_video_pred = video_noisy_latents - video_sigma_r * student_video_v
                    video_diff = student_video_pred.float() - batch['latents'].detach().float()
                    cosmos_video_endpoint_loss = (video_diff ** 2).mean()
                    video_loss = video_loss + endpoint_loss_weight * cosmos_video_endpoint_loss
            else:
                cdiff_mode = str(
                    getattr(self.config, "cosmos_video_cdiff_mode", "aux")
                ).lower()
                if cdiff_mode not in ("primary", "aux"):
                    raise ValueError(
                        "cfg.cosmos_video_cdiff_mode must be either 'primary' or 'aux'."
                    )
                if cdiff_mode == "primary" and not bool(
                    getattr(self.config, "cosmos_video_cdiff_aux", False)
                ):
                    raise RuntimeError(
                        "cfg.cosmos_video_cdiff_mode='primary' requires "
                        "cfg.cosmos_video_cdiff_aux=True."
                    )
                endpoint_loss_weight = float(
                    getattr(self.config, "cosmos_video_endpoint_loss_weight", 1.0))
                cdiff_loss_weight = float(
                    getattr(self.config, "cosmos_video_cdiff_loss_weight", 0.0))
                video_sigma_r = (
                    video_r / self.config.num_train_timesteps
                )[:, None, :, None, None].to(student_video_v)
                student_video_pred = video_noisy_latents - video_sigma_r * student_video_v
                video_diff = student_video_pred.float() - batch['latents'].detach().float()
                cosmos_video_endpoint_loss = (video_diff ** 2).mean()
                video_loss = endpoint_loss_weight * cosmos_video_endpoint_loss

                if bool(getattr(self.config, "cosmos_video_cdiff_aux", False)):
                    if lingbotva_stage1_targets is not None:
                        cdiff_target = lingbotva_stage1_targets["video_target"]
                    else:
                        cdiff_target = self._cosmos_video_cdiff_target(
                            input_dict=input_dict,
                            batch=batch,
                            video_noisy_latents=video_noisy_latents,
                            video_noise=video_noise,
                            video_t=video_t,
                            video_r=video_r,
                            ref_shape=ref_shape,
                            batch_size=B,
                            cfg_scale=cfg_scale,
                        )
                    cdiff_diff = student_video_v.float() - cdiff_target.float()
                    loss_type = getattr(self.config, "loss_type", "l2")
                    if loss_type == "huber":
                        huber_c = float(getattr(self.config, "huber_c", 0.001))
                        abs_diff = cdiff_diff.abs()
                        token_loss = torch.where(
                            abs_diff < huber_c,
                            0.5 * cdiff_diff ** 2,
                            huber_c * (abs_diff - 0.5 * huber_c),
                        ).mean(dim=1)
                    else:
                        token_loss = (cdiff_diff ** 2).mean(dim=1)
                    per_sample_loss = token_loss.flatten(1).mean(dim=1)
                    weight_type = getattr(self.config, "weight_type", "uniform")
                    weight = self._get_timestep_weight(
                        video_t.mean(dim=-1), weight_type).to(self.device)
                    cosmos_video_cdiff_loss = (per_sample_loss * weight).mean()
                    video_loss = video_loss + cdiff_loss_weight * cosmos_video_cdiff_loss

        loss = getattr(self.config, 'video_loss_weight', 1.0) * video_loss \
               + getattr(self.config, 'action_block_weight', 1.0) * (
            self.config.action_loss_weight * action_loss
            + self.gt_regression_weight * gt_regression_loss
            + getattr(self.config, 'action_aware_weight', 0.0) * action_aware_loss
        )
        loss = loss / self.gradient_accumulation_steps

        loss_clip_value = getattr(self.config, 'loss_clip_value', None)
        if loss_clip_value is not None and getattr(self.config, 'loss_clip_enabled', True):
            loss_clip_value = float(loss_clip_value)
            if loss_clip_value >= 0:
                scale = (loss_clip_value / loss.detach().clamp(min=1e-12)).clamp(max=1.0)
                loss = loss * scale

        local_nonfinite_origins = {
            "video": not bool(torch.isfinite(video_loss.detach()).all().item()),
            "action": not bool(torch.isfinite(action_loss.detach()).all().item())
            or not bool(torch.isfinite(action_aware_loss.detach()).all().item()),
            "teacher_or_gt": not bool(
                torch.isfinite(gt_regression_loss.detach()).all().item()
            ),
        }
        is_finite, nonfinite_origins = _synchronized_nonfinite_decision(
            loss=loss,
            local_flags=local_nonfinite_origins,
            device=self.device,
        )
        if not is_finite:
            if self.config.rank == 0:
                logger.warning(f"[step {self.step}] NaN/Inf loss, skipping")
            return {
                "loss": zero_metric,
                "video_loss": video_loss.detach(),
                "action_loss": action_loss.detach(),
                "action_local_fm_loss": action_local_fm_loss.detach(),
                "action_aware_loss": action_aware_loss.detach(),
                "gt_regression_loss": gt_regression_loss.detach(),
                "raw_teacher_gt_mse": raw_teacher_gt_mse.detach(),
                "raw_teacher_gt_l1": raw_teacher_gt_l1.detach(),
                "raw_teacher_abs_mean": raw_teacher_abs_mean.detach(),
                "raw_gt_abs_mean": raw_gt_abs_mean.detach(),
                "raw_teacher_enabled": raw_teacher_enabled.detach(),
                "cosmos_video_endpoint_loss": cosmos_video_endpoint_loss.detach(),
                "cosmos_video_cdiff_loss": cosmos_video_cdiff_loss.detach(),
                "kto_main_active": zero_metric,
                "kto_main_good_ratio": zero_metric,
                "kto_main_weight_mean": zero_metric,
                "kto_main_weight_min": zero_metric,
                "kto_main_weight_max": zero_metric,
                "kto_main_threshold": zero_metric,
                "should_sync": should_sync,
                "skip_step": True,
                "nonfinite_origins": nonfinite_origins,
            }

        _profile_mark("loss_compute")
        loss.backward()
        _profile_mark("backward")
        if profile_step and getattr(self.config, "rank", 0) == 0:
            logger.info(
                "[cosmos_step_profile] step=%s micro_step=%s mode=%s cdiff=%s "
                "teacher=%.3fs student_forward=%.3fs loss_compute=%.3fs backward=%.3fs",
                getattr(self, "step", 0),
                cosmos_latent_target_step,
                latent_target_mode,
                cosmos_latent_cdiff_active,
                profile_times.get("teacher", 0.0),
                profile_times.get("student_forward", 0.0),
                profile_times.get("loss_compute", 0.0),
                profile_times.get("backward", 0.0),
            )
        return {
            "loss": loss.detach(),
            "video_loss": video_loss.detach(),
            "action_loss": action_loss.detach(),
            "action_local_fm_loss": action_local_fm_loss.detach(),
            "action_aware_loss": action_aware_loss.detach(),
            "gt_regression_loss": gt_regression_loss.detach(),
            "raw_teacher_gt_mse": raw_teacher_gt_mse.detach(),
            "raw_teacher_gt_l1": raw_teacher_gt_l1.detach(),
            "raw_teacher_abs_mean": raw_teacher_abs_mean.detach(),
            "raw_gt_abs_mean": raw_gt_abs_mean.detach(),
            "raw_teacher_enabled": raw_teacher_enabled.detach(),
            "cosmos_video_endpoint_loss": cosmos_video_endpoint_loss.detach(),
            "cosmos_video_cdiff_loss": cosmos_video_cdiff_loss.detach(),
            "kto_main_active": zero_metric,
            "kto_main_good_ratio": zero_metric,
            "kto_main_weight_mean": zero_metric,
            "kto_main_weight_min": zero_metric,
            "kto_main_weight_max": zero_metric,
            "kto_main_threshold": zero_metric,
            "should_sync": should_sync,
            "skip_step": False,
            "nonfinite_origins": nonfinite_origins,
        }

    def _train_step(self, batch, batch_idx):
        """
        执行一步 Flow Map 蒸馏训练。

        与原始 StepMixin._train_step 的核心区别：
          1. 使用 sample_timestep_mixed() 采样 (t, r) 对（而非固定的 sigma_start/sigma_end）
          2. 教师 CFG 前向生成 v-prediction（而非 Euler 步推进）
          3. 中心差分法计算 dF/dt（flow map 梯度）
          4. 训练目标 = v_pred - (t-r) * dF/dt（而非 EMA 学生的一致性预测）
          5. 学生模型在 r 处预测（而非 sigma_start 处）
          6. 损失包含 GT 回归和 action_aware 辅助损失

        参数:
            batch:     数据批次，包含 latents、actions、text_emb 等
            batch_idx: 当前批次在梯度累积中的索引

        返回:
            包含损失值和是否需要梯度同步的字典
        """
        if getattr(self, 'is_cosmos_policy_teacher', False) and (
            not self.distill_video
            or getattr(self.config, 'cosmos_video_target', False)
            or getattr(self.config, 'cosmos_latent_target', False)
        ):
            return self._cosmos_policy_train_step(batch, batch_idx)

        batch = self.convert_input_format(batch)

        # ==============================================================
        # 消融开关（用于 ablation study，默认全部启用）
        # ==============================================================
        use_flowmap = getattr(self.config, 'use_flowmap', True)          # False 时 flowmap_ratio=0
        use_gt_regression = getattr(self.config, 'use_gt_regression', True)  # False 时跳过 gt_regression_loss
        use_central_diff = getattr(self.config, 'use_central_diff', True)    # False 时 dF_dt=0
        selective_cdiff = getattr(self.config, 'selective_cdiff', True)      # False 时对所有 batch 计算中心差分

        B = batch['latents'].shape[0]
        ref_shape = batch['latents'].shape     # [B, C, F, H, W]
        num_frames = ref_shape[2]
        actions_mask = batch.get('actions_mask')

        # ==============================================================
        # 步骤 1: 准备基础 input_dict（不加噪，避免冗余）
        # 只提取 grid_id、cond_timesteps、latent、text_emb 等基础字段
        # ==============================================================
        input_dict = self._prepare_base_dict(batch)

        # ==============================================================
        # 步骤 2: 混合时间步采样
        # 为视频和动作分别采样 (t, r) 对
        # ==============================================================
        # 消融开关 use_flowmap：为 False 时将 flowmap_ratio 设为 0（只保留扩散和一致性）
        # 临时保存原始比例，采样后恢复
        orig_diffusion_ratio = self.diffusion_ratio
        orig_consistency_ratio = self.consistency_ratio
        orig_flowmap_ratio = self.flowmap_ratio
        if not use_flowmap:
            # 将流映射比例分配给扩散和一致性（按原始比例归一化）
            total = self.diffusion_ratio + self.consistency_ratio
            if total > 0:
                self.diffusion_ratio = self.diffusion_ratio / total
                self.consistency_ratio = self.consistency_ratio / total
            else:
                self.diffusion_ratio = 0.5
                self.consistency_ratio = 0.5
            self.flowmap_ratio = 0.0

        # 视频时间步采样：sample_timestep_mixed 已应用 SNR shift 并转换为原始时间步
        video_t, video_r, video_is_diffusion = self.sample_timestep_mixed(
            B, num_frames, dtype=torch.float32, device=self.device,
        )

        # 恢复原始比例
        self.diffusion_ratio = orig_diffusion_ratio
        self.consistency_ratio = orig_consistency_ratio
        self.flowmap_ratio = orig_flowmap_ratio

        # 保留归一化 sigma（用于后续 x0 参数化等需要归一化值的场景）
        video_t_sigma = video_t / self.config.num_train_timesteps

        # 动作分支始终需要有效输入；未蒸馏动作时复用视频时间步。
        if self.distill_action:
            # 同样临时覆盖比例
            orig_diffusion_ratio = self.diffusion_ratio
            orig_consistency_ratio = self.consistency_ratio
            orig_flowmap_ratio = self.flowmap_ratio
            if not use_flowmap:
                total = self.diffusion_ratio + self.consistency_ratio
                if total > 0:
                    self.diffusion_ratio = self.diffusion_ratio / total
                    self.consistency_ratio = self.consistency_ratio / total
                else:
                    self.diffusion_ratio = 0.5
                    self.consistency_ratio = 0.5
                self.flowmap_ratio = 0.0

            action_t, action_r, action_is_diffusion = self.sample_timestep_mixed(
                B, num_frames, dtype=torch.float32, device=self.device,
                scheduler=self.train_scheduler_action,
            )

            # 恢复原始比例
            self.diffusion_ratio = orig_diffusion_ratio
            self.consistency_ratio = orig_consistency_ratio
            self.flowmap_ratio = orig_flowmap_ratio
        else:
            action_t = video_t
            action_r = video_r
            action_is_diffusion = video_is_diffusion

        action_t_sigma = action_t / self.config.num_train_timesteps
        action_r_sigma = action_r / self.config.num_train_timesteps

        # ==============================================================
        # 步骤 3: 加噪（使用采样的 t）
        # ==============================================================
        # 使用采样的时间步进行加噪（这是唯一的加噪步骤，避免冗余）
        # 生成新的噪声
        video_noise = torch.randn_like(batch['latents'])
        # 使用调度器的 add_noise 方法：x_t = (1-σ)*x_0 + σ*noise
        # 注意：add_noise 接受原始时间步（0~1000），内部会查找对应的 sigma
        video_noisy_latents = self.train_scheduler_latent.add_noise(
            batch['latents'], video_noise, video_t, t_dim=2
        )
        # 训练目标：v = noise - x_0
        video_v_target = self.train_scheduler_latent.training_target(
            batch['latents'], video_noise, video_t
        )

        action_noise = torch.randn_like(batch['actions'])
        action_noisy_latents = self.train_scheduler_action.add_noise(
            batch['actions'], action_noise, action_t, t_dim=2
        )
        action_v_target = self.train_scheduler_action.training_target(
            batch['actions'], action_noise, action_t
        )

        # ==============================================================
        # 步骤 3b: 填充 input_dict（使用加噪结果）
        # ==============================================================
        # 将加噪结果填充到 input_dict 中
        input_dict['latent_dict']['noisy_latents'] = video_noisy_latents
        input_dict['latent_dict']['timesteps'] = video_t  # [B, F]
        # targets 用于 action_aware loss
        input_dict['latent_dict']['targets'] = video_v_target

        input_dict['action_dict']['noisy_latents'] = action_noisy_latents
        input_dict['action_dict']['timesteps'] = action_t  # [B, F]
        input_dict['action_dict']['targets'] = action_v_target

        # ==============================================================
        # 步骤 4: 教师前向（CFG + 中心差分合并优化）
        # ==============================================================
        # 随机采样 CFG 引导强度（在 [cfg_min, cfg_max] 范围内均匀采样）
        cfg_scale = self.config.cfg_min + torch.rand(1).item() * (
            self.config.cfg_max - self.config.cfg_min)

        # 准备空文本嵌入（各 helper 内部按需要 expand）
        empty_emb = self.empty_emb

        # 预计算 t+ε 和 t-ε 的噪声样本和时间步（中心差分用）
        v_pred = video_noise - batch['latents']
        t_plus = (video_t + self.epsilon).clamp(max=self.config.num_train_timesteps)
        noisy_latents_plus = video_noisy_latents + v_pred * (self.epsilon / self.config.num_train_timesteps)
        t_minus = (video_t - self.epsilon).clamp(min=0)
        noisy_latents_minus = video_noisy_latents - v_pred * (self.epsilon / self.config.num_train_timesteps)

        action_use_flowmap_cfg = False
        action_has_flowmap = False
        action_eps = self.epsilon
        action_t_plus = action_t_minus = None
        action_noisy_plus = action_noisy_minus = None
        if self.distill_action:
            action_use_flowmap_cfg = getattr(self.config, "action_use_flowmap", False)
            if action_use_flowmap_cfg and use_central_diff:
                action_ds_for_mask = getattr(self.config, 'action_downsample_factor', 4)
                action_flowmap_token_mask = (
                    ((action_t[:, ::action_ds_for_mask] - action_r[:, ::action_ds_for_mask]).abs() > 1e-3) &
                    (action_r[:, ::action_ds_for_mask] > 1e-3)
                )
                action_has_flowmap = bool(action_flowmap_token_mask.any().item())
                if action_has_flowmap:
                    action_eps = float(getattr(self.config, "action_epsilon", self.epsilon))
                    action_t_center = action_t.clamp(
                        min=action_eps,
                        max=self.config.num_train_timesteps - action_eps,
                    )
                    action_t_plus = action_t_center + action_eps
                    action_t_minus = action_t_center - action_eps
                    action_noisy_plus = self.train_scheduler_action.add_noise(
                        batch["actions"], action_noise, action_t_plus, t_dim=2
                    )
                    action_noisy_minus = self.train_scheduler_action.add_noise(
                        batch["actions"], action_noise, action_t_minus, t_dim=2
                    )

        # ==============================================================
        # 步骤 5: 中心差分计算 dF/dt（与 CFG 前向合并优化）
        # ==============================================================
        if selective_cdiff:
            non_diffusion_mask = ~video_is_diffusion  # [B]
            has_non_diffusion = non_diffusion_mask.any()
        else:
            non_diffusion_mask = torch.ones(B, dtype=torch.bool, device=self.device)
            has_non_diffusion = True

        action_cond_plus_seq = None
        action_cond_minus_seq = None
        with torch.no_grad():
            if has_non_diffusion and use_central_diff:
                if selective_cdiff and B > 1:
                    # B > 1 选择性: 提取非扩散子集，分别做 CFG + 中心差分
                    non_diff_indices = non_diffusion_mask.nonzero(as_tuple=True)[0]
                    noisy_latents_sub = video_noisy_latents[non_diff_indices]
                    latents_sub = batch['latents'][non_diff_indices]
                    noise_sub = video_noise[non_diff_indices]
                    t_sub = video_t[non_diff_indices]
                    r_sub = video_r[non_diff_indices]

                    # CFG forward for full batch
                    video_v_cond, video_v_uncond, action_v_cond = \
                        self._batched_cfg_forward(input_dict, empty_emb)
                    video_v_cfg = video_v_uncond + cfg_scale * (video_v_cond - video_v_uncond)
                    video_v_cfg_5d = self._extract_video_v(video_v_cfg, ref_shape, B)

                    # Build sub-batch input_dict for the non-diffusion subset
                    sub_input_dict = {
                        'latent_dict': {
                            k: v[non_diff_indices] if isinstance(v, torch.Tensor) and v.shape[0] == B else v
                            for k, v in input_dict['latent_dict'].items()
                        },
                        'action_dict': {
                            k: v[non_diff_indices] if isinstance(v, torch.Tensor) and v.shape[0] == B else v
                            for k, v in input_dict['action_dict'].items()
                        },
                        'chunk_size': input_dict['chunk_size'],
                        'window_size': input_dict['window_size'],
                    }
                    dF_dt_sub = self.compute_central_difference_merged(
                        input_dict=sub_input_dict, empty_emb=self.empty_emb, cfg_scale=cfg_scale,
                        noisy_latents=noisy_latents_sub, latents=latents_sub,
                        noise=noise_sub, t=t_sub, r=r_sub,
                        eps=self.epsilon, ref_shape=ref_shape)
                    dF_dt = torch.zeros_like(video_v_cfg_5d)
                    dF_dt[non_diff_indices] = dF_dt_sub
                else:
                    # B=1 或不筛选: 全合并 1×6B（CFG + 中心差分 + action 中心差分）
                    (video_v_cfg_seq, action_v_cond, v_plus_seq, v_minus_seq,
                     action_cond_plus_seq, action_cond_minus_seq) = \
                        self._merged_cfg_central_diff_unified(
                            input_dict, empty_emb, cfg_scale,
                            noisy_latents_plus, noisy_latents_minus,
                            t_plus, t_minus, batch['latents'],
                            action_noisy_plus=action_noisy_plus,
                            action_noisy_minus=action_noisy_minus,
                            action_t_plus=action_t_plus,
                            action_t_minus=action_t_minus)
                    video_v_cfg_5d = self._extract_video_v(video_v_cfg_seq, ref_shape, B)
                    video_v_plus_5d = self._extract_video_v(v_plus_seq, ref_shape, B)
                    video_v_minus_5d = self._extract_video_v(v_minus_seq, ref_shape, B)
                    dF_dt = (video_v_plus_5d - video_v_minus_5d) / (2 * self.epsilon)
            else:
                # 扩散样本: 仅 1×2B CFG 前向
                video_v_cond, video_v_uncond, action_v_cond = \
                    self._batched_cfg_forward(input_dict, empty_emb)
                video_v_cfg = video_v_uncond + cfg_scale * (video_v_cond - video_v_uncond)
                video_v_cfg_5d = self._extract_video_v(video_v_cfg, ref_shape, B)
                dF_dt = torch.zeros_like(video_v_cfg_5d)

            # 动作 v-prediction（两种方案共用）
            action_dF_dt_5d = None
            if self.distill_action:
                action_v_5d = self._extract_action_v(action_v_cond, num_frames)
                if action_has_flowmap:
                    if action_cond_plus_seq is not None and action_cond_minus_seq is not None:
                        action_plus_5d = self._extract_action_v(action_cond_plus_seq, num_frames)
                        action_minus_5d = self._extract_action_v(action_cond_minus_seq, num_frames)
                        action_dF_dt_5d = (action_plus_5d - action_minus_5d) / (2 * action_eps)
                    else:
                        action_dF_dt_seq = self._compute_action_central_difference_merged(
                            input_dict=input_dict,
                            action_noisy_plus=action_noisy_plus,
                            action_noisy_minus=action_noisy_minus,
                            action_t_plus=action_t_plus,
                            action_t_minus=action_t_minus,
                            eps=action_eps,
                        )
                        action_dF_dt_5d = self._extract_action_v(action_dF_dt_seq, num_frames)

                # 教师 Euler 步推进（action）：从 action_t 推进到 action_r
                # 公式：x_prev = x_t + v * (σ_r - σ_t)
                # 这是原始 LCM 蒸馏的核心步骤，让教师生成"伪 ground truth"
                use_action_distill = getattr(self.config, 'use_action_distill', True)
                if use_action_distill:
                    sigma_s_a = action_t_sigma[:, None, :, None, None].to(action_v_5d)
                    sigma_r_a = action_r_sigma[:, None, :, None, None].to(action_v_5d)
                    x_prev_action = action_noisy_latents + action_v_5d * (sigma_r_a - sigma_s_a)

        # ==============================================================
        # 步骤 6: 计算 Flow Map 训练目标
        # target = v_pred - (t - r) * dF/dt
        # ==============================================================
        # 将 t 和 r 扩展为 5D 以匹配 latent 的形状 [B, C, F, H, W]
        # video_t/video_r 形状为 [B, T]，扩展为 [B, 1, T, 1, 1]
        t_5d = video_t[:, None, :, None, None].to(video_v_cfg_5d.dtype)  # [B, 1, T, 1, 1]
        r_5d = video_r[:, None, :, None, None].to(video_v_cfg_5d.dtype)  # [B, 1, T, 1, 1]

        # Flow Map 目标公式：
        #   当 r=t（扩散）：target = v_pred（标准 FlowMatch）
        #   当 r=0（一致性）：target = v_pred - t * dF/dt
        #   当 r<t（流映射）：target = v_pred - (t-r) * dF/dt
        video_target = video_v_cfg_5d - (t_5d - r_5d) * dF_dt

        # ==============================================================
        # 步骤 7: 学生模型在 r 处预测
        # ==============================================================
        # 梯度累积优化：只在需要同步时才启用梯度同步
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        if not should_sync:
            # PEFT (LoRA) 模型没有 set_requires_gradient_sync 方法
            if hasattr(self.student, 'set_requires_gradient_sync'):
                self.student.set_requires_gradient_sync(False)
        else:
            if hasattr(self.student, 'set_requires_gradient_sync'):
                self.student.set_requires_gradient_sync(True)

        # Action 时间下采样：学生只处理每 4 帧的 action（256 → 64 tokens）
        # 教师保持全分辨率（256 tokens），学生用低分辨率减少计算量
        # 机器人动作是连续的，相邻帧差异小，下采样信息损失有限
        _ACTION_DS = getattr(self.config, 'action_downsample_factor', 4)

        # 构建学生模型的 input_dict（使用 r 时间步）
        # 注意：学生模型接收 r_timestep 参数（由 patch_model_forward 添加）
        student_input = {
            'latent_dict': {
                **input_dict['latent_dict'],
                'timesteps': video_t,  # [B, F] 学生在 t 处预测（噪声等级对应 noisy_latents）
            },
            'action_dict': {
                'noisy_latents': action_noisy_latents[:, :, ::_ACTION_DS],
                'latent':        batch['actions'][:, :, ::_ACTION_DS],
                'timesteps':     action_t[:, ::_ACTION_DS],
                'cond_timesteps':input_dict['action_dict']['cond_timesteps'][:, ::_ACTION_DS],
                'text_emb':      input_dict['action_dict']['text_emb'],
            },
            'chunk_size': input_dict['chunk_size'],
            'window_size': input_dict['window_size'],
        }
        if 'grid_id' in input_dict['action_dict'] and input_dict['action_dict']['grid_id'] is not None:
            student_input['action_dict']['grid_id'] = _downsample_action_grid_id(
                input_dict['action_dict']['grid_id'], batch['actions'], _ACTION_DS)
        if 'actions_mask' in input_dict['action_dict'] and input_dict['action_dict']['actions_mask'] is not None:
            student_input['action_dict']['actions_mask'] = input_dict['action_dict']['actions_mask'][:, :, ::_ACTION_DS]

        # Action r_timestep 也需要下采样，与 action token 维度对齐
        action_r_ds = action_r[:, ::_ACTION_DS]

        # 学生前向（传入 r_timestep 以启用 delta_embedder）
        # 视频使用 r_timestep，动作使用 action_r_timestep（与 patched_forward 签名一致）
        #
        # 在 student forward 前显式调用 init_mask，确保 FlexAttnFunc 的 ClassVar
        # block_mask 大小与 student 的 B batch 匹配（teacher forward 用 2B 会污染缓存）。
        # 不依赖 @torch._dynamo.disable（在某些 PyTorch 版本下对 nn.Module.forward 无效）。
        from modules.model import FlexAttnFunc
        # 计算 student 的 padded_length（与 patched_forward_train 内部逻辑一致）
        _ld = student_input['latent_dict']
        _ad = student_input['action_dict']
        _total_length = (
            _ld['noisy_latents'].flatten(0, 1).shape[0] * 2 +
            _ad['noisy_latents'].flatten(0, 1).shape[0] * 2
        )
        _padded_length = (128 - _total_length % 128) % 128
        FlexAttnFunc.init_mask(
            _ld['noisy_latents'].shape,
            _ad['noisy_latents'].shape,
            _padded_length,
            student_input['chunk_size'],
            window_size=student_input['window_size'],
            patch_size=self.patch_size,
            device=self.device,
        )
        del _ld, _ad, _total_length, _padded_length

        student_video_v_seq, student_action_v_seq = self.student(
            student_input, train_mode=True,
            r_timestep=video_r,  # [B, F]
            action_r_timestep=action_r_ds,  # [B, F_ds]
        )

        # ==============================================================
        # 步骤 8: 提取学生预测
        # ==============================================================
        if self.distill_video:
            student_video_v = self._extract_video_v(student_video_v_seq, ref_shape, B)

        if self.distill_action or self.action_aware:
            _action_num_frames = student_input['action_dict']['noisy_latents'].shape[2]  # 下采样后的动作帧数
            student_action_v = self._extract_action_v(student_action_v_seq, _action_num_frames)

        # ==============================================================
        # 步骤 8b: 目标学生（EMA）生成 action 目标（教师蒸馏）
        # ==============================================================
        # 核心改进：使用 target_student 在 action_r 处生成蒸馏目标，
        # 而非直接使用 GT 动作。这是原始 LCM 蒸馏的核心思想。
        use_action_distill = getattr(self.config, 'use_action_distill', True)
        target_action_pred = None
        if self.distill_action and use_action_distill:
            video_sigma_s = self._timestep_to_sigma_5d(video_t)
            video_sigma_r = self._timestep_to_sigma_5d(video_r)
            x_prev_video = video_noisy_latents + video_v_cfg_5d * (video_sigma_r - video_sigma_s)
            # 构建 target_student 的 input_dict（action 也下采样）
            target_input = {
                'latent_dict': {
                    **input_dict['latent_dict'],
                    'noisy_latents': x_prev_video.detach(),
                    'timesteps': video_r,  # 视频使用 r 时间步
                },
                'action_dict': {
                    'noisy_latents': x_prev_action.detach()[:, :, ::_ACTION_DS],
                    'latent':        batch['actions'][:, :, ::_ACTION_DS],
                    'timesteps':     action_r[:, ::_ACTION_DS],
                    'cond_timesteps':input_dict['action_dict']['cond_timesteps'][:, ::_ACTION_DS],
                    'text_emb':      input_dict['action_dict']['text_emb'],
                },
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
            if 'grid_id' in input_dict['action_dict'] and input_dict['action_dict']['grid_id'] is not None:
                target_input['action_dict']['grid_id'] = _downsample_action_grid_id(
                    input_dict['action_dict']['grid_id'], batch['actions'], _ACTION_DS)
            if 'actions_mask' in input_dict['action_dict'] and input_dict['action_dict']['actions_mask'] is not None:
                target_input['action_dict']['actions_mask'] = input_dict['action_dict']['actions_mask'][:, :, ::_ACTION_DS]

            with torch.no_grad():
                # 目标学生（EMA）在 action_r 处做前向
                _, target_action_v_seq = self.target_student(
                    target_input, train_mode=True,
                    r_timestep=video_r,
                    action_r_timestep=action_r[:, ::_ACTION_DS],
                )
                target_action_v = self._extract_action_v(target_action_v_seq, target_input['action_dict']['noisy_latents'].shape[2])

                # x0 参数化：target_action_pred = x_prev_action - σ_r * target_action_v
                sigma_r_a_5d = action_r_sigma[:, None, ::_ACTION_DS, None, None].to(target_action_v)
                # x_prev_action 是教师 256 tokens 的输出，需要下采样到 64
                x_prev_action_ds = x_prev_action[:, :, ::_ACTION_DS]
                target_action_pred = x_prev_action_ds - sigma_r_a_5d * target_action_v

        # ==============================================================
        # 步骤 9: 计算损失
        # ==============================================================
        # --- 9a. 视频 Flow Map 损失（在步骤 10 中统一计算）---
        video_loss = torch.tensor(0.0, device=self.device)

        # --- 9b. 动作损失（支持流映射目标）---
        action_loss = torch.tensor(0.0, device=self.device)
        gt_regression_loss = torch.tensor(0.0, device=self.device)
        action_aware_loss = torch.tensor(0.0, device=self.device)
        action_local_fm_loss = torch.tensor(0.0, device=self.device)

        if self.distill_action or self.action_aware:
            # Action 时间下采样：与学生模型一致（每 _ACTION_DS 帧取 1 个）
            _ad = _ACTION_DS
            action_noisy_ds = action_noisy_latents[:, :, ::_ad]
            actions_mask_ds = actions_mask[:, :, ::_ad]
            actions_gt_ds = batch['actions'][:, :, ::_ad]
            action_sigma_s = action_r_sigma[:, None, ::_ad, None, None].to(student_action_v)
            student_action_pred = action_noisy_ds - action_sigma_s * student_action_v
            mask = actions_mask_ds.float()
            action_denom = (mask.sum() * student_action_pred.shape[1]).clamp(min=1)

            if self.distill_action:
                with torch.no_grad():
                    cosmos_action_target_pred = self._cosmos_action_x0_target(
                        input_dict['action_dict'],
                        raw_batch=batch,
                        downsample_factor=_ad,
                    )
                action_use_flowmap = getattr(self.config, 'action_use_flowmap', False)
                if action_use_flowmap:
                    flowmap_token_mask = (
                        ((action_t[:, ::_ad] - action_r[:, ::_ad]).abs() > 1e-3) &
                        (action_r[:, ::_ad] > 1e-3)
                    )
                else:
                    flowmap_token_mask = torch.zeros_like(action_r[:, ::_ad], dtype=torch.bool)

                if cosmos_action_target_pred is not None:
                    base_action_target_pred = cosmos_action_target_pred
                    flowmap_token_mask = torch.zeros_like(flowmap_token_mask)
                elif use_action_distill and target_action_pred is not None:
                    base_action_target_pred = target_action_pred
                elif self.action_distill_mode == "x0":
                    base_action_target_pred = actions_gt_ds
                else:
                    student_action_pred = self._consistency_function(
                        student_action_v, action_noisy_ds, action_r_sigma[:, ::_ad]
                    )
                    base_action_target_pred = action_v_5d[:, :, ::_ad]

                if action_use_flowmap and flowmap_token_mask.any() and action_dF_dt_5d is not None:
                    action_t_5d = action_t[:, None, ::_ad, None, None].to(action_v_5d)
                    action_r_5d = action_r[:, None, ::_ad, None, None].to(action_v_5d)
                    action_flowmap_v_target = action_v_5d[:, :, ::_ad] - (
                        action_t_5d - action_r_5d
                    ) * action_dF_dt_5d[:, :, ::_ad]

                    if self.action_distill_mode == "x0":
                        sigma_r_a_5d = action_r_sigma[:, None, ::_ad, None, None].to(action_flowmap_v_target)
                        flowmap_action_target_pred = action_noisy_ds - sigma_r_a_5d * action_flowmap_v_target
                    else:
                        flowmap_action_target_pred = action_flowmap_v_target

                    action_target_pred = torch.where(
                        flowmap_token_mask[:, None, :, None, None],
                        flowmap_action_target_pred,
                        base_action_target_pred,
                    )
                else:
                    action_target_pred = base_action_target_pred

                action_diff = (student_action_pred.float() * mask) - \
                              (action_target_pred.detach().float() * mask)
                action_loss = (action_diff ** 2).sum() / action_denom

                if use_gt_regression and self.gt_regression_weight > 0:
                    gt_diff = (student_action_pred.float() * mask) - \
                              (actions_gt_ds.float() * mask)
                    gt_regression_loss = (gt_diff ** 2).sum() / action_denom

            if self.action_aware:
                action_targets = input_dict['action_dict']['targets'][:, :, ::_ad]
                aa_diff = (student_action_v.float() - action_targets.float().detach()) * mask
                action_local_fm_loss = (aa_diff ** 2).sum() / action_denom
                action_aware_loss = action_local_fm_loss

        # ==============================================================
        # 步骤 10: 时间步加权 + 扩散样本缩放
        # ==============================================================
        # 时间步加权：根据 weight_type 选择权重策略
        weight_type = getattr(self.config, 'weight_type', 'uniform')
        # video_t 形状为 [B, T]，取帧平均得到 [B] 用于 per-sample 加权
        weight = self._get_timestep_weight(video_t.mean(dim=-1), weight_type).to(self.device)  # [B]
        kto_main_active = torch.tensor(0.0, device=self.device)
        kto_main_good_ratio = torch.tensor(0.0, device=self.device)
        kto_main_weight_mean = torch.tensor(0.0, device=self.device)
        kto_main_weight_min = torch.tensor(0.0, device=self.device)
        kto_main_weight_max = torch.tensor(0.0, device=self.device)
        kto_main_threshold = torch.tensor(0.0, device=self.device)

        # 对视频 loss 按样本加权
        if self.distill_video:
            # 逐样本 loss（支持 huber 和 mse）
            loss_type = getattr(self.config, 'loss_type', 'l2')
            huber_c = getattr(self.config, 'huber_c', 0.001)
            video_diff = (student_video_v.float() - video_target.detach().float())
            if loss_type == "huber":
                # Huber loss: |x| < c -> 0.5*x^2; otherwise c*(|x|-0.5*c)
                abs_diff = video_diff.abs()
                video_token_loss = torch.where(
                    abs_diff < huber_c,
                    0.5 * video_diff ** 2,
                    huber_c * (abs_diff - 0.5 * huber_c)
                ).mean(dim=1)
                per_sample_video_loss = video_token_loss.flatten(1).mean(dim=1)  # [B]
            else:
                # MSE loss（原始行为）
                video_token_loss = (video_diff ** 2).mean(dim=1)
                per_sample_video_loss = video_token_loss.flatten(1).mean(dim=1)  # [B]

            if bool(getattr(self.config, 'kto_main_video_reweight', False)):
                with torch.no_grad():
                    token_error = video_diff.detach().float().abs().mean(dim=1)
                    token_ref = video_target.detach().float().abs().mean(dim=1)
                    threshold_tensor = None
                    if bool(getattr(self.config, 'kto_main_use_ema_threshold', True)):
                        error_ratio = token_error / (
                            token_ref + float(getattr(self.config, 'kto_eps', 1e-5)))
                        q = float(getattr(self.config, 'kto_main_threshold_quantile', 0.70))
                        current_threshold = torch.quantile(
                            error_ratio.detach().float().flatten(), q)
                        ema_decay = float(getattr(
                            self.config, 'kto_main_threshold_ema_decay', 0.90))
                        ema_decay = min(max(ema_decay, 0.0), 0.9999)
                        previous_threshold = getattr(
                            self, '_kto_main_error_threshold_ema', None)
                        if previous_threshold is None:
                            threshold_tensor = current_threshold.detach()
                        else:
                            threshold_tensor = (
                                previous_threshold.to(current_threshold.device) * ema_decay
                                + current_threshold.detach() * (1.0 - ema_decay)
                            )
                        self._kto_main_error_threshold_ema = threshold_tensor.detach()

                    main_focal = compute_normalized_focal_weights(
                        token_error,
                        token_ref,
                        threshold=threshold_tensor,
                        threshold_quantile=float(getattr(
                            self.config, 'kto_main_threshold_quantile', 0.70)),
                        eps=float(getattr(self.config, 'kto_eps', 1e-5)),
                        alpha=float(getattr(self.config, 'kto_main_alpha', 1.0)),
                        temperature=float(getattr(
                            self.config, 'kto_main_temperature',
                            getattr(self.config, 'kto_temperature', 0.10))),
                        min_weight=float(getattr(
                            self.config, 'kto_main_min_weight',
                            getattr(self.config, 'kto_min_weight', 0.5))),
                        max_weight=float(getattr(
                            self.config, 'kto_main_max_weight',
                            getattr(self.config, 'kto_max_weight', 1.8))),
                    )
                    kto_main_active = torch.tensor(1.0, device=self.device)
                    kto_main_good_ratio = main_focal.hard_ratio
                    kto_main_weight_mean = main_focal.weights.float().mean()
                    kto_main_weight_min = main_focal.weights.float().min()
                    kto_main_weight_max = main_focal.weights.float().max()
                    kto_main_threshold = main_focal.threshold.float().mean()
                per_sample_video_loss = (
                    video_token_loss * main_focal.weights.detach()
                ).flatten(1).mean(dim=1)

            # AnyFlow 技巧：用扩散样本的 loss 均值缩放非扩散样本
            # 这有助于平衡不同模式的梯度贡献
            # 注意：缩放因子在 no_grad 块内计算，但最终 loss 赋值必须在块外以保留梯度
            scale_weights = torch.ones_like(per_sample_video_loss)
            with torch.no_grad():
                if video_is_diffusion.any():
                    diffusion_mean = per_sample_video_loss[video_is_diffusion].mean()
                    non_diffusion_mask = ~video_is_diffusion
                    if non_diffusion_mask.any():
                        non_diffusion_mean = per_sample_video_loss[non_diffusion_mask].mean().clamp(min=1e-5)
                        scale_weights[non_diffusion_mask] = diffusion_mean / non_diffusion_mean
            per_sample_video_loss = per_sample_video_loss * scale_weights

            # no_grad 块外：最终加权 loss 保留梯度
            video_loss = (per_sample_video_loss * weight).mean()

        # ==============================================================
        # 步骤 11: 总损失
        # ==============================================================
        video_loss_weight = getattr(self.config, 'video_loss_weight', 1.0)
        action_block_weight = getattr(self.config, 'action_block_weight', 1.0)
        loss = video_loss_weight * video_loss \
               + action_block_weight * (
                   self.config.action_loss_weight * action_loss \
                   + self.gt_regression_weight * gt_regression_loss \
                   + getattr(self.config, 'action_aware_weight', 0.0) * action_aware_loss
               )

        flowmap_aux_weight = float(getattr(self.config, 'flowmap_aux_weight', 1.0))
        loss = loss * flowmap_aux_weight
        loss = loss / self.gradient_accumulation_steps

        # Loss clipping（防止 outlier 梯度）
        loss_clip_value = getattr(self.config, 'loss_clip_value', None)
        if loss_clip_value is not None and getattr(self.config, 'loss_clip_enabled', True):
            loss_clip_value = float(loss_clip_value)
            if loss_clip_value >= 0:
                # Preserve gradient direction when clipping large FlowMap losses.
                scale = (loss_clip_value / loss.detach().clamp(min=1e-12)).clamp(max=1.0)
                loss = loss * scale

        # Agree before backward so every rank follows the same collective path.
        local_nonfinite_origins = {
            "video": not bool(torch.isfinite(video_loss.detach()).all().item()),
            "action": not bool(torch.isfinite(action_loss.detach()).all().item())
            or not bool(torch.isfinite(action_aware_loss.detach()).all().item()),
            "teacher_or_gt": not bool(
                torch.isfinite(gt_regression_loss.detach()).all().item()
            ),
        }
        is_finite, nonfinite_origins = _synchronized_nonfinite_decision(
            loss=loss,
            local_flags=local_nonfinite_origins,
            device=self.device,
        )
        if not is_finite:
            if self.config.rank == 0:
                logger.warning(f"[step {self.step}] NaN/Inf loss, skipping")
            zero_loss = torch.zeros((), device=self.device)
            return {
                "loss": zero_loss,
                "video_loss": video_loss.detach(),
                "action_loss": action_loss.detach() if self.distill_action else action_loss,
                "action_local_fm_loss": action_local_fm_loss.detach() if self.action_aware else action_local_fm_loss,
                "action_aware_loss": action_aware_loss.detach() if self.action_aware else action_aware_loss,
                "gt_regression_loss": gt_regression_loss.detach() if self.distill_action else gt_regression_loss,
                "kto_main_active": kto_main_active.detach(),
                "kto_main_good_ratio": kto_main_good_ratio.detach(),
                "kto_main_weight_mean": kto_main_weight_mean.detach(),
                "kto_main_weight_min": kto_main_weight_min.detach(),
                "kto_main_weight_max": kto_main_weight_max.detach(),
                "kto_main_threshold": kto_main_threshold.detach(),
                "should_sync": should_sync,
                "skip_step": True,
                "nonfinite_origins": nonfinite_origins,
            }

        loss.backward()


        return {
            "loss": loss.detach(),
            "video_loss": video_loss.detach(),
            "action_loss": action_loss.detach() if self.distill_action else action_loss,
            "action_local_fm_loss": action_local_fm_loss.detach() if self.action_aware else action_local_fm_loss,
            "action_aware_loss": action_aware_loss.detach() if self.action_aware else action_aware_loss,
            "gt_regression_loss": gt_regression_loss.detach() if self.distill_action else gt_regression_loss,
            "kto_main_active": kto_main_active.detach(),
            "kto_main_good_ratio": kto_main_good_ratio.detach(),
            "kto_main_weight_mean": kto_main_weight_mean.detach(),
            "kto_main_weight_min": kto_main_weight_min.detach(),
            "kto_main_weight_max": kto_main_weight_max.detach(),
            "kto_main_threshold": kto_main_threshold.detach(),
            "should_sync": should_sync,
            "skip_step": False,
            "nonfinite_origins": nonfinite_origins,
        }

    # ==================================================================
    # On-Policy Rollout + DMD 训练步（Phase 5: On-Policy DMD）
    # ==================================================================

    def _sync_student_nofsdp(self):
        """Sync weights from FSDP student to non-FSDP copy."""
        if not hasattr(self, '_student_nofsdp'):
            return
        import torch.distributed.tensor as _dt
        with torch.no_grad():
            nofsdp_dict = dict(self._student_nofsdp.named_parameters())
            synced = 0
            for name, fsdp_p in self.student.named_parameters():
                if name in nofsdp_dict:
                    nofsdp_p = nofsdp_dict[name]
                    if isinstance(fsdp_p, _dt.DTensor):
                        full = fsdp_p.full_tensor()
                    else:
                        full = fsdp_p
                    nofsdp_p.copy_(full.to(nofsdp_p.dtype))
                    synced += 1
            # Debug: check first param type
            first_name, first_p = next(self._student_nofsdp.named_parameters())

    # ==================================================================
    def _student_euler_integrate(self, noisy_latents, timesteps, target_r, base_input_dict,
                                   empty_emb, cfg_scale, ref_shape, B, num_frames,
                                   K_steps=1, action_target_r=None,
                                   return_last_step_start=False,
                                   return_final_action=False,
                                   return_trajectory=False,
                                   return_final_action_state=False,
                                   force_no_grad=False):
        """
        Student multi-step Euler integration from t to target_r.

        OPD can choose how much rollout gradient to keep:
          - endpoint: old behavior, rollout in no_grad and only terminal forward has grad.
          - last_step: no_grad for early rollout, keep grad through the final Euler step.
          - full: keep grad through the full student rollout.
        """
        from modules.model import FlexAttnFunc
        _ACTION_DS = getattr(self.config, 'action_downsample_factor', 4)
        action_target_r_ds = None
        if action_target_r is not None:
            action_target_r_ds = action_target_r[:, ::_ACTION_DS]
        joint_action_rollout = bool(
            getattr(self.config, 'opd_joint_action_rollout', False)
        ) and action_target_r_ds is not None
        if return_final_action_state and not joint_action_rollout:
            raise ValueError(
                "return_final_action_state requires opd_joint_action_rollout=True "
                "and action_target_r."
            )

        rollout_grad_mode = getattr(self.config, 'opd_rollout_grad_mode', 'endpoint')
        rollout_grad_mode = str(rollout_grad_mode).lower()
        rollout_grad_steps = int(getattr(
            self.config, 'opd_rollout_grad_steps', 1
        ))
        if rollout_grad_mode not in SUPPORTED_ROLLOUT_GRAD_MODES:
            raise ValueError(
                f"Invalid opd_rollout_grad_mode={rollout_grad_mode!r}; "
                "expected endpoint, last_step, suffix, or full."
            )
        force_eval_checkpointing = bool(
            getattr(self.config, 'offline_eval_force_gradient_checkpointing', False)
        )
        force_cfg = bool(getattr(self.config, 'offline_eval_force_cfg', True))

        use_nofsdp_rollout = any(
            not rollout_step_requires_grad(
                mode=rollout_grad_mode,
                step_index=step_index,
                num_steps=K_steps,
                suffix_steps=rollout_grad_steps,
            )
            for step_index in range(K_steps)
        )
        _rollout_model = (getattr(self, '_student_nofsdp', None) or self.student) if use_nofsdp_rollout else self.student
        if _rollout_model is not self.student:
            if not getattr(self, '_nofsdp_synced', False):
                self._sync_student_nofsdp()
                self._nofsdp_synced = True
            # Observation-only probes enter ``diagnostic_runtime`` in eval mode.
            # Do not silently undo that contract on the no-FSDP rollout copy.
            if not force_no_grad:
                _rollout_model.train()
            _rt_latent = {k: _to_regular_tensor(v) if isinstance(v, torch.Tensor) else v
                          for k, v in base_input_dict['latent_dict'].items()}
            _rt_action = {k: _to_regular_tensor(v) if isinstance(v, torch.Tensor) else v
                          for k, v in base_input_dict['action_dict'].items()}
            _rt_empty = _to_regular_tensor(empty_emb)
            noisy_latents = _to_regular_tensor(noisy_latents)
            timesteps = _to_regular_tensor(timesteps)
        else:
            _rt_latent = base_input_dict['latent_dict']
            _rt_action = base_input_dict['action_dict']
            _rt_empty = empty_emb

        current_action = None
        action_step_ts = None
        if joint_action_rollout:
            current_action = base_input_dict['action_dict']['noisy_latents']
            action_timesteps = base_input_dict['action_dict']['timesteps']
            if current_action.shape[2] != action_target_r_ds.shape[1]:
                raise ValueError(
                    "joint action rollout requires matching action state and target "
                    f"horizons, got {current_action.shape[2]} and {action_target_r_ds.shape[1]}"
                )
            action_step_ts = self._build_timestep_path(
                action_timesteps.float(), action_target_r_ds.float(), K_steps
            )
            if _rollout_model is not self.student:
                current_action = _to_regular_tensor(current_action)
                action_step_ts = _to_regular_tensor(action_step_ts)

        def _init_mask(step_input):
            _ld = step_input['latent_dict']
            _ad = step_input['action_dict']
            _total_length = (
                _ld['noisy_latents'].flatten(0, 1).shape[0] * 2 +
                _ad['noisy_latents'].flatten(0, 1).shape[0] * 2
            )
            _padded_length = (128 - _total_length % 128) % 128
            FlexAttnFunc.init_mask(
                _ld['noisy_latents'].shape,
                _ad['noisy_latents'].shape,
                _padded_length,
                step_input['chunk_size'],
                window_size=step_input['window_size'],
                patch_size=self.patch_size,
                device=self.device,
            )

        def _student_cfg_with_optional_uncompile(
            model,
            step_input,
            step_empty,
            r_timestep_i,
            action_r_timestep_i,
            *,
            return_action=False,
        ):
            saved_blocks = None
            if model is self.student and getattr(self, '_student_blocks_compiled', False):
                saved_blocks = list(self.student.blocks)
                for bi, block in enumerate(saved_blocks):
                    if hasattr(block, '_orig_mod'):
                        self.student.blocks[bi] = block._orig_mod
            try:
                return self._student_cfg_forward(
                    model,
                    step_input,
                    step_empty,
                    cfg_scale,
                    B,
                    ref_shape,
                    r_timestep_i,
                    action_r_timestep_i,
                    force_cfg=force_cfg,
                    return_action=return_action,
                )
            finally:
                if saved_blocks is not None:
                    for bi, block in enumerate(saved_blocks):
                        self.student.blocks[bi] = block

        step_ts = self._build_timestep_path(timesteps.float(), target_r.float(), K_steps)
        current_x = noisy_latents
        trajectory = [current_x.detach()] if return_trajectory else None
        last_step_start_x = current_x
        last_step_start_t = step_ts[0]

        for i in range(K_steps):
            t_i = step_ts[i]
            r_next = step_ts[min(i + 1, K_steps)]
            if i == K_steps - 1:
                last_step_start_x = current_x.detach()
                last_step_start_t = t_i.detach()

            keep_step_grad = (not force_no_grad) and rollout_step_requires_grad(
                mode=rollout_grad_mode,
                step_index=i,
                num_steps=K_steps,
                suffix_steps=rollout_grad_steps,
            )
            step_model = self.student if keep_step_grad else _rollout_model
            step_latent = base_input_dict['latent_dict'] if keep_step_grad else _rt_latent
            step_action = base_input_dict['action_dict'] if keep_step_grad else _rt_action
            step_empty = empty_emb if keep_step_grad else _rt_empty
            if keep_step_grad and current_x.device != self.device:
                current_x = current_x.to(self.device)
            if keep_step_grad and current_x.dtype != base_input_dict['latent_dict']['latent'].dtype:
                current_x = current_x.to(base_input_dict['latent_dict']['latent'].dtype)
            if joint_action_rollout and keep_step_grad:
                if current_action.device != self.device:
                    current_action = current_action.to(self.device)
                if current_action.dtype != base_input_dict['action_dict']['latent'].dtype:
                    current_action = current_action.to(
                        base_input_dict['action_dict']['latent'].dtype
                    )

            if joint_action_rollout:
                action_t_i = action_step_ts[i]
                action_r_i = action_step_ts[min(i + 1, K_steps)]
                step_action = {
                    **step_action,
                    'noisy_latents': current_action,
                    'timesteps': action_t_i,
                }
            else:
                action_t_i = action_r_i = None
            step_input = {
                'latent_dict': {
                    **step_latent,
                    'noisy_latents': current_x,
                    'timesteps': t_i,
                },
                'action_dict': step_action,
                'chunk_size': base_input_dict['chunk_size'],
                'window_size': base_input_dict['window_size'],
            }
            _init_mask(step_input)

            r_timestep_i = r_next
            action_r_timestep_i = (
                action_r_i if joint_action_rollout
                else (
                    action_target_r_ds if action_target_r_ds is not None
                    else (r_timestep_i[:, ::_ACTION_DS] if (self.distill_action or self.action_aware) else r_timestep_i)
                )
            )

            if force_no_grad:
                grad_context = torch.no_grad()
            elif force_eval_checkpointing:
                grad_context = torch.enable_grad()
            else:
                grad_context = torch.enable_grad() if keep_step_grad else torch.no_grad()
            with grad_context:
                step_forward = _student_cfg_with_optional_uncompile(
                    step_model,
                    step_input,
                    step_empty,
                    r_timestep_i,
                    action_r_timestep_i,
                    return_action=joint_action_rollout,
                )
                if joint_action_rollout:
                    v_cfg, step_action_seq = step_forward
                    if step_action_seq is None:
                        raise RuntimeError("joint action rollout requires the student action head")
                    step_action_v = self._extract_action_v(
                        step_action_seq, current_action.shape[2]
                    )
                else:
                    v_cfg = step_forward
                if force_eval_checkpointing and not keep_step_grad:
                    v_cfg = v_cfg.detach()
                sigma_i = self._timestep_to_sigma_5d(t_i)
                sigma_next = self._timestep_to_sigma_5d(r_next)
                current_x = current_x + v_cfg * (sigma_next - sigma_i)
                if joint_action_rollout:
                    action_sigma_i = action_t_i[:, None, :, None, None] / self.config.num_train_timesteps
                    action_sigma_next = action_r_i[:, None, :, None, None] / self.config.num_train_timesteps
                    current_action = current_action + step_action_v * (
                        action_sigma_next.to(step_action_v) - action_sigma_i.to(step_action_v)
                    )
                if force_eval_checkpointing and not keep_step_grad:
                    current_x = current_x.detach()
                    if joint_action_rollout:
                        current_action = current_action.detach()
                if return_trajectory:
                    trajectory.append(current_x.detach())

        current_x_for_loss = current_x if rollout_grad_mode != 'endpoint' else current_x.detach()
        current_x_for_forward = current_x.detach()
        current_action_for_loss = (
            current_action if rollout_grad_mode != 'endpoint' else current_action.detach()
        ) if joint_action_rollout else None
        current_action_for_forward = current_action.detach() if joint_action_rollout else None

        final_input = {
            'latent_dict': {
                **base_input_dict['latent_dict'],
                'noisy_latents': current_x_for_forward,
                'timesteps': target_r,
            },
            'action_dict': (
                {
                    **base_input_dict['action_dict'],
                    'noisy_latents': current_action_for_forward,
                    'timesteps': action_target_r_ds,
                }
                if joint_action_rollout else base_input_dict['action_dict']
            ),
            'chunk_size': base_input_dict['chunk_size'],
            'window_size': base_input_dict['window_size'],
        }
        _init_mask(final_input)

        _act_r_final = (
            action_target_r_ds if action_target_r_ds is not None
            else (target_r[:, ::_ACTION_DS] if (self.distill_action or self.action_aware) else target_r)
        )
        _saved_blocks = None
        _is_compiled = getattr(self, '_student_blocks_compiled', False)
        if _is_compiled:
            _saved_blocks = list(self.student.blocks)
            for _bi, _block in enumerate(_saved_blocks):
                if hasattr(_block, '_orig_mod'):
                    self.student.blocks[_bi] = _block._orig_mod
        try:
            final_grad_context = (
                torch.no_grad()
                if force_no_grad
                else (
                    torch.enable_grad()
                    if force_eval_checkpointing
                    else contextlib.nullcontext()
                )
            )
            with final_grad_context:
                final_forward = self._student_cfg_forward(
                    self.student, final_input, empty_emb, cfg_scale,
                    B, ref_shape, target_r, _act_r_final, force_cfg=False,
                    return_action=return_final_action,
                )
                if return_final_action:
                    v_final_cfg, final_action_seq = final_forward
                else:
                    v_final_cfg = final_forward
                    final_action_seq = None
                if force_eval_checkpointing:
                    v_final_cfg = v_final_cfg.detach()
                    if isinstance(final_action_seq, torch.Tensor):
                        final_action_seq = final_action_seq.detach()
        finally:
            if _saved_blocks is not None:
                for _bi, _block in enumerate(_saved_blocks):
                    self.student.blocks[_bi] = _block

        if return_last_step_start:
            if return_trajectory:
                raise ValueError("return_trajectory is incompatible with return_last_step_start.")
            if return_final_action:
                return (
                    current_x_for_loss,
                    v_final_cfg,
                    final_action_seq,
                    last_step_start_x.detach(),
                    last_step_start_t.detach(),
                )
            return current_x_for_loss, v_final_cfg, last_step_start_x.detach(), last_step_start_t.detach()

        if return_final_action:
            if return_trajectory:
                if return_final_action_state:
                    return (
                        current_x_for_loss,
                        v_final_cfg,
                        final_action_seq,
                        current_action_for_loss,
                        tuple(trajectory),
                    )
                return current_x_for_loss, v_final_cfg, final_action_seq, tuple(trajectory)
            if return_final_action_state:
                return (
                    current_x_for_loss,
                    v_final_cfg,
                    final_action_seq,
                    current_action_for_loss,
                )
            return current_x_for_loss, v_final_cfg, final_action_seq
        if return_trajectory:
            return current_x_for_loss, v_final_cfg, tuple(trajectory)
        return current_x_for_loss, v_final_cfg


    def _teacher_integrate_to_r(self, noisy_latents, timesteps, target_r, input_dict, empty_emb,
                                 cfg_scale, ref_shape, B, num_frames, num_steps=2,
                                 return_trajectory=False):
        """
        Teacher multi-step Euler integration from t to target_r (no_grad).

        使用 teacher 模型做 num_steps 步 Euler 积分，
        从 t 的噪声样本出发，到达 target_r 处的"teacher 认为的"状态。

        参数:
            noisy_latents: 起始噪声 latent [B, C, F, H, W]
            timesteps: 起始时间步 [B, T]
            target_r: 目标时间步 [B, T]
            input_dict: 基础 input_dict
            empty_emb: 空文本嵌入
            cfg_scale: CFG 引导强度
            ref_shape: 参考形状
            B: batch size
            num_frames: 帧数
            num_steps: 积分步数

        返回:
            current_x: teacher rollout 到 target_r 后的 latent [B, C, F, H, W]
            v_teacher: teacher 在 target_r 处的 v-prediction [B, C, F, H, W]
        """
        _ACTION_DS = getattr(self.config, 'action_downsample_factor', 4)

        # 计算按样本的时间步序列
        step_ts = self._build_timestep_path(timesteps.float(), target_r.float(), num_steps)

        current_x = noisy_latents
        trajectory = [current_x.detach()] if return_trajectory else None

        with torch.no_grad():
            for i in range(num_steps):
                t_i = step_ts[i]
                r_i = step_ts[i + 1]

                # 构建 teacher input
                teacher_input = {
                    'latent_dict': {
                        **input_dict['latent_dict'],
                        'noisy_latents': current_x,
                        'timesteps': t_i,
                    },
                    'action_dict': input_dict['action_dict'],
                    'chunk_size': input_dict['chunk_size'],
                    'window_size': input_dict['window_size'],
                }

                # Teacher CFG forward
                v_cond, v_uncond, _ = self._batched_cfg_forward(teacher_input, empty_emb)
                v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
                v_cfg_5d = self._extract_video_v(v_cfg, ref_shape, B)

                # Euler step
                sigma_i = self._timestep_to_sigma_5d(t_i)
                sigma_next = self._timestep_to_sigma_5d(r_i)
                current_x = current_x + v_cfg_5d * (sigma_next - sigma_i)
                if return_trajectory:
                    trajectory.append(current_x.detach())

            # 最终 teacher v-prediction at target_r
            final_teacher_input = {
                'latent_dict': {
                    **input_dict['latent_dict'],
                    'noisy_latents': current_x,
                    'timesteps': target_r,
                },
                'action_dict': input_dict['action_dict'],
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
            v_cond_final, v_uncond_final, _ = self._batched_cfg_forward(final_teacher_input, empty_emb)
            v_cfg_final = v_uncond_final + cfg_scale * (v_cond_final - v_uncond_final)
            v_teacher = self._extract_video_v(v_cfg_final, ref_shape, B)

        if return_trajectory:
            return current_x, v_teacher, tuple(trajectory)
        return current_x, v_teacher

    def _teacher_forward_at_student_state(self, student_x_r, target_r, input_dict, empty_emb,
                                          cfg_scale, ref_shape, B,
                                          action_input_dict=None,
                                          action_frames=None):
        """
        Query the teacher at the state actually visited by the student rollout.

        This keeps the transition target on the student's rollout distribution
        instead of comparing student and teacher predictions at two different
        terminal states.
        """
        with torch.no_grad():
            teacher_input = {
                'latent_dict': {
                    **input_dict['latent_dict'],
                    'noisy_latents': student_x_r.detach(),
                    'timesteps': target_r,
                },
                'action_dict': (
                    action_input_dict
                    if action_input_dict is not None
                    else input_dict['action_dict']
                ),
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
            v_cond, v_uncond, action_cond = self._batched_cfg_forward(
                teacher_input, empty_emb)
            v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
            v_teacher = self._extract_video_v(v_cfg, ref_shape, B)
            if action_input_dict is not None:
                if action_cond is None:
                    raise RuntimeError(
                        "Teacher did not return action output for fused OPD action target.")
                if action_frames is None:
                    action_frames = action_input_dict['noisy_latents'].shape[2]
                action_teacher = self._extract_action_v(action_cond, action_frames)
                return student_x_r.detach(), v_teacher, action_teacher

        return student_x_r.detach(), v_teacher

    def _onpolicy_transition_step(self, batch, batch_idx):
        """
        On-Policy Transition Matching 训练步（替代 DMD）。

        核心思路：
          1. 采样 (t, r) 对，加噪得到 x_t
          2. Student 多步 Euler rollout：x_t → x_r（保留梯度图）
          3. Teacher 多步 Euler 积分：x_t → x_r_teacher（no_grad，提供 target）
          4. Loss = MSE(v_student(x_r, r), v_teacher(x_r, r))
                 + local_fm_loss(v_student(x_r, r), v_gt(x_r, r))

        与 _train_step 的核心区别：
          - _train_step: teacher 单步 CFG forward + 中心差分，student 单步预测
          - _onpolicy_transition: student 多步 rollout（on-policy），teacher 多步积分（target）

        返回:
            loss_dict: 包含各项损失的字典
        """

        batch = self.convert_input_format(batch)

        B = batch['latents'].shape[0]
        ref_shape = batch['latents'].shape
        num_frames = ref_shape[2]
        actions_mask = batch.get('actions_mask')
        _ACTION_DS = getattr(self.config, 'action_downsample_factor', 4)

        # ==============================================================
        # 步骤 1: 准备基础 input_dict
        # ==============================================================
        input_dict = self._prepare_base_dict(batch)

        # ==============================================================
        # 步骤 2: 混合时间步采样（复用 FlowMap 的采样逻辑）
        # ==============================================================
        video_t, video_r, video_is_diffusion = self.sample_timestep_mixed(
            B, num_frames, dtype=torch.float32, device=self.device,
        )
        video_t_sigma = video_t / self.config.num_train_timesteps
        video_r_sigma = video_r / self.config.num_train_timesteps

        if self.distill_action or self.action_aware:
            action_t, action_r, action_is_diffusion = self.sample_timestep_mixed(
                B, num_frames, dtype=torch.float32, device=self.device,
                scheduler=self.train_scheduler_action,
            )
            action_t_sigma = action_t / self.config.num_train_timesteps
            action_r_sigma = action_r / self.config.num_train_timesteps

        # ==============================================================
        # 步骤 3: 加噪
        # ==============================================================
        video_noise = torch.randn_like(batch['latents'])
        video_noisy_latents = self.train_scheduler_latent.add_noise(
            batch['latents'], video_noise, video_t, t_dim=2
        )
        video_v_target = self.train_scheduler_latent.training_target(
            batch['latents'], video_noise, video_t
        )

        if self.distill_action or self.action_aware:
            action_noise = torch.randn_like(batch['actions'])
            action_noisy_latents = self.train_scheduler_action.add_noise(
                batch['actions'], action_noise, action_t, t_dim=2
            )

        # 填充 input_dict（为 student/teacher Euler 积分准备）
        input_dict['latent_dict']['noisy_latents'] = video_noisy_latents
        input_dict['latent_dict']['timesteps'] = video_t
        input_dict['latent_dict']['targets'] = video_v_target
        if self.distill_action or self.action_aware:
            input_dict['action_dict']['noisy_latents'] = action_noisy_latents
            input_dict['action_dict']['timesteps'] = action_t
            input_dict['action_dict']['targets'] = self.train_scheduler_action.training_target(batch['actions'], action_noise, action_t)

        # ==============================================================
        # 步骤 4: 采样 (N, K) 步数对（分布式一致）
        # ==============================================================

        if self.distill_action or self.action_aware:
            _ad = input_dict['action_dict']
            student_input_dict = {
                'latent_dict': input_dict['latent_dict'],
                'action_dict': {
                    'noisy_latents': _ad['noisy_latents'][:, :, ::_ACTION_DS],
                    'latent':        _ad['latent'][:, :, ::_ACTION_DS],
                    'timesteps':     _ad['timesteps'][:, ::_ACTION_DS],
                    'cond_timesteps':_ad['cond_timesteps'][:, ::_ACTION_DS],
                    'text_emb':      _ad['text_emb'],
                    'grid_id':       _downsample_action_grid_id(
                        _ad['grid_id'], _ad['latent'], _ACTION_DS) if _ad.get('grid_id') is not None else None,
                    'actions_mask':  _ad['actions_mask'][:, :, ::_ACTION_DS] if _ad.get('actions_mask') is not None else None,
                },
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
        else:
            student_input_dict = input_dict

        rollout_step_pairs = getattr(
            self.config,
            'opd_rollout_step_pairs',
            getattr(self.config, 'rollout_step_pairs', [[1, 1]]),
        )
        if dist.is_initialized():
            idx = torch.randint(0, len(rollout_step_pairs), (1,), device=self.device)
            dist.broadcast(idx, src=0)
            N_steps, K_steps = rollout_step_pairs[idx.item()]
        else:
            import random
            N_steps, K_steps = random.choice(rollout_step_pairs)

        # ==============================================================
        # 步骤 5: CFG scale
        # ==============================================================
        cfg_scale = self.config.cfg_min + torch.rand(1).item() * (
            self.config.cfg_max - self.config.cfg_min)

        # 准备空文本嵌入
        B_emb = input_dict['latent_dict']['text_emb'].shape[0]
        empty_emb = self.empty_emb.expand(B_emb, -1, -1)

        # ==============================================================
        # 步骤 6: 梯度累积控制
        # ==============================================================
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        if not should_sync:
            if hasattr(self.student, 'set_requires_gradient_sync'):
                self.student.set_requires_gradient_sync(False)
        else:
            if hasattr(self.student, 'set_requires_gradient_sync'):
                self.student.set_requires_gradient_sync(True)

        # 步骤 7: Student 多步 Euler rollout（保留梯度图）
        # ==============================================================
        if self.distill_video:
            student_x_r, student_v_at_r, student_last_x, student_last_t = self._student_euler_integrate(
                noisy_latents=video_noisy_latents,
                timesteps=video_t,
                target_r=video_r,
                base_input_dict=student_input_dict,
                empty_emb=empty_emb,
                cfg_scale=cfg_scale,
                ref_shape=ref_shape,
                B=B,
                num_frames=num_frames,
                K_steps=K_steps,
                action_target_r=action_r if (self.distill_action or self.action_aware) else None,
                return_last_step_start=True,
            )

        # ==============================================================
        # 步骤 8: Teacher target（no_grad，提供 student-state target）
        # ==============================================================
        effective_teacher_steps = 1
        if self.distill_video:
            video_transition_param = getattr(self.config, 'video_transition_param', 'x0')
            effective_teacher_steps = max(1, N_steps)
            teacher_x_r, teacher_v_at_r = self._teacher_integrate_to_r(
                noisy_latents=video_noisy_latents,
                timesteps=video_t,
                target_r=video_r,
                input_dict=input_dict,
                empty_emb=empty_emb,
                cfg_scale=cfg_scale,
                ref_shape=ref_shape,
                B=B,
                num_frames=num_frames,
                num_steps=effective_teacher_steps,
            )
            if video_transition_param == 'velocity':
                video_transition_param = 'x0'
                if self.config.rank == 0 and not getattr(
                    self, '_warned_video_endpoint_uses_x0', False
                ):
                    logger.warning(
                        "Using x0 video transition loss for K-step student vs "
                        "N-step teacher endpoint distillation; velocity MSE is "
                        "only valid for same-state teacher labels."
                    )
                    self._warned_video_endpoint_uses_x0 = True

        # ==============================================================
        # 步骤 9: 计算 transition loss
        # ==============================================================
        video_transition_loss = torch.tensor(0.0, device=self.device)
        local_fm_loss = torch.tensor(0.0, device=self.device)
        action_loss = torch.tensor(0.0, device=self.device)
        action_aware_loss = torch.tensor(0.0, device=self.device)
        action_local_fm_loss = torch.tensor(0.0, device=self.device)
        gt_regression_loss = torch.tensor(0.0, device=self.device)

        if self.distill_video:
            # 9a. Transition loss.
            # Default is the original OPD velocity objective; distill-style mode
            # matches the validated consistency-distillation prediction target.
            # KTO-style pointwise adaptive weighting: bad tokens get higher weight
            transition_loss_type = getattr(self.config, 'transition_loss_type', 'huber')
            transition_huber_c = getattr(self.config, 'transition_huber_c', 1e-3)
            if video_transition_param == 'consistency':
                student_video_pred = self._consistency_function(
                    student_v_at_r, student_x_r, video_r_sigma
                )
                with torch.no_grad():
                    teacher_video_pred = self._consistency_function(
                        teacher_v_at_r, teacher_x_r, video_r_sigma
                    )
                video_diff = (student_video_pred.float() - teacher_video_pred.detach().float())
            elif video_transition_param == 'x0':
                sigma_r = video_r_sigma[:, None, :, None, None].to(student_v_at_r)
                student_video_pred = student_x_r - sigma_r * student_v_at_r
                teacher_video_pred = teacher_x_r - sigma_r * teacher_v_at_r
                video_diff = (student_video_pred.float() - teacher_video_pred.detach().float())
            else:
                video_diff = (student_v_at_r.float() - teacher_v_at_r.detach().float())

            if transition_loss_type == "huber":
                abs_diff = video_diff.abs()
                per_sample_loss = torch.where(
                    abs_diff < transition_huber_c,
                    0.5 * video_diff ** 2,
                    transition_huber_c * (abs_diff - 0.5 * transition_huber_c)
                ).mean(dim=[1, 2, 3, 4])
            else:
                per_sample_loss = (video_diff ** 2).mean(dim=[1, 2, 3, 4])

            # KTO-style pointwise adaptive weighting
            # Per-token similarity → per-sample good_ratio → adaptive weight
            # Bad tokens get higher weight ("losses loom larger than gains")
            kto_adaptive = getattr(self.config, 'kto_adaptive', False)
            if kto_adaptive:
                with torch.no_grad():
                    # Per-token relative error to teacher
                    token_error = video_diff.abs().mean(dim=[1]).flatten(1)  # [B, F*H*W]
                    teacher_mag = teacher_v_at_r.detach().float().abs().mean(dim=[1]).flatten(1)
                    token_similarity = (1.0 - token_error / (teacher_mag + 1e-5)).clamp(min=0.0)

                    # Dynamic threshold: median similarity per batch
                    threshold = getattr(self.config, 'kto_threshold', None)
                    if threshold is None:
                        threshold = token_similarity.median()
                    threshold = threshold.clamp(min=1e-3)

                    is_good = token_similarity > threshold

                good_ratio = is_good.float().mean(dim=1)  # [B]
                good_weight = getattr(self.config, 'kto_good_weight', 0.3)
                bad_weight = getattr(self.config, 'kto_bad_weight', 1.0)
                adaptive_weight = good_ratio * good_weight + (1.0 - good_ratio) * bad_weight
            else:
                adaptive_weight = torch.ones_like(per_sample_loss)

            # 时间步加权
            weight_type = getattr(self.config, 'weight_type', 'uniform')
            weight = self._get_timestep_weight(video_t.mean(dim=-1), weight_type).to(self.device)
            video_transition_loss = (per_sample_loss * adaptive_weight * weight).mean()

            # 9b. Local FM loss: student v vs GT v at (x_r, r)
            local_fm_weight = getattr(self.config, 'local_fm_weight', 0.05)
            if local_fm_weight > 0:
                video_v_target_at_r = self.train_scheduler_latent.training_target(
                    batch['latents'], video_noise, video_r
                )
                # 用 student rollout 后的 x_r 做 student 预测
                # student_v_at_r 已经是 student 在 x_r_student 处的预测
                # GT target 是 noise - x0（与 x 无关）
                local_diff = (student_v_at_r.float() - video_v_target_at_r.detach().float())
                local_fm_loss = (local_diff ** 2).mean()

        # ==============================================================
        # 步骤 10: 动作损失（如果有）
        # ==============================================================
        if self.distill_action or self.action_aware:
            # 动作使用标准 FlowMap 逻辑（不做 rollout）；action_aware-only 也需要完整动作前向。
            action_noisy_ds = action_noisy_latents[:, :, ::_ACTION_DS]
            video_context_r = student_x_r.detach() if self.distill_video else video_noisy_latents

            action_v_5d = None
            x_prev_action = None
            if self.distill_action:
                with torch.no_grad():
                    _, _, action_v_cond = self._batched_cfg_forward(input_dict, empty_emb)
                    action_v_5d = self._extract_action_v(action_v_cond, num_frames)
                    sigma_s_a = action_t_sigma[:, None, :, None, None].to(action_v_5d)
                    sigma_r_a = action_r_sigma[:, None, :, None, None].to(action_v_5d)
                    x_prev_action = action_noisy_latents + action_v_5d * (sigma_r_a - sigma_s_a)

            # Student action prediction at r
            student_action_input = {
                'latent_dict': {
                    **input_dict['latent_dict'],
                    'noisy_latents': video_context_r,
                    'timesteps': video_r,
                },
                'action_dict': {
                    'noisy_latents': action_noisy_latents[:, :, ::_ACTION_DS],
                    'latent': batch['actions'][:, :, ::_ACTION_DS],
                    'timesteps': action_t[:, ::_ACTION_DS],
                    'cond_timesteps': input_dict['action_dict']['cond_timesteps'][:, ::_ACTION_DS],
                    'text_emb': input_dict['action_dict']['text_emb'],
                },
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
            if 'grid_id' in input_dict['action_dict'] and input_dict['action_dict']['grid_id'] is not None:
                student_action_input['action_dict']['grid_id'] = _downsample_action_grid_id(
                    input_dict['action_dict']['grid_id'], batch['actions'], _ACTION_DS)
            if 'actions_mask' in input_dict['action_dict'] and input_dict['action_dict']['actions_mask'] is not None:
                student_action_input['action_dict']['actions_mask'] = input_dict['action_dict']['actions_mask'][:, :, ::_ACTION_DS]

            from modules.model import FlexAttnFunc
            _ld = student_action_input['latent_dict']
            _ad = student_action_input['action_dict']
            _total_length = (
                _ld['noisy_latents'].flatten(0, 1).shape[0] * 2 +
                _ad['noisy_latents'].flatten(0, 1).shape[0] * 2
            )
            _padded_length = (128 - _total_length % 128) % 128
            FlexAttnFunc.init_mask(
                _ld['noisy_latents'].shape,
                _ad['noisy_latents'].shape,
                _padded_length,
                student_action_input['chunk_size'],
                window_size=student_action_input['window_size'],
                patch_size=self.patch_size,
                device=self.device,
            )

            action_r_ds = action_r[:, ::_ACTION_DS]
            _, student_action_v_seq = self.student(
                student_action_input, train_mode=True,
                r_timestep=video_r,
                action_r_timestep=action_r_ds,
            )
            student_action_v = self._extract_action_v(student_action_v_seq, student_action_input['action_dict']['noisy_latents'].shape[2])

            # Action loss (x0 prediction vs EMA target)
            actions_mask_ds = actions_mask[:, :, ::_ACTION_DS]
            actions_gt_ds = batch['actions'][:, :, ::_ACTION_DS]
            action_sigma_s = action_r_sigma[:, None, ::_ACTION_DS, None, None].to(student_action_v)
            student_action_pred = action_noisy_ds - action_sigma_s * student_action_v
            mask = actions_mask_ds.float()

            action_denom = (mask.sum() * student_action_pred.shape[1]).clamp(min=1)
            if self.distill_action:
                with torch.no_grad():
                    target_input = {
                        'latent_dict': {
                            **input_dict['latent_dict'],
                            'noisy_latents': video_context_r,
                            'timesteps': video_r,
                        },
                        'action_dict': {
                            'noisy_latents': x_prev_action.detach()[:, :, ::_ACTION_DS],
                            'latent': batch['actions'][:, :, ::_ACTION_DS],
                            'timesteps': action_r[:, ::_ACTION_DS],
                            'cond_timesteps': input_dict['action_dict']['cond_timesteps'][:, ::_ACTION_DS],
                            'text_emb': input_dict['action_dict']['text_emb'],
                        },
                        'chunk_size': input_dict['chunk_size'],
                        'window_size': input_dict['window_size'],
                    }
                    if 'grid_id' in input_dict['action_dict'] and input_dict['action_dict']['grid_id'] is not None:
                        target_input['action_dict']['grid_id'] = _downsample_action_grid_id(
                            input_dict['action_dict']['grid_id'], batch['actions'], _ACTION_DS)
                    if 'actions_mask' in input_dict['action_dict'] and input_dict['action_dict']['actions_mask'] is not None:
                        target_input['action_dict']['actions_mask'] = input_dict['action_dict']['actions_mask'][:, :, ::_ACTION_DS]

                    _, target_action_v_seq = self.target_student(
                        target_input, train_mode=True,
                        r_timestep=video_r,
                        action_r_timestep=action_r_ds,
                    )
                    target_action_v = self._extract_action_v(target_action_v_seq, target_input['action_dict']['noisy_latents'].shape[2])
                    sigma_r_a_5d = action_r_sigma[:, None, ::_ACTION_DS, None, None].to(target_action_v)
                    x_prev_action_ds = x_prev_action[:, :, ::_ACTION_DS]
                    target_action_pred = x_prev_action_ds - sigma_r_a_5d * target_action_v

                action_diff = (student_action_pred.float() * mask) - (target_action_pred.detach().float() * mask)
                action_loss = (action_diff ** 2).sum() / action_denom

                # GT regression
                if getattr(self.config, 'use_gt_regression', True) and self.gt_regression_weight > 0:
                    gt_diff = (student_action_pred.float() * mask) - (actions_gt_ds.float() * mask)
                    gt_regression_loss = (gt_diff ** 2).sum() / action_denom

            # Action local FM regularization: student action velocity vs GT FM target.
            # action_aware_loss is kept as a backward-compatible alias.
            if self.action_aware:
                action_targets = input_dict['action_dict']['targets'][:, :, ::_ACTION_DS]
                aa_diff = (student_action_v.float() - action_targets.float().detach()) * mask
                action_local_fm_loss = (aa_diff ** 2).sum() / action_denom
                action_aware_loss = action_local_fm_loss

        # ==============================================================
        # 步骤 11: 总损失
        # ==============================================================
        video_transition_weight = getattr(self.config, 'video_transition_weight', 1.0)
        local_fm_w = getattr(self.config, 'local_fm_weight', 0.05)

        action_block_weight = getattr(self.config, 'action_block_weight', 1.0)
        loss = (
            video_transition_weight * video_transition_loss
            + local_fm_w * local_fm_loss
            + action_block_weight * (
                self.config.action_loss_weight * action_loss
                + self.gt_regression_weight * gt_regression_loss
                + getattr(self.config, 'action_aware_weight', 0.0) * action_aware_loss
            )
        )

        loss = loss / self.gradient_accumulation_steps

        # Loss clipping
        loss_clip_value = getattr(self.config, 'loss_clip_value', None)
        if loss_clip_value is not None and getattr(self.config, 'loss_clip_enabled', True):
            loss = torch.clamp(loss, min=0.0, max=loss_clip_value)
        is_finite, nonfinite_origins = _synchronized_nonfinite_decision(
            loss=loss,
            local_flags={
                "video": not bool(
                    torch.isfinite(video_transition_loss.detach()).all().item()
                )
                or not bool(torch.isfinite(local_fm_loss.detach()).all().item()),
                "action": not bool(
                    torch.isfinite(action_loss.detach()).all().item()
                )
                or not bool(
                    torch.isfinite(action_aware_loss.detach()).all().item()
                ),
                "teacher_or_gt": not bool(
                    torch.isfinite(gt_regression_loss.detach()).all().item()
                ),
            },
            device=self.device,
        )
        if not is_finite:
            if self.config.rank == 0:
                logger.warning(f"[step {self.step}] NaN/Inf loss, skipping")
            zero_loss = torch.zeros((), device=self.device)
            return {
                'loss': zero_loss,
                'video_transition_loss': video_transition_loss.detach(),
                'local_fm_loss': local_fm_loss.detach(),
                'action_loss': action_loss.detach() if self.distill_action else action_loss,
                'action_local_fm_loss': action_local_fm_loss.detach() if self.action_aware else action_local_fm_loss,
                'action_aware_loss': action_aware_loss.detach() if self.action_aware else action_aware_loss,
                'gt_regression_loss': gt_regression_loss.detach() if self.distill_action else gt_regression_loss,
                'should_sync': should_sync,
                'skip_step': True,
                'nonfinite_origins': nonfinite_origins,
            }

        loss.backward()

        empty_cache_interval = getattr(self.config, 'onpolicy_empty_cache_interval', 0)
        if empty_cache_interval and (self.step % empty_cache_interval == 0):
            torch.cuda.empty_cache()

        return {
            'loss': loss.detach(),
            'video_transition_loss': video_transition_loss.detach(),
            'local_fm_loss': local_fm_loss.detach(),
            'action_loss': action_loss.detach() if self.distill_action else action_loss,
            'action_local_fm_loss': action_local_fm_loss.detach() if self.action_aware else action_local_fm_loss,
            'action_aware_loss': action_aware_loss.detach() if self.action_aware else action_aware_loss,
            'gt_regression_loss': gt_regression_loss.detach() if self.distill_action else gt_regression_loss,
            'rollout_steps': K_steps,
            'teacher_steps': effective_teacher_steps,
            'video_t_mean': video_t.mean().item(),
            'video_r_mean': video_r.mean().item(),
            'should_sync': should_sync,
            'skip_step': False,
            'nonfinite_origins': nonfinite_origins,
        }

    def _opd_aux_transition_step_kto_paopd(self, batch, batch_idx):
        return self._opd_aux_transition_step(
            batch,
            batch_idx,
            kto_paopd=True,
        )

    def _build_joint_input(
        self,
        video_x,
        video_t,
        action_x,
        action_t,
        *,
        video_base,
        action_latent,
        action_cond_t,
        action_text,
        action_grid,
        action_valid_mask,
        chunk_size,
        window_size,
    ):
        """Build one explicit Cosmos/WanVA joint state without stale GT tensors."""
        action_dict = {
            "noisy_latents": action_x,
            "latent": action_latent,
            "timesteps": action_t,
            "cond_timesteps": action_cond_t,
            "text_emb": action_text,
        }
        if action_grid is not None:
            action_dict["grid_id"] = action_grid
        if action_valid_mask is not None:
            action_dict["actions_mask"] = action_valid_mask
        return {
            "latent_dict": {
                **video_base,
                "noisy_latents": video_x,
                "timesteps": video_t,
            },
            "action_dict": action_dict,
            "chunk_size": chunk_size,
            "window_size": window_size,
        }

    def _init_joint_mask(self, joint_input):
        """Initialize the actual packed attention mask for a joint query."""
        from modules.model import FlexAttnFunc

        latent_dict = joint_input["latent_dict"]
        action_dict = joint_input["action_dict"]
        total_length = (
            latent_dict["noisy_latents"].flatten(0, 1).shape[0] * 2
            + action_dict["noisy_latents"].flatten(0, 1).shape[0] * 2
        )
        padded_length = (128 - total_length % 128) % 128
        FlexAttnFunc.init_mask(
            latent_dict["noisy_latents"].shape,
            action_dict["noisy_latents"].shape,
            padded_length,
            joint_input["chunk_size"],
            window_size=joint_input["window_size"],
            patch_size=self.patch_size,
            device=self.device,
        )

    def _student_joint_forward(
        self,
        model,
        joint_input,
        model_empty_emb,
        video_r_t,
        action_r_t,
        *,
        cfg_scale,
        batch_size,
        ref_shape,
        require_action,
    ):
        """Run one joint student query while safely bypassing compiled blocks."""
        saved_blocks = None
        if model is self.student and getattr(self, "_student_blocks_compiled", False):
            saved_blocks = list(self.student.blocks)
            for block_index, block in enumerate(saved_blocks):
                if hasattr(block, "_orig_mod"):
                    self.student.blocks[block_index] = block._orig_mod
        try:
            return self._student_cfg_forward(
                model,
                joint_input,
                model_empty_emb,
                cfg_scale,
                batch_size,
                ref_shape,
                video_r_t,
                action_r_t,
                force_cfg=True,
                return_action=require_action,
            )
        finally:
            if saved_blocks is not None:
                for block_index, block in enumerate(saved_blocks):
                    self.student.blocks[block_index] = block

    def _joint_euler_update(
        self,
        video_x,
        action_x,
        video_velocity,
        action_velocity,
        video_t,
        video_r,
        action_t,
        action_r,
    ):
        video_sigma = self._timestep_to_sigma_5d(video_t)
        video_sigma_next = self._timestep_to_sigma_5d(video_r)
        action_sigma = (
            action_t[:, None, :, None, None] / self.config.num_train_timesteps
        )
        action_sigma_next = (
            action_r[:, None, :, None, None] / self.config.num_train_timesteps
        )
        return (
            video_x
            + video_velocity
            * (video_sigma_next.to(video_velocity) - video_sigma.to(video_velocity)),
            action_x
            + action_velocity
            * (
                action_sigma_next.to(action_velocity)
                - action_sigma.to(action_velocity)
            ),
        )

    def _prepare_cosmos_mechanism_context(self, batch):
        """Prepare immutable conditioning fields shared by every probe route."""
        if hasattr(self, "convert_input_format"):
            batch = self.convert_input_format(batch)
        batch_size = int(batch["latents"].shape[0])
        action_downsample = int(self.config.action_downsample_factor)
        action_clean = batch["actions"][:, :, ::action_downsample]
        if action_clean.shape[2] <= 0:
            raise ValueError("Cosmos mechanism probe requires action frames")
        input_dict = self._prepare_base_dict(batch)
        action_base = input_dict["action_dict"]
        action_mask = action_base.get("actions_mask")
        if action_mask is not None:
            action_mask = action_mask[:, :, ::action_downsample]
        action_grid = _downsample_action_grid_id(
            action_base.get("grid_id"),
            action_base["latent"],
            action_downsample,
        )
        student_model = getattr(self, "_student_nofsdp", None) or self.student
        if student_model is not self.student:
            if not getattr(self, "_nofsdp_synced", False):
                self._sync_student_nofsdp()
                self._nofsdp_synced = True
            video_base = {
                key: _to_regular_tensor(value)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in input_dict["latent_dict"].items()
            }
            action_latent = _to_regular_tensor(
                action_base["latent"][:, :, ::action_downsample]
            )
            action_cond_t = _to_regular_tensor(
                action_base["cond_timesteps"][:, ::action_downsample]
            )
            action_text = _to_regular_tensor(action_base["text_emb"])
            action_grid = _to_regular_tensor(action_grid)
            action_mask = _to_regular_tensor(action_mask)
            empty_emb = _to_regular_tensor(
                self.empty_emb.expand(batch_size, -1, -1)
            )
        else:
            video_base = input_dict["latent_dict"]
            action_latent = action_base["latent"][:, :, ::action_downsample]
            action_cond_t = action_base["cond_timesteps"][:, ::action_downsample]
            action_text = action_base["text_emb"]
            empty_emb = self.empty_emb.expand(batch_size, -1, -1)
        return {
            "batch": batch,
            "input_dict": input_dict,
            "batch_size": batch_size,
            "ref_shape": tuple(batch["latents"].shape),
            "action_clean": action_clean,
            "action_frames": int(action_clean.shape[2]),
            "action_mask": action_mask,
            "student_model": student_model,
            "video_base": video_base,
            "action_latent": action_latent,
            "action_cond_t": action_cond_t,
            "action_text": action_text,
            "action_grid": action_grid,
            "empty_emb": empty_emb,
            "chunk_size": input_dict["chunk_size"],
            "window_size": input_dict["window_size"],
            "cfg_scale": (
                float(getattr(self.config, "cfg_min", 1.0))
                + float(getattr(self.config, "cfg_max", 1.0))
            )
            / 2.0,
        }

    def _mechanism_joint_input(
        self, video_x, action_x, video_t, action_t, context
    ):
        return self._build_joint_input(
            video_x,
            video_t,
            action_x,
            action_t,
            video_base=context["video_base"],
            action_latent=context["action_latent"],
            action_cond_t=context["action_cond_t"],
            action_text=context["action_text"],
            action_grid=context["action_grid"],
            action_valid_mask=context["action_mask"],
            chunk_size=context["chunk_size"],
            window_size=context["window_size"],
        )

    def _cosmos_student_joint_map(
        self,
        video_x,
        action_x,
        video_t,
        action_t,
        video_r,
        action_r,
        *,
        context,
    ):
        joint_input = self._mechanism_joint_input(
            video_x, action_x, video_t, action_t, context
        )
        self._init_joint_mask(joint_input)
        video_velocity, action_velocity_seq = self._student_joint_forward(
            context["student_model"],
            joint_input,
            context["empty_emb"],
            video_r,
            action_r,
            cfg_scale=context["cfg_scale"],
            batch_size=context["batch_size"],
            ref_shape=context["ref_shape"],
            require_action=True,
        )
        if action_velocity_seq is None:
            raise RuntimeError("Cosmos mechanism probe requires an action head")
        action_velocity = self._extract_action_v(
            action_velocity_seq, context["action_frames"]
        )
        next_video, next_action = self._joint_euler_update(
            video_x,
            action_x,
            video_velocity,
            action_velocity,
            video_t,
            video_r,
            action_t,
            action_r,
        )
        return next_video, next_action, video_velocity, action_velocity

    def _cosmos_action_context_predictions(
        self,
        video_contexts,
        *,
        common_action,
        action_t,
        context,
        action_s=None,
    ):
        """Query the real action head with replaced video and one held action state."""
        predictions = {}
        video_t = torch.zeros(
            (context["batch_size"], context["ref_shape"][2]),
            device=common_action.device,
            dtype=action_t.dtype,
        )
        video_r = torch.zeros_like(video_t)
        action_r = torch.zeros_like(action_t)
        with torch.no_grad():
            for name, video_x in video_contexts.items():
                current_action = common_action
                action_path = (
                    (action_t, action_r)
                    if action_s is None
                    else (action_t, action_s, action_r)
                )
                for index in range(len(action_path) - 1):
                    current_t = action_path[index]
                    next_t = action_path[index + 1]
                    joint_input = self._mechanism_joint_input(
                        video_x, current_action, video_t, current_t, context
                    )
                    # Tiny unit harnesses intercept the forward before mask
                    # setup; production always owns a real student module.
                    if hasattr(self.student, "parameters"):
                        self._init_joint_mask(joint_input)
                    _, action_velocity_seq = self._student_joint_forward(
                        context["student_model"],
                        joint_input,
                        context["empty_emb"],
                        video_r,
                        next_t,
                        cfg_scale=context["cfg_scale"],
                        batch_size=context["batch_size"],
                        ref_shape=context["ref_shape"],
                        require_action=True,
                    )
                    action_velocity = self._extract_action_v(
                        action_velocity_seq, context["action_frames"]
                    )
                    current_sigma = (
                        current_t[:, None, :, None, None]
                        / self.config.num_train_timesteps
                    )
                    next_sigma = (
                        next_t[:, None, :, None, None]
                        / self.config.num_train_timesteps
                    )
                    current_action = current_action + action_velocity * (
                        next_sigma.to(action_velocity)
                        - current_sigma.to(action_velocity)
                    )
                predictions[name] = current_action.detach()
        return predictions

    def _validate_cosmos_mechanism_teacher_band(self, t_min, t_max):
        t_min = float(t_min)
        t_max = float(t_max)
        calibrated_min = 4.0 / 5.0
        calibrated_max = 80.0 / 81.0
        if not (
            calibrated_min <= t_min < t_max <= calibrated_max
        ):
            raise ValueError(
                "Cosmos mechanism teacher band must remain inside the calibrated "
                f"[{calibrated_min}, {calibrated_max}] interval"
            )
        return t_min, t_max

    @staticmethod
    def _cosmos_calibrated_noisy_state(*, teacher_x0, noise, normalized_t):
        """Construct a calibrated-band state from the official teacher endpoint."""
        normalized_t = torch.as_tensor(
            normalized_t, device=teacher_x0.device, dtype=teacher_x0.dtype
        )
        return (1.0 - normalized_t) * teacher_x0 + normalized_t * noise

    def _validate_cosmos_mechanism_preflight(
        self, batch, *, teacher_steps, student_steps
    ):
        """Validate deterministic local contracts before entering model forwards."""
        teacher_steps = int(teacher_steps)
        student_steps = int(student_steps)
        if teacher_steps != 8:
            raise ValueError(
                "Cosmos official mechanism teacher budget is fixed at 8 steps"
            )
        if student_steps not in (2, 4):
            raise ValueError("mechanism student_steps must be 2 or 4")
        if not isinstance(batch, dict):
            raise TypeError("Cosmos mechanism diagnostic batch must be a mapping")
        for key in ("latents", "actions"):
            value = batch.get(key)
            if not torch.is_tensor(value) or value.ndim < 3 or value.shape[0] <= 0:
                raise ValueError(
                    f"Cosmos mechanism diagnostic requires a non-empty {key} tensor"
                )
        teacher = getattr(self, "_action_teacher_model", None)
        if not (
            getattr(teacher, "raw_inference_enabled", False)
            and hasattr(teacher, "predict_raw_latent_target")
            and hasattr(teacher, "predict_raw_joint_latent_velocity")
            and hasattr(teacher, "predict_raw_same_prior_endpoint")
            and hasattr(teacher, "predict_raw_joint_continuation_endpoint")
        ):
            raise RuntimeError(
                "Cosmos mechanism diagnostics require the official raw teacher "
                "with same-prior and joint-continuation endpoints"
            )
        self._validate_cosmos_mechanism_teacher_band(
            getattr(self.config, "mechanism_cosmos_t_min", 4.0 / 5.0),
            getattr(self.config, "mechanism_cosmos_t_max", 80.0 / 81.0),
        )

    def _cosmos_teacher_field_probe(
        self,
        *,
        teacher,
        batch,
        query_latent,
        query_action,
        t_norm,
        raw_video_t,
        raw_video_r,
        raw_action_t,
        raw_action_r,
        context,
    ):
        """Query the raw worker, then route its post-injection state to student."""
        calibrated_min, calibrated_max = 4.0 / 5.0, 80.0 / 81.0
        if not bool(
            ((t_norm >= calibrated_min) & (t_norm <= calibrated_max)).all()
        ):
            raise ValueError(
                "Cosmos mechanism field query must stay in the calibrated band"
            )
        teacher_result = teacher.predict_raw_joint_latent_velocity(
            batch,
            query_latent=query_latent.float(),
            query_action=query_action,
            t=t_norm.detach().float(),
        )
        student_query_video = teacher_result["cosmos_joint_query"].to(
            device=query_latent.device, dtype=query_latent.dtype
        )
        teacher_field_video = teacher_result["cosmos_latent_velocity"].to(
            device=query_latent.device, dtype=query_latent.dtype
        )
        video_frame_mask = teacher_result["cosmos_video_frame_mask"].to(
            device=query_latent.device, dtype=torch.bool
        )
        (
            student_direct_video,
            student_direct_action,
            student_field_video,
            student_action_field,
        ) = self._cosmos_student_joint_map(
            student_query_video,
            query_action,
            raw_video_t,
            raw_action_t,
            raw_video_r,
            raw_action_r,
            context=context,
        )
        return {
            "student_query_video": student_query_video,
            "teacher_field_video": teacher_field_video,
            "video_frame_mask": video_frame_mask,
            "student_direct_video": student_direct_video,
            "student_direct_action": student_direct_action,
            "student_field_video": student_field_video,
            "student_action_field": student_action_field,
        }

    def _cosmos_teacher_video_continuation(
        self,
        *,
        teacher,
        batch,
        start_video,
        held_action,
        t_min,
        t_max,
        teacher_steps,
    ):
        """Integrate only the supported teacher video field with held action."""
        t_min, t_max = self._validate_cosmos_mechanism_teacher_band(
            t_min, t_max
        )
        teacher_steps = int(teacher_steps)
        if teacher_steps <= 0:
            raise ValueError("Cosmos mechanism teacher_steps must be positive")
        path = torch.linspace(
            t_max,
            t_min,
            teacher_steps + 1,
            device=start_video.device,
            dtype=torch.float32,
        )
        current = start_video
        for index in range(teacher_steps):
            current_t = torch.full(
                (start_video.shape[0], start_video.shape[2]),
                float(path[index].item()),
                device=start_video.device,
            )
            result = teacher.predict_raw_joint_latent_velocity(
                batch,
                query_latent=current.float(),
                query_action=held_action,
                t=current_t,
            )
            joint_query = result["cosmos_joint_query"].to(
                device=start_video.device, dtype=start_video.dtype
            )
            velocity = result["cosmos_latent_velocity"].to(
                device=start_video.device, dtype=start_video.dtype
            )
            current = joint_query + velocity * (
                path[index + 1].to(velocity) - path[index].to(velocity)
            )
        return current

    def _cosmos_teacher_direct_endpoint(
        self,
        *,
        start_video,
        teacher_field_video,
        t_min,
        t_max,
    ):
        """Take the direct teacher Euler route over the same calibrated interval."""
        t_min, t_max = self._validate_cosmos_mechanism_teacher_band(
            t_min, t_max
        )
        return start_video + teacher_field_video * (t_min - t_max)

    @torch.no_grad()
    def _run_cosmos_mechanism_probe_legacy(
        self, batch, *, seed, teacher_steps, student_steps
    ):
        """Return additive Cosmos-native mechanism statistics without training."""
        teacher_steps = int(teacher_steps)
        student_steps = int(student_steps)
        self._validate_cosmos_mechanism_preflight(
            batch, teacher_steps=teacher_steps, student_steps=student_steps
        )
        dataset_gt_video = batch["latents"].detach().clone()
        context = self._prepare_cosmos_mechanism_context(batch)
        batch = context["batch"]
        teacher = self._action_teacher_model
        if not (
            getattr(teacher, "raw_inference_enabled", False)
            and hasattr(teacher, "predict_raw_latent_target")
            and hasattr(teacher, "predict_raw_joint_latent_velocity")
        ):
            raise RuntimeError(
                "Cosmos mechanism diagnostics require the official raw teacher"
            )
        t_min, t_max = self._validate_cosmos_mechanism_teacher_band(
            getattr(self.config, "mechanism_cosmos_t_min", 4.0 / 5.0),
            getattr(self.config, "mechanism_cosmos_t_max", 80.0 / 81.0),
        )
        B = context["batch_size"]
        video_frames = context["ref_shape"][2]
        action_frames = context["action_frames"]
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        video_noise = torch.randn(
            context["ref_shape"],
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )
        native_action_prior = torch.randn(
            B,
            16,
            7,
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )
        action_channels = int(batch["actions"].shape[1])
        used_action_channel_ids = torch.as_tensor(
            self.config.used_action_channel_ids,
            device=self.device,
            dtype=torch.long,
        )
        if (
            used_action_channel_ids.numel() != 7
            or int(used_action_channel_ids.min().item()) < 0
            or int(used_action_channel_ids.max().item()) >= action_channels
        ):
            raise ValueError(
                "Cosmos mechanism diagnostics require seven valid action "
                "channel IDs"
            )
        aligned_action_prior = torch.zeros(
            B,
            16,
            action_channels,
            device=self.device,
            dtype=torch.float32,
        )
        aligned_action_prior[..., used_action_channel_ids] = native_action_prior
        action_downsample = int(self.config.action_downsample_factor)
        full_action_prior = pack_actions_for_downsample(
            aligned_action_prior,
            tuple(batch["actions"].shape),
            downsample_factor=action_downsample,
            schema=self.config.action_packing_schema,
        )
        action_noise = full_action_prior[:, :, ::action_downsample]
        t_max_norm = torch.full(
            (B, video_frames), t_max, device=self.device, dtype=torch.float32
        )
        t_min_norm = torch.full_like(t_max_norm, t_min)
        endpoint_result = teacher.predict_raw_latent_target(
            batch,
            noise=video_noise,
            t=t_max_norm,
            r=t_min_norm,
            epsilon=float(self.config.cosmos_latent_epsilon),
            include_cdiff=False,
        )
        teacher_clean_video = endpoint_result["cosmos_latent_x0"].to(
            device=self.device, dtype=batch["latents"].dtype
        )
        teacher_endpoint_action_full = cosmos_actions_to_flowmap_x0(
            endpoint_result["actions"],
            target_shape=tuple(batch["actions"].shape),
            q01=self.config.norm_stat["q01"],
            q99=self.config.norm_stat["q99"],
            inverse_used_action_channel_ids=self.config.inverse_used_action_channel_ids,
            device=self.device,
            dtype=batch["actions"].dtype,
            packing_schema=self.config.action_packing_schema,
            downsample_factor=self.config.action_downsample_factor,
        )
        teacher_endpoint_action = teacher_endpoint_action_full[
            :, :, ::action_downsample
        ]

        # The mechanism routes depart from one Student-reached z_r generated
        # from the exact terminal priors later supplied to the same-prior
        # Teacher endpoint.
        terminal_video_t = torch.full(
            (B, video_frames),
            float(self.config.num_train_timesteps),
            device=self.device,
        )
        raw_t_max = t_max_norm * self.config.num_train_timesteps
        raw_t_min = t_min_norm * self.config.num_train_timesteps
        terminal_action_t = torch.full(
            (B, batch["actions"].shape[2]),
            float(self.config.num_train_timesteps),
            device=self.device,
        )
        full_action_t_max = torch.full_like(
            terminal_action_t, t_max * self.config.num_train_timesteps
        )
        raw_action_t_max = torch.full(
            (B, action_frames),
            t_max * self.config.num_train_timesteps,
            device=self.device,
        )
        raw_action_t_min = torch.full_like(
            raw_action_t_max, t_min * self.config.num_train_timesteps
        )
        deployment_input = {
            "latent_dict": {
                **context["input_dict"]["latent_dict"],
                "latent": video_noise,
                "noisy_latents": video_noise,
                "timesteps": terminal_video_t,
            },
            "action_dict": {
                **context["input_dict"]["action_dict"],
                "noisy_latents": action_noise,
                "latent": action_noise,
                "timesteps": terminal_action_t[:, ::action_downsample],
                "cond_timesteps": context["action_cond_t"],
                "grid_id": context["action_grid"],
                "actions_mask": context["action_mask"],
            },
            "chunk_size": context["chunk_size"],
            "window_size": context["window_size"],
        }
        deployment_result = self._student_euler_integrate(
            noisy_latents=video_noise,
            timesteps=terminal_video_t,
            target_r=raw_t_max,
            base_input_dict=deployment_input,
            empty_emb=context["empty_emb"],
            cfg_scale=context["cfg_scale"],
            ref_shape=context["ref_shape"],
            B=B,
            num_frames=video_frames,
            K_steps=student_steps,
            action_target_r=full_action_t_max,
            return_final_action=True,
            return_final_action_state=True,
            force_no_grad=True,
        )
        shared_student_video = deployment_result[0].detach()
        shared_student_action = deployment_result[3].detach()
        field_result = self._cosmos_teacher_field_probe(
            teacher=teacher,
            batch=batch,
            query_latent=shared_student_video,
            query_action=shared_student_action,
            t_norm=t_max_norm,
            raw_video_t=raw_t_max,
            raw_video_r=raw_t_min,
            raw_action_t=raw_action_t_max,
            raw_action_r=raw_action_t_min,
            context=context,
        )
        joint_video = field_result["student_query_video"]
        teacher_field_video = field_result["teacher_field_video"]
        video_frame_mask = field_result["video_frame_mask"]
        student_direct_video = field_result["student_direct_video"]
        student_field_video = field_result["student_field_video"]
        expanded_mask = video_frame_mask[:, None, :, None, None].expand_as(
            joint_video
        )
        shared_state_verified = torch.equal(
            joint_video.masked_select(expanded_mask),
            shared_student_video.to(joint_video).masked_select(expanded_mask),
        )
        if not shared_state_verified:
            raise RuntimeError(
                "Cosmos mechanism Teacher canonicalization changed valid "
                "shared Student-state frames"
            )
        same_prior_result = teacher.predict_raw_same_prior_endpoint(
            batch,
            video_prior=video_noise,
            action_prior=native_action_prior,
            teacher_steps=teacher_steps,
        )
        required_same_prior = (
            "endpoint_video",
            "video_frame_mask",
            "effective_teacher_steps",
            "video_prior_sha256",
            "action_prior_sha256",
        )
        if not all(key in same_prior_result for key in required_same_prior):
            raise RuntimeError(
                "Cosmos mechanism same-prior Teacher response is incomplete"
            )
        same_prior_teacher_endpoint_video = same_prior_result[
            "endpoint_video"
        ].to(device=self.device, dtype=joint_video.dtype)
        same_prior_mask = same_prior_result["video_frame_mask"].to(
            device=self.device, dtype=torch.bool
        )
        effective_teacher_steps = same_prior_result["effective_teacher_steps"]
        same_prior_verified = (
            same_prior_teacher_endpoint_video.shape == joint_video.shape
            and same_prior_mask.shape == video_frame_mask.shape
            and torch.equal(same_prior_mask, video_frame_mask)
            and type(effective_teacher_steps) is int
            and effective_teacher_steps == 8
        )
        if not same_prior_verified:
            raise RuntimeError(
                "Cosmos mechanism same-prior endpoint shape, mask, or "
                "effective eight-step contract mismatch"
            )
        midpoint = (t_min + t_max) / 2.0
        raw_mid_video = torch.full_like(
            raw_t_max, midpoint * self.config.num_train_timesteps
        )
        raw_mid_action = torch.full_like(
            raw_action_t_max, midpoint * self.config.num_train_timesteps
        )
        mid_video, mid_action, _, _ = self._cosmos_student_joint_map(
            joint_video,
            shared_student_action,
            raw_t_max,
            raw_action_t_max,
            raw_mid_video,
            raw_mid_action,
            context=context,
        )
        student_composed_video, _, _, _ = self._cosmos_student_joint_map(
            mid_video,
            mid_action,
            raw_mid_video,
            raw_mid_action,
            raw_t_min,
            raw_action_t_min,
            context=context,
        )
        teacher_cont_video = self._cosmos_teacher_video_continuation(
            teacher=teacher,
            batch=batch,
            start_video=joint_video,
            held_action=shared_student_action,
            t_min=t_min,
            t_max=t_max,
            teacher_steps=teacher_steps,
        )

        diagnostic_r = float(getattr(self.config, "mechanism_diagnostic_r", 500))
        common_action = (
            (1.0 - diagnostic_r / self.config.num_train_timesteps)
            * teacher_endpoint_action
            + (diagnostic_r / self.config.num_train_timesteps) * action_noise
        )
        action_t = torch.full(
            (B, action_frames), diagnostic_r, device=self.device
        )
        diagnostic_s = float(getattr(self.config, "mechanism_diagnostic_s", 250))
        action_s = torch.full(
            (B, action_frames), diagnostic_s, device=self.device
        )
        action_predictions = self._cosmos_action_context_predictions(
            {
                "gt": dataset_gt_video,
                "student": student_direct_video,
                "teacher_video": teacher_clean_video,
            },
            common_action=common_action,
            action_t=action_t,
            action_s=action_s,
            context=context,
        )
        samples = compute_mechanism_metric_samples(
            teacher_continuation_video=teacher_cont_video,
            same_prior_teacher_endpoint_video=(
                same_prior_teacher_endpoint_video
            ),
            direct_route_video=student_direct_video,
            composed_route_video=student_composed_video,
            student_field_video=student_field_video,
            teacher_field_video=teacher_field_video,
            video_frame_mask=video_frame_mask,
            teacher_endpoint_action=teacher_endpoint_action,
            action_gt_context=action_predictions["gt"],
            action_student_context=action_predictions["student"],
            action_teacher_video_context=action_predictions["teacher_video"],
            action_teacher_joint_context=None,
            action_mask=context["action_mask"],
            teacher_joint_available=False,
            shared_state_verified=shared_state_verified,
            same_prior_verified=same_prior_verified,
            effective_teacher_steps=effective_teacher_steps,
        )
        stats = pack_finite_metric_stats(samples)
        stats["mechanism/teacher_steps_sum"] = torch.as_tensor(
            float(teacher_steps * B), device=self.device
        )
        stats["mechanism/teacher_steps_count"] = torch.as_tensor(
            float(B), device=self.device
        )
        stats["mechanism/student_steps_sum"] = torch.as_tensor(
            float(student_steps * B), device=self.device
        )
        stats["mechanism/student_steps_count"] = torch.as_tensor(
            float(B), device=self.device
        )
        return {name: value.detach() for name, value in stats.items()}

    @torch.no_grad()
    def _run_cosmos_mechanism_probe(
        self, batch, *, seed, teacher_steps, student_steps
    ):
        """Run aligned observation-only G metrics from one shifted Student z_r."""
        teacher_steps = int(teacher_steps)
        student_steps = int(student_steps)
        self._validate_cosmos_mechanism_preflight(
            batch, teacher_steps=teacher_steps, student_steps=student_steps
        )
        dataset_gt_video = batch["latents"].detach().clone()
        context = self._prepare_cosmos_mechanism_context(batch)
        batch = context["batch"]
        teacher = self._action_teacher_model
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        shared = self._build_cosmos_shifted_shared_query(
            batch,
            context["input_dict"],
            student_steps=student_steps,
            generator=generator,
            student_model=context["student_model"],
            cfg_scale=context["cfg_scale"],
        )
        query_video = shared["query_video"]
        query_action = shared["query_action"]
        query_video_t = shared["query_video_t"]
        query_action_t = shared["query_action_t"]
        query_sigma_frames = shared["query_sigma_frames"]

        field_result = teacher.predict_raw_joint_latent_velocity(
            batch,
            query_latent=query_video.float(),
            query_action=query_action,
            t=query_sigma_frames.float(),
        )
        required_field = (
            "cosmos_joint_query",
            "cosmos_latent_velocity",
            "cosmos_video_frame_mask",
        )
        if not all(key in field_result for key in required_field):
            raise RuntimeError(
                "Cosmos mechanism same-state Teacher response is incomplete"
            )
        canonical_video = field_result["cosmos_joint_query"].to(
            device=self.device, dtype=query_video.dtype
        ).detach()
        teacher_field_video = field_result["cosmos_latent_velocity"].to(
            device=self.device, dtype=canonical_video.dtype
        ).detach()
        video_frame_mask = field_result["cosmos_video_frame_mask"].to(
            device=self.device, dtype=torch.bool
        )
        shape_ok = (
            canonical_video.shape == query_video.shape
            and teacher_field_video.shape == canonical_video.shape
            and video_frame_mask.shape
            == (shared["batch_size"], canonical_video.shape[2])
            and bool(video_frame_mask.any(dim=1).all().item())
        )
        if not shape_ok:
            raise RuntimeError(
                "Cosmos mechanism canonical state, field, or nonempty mask "
                "contract mismatch"
            )
        expanded_mask = video_frame_mask[:, None, :, None, None].expand_as(
            canonical_video
        )
        shared_state_verified = torch.equal(
            canonical_video.masked_select(expanded_mask),
            query_video.to(canonical_video).masked_select(expanded_mask),
        )
        if not shared_state_verified:
            raise RuntimeError(
                "Cosmos mechanism canonical valid frames differ from shared z_r"
            )

        same_prior_result = teacher.predict_raw_same_prior_endpoint(
            batch,
            video_prior=shared["video_prior"],
            action_prior=shared["native_action_prior"],
            teacher_steps=teacher_steps,
        )
        required_same_prior = (
            "endpoint_video",
            "video_frame_mask",
            "effective_teacher_steps",
            "video_prior_sha256",
            "action_prior_sha256",
        )
        if not all(key in same_prior_result for key in required_same_prior):
            raise RuntimeError(
                "Cosmos mechanism same-prior Teacher response is incomplete"
            )
        same_prior_endpoint = same_prior_result["endpoint_video"].to(
            device=self.device, dtype=canonical_video.dtype
        ).detach()
        same_prior_mask = same_prior_result["video_frame_mask"].to(
            device=self.device, dtype=torch.bool
        )
        same_prior_steps = same_prior_result["effective_teacher_steps"]
        same_prior_verified = (
            same_prior_endpoint.shape == canonical_video.shape
            and same_prior_mask.shape == video_frame_mask.shape
            and bool(same_prior_mask.any(dim=1).all().item())
            and torch.equal(same_prior_mask, video_frame_mask)
            and type(same_prior_steps) is int
            and same_prior_steps == 8
        )
        if not same_prior_verified:
            raise RuntimeError(
                "Cosmos mechanism same-prior mask or observed eight-step "
                "provenance mismatch"
            )

        normalized_action_clock = query_action_t / float(
            self.config.num_train_timesteps
        )
        video_clock_per_sample = query_sigma_frames[:, :1]
        action_clock_per_sample = normalized_action_clock[:, :1]
        if not torch.equal(
            query_sigma_frames,
            video_clock_per_sample.expand_as(query_sigma_frames),
        ):
            raise RuntimeError(
                "Cosmos mechanism video clock differs across frames"
            )
        if not torch.equal(
            normalized_action_clock,
            action_clock_per_sample.expand_as(normalized_action_clock),
        ):
            raise RuntimeError(
                "Cosmos mechanism action clock differs across frames"
            )
        anchor_available = torch.equal(
            video_clock_per_sample, action_clock_per_sample
        )
        mixed_clock = not anchor_available
        teacher_continuation = None
        continuation_steps = None
        continuation_verified = False
        if anchor_available:
            continuation_result = (
                teacher.predict_raw_joint_continuation_endpoint(
                    batch,
                    canonical_joint_state=canonical_video.float(),
                    normalized_t=query_sigma_frames.float(),
                    teacher_steps=teacher_steps,
                )
            )
            required_continuation = (
                "endpoint_video",
                "video_frame_mask",
                "effective_teacher_steps",
                "joint_state_sha256",
                "normalized_t_sha256",
                "normalized_t",
                "edm_sigma",
            )
            if not all(
                key in continuation_result for key in required_continuation
            ):
                raise RuntimeError(
                    "Cosmos mechanism continuation Teacher response is "
                    "incomplete"
                )
            teacher_continuation = continuation_result["endpoint_video"].to(
                device=self.device, dtype=canonical_video.dtype
            ).detach()
            continuation_mask = continuation_result["video_frame_mask"].to(
                device=self.device, dtype=torch.bool
            )
            continuation_steps = continuation_result[
                "effective_teacher_steps"
            ]
            continuation_verified = (
                teacher_continuation.shape == canonical_video.shape
                and continuation_mask.shape == video_frame_mask.shape
                and bool(continuation_mask.any(dim=1).all().item())
                and torch.equal(continuation_mask, video_frame_mask)
                and type(continuation_steps) is int
                and continuation_steps == 8
            )
            if not continuation_verified:
                raise RuntimeError(
                    "Cosmos mechanism continuation mask or observed "
                    "eight-step provenance mismatch"
                )

        route_context = shared["joint_context"](canonical_video, query_action)
        field_input = self._mechanism_joint_input(
            canonical_video,
            query_action,
            query_video_t,
            query_action_t,
            route_context,
        )
        self._init_joint_mask(field_input)
        student_field_video = self._student_joint_forward(
            route_context["student_model"],
            field_input,
            route_context["empty_emb"],
            query_video_t,
            query_action_t,
            cfg_scale=route_context["cfg_scale"],
            batch_size=route_context["batch_size"],
            ref_shape=route_context["ref_shape"],
            require_action=False,
        )

        zero_video_t = torch.zeros_like(query_video_t)
        zero_action_t = torch.zeros_like(query_action_t)
        direct_video, _, _, _ = self._cosmos_student_joint_map(
            canonical_video,
            query_action,
            query_video_t,
            query_action_t,
            zero_video_t,
            zero_action_t,
            context=route_context,
        )
        diagnostic_s = float(
            getattr(self.config, "mechanism_diagnostic_s", 250.0)
        )
        composed_video_s = torch.full_like(query_video_t, diagnostic_s)
        if not bool(
            ((composed_video_s >= 0) & (composed_video_s < query_video_t)).all()
        ):
            raise ValueError(
                "mechanism diagnostic s must satisfy 0 <= s < every shared r"
            )
        action_fraction = (
            composed_video_s[:, :1] / query_video_t[:, :1]
        )
        composed_action_s = query_action_t * action_fraction
        mid_video, mid_action, _, _ = self._cosmos_student_joint_map(
            canonical_video,
            query_action,
            query_video_t,
            query_action_t,
            composed_video_s,
            composed_action_s,
            context=route_context,
        )
        mid_context = shared["joint_context"](mid_video, mid_action)
        composed_video, _, _, _ = self._cosmos_student_joint_map(
            mid_video,
            mid_action,
            composed_video_s,
            composed_action_s,
            zero_video_t,
            zero_action_t,
            context=mid_context,
        )

        # Legacy target inference remains isolated to action-context sanity
        # metrics and cannot affect either G input.
        legacy_t = torch.full_like(query_sigma_frames, 80.0 / 81.0)
        legacy_r = torch.full_like(query_sigma_frames, 4.0 / 5.0)
        legacy_target = teacher.predict_raw_latent_target(
            batch,
            noise=shared["video_prior"],
            t=legacy_t,
            r=legacy_r,
            epsilon=float(self.config.cosmos_latent_epsilon),
            include_cdiff=False,
        )
        teacher_clean_video = legacy_target["cosmos_latent_x0"].to(
            device=self.device, dtype=batch["latents"].dtype
        )
        teacher_endpoint_action_full = cosmos_actions_to_flowmap_x0(
            legacy_target["actions"],
            target_shape=tuple(batch["actions"].shape),
            q01=self.config.norm_stat["q01"],
            q99=self.config.norm_stat["q99"],
            inverse_used_action_channel_ids=(
                self.config.inverse_used_action_channel_ids
            ),
            device=self.device,
            dtype=batch["actions"].dtype,
            packing_schema=self.config.action_packing_schema,
            downsample_factor=self.config.action_downsample_factor,
        )
        action_downsample = int(self.config.action_downsample_factor)
        teacher_endpoint_action = teacher_endpoint_action_full[
            :, :, ::action_downsample
        ]
        diagnostic_r = float(
            getattr(self.config, "mechanism_diagnostic_r", 500.0)
        )
        action_t = torch.full(
            (shared["batch_size"], shared["action_frame_count"]),
            diagnostic_r,
            device=self.device,
        )
        action_s = torch.full_like(action_t, diagnostic_s)
        action_predictions = self._cosmos_action_context_predictions(
            {
                "gt": dataset_gt_video,
                "student": direct_video,
                "teacher_video": teacher_clean_video,
            },
            common_action=query_action,
            action_t=action_t,
            action_s=action_s,
            context=context,
        )
        samples = compute_mechanism_metric_samples(
            teacher_continuation_video=teacher_continuation,
            same_prior_teacher_endpoint_video=same_prior_endpoint,
            direct_route_video=direct_video,
            composed_route_video=composed_video,
            student_field_video=student_field_video,
            teacher_field_video=teacher_field_video,
            video_frame_mask=video_frame_mask,
            teacher_endpoint_action=teacher_endpoint_action,
            action_gt_context=action_predictions["gt"],
            action_student_context=action_predictions["student"],
            action_teacher_video_context=action_predictions["teacher_video"],
            action_teacher_joint_context=None,
            action_mask=context["action_mask"],
            teacher_joint_available=False,
            shared_state_verified=shared_state_verified,
            same_prior_verified=same_prior_verified,
            continuation_verified=continuation_verified,
            effective_teacher_steps=continuation_steps,
            anchor_available=anchor_available,
            anchor_unavailable_mixed_clock=mixed_clock,
        )
        stats = pack_finite_metric_stats(samples)
        batch_size = shared["batch_size"]
        for name, value in (
            ("teacher_steps", same_prior_steps),
            ("student_steps", shared["student_steps"]),
        ):
            stats[f"mechanism/{name}_sum"] = torch.as_tensor(
                float(value * batch_size), device=self.device
            )
            stats[f"mechanism/{name}_count"] = torch.as_tensor(
                float(batch_size), device=self.device
            )
        return {name: value.detach() for name, value in stats.items()}

    def _cosmos_danceopd_velocity_loss(
        self, batch, teacher, input_dict, *, cfg_scale, teacher_crop_context=None
    ):
        """Return DanceOPD's local video-field loss for a raw Cosmos teacher.

        The configured trajectory jointly evolves the student video and action
        states, but the official Cosmos API only exposes a video latent field.
        Thus the local loss supervises the video field at a state conditioned
        on the on-policy action state instead of fabricating an action field.
        """
        B = batch['latents'].shape[0]
        ref_shape = batch['latents'].shape
        video_frames = ref_shape[2]
        action_downsample = int(getattr(self.config, 'action_downsample_factor', 4))
        action_clean = batch['actions'][:, :, ::action_downsample]
        action_frames = action_clean.shape[2]
        if action_frames <= 0:
            raise ValueError("Cosmos DanceOPD requires action frames after downsampling")

        rollout_step_choices = tuple(
            int(value)
            for value in getattr(
                self.config,
                'opd_danceopd_rollout_step_choices',
                (getattr(self.config, 'opd_danceopd_rollout_steps', 4),),
            )
        )
        if not rollout_step_choices or any(value <= 0 for value in rollout_step_choices):
            raise ValueError('opd_danceopd_rollout_step_choices must be positive')
        if len(rollout_step_choices) == 1:
            rollout_steps = rollout_step_choices[0]
        else:
            choice_index = torch.randint(
                len(rollout_step_choices), (1,), device=self.device
            )
            if dist.is_initialized():
                dist.broadcast(choice_index, src=0)
            rollout_steps = rollout_step_choices[choice_index.item()]
        query_alpha = float(getattr(self.config, 'opd_danceopd_query_alpha', 5.0))
        query_beta = float(getattr(self.config, 'opd_danceopd_query_beta', 2.0))
        cosmos_t_min = float(getattr(self.config, 'cosmos_latent_t_min', 4.0 / 5.0))
        cosmos_t_max = float(getattr(self.config, 'cosmos_latent_t_max', 80.0 / 81.0))
        video_path = build_cosmos_teacher_window_path(
            batch_size=B,
            num_frames=video_frames,
            num_steps=rollout_steps,
            t_min=cosmos_t_min,
            t_max=cosmos_t_max,
            num_train_timesteps=self.config.num_train_timesteps,
            device=self.device,
            dtype=torch.float32,
        )
        action_path = build_cosmos_teacher_window_path(
            batch_size=B,
            num_frames=action_frames,
            num_steps=rollout_steps,
            t_min=cosmos_t_min,
            t_max=cosmos_t_max,
            num_train_timesteps=self.config.num_train_timesteps,
            device=self.device,
            dtype=torch.float32,
        )
        full_dance_noise = None
        if teacher_crop_context is None:
            video_noise = torch.randn_like(batch['latents'])
        else:
            full_dance_noise = torch.randn_like(
                teacher_crop_context['full_anchor']
            )
            video_noise = full_dance_noise[
                ..., teacher_crop_context['h_slice'], teacher_crop_context['w_slice']
            ].to(batch['latents'])
        action_noise = torch.randn_like(action_clean)
        video_start_sigma = self._timestep_to_sigma_5d(video_path[0]).to(batch['latents'])
        action_start_sigma = self._timestep_to_sigma_5d(action_path[0]).to(action_clean)
        # Raw Cosmos velocity uses x_t=(1-t)x0+t*noise.  Keep the entire
        # student rollout in the same supported raw-teacher coordinates.
        current_video = (
            (1.0 - video_start_sigma) * batch['latents']
            + video_start_sigma * video_noise
        )
        current_action = (
            (1.0 - action_start_sigma) * action_clean
            + action_start_sigma * action_noise
        )
        expected_video_start = (
            (1.0 - video_start_sigma) * batch['latents']
            + video_start_sigma * video_noise
        )
        expected_action_start = (
            (1.0 - action_start_sigma) * action_clean
            + action_start_sigma * action_noise
        )
        (
            terminal_prior_validation,
            terminal_prior_diagnostics,
        ) = _validate_terminal_prior_pair(
            config=self.config,
            video_actual=current_video,
            video_expected=expected_video_start,
            video_source_dtype=batch["latents"].dtype,
            action_actual=current_action,
            action_expected=expected_action_start,
            action_source_dtype=action_clean.dtype,
            reference=current_video,
            error_context=(
                "Cosmos DanceOPD window start state is inconsistent with "
                "the raw-teacher interpolation"
            ),
        )
        action_base = input_dict['action_dict']
        action_grid_id = _downsample_action_grid_id(
            action_base.get('grid_id'), action_base['latent'], action_downsample
        )
        action_mask = action_base.get('actions_mask')
        if action_mask is not None:
            action_mask = action_mask[:, :, ::action_downsample]

        joint_context = {
            "video_base": input_dict["latent_dict"],
            "action_latent": action_base["latent"][:, :, ::action_downsample],
            "action_cond_t": action_base["cond_timesteps"][:, ::action_downsample],
            "action_text": action_base["text_emb"],
            "action_grid": action_grid_id,
            "action_mask": action_mask,
            "chunk_size": input_dict["chunk_size"],
            "window_size": input_dict["window_size"],
            "student_model": self.student,
            "empty_emb": self.empty_emb.expand(B, -1, -1),
            "cfg_scale": cfg_scale,
            "batch_size": B,
            "ref_shape": ref_shape,
        }

        video_states = []
        action_states = []
        video_timesteps = []
        action_timesteps = []
        with torch.no_grad():
            for step_index in range(rollout_steps):
                video_t = video_path[step_index]
                video_r = video_path[step_index + 1]
                action_t = action_path[step_index]
                action_r = action_path[step_index + 1]
                video_states.append(current_video.detach().clone())
                action_states.append(current_action.detach().clone())
                video_timesteps.append(video_t.detach().clone())
                action_timesteps.append(action_t.detach().clone())

                joint_input = self._mechanism_joint_input(
                    current_video, current_action, video_t, action_t, joint_context
                )
                self._init_joint_mask(joint_input)
                video_velocity, action_velocity_seq = self._student_joint_forward(
                    self.student,
                    joint_input,
                    joint_context["empty_emb"],
                    video_r,
                    action_r,
                    cfg_scale=cfg_scale,
                    batch_size=B,
                    ref_shape=ref_shape,
                    require_action=True,
                )
                if action_velocity_seq is None:
                    raise RuntimeError(
                        "Cosmos DanceOPD joint rollout requires a student action output"
                    )
                action_velocity = self._extract_action_v(
                    action_velocity_seq, action_frames
                )
                video_sigma = self._timestep_to_sigma_5d(video_t)
                video_sigma_next = self._timestep_to_sigma_5d(video_r)
                action_sigma = action_t[:, None, :, None, None] / self.config.num_train_timesteps
                action_sigma_next = action_r[:, None, :, None, None] / self.config.num_train_timesteps
                current_video = current_video + video_velocity * (
                    video_sigma_next.to(video_velocity) - video_sigma.to(video_velocity)
                )
                current_action = current_action + action_velocity * (
                    action_sigma_next.to(action_velocity) - action_sigma.to(action_velocity)
                )

        query_indices = sample_low_noise_query_indices(
            n_states=rollout_steps,
            batch_size=B,
            alpha=query_alpha,
            beta=query_beta,
            device=current_video.device,
        )
        query_video = select_per_sample_trajectory_state(
            torch.stack(video_states, dim=0), query_indices
        ).detach()
        query_action = select_per_sample_trajectory_state(
            torch.stack(action_states, dim=0), query_indices
        ).detach()
        query_video_t = select_per_sample_trajectory_state(
            torch.stack(video_timesteps, dim=0), query_indices
        ).detach()
        query_action_t = select_per_sample_trajectory_state(
            torch.stack(action_timesteps, dim=0), query_indices
        ).detach()
        # Only the selected query state is needed by the trainable forwards
        # below. Release the no-grad rollout trajectory before building those
        # graphs, which keeps the Cosmos joint rollout's peak memory bounded.
        del video_states, action_states, video_timesteps, action_timesteps
        del current_video, current_action, joint_input
        del video_velocity, action_velocity_seq, action_velocity
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with torch.no_grad():
            teacher_query = query_video.detach().float()
            if teacher_crop_context is not None:
                query_sigma = (
                    query_video_t / self.config.num_train_timesteps
                )[:, None, :, None, None].to(
                    teacher_crop_context['full_anchor']
                )
                teacher_query = (
                    (1.0 - query_sigma) * teacher_crop_context['full_anchor']
                    + query_sigma * full_dance_noise
                )
                teacher_query = teacher_query.clone()
                teacher_query[
                    ..., teacher_crop_context['h_slice'], teacher_crop_context['w_slice']
                ] = query_video.detach().float().to(teacher_query)
            teacher_result = teacher.predict_raw_joint_latent_velocity(
                batch,
                query_latent=teacher_query,
                query_action=query_action,
                t=(query_video_t / self.config.num_train_timesteps).detach().float(),
            )
            teacher_velocity = teacher_result['cosmos_latent_velocity'].to(
                device=query_video.device, dtype=query_video.dtype
            )
            student_query_video = teacher_result['cosmos_joint_query'].to(
                device=query_video.device, dtype=query_video.dtype
            )
            video_frame_mask = teacher_result['cosmos_video_frame_mask'].to(
                device=query_video.device, dtype=torch.bool
            )
            if teacher_crop_context is not None:
                teacher_velocity = teacher_velocity[
                    ..., teacher_crop_context['h_slice'], teacher_crop_context['w_slice']
                ]
                student_query_video = student_query_video[
                    ..., teacher_crop_context['h_slice'], teacher_crop_context['w_slice']
                ]
        query_input = self._mechanism_joint_input(
            student_query_video,
            query_action,
            query_video_t,
            query_action_t,
            joint_context,
        )
        self._init_joint_mask(query_input)
        student_velocity = self._student_joint_forward(
            self.student,
            query_input,
            joint_context["empty_emb"],
            query_video_t,
            query_action_t,
            cfg_scale=cfg_scale,
            batch_size=B,
            ref_shape=ref_shape,
            require_action=False,
        )
        velocity_loss = masked_video_velocity_mse(
            student_velocity, teacher_velocity, video_frame_mask
        )
        diagnostics = {
            'query_index_mean': query_indices.float().mean().detach(),
            'query_sigma_mean': (
                query_video_t.float() / self.config.num_train_timesteps
            ).mean().detach(),
            'terminal_prior_max_error': terminal_prior_diagnostics['max_error'],
            'terminal_prior_reference_scale': terminal_prior_diagnostics['reference_scale'],
            'terminal_prior_atol': terminal_prior_diagnostics['atol'],
            'terminal_prior_rtol': terminal_prior_diagnostics['rtol'],
            'terminal_prior_threshold': terminal_prior_diagnostics['threshold'],
            'terminal_prior_severity': terminal_prior_diagnostics['severity'],
        }
        return velocity_loss, diagnostics

    def _cosmos_deployment_joint_rollout_step(
        self, batch, batch_idx, *, student_steps
    ):
        """Train one full joint student rollout from deployment noise to x0."""
        student_steps = int(student_steps)
        if student_steps not in (1, 2, 4):
            raise ValueError("deployment student_steps must be one of 1, 2, 4")
        if self.gradient_accumulation_steps != 1:
            raise ValueError(
                "deployment joint rollout requires gradient_accumulation_steps=1"
            )
        rollout_grad_mode = str(
            getattr(self.config, "opd_rollout_grad_mode", "endpoint")
        ).lower()
        if rollout_grad_mode == "endpoint":
            raise ValueError(
                "deployment joint rollout is incompatible with "
                "opd_rollout_grad_mode='endpoint' because final video/action "
                "states are detached"
            )

        batch = self.convert_input_format(batch)
        teacher = self._action_teacher_model
        if not (
            hasattr(teacher, "predict_raw_latent_target")
            and getattr(teacher, "raw_inference_enabled", False)
        ):
            raise RuntimeError(
                "Cosmos deployment rollout requires raw latent endpoint inference."
            )

        B = int(batch["actions"].shape[0])
        latent_shape = (
            B,
            int(getattr(self.config, "cosmos_latent_channels", 16)),
            int(getattr(self.config, "cosmos_latent_frames", 9)),
            int(getattr(self.config, "cosmos_latent_height", 28)),
            int(getattr(self.config, "cosmos_latent_width", 28)),
        )
        video_frames = latent_shape[2]
        anchor_t_norm = torch.full(
            (B, video_frames),
            float(getattr(self.config, "cosmos_latent_t_max", 80.0 / 81.0)),
            device=self.device,
            dtype=torch.float32,
        )
        anchor_r_norm = torch.full(
            (B, video_frames),
            float(getattr(self.config, "cosmos_latent_t_min", 4.0 / 5.0)),
            device=self.device,
            dtype=torch.float32,
        )
        anchor_noise = torch.randn(
            latent_shape, device=self.device, dtype=batch["actions"].dtype
        )
        with torch.no_grad():
            endpoint_result = teacher.predict_raw_latent_target(
                batch,
                noise=anchor_noise,
                t=anchor_t_norm,
                r=anchor_r_norm,
                epsilon=float(self.config.cosmos_latent_epsilon),
                include_cdiff=False,
            )
        video_x0 = endpoint_result["cosmos_latent_x0"].to(
            device=self.device, dtype=batch["latents"].dtype
        )
        action_x0_full = cosmos_actions_to_flowmap_x0(
            endpoint_result["actions"],
            target_shape=tuple(batch["actions"].shape),
            q01=self.config.norm_stat["q01"],
            q99=self.config.norm_stat["q99"],
            inverse_used_action_channel_ids=self.config.inverse_used_action_channel_ids,
            device=self.device,
            dtype=batch["actions"].dtype,
            packing_schema=self.config.action_packing_schema,
            downsample_factor=self.config.action_downsample_factor,
        )

        batch["latents"] = video_x0
        ref_shape = video_x0.shape
        video_t = torch.full(
            (B, video_frames), 1000.0, device=self.device
        )
        video_r = torch.zeros_like(video_t)
        action_t, action_r = broadcast_joint_action_timesteps(
            video_t, video_r, action_frames=batch["actions"].shape[2]
        )
        video_noise = anchor_noise.to(device=self.device, dtype=video_x0.dtype)
        action_noise = torch.randn_like(action_x0_full)

        input_dict = self._prepare_base_dict(batch)
        input_dict["latent_dict"]["noisy_latents"] = video_noise
        input_dict["latent_dict"]["timesteps"] = video_t
        input_dict["latent_dict"]["targets"] = (
            self.train_scheduler_latent.training_target(
                video_x0, video_noise, video_t
            )
        )
        action_base = input_dict["action_dict"]
        action_base["noisy_latents"] = action_noise
        action_base["timesteps"] = action_t
        action_base["targets"] = self.train_scheduler_action.training_target(
            batch["actions"], action_noise, action_t
        )
        action_downsample = int(self.config.action_downsample_factor)
        student_action_dict = {
            "noisy_latents": action_noise[:, :, ::action_downsample],
            "latent": action_base["latent"][:, :, ::action_downsample],
            "timesteps": action_t[:, ::action_downsample],
            "cond_timesteps": action_base["cond_timesteps"][
                :, ::action_downsample
            ],
            "text_emb": action_base["text_emb"],
            "grid_id": _downsample_action_grid_id(
                action_base.get("grid_id"),
                action_base["latent"],
                action_downsample,
            )
            if action_base.get("grid_id") is not None
            else None,
            "actions_mask": batch["actions_mask"][:, :, ::action_downsample],
        }
        student_input = {
            "latent_dict": input_dict["latent_dict"],
            "action_dict": student_action_dict,
            "chunk_size": input_dict["chunk_size"],
            "window_size": input_dict["window_size"],
        }
        cfg_scale = self.config.cfg_min + torch.rand(1).item() * (
            self.config.cfg_max - self.config.cfg_min
        )
        empty_emb = self.empty_emb.expand(B, -1, -1)
        video_final, _, _, action_final = self._student_euler_integrate(
            noisy_latents=video_noise,
            timesteps=video_t,
            target_r=video_r,
            base_input_dict=student_input,
            empty_emb=empty_emb,
            cfg_scale=cfg_scale,
            ref_shape=ref_shape,
            B=B,
            num_frames=video_frames,
            K_steps=student_steps,
            action_target_r=action_r,
            return_final_action=True,
            return_final_action_state=True,
        )
        endpoint_losses = deployment_endpoint_losses(
            video_final,
            video_x0,
            action_final,
            batch["actions"][:, :, ::action_downsample],
            batch["actions_mask"][:, :, ::action_downsample],
            action_weight=float(self.config.deployment_action_weight),
        )
        loss = endpoint_losses["total"]
        is_finite, nonfinite_origins = _synchronized_nonfinite_decision(
            loss=loss,
            local_flags={
                "video": not bool(
                    torch.isfinite(endpoint_losses["video"].detach()).all().item()
                ),
                "action": not bool(
                    torch.isfinite(endpoint_losses["action"].detach()).all().item()
                ),
            },
            device=self.device,
        )
        zero = torch.zeros((), device=self.device, dtype=loss.dtype)
        if is_finite:
            loss.backward()
        deployment_video_loss = (
            endpoint_losses["video"].detach() if is_finite else zero
        )
        deployment_action_loss = (
            endpoint_losses["action"].detach() if is_finite else zero
        )
        deployment_total_loss = loss.detach() if is_finite else zero
        return {
            "loss": deployment_total_loss,
            "deployment_video_endpoint_loss": deployment_video_loss,
            "deployment_action_endpoint_loss": deployment_action_loss,
            "deployment_total_loss": deployment_total_loss,
            "deployment_student_steps": torch.tensor(
                float(student_steps), device=self.device
            ),
            "deployment_t_start": video_t[0, 0].detach(),
            "deployment_t_end": video_r[0, 0].detach(),
            "should_sync": True,
            "skip_step": not is_finite,
            "nonfinite_origins": nonfinite_origins,
        }

    def _build_cosmos_shifted_shared_query(
        self,
        batch,
        input_dict,
        *,
        student_steps=None,
        generator=None,
        student_model=None,
        cfg_scale=None,
    ):
        """Build the detached shifted-grid joint z_r shared by training/probes."""
        batch_size = int(batch["latents"].shape[0])
        latent_shape = (
            batch_size,
            int(getattr(self.config, "cosmos_latent_channels", 16)),
            int(getattr(self.config, "cosmos_latent_frames", 9)),
            int(getattr(self.config, "cosmos_latent_height", 28)),
            int(getattr(self.config, "cosmos_latent_width", 28)),
        )
        action_shape = tuple(batch["actions"].shape)
        if len(action_shape) != 5:
            raise ValueError("Aligned Cosmos query requires rank-5 actions")
        action_downsample = int(self.config.action_downsample_factor)
        action_frames = action_shape[2] // action_downsample
        if action_frames <= 0:
            raise ValueError("Aligned Cosmos query requires compact action frames")

        rollout_choices = getattr(
            self.config, "opd_danceopd_rollout_steps", (2, 4)
        )
        if isinstance(rollout_choices, int):
            rollout_choices = (rollout_choices,)
        rollout_choices = tuple(int(value) for value in rollout_choices)
        canonical_rollout_choices = {
            (1,),
            (2,),
            (4,),
            (2, 4),
        }
        if rollout_choices not in canonical_rollout_choices:
            raise ValueError(
                "Aligned Cosmos rollout steps must be one canonical choice set: "
                "(1,), (2,), (4,), or (2, 4)"
            )
        if student_steps is None:
            choice_index = torch.randint(
                0, len(rollout_choices), (1,), device=self.device
            )
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(choice_index, src=0)
            student_steps = rollout_choices[int(choice_index.item())]
        else:
            student_steps = int(student_steps)
            if student_steps not in rollout_choices:
                raise ValueError(
                    "Aligned Cosmos shared query student_steps must belong to "
                    f"the canonical configured choices {rollout_choices!r}"
                )
        if (
            student_steps == 1
            and float(
                getattr(
                    self.config,
                    "opd_danceopd_velocity_weight",
                    1.0,
                )
            )
            > 0.0
        ):
            raise ValueError(
                "Aligned Cosmos K=1 has no nonterminal state for a field loss"
            )

        video_sigmas = build_shifted_terminal_path(
            steps=student_steps,
            shift=float(self.config.snr_shift),
            device=self.device,
            dtype=torch.float32,
        )
        action_sigmas = build_shifted_terminal_path(
            steps=student_steps,
            shift=float(self.config.action_snr_shift),
            device=self.device,
            dtype=torch.float32,
        )
        num_train_timesteps = float(self.config.num_train_timesteps)
        video_path = (
            video_sigmas[:, None, None]
            .expand(student_steps + 1, batch_size, latent_shape[2])
            * num_train_timesteps
        )
        action_path = (
            action_sigmas[:, None, None]
            .expand(student_steps + 1, batch_size, action_frames)
            * num_train_timesteps
        )

        video_prior = torch.randn(
            latent_shape,
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )
        native_action_prior = torch.randn(
            batch_size,
            16,
            7,
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )
        action_channels = action_shape[1]
        used_action_channel_ids = torch.as_tensor(
            self.config.used_action_channel_ids,
            device=self.device,
            dtype=torch.long,
        )
        if (
            used_action_channel_ids.numel() != 7
            or int(used_action_channel_ids.min().item()) < 0
            or int(used_action_channel_ids.max().item()) >= action_channels
        ):
            raise ValueError(
                "Aligned Cosmos query requires seven valid action channel IDs"
            )
        aligned_action_prior = torch.zeros(
            batch_size,
            16,
            action_channels,
            device=self.device,
            dtype=torch.float32,
        )
        aligned_action_prior[..., used_action_channel_ids] = native_action_prior
        full_action_prior = pack_actions_for_downsample(
            aligned_action_prior,
            action_shape,
            downsample_factor=action_downsample,
            schema=self.config.action_packing_schema,
        )
        current_video = video_prior
        current_action = full_action_prior[:, :, ::action_downsample]

        video_base = {
            key: value
            for key, value in input_dict["latent_dict"].items()
            if key not in ("latent", "noisy_latents", "targets", "timesteps")
        }
        action_base = input_dict["action_dict"]
        action_cond_t = action_base["cond_timesteps"][:, ::action_downsample]
        action_grid = _downsample_action_grid_id(
            action_base.get("grid_id"),
            action_base["latent"],
            action_downsample,
        )
        action_mask = action_base.get("actions_mask")
        if action_mask is not None:
            action_mask = action_mask[:, :, ::action_downsample]
        empty_emb = self.empty_emb.expand(batch_size, -1, -1)
        if cfg_scale is None:
            cfg_scale = float(self.config.cfg_min) + torch.rand(1).item() * (
                float(self.config.cfg_max) - float(self.config.cfg_min)
            )
        if student_model is None:
            student_model = self.student

        def joint_context(video_state, action_state):
            return {
                "video_base": {**video_base, "latent": video_state},
                "action_latent": action_state,
                "action_cond_t": action_cond_t,
                "action_text": action_base["text_emb"],
                "action_grid": action_grid,
                "action_mask": action_mask,
                "chunk_size": input_dict["chunk_size"],
                "window_size": input_dict["window_size"],
                "student_model": student_model,
                "empty_emb": empty_emb,
                "cfg_scale": cfg_scale,
                "batch_size": batch_size,
                "ref_shape": latent_shape,
                "action_frames": action_frames,
            }

        if student_steps == 1:
            # K=1 has no nonterminal deployment state.  Its endpoint-only arm
            # compares the Student's direct prior->endpoint prediction with
            # the Teacher's same-prior 8-step endpoint.  Do not invent a
            # nonterminal z_r or a meaningless field query.
            query_indices = torch.zeros(
                batch_size, device=self.device, dtype=torch.long
            )
            query_video = current_video.detach()
            query_action = current_action.detach()
            query_video_t = video_path[0].detach()
            query_action_t = action_path[0].detach()
        else:
            video_states = []
            action_states = []
            with torch.no_grad():
                for step_index in range(student_steps):
                    video_t = video_path[step_index]
                    video_r = video_path[step_index + 1]
                    action_t = action_path[step_index]
                    action_r = action_path[step_index + 1]
                    context = joint_context(current_video, current_action)
                    joint_input = self._mechanism_joint_input(
                        current_video,
                        current_action,
                        video_t,
                        action_t,
                        context,
                    )
                    self._init_joint_mask(joint_input)
                    video_velocity, action_velocity_seq = (
                        self._student_joint_forward(
                            student_model,
                            joint_input,
                            empty_emb,
                            video_r,
                            action_r,
                            cfg_scale=cfg_scale,
                            batch_size=batch_size,
                            ref_shape=latent_shape,
                            require_action=True,
                        )
                    )
                    if action_velocity_seq is None:
                        raise RuntimeError(
                            "Aligned Cosmos query requires a Student action head"
                        )
                    action_velocity = self._extract_action_v(
                        action_velocity_seq, action_frames
                    )
                    current_video, current_action = self._joint_euler_update(
                        current_video,
                        current_action,
                        video_velocity,
                        action_velocity,
                        video_t,
                        video_r,
                        action_t,
                        action_r,
                    )
                    video_states.append(current_video.detach())
                    action_states.append(current_action.detach())

            query_indices = sample_nonterminal_semantic_query_indices(
                video_sigmas, batch_size
            )
            state_indices = query_indices - 1
            query_video = select_per_sample_trajectory_state(
                torch.stack(video_states), state_indices
            ).detach()
            query_action = select_per_sample_trajectory_state(
                torch.stack(action_states), state_indices
            ).detach()
            query_video_t = select_per_sample_trajectory_state(
                video_path, query_indices
            ).detach()
            query_action_t = select_per_sample_trajectory_state(
                action_path, query_indices
            ).detach()
        return {
            "student_steps": student_steps,
            "video_prior": video_prior,
            "native_action_prior": native_action_prior,
            "query_indices": query_indices,
            "query_video": query_video,
            "query_action": query_action,
            "query_video_t": query_video_t,
            "query_action_t": query_action_t,
            "query_sigma_frames": query_video_t / num_train_timesteps,
            "query_sigma": query_video_t[:, 0] / num_train_timesteps,
            "joint_context": joint_context,
            "video_frame_count": latent_shape[2],
            "action_frame_count": action_frames,
            "batch_size": batch_size,
            "latent_shape": latent_shape,
            "empty_emb": empty_emb,
            "cfg_scale": cfg_scale,
            "action_mask": action_mask,
        }

    def _cosmos_aligned_video_opd_step(self, batch, batch_idx):
        """Match one canonical deployment-reached joint state to Cosmos video targets."""
        del batch_idx
        endpoint_weight = float(
            getattr(self.config, "opd_danceopd_endpoint_weight", 1.0)
        )
        field_weight = float(
            getattr(self.config, "opd_danceopd_velocity_weight", 1.0)
        )
        if (
            not float("-inf") < endpoint_weight < float("inf")
            or endpoint_weight < 0.0
            or not float("-inf") < field_weight < float("inf")
            or field_weight < 0.0
        ):
            raise ValueError(
                "Aligned Cosmos OPD endpoint/field weights must be finite and "
                "non-negative"
            )
        endpoint_enabled = endpoint_weight > 0.0
        field_enabled = field_weight > 0.0
        if not endpoint_enabled and not field_enabled:
            zero = torch.zeros((), device=self.device, dtype=torch.float32)
            return {
                "loss": zero,
                "opd_endpoint_loss": zero,
                "opd_same_state_velocity_loss": zero,
                "opd_endpoint_contrib": zero,
                "opd_same_state_velocity_contrib": zero,
                "opd_query_sigma": zero,
                "opd_query_index": zero,
                "opd_student_steps": zero,
                "opd_teacher_steps": zero,
                "opd_valid_video_frames": zero,
                "opd_same_prior_verified": zero,
                "opd_canonical_state_verified": zero,
                "should_sync": True,
                "skip_step": False,
                "nonfinite_origins": {},
            }

        batch = self.convert_input_format(batch)
        teacher = self._action_teacher_model
        teacher_contract_ok = getattr(
            teacher, "raw_inference_enabled", False
        )
        teacher_contract_ok &= (
            not field_enabled
            or hasattr(teacher, "predict_raw_joint_latent_velocity")
        )
        teacher_contract_ok &= (
            not endpoint_enabled
            or hasattr(teacher, "predict_raw_same_prior_endpoint")
        )
        if not teacher_contract_ok:
            raise RuntimeError(
                "Aligned Cosmos video OPD requires each enabled raw Teacher API"
            )

        input_dict = self._prepare_base_dict(batch)
        shared = self._build_cosmos_shifted_shared_query(
            batch, input_dict
        )
        student_steps = shared["student_steps"]
        batch_size = shared["batch_size"]
        latent_shape = shared["latent_shape"]
        action_frames = shared["action_frame_count"]
        video_prior = shared["video_prior"]
        native_action_prior = shared["native_action_prior"]
        query_indices = shared["query_indices"]
        query_video = shared["query_video"]
        query_action = shared["query_action"]
        query_video_t = shared["query_video_t"]
        query_action_t = shared["query_action_t"]
        query_sigma_frames = shared["query_sigma_frames"]
        query_sigma = shared["query_sigma"]
        joint_context = shared["joint_context"]
        empty_emb = shared["empty_emb"]
        cfg_scale = shared["cfg_scale"]

        def synchronized_teacher_call(label, callback):
            local_error = None
            result = None
            try:
                result = callback()
            except Exception as exc:
                local_error = exc
            if not all_ranks_finite(local_error is None, device=self.device):
                raise RuntimeError(
                    f"Aligned Cosmos OPD {label} failed on at least one rank"
                ) from local_error
            return result

        field_result = None
        endpoint_result = None
        with torch.no_grad():
            if field_enabled:
                field_result = synchronized_teacher_call(
                    "same-state Teacher query",
                    lambda: teacher.predict_raw_joint_latent_velocity(
                        batch,
                        query_latent=query_video.float(),
                        query_action=query_action,
                        t=query_sigma_frames.float(),
                    ),
                )
            if endpoint_enabled:
                endpoint_result = synchronized_teacher_call(
                    "same-prior Teacher endpoint",
                    lambda: teacher.predict_raw_same_prior_endpoint(
                        batch,
                        video_prior=video_prior,
                        action_prior=native_action_prior,
                        teacher_steps=8,
                    ),
                )

        required_field_keys = (
            "cosmos_joint_query",
            "cosmos_latent_velocity",
            "cosmos_video_frame_mask",
        )
        required_endpoint_keys = (
            "endpoint_video",
            "video_frame_mask",
            "effective_teacher_steps",
            "video_prior_sha256",
            "action_prior_sha256",
        )
        local_contract_ok = (
            not field_enabled
            or all(key in field_result for key in required_field_keys)
        )
        local_contract_ok &= (
            not endpoint_enabled
            or all(key in endpoint_result for key in required_endpoint_keys)
        )
        if not all_ranks_finite(local_contract_ok, device=self.device):
            raise RuntimeError(
                "Aligned Cosmos OPD Teacher response contract mismatch"
            )

        if field_enabled:
            canonical_video = field_result["cosmos_joint_query"].to(
                device=self.device, dtype=query_video.dtype
            ).detach()
            teacher_field = field_result["cosmos_latent_velocity"].to(
                device=self.device, dtype=canonical_video.dtype
            ).detach()
            field_mask = field_result["cosmos_video_frame_mask"].to(
                device=self.device, dtype=torch.bool
            )
        else:
            canonical_video = query_video.detach()
            teacher_field = None
            field_mask = None

        if endpoint_enabled:
            teacher_endpoint = endpoint_result["endpoint_video"].to(
                device=self.device, dtype=canonical_video.dtype
            ).detach()
            endpoint_mask = endpoint_result["video_frame_mask"].to(
                device=self.device, dtype=torch.bool
            )
            effective_teacher_steps = endpoint_result[
                "effective_teacher_steps"
            ]
        else:
            teacher_endpoint = None
            endpoint_mask = None
            effective_teacher_steps = 0

        video_frame_mask = field_mask if field_enabled else endpoint_mask
        shape_contract_ok = (
            canonical_video.shape == query_video.shape
            and video_frame_mask.shape
            == (batch_size, canonical_video.shape[2])
            and bool(video_frame_mask.any(dim=1).all().item())
        )
        shape_contract_ok &= (
            not field_enabled
            or teacher_field.shape == canonical_video.shape
        )
        shape_contract_ok &= (
            not endpoint_enabled
            or (
                teacher_endpoint.shape == canonical_video.shape
                and endpoint_mask.shape == video_frame_mask.shape
                and type(effective_teacher_steps) is int
                and effective_teacher_steps == 8
            )
        )
        shape_contract_ok &= (
            not (field_enabled and endpoint_enabled)
            or torch.equal(video_frame_mask, endpoint_mask)
        )
        if not all_ranks_finite(shape_contract_ok, device=self.device):
            raise RuntimeError(
                "Aligned Cosmos OPD Teacher shape, mask, or step contract mismatch"
            )

        expanded_mask = video_frame_mask[:, None, :, None, None].expand_as(
            canonical_video
        )
        canonical_state_verified = torch.equal(
            canonical_video.masked_select(expanded_mask),
            query_video.to(canonical_video).masked_select(expanded_mask),
        )
        if not all_ranks_finite(
            canonical_state_verified, device=self.device
        ):
            raise RuntimeError(
                "Aligned Cosmos OPD canonical valid-video frames differ from "
                "the selected Student state"
            )

        zero = torch.zeros(
            (), device=self.device, dtype=canonical_video.dtype
        )
        field_loss = zero
        if field_enabled:
            field_context = joint_context(canonical_video, query_action)
            field_input = self._mechanism_joint_input(
                canonical_video,
                query_action,
                query_video_t,
                query_action_t,
                field_context,
            )
            self._init_joint_mask(field_input)
            student_field = self._student_joint_forward(
                self.student,
                field_input,
                empty_emb,
                query_video_t,
                query_action_t,
                cfg_scale=cfg_scale,
                batch_size=batch_size,
                ref_shape=latent_shape,
                require_action=False,
            )
            field_loss = masked_video_velocity_mse(
                student_field,
                teacher_field,
                video_frame_mask,
            )

        endpoint_loss = zero
        if endpoint_enabled:
            anchor_context = joint_context(canonical_video, query_action)
            anchor_input = self._mechanism_joint_input(
                canonical_video,
                query_action,
                query_video_t,
                query_action_t,
                anchor_context,
            )
            self._init_joint_mask(anchor_input)
            student_anchor = self._student_joint_forward(
                self.student,
                anchor_input,
                empty_emb,
                torch.zeros_like(query_video_t),
                torch.zeros_like(query_action_t),
                cfg_scale=cfg_scale,
                batch_size=batch_size,
                ref_shape=latent_shape,
                require_action=False,
            )
            endpoint_loss = aligned_anchor_mse(
                canonical_video,
                query_sigma,
                student_anchor,
                teacher_endpoint,
                video_frame_mask,
            )

        endpoint_contrib = endpoint_loss * endpoint_weight
        field_contrib = field_loss * field_weight
        loss = endpoint_contrib + field_contrib
        is_finite, nonfinite_origins = _synchronized_nonfinite_decision(
            loss=loss,
            local_flags={
                "opd_endpoint": bool(
                    endpoint_enabled
                    and not torch.isfinite(endpoint_loss.detach()).all().item()
                ),
                "opd_compositional": bool(
                    field_enabled
                    and not torch.isfinite(field_loss.detach()).all().item()
                ),
            },
            device=self.device,
        )
        if is_finite and loss.requires_grad:
            loss.backward()
        return {
            "loss": loss.detach() if is_finite else zero,
            "opd_endpoint_loss": endpoint_loss.detach() if is_finite else zero,
            "opd_same_state_velocity_loss": (
                field_loss.detach() if is_finite else zero
            ),
            "opd_endpoint_contrib": (
                endpoint_contrib.detach() if is_finite else zero
            ),
            "opd_same_state_velocity_contrib": (
                field_contrib.detach() if is_finite else zero
            ),
            "opd_query_sigma": query_sigma.float().mean().detach(),
            "opd_query_index": query_indices.float().mean().detach(),
            "opd_student_steps": torch.tensor(
                float(student_steps), device=self.device
            ),
            "opd_teacher_steps": torch.tensor(
                float(effective_teacher_steps), device=self.device
            ),
            "opd_valid_video_frames": video_frame_mask.float().sum().detach(),
            "opd_same_prior_verified": torch.tensor(
                float(endpoint_enabled), device=self.device
            ),
            "opd_canonical_state_verified": torch.tensor(
                float(canonical_state_verified), device=self.device
            ),
            "should_sync": True,
            "skip_step": not is_finite,
            "nonfinite_origins": nonfinite_origins,
        }

    def _cosmos_latent_full_opd_aux_transition_step(self, batch, batch_idx, kto_paopd=False):
        """Cosmos full OPD: independent endpoint rollout plus DanceOPD field loss."""
        if kto_paopd:
            raise NotImplementedError(
                "KTO-PAOPD is not wired for Cosmos full OPD; use default OPD."
            )
        batch = self.convert_input_format(batch)
        teacher = self._action_teacher_model
        if not (
            hasattr(teacher, 'predict_raw_latent_target')
            and hasattr(teacher, 'predict_raw_latent_velocity')
            and hasattr(teacher, 'predict_raw_joint_latent_velocity')
            and getattr(teacher, 'raw_inference_enabled', False)
        ):
            raise RuntimeError(
                "Cosmos full OPD requires raw latent target and velocity APIs."
            )

        B = int(batch['actions'].shape[0])
        latent_shape = (
            B,
            int(getattr(self.config, 'cosmos_latent_channels', 16)),
            int(getattr(self.config, 'cosmos_latent_frames', 9)),
            int(getattr(self.config, 'cosmos_latent_height', 28)),
            int(getattr(self.config, 'cosmos_latent_width', 28)),
        )
        num_frames = latent_shape[2]
        video_t, video_r, video_t_norm, video_r_norm, _ = (
            self.sample_cosmos_latent_timestep_mixed(
                B, num_frames, dtype=torch.float32, device=self.device
            )
        )
        video_t, video_r, focused_mask = apply_full_endpoint_focus(
            video_t,
            video_r,
            probability=float(getattr(self.config, 'opd_endpoint_focus_prob', 0.0)),
            num_train_timesteps=self.config.num_train_timesteps,
            focus_timestep=float(getattr(
                self.config, 'cosmos_latent_t_max', 80.0 / 81.0
            )) * self.config.num_train_timesteps,
            focus_target_timestep=float(getattr(
                self.config, 'cosmos_latent_t_min', 4.0 / 5.0
            )) * self.config.num_train_timesteps,
        )
        video_t, video_r = constrain_cosmos_teacher_timestep_pair(
            video_t,
            video_r,
            t_min=float(getattr(self.config, 'cosmos_latent_t_min', 4.0 / 5.0)),
            t_max=float(getattr(self.config, 'cosmos_latent_t_max', 80.0 / 81.0)),
            num_train_timesteps=self.config.num_train_timesteps,
        )
        teacher_target_r = video_r
        endpoint_sigma = sample_endpoint_sigmas(
            batch_size=B,
            alpha=5.0,
            beta=2.0,
            max_sigma=0.25,
            device=self.device,
        )
        video_r = (
            endpoint_sigma[:, None].expand(B, num_frames)
            * self.config.num_train_timesteps
        )
        video_t_norm = (video_t / self.config.num_train_timesteps).clamp(0.0, 1.0)
        video_r_norm = (video_r / self.config.num_train_timesteps).clamp(0.0, 1.0)
        teacher_target_r_norm = (
            teacher_target_r / self.config.num_train_timesteps
        ).clamp(0.0, 1.0)
        video_noise_teacher = torch.randn(
            latent_shape, device=self.device, dtype=torch.float32
        )
        with torch.no_grad():
            anchor_result = teacher.predict_raw_latent_target(
                batch,
                noise=video_noise_teacher,
                t=video_t_norm,
                r=teacher_target_r_norm,
                epsilon=float(getattr(self.config, 'cosmos_latent_epsilon', 0.001)),
                include_cdiff=False,
            )
        full_anchor = anchor_result['cosmos_latent_x0'].to(
            device=self.device, dtype=batch['actions'].dtype
        )
        use_teacher_action_anchor = bool(
            getattr(self.config, 'cosmos_use_teacher_action_anchor', False)
        )
        if use_teacher_action_anchor:
            batch['actions'] = cosmos_actions_to_flowmap_x0(
                anchor_result['actions'],
                target_shape=tuple(batch['actions'].shape),
                q01=self.config.norm_stat['q01'],
                q99=self.config.norm_stat['q99'],
                inverse_used_action_channel_ids=self.config.inverse_used_action_channel_ids,
                device=self.device,
                dtype=batch['actions'].dtype,
                packing_schema=self.config.action_packing_schema,
                downsample_factor=self.config.action_downsample_factor,
            )
        full_video_noise = video_noise_teacher.to(
            device=self.device, dtype=full_anchor.dtype
        )
        sigma_t = video_t_norm[:, None, :, None, None].to(full_anchor)
        full_video_noisy_t = (
            (1.0 - sigma_t) * full_anchor + sigma_t * full_video_noise
        )
        h_slice, w_slice = center_spatial_crop_slices(
            full_anchor.shape[-2],
            full_anchor.shape[-1],
            crop_size=int(getattr(self.config, 'opd_cosmos_spatial_crop_size', 0)),
        )
        crop_is_active = (
            h_slice.start != 0
            or h_slice.stop != full_anchor.shape[-2]
            or w_slice.start != 0
            or w_slice.stop != full_anchor.shape[-1]
        )
        if crop_is_active:
            batch['latents'] = full_anchor[..., h_slice, w_slice].contiguous()
            video_noise = full_video_noise[..., h_slice, w_slice].contiguous()
            video_noisy_t = full_video_noisy_t[..., h_slice, w_slice].contiguous()
            teacher_crop_context = {
                'full_anchor': full_anchor.detach(),
                'h_slice': h_slice,
                'w_slice': w_slice,
            }
        else:
            batch['latents'] = full_anchor
            video_noise = full_video_noise
            video_noisy_t = full_video_noisy_t
            teacher_crop_context = None
        ref_shape = batch['latents'].shape
        input_dict = self._prepare_base_dict(batch)
        input_dict['latent_dict']['noisy_latents'] = video_noisy_t
        input_dict['latent_dict']['timesteps'] = video_t
        input_dict['latent_dict']['targets'] = self.train_scheduler_latent.training_target(
            batch['latents'], video_noise, video_t
        )

        action_r = None
        if self.distill_action or self.action_aware:
            action_frames = batch['actions'].shape[2]
            if use_teacher_action_anchor:
                action_t, action_r = broadcast_joint_action_timesteps(
                    video_t, video_r, action_frames=action_frames
                )
            else:
                action_t, action_r, _ = self.sample_timestep_mixed(
                    B,
                    action_frames,
                    dtype=torch.float32,
                    device=self.device,
                    scheduler=self.train_scheduler_action,
                )
            action_noise = torch.randn_like(batch['actions'])
            action_noisy = self.train_scheduler_action.add_noise(
                batch['actions'], action_noise, action_t, t_dim=2
            )
            action_base = input_dict['action_dict']
            action_base['noisy_latents'] = action_noisy
            action_base['timesteps'] = action_t
            action_base['targets'] = self.train_scheduler_action.training_target(
                batch['actions'], action_noise, action_t
            )
            action_downsample = int(getattr(self.config, 'action_downsample_factor', 4))
            student_action_dict = {
                'noisy_latents': action_noisy[:, :, ::action_downsample],
                'latent': action_base['latent'][:, :, ::action_downsample],
                'timesteps': action_t[:, ::action_downsample],
                'cond_timesteps': action_base['cond_timesteps'][:, ::action_downsample],
                'text_emb': action_base['text_emb'],
                'grid_id': _downsample_action_grid_id(
                    action_base.get('grid_id'), action_base['latent'], action_downsample
                ) if action_base.get('grid_id') is not None else None,
                'actions_mask': action_base['actions_mask'][:, :, ::action_downsample]
                if action_base.get('actions_mask') is not None else None,
            }
            student_input = {
                'latent_dict': input_dict['latent_dict'],
                'action_dict': student_action_dict,
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
        else:
            student_input = input_dict

        rollout_step_pairs = getattr(self.config, 'opd_rollout_step_pairs', [[8, 4]])
        if not rollout_step_pairs:
            raise ValueError('opd_rollout_step_pairs must contain at least one (N, K) pair')
        broadcast_index = (
            (lambda index: dist.broadcast(index, src=0))
            if dist.is_initialized()
            else None
        )
        teacher_steps, student_steps = sample_uniform_rollout_step_pair(
            rollout_step_pairs,
            device=self.device,
            broadcast_index=broadcast_index,
        )
        teacher_steps = max(1, int(teacher_steps))
        student_steps = max(1, int(student_steps))
        cfg_scale = self.config.cfg_min + torch.rand(1).item() * (
            self.config.cfg_max - self.config.cfg_min
        )
        empty_emb = self.empty_emb.expand(B, -1, -1)
        joint_action_rollout = bool(
            getattr(self.config, 'opd_joint_action_rollout', False)
        ) and action_r is not None
        rollout_out = self._student_euler_integrate(
            noisy_latents=video_noisy_t,
            timesteps=video_t,
            target_r=video_r,
            base_input_dict=student_input,
            empty_emb=empty_emb,
            cfg_scale=cfg_scale,
            ref_shape=ref_shape,
            B=B,
            num_frames=num_frames,
            K_steps=student_steps,
            action_target_r=action_r,
            return_final_action=joint_action_rollout,
            return_final_action_state=joint_action_rollout,
        )
        if joint_action_rollout:
            student_x_r, student_v_r, student_action_seq, student_action_x_r = rollout_out
        else:
            student_x_r, student_v_r = rollout_out
            student_action_seq = student_action_x_r = None

        sigma_r = video_r_norm[:, None, :, None, None]
        student_x0 = student_x_r - sigma_r.to(student_v_r) * student_v_r
        video_endpoint_loss = F.mse_loss(
            student_x0.float(), batch['latents'].detach().float()
        )
        action_endpoint_weight = float(getattr(
            self.config, 'opd_danceopd_action_endpoint_weight', 0.0
        ))
        if (
            not math.isfinite(action_endpoint_weight)
            or action_endpoint_weight < 0.0
        ):
            raise ValueError(
                'Cosmos OPD action endpoint weight must be finite and non-negative'
            )
        action_endpoint_loss = torch.zeros(
            (), device=self.device, dtype=video_endpoint_loss.dtype
        )
        if joint_action_rollout and action_endpoint_weight > 0.0:
            action_downsample = int(getattr(self.config, 'action_downsample_factor', 4))
            student_action_v = self._extract_action_v(
                student_action_seq, student_action_x_r.shape[2]
            )
            action_sigma_r = (
                action_r[:, None, ::action_downsample, None, None]
                / self.config.num_train_timesteps
            ).to(student_action_v)
            student_action_x0 = student_action_x_r - action_sigma_r * student_action_v
            action_target_x0 = batch['actions'][:, :, ::action_downsample]
            action_mask = batch.get('actions_mask')
            if action_mask is None:
                action_mask = torch.ones_like(action_target_x0[:, :1], dtype=torch.float32)
            else:
                action_mask = action_mask[:, :, ::action_downsample].float()
            action_diff = (student_action_x0.float() - action_target_x0.detach().float()) * action_mask
            action_denom = (action_mask.sum() * student_action_x0.shape[1]).clamp(min=1.0)
            action_endpoint_loss = action_diff.square().sum() / action_denom
        endpoint_loss = compose_cosmos_endpoint_loss(
            video_endpoint_loss,
            action_endpoint_loss,
            action_endpoint_weight=action_endpoint_weight,
        )
        endpoint_weight = float(getattr(self.config, 'opd_danceopd_endpoint_weight', 1.0))
        velocity_weight = float(getattr(self.config, 'opd_danceopd_velocity_weight', 1.0))
        if (
            not math.isfinite(endpoint_weight)
            or endpoint_weight < 0
            or not math.isfinite(velocity_weight)
            or velocity_weight < 0
        ):
            raise ValueError('Cosmos OPD endpoint and velocity weights must be finite and non-negative')
        if endpoint_weight == 0.0 and velocity_weight == 0.0:
            raise ValueError('Cosmos OPD requires a non-zero endpoint or velocity weight')
        if velocity_weight > 0:
            velocity_loss, dance_diagnostics = self._cosmos_danceopd_velocity_loss(
                batch,
                teacher,
                self._prepare_base_dict(batch),
                cfg_scale=cfg_scale,
                teacher_crop_context=teacher_crop_context,
            )
        else:
            velocity_loss = torch.zeros(
                (), device=self.device, dtype=endpoint_loss.dtype
            )
            dance_diagnostics = {
                'query_index_mean': velocity_loss,
                'query_sigma_mean': velocity_loss,
                'terminal_prior_max_error': velocity_loss,
                'terminal_prior_reference_scale': velocity_loss,
                'terminal_prior_atol': velocity_loss,
                'terminal_prior_rtol': velocity_loss,
                'terminal_prior_threshold': velocity_loss,
                'terminal_prior_severity': velocity_loss,
            }
        aux_weight = float(getattr(self.config, 'opd_aux_weight', 1.0))
        endpoint_contrib = endpoint_weight * endpoint_loss * aux_weight
        velocity_contrib = velocity_weight * velocity_loss * aux_weight
        loss = endpoint_contrib + velocity_contrib
        zero = torch.zeros((), device=self.device, dtype=loss.dtype)
        contrib_denom = (
            endpoint_contrib.detach().abs() + velocity_contrib.detach().abs()
        ).clamp(min=1e-12)
        is_finite, nonfinite_origins = _synchronized_nonfinite_decision(
            loss=loss,
            local_flags={
                "opd_endpoint": not bool(
                    torch.isfinite(video_endpoint_loss.detach()).all().item()
                ),
                "opd_action": not bool(
                    torch.isfinite(action_endpoint_loss.detach()).all().item()
                ),
                "opd_compositional": not bool(
                    torch.isfinite(velocity_loss.detach()).all().item()
                ),
            },
            device=self.device,
        )
        result = {
            'loss': loss.detach() if is_finite else zero,
            'opd_aux_loss': loss.detach() if is_finite else zero,
            'opd_same_state_velocity_loss': velocity_loss.detach(),
            'opd_same_state_velocity_contrib': velocity_contrib.detach(),
            'opd_same_state_velocity_ratio': velocity_contrib.detach().abs() / contrib_denom,
            'opd_video_transition_loss': zero,
            'opd_endpoint_aux_loss': endpoint_loss.detach(),
            'opd_video_endpoint_loss': video_endpoint_loss.detach(),
            'opd_action_endpoint_loss': action_endpoint_loss.detach(),
            'opd_local_fm_loss': zero,
            'opd_action_transition_loss': zero,
            'opd_action_local_fm_loss': zero,
            'opd_video_transition_contrib': zero,
            'opd_endpoint_aux_contrib': endpoint_contrib.detach(),
            'opd_action_endpoint_weight': torch.tensor(
                action_endpoint_weight, device=self.device, dtype=loss.dtype
            ),
            'opd_local_fm_contrib': zero,
            'opd_action_transition_contrib': zero,
            'opd_action_local_fm_contrib': zero,
            'opd_video_transition_ratio': zero,
            'opd_endpoint_aux_ratio': endpoint_contrib.detach().abs() / contrib_denom,
            'opd_local_fm_ratio': zero,
            'opd_action_transition_ratio': zero,
            'opd_action_local_fm_ratio': zero,
            'danceopd_query_index_mean': dance_diagnostics['query_index_mean'],
            'danceopd_query_sigma_mean': dance_diagnostics['query_sigma_mean'],
            'danceopd_terminal_prior_max_error': dance_diagnostics['terminal_prior_max_error'],
            'danceopd_terminal_prior_reference_scale': dance_diagnostics['terminal_prior_reference_scale'],
            'danceopd_terminal_prior_atol': dance_diagnostics['terminal_prior_atol'],
            'danceopd_terminal_prior_rtol': dance_diagnostics['terminal_prior_rtol'],
            'danceopd_terminal_prior_threshold': dance_diagnostics['terminal_prior_threshold'],
            'danceopd_terminal_prior_severity': dance_diagnostics['terminal_prior_severity'],
            'cosmos_opd_endpoint_focus_ratio': focused_mask.float().mean().detach(),
            'danceopd_endpoint_teacher_steps': teacher_steps,
            'danceopd_endpoint_student_steps': student_steps,
            'danceopd_endpoint_t_mean': video_t.float().mean().detach(),
            'danceopd_endpoint_r_mean': video_r.float().mean().detach(),
            'rollout_steps': student_steps,
            'teacher_steps': teacher_steps,
            'should_sync': True,
            'skip_step': not is_finite,
            'nonfinite_origins': nonfinite_origins,
        }
        if not is_finite:
            if getattr(self.config, 'rank', 0) == 0:
                logger.warning('[step %s] non-finite Cosmos full OPD loss, skipping', self.step)
            return result
        loss.backward()
        return result

    def _cosmos_latent_opd_aux_transition_step(self, batch, batch_idx, kto_paopd=False):
        """Pure Cosmos latent OPD queried on the student-visited state."""
        if kto_paopd:
            raise NotImplementedError(
                "KTO-PAOPD is not wired for pure Cosmos latent OPD yet; "
                "use OPD_AUX_VARIANT=default for this path."
            )

        profile_opd = bool(getattr(self.config, 'opd_profile', False))
        profile_times = {}
        profile_last = None

        def _profile_mark(name):
            nonlocal profile_last
            if not profile_opd:
                return
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            now = time.perf_counter()
            if profile_last is not None:
                profile_times[name] = now - profile_last
            profile_last = now

        _profile_mark('start')
        batch = self.convert_input_format(batch)
        _profile_mark('prepare_batch')

        teacher = self._action_teacher_model
        if not hasattr(teacher, 'predict_raw_latent_target'):
            raise RuntimeError(
                "Pure Cosmos latent OPD requires a teacher with "
                "predict_raw_latent_target()."
            )
        if not hasattr(teacher, 'predict_raw_latent_velocity'):
            raise RuntimeError(
                "Pure Cosmos latent OPD requires a teacher with "
                "predict_raw_latent_velocity()."
            )
        if not getattr(teacher, 'raw_inference_enabled', False):
            raise RuntimeError(
                "Pure Cosmos latent OPD requires "
                "cfg.cosmos_policy_use_raw_inference=True."
            )

        B = int(batch['actions'].shape[0])
        latent_shape = (
            B,
            int(getattr(self.config, "cosmos_latent_channels", 16)),
            int(getattr(self.config, "cosmos_latent_frames", 9)),
            int(getattr(self.config, "cosmos_latent_height", 28)),
            int(getattr(self.config, "cosmos_latent_width", 28)),
        )
        num_frames = latent_shape[2]
        video_t, video_r, video_t_norm, video_r_norm, _ = (
            self.sample_cosmos_latent_timestep_mixed(
                B, num_frames, dtype=torch.float32, device=self.device,
            )
        )
        video_r = self._apply_opd_low_noise_query_bias(video_t, video_r)
        video_t, video_r = constrain_cosmos_teacher_timestep_pair(
            video_t,
            video_r,
            t_min=float(getattr(self.config, 'cosmos_latent_t_min', 4.0 / 5.0)),
            t_max=float(getattr(self.config, 'cosmos_latent_t_max', 80.0 / 81.0)),
            num_train_timesteps=self.config.num_train_timesteps,
        )
        video_r_norm = (video_r / self.config.num_train_timesteps).clamp(0.0, 1.0)

        video_noise_teacher = torch.randn(
            latent_shape, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            cosmos_latent_teacher_result = teacher.predict_raw_latent_target(
                batch,
                noise=video_noise_teacher,
                t=video_t_norm,
                r=video_r_norm,
                epsilon=float(getattr(self.config, "cosmos_latent_epsilon", 0.001)),
                include_cdiff=False,
            )
        batch["latents"] = cosmos_latent_teacher_result[
            "cosmos_latent_x0"
        ].to(device=self.device, dtype=batch["actions"].dtype)
        video_noise = video_noise_teacher.to(
            device=batch["latents"].device, dtype=batch["latents"].dtype)
        ref_shape = batch["latents"].shape
        _profile_mark('cosmos_anchor')

        input_dict = self._prepare_base_dict(batch)
        sigma_t = video_t_norm[:, None, :, None, None].to(batch["latents"])
        video_noisy_latents = (
            (1.0 - sigma_t) * batch["latents"] + sigma_t * video_noise
        )
        video_v_target = self.train_scheduler_latent.training_target(
            batch["latents"], video_noise, video_t
        )
        input_dict['latent_dict']['noisy_latents'] = video_noisy_latents
        input_dict['latent_dict']['timesteps'] = video_t
        input_dict['latent_dict']['targets'] = video_v_target

        action_t = action_r = None
        if self.distill_action or self.action_aware:
            action_frames = batch['actions'].shape[2]
            action_t, action_r, _ = self.sample_timestep_mixed(
                B, action_frames, dtype=torch.float32, device=self.device,
                scheduler=self.train_scheduler_action,
            )
            action_noise = torch.randn_like(batch['actions'])
            action_noisy_latents = self.train_scheduler_action.add_noise(
                batch['actions'], action_noise, action_t, t_dim=2
            )
            input_dict['action_dict']['noisy_latents'] = action_noisy_latents
            input_dict['action_dict']['timesteps'] = action_t
            input_dict['action_dict']['targets'] = self.train_scheduler_action.training_target(
                batch['actions'], action_noise, action_t
            )
            action_ds = getattr(self.config, 'action_downsample_factor', 4)
            _ad = input_dict['action_dict']
            student_input_dict = {
                'latent_dict': input_dict['latent_dict'],
                'action_dict': {
                    'noisy_latents': _ad['noisy_latents'][:, :, ::action_ds],
                    'latent': _ad['latent'][:, :, ::action_ds],
                    'timesteps': _ad['timesteps'][:, ::action_ds],
                    'cond_timesteps': _ad['cond_timesteps'][:, ::action_ds],
                    'text_emb': _ad['text_emb'],
                    'grid_id': _downsample_action_grid_id(
                        _ad['grid_id'], _ad['latent'], action_ds) if _ad.get('grid_id') is not None else None,
                    'actions_mask': _ad['actions_mask'][:, :, ::action_ds] if _ad.get('actions_mask') is not None else None,
                },
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
        else:
            student_input_dict = input_dict

        rollout_step_pairs = getattr(
            self.config,
            'opd_rollout_step_pairs',
            getattr(self.config, 'rollout_step_pairs', [[1, 1]]),
        )
        if dist.is_initialized():
            idx = torch.randint(0, len(rollout_step_pairs), (1,), device=self.device)
            dist.broadcast(idx, src=0)
            N_steps, K_steps = rollout_step_pairs[idx.item()]
        else:
            import random
            N_steps, K_steps = random.choice(rollout_step_pairs)

        cfg_scale = self.config.cfg_min + torch.rand(1).item() * (
            self.config.cfg_max - self.config.cfg_min)
        empty_emb = self.empty_emb.expand(
            input_dict['latent_dict']['text_emb'].shape[0], -1, -1)

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        if hasattr(self.student, 'set_requires_gradient_sync'):
            self.student.set_requires_gradient_sync(should_sync)

        student_x_r, student_v_at_r = self._student_euler_integrate(
            noisy_latents=video_noisy_latents,
            timesteps=video_t,
            target_r=video_r,
            base_input_dict=student_input_dict,
            empty_emb=empty_emb,
            cfg_scale=cfg_scale,
            ref_shape=ref_shape,
            B=B,
            num_frames=num_frames,
            K_steps=max(1, int(K_steps)),
            action_target_r=action_r if (self.distill_action or self.action_aware) else None,
        )
        _profile_mark('student_rollout')

        with torch.no_grad():
            teacher_velocity_result = teacher.predict_raw_latent_velocity(
                batch,
                query_latent=student_x_r.detach().float(),
                t=video_r_norm.detach().float(),
            )
            teacher_v_at_r = teacher_velocity_result[
                "cosmos_latent_velocity"
            ].to(device=student_v_at_r.device, dtype=student_v_at_r.dtype)
        _profile_mark('cosmos_velocity_query')

        video_transition_param = str(getattr(
            self.config, 'video_transition_param', 'velocity')).lower()
        if video_transition_param not in ('velocity', 'x0', 'consistency'):
            raise ValueError(
                f"Invalid video_transition_param={video_transition_param!r}; "
                "expected velocity, x0, or consistency."
            )
        video_r_sigma = video_r / self.config.num_train_timesteps
        if video_transition_param == 'consistency':
            student_video_pred = self._consistency_function(
                student_v_at_r, student_x_r, video_r_sigma
            )
            teacher_video_pred = self._consistency_function(
                teacher_v_at_r, student_x_r.detach(), video_r_sigma
            )
            video_diff = student_video_pred.float() - teacher_video_pred.detach().float()
        elif video_transition_param == 'x0':
            sigma_r = video_r_sigma[:, None, :, None, None].to(student_v_at_r)
            student_video_pred = student_x_r - sigma_r * student_v_at_r
            teacher_video_pred = student_x_r.detach() - sigma_r.to(
                teacher_v_at_r) * teacher_v_at_r
            video_diff = student_video_pred.float() - teacher_video_pred.detach().float()
        else:
            video_diff = student_v_at_r.float() - teacher_v_at_r.detach().float()

        transition_loss_type = getattr(self.config, 'transition_loss_type', 'huber')
        transition_huber_c = getattr(self.config, 'transition_huber_c', 1e-3)
        if transition_loss_type == "huber":
            abs_diff = video_diff.abs()
            per_sample_loss = torch.where(
                abs_diff < transition_huber_c,
                0.5 * video_diff ** 2,
                transition_huber_c * (abs_diff - 0.5 * transition_huber_c),
            ).mean(dim=[1, 2, 3, 4])
        else:
            per_sample_loss = (video_diff ** 2).mean(dim=[1, 2, 3, 4])

        weight_type = getattr(self.config, 'weight_type', 'uniform')
        weight = self._get_timestep_weight(video_t.mean(dim=-1), weight_type).to(self.device)
        video_transition_loss = (per_sample_loss * weight).mean()

        opd_endpoint_aux_loss = torch.tensor(0.0, device=self.device)
        opd_endpoint_aux_weight = float(getattr(self.config, 'opd_endpoint_aux_weight', 0.0))
        if opd_endpoint_aux_weight > 0:
            sigma_r = video_r_sigma[:, None, :, None, None].to(student_v_at_r)
            student_endpoint_pred = student_x_r - sigma_r * student_v_at_r
            teacher_endpoint_pred = student_x_r.detach() - sigma_r.to(
                teacher_v_at_r) * teacher_v_at_r
            endpoint_diff = (
                student_endpoint_pred.float()
                - teacher_endpoint_pred.detach().float()
            )
            if transition_loss_type == "huber":
                abs_endpoint_diff = endpoint_diff.abs()
                endpoint_per_sample = torch.where(
                    abs_endpoint_diff < transition_huber_c,
                    0.5 * endpoint_diff ** 2,
                    transition_huber_c * (abs_endpoint_diff - 0.5 * transition_huber_c),
                ).mean(dim=[1, 2, 3, 4])
            else:
                endpoint_per_sample = (endpoint_diff ** 2).mean(dim=[1, 2, 3, 4])
            opd_endpoint_aux_loss = (endpoint_per_sample * weight).mean()

        local_fm_loss = torch.tensor(0.0, device=self.device)
        local_fm_weight = float(getattr(self.config, 'local_fm_weight', 0.0))
        if local_fm_weight > 0:
            video_v_target_at_r = self.train_scheduler_latent.training_target(
                batch['latents'], video_noise, video_r
            )
            local_fm_loss = (
                student_v_at_r.float() - video_v_target_at_r.detach().float()
            ).pow(2).mean()

        zero = torch.tensor(0.0, device=self.device)
        opd_action_transition_loss = zero
        opd_action_local_fm_loss = zero
        video_transition_weight = float(getattr(self.config, 'video_transition_weight', 1.0))
        raw_opd_video_transition_contrib = video_transition_weight * video_transition_loss
        raw_opd_endpoint_aux_contrib = opd_endpoint_aux_weight * opd_endpoint_aux_loss
        raw_opd_local_fm_contrib = local_fm_weight * local_fm_loss
        raw_opd_action_transition_contrib = zero
        raw_opd_action_local_fm_contrib = zero
        raw_transition_group = raw_opd_video_transition_contrib
        raw_anchor_group = raw_opd_endpoint_aux_contrib + raw_opd_local_fm_contrib
        transition_scale = torch.as_tensor(
            float(getattr(self.config, 'opd_transition_group_weight', 1.0)),
            device=self.device,
            dtype=raw_transition_group.dtype,
        )
        scaled_transition_group = raw_transition_group * transition_scale
        anchor_cap_ratio = float(getattr(self.config, 'opd_anchor_cap_ratio', -1.0))
        anchor_scale = torch.ones((), device=self.device, dtype=raw_anchor_group.dtype)
        if anchor_cap_ratio >= 0:
            anchor_cap = scaled_transition_group.detach().abs() * anchor_cap_ratio
            raw_anchor_abs = raw_anchor_group.detach().abs().clamp(min=1e-12)
            anchor_scale = torch.minimum(anchor_scale, anchor_cap / raw_anchor_abs)
        scaled_anchor_group = raw_anchor_group * anchor_scale

        opd_video_transition_contrib = raw_opd_video_transition_contrib * transition_scale
        opd_endpoint_aux_contrib = raw_opd_endpoint_aux_contrib * anchor_scale
        opd_local_fm_contrib = raw_opd_local_fm_contrib * anchor_scale
        opd_action_transition_contrib = zero
        opd_action_local_fm_contrib = zero
        raw_aux_loss = scaled_transition_group + scaled_anchor_group
        contrib_denom = (
            scaled_transition_group.detach().abs() + scaled_anchor_group.detach().abs()
        ).clamp(min=1e-12)
        opd_transition_group_ratio = scaled_transition_group.detach().abs() / contrib_denom
        opd_anchor_group_ratio = scaled_anchor_group.detach().abs() / contrib_denom
        opd_aux_weight = float(getattr(self.config, 'opd_aux_weight', 0.1))
        weighted_aux_loss = raw_aux_loss * opd_aux_weight
        loss = weighted_aux_loss

        loss_clip_value = getattr(self.config, 'opd_aux_loss_clip_value', None)
        if loss_clip_value is not None and getattr(self.config, 'loss_clip_enabled', True):
            loss_clip_value = float(loss_clip_value)
            if loss_clip_value >= 0:
                scale = (loss_clip_value / loss.detach().clamp(min=1e-12)).clamp(max=1.0)
                loss = loss * scale

        is_finite, nonfinite_origins = _synchronized_nonfinite_decision(
            loss=loss,
            local_flags={
                "opd_endpoint": not bool(
                    torch.isfinite(opd_endpoint_aux_loss.detach()).all().item()
                ),
                "opd_compositional": not bool(
                    torch.isfinite(video_transition_loss.detach()).all().item()
                )
                or not bool(torch.isfinite(local_fm_loss.detach()).all().item()),
                "opd_action": not bool(
                    torch.isfinite(opd_action_transition_loss.detach()).all().item()
                )
                or not bool(
                    torch.isfinite(opd_action_local_fm_loss.detach()).all().item()
                ),
            },
            device=self.device,
        )
        result = {
            'loss': loss.detach() if is_finite else zero,
            'opd_aux_loss': weighted_aux_loss.detach(),
            'opd_video_transition_loss': video_transition_loss.detach(),
            'opd_endpoint_aux_loss': opd_endpoint_aux_loss.detach(),
            'opd_local_fm_loss': local_fm_loss.detach(),
            'opd_action_transition_loss': opd_action_transition_loss.detach(),
            'opd_action_local_fm_loss': opd_action_local_fm_loss.detach(),
            'opd_video_transition_contrib': opd_video_transition_contrib.detach(),
            'opd_endpoint_aux_contrib': opd_endpoint_aux_contrib.detach(),
            'opd_local_fm_contrib': opd_local_fm_contrib.detach(),
            'opd_action_transition_contrib': opd_action_transition_contrib.detach(),
            'opd_action_local_fm_contrib': opd_action_local_fm_contrib.detach(),
            'opd_transition_group_scaled': scaled_transition_group.detach(),
            'opd_anchor_group_scaled': scaled_anchor_group.detach(),
            'opd_transition_scale': transition_scale.detach(),
            'opd_anchor_scale': anchor_scale.detach(),
            'opd_transition_group_ratio': opd_transition_group_ratio.detach(),
            'opd_anchor_group_ratio': opd_anchor_group_ratio.detach(),
            'opd_video_transition_ratio': (opd_video_transition_contrib.detach().abs() / contrib_denom),
            'opd_endpoint_aux_ratio': (opd_endpoint_aux_contrib.detach().abs() / contrib_denom),
            'opd_local_fm_ratio': (opd_local_fm_contrib.detach().abs() / contrib_denom),
            'opd_action_transition_ratio': zero,
            'opd_action_local_fm_ratio': zero,
            'rollout_steps': max(1, int(K_steps)),
            'teacher_steps': 1,
            'should_sync': should_sync,
            'skip_step': not is_finite,
            'nonfinite_origins': nonfinite_origins,
        }

        if not is_finite:
            if self.config.rank == 0:
                logger.warning(f"[step {self.step}] NaN/Inf Cosmos OPD aux loss, skipping")
            return result

        _profile_mark('loss_build')
        loss.backward()
        _profile_mark('backward')
        if profile_opd and getattr(self.config, 'rank', 0) == 0:
            logger.info(
                "[Cosmos OPD profile] " + ", ".join(
                    f"{k}={v:.3f}s" for k, v in profile_times.items()
                )
            )
        return result

    def _danceopd_independent_endpoint_loss(self, batch, *, cfg_scale):
        """Return video x0 loss for an independent teacher endpoint rollout."""
        B = batch['latents'].shape[0]
        ref_shape = batch['latents'].shape
        num_frames = ref_shape[2]
        action_downsample = int(getattr(self.config, 'action_downsample_factor', 4))

        input_dict = self._prepare_base_dict(batch)
        video_t, video_r, _ = self.sample_timestep_mixed(
            B,
            num_frames,
            dtype=torch.float32,
            device=self.device,
            pair_mode=getattr(self.config, 'opd_pair_mode', None),
        )
        video_r = self._apply_opd_low_noise_query_bias(video_t, video_r)
        video_noise = torch.randn_like(batch['latents'])
        video_noisy_t = self.train_scheduler_latent.add_noise(
            batch['latents'], video_noise, video_t, t_dim=2
        )
        input_dict['latent_dict']['noisy_latents'] = video_noisy_t
        input_dict['latent_dict']['timesteps'] = video_t
        input_dict['latent_dict']['targets'] = self.train_scheduler_latent.training_target(
            batch['latents'], video_noise, video_t
        )

        action_t = action_r = None
        if self.distill_action or self.action_aware:
            action_t, action_r, _ = self.sample_timestep_mixed(
                B,
                num_frames,
                dtype=torch.float32,
                device=self.device,
                scheduler=self.train_scheduler_action,
                pair_mode=getattr(self.config, 'opd_pair_mode', None),
            )
            action_r = self._apply_opd_low_noise_query_bias(action_t, action_r)
            action_noise = torch.randn_like(batch['actions'])
            action_noisy_t = self.train_scheduler_action.add_noise(
                batch['actions'], action_noise, action_t, t_dim=2
            )
            action_dict = input_dict['action_dict']
            action_dict['noisy_latents'] = action_noisy_t
            action_dict['timesteps'] = action_t
            action_dict['targets'] = self.train_scheduler_action.training_target(
                batch['actions'], action_noise, action_t
            )
            student_input = {
                'latent_dict': input_dict['latent_dict'],
                'action_dict': {
                    'noisy_latents': action_noisy_t[:, :, ::action_downsample],
                    'latent': action_dict['latent'][:, :, ::action_downsample],
                    'timesteps': action_t[:, ::action_downsample],
                    'cond_timesteps': action_dict['cond_timesteps'][:, ::action_downsample],
                    'text_emb': action_dict['text_emb'],
                    'grid_id': _downsample_action_grid_id(
                        action_dict.get('grid_id'), action_dict['latent'], action_downsample
                    ) if action_dict.get('grid_id') is not None else None,
                    'actions_mask': action_dict['actions_mask'][:, :, ::action_downsample]
                    if action_dict.get('actions_mask') is not None else None,
                },
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
        else:
            student_input = input_dict

        rollout_step_pairs = getattr(
            self.config,
            'opd_rollout_step_pairs',
            getattr(self.config, 'rollout_step_pairs', [[1, 1]]),
        )
        if not rollout_step_pairs:
            raise ValueError('opd_rollout_step_pairs must contain at least one (N, K) pair')
        if dist.is_initialized():
            pair_index = torch.randint(
                0, len(rollout_step_pairs), (1,), device=self.device
            )
            dist.broadcast(pair_index, src=0)
            teacher_steps, student_steps = rollout_step_pairs[pair_index.item()]
        else:
            import random
            teacher_steps, student_steps = random.choice(rollout_step_pairs)
        teacher_steps = max(1, int(teacher_steps))
        student_steps = max(1, int(student_steps))

        empty_emb = self.empty_emb.expand(
            input_dict['latent_dict']['text_emb'].shape[0], -1, -1
        )
        student_x_r, student_v_r = self._student_euler_integrate(
            noisy_latents=video_noisy_t,
            timesteps=video_t,
            target_r=video_r,
            base_input_dict=student_input,
            empty_emb=empty_emb,
            cfg_scale=cfg_scale,
            ref_shape=ref_shape,
            B=B,
            num_frames=num_frames,
            K_steps=student_steps,
            action_target_r=action_r,
        )
        teacher_x_r, teacher_v_r = self._teacher_integrate_to_r(
            noisy_latents=video_noisy_t,
            timesteps=video_t,
            target_r=video_r,
            input_dict=input_dict,
            empty_emb=empty_emb,
            cfg_scale=cfg_scale,
            ref_shape=ref_shape,
            B=B,
            num_frames=num_frames,
            num_steps=teacher_steps,
        )
        sigma_r = (video_r / self.config.num_train_timesteps)[:, None, :, None, None]
        endpoint_loss = denoised_endpoint_mse(
            student_x_r,
            student_v_r,
            teacher_x_r,
            teacher_v_r,
            sigma_r,
        )
        diagnostics = {
            'teacher_steps': teacher_steps,
            'student_steps': student_steps,
            't_mean': video_t.float().mean().detach(),
            'r_mean': video_r.float().mean().detach(),
        }
        return endpoint_loss, diagnostics

    def _danceopd_aux_transition_step(self, batch, batch_idx, kto_paopd=False):
        """Run one DanceOPD-style local video field-matching update.

        The rollout starts at each scheduler's terminal noise distribution and
        evolves video and action noisy states together without gradient.  One
        semantic-side state is selected per sample, detached, and then queried
        by the frozen teacher and trainable student at the same state/time.
        This is intentionally separate from the endpoint OPD objective.
        """
        if kto_paopd:
            raise ValueError("DanceOPD query mode cannot be combined with KTO-PAOPD")

        batch = self.convert_input_format(batch)
        B = batch['latents'].shape[0]
        ref_shape = batch['latents'].shape
        video_frames = ref_shape[2]
        action_downsample = int(getattr(self.config, 'action_downsample_factor', 4))
        action_clean = batch['actions'][:, :, ::action_downsample]
        action_frames = action_clean.shape[2]
        if action_frames <= 0:
            raise ValueError("DanceOPD requires at least one action frame after downsampling")

        input_dict = self._prepare_base_dict(batch)
        empty_emb = self.empty_emb.expand(B, -1, -1)
        cfg_scale = self.config.cfg_min + torch.rand(1).item() * (
            self.config.cfg_max - self.config.cfg_min
        )
        rollout_step_choices = tuple(
            int(value)
            for value in getattr(
                self.config,
                "opd_danceopd_rollout_step_choices",
                (getattr(self.config, "opd_danceopd_rollout_steps", 4),),
            )
        )
        if not rollout_step_choices or any(value <= 0 for value in rollout_step_choices):
            raise ValueError("opd_danceopd_rollout_step_choices must be positive")
        if len(rollout_step_choices) == 1:
            rollout_steps = rollout_step_choices[0]
        else:
            choice_index = torch.randint(len(rollout_step_choices), (1,), device=self.device)
            if dist.is_initialized():
                dist.broadcast(choice_index, src=0)
            rollout_steps = rollout_step_choices[choice_index.item()]
        query_alpha = float(getattr(self.config, 'opd_danceopd_query_alpha', 5.0))
        query_beta = float(getattr(self.config, 'opd_danceopd_query_beta', 2.0))

        terminal_video_t = torch.full(
            (B, video_frames),
            float(self.config.num_train_timesteps),
            device=self.device,
            dtype=torch.float32,
        )
        terminal_action_t = torch.full(
            (B, action_frames),
            float(self.config.num_train_timesteps),
            device=self.device,
            dtype=torch.float32,
        )
        zero_video_t = torch.zeros_like(terminal_video_t)
        zero_action_t = torch.zeros_like(terminal_action_t)

        # At the FlowMatch terminal timestep sigma is exactly one, so these are
        # pure noise states while preserving any scheduler-specific convention.
        video_noise = torch.randn_like(batch['latents'])
        action_noise = torch.randn_like(action_clean)
        current_video = self.train_scheduler_latent.add_noise(
            batch['latents'], video_noise, terminal_video_t, t_dim=2
        )
        current_action = self.train_scheduler_action.add_noise(
            action_clean, action_noise, terminal_action_t, t_dim=2
        )
        (
            terminal_prior_validation,
            terminal_prior_diagnostics,
        ) = _validate_terminal_prior_pair(
            config=self.config,
            video_actual=current_video,
            video_expected=video_noise,
            video_source_dtype=batch["latents"].dtype,
            action_actual=current_action,
            action_expected=action_noise,
            action_source_dtype=action_clean.dtype,
            reference=current_video,
            error_context="DanceOPD terminal state is not pure scheduler noise",
        )
        danceopd_terminal_prior_max_error = terminal_prior_diagnostics['max_error']

        video_path = self._build_timestep_path(
            terminal_video_t, zero_video_t, rollout_steps
        )
        action_path = self._build_timestep_path(
            terminal_action_t, zero_action_t, rollout_steps
        )

        action_base = input_dict['action_dict']
        action_grid_id = _downsample_action_grid_id(
            action_base.get('grid_id'), action_base['latent'], action_downsample
        )
        action_mask = action_base.get('actions_mask')
        if action_mask is not None:
            action_mask = action_mask[:, :, ::action_downsample]

        use_nofsdp_rollout = getattr(self, '_student_nofsdp', None) is not None
        rollout_model = self._student_nofsdp if use_nofsdp_rollout else self.student
        rollout_video_base = input_dict['latent_dict']
        rollout_action_latent = action_base['latent'][:, :, ::action_downsample]
        rollout_action_cond_t = action_base['cond_timesteps'][:, ::action_downsample]
        rollout_action_text = action_base['text_emb']
        rollout_empty_emb = empty_emb
        if use_nofsdp_rollout:
            if not getattr(self, '_nofsdp_synced', False):
                self._sync_student_nofsdp()
                self._nofsdp_synced = True
            rollout_model.train()
            rollout_video_base = {
                key: _to_regular_tensor(value) if isinstance(value, torch.Tensor) else value
                for key, value in rollout_video_base.items()
            }
            rollout_action_latent = _to_regular_tensor(rollout_action_latent)
            rollout_action_cond_t = _to_regular_tensor(rollout_action_cond_t)
            rollout_action_text = _to_regular_tensor(rollout_action_text)
            action_grid_id = _to_regular_tensor(action_grid_id)
            action_mask = _to_regular_tensor(action_mask)
            rollout_empty_emb = _to_regular_tensor(empty_emb)
            current_video = _to_regular_tensor(current_video)
            current_action = _to_regular_tensor(current_action)
            video_path = _to_regular_tensor(video_path)
            action_path = _to_regular_tensor(action_path)

        rollout_context = {
            "video_base": rollout_video_base,
            "action_latent": rollout_action_latent,
            "action_cond_t": rollout_action_cond_t,
            "action_text": rollout_action_text,
            "action_grid": action_grid_id,
            "action_mask": action_mask,
            "chunk_size": input_dict["chunk_size"],
            "window_size": input_dict["window_size"],
            "student_model": rollout_model,
            "empty_emb": rollout_empty_emb,
            "cfg_scale": cfg_scale,
            "batch_size": B,
            "ref_shape": ref_shape,
        }

        video_states = []
        action_states = []
        video_timesteps = []
        action_timesteps = []
        with torch.no_grad():
            for step_index in range(rollout_steps):
                video_t = video_path[step_index]
                video_r = video_path[step_index + 1]
                action_t = action_path[step_index]
                action_r = action_path[step_index + 1]
                video_states.append(current_video.detach().clone())
                action_states.append(current_action.detach().clone())
                video_timesteps.append(video_t.detach().clone())
                action_timesteps.append(action_t.detach().clone())

                rollout_input = self._mechanism_joint_input(
                    current_video,
                    current_action,
                    video_t,
                    action_t,
                    rollout_context,
                )
                self._init_joint_mask(rollout_input)
                video_velocity, action_velocity_seq = self._student_joint_forward(
                    rollout_model,
                    rollout_input,
                    rollout_empty_emb,
                    video_r,
                    action_r,
                    cfg_scale=cfg_scale,
                    batch_size=B,
                    ref_shape=ref_shape,
                    require_action=True,
                )
                if action_velocity_seq is None:
                    raise RuntimeError("DanceOPD joint rollout requires a student action output")
                action_velocity = self._extract_action_v(action_velocity_seq, action_frames)
                video_sigma = self._timestep_to_sigma_5d(video_t)
                video_sigma_next = self._timestep_to_sigma_5d(video_r)
                action_sigma = action_t[:, None, :, None, None] / self.config.num_train_timesteps
                action_sigma_next = action_r[:, None, :, None, None] / self.config.num_train_timesteps
                current_video = current_video + video_velocity * (
                    video_sigma_next.to(video_velocity) - video_sigma.to(video_velocity)
                )
                current_action = current_action + action_velocity * (
                    action_sigma_next.to(action_velocity) - action_sigma.to(action_velocity)
                )

        query_indices = sample_low_noise_query_indices(
            n_states=rollout_steps,
            batch_size=B,
            alpha=query_alpha,
            beta=query_beta,
            device=current_video.device,
        )
        query_video = select_per_sample_trajectory_state(
            torch.stack(video_states, dim=0), query_indices
        ).detach()
        query_action = select_per_sample_trajectory_state(
            torch.stack(action_states, dim=0), query_indices
        ).detach()
        query_video_t = select_per_sample_trajectory_state(
            torch.stack(video_timesteps, dim=0), query_indices
        ).detach()
        query_action_t = select_per_sample_trajectory_state(
            torch.stack(action_timesteps, dim=0), query_indices
        ).detach()
        danceopd_query_index_mean = query_indices.float().mean().detach()
        danceopd_query_sigma_mean = (
            query_video_t.float() / self.config.num_train_timesteps
        ).mean().detach()
        diagnostic_interval = int(getattr(
            self.config, 'opd_danceopd_diagnostic_interval', 50
        ))
        if (
            getattr(self.config, 'rank', 0) == 0
            and int(getattr(self, 'step', 0)) % diagnostic_interval == 0
        ):
            logger.info(
                "[DanceOPD] rollout_steps=%d query_index=%.2f query_sigma=%.4f "
                "terminal_prior_max_error=%.3e",
                rollout_steps,
                danceopd_query_index_mean.item(),
                danceopd_query_sigma_mean.item(),
                danceopd_terminal_prior_max_error.detach().item(),
            )

        query_context = {
            "video_base": input_dict["latent_dict"],
            "action_latent": action_base["latent"][:, :, ::action_downsample],
            "action_cond_t": action_base["cond_timesteps"][:, ::action_downsample],
            "action_text": action_base["text_emb"],
            "action_grid": _downsample_action_grid_id(
                action_base.get('grid_id'), action_base['latent'], action_downsample
            ),
            "action_mask": (
                action_base['actions_mask'][:, :, ::action_downsample]
                if action_base.get('actions_mask') is not None else None
            ),
            "chunk_size": input_dict["chunk_size"],
            "window_size": input_dict["window_size"],
        }
        student_query_input = self._mechanism_joint_input(
            query_video,
            query_action,
            query_video_t,
            query_action_t,
            query_context,
        )
        self._init_joint_mask(student_query_input)
        student_video_velocity = self._student_joint_forward(
            self.student,
            student_query_input,
            empty_emb,
            query_video_t,
            query_action_t,
            cfg_scale=cfg_scale,
            batch_size=B,
            ref_shape=ref_shape,
            require_action=False,
        )
        with torch.no_grad():
            teacher_cond, teacher_uncond, _ = self._batched_cfg_forward(
                student_query_input, empty_emb
            )
            teacher_video_velocity = self._extract_video_v(
                teacher_uncond + cfg_scale * (teacher_cond - teacher_uncond),
                ref_shape,
                B,
            )

        velocity_loss = direct_velocity_mse(
            student_video_velocity, teacher_video_velocity
        )
        velocity_weight = float(getattr(
            self.config, 'opd_danceopd_velocity_weight', 1.0
        ))
        endpoint_weight = float(getattr(
            self.config, 'opd_danceopd_endpoint_weight', 0.0
        ))
        if (
            not math.isfinite(velocity_weight)
            or velocity_weight < 0
            or not math.isfinite(endpoint_weight)
            or endpoint_weight < 0
        ):
            raise ValueError(
                'DanceOPD endpoint and velocity weights must be finite and non-negative'
            )
        zero = torch.zeros((), device=self.device, dtype=velocity_loss.dtype)
        endpoint_loss = zero
        endpoint_diagnostics = {}
        if endpoint_weight > 0:
            endpoint_loss, endpoint_diagnostics = self._danceopd_independent_endpoint_loss(
                batch, cfg_scale=cfg_scale
            )
        raw_velocity_contrib = velocity_weight * velocity_loss
        raw_endpoint_contrib = endpoint_weight * endpoint_loss
        aux_weight = float(getattr(self.config, 'opd_aux_weight', 1.0))
        loss = (raw_velocity_contrib + raw_endpoint_contrib) * aux_weight
        velocity_contrib = raw_velocity_contrib * aux_weight
        endpoint_contrib = raw_endpoint_contrib * aux_weight
        contrib_denom = (
            velocity_contrib.detach().abs() + endpoint_contrib.detach().abs()
        ).clamp(min=1e-12)
        is_finite, nonfinite_origins = _synchronized_nonfinite_decision(
            loss=loss,
            local_flags={
                "opd_endpoint": not bool(
                    torch.isfinite(endpoint_loss.detach()).all().item()
                ),
                "opd_compositional": not bool(
                    torch.isfinite(velocity_loss.detach()).all().item()
                ),
            },
            device=self.device,
        )
        result = {
            'loss': loss.detach(),
            'opd_aux_loss': loss.detach(),
            'opd_same_state_velocity_loss': velocity_loss.detach(),
            'opd_same_state_velocity_contrib': velocity_contrib.detach(),
            'opd_same_state_velocity_ratio': (
                velocity_contrib.detach().abs() / contrib_denom
            ),
            'opd_video_transition_loss': zero,
            'opd_endpoint_aux_loss': endpoint_loss.detach(),
            'opd_local_fm_loss': zero,
            'opd_action_transition_loss': zero,
            'opd_action_local_fm_loss': zero,
            'opd_video_transition_contrib': zero,
            'opd_endpoint_aux_contrib': endpoint_contrib.detach(),
            'opd_local_fm_contrib': zero,
            'opd_action_transition_contrib': zero,
            'opd_action_local_fm_contrib': zero,
            'opd_video_transition_ratio': zero,
            'opd_endpoint_aux_ratio': (
                endpoint_contrib.detach().abs() / contrib_denom
            ),
            'opd_local_fm_ratio': zero,
            'opd_action_transition_ratio': zero,
            'opd_action_local_fm_ratio': zero,
            'danceopd_query_index_mean': danceopd_query_index_mean,
            'danceopd_query_sigma_mean': danceopd_query_sigma_mean,
            'danceopd_terminal_prior_max_error': danceopd_terminal_prior_max_error.detach(),
            'danceopd_terminal_prior_reference_scale': terminal_prior_diagnostics['reference_scale'],
            'danceopd_terminal_prior_atol': terminal_prior_diagnostics['atol'],
            'danceopd_terminal_prior_rtol': terminal_prior_diagnostics['rtol'],
            'danceopd_terminal_prior_threshold': terminal_prior_diagnostics['threshold'],
            'danceopd_terminal_prior_severity': terminal_prior_diagnostics['severity'],
            'danceopd_endpoint_teacher_steps': endpoint_diagnostics.get(
                'teacher_steps', 1
            ),
            'danceopd_endpoint_student_steps': endpoint_diagnostics.get(
                'student_steps', 0
            ),
            'danceopd_endpoint_t_mean': endpoint_diagnostics.get('t_mean', zero),
            'danceopd_endpoint_r_mean': endpoint_diagnostics.get('r_mean', zero),
            'rollout_steps': rollout_steps,
            'teacher_steps': endpoint_diagnostics.get('teacher_steps', 1),
            'should_sync': True,
            'skip_step': not is_finite,
            'nonfinite_origins': nonfinite_origins,
        }
        if result['skip_step']:
            if self.config.rank == 0:
                logger.warning(f"[step {self.step}] non-finite DanceOPD loss, skipping")
            result['loss'] = zero
            result['opd_aux_loss'] = zero
            return result

        loss.backward()
        return result

    def _opd_aux_transition_step(self, batch, batch_idx, kto_paopd=False):
        """
        Auxiliary teacher-transition loss on top of the regular FlowMap step.

        This keeps the regular AnyFlow step as the main objective and adds a
        teacher-transition auxiliary for both video and action.
        """
        teacher_target_mode = str(getattr(
            self.config, 'opd_teacher_target_mode', 'student_state')).lower()
        if (
            getattr(self.config, 'cosmos_latent_target', False)
            and teacher_target_mode == 'cosmos_latent_full'
        ):
            return self._cosmos_latent_full_opd_aux_transition_step(
                batch,
                batch_idx,
                kto_paopd=kto_paopd,
            )
        if str(getattr(self.config, 'opd_query_mode', 'legacy')).lower() == 'danceopd':
            return self._danceopd_aux_transition_step(
                batch,
                batch_idx,
                kto_paopd=kto_paopd,
            )
        if (
            getattr(self.config, 'cosmos_latent_target', False)
            and teacher_target_mode
            in ('cosmos_latent_student_state', 'cosmos_latent_velocity')
        ):
            return self._cosmos_latent_opd_aux_transition_step(
                batch,
                batch_idx,
                kto_paopd=kto_paopd,
            )

        profile_opd = bool(getattr(self.config, 'opd_profile', False))
        profile_times = {}
        profile_last = None

        def _profile_mark(name):
            nonlocal profile_last
            if not profile_opd:
                return
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            now = time.perf_counter()
            if profile_last is not None:
                profile_times[name] = now - profile_last
            profile_last = now

        _profile_mark('start')
        batch = self.convert_input_format(batch)
        _profile_mark('prepare_batch')

        B = batch['latents'].shape[0]
        ref_shape = batch['latents'].shape
        num_frames = ref_shape[2]
        actions_mask = batch.get('actions_mask')
        _ACTION_DS = getattr(self.config, 'action_downsample_factor', 4)

        input_dict = self._prepare_base_dict(batch)

        video_t, video_r, _ = self.sample_timestep_mixed(
            B, num_frames, dtype=torch.float32, device=self.device,
            pair_mode=getattr(self.config, 'opd_pair_mode', None),
        )
        video_r = self._apply_opd_low_noise_query_bias(video_t, video_r)
        video_r_sigma = video_r / self.config.num_train_timesteps

        action_t = action_r = None
        action_t_sigma = action_r_sigma = None
        if self.distill_action or self.action_aware:
            action_t, action_r, _ = self.sample_timestep_mixed(
                B, num_frames, dtype=torch.float32, device=self.device,
                scheduler=self.train_scheduler_action,
                pair_mode=getattr(self.config, 'opd_pair_mode', None),
            )
            action_r = self._apply_opd_low_noise_query_bias(action_t, action_r)
            action_t_sigma = action_t / self.config.num_train_timesteps
            action_r_sigma = action_r / self.config.num_train_timesteps

        video_noise = torch.randn_like(batch['latents'])
        video_noisy_latents = self.train_scheduler_latent.add_noise(
            batch['latents'], video_noise, video_t, t_dim=2
        )
        video_v_target = self.train_scheduler_latent.training_target(
            batch['latents'], video_noise, video_t
        )
        input_dict['latent_dict']['noisy_latents'] = video_noisy_latents
        input_dict['latent_dict']['timesteps'] = video_t
        input_dict['latent_dict']['targets'] = video_v_target

        if self.distill_action or self.action_aware:
            action_noise = torch.randn_like(batch['actions'])
            action_noisy_latents = self.train_scheduler_action.add_noise(
                batch['actions'], action_noise, action_t, t_dim=2
            )
            input_dict['action_dict']['noisy_latents'] = action_noisy_latents
            input_dict['action_dict']['timesteps'] = action_t
            input_dict['action_dict']['targets'] = self.train_scheduler_action.training_target(
                batch['actions'], action_noise, action_t
            )
            _ad = input_dict['action_dict']
            student_input_dict = {
                'latent_dict': input_dict['latent_dict'],
                'action_dict': {
                    'noisy_latents': _ad['noisy_latents'][:, :, ::_ACTION_DS],
                    'latent':        _ad['latent'][:, :, ::_ACTION_DS],
                    'timesteps':     _ad['timesteps'][:, ::_ACTION_DS],
                    'cond_timesteps':_ad['cond_timesteps'][:, ::_ACTION_DS],
                    'text_emb':      _ad['text_emb'],
                    'grid_id':       _downsample_action_grid_id(
                        _ad['grid_id'], _ad['latent'], _ACTION_DS) if _ad.get('grid_id') is not None else None,
                    'actions_mask':  _ad['actions_mask'][:, :, ::_ACTION_DS] if _ad.get('actions_mask') is not None else None,
                },
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
        else:
            student_input_dict = input_dict

        rollout_step_pairs = getattr(
            self.config,
            'opd_rollout_step_pairs',
            getattr(self.config, 'rollout_step_pairs', [[1, 1]]),
        )
        if dist.is_initialized():
            idx = torch.randint(0, len(rollout_step_pairs), (1,), device=self.device)
            dist.broadcast(idx, src=0)
            N_steps, K_steps = rollout_step_pairs[idx.item()]
        else:
            import random
            N_steps, K_steps = random.choice(rollout_step_pairs)

        cfg_scale = self.config.cfg_min + torch.rand(1).item() * (
            self.config.cfg_max - self.config.cfg_min)
        empty_emb = self.empty_emb.expand(
            input_dict['latent_dict']['text_emb'].shape[0], -1, -1)

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        if hasattr(self.student, 'set_requires_gradient_sync'):
            self.student.set_requires_gradient_sync(should_sync)

        student_x_r, student_v_at_r, student_last_x, student_last_t = self._student_euler_integrate(
            noisy_latents=video_noisy_latents,
            timesteps=video_t,
            target_r=video_r,
            base_input_dict=student_input_dict,
            empty_emb=empty_emb,
            cfg_scale=cfg_scale,
            ref_shape=ref_shape,
            B=B,
            num_frames=num_frames,
            K_steps=K_steps,
            action_target_r=action_r if (self.distill_action or self.action_aware) else None,
            return_last_step_start=True,
        )
        _profile_mark('video_student_rollout')

        video_transition_param = getattr(self.config, 'video_transition_param', 'x0')
        teacher_target_mode = str(getattr(
            self.config, 'opd_teacher_target_mode', 'student_state')).lower()
        if teacher_target_mode not in ('student_state', 'endpoint'):
            raise ValueError(
                f"Invalid opd_teacher_target_mode={teacher_target_mode!r}; "
                "expected student_state or endpoint."
            )

        opd_action_transition_loss = torch.tensor(0.0, device=self.device)
        opd_action_local_fm_loss = torch.tensor(0.0, device=self.device)
        kto_good_ratio = torch.tensor(0.0, device=self.device)
        kto_weight_mean = torch.tensor(0.0, device=self.device)
        kto_weight_min = torch.tensor(0.0, device=self.device)
        kto_weight_max = torch.tensor(0.0, device=self.device)
        kto_threshold_value = torch.tensor(0.0, device=self.device)
        use_opd_aux_action = getattr(self.config, 'opd_aux_action', True)
        action_opd_active = use_opd_aux_action and (self.distill_action or self.action_aware)
        target_action_v_from_fused = None
        fused_action_teacher_input = None
        fused_action_frames = None
        student_action_v = None

        if action_opd_active:
            action_noisy_ds = action_noisy_latents[:, :, ::_ACTION_DS]
            video_context_r = student_x_r.detach()
            from modules.model import FlexAttnFunc
            action_r_ds = action_r[:, ::_ACTION_DS]
            action_t_ds = action_t[:, ::_ACTION_DS]
            action_frames_ds = action_noisy_ds.shape[2]
            actions_gt_ds = batch['actions'][:, :, ::_ACTION_DS]
            if actions_mask is None:
                actions_mask_ds = torch.ones(
                    actions_gt_ds.shape[0], 1, actions_gt_ds.shape[2],
                    device=actions_gt_ds.device, dtype=actions_gt_ds.dtype)
            else:
                actions_mask_ds = actions_mask[:, :, ::_ACTION_DS]
            mask = actions_mask_ds.float()
            mask_5d = mask[:, :, :, None, None]
            action_channels = actions_gt_ds.shape[1]

            def _masked_action_mean(diff):
                per_sample_num = (diff ** 2 * mask_5d).sum(dim=[1, 2, 3, 4])
                per_sample_den = (
                    mask.sum(dim=[1, 2]).clamp(min=1.0)
                    * action_channels
                    * diff.shape[3]
                    * diff.shape[4]
                )
                return (per_sample_num / per_sample_den).mean()

            def _build_action_rollout_input(action_x, action_timesteps):
                action_input = {
                    'latent_dict': {
                        **input_dict['latent_dict'],
                        'noisy_latents': video_context_r,
                        'timesteps': video_r,
                    },
                    'action_dict': {
                        'noisy_latents': action_x,
                        'latent': actions_gt_ds,
                        'timesteps': action_timesteps,
                        'cond_timesteps': input_dict['action_dict']['cond_timesteps'][:, ::_ACTION_DS],
                        'text_emb': input_dict['action_dict']['text_emb'],
                    },
                    'chunk_size': input_dict['chunk_size'],
                    'window_size': input_dict['window_size'],
                }
                if 'grid_id' in input_dict['action_dict'] and input_dict['action_dict']['grid_id'] is not None:
                    action_input['action_dict']['grid_id'] = _downsample_action_grid_id(
                        input_dict['action_dict']['grid_id'], batch['actions'], _ACTION_DS)
                if 'actions_mask' in input_dict['action_dict'] and input_dict['action_dict']['actions_mask'] is not None:
                    action_input['action_dict']['actions_mask'] = actions_mask_ds
                return action_input

            def _init_action_mask(action_input):
                _ld = action_input['latent_dict']
                _ad = action_input['action_dict']
                _total_length = (
                    _ld['noisy_latents'].flatten(0, 1).shape[0] * 2 +
                    _ad['noisy_latents'].flatten(0, 1).shape[0] * 2
                )
                _padded_length = (128 - _total_length % 128) % 128
                FlexAttnFunc.init_mask(
                    _ld['noisy_latents'].shape,
                    _ad['noisy_latents'].shape,
                    _padded_length,
                    action_input['chunk_size'],
                    window_size=action_input['window_size'],
                    patch_size=self.patch_size,
                    device=self.device,
                )

            def _student_action_forward(model, action_x, action_timesteps, action_r_timestep):
                action_input = _build_action_rollout_input(action_x, action_timesteps)
                _init_action_mask(action_input)
                saved_blocks = None
                if model is self.student and getattr(self, '_student_blocks_compiled', False):
                    saved_blocks = list(self.student.blocks)
                    for bi, block in enumerate(saved_blocks):
                        if hasattr(block, '_orig_mod'):
                            self.student.blocks[bi] = block._orig_mod
                try:
                    _, action_v_seq = model(
                        action_input, train_mode=True,
                        r_timestep=video_r,
                        action_r_timestep=action_r_timestep,
                    )
                finally:
                    if saved_blocks is not None:
                        for bi, block in enumerate(saved_blocks):
                            self.student.blocks[bi] = block
                return self._extract_action_v(action_v_seq, action_frames_ds)

            def _teacher_action_forward(action_x, action_timesteps, action_r_timestep):
                teacher_input = _build_action_rollout_input(action_x, action_timesteps)
                teacher_input['action_dict']['timesteps'] = action_timesteps
                with torch.no_grad():
                    _, action_v_cond = self._teacher_forward_preserve_mask(teacher_input)
                    return self._extract_action_v(action_v_cond, action_frames_ds)

            if self.distill_action:
                action_grad_mode = str(getattr(
                    self.config, 'opd_action_rollout_grad_mode',
                    getattr(self.config, 'opd_rollout_grad_mode', 'endpoint'))).lower()
                action_grad_steps = int(getattr(
                    self.config,
                    'opd_action_rollout_grad_steps',
                    getattr(self.config, 'opd_rollout_grad_steps', 1),
                ))
                if action_grad_mode not in SUPPORTED_ROLLOUT_GRAD_MODES:
                    raise ValueError(
                        f"Invalid opd_action_rollout_grad_mode={action_grad_mode!r}; "
                        "expected endpoint, last_step, suffix, or full."
                    )

                action_num_steps = max(1, K_steps)
                action_student_path = self._build_timestep_path(
                    action_t_ds.float(), action_r_ds.float(), action_num_steps)
                student_action_x = action_noisy_ds
                for i in range(action_num_steps):
                    t_i = action_student_path[i]
                    r_i = action_student_path[i + 1]
                    keep_step_grad = rollout_step_requires_grad(
                        mode=action_grad_mode,
                        step_index=i,
                        num_steps=action_num_steps,
                        suffix_steps=action_grad_steps,
                    )
                    action_rollout_model = self.student
                    if not keep_step_grad and getattr(self, '_student_nofsdp', None) is not None:
                        if not getattr(self, '_nofsdp_synced', False):
                            self._sync_student_nofsdp()
                            self._nofsdp_synced = True
                        action_rollout_model = self._student_nofsdp
                    grad_context = torch.enable_grad() if keep_step_grad else torch.no_grad()
                    with grad_context:
                        action_v_i = _student_action_forward(action_rollout_model, student_action_x, t_i, r_i)
                        sigma_i = t_i[:, None, :, None, None] / self.config.num_train_timesteps
                        sigma_next = r_i[:, None, :, None, None] / self.config.num_train_timesteps
                        student_action_x = student_action_x + action_v_i * (
                            sigma_next.to(action_v_i) - sigma_i.to(action_v_i))
                student_action_x_r = (
                    student_action_x if action_grad_mode != 'endpoint'
                    else student_action_x.detach()
                )

                action_transition_param = str(getattr(
                    self.config, 'action_transition_param',
                    getattr(self.config, 'video_transition_param', 'velocity'))).lower()
                if action_transition_param not in ('velocity', 'x0'):
                    raise ValueError(
                        f"Invalid action_transition_param={action_transition_param!r}; "
                        "expected velocity or x0."
                    )

                if teacher_target_mode == 'student_state':
                    teacher_action_x_r = student_action_x_r.detach()
                    if bool(getattr(self.config, 'opd_fuse_action_teacher', True)):
                        fused_action_teacher_input = _build_action_rollout_input(
                            teacher_action_x_r, action_r_ds)
                        fused_action_frames = action_frames_ds
                    else:
                        with torch.no_grad():
                            target_action_v_from_fused = _teacher_action_forward(
                                teacher_action_x_r, action_r_ds, action_r_ds)
                else:
                    with torch.no_grad():
                        action_teacher_path = self._build_timestep_path(
                            action_t_ds.float(), action_r_ds.float(), max(1, N_steps))
                        teacher_action_x = action_noisy_ds
                        for i in range(max(1, N_steps)):
                            t_i = action_teacher_path[i]
                            r_i = action_teacher_path[i + 1]
                            action_v_i = _teacher_action_forward(teacher_action_x, t_i, r_i)
                            sigma_i = t_i[:, None, :, None, None] / self.config.num_train_timesteps
                            sigma_next = r_i[:, None, :, None, None] / self.config.num_train_timesteps
                            teacher_action_x = teacher_action_x + action_v_i * (
                                sigma_next.to(action_v_i) - sigma_i.to(action_v_i))
                        teacher_action_x_r = teacher_action_x.detach()
                        target_action_v = _teacher_action_forward(
                            teacher_action_x_r, action_r_ds, action_r_ds)
                    if action_transition_param == 'velocity':
                        action_transition_param = 'x0'
                        if self.config.rank == 0 and not getattr(
                            self, '_warned_opd_action_endpoint_uses_x0', False
                        ):
                            logger.warning(
                                "Using x0 OPD action transition loss for K-step student "
                                "vs N-step teacher endpoint distillation; velocity MSE is "
                                "only valid for same-state teacher labels."
                            )
                            self._warned_opd_action_endpoint_uses_x0 = True
                    student_action_v = _student_action_forward(
                        self.student, student_action_x_r.detach(), action_r_ds, action_r_ds)
                    sigma_r_a_5d = action_r_sigma[:, None, ::_ACTION_DS, None, None].to(
                        student_action_v)
                    student_action_pred = student_action_x_r - sigma_r_a_5d * student_action_v
                    target_action_pred = teacher_action_x_r - sigma_r_a_5d.to(
                        target_action_v) * target_action_v
                    action_diff = (
                        student_action_pred.float()
                        - target_action_pred.detach().float()
                    )
                    opd_action_transition_loss = _masked_action_mean(action_diff)

        effective_teacher_steps = max(1, N_steps)
        if teacher_target_mode == 'student_state':
            if fused_action_teacher_input is not None:
                teacher_x_r, teacher_v_at_r, target_action_v_from_fused = (
                    self._teacher_forward_at_student_state(
                        student_x_r=student_x_r,
                        target_r=video_r,
                        input_dict=input_dict,
                        empty_emb=empty_emb,
                        cfg_scale=cfg_scale,
                        ref_shape=ref_shape,
                        B=B,
                        action_input_dict=fused_action_teacher_input['action_dict'],
                        action_frames=fused_action_frames,
                    )
                )
            else:
                teacher_x_r, teacher_v_at_r = self._teacher_forward_at_student_state(
                    student_x_r=student_x_r,
                    target_r=video_r,
                    input_dict=input_dict,
                    empty_emb=empty_emb,
                    cfg_scale=cfg_scale,
                    ref_shape=ref_shape,
                    B=B,
                )
            effective_teacher_steps = 1
            _profile_mark('video_teacher_query')
        else:
            teacher_x_r, teacher_v_at_r = self._teacher_integrate_to_r(
                noisy_latents=video_noisy_latents,
                timesteps=video_t,
                target_r=video_r,
                input_dict=input_dict,
                empty_emb=empty_emb,
                cfg_scale=cfg_scale,
                ref_shape=ref_shape,
                B=B,
                num_frames=num_frames,
                num_steps=effective_teacher_steps,
            )
            _profile_mark('video_teacher_rollout')
        if video_transition_param == 'velocity' and teacher_target_mode == 'endpoint':
            video_transition_param = 'x0'
            if self.config.rank == 0 and not getattr(
                self, '_warned_opd_video_endpoint_uses_x0', False
            ):
                logger.warning(
                    "Using x0 OPD video transition loss for K-step student vs "
                    "N-step teacher endpoint distillation; velocity MSE is only "
                    "valid for same-state teacher labels."
                )
                self._warned_opd_video_endpoint_uses_x0 = True

        transition_loss_type = getattr(self.config, 'transition_loss_type', 'huber')
        transition_huber_c = getattr(self.config, 'transition_huber_c', 1e-3)
        if video_transition_param == 'consistency':
            student_video_pred = self._consistency_function(
                student_v_at_r, student_x_r, video_r_sigma
            )
            with torch.no_grad():
                teacher_video_pred = self._consistency_function(
                    teacher_v_at_r, teacher_x_r, video_r_sigma
                )
            video_diff = student_video_pred.float() - teacher_video_pred.detach().float()
        elif video_transition_param == 'x0':
            sigma_r = video_r_sigma[:, None, :, None, None].to(student_v_at_r)
            student_video_pred = student_x_r - sigma_r * student_v_at_r
            teacher_video_pred = teacher_x_r - sigma_r * teacher_v_at_r
            video_diff = student_video_pred.float() - teacher_video_pred.detach().float()
        else:
            video_diff = student_v_at_r.float() - teacher_v_at_r.detach().float()

        if transition_loss_type == "huber":
            abs_diff = video_diff.abs()
            per_sample_loss = torch.where(
                abs_diff < transition_huber_c,
                0.5 * video_diff ** 2,
                transition_huber_c * (abs_diff - 0.5 * transition_huber_c),
            ).mean(dim=[1, 2, 3, 4])
        else:
            per_sample_loss = (video_diff ** 2).mean(dim=[1, 2, 3, 4])

        weight_type = getattr(self.config, 'weight_type', 'uniform')
        weight = self._get_timestep_weight(video_t.mean(dim=-1), weight_type).to(self.device)
        if kto_paopd and bool(getattr(self.config, 'kto_adaptive', True)):
            kto_eps = float(getattr(self.config, 'kto_eps', 1e-5))
            reweight_mode = str(getattr(self.config, 'kto_reweight_mode', 'hard')).lower()
            kto_video_loss_scale = 1.0

            if transition_loss_type == "huber":
                token_loss = torch.where(
                    abs_diff < transition_huber_c,
                    0.5 * video_diff ** 2,
                    transition_huber_c * (abs_diff - 0.5 * transition_huber_c),
                ).mean(dim=1)
            else:
                token_loss = (video_diff ** 2).mean(dim=1)

            with torch.no_grad():
                if video_transition_param == 'velocity':
                    teacher_ref = teacher_v_at_r.detach().float()
                else:
                    teacher_ref = teacher_video_pred.detach().float()
                token_error = video_diff.detach().float().abs().mean(dim=1)
                token_teacher_ref = teacher_ref.abs().mean(dim=1)
                if reweight_mode in ('normalized_focal', 'norm_focal'):
                    fixed_threshold = getattr(self.config, 'kto_threshold', None)
                    threshold_tensor = None
                    if fixed_threshold is not None:
                        threshold_tensor = torch.as_tensor(
                            fixed_threshold,
                            device=token_error.device,
                            dtype=token_error.dtype,
                        )
                    elif bool(getattr(self.config, 'kto_use_ema_threshold', True)):
                        error_ratio = token_error / (token_teacher_ref + kto_eps)
                        q = float(getattr(self.config, 'kto_threshold_quantile', 0.70))
                        current_threshold = torch.quantile(
                            error_ratio.detach().float().flatten(), q
                        )
                        ema_decay = float(getattr(self.config, 'kto_threshold_ema_decay', 0.90))
                        ema_decay = min(max(ema_decay, 0.0), 0.9999)
                        previous_threshold = getattr(self, '_kto_error_threshold_ema', None)
                        if previous_threshold is None:
                            threshold_tensor = current_threshold.detach()
                        else:
                            threshold_tensor = (
                                previous_threshold.to(current_threshold.device) * ema_decay
                                + current_threshold.detach() * (1.0 - ema_decay)
                            )
                        self._kto_error_threshold_ema = threshold_tensor.detach()

                    alpha = float(getattr(self.config, 'kto_alpha', 1.0))
                    warmup_steps = int(getattr(self.config, 'kto_warmup_steps', 0))
                    ramp_steps = int(getattr(self.config, 'kto_ramp_steps', 0))
                    current_step = int(getattr(self, 'step', 0))
                    if current_step < warmup_steps:
                        effective_alpha = 0.0
                    elif ramp_steps > 0:
                        effective_alpha = alpha * min(
                            1.0,
                            float(current_step - warmup_steps + 1) / float(ramp_steps),
                        )
                    else:
                        effective_alpha = alpha
                    decay_hold_steps = int(getattr(
                        self.config, 'kto_alpha_decay_hold_steps', -1))
                    if decay_hold_steps >= 0:
                        decay_scale = piecewise_linear_scale(
                            current_step,
                            start=1.0,
                            end=0.0,
                            hold_steps=decay_hold_steps,
                            ramp_steps=int(getattr(
                                self.config, 'kto_alpha_decay_ramp_steps', 0)),
                        )
                        effective_alpha = effective_alpha * decay_scale

                    focal = compute_normalized_focal_weights(
                        token_error,
                        token_teacher_ref,
                        threshold=threshold_tensor,
                        threshold_quantile=float(getattr(
                            self.config, 'kto_threshold_quantile', 0.70)),
                        eps=kto_eps,
                        alpha=effective_alpha,
                        temperature=float(getattr(self.config, 'kto_temperature', 0.10)),
                        min_weight=float(getattr(self.config, 'kto_min_weight', 0.5)),
                        max_weight=float(getattr(self.config, 'kto_max_weight', 1.8)),
                    )
                    token_weights = focal.weights
                    kto_good_ratio = focal.hard_ratio
                    kto_weight_mean = token_weights.float().mean()
                    kto_weight_min = token_weights.float().min()
                    kto_weight_max = token_weights.float().max()
                    kto_threshold_value = focal.threshold.float().mean()
                    kto_video_loss_scale = piecewise_linear_scale(
                        current_step,
                        start=float(getattr(self.config, 'kto_video_scale_start', 0.65)),
                        end=float(getattr(self.config, 'kto_video_scale_end', 1.0)),
                        hold_steps=int(getattr(self.config, 'kto_video_scale_hold_steps', 25)),
                        ramp_steps=int(getattr(self.config, 'kto_video_scale_ramp_steps', 25)),
                    )
                else:
                    good_weight_value = float(getattr(self.config, 'kto_good_weight', 0.3))
                    bad_weight_value = float(getattr(self.config, 'kto_bad_weight', 1.0))
                    if not math.isfinite(good_weight_value) or not math.isfinite(bad_weight_value):
                        raise ValueError("kto_good_weight and kto_bad_weight must be finite")
                    if good_weight_value < 0.0 or bad_weight_value < 0.0:
                        raise ValueError("kto_good_weight and kto_bad_weight must be non-negative")
                    token_similarity = (
                        1.0 - token_error / (token_teacher_ref + kto_eps)
                    ).clamp(0.0, 1.0)
                    threshold = getattr(self.config, 'kto_threshold', None)
                    if threshold is None:
                        threshold_tensor = token_similarity.median()
                    else:
                        threshold_tensor = torch.as_tensor(
                            threshold,
                            device=token_similarity.device,
                            dtype=token_similarity.dtype,
                        )
                    threshold_tensor = threshold_tensor.clamp(1e-6, 1.0)
                    is_good = token_similarity > threshold_tensor
                    token_weights = torch.where(
                        is_good,
                        torch.full_like(token_similarity, good_weight_value),
                        torch.full_like(token_similarity, bad_weight_value),
                    )
                    kto_good_ratio = is_good.float().mean()
                    kto_weight_mean = token_weights.mean()
                    kto_weight_min = token_weights.min()
                    kto_weight_max = token_weights.max()
                    kto_threshold_value = threshold_tensor.float().mean()

            per_sample_loss = (token_loss * token_weights.detach()).flatten(1).mean(dim=1)
            video_transition_loss = (per_sample_loss * weight).mean() * kto_video_loss_scale
        else:
            video_transition_loss = (per_sample_loss * weight).mean()

        opd_loss_composition = str(getattr(
            self.config, 'opd_loss_composition', 'legacy'
        )).lower()
        opd_endpoint_aux_loss = torch.tensor(0.0, device=self.device)
        opd_endpoint_aux_weight = float(getattr(self.config, 'opd_endpoint_aux_weight', 0.0))
        if (
            opd_endpoint_aux_weight > 0
            and opd_loss_composition != 'explicit_hybrid'
        ):
            sigma_r = video_r_sigma[:, None, :, None, None].to(student_v_at_r)
            student_endpoint_pred = student_x_r - sigma_r * student_v_at_r
            teacher_endpoint_pred = teacher_x_r - sigma_r.to(teacher_v_at_r) * teacher_v_at_r
            endpoint_diff = (
                student_endpoint_pred.float() -
                teacher_endpoint_pred.detach().float()
            )
            if transition_loss_type == "huber":
                abs_endpoint_diff = endpoint_diff.abs()
                endpoint_per_sample = torch.where(
                    abs_endpoint_diff < transition_huber_c,
                    0.5 * endpoint_diff ** 2,
                    transition_huber_c * (abs_endpoint_diff - 0.5 * transition_huber_c),
                ).mean(dim=[1, 2, 3, 4])
            else:
                endpoint_per_sample = (endpoint_diff ** 2).mean(dim=[1, 2, 3, 4])
            opd_endpoint_aux_loss = (endpoint_per_sample * weight).mean()

        opd_same_state_velocity_loss = torch.tensor(0.0, device=self.device)
        opd_same_state_velocity_weight = float(getattr(
            self.config, 'opd_same_state_velocity_weight', 0.0))
        if opd_same_state_velocity_weight > 0 and teacher_target_mode == 'endpoint':
            _, teacher_v_same_state = self._teacher_forward_at_student_state(
                student_x_r=student_x_r,
                target_r=video_r,
                input_dict=input_dict,
                empty_emb=empty_emb,
                cfg_scale=cfg_scale,
                ref_shape=ref_shape,
                B=B,
            )
            opd_same_state_velocity_loss = _same_state_velocity_loss(
                student_v_at_r,
                teacher_v_same_state,
                weight,
                transition_loss_type=transition_loss_type,
                transition_huber_c=transition_huber_c,
            )
            _profile_mark('video_teacher_same_state_query')

        local_fm_loss = torch.tensor(0.0, device=self.device)
        local_fm_weight = getattr(self.config, 'local_fm_weight', 0.05)
        if local_fm_weight > 0:
            video_v_target_at_r = self.train_scheduler_latent.training_target(
                batch['latents'], video_noise, video_r
            )
            local_fm_loss = (
                student_v_at_r.float() - video_v_target_at_r.detach().float()
            ).pow(2).mean()

        if action_opd_active:
            if self.distill_action and teacher_target_mode == 'student_state':
                teacher_action_x_r = student_action_x_r.detach()
                target_action_v = target_action_v_from_fused
                if target_action_v is None:
                    with torch.no_grad():
                        target_action_v = _teacher_action_forward(
                            teacher_action_x_r, action_r_ds, action_r_ds)
                if action_transition_param == 'x0':
                    sigma_r_a_5d = action_r_sigma[:, None, ::_ACTION_DS, None, None]
                    sigma_r_a_5d = sigma_r_a_5d.to(student_action_x_r)
                    student_action_v = _student_action_forward(
                        self.student, student_action_x_r.detach(), action_r_ds, action_r_ds)
                    student_action_pred = student_action_x_r - sigma_r_a_5d.to(student_action_v) * student_action_v
                    target_action_pred = teacher_action_x_r - sigma_r_a_5d.to(target_action_v) * target_action_v
                    action_diff = (
                        student_action_pred.float()
                        - target_action_pred.detach().float()
                    )
                else:
                    student_action_v = _student_action_forward(
                        self.student, student_action_x_r.detach(), action_r_ds, action_r_ds)
                    action_diff = (
                        student_action_v.float()
                        - target_action_v.detach().float()
                    )
                opd_action_transition_loss = _masked_action_mean(action_diff)

            if self.action_aware:
                action_targets = self.train_scheduler_action.training_target(
                    batch['actions'], action_noise, action_r)[:, :, ::_ACTION_DS]
                if not self.distill_action:
                    student_action_v = _student_action_forward(
                        self.student, action_noisy_ds, action_t_ds, action_r_ds)
                action_local_diff = (
                    student_action_v.float() - action_targets.float().detach()
                )
                opd_action_local_fm_loss = _masked_action_mean(action_local_diff)
            _profile_mark('action_opd_block')

        video_transition_weight = getattr(self.config, 'video_transition_weight', 1.0)
        action_transition_block_weight = float(getattr(
            self.config, 'action_transition_block_weight',
            getattr(self.config, 'action_block_weight', 1.0),
        ))
        action_local_fm_block_weight = float(getattr(
            self.config, 'action_local_fm_block_weight', 1.0,
        ))
        action_transition_weight = getattr(self.config, 'action_loss_weight', 1.0)
        action_local_fm_weight = getattr(self.config, 'action_aware_weight', 0.0)

        if opd_loss_composition == 'explicit_hybrid':
            explicit_loss = compose_explicit_hybrid_opd(
                endpoint_video_loss=video_transition_loss,
                velocity_video_loss=opd_same_state_velocity_loss,
                endpoint_action_loss=opd_action_transition_loss,
                beta_end_video=video_transition_weight,
                beta_vel_video=opd_same_state_velocity_weight,
                beta_end_action=(
                    action_transition_block_weight * action_transition_weight
                ),
            )
            zero_contrib = torch.zeros_like(explicit_loss.loss)
            opd_video_transition_contrib = explicit_loss.contributions[
                'endpoint_video'
            ]
            opd_same_state_velocity_contrib = explicit_loss.contributions[
                'velocity_video'
            ]
            opd_action_transition_contrib = explicit_loss.contributions[
                'endpoint_action'
            ]
            opd_endpoint_aux_contrib = zero_contrib
            opd_local_fm_contrib = zero_contrib
            opd_action_local_fm_contrib = zero_contrib
            scaled_transition_group = (
                opd_video_transition_contrib + opd_action_transition_contrib
            )
            scaled_anchor_group = opd_same_state_velocity_contrib
            transition_scale = torch.ones_like(explicit_loss.loss)
            anchor_scale = torch.ones_like(explicit_loss.loss)
            raw_aux_loss = explicit_loss.loss
            contrib_denom = sum(
                value.detach().abs()
                for value in explicit_loss.contributions.values()
            ).clamp(min=1e-12)
            opd_transition_group_ratio = (
                scaled_transition_group.detach().abs() / contrib_denom
            )
            opd_anchor_group_ratio = (
                scaled_anchor_group.detach().abs() / contrib_denom
            )
        else:
            raw_opd_video_transition_contrib = (
                video_transition_weight * video_transition_loss
            )
            raw_opd_endpoint_aux_contrib = (
                opd_endpoint_aux_weight * opd_endpoint_aux_loss
            )
            raw_opd_same_state_velocity_contrib = (
                opd_same_state_velocity_weight * opd_same_state_velocity_loss
            )
            raw_opd_local_fm_contrib = local_fm_weight * local_fm_loss
            raw_opd_action_transition_contrib = (
                action_transition_block_weight
                * action_transition_weight
                * opd_action_transition_loss
            )
            raw_opd_action_local_fm_contrib = (
                action_local_fm_block_weight
                * action_local_fm_weight
                * opd_action_local_fm_loss
            )

            raw_transition_group = (
                raw_opd_video_transition_contrib
                + raw_opd_action_transition_contrib
            )
            raw_anchor_group = (
                raw_opd_endpoint_aux_contrib
                + raw_opd_same_state_velocity_contrib
                + raw_opd_local_fm_contrib
                + raw_opd_action_local_fm_contrib
            )
            transition_scale = torch.as_tensor(
                float(getattr(self.config, 'opd_transition_group_weight', 1.0)),
                device=self.device,
                dtype=raw_transition_group.dtype,
            )
            scaled_transition_group = raw_transition_group * transition_scale
            anchor_cap_ratio = float(getattr(
                self.config, 'opd_anchor_cap_ratio', -1.0
            ))
            anchor_scale = torch.ones(
                (), device=self.device, dtype=raw_anchor_group.dtype
            )
            if anchor_cap_ratio >= 0:
                anchor_cap = (
                    scaled_transition_group.detach().abs() * anchor_cap_ratio
                )
                raw_anchor_abs = raw_anchor_group.detach().abs().clamp(min=1e-12)
                anchor_scale = torch.minimum(
                    anchor_scale, anchor_cap / raw_anchor_abs
                )
            scaled_anchor_group = raw_anchor_group * anchor_scale

            opd_video_transition_contrib = (
                raw_opd_video_transition_contrib * transition_scale
            )
            opd_action_transition_contrib = (
                raw_opd_action_transition_contrib * transition_scale
            )
            opd_endpoint_aux_contrib = (
                raw_opd_endpoint_aux_contrib * anchor_scale
            )
            opd_same_state_velocity_contrib = (
                raw_opd_same_state_velocity_contrib * anchor_scale
            )
            opd_local_fm_contrib = raw_opd_local_fm_contrib * anchor_scale
            opd_action_local_fm_contrib = (
                raw_opd_action_local_fm_contrib * anchor_scale
            )
            raw_aux_loss = scaled_transition_group + scaled_anchor_group
            contrib_denom = (
                scaled_transition_group.detach().abs()
                + scaled_anchor_group.detach().abs()
            ).clamp(min=1e-12)
            opd_transition_group_ratio = (
                scaled_transition_group.detach().abs() / contrib_denom
            )
            opd_anchor_group_ratio = (
                scaled_anchor_group.detach().abs() / contrib_denom
            )
        opd_aux_weight = float(getattr(self.config, 'opd_aux_weight', 0.1))
        weighted_aux_loss = raw_aux_loss * opd_aux_weight
        # OPD aux is scheduled only on the final accumulation microbatch
        # (should_sync=True), so do not divide by gradient_accumulation_steps
        # here. Dividing made OPD weaker by ACCUM after de-duplicating it.
        loss = weighted_aux_loss

        loss_clip_value = getattr(self.config, 'opd_aux_loss_clip_value', None)
        if loss_clip_value is not None and getattr(self.config, 'loss_clip_enabled', True):
            loss_clip_value = float(loss_clip_value)
            if loss_clip_value >= 0:
                # Preserve the gradient direction when clipping large OPD losses.
                # torch.clamp would zero the gradient above the cap.
                scale = (loss_clip_value / loss.detach().clamp(min=1e-12)).clamp(max=1.0)
                loss = loss * scale

        is_finite, nonfinite_origins = _synchronized_nonfinite_decision(
            loss=loss,
            local_flags={
                "opd_endpoint": not bool(
                    torch.isfinite(opd_endpoint_aux_loss.detach()).all().item()
                ),
                "opd_compositional": not bool(
                    torch.isfinite(video_transition_loss.detach()).all().item()
                )
                or not bool(
                    torch.isfinite(opd_same_state_velocity_loss.detach()).all().item()
                )
                or not bool(torch.isfinite(local_fm_loss.detach()).all().item()),
                "opd_action": not bool(
                    torch.isfinite(opd_action_transition_loss.detach()).all().item()
                )
                or not bool(
                    torch.isfinite(opd_action_local_fm_loss.detach()).all().item()
                ),
            },
            device=self.device,
        )
        if not is_finite:
            if self.config.rank == 0:
                logger.warning(f"[step {self.step}] NaN/Inf OPD aux loss, skipping")
            zero_loss = torch.zeros((), device=self.device)
            result = {
                'loss': zero_loss,
                'opd_aux_loss': weighted_aux_loss.detach(),
                'opd_video_transition_loss': video_transition_loss.detach(),
                'opd_endpoint_aux_loss': opd_endpoint_aux_loss.detach(),
                'opd_same_state_velocity_loss': opd_same_state_velocity_loss.detach(),
                'opd_local_fm_loss': local_fm_loss.detach(),
                'opd_action_transition_loss': opd_action_transition_loss.detach(),
                'opd_action_local_fm_loss': opd_action_local_fm_loss.detach(),
                'opd_video_transition_contrib': opd_video_transition_contrib.detach(),
                'opd_endpoint_aux_contrib': opd_endpoint_aux_contrib.detach(),
                'opd_same_state_velocity_contrib': opd_same_state_velocity_contrib.detach(),
                'opd_local_fm_contrib': opd_local_fm_contrib.detach(),
                'opd_action_transition_contrib': opd_action_transition_contrib.detach(),
                'opd_action_local_fm_contrib': opd_action_local_fm_contrib.detach(),
                'opd_transition_group_scaled': scaled_transition_group.detach(),
                'opd_anchor_group_scaled': scaled_anchor_group.detach(),
                'opd_transition_scale': transition_scale.detach(),
                'opd_anchor_scale': anchor_scale.detach(),
                'opd_transition_group_ratio': opd_transition_group_ratio.detach(),
                'opd_anchor_group_ratio': opd_anchor_group_ratio.detach(),
                'opd_video_transition_ratio': (opd_video_transition_contrib.detach().abs() / contrib_denom),
                'opd_endpoint_aux_ratio': (opd_endpoint_aux_contrib.detach().abs() / contrib_denom),
                'opd_same_state_velocity_ratio': (opd_same_state_velocity_contrib.detach().abs() / contrib_denom),
                'opd_local_fm_ratio': (opd_local_fm_contrib.detach().abs() / contrib_denom),
                'opd_action_transition_ratio': (opd_action_transition_contrib.detach().abs() / contrib_denom),
                'opd_action_local_fm_ratio': (opd_action_local_fm_contrib.detach().abs() / contrib_denom),
                'should_sync': should_sync,
                'skip_step': True,
                'nonfinite_origins': nonfinite_origins,
            }
            if kto_paopd:
                result.update({
                    'kto_good_ratio': kto_good_ratio.detach(),
                    'kto_weight_mean': kto_weight_mean.detach(),
                    'kto_weight_min': kto_weight_min.detach(),
                    'kto_weight_max': kto_weight_max.detach(),
                    'kto_threshold': kto_threshold_value.detach(),
                })
            return result

        _profile_mark('loss_build')
        loss.backward()
        _profile_mark('backward')
        if profile_opd and getattr(self.config, 'rank', 0) == 0:
            logger.info(
                "[OPD profile] " + ", ".join(
                    f"{k}={v:.3f}s" for k, v in profile_times.items()
                )
            )

        result = {
            'loss': loss.detach(),
            'opd_aux_loss': weighted_aux_loss.detach(),
            'opd_video_transition_loss': video_transition_loss.detach(),
            'opd_endpoint_aux_loss': opd_endpoint_aux_loss.detach(),
            'opd_same_state_velocity_loss': opd_same_state_velocity_loss.detach(),
            'opd_local_fm_loss': local_fm_loss.detach(),
            'opd_action_transition_loss': opd_action_transition_loss.detach(),
            'opd_action_local_fm_loss': opd_action_local_fm_loss.detach(),
            'opd_video_transition_contrib': opd_video_transition_contrib.detach(),
            'opd_endpoint_aux_contrib': opd_endpoint_aux_contrib.detach(),
            'opd_same_state_velocity_contrib': opd_same_state_velocity_contrib.detach(),
            'opd_local_fm_contrib': opd_local_fm_contrib.detach(),
            'opd_action_transition_contrib': opd_action_transition_contrib.detach(),
            'opd_action_local_fm_contrib': opd_action_local_fm_contrib.detach(),
            'opd_transition_group_scaled': scaled_transition_group.detach(),
            'opd_anchor_group_scaled': scaled_anchor_group.detach(),
            'opd_transition_scale': transition_scale.detach(),
            'opd_anchor_scale': anchor_scale.detach(),
            'opd_transition_group_ratio': opd_transition_group_ratio.detach(),
            'opd_anchor_group_ratio': opd_anchor_group_ratio.detach(),
            'opd_video_transition_ratio': (opd_video_transition_contrib.detach().abs() / contrib_denom),
            'opd_endpoint_aux_ratio': (opd_endpoint_aux_contrib.detach().abs() / contrib_denom),
            'opd_same_state_velocity_ratio': (opd_same_state_velocity_contrib.detach().abs() / contrib_denom),
            'opd_local_fm_ratio': (opd_local_fm_contrib.detach().abs() / contrib_denom),
            'opd_action_transition_ratio': (opd_action_transition_contrib.detach().abs() / contrib_denom),
            'opd_action_local_fm_ratio': (opd_action_local_fm_contrib.detach().abs() / contrib_denom),
            'rollout_steps': K_steps,
            'teacher_steps': effective_teacher_steps,
            'should_sync': should_sync,
            'skip_step': False,
            'nonfinite_origins': nonfinite_origins,
        }
        if kto_paopd:
            result.update({
                'kto_good_ratio': kto_good_ratio.detach(),
                'kto_weight_mean': kto_weight_mean.detach(),
                'kto_weight_min': kto_weight_min.detach(),
                'kto_weight_max': kto_weight_max.detach(),
                'kto_threshold': kto_threshold_value.detach(),
            })
        return result

    def _on_policy_rollout(self, batch, num_steps=None, cfg_scale=None, retain_grad=False):
        """
        用当前学生模型做 on-policy 推理，生成假动作。

        从纯噪声出发，用学生模型做 num_steps 步 Euler 去噪。
        视频 latent 保持 GT 不动（仅作为条件），只对动作做去噪。
        这样节省计算开销，且判别器条件使用 GT 视频 latent + GT 文本。

        参数:
            batch: 数据批次（包含 latents、actions、text_emb 等）
            num_steps: 推理步数（None 时从配置随机采样）
            cfg_scale: CFG 引导强度（None 时使用配置值）
            retain_grad: 是否保留梯度图（用于 DMD 梯度计算）

        返回:
            fake_action: 学生生成的动作 latent [B, C, F, N, 1]
        """
        B = batch['latents'].shape[0]
        # 使用非 FSDP 副本进行 DMD rollout（避免 FSDP + CheckpointWrapper 的 DTensor 问题）
        _rollout_model = getattr(self, '_student_nofsdp', self.student)
        if _rollout_model is not self.student:
            if not getattr(self, '_nofsdp_synced', False):
                self._sync_student_nofsdp()
                self._nofsdp_synced = True
            _rollout_model.train()
            # 将所有输入转为纯 Tensor（避免 DTensor + Tensor 混合报错）
            batch = {k: _to_regular_tensor(v) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _empty_emb = _to_regular_tensor(self.empty_emb)
        else:
            _empty_emb = self.empty_emb

        # 随机采样推理步数（所有 rank 必须使用相同步数，否则 FSDP allgather 会死锁）
        if num_steps is None:
            step_min = getattr(self.config, 'dmd_rollout_steps_min', 2)
            step_max = getattr(self.config, 'dmd_rollout_steps_max', 8)
            if dist.is_initialized():
                # rank 0 采样，broadcast 到所有 rank
                num_steps_tensor = torch.randint(step_min, step_max + 1, (1,), device=self.device)
                dist.broadcast(num_steps_tensor, src=0)
                num_steps = num_steps_tensor.item()
            else:
                num_steps = torch.randint(step_min, step_max + 1, (1,)).item()

        if cfg_scale is None:
            cfg_scale = getattr(self.config, 'dmd_cfg_scale', 5.0)

        # 从纯噪声出发（所有 rank 使用相同噪声，保证一致性）
        # 当 retain_grad=True 时，需要从一开始就保留梯度
        current_action = torch.randn_like(batch['actions'])
        if dist.is_initialized():
            # rank 0 生成噪声，broadcast 到所有 rank
            dist.broadcast(current_action, src=0)
        if retain_grad:
            current_action.requires_grad_(True)

        # 生成时间步序列
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        action_sigmas = self.train_scheduler_action.apply_shift(sigmas) * self.config.num_train_timesteps

        # 准备空文本嵌入（用于 CFG 无条件推理）
        empty_emb = _empty_emb.expand(B, -1, -1)

        # 获取动作帧数
        latent_num_frames = batch['latents'].shape[2]
        action_num_frames = batch['actions'].shape[2]

        # 生成位置编码网格（grid_id）
        # 需要为视频和动作分别生成 grid_id
        from wan_va.utils.utils import get_mesh_id
        patch_f, patch_h, patch_w = self.patch_size

        # 视频 grid_id
        latent_grid_id = get_mesh_id(
            batch['latents'].shape[-3] // patch_f,
            batch['latents'].shape[-2] // patch_h,
            batch['latents'].shape[-1] // patch_w,
            t=0, f_w=1, f_shift=0, action=False,
        ).to(self.device)
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        # 动作 grid_id
        action_grid_id = get_mesh_id(
            batch['actions'].shape[-3],  # 动作帧数
            batch['actions'].shape[-2],  # 动作高度
            batch['actions'].shape[-1],  # 动作宽度
            t=1, f_w=1, f_shift=0, action=True,
        ).to(self.device)
        action_grid_id = action_grid_id[None].repeat(B, 1, 1)

        # 使用 torch.no_grad() 包裹学生前向（不计算学生梯度）
        # 当 retain_grad=True 时，不用 no_grad() 以保留计算图
        grad_context = torch.enable_grad() if retain_grad else torch.no_grad()
        # 使用 FSDP no_sync() 禁用梯度同步，避免多卡 NCCL 超时
        # 在 rollout 阶段，多次前向传播不需要同步梯度
        # rollout model 不需要 FSDP no_sync，直接用 nullcontext
        fsdp_context = contextlib.nullcontext()
        with grad_context, fsdp_context:
            for i in range(num_steps):
                t_action = action_sigmas[i]
                r_action = action_sigmas[i + 1]

                # 构建 input_dict（视频用 GT latent 作为条件，不参与去噪）
                # 动作用当前去噪状态
                # t_action/r_action 是标量张量，需要先 unsqueeze 再 expand 到 [B, T]
                action_t_expanded = t_action.unsqueeze(0).expand(B, action_num_frames)  # [B, T]
                action_r_expanded = r_action.unsqueeze(0).expand(B, action_num_frames)  # [B, T]
                latent_zero_timesteps = torch.zeros(
                    B, latent_num_frames, device=self.device, dtype=action_t_expanded.dtype
                )

                rollout_input = {
                    'latent_dict': {
                        'noisy_latents': batch['latents'],
                        'latent': batch['latents'],
                        'timesteps': latent_zero_timesteps,  # 视频时间步为 0（GT 条件）
                        'cond_timesteps': latent_zero_timesteps,
                        'text_emb': batch['text_emb'],
                        'grid_id': latent_grid_id,
                    },
                    'action_dict': {
                        'noisy_latents': current_action,
                        'latent': batch['actions'],  # GT 动作作为条件参考
                        'timesteps': action_t_expanded,
                        'cond_timesteps': torch.zeros_like(action_t_expanded),
                        'text_emb': batch['text_emb'],
                        'grid_id': action_grid_id,
                    },
                    'chunk_size': self.config.frame_chunk_size,  # 从 config 获取
                    'window_size': self.config.attn_window,       # 从 config 获取
                }

                # DEBUG: check all input types
                if i == 0:
                    import torch.distributed.tensor as _dt
                    def _dt_check(name, obj):
                        if isinstance(obj, _dt.DTensor):
                            print(f"[DT-DEBUG] {name}: DTensor")
                        elif isinstance(obj, torch.Tensor):
                            print(f"[DT-DEBUG] {name}: Tensor {obj.shape} {obj.dtype}")
                        elif isinstance(obj, dict):
                            for k, v in obj.items():
                                _dt_check(f'{name}.{k}', v)
                    _dt_check('batch_latents', batch['latents'])
                    _dt_check('batch_text_emb', batch['text_emb'])
                    _dt_check('batch_actions', batch['actions'])
                    _dt_check('empty_emb', empty_emb)
                    _dt_check('current_action', current_action)
                    _dt_check('rollout_input', rollout_input)
                    _dt_check('action_r_expanded', action_r_expanded)
                    # Check model buffers
                    for bn, buf in _rollout_model.named_buffers():
                        if isinstance(buf, _dt.DTensor):
                            print(f"[DT-DEBUG] BUFFER {bn}: DTensor")

                # 学生有条件前向
                _, v_action_cond = _rollout_model(
                    rollout_input, train_mode=True,
                    r_timestep=latent_zero_timesteps,
                    action_r_timestep=action_r_expanded,
                )

                # 学生无条件前向（CFG）
                if cfg_scale > 1.0:
                    rollout_input_uncond = {
                        'latent_dict': {**rollout_input['latent_dict'], 'text_emb': empty_emb},
                        'action_dict': {**rollout_input['action_dict'], 'text_emb': empty_emb},
                        'chunk_size': rollout_input['chunk_size'],
                        'window_size': rollout_input['window_size'],
                    }
                    _, v_action_uncond = _rollout_model(
                        rollout_input_uncond, train_mode=True,
                        r_timestep=latent_zero_timesteps,
                        action_r_timestep=action_r_expanded,
                    )
                    v_action = v_action_uncond + cfg_scale * (v_action_cond - v_action_uncond)
                else:
                    v_action = v_action_cond

                # Euler 更新（仅更新动作）
                # v_action 形状: [B, F*N, C]，需要 reshape 为 [B, C, F, N, 1]
                v_action_5d = self._extract_action_v(v_action, batch['actions'].shape[2])
                v_action_5d = _to_regular_tensor(v_action_5d)  # Convert DTensor to regular Tensor for Euler update
                current_action = current_action - (t_action - r_action) * v_action_5d


        # 如果需要保留梯度图（用于 DMD 梯度计算）
        if retain_grad:
            current_action.requires_grad_(True)

        return current_action

    def _dmd_train_step(self, batch):
        """
        DMD 训练步：计算学生 DMD loss + 更新判别器。

        关键顺序约束：
          - 学生梯度必须在判别器更新之前计算
          - 因为判别器参数 inplace 更新会破坏计算图

        流程：
          1. On-policy rollout（保留计算图）：用学生生成假动作
          2. 判别器前向（非 detach）：构建学生梯度的计算图
          3. 计算 DMD loss 并 backward 到学生（判别器参数还未被修改）
          4. 判别器前向（detach 假动作）+ backward + 更新判别器

        参数:
            batch: 数据批次

        返回:
            d_loss: 判别器 loss（标量，用于日志）
            dmd_loss: DMD loss（标量，已 backward，用于日志）
        """
        from discriminator import train_discriminator_step

        # 1. On-policy rollout（保留计算图）
        fake_action = self._on_policy_rollout(batch, retain_grad=True)

        # 2. 计算判别器对假动作的打分
        # 关键：禁用判别器梯度，避免 backward 时触发 FSDP reshard 导致 DTensor/Tensor 混用
        # 参考 DMD2: https://github.com/tianweiy/DMD2/blob/main/main/sd_unified_model.py
        disc_was_training = self.discriminator.training
        self.discriminator.requires_grad_(False)
        self.discriminator.eval()
        try:
            fake_action_for_disc = _to_regular_tensor(fake_action)
            fake_logits = self.discriminator(
                fake_action_for_disc,
                _to_regular_tensor(batch['latents']),
                _to_regular_tensor(batch['text_emb']),
            )
            # Generator loss: make fake actions look real to the discriminator.
            dmd_loss = getattr(self.config, 'dmd_weight', 0.1) * F.binary_cross_entropy_with_logits(
                fake_logits, torch.ones_like(fake_logits)
            )

            # 3. backward 到学生（判别器梯度已禁用，不会触发 reshard）
            dmd_loss.backward()
        finally:
            self.discriminator.requires_grad_(True)
            if disc_was_training:
                self.discriminator.train()

        # 4. 再更新判别器（detach 假动作，不回传梯度到学生）
        d_loss = train_discriminator_step(
            self.discriminator,
            real_actions=_to_regular_tensor(batch['actions']),
            fake_actions=_to_regular_tensor(fake_action.detach()),
            video_latent=_to_regular_tensor(batch['latents']),
            text_emb=_to_regular_tensor(batch['text_emb']),
        )
        d_loss.backward()
        self.discriminator_optimizer.step()
        self.discriminator_optimizer.zero_grad()

        return d_loss.detach(), dmd_loss.detach()
