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
    """Weighted per-sample velocity loss for optional DanceOPD-style regularization."""
    diff = student_v.float() - teacher_v.detach().float()
    if transition_loss_type == "huber":
        abs_diff = diff.abs()
        elem_loss = torch.where(
            abs_diff < transition_huber_c,
            0.5 * diff ** 2,
            transition_huber_c * (abs_diff - 0.5 * transition_huber_c),
        )
    else:
        elem_loss = diff ** 2
    per_sample_loss = elem_loss.flatten(1).mean(dim=1)
    return (per_sample_loss * sample_weight.to(per_sample_loss)).mean()


from einops import rearrange

from utils import data_seq_to_patch, logger
from distillation.consistency import scalings_for_boundary_conditions


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
        """Return _teacher_nofsdp if available, else fall back to teacher."""
        return getattr(self, '_teacher_nofsdp', self.teacher)

    # ==================================================================
    # 混合时间步采样：扩散目标 + 一致性目标 + 流映射目标
    # ==================================================================
    def sample_timestep_mixed(
        self, batch_size, num_frames, dtype, device, scheduler=None,
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
            v_plus_all, _ = self._teacher_model(input_plus, train_mode=True)

            # 恢复 mask（teacher forward 会更新 mask，需要在下次前向前重置）
            FlexAttnFunc.attention_mask = None
            FlexAttnFunc.cross_attention_mask = None

            # 前向 2: t-eps, cond+uncond (2B)
            input_minus = _build_2b_input(noisy_latents_minus, latents, t_minus)
            v_minus_all, _ = self._teacher_model(input_minus, train_mode=True)
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
            v_cond_all, action_cond_all = self._teacher_model(cond_input, train_mode=True)

            # 重置 mask（第二次前向需要重新创建）
            FlexAttnFunc.attention_mask = None
            FlexAttnFunc.cross_attention_mask = None

            # 前向 2 (3B): uncond at (t, t+ε, t-ε)
            uncond_input = _build_3b_input(torch.cat([empty_expanded] * 3, dim=0))
            v_uncond_all, _ = self._teacher_model(uncond_input, train_mode=True)
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
            v_cond_all, action_cond_all = self._teacher_model(cond_input, train_mode=True)

            FlexAttnFunc.attention_mask = None
            FlexAttnFunc.cross_attention_mask = None

            # 前向 2 (3B): uncond at (t, t+ε, t-ε)
            uncond_input = _build_3b_input(torch.cat([empty_expanded] * 3, dim=0))
            v_uncond_all, _ = self._teacher_model(uncond_input, train_mode=True)
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
            return self._teacher_model(input_dict, train_mode=True)
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
            v_all, action_all = self._teacher_model(doubled_input, train_mode=True)
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
            _, action_all = self._teacher_model(doubled_input, train_mode=True)
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
                              force_cfg=False):
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
        """
        if cfg_scale <= 1.0 and not force_cfg:
            from modules.model import FlexAttnFunc
            saved_attn = FlexAttnFunc.attention_mask
            saved_cross = FlexAttnFunc.cross_attention_mask
            try:
                FlexAttnFunc.attention_mask = None
                FlexAttnFunc.cross_attention_mask = None
                v, _ = model(input_dict, train_mode=True,
                              r_timestep=r_timestep,
                              action_r_timestep=action_r_timestep)
            finally:
                FlexAttnFunc.attention_mask = saved_attn
                FlexAttnFunc.cross_attention_mask = saved_cross
            return self._extract_video_v(v, ref_shape, B)

        from modules.model import FlexAttnFunc

        def _cat(a, b):
            return torch.cat([a, b], dim=0)

        text_emb = input_dict['latent_dict']['text_emb']
        empty_expanded = empty_emb.expand(B, -1, -1)

        ld = input_dict['latent_dict']
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

        ad = input_dict['action_dict']
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
            v_all, _ = model(doubled_input, train_mode=True,
                              r_timestep=doubled_r,
                              action_r_timestep=doubled_ar)
        finally:
            FlexAttnFunc.attention_mask = saved_attn
            FlexAttnFunc.cross_attention_mask = saved_cross

        v_cond, v_uncond = v_all.chunk(2, dim=0)
        v_cond_5d = self._extract_video_v(v_cond, ref_shape, B)
        v_uncond_5d = self._extract_video_v(v_uncond, ref_shape, B)
        return v_uncond_5d + cfg_scale * (v_cond_5d - v_uncond_5d)


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

    # ==================================================================
    # 核心训练步：Flow Map 蒸馏
    # ==================================================================
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
                action_use_flowmap = getattr(self.config, 'action_use_flowmap', False)
                if action_use_flowmap:
                    flowmap_token_mask = (
                        ((action_t[:, ::_ad] - action_r[:, ::_ad]).abs() > 1e-3) &
                        (action_r[:, ::_ad] > 1e-3)
                    )
                else:
                    flowmap_token_mask = torch.zeros_like(action_r[:, ::_ad], dtype=torch.bool)

                if use_action_distill and target_action_pred is not None:
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

        # 对视频 loss 按样本加权
        if self.distill_video:
            # 逐样本 loss（支持 huber 和 mse）
            loss_type = getattr(self.config, 'loss_type', 'l2')
            huber_c = getattr(self.config, 'huber_c', 0.001)
            video_diff = (student_video_v.float() - video_target.detach().float())
            if loss_type == "huber":
                # Huber loss: |x| < c -> 0.5*x^2; otherwise c*(|x|-0.5*c)
                abs_diff = video_diff.abs()
                per_sample_video_loss = torch.where(
                    abs_diff < huber_c,
                    0.5 * video_diff ** 2,
                    huber_c * (abs_diff - 0.5 * huber_c)
                ).mean(dim=[1, 2, 3, 4])  # [B]
            else:
                # MSE loss（原始行为）
                per_sample_video_loss = (video_diff ** 2).mean(dim=[1, 2, 3, 4])  # [B]

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

        # 检查损失是否为 NaN/Inf，如果是则跳过这一步

        if not torch.isfinite(loss):
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
                "should_sync": should_sync,
                "skip_step": True,
            }

        loss.backward()


        return {
            "loss": loss.detach(),
            "video_loss": video_loss.detach(),
            "action_loss": action_loss.detach() if self.distill_action else action_loss,
            "action_local_fm_loss": action_local_fm_loss.detach() if self.action_aware else action_local_fm_loss,
            "action_aware_loss": action_aware_loss.detach() if self.action_aware else action_aware_loss,
            "gt_regression_loss": gt_regression_loss.detach() if self.distill_action else gt_regression_loss,
            "should_sync": should_sync,
            "skip_step": False,
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
                                   return_last_step_start=False):
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

        rollout_grad_mode = getattr(self.config, 'opd_rollout_grad_mode', 'endpoint')
        rollout_grad_mode = str(rollout_grad_mode).lower()
        if rollout_grad_mode not in ('endpoint', 'last_step', 'full'):
            raise ValueError(
                f"Invalid opd_rollout_grad_mode={rollout_grad_mode!r}; "
                "expected endpoint, last_step, or full."
            )

        use_nofsdp_rollout = (
            rollout_grad_mode == 'endpoint' or
            (rollout_grad_mode == 'last_step' and K_steps > 1)
        )
        _rollout_model = (getattr(self, '_student_nofsdp', None) or self.student) if use_nofsdp_rollout else self.student
        if _rollout_model is not self.student:
            if not getattr(self, '_nofsdp_synced', False):
                self._sync_student_nofsdp()
                self._nofsdp_synced = True
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

        def _student_cfg_with_optional_uncompile(model, step_input, step_empty, r_timestep_i, action_r_timestep_i):
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
                    force_cfg=True,
                )
            finally:
                if saved_blocks is not None:
                    for bi, block in enumerate(saved_blocks):
                        self.student.blocks[bi] = block

        step_ts = self._build_timestep_path(timesteps.float(), target_r.float(), K_steps)
        current_x = noisy_latents
        last_step_start_x = current_x
        last_step_start_t = step_ts[0]

        for i in range(K_steps):
            t_i = step_ts[i]
            r_next = step_ts[min(i + 1, K_steps)]
            if i == K_steps - 1:
                last_step_start_x = current_x.detach()
                last_step_start_t = t_i.detach()

            keep_step_grad = (
                rollout_grad_mode == 'full' or
                (rollout_grad_mode == 'last_step' and i == K_steps - 1)
            )
            step_model = self.student if keep_step_grad else _rollout_model
            step_latent = base_input_dict['latent_dict'] if keep_step_grad else _rt_latent
            step_action = base_input_dict['action_dict'] if keep_step_grad else _rt_action
            step_empty = empty_emb if keep_step_grad else _rt_empty
            if keep_step_grad and current_x.device != self.device:
                current_x = current_x.to(self.device)
            if keep_step_grad and current_x.dtype != base_input_dict['latent_dict']['latent'].dtype:
                current_x = current_x.to(base_input_dict['latent_dict']['latent'].dtype)

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
                action_target_r_ds if action_target_r_ds is not None
                else (r_timestep_i[:, ::_ACTION_DS] if (self.distill_action or self.action_aware) else r_timestep_i)
            )

            grad_context = torch.enable_grad() if keep_step_grad else torch.no_grad()
            with grad_context:
                v_cfg = _student_cfg_with_optional_uncompile(
                    step_model, step_input, step_empty, r_timestep_i, action_r_timestep_i)
                sigma_i = self._timestep_to_sigma_5d(t_i)
                sigma_next = self._timestep_to_sigma_5d(r_next)
                current_x = current_x + v_cfg * (sigma_next - sigma_i)

        current_x_for_loss = current_x if rollout_grad_mode != 'endpoint' else current_x.detach()
        current_x_for_forward = current_x.detach()

        final_input = {
            'latent_dict': {
                **base_input_dict['latent_dict'],
                'noisy_latents': current_x_for_forward,
                'timesteps': target_r,
            },
            'action_dict': base_input_dict['action_dict'],
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
            v_final_cfg = self._student_cfg_forward(
                self.student, final_input, empty_emb, cfg_scale,
                B, ref_shape, target_r, _act_r_final, force_cfg=False,
            )
        finally:
            if _saved_blocks is not None:
                for _bi, _block in enumerate(_saved_blocks):
                    self.student.blocks[_bi] = _block

        if return_last_step_start:
            return current_x_for_loss, v_final_cfg, last_step_start_x.detach(), last_step_start_t.detach()

        return current_x_for_loss, v_final_cfg


    def _teacher_integrate_to_r(self, noisy_latents, timesteps, target_r, input_dict, empty_emb,
                                 cfg_scale, ref_shape, B, num_frames, num_steps=2):
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
        if not torch.isfinite(loss):
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
        }

    def _opd_aux_transition_step(self, batch, batch_idx):
        """
        Auxiliary teacher-transition loss on top of the regular FlowMap step.

        This keeps the regular AnyFlow step as the main objective and adds a
        teacher-transition auxiliary for both video and action.
        """
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
        )
        video_r = self._apply_opd_low_noise_query_bias(video_t, video_r)
        video_r_sigma = video_r / self.config.num_train_timesteps

        action_t = action_r = None
        action_t_sigma = action_r_sigma = None
        if self.distill_action or self.action_aware:
            action_t, action_r, _ = self.sample_timestep_mixed(
                B, num_frames, dtype=torch.float32, device=self.device,
                scheduler=self.train_scheduler_action,
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
                if action_grad_mode not in ('endpoint', 'last_step', 'full'):
                    raise ValueError(
                        f"Invalid opd_action_rollout_grad_mode={action_grad_mode!r}; "
                        "expected endpoint, last_step, or full."
                    )

                action_student_path = self._build_timestep_path(
                    action_t_ds.float(), action_r_ds.float(), max(1, K_steps))
                student_action_x = action_noisy_ds
                for i in range(max(1, K_steps)):
                    t_i = action_student_path[i]
                    r_i = action_student_path[i + 1]
                    keep_step_grad = (
                        action_grad_mode == 'full' or
                        (action_grad_mode == 'last_step' and i == max(1, K_steps) - 1)
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
        video_transition_loss = (per_sample_loss * weight).mean()

        opd_endpoint_aux_loss = torch.tensor(0.0, device=self.device)
        opd_endpoint_aux_weight = float(getattr(self.config, 'opd_endpoint_aux_weight', 0.0))
        if opd_endpoint_aux_weight > 0:
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

        raw_opd_video_transition_contrib = video_transition_weight * video_transition_loss
        raw_opd_endpoint_aux_contrib = opd_endpoint_aux_weight * opd_endpoint_aux_loss
        raw_opd_same_state_velocity_contrib = (
            opd_same_state_velocity_weight * opd_same_state_velocity_loss
        )
        raw_opd_local_fm_contrib = local_fm_weight * local_fm_loss
        raw_opd_action_transition_contrib = (
            action_transition_block_weight * action_transition_weight * opd_action_transition_loss
        )
        raw_opd_action_local_fm_contrib = (
            action_local_fm_block_weight * action_local_fm_weight * opd_action_local_fm_loss
        )

        raw_transition_group = (
            raw_opd_video_transition_contrib + raw_opd_action_transition_contrib
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
        anchor_cap_ratio = float(getattr(self.config, 'opd_anchor_cap_ratio', -1.0))
        anchor_scale = torch.ones((), device=self.device, dtype=raw_anchor_group.dtype)
        if anchor_cap_ratio >= 0:
            anchor_cap = scaled_transition_group.detach().abs() * anchor_cap_ratio
            raw_anchor_abs = raw_anchor_group.detach().abs().clamp(min=1e-12)
            anchor_scale = torch.minimum(anchor_scale, anchor_cap / raw_anchor_abs)
        scaled_anchor_group = raw_anchor_group * anchor_scale

        opd_video_transition_contrib = raw_opd_video_transition_contrib * transition_scale
        opd_action_transition_contrib = raw_opd_action_transition_contrib * transition_scale
        opd_endpoint_aux_contrib = raw_opd_endpoint_aux_contrib * anchor_scale
        opd_same_state_velocity_contrib = raw_opd_same_state_velocity_contrib * anchor_scale
        opd_local_fm_contrib = raw_opd_local_fm_contrib * anchor_scale
        opd_action_local_fm_contrib = raw_opd_action_local_fm_contrib * anchor_scale
        raw_aux_loss = scaled_transition_group + scaled_anchor_group
        contrib_denom = (
            scaled_transition_group.detach().abs() + scaled_anchor_group.detach().abs()
        ).clamp(min=1e-12)
        opd_transition_group_ratio = scaled_transition_group.detach().abs() / contrib_denom
        opd_anchor_group_ratio = scaled_anchor_group.detach().abs() / contrib_denom
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

        if not torch.isfinite(loss):
            if self.config.rank == 0:
                logger.warning(f"[step {self.step}] NaN/Inf OPD aux loss, skipping")
            zero_loss = torch.zeros((), device=self.device)
            return {
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
            }

        _profile_mark('loss_build')
        loss.backward()
        _profile_mark('backward')
        if profile_opd and getattr(self.config, 'rank', 0) == 0:
            logger.info(
                "[OPD profile] " + ", ".join(
                    f"{k}={v:.3f}s" for k, v in profile_times.items()
                )
            )

        return {
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
        }

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
