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

import torch
import torch.nn.functional as F
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

    # ==================================================================
    # 混合时间步采样：扩散目标 + 一致性目标 + 流映射目标
    # ==================================================================
    def sample_timestep_mixed(self, batch_size, num_frames, dtype, device):
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

        # 步骤 3：根据配置比例决定 r 的取值（向量化实现）
        n_diffusion = round(self.diffusion_ratio * batch_size)
        n_consistency = round(self.consistency_ratio * batch_size)

        # 创建布尔掩码
        is_diffusion = torch.zeros(batch_size, dtype=torch.bool, device=device)
        is_diffusion[:n_diffusion] = True

        # 向量化赋值：扩散目标 r = t，一致性目标 r = 0
        r[:n_diffusion] = t[:n_diffusion]  # 扩散：r = t
        r[n_diffusion:n_diffusion + n_consistency] = 0.0  # 一致性：r = 0
        # 剩余的是流映射目标（不修改 r，保持 r < t）

        # 步骤 4：扩展为 per-frame [B, T]（与 AnyFlow 参考实现一致）
        t = t.unsqueeze(1).expand(-1, num_frames)
        r = r.unsqueeze(1).expand(-1, num_frames)

        # 应用 SNR shift（与 FlowMatchScheduler 一致）并转换为原始时间步
        t = self.train_scheduler_latent.apply_shift(t) * self.config.num_train_timesteps
        r = self.train_scheduler_latent.apply_shift(r) * self.config.num_train_timesteps
        return t, r, is_diffusion

    # ==================================================================
    # 中心差分法计算 flow map 梯度 dF/dt
    # ==================================================================
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
            v_plus_all, _ = self.teacher(input_plus, train_mode=True)

            # 恢复 mask（teacher forward 会更新 mask，需要在下次前向前重置）
            FlexAttnFunc.attention_mask = None
            FlexAttnFunc.cross_attention_mask = None

            # 前向 2: t-eps, cond+uncond (2B)
            input_minus = _build_2b_input(noisy_latents_minus, latents, t_minus)
            v_minus_all, _ = self.teacher(input_minus, train_mode=True)
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
        t_plus, t_minus, latents
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

        def _build_3b_input(text_emb_3b):
            """构建 3B input_dict: (t, t+ε, t-ε) 三个时间步。"""
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
                    'noisy_latents': torch.cat([ad['noisy_latents']] * 3, dim=0),
                    'latent':        torch.cat([ad['latent']] * 3, dim=0),
                    'timesteps':     torch.cat([ad['timesteps']] * 3, dim=0),
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
            v_cond_all, action_cond_all = self.teacher(cond_input, train_mode=True)

            # 重置 mask（第二次前向需要重新创建）
            FlexAttnFunc.attention_mask = None
            FlexAttnFunc.cross_attention_mask = None

            # 前向 2 (3B): uncond at (t, t+ε, t-ε)
            uncond_input = _build_3b_input(torch.cat([empty_expanded] * 3, dim=0))
            v_uncond_all, _ = self.teacher(uncond_input, train_mode=True)
        finally:
            FlexAttnFunc.attention_mask = saved_attn_mask
            FlexAttnFunc.cross_attention_mask = saved_cross_mask

        # 拆分 3B → 3 个 B: (t, t+ε, t-ε)
        v_cond_t, v_cond_plus, v_cond_minus = v_cond_all.chunk(3, dim=0)
        v_uncond_t, v_uncond_plus, v_uncond_minus = v_uncond_all.chunk(3, dim=0)
        action_cond_t, _, _ = action_cond_all.chunk(3, dim=0)

        # CFG 组合
        v_cfg_t = v_uncond_t + cfg_scale * (v_cond_t - v_uncond_t)
        v_cfg_plus = v_uncond_plus + cfg_scale * (v_cond_plus - v_uncond_plus)
        v_cfg_minus = v_uncond_minus + cfg_scale * (v_cond_minus - v_uncond_minus)

        return v_cfg_t, action_cond_t, v_cfg_plus, v_cfg_minus

    # ==================================================================
    # 从模型输出中提取视频 v-prediction → [B, C, F, H, W]
    # ==================================================================

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

        # latent_dict：所有 batch 维张量翻倍，text_emb 特殊处理
        ld = input_dict['latent_dict']
        doubled_latent_dict = {
            'noisy_latents': _cat(ld['noisy_latents'], ld['noisy_latents']),
            'latent':        _cat(ld['latent'], ld['latent']),
            'timesteps':     _cat(ld['timesteps'], ld['timesteps']),
            'cond_timesteps':_cat(ld['cond_timesteps'], ld['cond_timesteps']),
            'text_emb':      _cat(text_emb, empty_expanded),
        }
        if 'grid_id' in ld and ld['grid_id'] is not None:
            doubled_latent_dict['grid_id'] = _cat(ld['grid_id'], ld['grid_id'])
        # targets 可能存在也可能不存在，如果存在就复制
        if 'targets' in ld:
            doubled_latent_dict['targets'] = _cat(ld['targets'], ld['targets'])

        # action_dict：同理
        ad = input_dict['action_dict']
        doubled_action_dict = {
            'noisy_latents': _cat(ad['noisy_latents'], ad['noisy_latents']),
            'latent':        _cat(ad['latent'], ad['latent']),
            'timesteps':     _cat(ad['timesteps'], ad['timesteps']),
            'cond_timesteps':_cat(ad['cond_timesteps'], ad['cond_timesteps']),
            'text_emb':      _cat(ad['text_emb'], empty_expanded),
        }
        if 'grid_id' in ad and ad['grid_id'] is not None:
            doubled_action_dict['grid_id'] = _cat(ad['grid_id'], ad['grid_id'])
        if 'targets' in ad:
            doubled_action_dict['targets'] = _cat(ad['targets'], ad['targets'])
        if 'actions_mask' in ad:
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
            v_all, action_all = self.teacher(doubled_input, train_mode=True)
        finally:
            FlexAttnFunc.attention_mask = saved_attn_mask
            FlexAttnFunc.cross_attention_mask = saved_cross_mask

        # 拆分：前半 = 有条件，后半 = 无条件
        v_cond, v_uncond = v_all.chunk(2, dim=0)
        action_cond, _ = action_all.chunk(2, dim=0) if action_all is not None else (None, None)

        return v_cond, v_uncond, action_cond

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
        if not use_flowmap:
            # 将流映射比例分配给扩散和一致性（按原始比例归一化）
            total = self.diffusion_ratio + self.consistency_ratio
            if total > 0:
                self.diffusion_ratio = self.diffusion_ratio / total
                self.consistency_ratio = self.consistency_ratio / total
            else:
                self.diffusion_ratio = 0.5
                self.consistency_ratio = 0.5

        # 视频时间步采样：sample_timestep_mixed 已应用 SNR shift 并转换为原始时间步
        video_t, video_r, video_is_diffusion = self.sample_timestep_mixed(
            B, num_frames, dtype=torch.float32, device=self.device
        )

        # 恢复原始比例
        self.diffusion_ratio = orig_diffusion_ratio
        self.consistency_ratio = orig_consistency_ratio

        # 保留归一化 sigma（用于后续 x0 参数化等需要归一化值的场景）
        video_t_sigma = video_t / self.config.num_train_timesteps

        # 动作时间步采样（如果启用动作蒸馏）
        if self.distill_action:
            # 同样临时覆盖比例
            orig_diffusion_ratio = self.diffusion_ratio
            orig_consistency_ratio = self.consistency_ratio
            if not use_flowmap:
                total = self.diffusion_ratio + self.consistency_ratio
                if total > 0:
                    self.diffusion_ratio = self.diffusion_ratio / total
                    self.consistency_ratio = self.consistency_ratio / total
                else:
                    self.diffusion_ratio = 0.5
                    self.consistency_ratio = 0.5

            action_t, action_r, action_is_diffusion = self.sample_timestep_mixed(
                B, num_frames, dtype=torch.float32, device=self.device
            )

            # 恢复原始比例
            self.diffusion_ratio = orig_diffusion_ratio
            self.consistency_ratio = orig_consistency_ratio

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

        if self.distill_action:
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

        if self.distill_action:
            input_dict['action_dict']['noisy_latents'] = action_noisy_latents
            input_dict['action_dict']['timesteps'] = action_t  # [B, F]
            input_dict['action_dict']['targets'] = action_v_target

        # ==============================================================
        # 步骤 4: 教师前向（CFG + 中心差分合并优化）
        # ==============================================================
        # 随机采样 CFG 引导强度（在 [cfg_min, cfg_max] 范围内均匀采样）
        cfg_scale = self.config.cfg_min + torch.rand(1).item() * (
            self.config.cfg_max - self.config.cfg_min)

        # 准备空文本嵌入（用于批量 CFG 前向）
        B_emb = input_dict['latent_dict']['text_emb'].shape[0]
        empty_emb = self.empty_emb.expand(B_emb, -1, -1)

        # 预计算 t+ε 和 t-ε 的噪声样本和时间步（中心差分用）
        v_pred = video_noise - batch['latents']
        t_plus = (video_t + self.epsilon).clamp(max=self.config.num_train_timesteps)
        noisy_latents_plus = video_noisy_latents + v_pred * (self.epsilon / self.config.num_train_timesteps)
        t_minus = (video_t - self.epsilon).clamp(min=0)
        noisy_latents_minus = video_noisy_latents - v_pred * (self.epsilon / self.config.num_train_timesteps)

        # ==============================================================
        # 步骤 5: 中心差分计算 dF/dt（与 CFG 前向合并优化）
        # ==============================================================
        if selective_cdiff:
            non_diffusion_mask = ~video_is_diffusion  # [B]
            has_non_diffusion = non_diffusion_mask.any()
        else:
            non_diffusion_mask = torch.ones(B, dtype=torch.bool, device=self.device)
            has_non_diffusion = True

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

                    # Central difference for sub-batch
                    dF_dt_sub = self.compute_central_difference_merged(
                        input_dict=input_dict, empty_emb=empty_emb, cfg_scale=cfg_scale,
                        noisy_latents=noisy_latents_sub, latents=latents_sub,
                        noise=noise_sub, t=t_sub, r=r_sub,
                        eps=self.epsilon, ref_shape=ref_shape)
                    dF_dt = torch.zeros_like(video_v_cfg_5d)
                    dF_dt[non_diff_indices] = dF_dt_sub
                else:
                    # B=1 或不筛选: 合并方案 2×3B（替代 3×2B）
                    video_v_cfg_seq, action_v_cond, v_plus_seq, v_minus_seq = \
                        self._merged_cfg_central_diff(
                            input_dict, empty_emb, cfg_scale,
                            noisy_latents_plus, noisy_latents_minus,
                            t_plus, t_minus, batch['latents'])
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
            if self.distill_action:
                action_v_5d = self._extract_action_v(action_v_cond, num_frames)

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
                'timesteps': video_r,  # [B, F] 学生在 r 处预测
            },
            'action_dict': {
                'noisy_latents': action_noisy_latents[:, :, ::_ACTION_DS],
                'latent':        batch['actions'][:, :, ::_ACTION_DS],
                'timesteps':     action_r[:, ::_ACTION_DS],
                'cond_timesteps':input_dict['action_dict']['cond_timesteps'][:, ::_ACTION_DS],
                'text_emb':      input_dict['action_dict']['text_emb'],
            },
            'chunk_size': input_dict['chunk_size'],
            'window_size': input_dict['window_size'],
        }
        if 'grid_id' in input_dict['action_dict'] and input_dict['action_dict']['grid_id'] is not None:
            student_input['action_dict']['grid_id'] = input_dict['action_dict']['grid_id'][:, :, ::_ACTION_DS]

        # Action r_timestep 也需要下采样，与 action token 维度对齐
        action_r_ds = action_r[:, ::_ACTION_DS] if self.distill_action else video_r

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

        if self.distill_action:
            _action_num_frames = batch['actions'].shape[2] // _ACTION_DS  # 下采样后的动作帧数
            student_action_v = self._extract_action_v(student_action_v_seq, _action_num_frames)

        # ==============================================================
        # 步骤 8b: 目标学生（EMA）生成 action 目标（教师蒸馏）
        # ==============================================================
        # 核心改进：使用 target_student 在 action_r 处生成蒸馏目标，
        # 而非直接使用 GT 动作。这是原始 LCM 蒸馏的核心思想。
        use_action_distill = getattr(self.config, 'use_action_distill', True)
        target_action_pred = None
        if self.distill_action and use_action_distill:
            # 构建 target_student 的 input_dict（action 也下采样）
            target_input = {
                'latent_dict': {
                    **input_dict['latent_dict'],
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
                target_input['action_dict']['grid_id'] = input_dict['action_dict']['grid_id'][:, :, ::_ACTION_DS]

            with torch.no_grad():
                # 目标学生（EMA）在 action_r 处做前向
                _, target_action_v_seq = self.target_student(
                    target_input, train_mode=True,
                    r_timestep=video_r,
                    action_r_timestep=action_r[:, ::_ACTION_DS],
                )
                target_action_v = self._extract_action_v(target_action_v_seq, batch['actions'].shape[2] // _ACTION_DS)

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

        if self.distill_action:
            # Action 时间下采样：与学生模型一致（每 _ACTION_DS 帧取 1 个）
            # 学生输出 64 tokens，loss 也要用 64 tokens 的 target
            _ad = _ACTION_DS  # 缩写
            action_noisy_ds = action_noisy_latents[:, :, ::_ad]
            actions_mask_ds = actions_mask[:, :, ::_ad]
            actions_gt_ds = batch['actions'][:, :, ::_ad]

            # 学生的 x0 预测（只计算一次，后续复用）
            action_sigma_s = action_r_sigma[:, None, ::_ad, None, None].to(student_action_v)
            student_action_pred = action_noisy_ds - action_sigma_s * student_action_v
            # mask 也只转换一次
            mask = actions_mask_ds.float()

            # 判断使用哪种目标
            action_use_flowmap = getattr(self.config, 'action_use_flowmap', False)
            has_flowmap = action_use_flowmap and (
                (action_t - action_r).abs().max() > 1e-3 and
                action_r.min() > 1e-3  # 排除一致性（r=0）和扩散（r≈t）
            )

            if has_flowmap:
                # Flow Map 目标：与视频流映射目标策略一致
                # 视频: video_target = video_v_cfg_5d - (t_5d - r_5d) * dF_dt
                # 动作（x0 空间）: 优先使用 EMA 目标学生的预测（已包含教师 Euler 步 + EMA 在 r 处的 v-prediction）
                # 回退: 使用教师的 v-prediction 转换到 x0 空间
                if target_action_pred is not None:
                    # EMA 目标学生已计算: target = x_prev_action - sigma_r * v_ema_at_r
                    action_target_pred = target_action_pred
                else:
                    # 教师的 x0 预测: x0 = x_t - sigma_t * v_teacher
                    sigma_t_a_5d = action_t_sigma[:, None, ::_ad, None, None].to(action_v_5d)
                    action_v_5d_ds = action_v_5d[:, :, ::_ad]
                    action_target_pred = action_noisy_ds - sigma_t_a_5d * action_v_5d_ds
            elif use_action_distill and target_action_pred is not None:
                action_target_pred = target_action_pred
            elif self.action_distill_mode == "x0":
                action_target_pred = actions_gt_ds
            else:
                student_action_pred = self._consistency_function(
                    student_action_v, action_noisy_ds, action_t_sigma
                )
                # 教师 v-prediction 也要下采样
                action_v_5d_ds = action_v_5d[:, :, :, 0:1, :]
                action_target_pred = action_v_5d_ds

            action_diff = (student_action_pred.float() * mask) - \
                          (action_target_pred.detach().float() * mask)
            action_loss = (action_diff ** 2).sum() / mask.sum().clamp(min=1)

            # --- 9c. GT 回归损失（Flow Map 特有辅助损失）---
            # 复用 student_action_pred（与 student_action_x0 相同）
            if use_gt_regression and self.gt_regression_weight > 0:
                gt_diff = (student_action_pred.float() * mask) - \
                          (actions_gt_ds.float() * mask)
                gt_regression_loss = (gt_diff ** 2).sum() / mask.sum().clamp(min=1)

            # --- 9d. 动作感知正则化（原生 flow matching MSE）---
            if self.action_aware:
                action_targets = input_dict['action_dict']['targets'][:, :, ::_ad]
                aa_diff = (student_action_v.float() - action_targets.float().detach()) * mask
                action_aware_loss = (aa_diff ** 2).sum() / mask.sum().clamp(min=1)

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
                # Huber loss: 对 outlier 更鲁棒
                # |x| < c: 0.5 * x^2 / c
                # |x| >= c: |x| - 0.5 * c
                abs_diff = video_diff.abs()
                per_sample_video_loss = torch.where(
                    abs_diff < huber_c,
                    0.5 * video_diff ** 2 / huber_c,
                    abs_diff - 0.5 * huber_c
                ).mean(dim=[1, 2, 3, 4])  # [B]
            else:
                # MSE loss（原始行为）
                per_sample_video_loss = (video_diff ** 2).mean(dim=[1, 2, 3, 4])  # [B]

            # AnyFlow 技巧：用扩散样本的 loss 均值缩放非扩散样本
            # 这有助于平衡不同模式的梯度贡献
            # 注意：缩放因子在 no_grad 块内计算，但最终 loss 赋值必须在块外以保留梯度
            with torch.no_grad():
                if video_is_diffusion.any():
                    diffusion_mean = per_sample_video_loss[video_is_diffusion].mean()
                    non_diffusion_mask = ~video_is_diffusion
                    if non_diffusion_mask.any():
                        non_diffusion_mean = per_sample_video_loss[non_diffusion_mask].mean().clamp(min=1e-5)
                        scale_weight = diffusion_mean / non_diffusion_mean
                        # 对非扩散样本的 loss 进行缩放
                        per_sample_video_loss[non_diffusion_mask] = \
                            per_sample_video_loss[non_diffusion_mask] * scale_weight

            # no_grad 块外：最终加权 loss 保留梯度
            video_loss = (per_sample_video_loss * weight).mean()

        # ==============================================================
        # 步骤 11: 总损失
        # ==============================================================
        video_loss_weight = getattr(self.config, 'video_loss_weight', 1.0)
        loss = video_loss_weight * video_loss \
               + self.config.action_loss_weight * action_loss \
               + self.gt_regression_weight * gt_regression_loss \
               + getattr(self.config, 'action_aware_weight', 0.0) * action_aware_loss

        loss = loss / self.gradient_accumulation_steps

        # Loss clipping（防止 outlier 梯度）
        loss_clip_value = getattr(self.config, 'loss_clip_value', None)
        if loss_clip_value is not None and getattr(self.config, 'loss_clip_enabled', True):
            loss = torch.clamp(loss, max=loss_clip_value)

        # 检查损失是否为 NaN/Inf，如果是则跳过这一步
        if not torch.isfinite(loss):
            if self.config.rank == 0:
                logger.warning(f"[step {self.step}] NaN/Inf loss, skipping")
            loss = torch.zeros(1, device=self.device, requires_grad=True)

        loss.backward()

        return {
            "loss": loss.detach(),
            "video_loss": video_loss.detach(),
            "action_loss": action_loss.detach() if self.distill_action else action_loss,
            "action_aware_loss": action_aware_loss.detach() if self.action_aware else action_aware_loss,
            "gt_regression_loss": gt_regression_loss.detach() if self.distill_action else gt_regression_loss,
            "should_sync": should_sync,
        }

    # ==================================================================
    # On-Policy Rollout + DMD 训练步（Phase 5: On-Policy DMD）
    # ==================================================================

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

        # 随机采样推理步数
        if num_steps is None:
            step_min = getattr(self.config, 'dmd_rollout_steps_min', 2)
            step_max = getattr(self.config, 'dmd_rollout_steps_max', 8)
            num_steps = torch.randint(step_min, step_max + 1, (1,)).item()

        if cfg_scale is None:
            cfg_scale = getattr(self.config, 'dmd_cfg_scale', 5.0)

        # 从纯噪声出发
        # 当 retain_grad=True 时，需要从一开始就保留梯度
        current_action = torch.randn_like(batch['actions'])
        if retain_grad:
            current_action.requires_grad_(True)

        # 生成时间步序列
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        action_sigmas = self.train_scheduler_action.apply_shift(sigmas) * self.config.num_train_timesteps

        # 准备空文本嵌入（用于 CFG 无条件推理）
        empty_emb = self.empty_emb.expand(B, -1, -1)

        # 获取动作帧数
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
        with grad_context:
            for i in range(num_steps):
                t_action = action_sigmas[i]
                r_action = action_sigmas[i + 1]

                # 构建 input_dict（视频用 GT latent 作为条件，不参与去噪）
                # 动作用当前去噪状态
                # t_action/r_action 是标量张量，需要先 unsqueeze 再 expand 到 [B, T]
                action_t_expanded = t_action.unsqueeze(0).expand(B, action_num_frames)  # [B, T]
                action_r_expanded = r_action.unsqueeze(0).expand(B, action_num_frames)  # [B, T]

                rollout_input = {
                    'latent_dict': {
                        'noisy_latents': batch['latents'],
                        'latent': batch['latents'],
                        'timesteps': torch.zeros_like(action_t_expanded),  # 视频时间步为 0（GT 条件）
                        'cond_timesteps': torch.zeros_like(action_t_expanded),
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

                # 学生有条件前向
                _, v_action_cond = self.student(
                    rollout_input, train_mode=True,
                    r_timestep=action_r_expanded,
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
                    _, v_action_uncond = self.student(
                        rollout_input_uncond, train_mode=True,
                        r_timestep=action_r_expanded,
                        action_r_timestep=action_r_expanded,
                    )
                    v_action = v_action_uncond + cfg_scale * (v_action_cond - v_action_uncond)
                else:
                    v_action = v_action_cond

                # Euler 更新（仅更新动作）
                # v_action 形状: [B, F*N, C]，需要 reshape 为 [B, C, F, N, 1]
                v_action_5d = self._extract_action_v(v_action, batch['actions'].shape[2])
                current_action = current_action - (t_action - r_action) * v_action_5d

        # 如果需要保留梯度图（用于 DMD 梯度计算）
        if retain_grad:
            current_action.requires_grad_(True)

        return current_action

    def _dmd_train_step(self, batch):
        """
        DMD 训练步：更新判别器 + 计算 DMD 梯度。

        优化版本：只做一次 rollout，同时用于判别器训练和 DMD 梯度计算。
        原版本需要两次 rollout（一次无梯度、一次有梯度），开销翻倍。

        流程：
          1. On-policy rollout（保留计算图）：用学生生成假动作
          2. 判别器更新：真假样本二分类（使用 detach 后的假动作）
          3. DMD 梯度计算：normalize(D(fake) - teacher_score)（复用同一次 rollout）

        参数:
            batch: 数据批次

        返回:
            dmd_grad: DMD 梯度 [B, C, F, N, 1]（注入学生 action loss 用）
            d_loss: 判别器 loss（标量，用于日志）
        """
        from discriminator import train_discriminator_step, compute_dmd_gradient

        # 1. On-policy rollout（保留计算图，用于后续 DMD 梯度计算）
        fake_action = self._on_policy_rollout(batch, retain_grad=True)

        # 2. 判别器更新（使用 detach 后的假动作，不回传梯度到学生）
        d_loss = train_discriminator_step(
            self.discriminator,
            real_actions=batch['actions'],
            fake_actions=fake_action.detach(),  # detach 切断到学生模型的梯度
            video_latent=batch['latents'],
            text_emb=batch['text_emb'],
        )
        d_loss.backward()
        self.discriminator_optimizer.step()
        self.discriminator_optimizer.zero_grad()

        # 3. DMD 梯度计算（复用同一次 rollout，无需重新 rollout）
        dmd_grad = compute_dmd_gradient(
            self.discriminator,
            fake_actions=fake_action,  # 使用保留计算图的 fake_action
            video_latent=batch['latents'],
            text_emb=batch['text_emb'],
            teacher_score=0.0,
        )

        return dmd_grad, d_loss.detach()
