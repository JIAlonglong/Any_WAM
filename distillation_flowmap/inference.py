"""
Flow Map 蒸馏的推理模块。

支持 2~50 步的灵活推理，基于 AnyFlow 的流映射方法。
核心思想：通过 SNR shift 生成非均匀时间步序列，利用模型的 flow map 能力
在任意时间步对之间进行去噪，从而实现少步高质量生成。

时间步语义：
  - sigma = 1.0 对应纯噪声
  - sigma = 0.0 对应干净数据
  - 推理从 sigma=1.0 逐步推进到 sigma=0.0

Flow Map 更新公式：
  x_r = x_t - (t - r) * v(x_t, t, r)
  其中 v 是模型预测的速度场（velocity field）

参考实现：
  - FlowMatchScheduler: wan_va/utils/scheduler.py
  - FlowMapDiscreteScheduler: AnyFlow/far/schedulers/scheduling_flowmap_euler_discrete.py
  - AnyFlow training_rollout: AnyFlow/far/pipelines/pipeline_wan_anyflow.py
"""

import torch
import torch.nn as nn
from einops import rearrange

# 从 wan_va 工具模块导入辅助函数
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wan_va"))
from utils.utils import data_seq_to_patch, get_mesh_id


def apply_shift(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    """对 sigma 序列应用 SNR 偏移变换。

    SNR shift 的作用是重新分配时间步的密度分布：
      - shift > 1 时，高噪声区域（sigma 接近 1）被压缩，
        低噪声区域（sigma 接近 0）被展开，分配更多推理步数
      - shift = 1 时，不做任何变换（恒等映射）
      - shift < 1 时，效果相反

    数学公式：
      shifted_sigma = shift * sigma / (1 + (shift - 1) * sigma)

    这个公式来源于 FlowMatch 的 SNR 重新参数化：
      原始 SNR = (1-sigma)/sigma
      偏移后 SNR = shift * (1-sigma)/sigma
      反解得到上述公式

    参数:
        sigmas: 原始 sigma 序列，任意形状
        shift:  SNR 偏移系数（> 0）

    返回:
        偏移后的 sigma 序列，形状与输入相同
    """
    if shift == 1.0:
        # shift=1 时为恒等变换，直接返回
        return sigmas
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


def flowmap_inference(
    model,
    noisy_latent,
    noisy_action,
    text_emb,
    empty_emb,
    num_steps=2,
    cfg_scale=5.0,
    num_train_timesteps=1000,
    snr_shift=5.0,
    action_snr_shift=1.0,
    patch_size=(1, 2, 2),
):
    """Flow Map 推理：支持任意步数的去噪生成。

    推理流程：
      1. 生成从 sigma=1.0（纯噪声）到 sigma=0.0（干净数据）的时间步序列
      2. 对视频和动作分别应用各自的 SNR shift
      3. 转换为训练时间步（乘以 num_train_timesteps）
      4. 逐步去噪：每步执行条件预测 + 无条件预测 + CFG 组合 + Euler 更新

    参数:
        model:              带有 flowmap 能力的学生模型（已经过 patch_model_forward 处理）
        noisy_latent:       初始噪声 latent [B, C, F, H, W]，F=帧数
        noisy_action:       初始噪声 action [B, C, F, N, 1]，N=每帧动作步数
        text_emb:           文本嵌入 [B, L, D]，L=序列长度，D=嵌入维度
        empty_emb:          空文本嵌入 [B, L, D]（用于 CFG 无条件推理）
        num_steps:          推理步数（2/4/8/16/50 等）
        cfg_scale:          CFG（Classifier-Free Guidance）引导强度
        num_train_timesteps: 训练时间步总数（默认 1000）
        snr_shift:          视频的 SNR 偏移系数（默认 5.0，与训练配置一致）
        action_snr_shift:   动作的 SNR 偏移系数（默认 1.0，与训练配置一致）
        patch_size:         视频 patch 大小 (p_t, p_h, p_w)，默认 (1, 2, 2)

    返回:
        denoised_latent:    去噪后的视频 latent [B, C, F, H, W]
        denoised_action:    去噪后的动作 [B, C, F, N, 1]
    """
    device = noisy_latent.device
    dtype = noisy_latent.dtype
    B = noisy_latent.shape[0]
    num_frames = noisy_latent.shape[2]

    # ================================================================
    # 步骤 1: 生成时间步序列
    # ================================================================
    # 从 1.0（纯噪声）到 0.0（干净数据）均匀采样 num_steps + 1 个点
    # 例如 num_steps=2 时：[1.0, 0.5, 0.0]，共 3 个点，2 个去噪区间
    # 例如 num_steps=50 时：[1.0, 0.98, ..., 0.02, 0.0]，共 51 个点，50 个去噪区间
    sigmas_raw = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64, device=device)

    # ================================================================
    # 步骤 2: 对视频和动作分别应用 SNR shift
    # ================================================================
    # 视频使用 snr_shift（默认 5.0），压缩高噪声区域，展开低噪声区域
    video_sigmas = apply_shift(sigmas_raw, snr_shift)
    # 动作使用 action_snr_shift（默认 1.0，即不偏移）
    action_sigmas = apply_shift(sigmas_raw, action_snr_shift)

    # ================================================================
    # 步骤 3: 转换为训练时间步
    # ================================================================
    # 将 sigma（0~1 范围）转换为训练时间步（0~1000 范围）
    # 例如 sigma=0.5 → timestep=500
    video_timesteps = video_sigmas * num_train_timesteps    # [num_steps + 1]
    action_timesteps = action_sigmas * num_train_timesteps  # [num_steps + 1]

    # ================================================================
    # 步骤 4: 初始化当前 latent 和 action
    # ================================================================
    # 从纯噪声开始
    current_latent = noisy_latent.clone()
    current_action = noisy_action.clone()

    # ================================================================
    # 步骤 5: 逐步去噪推理
    # ================================================================
    # 遍历每一对相邻时间步 (t_i, t_{i+1})
    # t_i 是当前噪声水平，t_{i+1} 是目标噪声水平（更低）
    for i in range(num_steps):
        # 当前时间步 t 和目标时间步 r
        t_video = video_timesteps[i]
        r_video = video_timesteps[i + 1]
        t_action = action_timesteps[i]
        r_action = action_timesteps[i + 1]

        # 如果 t == r，跳过（不会发生，但作为安全检查）
        if t_video == r_video and t_action == r_action:
            continue

        # ------------------------------------------------------------
        # 步骤 5a: 条件预测（使用真实文本嵌入）
        # ------------------------------------------------------------
        # 构建 input_dict，与 forward_train 的接口对齐
        # 时间步需要扩展为 [B, T] 形状，T 为帧数
        input_dict_cond = {
            'latent_dict': {
                'noisy_latents': current_latent.to(dtype),
                'latent': current_latent.to(dtype),  # 条件 latent（推理时不使用条件分支）
                'text_emb': text_emb,
                'timesteps': torch.full((B, num_frames), t_video, device=device, dtype=dtype),
                'cond_timesteps': torch.full((B, num_frames), r_video, device=device, dtype=dtype),
                'grid_id': _make_grid_id(current_latent, model, device),
            },
            'action_dict': {
                'noisy_latents': current_action.to(dtype),
                'latent': current_action.to(dtype),
                'text_emb': text_emb,
                'timesteps': torch.full((B, num_frames), t_action, device=device, dtype=dtype),
                'cond_timesteps': torch.full((B, num_frames), r_action, device=device, dtype=dtype),
                'grid_id': _make_action_grid_id(current_action, device),
            },
            'chunk_size': getattr(model, '_chunk_size', 2),
            'window_size': getattr(model, '_window_size', 72),
        }

        # 条件前向推理（不使用 CFG）
        # r_timestep 和 action_r_timestep 分别传递视频和动作的目标时间步
        # 因为视频和动作使用不同的 SNR shift，所以 r 值不同
        video_v_cond, action_v_cond = model(
            input_dict_cond,
            train_mode=True,
            r_timestep=torch.full((B, num_frames), r_video, device=device, dtype=dtype),
            action_r_timestep=torch.full((B, num_frames), r_action, device=device, dtype=dtype),
        )

        # 将扁平输出转换为 5D 张量
        video_v_cond_5d = _extract_video_v(video_v_cond, current_latent.shape, B, patch_size)
        action_v_cond_5d = _extract_action_v(action_v_cond, num_frames)

        # ------------------------------------------------------------
        # 步骤 5b: 无条件预测（使用空文本嵌入）
        # ------------------------------------------------------------
        input_dict_uncond = {
            'latent_dict': {
                **input_dict_cond['latent_dict'],
                'text_emb': empty_emb,  # 替换为空文本嵌入
            },
            'action_dict': {
                **input_dict_cond['action_dict'],
                'text_emb': empty_emb,  # 替换为空文本嵌入
            },
            'chunk_size': input_dict_cond['chunk_size'],
            'window_size': input_dict_cond['window_size'],
        }

        # 无条件推理同样需要传递 action_r_timestep
        video_v_uncond, _ = model(
            input_dict_uncond,
            train_mode=True,
            r_timestep=torch.full((B, num_frames), r_video, device=device, dtype=dtype),
            action_r_timestep=torch.full((B, num_frames), r_action, device=device, dtype=dtype),
        )
        video_v_uncond_5d = _extract_video_v(video_v_uncond, current_latent.shape, B, patch_size)

        # ------------------------------------------------------------
        # 步骤 5c: CFG 组合
        # ------------------------------------------------------------
        # 标准 CFG 公式：v = v_uncond + cfg_scale * (v_cond - v_uncond)
        # cfg_scale=1 时退化为纯条件预测
        # cfg_scale 越大，条件引导越强，但可能引入伪影
        video_v_cfg = video_v_uncond_5d + cfg_scale * (video_v_cond_5d - video_v_uncond_5d)
        # 动作不使用 CFG（action_guidance_scale=1），直接使用条件预测
        action_v_cfg = action_v_cond_5d

        # ------------------------------------------------------------
        # 步骤 5d: Flow Map Euler 更新
        # ------------------------------------------------------------
        # 核心公式：x_r = x_t - (t - r) * v
        # 其中 t 和 r 是归一化后的 sigma 值（0~1 范围）
        # 注意：这里使用 sigma 而非训练时间步，因为 FlowMatch 的更新公式
        #       基于归一化的 sigma 空间
        sigma_t_video = video_sigmas[i]
        sigma_r_video = video_sigmas[i + 1]
        sigma_t_action = action_sigmas[i]
        sigma_r_action = action_sigmas[i + 1]

        # 视频 Euler 步
        current_latent = current_latent - (sigma_t_video - sigma_r_video) * video_v_cfg
        # 动作 Euler 步
        current_action = current_action - (sigma_t_action - sigma_r_action) * action_v_cfg

    # ================================================================
    # 步骤 6: 返回去噪结果
    # ================================================================
    return current_latent, current_action


def _extract_video_v(video_pred, ref_shape, batch_size, patch_size=(1, 2, 2)):
    """将模型输出的扁平视频序列转换为 5D 视频张量。

    模型的 forward_train 输出扁平的 patch 序列 [B, seq_len, C*p_t*p_h*p_w]，
    需要通过 data_seq_to_patch 重排为 [B, C, F, H, W]。

    参数:
        video_pred: 模型输出 [B, seq_len, C*p_t*p_h*p_w]
        ref_shape:  参考形状 [B, C, F, H, W]，用于确定时空维度
        batch_size: 批次大小
        patch_size: 视频 patch 大小 (p_t, p_h, p_w)，默认 (1, 2, 2)

    返回:
        5D 视频张量 [B, C, F, H, W]
    """
    # 从参考形状中获取维度信息
    latent_num_frames = ref_shape[2]
    latent_height = ref_shape[3]
    latent_width = ref_shape[4]

    # 使用 data_seq_to_patch 进行重排（与 step.py 中的 _extract_video_v 一致）
    return data_seq_to_patch(
        patch_size=patch_size,
        data_seq=video_pred,
        latent_num_frames=latent_num_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        batch_size=batch_size,
    )


def _extract_action_v(action_pred, num_frames):
    """将模型输出的扁平动作序列转换为 5D 动作张量。

    模型输出的是扁平的动作 token [B, F*N, C]，
    需要重排为 [B, C, F, N, 1] 以匹配动作的标准格式。

    参数:
        action_pred: 模型输出 [B, F*N, C]
        num_frames:  帧数 F

    返回:
        5D 动作张量 [B, C, F, N, 1]
    """
    return rearrange(action_pred, 'b (f n) c -> b c f n 1', f=num_frames)


def _make_grid_id(latent, model, device):
    """为视频 latent 构造网格位置 ID。

    使用 get_mesh_id 工具函数生成 RoPE 所需的位置编码网格。
    get_mesh_id 返回 [4, L] 的张量，其中 4 代表 (f_idx, h_idx, w_idx, t)，
    L 是 patch 后的序列长度。

    在 forward_train 中，grid_id 会被 permute 为 [B, L, 4] 格式。

    参数:
        latent: 视频 latent [B, C, F, H, W]
        model:  模型实例（用于获取 patch_size 等配置）
        device: 目标设备

    返回:
        grid_id: 网格位置 ID [B, L, 4]
    """
    B, C, F, H, W = latent.shape
    # 获取 patch 大小（视频模式）
    patch_f, patch_h, patch_w = getattr(model, 'patch_size', (1, 2, 2))

    # 计算 patch 后的维度
    post_f = F // patch_f
    post_h = H // patch_h
    post_w = W // patch_w

    # 使用 get_mesh_id 生成网格（t=0 表示视频模式）
    # get_mesh_id 返回 [4, L]，其中 L = post_f * post_h * post_w
    latent_grid_id = get_mesh_id(
        post_f, post_h, post_w,
        t=0,          # 视频模式
        f_w=1, f_shift=0,
        action=False,  # 非动作模式
    ).to(device)

    # 扩展到 batch 维度：[B, L, 4]
    latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

    return latent_grid_id.to(dtype=latent.dtype)


def _make_action_grid_id(action, device):
    """为动作构造网格位置 ID。

    使用 get_mesh_id 工具函数，以 action=True 模式生成动作的网格位置 ID。
    动作模式下，patch_size 为 (1, 1, 1)，所以 post_f=F, post_h=N, post_w=1。

    参数:
        action: 动作张量 [B, C, F, N, 1]
        device: 目标设备

    返回:
        grid_id: 网格位置 ID [B, F*N, 4]
    """
    B, C, F, N, _ = action.shape

    # 使用 get_mesh_id 生成网格（t=1 表示动作模式）
    # 动作模式下 patch_size 为 (1, 1, 1)，所以 post_f=F, post_h=N, post_w=1
    action_grid_id = get_mesh_id(
        F, N, 1,
        t=1,          # 动作模式
        f_w=1, f_shift=0,
        action=True,  # 动作模式：会应用 ff_offset 和 hh/ww 置 -1
    ).to(device)

    # 扩展到 batch 维度：[B, L, 4]
    action_grid_id = action_grid_id[None].repeat(B, 1, 1)

    return action_grid_id.to(dtype=action.dtype)


# ================================================================
# 便捷函数：从配置创建推理参数
# ================================================================
def create_inference_kwargs(config):
    """从 FlowMap 配置中提取推理所需的默认参数。

    参数:
        config: FlowMap 配置对象（distillation_flowmap.config.cfg）

    返回:
        dict: 包含推理参数的字典，可直接解包传入 flowmap_inference
    """
    return {
        'num_train_timesteps': config.num_train_timesteps,
        'snr_shift': config.snr_shift,
        'action_snr_shift': config.action_snr_shift,
    }
