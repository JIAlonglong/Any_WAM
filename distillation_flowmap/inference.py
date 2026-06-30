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
    action_num_steps=None,
    cfg_scale=5.0,
    num_train_timesteps=1000,
    snr_shift=5.0,
    action_snr_shift=1.0,
    action_downsample_factor=1,
    patch_size=(1, 2, 2),
    init_latent=None,
    action_mask=None,
    update_cache=0,
    cache_name="pos",
    frame_st_id=0,
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
        num_steps:          视频推理步数（2/4/8/16/20/50 等）
        action_num_steps:   动作推理步数（默认与 num_steps 相同）。视频/动作各自
                            使用独立的时间步网格，较早完成的分支会冻结在 t=r=0。
        cfg_scale:          CFG（Classifier-Free Guidance）引导强度
        num_train_timesteps: 训练时间步总数（默认 1000）
        snr_shift:          视频的 SNR 偏移系数（默认 5.0，与训练配置一致）
        action_snr_shift:   动作的 SNR 偏移系数（默认 1.0，与训练配置一致）
        patch_size:         视频 patch 大小 (p_t, p_h, p_w)，默认 (1, 2, 2)
        init_latent:        首帧条件 latent [B, C, 1, H, W]，为 None 时不注入首帧条件。
                            注入首帧条件后，第 0 帧始终保持在干净的 GT 状态，
                            作为其他帧去噪的空间-时间锚点。
        action_mask:        可选动作通道 mask [C]。未使用通道会在每次 forward 前置零，
                            与 teacher 推理的 padding-channel 处理保持一致。
        update_cache:       KV cache 模式 (0=只读, 1=追加, 2=覆盖)。去噪完成后会
                            用 t=r=0 的 clean latent/action 额外写入一次 cache。
        cache_name:         KV cache 名称，用于跨 chunk 缓存注意力状态。

    返回:
        denoised_latent:    去噪后的视频 latent [B, C, F, H, W]
        denoised_action:    去噪后的动作 [B, C, F, N, 1]
    """
    device = noisy_latent.device
    dtype = noisy_latent.dtype
    timestep_dtype = torch.float32
    B = noisy_latent.shape[0]
    num_frames = noisy_latent.shape[2]
    action_num_frames = noisy_action.shape[2]
    action_ds = max(1, int(action_downsample_factor))
    action_num_frames_ds = max(1, action_num_frames // action_ds)

    # 如果 action_num_steps 未指定，使用与 num_steps 相同的值
    if action_num_steps is None:
        action_num_steps = num_steps

    # ================================================================
    # 步骤 1: 生成时间步序列
    # ================================================================
    # Use each branch's own schedule. The loop still runs max_steps times so
    # video/action can be forwarded jointly, but finished branches are frozen.
    max_steps = max(num_steps, action_num_steps)
    video_sigmas_raw = torch.linspace(
        1.0, 0.0, num_steps + 1, dtype=torch.float64, device=device)
    action_sigmas_raw = torch.linspace(
        1.0, 0.0, action_num_steps + 1, dtype=torch.float64, device=device)

    # ================================================================
    # 步骤 2: 对视频和动作分别应用 SNR shift
    # ================================================================
    # 视频使用 snr_shift（默认 5.0），压缩高噪声区域，展开低噪声区域
    video_sigmas = apply_shift(video_sigmas_raw, snr_shift)
    # 动作使用 action_snr_shift（默认 1.0，即不偏移）
    action_sigmas = apply_shift(action_sigmas_raw, action_snr_shift)

    # ================================================================
    # 步骤 3: 转换为训练时间步
    # ================================================================
    # 将 sigma（0~1 范围）转换为训练时间步（0~1000 范围）
    # 例如 sigma=0.5 → timestep=500
    video_timesteps = video_sigmas * num_train_timesteps    # [max_steps + 1]
    action_timesteps = action_sigmas * num_train_timesteps  # [max_steps + 1]

    # ================================================================
    # 步骤 4: 初始化当前 latent 和 action
    # ================================================================
    # 从纯噪声开始
    current_latent = noisy_latent.clone()
    current_action = noisy_action.clone()

    was_training = model.training
    model.eval()

    try:
        # ================================================================
        # 步骤 5: 逐步去噪推理
        # ================================================================
        # 使用统一的步数循环，video 和 action 使用相同的步数但不同的时间步
        with torch.no_grad():
            for i in range(max_steps):
                video_active = i < num_steps
                action_active = i < action_num_steps
                if not video_active and not action_active:
                    continue

                # 当前时间步 t 和目标时间步 r. Inactive branches are kept at
                # t=r=0 so their state is available as context but not updated.
                if video_active:
                    t_video = video_timesteps[i]
                    r_video = video_timesteps[i + 1]
                    sigma_t_video = video_sigmas[i]
                    sigma_r_video = video_sigmas[i + 1]
                else:
                    t_video = r_video = video_timesteps[-1]
                    sigma_t_video = sigma_r_video = video_sigmas[-1]

                if action_active:
                    t_action = action_timesteps[i]
                    r_action = action_timesteps[i + 1]
                    sigma_t_action = action_sigmas[i]
                    sigma_r_action = action_sigmas[i + 1]
                else:
                    t_action = r_action = action_timesteps[-1]
                    sigma_t_action = sigma_r_action = action_sigmas[-1]

        # ------------------------------------------------------------
        # 注入首帧条件：将第 0 帧替换为 GT 干净 latent，保持锚点
        # ------------------------------------------------------------
                if init_latent is not None:
                    current_latent[:, :, 0:1] = init_latent[:, :, 0:1].to(current_latent.dtype)
                    current_action[:, :, 0:1] = 0.0
                _zero_invalid_actions_(current_action, action_mask)

        # ------------------------------------------------------------
        # 步骤 5a/5b: 条件/无条件预测
        # ------------------------------------------------------------
        # forward_train 会把 batch 展平成一个长序列。CFG 必须合并成一次
        # forward，这样最后一步写入 KV cache 时 cond/uncond 的历史是一致的。
                use_video_cfg = cfg_scale > 1.0 and empty_emb is not None
                infer_B = B * 2 if use_video_cfg else B
                latent_model_input = (
                    current_latent.repeat(2, 1, 1, 1, 1)
                    if use_video_cfg else current_latent
                )
                current_action_ds = current_action[:, :, ::action_ds]
                action_model_input = (
                    current_action_ds.repeat(2, 1, 1, 1, 1)
                    if use_video_cfg else current_action_ds
                )
                text_model_input = (
                    torch.cat([text_emb, empty_emb], dim=0)
                    if use_video_cfg else text_emb
                )

                video_r_timestep = _make_cond_timesteps(
                    infer_B, num_frames, r_video, init_latent, device, timestep_dtype
                )
                action_r_timestep = _expand_timestep(
                    infer_B, action_num_frames, r_action, device, timestep_dtype
                )[:, ::action_ds]
                action_timestep = _expand_timestep(
                    infer_B, action_num_frames, t_action, device, timestep_dtype
                )[:, ::action_ds]
                input_dict = {
                    'latent_dict': {
                        'noisy_latents': latent_model_input.to(dtype),
                        'latent': latent_model_input.to(dtype),
                        'text_emb': text_model_input,
                        'timesteps': _make_timesteps(
                            infer_B, num_frames, t_video, init_latent, device, timestep_dtype
                        ),
                        'cond_timesteps': video_r_timestep,
                        'grid_id': _make_grid_id(latent_model_input, model, device, frame_st_id),
                    },
                    'action_dict': {
                        'noisy_latents': action_model_input.to(dtype),
                        'latent': action_model_input.to(dtype),
                        'text_emb': text_model_input,
                        'timesteps': action_timestep,
                        'cond_timesteps': action_r_timestep,
                        'grid_id': _make_action_grid_id(action_model_input, device, frame_st_id),
                    },
                    'chunk_size': getattr(model, '_chunk_size', 2),
                    'window_size': getattr(model, '_window_size', 72),
                }

                video_v_all, action_v_all = model(
                    input_dict,
                    train_mode=True,
                    r_timestep=video_r_timestep,
                    action_r_timestep=action_r_timestep,
                    update_cache=0,
                    cache_name=cache_name,
                )

        # 将扁平输出转换为 5D 张量
                video_v_all_5d = _extract_video_v(video_v_all, latent_model_input.shape, infer_B, patch_size)
                action_v_all_5d = _extract_action_v(action_v_all, action_num_frames_ds)
                if use_video_cfg:
                    video_v_cond_5d = video_v_all_5d[:B]
                    video_v_uncond_5d = video_v_all_5d[B:]
                    action_v_cond_5d = action_v_all_5d[:B]
                else:
                    video_v_cond_5d = video_v_all_5d
                    video_v_uncond_5d = video_v_all_5d
                    action_v_cond_5d = action_v_all_5d

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
        # 视频 Euler 步
                if video_active:
                    current_latent = current_latent - (sigma_t_video - sigma_r_video) * video_v_cfg
                # 动作 Euler 步
                if action_active:
                    current_action[:, :, ::action_ds] = (
                        current_action[:, :, ::action_ds]
                        - (sigma_t_action - sigma_r_action) * action_v_cfg
                    )

            if init_latent is not None:
                current_latent[:, :, 0:1] = init_latent[:, :, 0:1].to(current_latent.dtype)
                current_action[:, :, 0:1] = 0.0
            _zero_invalid_actions_(current_action, action_mask)
            if update_cache:
                flowmap_update_cache(
                    model=model,
                    latent=current_latent,
                    action=current_action,
                    text_emb=text_emb,
                    empty_emb=empty_emb,
                    cfg_scale=cfg_scale,
                    action_mask=action_mask,
                    patch_size=patch_size,
                    update_cache=update_cache,
                    cache_name=cache_name,
                    frame_st_id=frame_st_id,
                )
    finally:
        if was_training:
            model.train()

    # ================================================================
    # 步骤 6: 返回去噪结果
    # ================================================================
    return current_latent, current_action


def flowmap_update_cache(
    model,
    latent,
    action,
    text_emb,
    empty_emb=None,
    cfg_scale=5.0,
    action_mask=None,
    patch_size=(1, 2, 2),
    update_cache=1,
    cache_name="pos",
    frame_st_id=0,
):
    """Write clean joint video/action tokens into the streaming KV cache.

    FlowMap denoising predicts x_r from x_t. For streaming inference the cache
    should represent the finalized chunk, matching the original extra t=0
    teacher forward, not the last noisy denoising input.
    """
    if not update_cache:
        return
    _zero_invalid_actions_(action, action_mask)

    device = latent.device
    dtype = latent.dtype
    timestep_dtype = torch.float32
    B = latent.shape[0]
    num_frames = latent.shape[2]
    action_num_frames = action.shape[2]

    use_video_cfg = cfg_scale > 1.0 and empty_emb is not None
    infer_B = B * 2 if use_video_cfg else B
    latent_model_input = latent.repeat(2, 1, 1, 1, 1) if use_video_cfg else latent
    action_model_input = action.repeat(2, 1, 1, 1, 1) if use_video_cfg else action
    text_model_input = (
        torch.cat([text_emb, empty_emb], dim=0) if use_video_cfg else text_emb
    )

    video_zero = _expand_timestep(
        infer_B, num_frames, 0.0, device, timestep_dtype
    )
    action_zero = _expand_timestep(
        infer_B, action_num_frames, 0.0, device, timestep_dtype
    )
    input_dict = {
        'latent_dict': {
            'noisy_latents': latent_model_input.to(dtype),
            'latent': latent_model_input.to(dtype),
            'text_emb': text_model_input,
            'timesteps': video_zero,
            'cond_timesteps': video_zero,
            'grid_id': _make_grid_id(latent_model_input, model, device, frame_st_id),
        },
        'action_dict': {
            'noisy_latents': action_model_input.to(dtype),
            'latent': action_model_input.to(dtype),
            'text_emb': text_model_input,
            'timesteps': action_zero,
            'cond_timesteps': action_zero,
            'grid_id': _make_action_grid_id(action_model_input, device, frame_st_id),
        },
        'chunk_size': getattr(model, '_chunk_size', 2),
        'window_size': getattr(model, '_window_size', 72),
    }
    model(
        input_dict,
        train_mode=True,
        r_timestep=video_zero,
        action_r_timestep=action_zero,
        update_cache=update_cache,
        cache_name=cache_name,
    )


def _zero_invalid_actions_(action, action_mask):
    """Zero padded action channels in-place when a channel mask is provided."""
    if action_mask is None:
        return
    mask = action_mask.to(device=action.device, dtype=torch.bool)
    action[:, ~mask] *= 0


def _make_timesteps(B, num_frames, t_val, init_latent, device, dtype):
    """Create timestep tensor, zeroing frame 0 if init_latent is provided."""
    ts = _expand_timestep(B, num_frames, t_val, device, dtype)
    if init_latent is not None:
        ts[:, 0:1] = 0.0  # Frame 0 is clean (no noise)
    return ts


def _make_cond_timesteps(B, num_frames, r_val, init_latent, device, dtype):
    """Create cond_timestep tensor, zeroing frame 0 if init_latent is provided."""
    ts = _expand_timestep(B, num_frames, r_val, device, dtype)
    if init_latent is not None:
        ts[:, 0:1] = 0.0  # Frame 0 stays clean (no movement)
    return ts


def _expand_timestep(B, num_frames, value, device, dtype):
    """Expand a scalar tensor timestep without casting it to the latent dtype."""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=device)
    return torch.ones((B, num_frames), device=device, dtype=dtype) * value.to(device=device, dtype=dtype)


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


def _make_grid_id(latent, model, device, frame_st_id=0):
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
        f_w=1, f_shift=frame_st_id,
        action=False,  # 非动作模式
    ).to(device)

    # 扩展到 batch 维度：[B, L, 4]
    latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

    return latent_grid_id


def _make_action_grid_id(action, device, frame_st_id=0):
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
        f_w=1, f_shift=frame_st_id,
        action=True,  # 动作模式：会应用 ff_offset 和 hh/ww 置 -1
    ).to(device)

    # 扩展到 batch 维度：[B, L, 4]
    action_grid_id = action_grid_id[None].repeat(B, 1, 1)

    return action_grid_id


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
        'action_downsample_factor': getattr(config, 'action_downsample_factor', 1),
    }
