# Copyright 2024-2025 The Flash-WAM Team Authors. All rights reserved.
# Flow Map 蒸馏的模型改造模块
# 参考 AnyFlow 的实现，适配 Flash-WAM 的 WanTimeTextImageEmbedding 接口

import copy
import math
from functools import wraps
from typing import Optional

import einops
import torch
import torch.nn as nn
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)


class WanTwoTimeTextImageEmbedding(nn.Module):
    """双时间步嵌入模块，支持 Flow Map 蒸馏。

    在原始 WanTimeTextImageEmbedding 的基础上，新增一个 delta_embedder，
    用于编码 delta 时间步（可以是 r 或 t-r）。通过门控机制融合两个时间步嵌入：
        rt_emb = (1 - gate) * temb + gate * delta_emb

    Args:
        dim: 嵌入维度（等于 transformer 的 inner_dim）
        gate_value: 门控初始值，控制 delta_embedder 的融合比例
        deltatime_type: delta 时间步类型，支持 'r' 和 't-r'
        time_freq_dim: 时间步频率编码维度
        time_proj_dim: 时间步投影维度（通常为 inner_dim * 6）
        text_embed_dim: 文本嵌入维度
    """

    def __init__(
        self,
        dim: int,
        gate_value: float,
        deltatime_type: str,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        # 时间步频率编码器（正弦/余弦编码）
        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        # 主时间步嵌入器
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim
        )
        # delta 时间步嵌入器（用于 Flow Map 的参考时间步）
        self.delta_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim
        )
        # 激活函数和时间步投影层
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        # 文本嵌入投影层
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh"
        )
        # 图像嵌入投影层（用于 image-to-video 模式）
        # 当 image_embed_dim 不为 None 时创建，否则设为 None
        self.image_embedder = (
            PixArtAlphaTextProjection(
                image_embed_dim, dim, act_fn="gelu_tanh"
            )
            if image_embed_dim is not None
            else None
        )

        # 门控参数（不可训练，由 buffer 注册）
        self.register_buffer(
            "delta_emb_gate",
            torch.tensor([gate_value], dtype=torch.float32),
            persistent=False,
        )
        # delta 时间步类型：'r' 表示直接使用 r_timestep，'t-r' 表示使用 timestep - r_timestep
        self.deltatime_type = deltatime_type

    def forward(
        self,
        timestep: torch.Tensor,
        dtype=None,
        r_timestep: Optional[torch.Tensor] = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        """前向传播，兼容 Flash-WAM 的调用方式。

        当 r_timestep 为 None 时，退化为原始 WanTimeTextImageEmbedding 的行为。

        Args:
            timestep: 时间步张量，形状 [B, L]
            dtype: 输出数据类型
            r_timestep: 参考时间步张量，形状 [B, L]，用于 Flow Map 蒸馏
            encoder_hidden_states_image: 图像编码器隐藏状态，用于 image-to-video 模式

        Returns:
            temb: 时间步嵌入，形状 [B, L, dim]
            timestep_proj: 时间步投影，形状 [B, L, time_proj_dim]
        """
        B, L = timestep.shape

        # 将时间步展平为一维进行编码
        timestep_flat = timestep.reshape(-1)

        # 频率编码
        timestep_proj_input = self.timesteps_proj(timestep_flat)

        # 确保数据类型与 time_embedder 参数一致
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep_proj_input.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep_proj_input = timestep_proj_input.to(time_embedder_dtype)

        # 计算主时间步嵌入
        temb = self.time_embedder(timestep_proj_input).to(dtype=dtype)

        if r_timestep is not None:
            # --- Flow Map 模式：融合双时间步嵌入 ---
            # 计算 delta 时间步
            if self.deltatime_type == "r":
                delta_timestep = r_timestep.reshape(-1)
            elif self.deltatime_type == "t-r":
                delta_timestep = (timestep - r_timestep).reshape(-1)
            else:
                raise NotImplementedError(
                    f"不支持的 deltatime_type: {self.deltatime_type}，仅支持 'r' 和 't-r'"
                )

            # delta 时间步频率编码
            delta_proj_input = self.timesteps_proj(delta_timestep)
            delta_embedder_dtype = self.delta_embedder.linear_1.weight.dtype
            if delta_proj_input.dtype != delta_embedder_dtype and delta_embedder_dtype != torch.int8:
                delta_proj_input = delta_proj_input.to(delta_embedder_dtype)

            # 计算 delta 嵌入
            delta_emb = self.delta_embedder(delta_proj_input).to(dtype=dtype)

            # 门控融合：rt_emb = (1 - gate) * temb + gate * delta_emb
            gate = self.delta_emb_gate.to(dtype)
            temb = (1 - gate) * temb + gate * delta_emb

        # 计算时间步投影
        timestep_proj = self.time_proj(self.act_fn(temb))

        # 图像嵌入：如果提供了 encoder_hidden_states_image 且 image_embedder 存在，
        # 将图像嵌入加到时间步嵌入上（与 text_embedder 的处理方式对齐）
        if encoder_hidden_states_image is not None and self.image_embedder is not None:
            image_emb = self.image_embedder(encoder_hidden_states_image).to(dtype=dtype)
            temb = temb + image_emb

        return temb.reshape(B, L, -1), timestep_proj.reshape(B, L, -1)


def setup_flowmap_model(model, gate_value=0.0, deltatime_type="r"):
    """将模型的 condition_embedder 替换为支持 Flow Map 的双时间步版本。

    该函数不会修改 wan_va/modules/model.py 源文件，而是在运行时动态替换模块。

    Args:
        model: WanTransformer3DModel 实例
        gate_value: 门控初始值，默认 0.0（完全使用原始时间步嵌入）
        deltatime_type: delta 时间步类型，支持 'r' 和 't-r'

    Returns:
        修改后的模型（原地修改，同时返回模型以支持链式调用）
    """
    inner_dim = model.num_attention_heads * model.attention_head_dim

    # 从模型配置中获取频率维度和文本嵌入维度
    freq_dim = model.condition_embedder.timesteps_proj.num_channels
    # PixArtAlphaTextProjection 的第一个 Linear 层的 in_features 即为文本嵌入维度
    text_embed_dim = model.condition_embedder.text_embedder.linear_1.in_features

    # 获取图像嵌入维度（如果原模型有 image_embedder）
    image_embed_dim = None
    if (
        hasattr(model.condition_embedder, "image_embedder")
        and model.condition_embedder.image_embedder is not None
    ):
        image_embed_dim = model.condition_embedder.image_embedder.linear_1.in_features

    # 创建新的双时间步嵌入模块
    new_condition_embedder = WanTwoTimeTextImageEmbedding(
        dim=inner_dim,
        gate_value=gate_value,
        deltatime_type=deltatime_type,
        time_freq_dim=freq_dim,
        time_proj_dim=inner_dim * 6,
        text_embed_dim=text_embed_dim,
        image_embed_dim=image_embed_dim,
    )

    # 从原模型的 condition_embedder 深拷贝权重
    new_condition_embedder.time_embedder = copy.deepcopy(
        model.condition_embedder.time_embedder
    )
    # delta_embedder 从 time_embedder 深拷贝初始化（共享初始权重）
    new_condition_embedder.delta_embedder = copy.deepcopy(
        model.condition_embedder.time_embedder
    )
    new_condition_embedder.time_proj = copy.deepcopy(
        model.condition_embedder.time_proj
    )
    new_condition_embedder.text_embedder = copy.deepcopy(
        model.condition_embedder.text_embedder
    )
    # 深拷贝 image_embedder（如果原模型存在）
    if (
        hasattr(model.condition_embedder, "image_embedder")
        and model.condition_embedder.image_embedder is not None
    ):
        new_condition_embedder.image_embedder = copy.deepcopy(
            model.condition_embedder.image_embedder
        )

    # 替换原模型的 condition_embedder
    del model.condition_embedder
    model.condition_embedder = new_condition_embedder

    # 同样替换 condition_embedder_action（如果存在）
    if hasattr(model, "condition_embedder_action"):
        # 获取 action 侧的图像嵌入维度
        action_image_embed_dim = None
        if (
            hasattr(model.condition_embedder_action, "image_embedder")
            and model.condition_embedder_action.image_embedder is not None
        ):
            action_image_embed_dim = model.condition_embedder_action.image_embedder.linear_1.in_features

        new_condition_embedder_action = WanTwoTimeTextImageEmbedding(
            dim=inner_dim,
            gate_value=gate_value,
            deltatime_type=deltatime_type,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_embed_dim,
            image_embed_dim=action_image_embed_dim,
        )
        new_condition_embedder_action.time_embedder = copy.deepcopy(
            model.condition_embedder_action.time_embedder
        )
        new_condition_embedder_action.delta_embedder = copy.deepcopy(
            model.condition_embedder_action.time_embedder
        )
        new_condition_embedder_action.time_proj = copy.deepcopy(
            model.condition_embedder_action.time_proj
        )
        new_condition_embedder_action.text_embedder = copy.deepcopy(
            model.condition_embedder_action.text_embedder
        )
        # 深拷贝 action 侧的 image_embedder（如果原模型存在）
        if (
            hasattr(model.condition_embedder_action, "image_embedder")
            and model.condition_embedder_action.image_embedder is not None
        ):
            new_condition_embedder_action.image_embedder = copy.deepcopy(
                model.condition_embedder_action.image_embedder
            )
        del model.condition_embedder_action
        model.condition_embedder_action = new_condition_embedder_action

    return model


def patch_model_forward(model):
    """给模型的 forward 和 _time_embed 方法添加 r_timestep 参数支持。

    当 r_timestep=None 时退化为原始行为（完全兼容）。
    使用 functools.wraps 保留原函数的元信息。

    Args:
        model: WanTransformer3DModel 实例

    Returns:
        修改后的模型（原地修改，同时返回模型以支持链式调用）

    Note:
        该函数修改 _time_embed 和 forward_train 方法，使其接受可选的 r_timestep 参数，
        并将其传递给 condition_embedder 的 forward 方法。
    """
    # 保存原始方法，便于 unpatch_model_forward 恢复
    original_time_embed = model._time_embed
    original_forward_train = model.forward_train
    original_forward = model.forward

    @wraps(original_time_embed)
    def patched_time_embed(timesteps, H, W, dtype, action_mode=False, r_timestep=None):
        """为 _time_embed 添加 r_timestep 支持。

        当 r_timestep 不为 None 时，将其传递给 condition_embedder 以启用 Flow Map 蒸馏。
        """
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            model.patch_size[1], model.patch_size[2])
        latent_time_steps = torch.repeat_interleave(
            timesteps,
            (H // pach_scale_h) * (W // pach_scale_w),
            dim=1,
        )
        current_condition_embedder = (
            model.condition_embedder_action if action_mode else model.condition_embedder
        )

        # 如果提供了 r_timestep，同样需要对齐维度
        r_latent_time_steps = None
        if r_timestep is not None:
            r_latent_time_steps = torch.repeat_interleave(
                r_timestep,
                (H // pach_scale_h) * (W // pach_scale_w),
                dim=1,
            )

        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=dtype, r_timestep=r_latent_time_steps
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C
        return temb, timestep_proj

    # 替换 _time_embed 方法
    model._time_embed = patched_time_embed

    @wraps(original_forward_train)
    def patched_forward_train(input_dict, fdm=False, r_timestep=None, action_r_timestep=None):
        """为 forward_train 添加 r_timestep 和 action_r_timestep 支持。

        当 r_timestep 不为 None 时，将其传递给 latent 的 _time_embed 以启用 Flow Map 蒸馏。
        当 action_r_timestep 不为 None 时，将其用于 action 的 _time_embed；
        否则 action 也使用 r_timestep（向后兼容）。

        Args:
            input_dict: 输入字典，包含 latent_dict 和 action_dict
            fdm: 是否启用 Forward Dynamics Model 模式
            r_timestep: latent 的参考时间步，形状 [B, T]
            action_r_timestep: action 的独立参考时间步，形状 [B, T]，
                为 None 时退化为使用 r_timestep
        """
        # 转换 dtype 到局部变量，避免原地修改调用者的 input_dict
        latent_dict = {
            **input_dict["latent_dict"],
            "noisy_latents": input_dict["latent_dict"]["noisy_latents"].to(torch.bfloat16),
            "latent": input_dict["latent_dict"]["latent"].to(torch.bfloat16),
        }
        action_dict = {
            **input_dict["action_dict"],
            "noisy_latents": input_dict["action_dict"]["noisy_latents"].to(torch.bfloat16),
            "latent": input_dict["action_dict"]["latent"].to(torch.bfloat16),
        }
        batch_size = latent_dict["noisy_latents"].shape[0]

        latent_hidden_states = model._input_embed(
            latent_dict["noisy_latents"], input_type="latent"
        ).flatten(0, 1)[None]
        action_hidden_states = model._input_embed(
            action_dict["noisy_latents"], input_type="action"
        ).flatten(0, 1)[None]
        text_hidden_states = model._input_embed(
            latent_dict["text_emb"], input_type="text"
        )
        text_hidden_states = text_hidden_states.flatten(0, 1)[None]

        condition_latent_hidden_states = model._input_embed(
            latent_dict["latent"], input_type="latent"
        ).flatten(0, 1)[None]
        condition_action_hidden_states = model._input_embed(
            action_dict["latent"], input_type="action"
        ).flatten(0, 1)[None]

        hidden_states = torch.cat(
            [
                latent_hidden_states,
                condition_latent_hidden_states,
                action_hidden_states,
                condition_action_hidden_states,
            ],
            dim=1,
        )

        latent_grid_id = latent_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        action_grid_id = action_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        full_grid_id = torch.cat(
            [latent_grid_id] * 2 + [action_grid_id] * 2, dim=2
        )

        rotary_emb = model.rope(full_grid_id)[:, :, None]

        latent_time_steps = torch.cat(
            [
                latent_dict["timesteps"].flatten(0, 1),
                latent_dict["cond_timesteps"].flatten(0, 1),
            ]
        )[None]
        action_time_steps = torch.cat(
            [
                action_dict["timesteps"].flatten(0, 1),
                action_dict["cond_timesteps"].flatten(0, 1),
            ]
        )[None]

        # latent 使用 r_timestep，action 使用 action_r_timestep（如果提供）否则用 r_timestep
        latent_r_timestep = None
        act_r = action_r_timestep if action_r_timestep is not None else r_timestep
        act_r_timestep_tensor = None
        if r_timestep is not None:
            # r_timestep 形状为 [B, T]，需要与 timesteps 对齐
            latent_r_timestep = torch.cat(
                [r_timestep.flatten(0, 1), r_timestep.flatten(0, 1)]
            )[None]
        if act_r is not None:
            # action 的参考时间步，可能与 latent 不同
            act_r_timestep_tensor = torch.cat(
                [act_r.flatten(0, 1), act_r.flatten(0, 1)]
            )[None]

        latent_temb, latent_timestep_proj = model._time_embed(
            latent_time_steps,
            latent_dict["noisy_latents"].shape[-2],
            latent_dict["noisy_latents"].shape[-1],
            dtype=hidden_states.dtype,
            action_mode=False,
            r_timestep=latent_r_timestep,
        )
        action_temb, action_timestep_proj = model._time_embed(
            action_time_steps,
            action_dict["noisy_latents"].shape[-2],
            action_dict["noisy_latents"].shape[-1],
            dtype=hidden_states.dtype,
            action_mode=True,
            r_timestep=act_r_timestep_tensor,
        )
        temb = torch.cat([latent_temb, action_temb], dim=1)
        timestep_proj = torch.cat(
            [latent_timestep_proj, action_timestep_proj], dim=1
        )

        total_length = hidden_states.shape[1]
        padded_length = (128 - total_length % 128) % 128
        hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = nn.functional.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = nn.functional.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = nn.functional.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

        split_list = [
            latent_hidden_states.shape[1],
            condition_latent_hidden_states.shape[1],
            action_hidden_states.shape[1],
            condition_action_hidden_states.shape[1],
            padded_length,
        ]

        # 延迟导入 FlexAttnFunc 以避免循环依赖
        from modules.model import FlexAttnFunc  # noqa: E402

        FlexAttnFunc.init_mask(
            latent_dict["noisy_latents"].shape,
            action_dict["noisy_latents"].shape,
            padded_length,
            input_dict["chunk_size"],
            window_size=input_dict["window_size"],
            patch_size=model.patch_size,
            device=hidden_states.device,
            fdm=fdm,
        )

        for i, block in enumerate(model.blocks):
            hidden_states = block(
                hidden_states,
                text_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=False,
            )
            if i == 0:
                pass  # Removed debug print

        temb_scale_shift_table = model.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = einops.rearrange(
            temb_scale_shift_table, "b l n c -> b n l c"
        ).chunk(2, dim=1)
        shift = shift.to(hidden_states.device).squeeze(1)
        scale = scale.to(hidden_states.device).squeeze(1)
        hidden_states = (
            model.norm_out(hidden_states.float()) * (1.0 + scale) + shift
        ).type_as(hidden_states)

        latent_hidden_states, _, action_hidden_states, _, _ = torch.split(
            hidden_states, split_list, dim=1
        )
        B_actual = latent_hidden_states.shape[0]
        latent_hidden_states = model.proj_out(latent_hidden_states)
        latent_hidden_states = einops.rearrange(
            latent_hidden_states,
            "b (b1 l) (n c) -> (b b1) l (n c)",
            n=math.prod(model.patch_size),
            b1=B_actual,
        )
        action_hidden_states = model.action_proj_out(action_hidden_states)
        action_hidden_states = einops.rearrange(
            action_hidden_states, "b (b1 l) c -> (b b1) l c", b1=B_actual
        )

        return latent_hidden_states, action_hidden_states

    # 替换 forward_train 方法
    model.forward_train = patched_forward_train

    # 保存原始 forward 方法
    original_forward = model.forward

    @wraps(original_forward)
    def patched_forward(
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        train_mode=False,
        fdm=False,
        r_timestep=None,
        action_r_timestep=None,
    ):
        """为 forward 添加 r_timestep 和 action_r_timestep 支持。

        当 train_mode=True 时，将 r_timestep 和 action_r_timestep 传递给 forward_train。
        非训练模式下这些参数被忽略（推理时不需要）。

        Args:
            action_r_timestep: action 的独立参考时间步，为 None 时 action 与 latent 共享 r_timestep
        """
        if train_mode:
            return model.forward_train(
                input_dict, fdm=fdm, r_timestep=r_timestep,
                action_r_timestep=action_r_timestep,
            )
        # 非训练模式调用原始 forward 逻辑（不需要 r_timestep）
        return original_forward(
            input_dict,
            update_cache=update_cache,
            cache_name=cache_name,
            action_mode=action_mode,
            train_mode=train_mode,
            fdm=fdm,
        )

    # 替换 forward 方法
    model.forward = patched_forward

    # 保存原始方法引用，便于 unpatch_model_forward 恢复
    model._original_methods = {
        '_time_embed': original_time_embed,
        'forward_train': original_forward_train,
        'forward': original_forward,
    }

    return model


def unpatch_model_forward(model):
    """恢复 patch_model_forward 对模型的所有修改。

    Args:
        model: 经过 patch_model_forward 修改的 WanTransformer3DModel 实例
    """
    if not hasattr(model, '_original_methods'):
        return
    for name, func in model._original_methods.items():
        setattr(model, name, func)
    del model._original_methods
