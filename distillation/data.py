"""
数据批次准备和噪声添加（DataMixin）。

这个文件负责：
  1. 从数据集加载数据批次
  2. 对视频和动作 latent 添加噪声（FlowMatch 的核心操作）
  3. 构建模型输入字典（input_dict）

噪声添加的数学原理（Flow Matching）：
  x_t = (1 - σ) * x_0 + σ * noise
  其中：
    x_0: 干净样本
    noise: 随机高斯噪声
    σ: 噪声水平（从 schedule 中采样）
    x_t: 加噪后的样本

训练目标（v-prediction）：
  target = noise - x_0
  即模型需要预测"噪声方向"
"""

import torch

from utils import sample_timestep_id, get_mesh_id


class DataMixin:
    # ==================================================================
    # 数据迭代器 — 与 wan_va/train.py 完全一致
    # ==================================================================
    def _get_next_batch(self):
        """
        获取下一个数据批次。

        功能：
          - 懒加载数据迭代器
          - 处理数据集遍历完成后的重置
          - 更新分布式采样器的 epoch（确保数据打乱）

        返回:
            包含 latents、actions、text_emb 等字段的数据批次
        """
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # 数据集遍历完成，重置迭代器
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        return batch

    # ==================================================================
    # _add_noise — 与 wan_va/train.py 完全一致，额外返回 timestep_ids
    # ==================================================================
    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False,
                   action_mode=False, noisy_cond_prob=0.):
        """
        对 latent 添加噪声（Flow Matching 的核心操作）。

        参数:
            latent:          干净的 latent 表示 [B, C, F, H, W]
            train_scheduler: FlowMatch 调度器
            action_mask:     动作掩码（用于过滤无效动作位置）
            action_mode:     是否是动作模式（影响 patch 大小）
            noisy_cond_prob: 条件加噪概率（蒸馏时设为 0）

        返回:
            包含以下字段的字典：
              - timesteps:     采样的时间步 [B, F]
              - noisy_latents: 加噪后的 latent [B, C, F, H, W]
              - targets:       训练目标（v-prediction）[B, C, F, H, W]
              - latent:        条件 latent（可能被加噪）
              - cond_timesteps: 条件时间步
              - grid_id:       位置编码网格

        噪声添加过程：
          1. 随机采样时间步 ID（每帧独立采样）
          2. 生成随机高斯噪声
          3. 根据 FlowMatch 公式加噪：x_t = (1-σ)*x_0 + σ*noise
          4. 计算训练目标：target = noise - x_0
        """
        B, C, F, H, W = latent.shape

        # 随机采样时间步 ID（每帧独立采样，实现帧级噪声调度）
        timestep_ids = sample_timestep_id(
            batch_size=F,
            num_train_timesteps=train_scheduler.num_train_timesteps,
        )
        noise = torch.zeros_like(latent).normal_()  # 标准高斯噪声
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        # FlowMatch 加噪：x_t = (1-σ)*x_0 + σ*noise
        noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        # 训练目标：v = noise - x_0
        targets = train_scheduler.training_target(latent, noise, timesteps)

        # 确定 patch 大小（动作模式使用 1x1x1）
        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1

        # 生成位置编码网格
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,
            latent.shape[-2] // patch_h,
            latent.shape[-1] // patch_w,
            t=1 if action_mode else 0,
            f_w=1, f_shift=0,
            action=action_mode,
        ).to(self.device)
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        # 条件加噪（蒸馏时关闭，概率为 0）
        if torch.rand(1).item() < noisy_cond_prob:
            # 对条件 latent 也添加少量噪声（数据增强）
            cond_timestep_ids = sample_timestep_id(
                batch_size=F,
                min_timestep_bd=0.5,
                max_timestep_bd=1.0,
                num_train_timesteps=train_scheduler.num_train_timesteps,
            )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        # 应用动作掩码（过滤无效动作位置）
        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    # ==================================================================
    # _prepare_base_dict — 轻量版，只提取不依赖噪声的字段
    # 用于 FlowMap 蒸馏，避免冗余加噪
    # ==================================================================
    @torch.no_grad()
    def _prepare_base_dict(self, batch_dict):
        """
        准备基础输入字典（不加噪）。

        与 _prepare_input_dict 的区别：
          - 不调用 _add_noise（避免冗余加噪）
          - 只提取 grid_id、cond_timesteps、latent、text_emb 等基础字段
          - noisy_latents 和 targets 由调用者后续填充

        参数:
            batch_dict: 数据批次，包含 latents、actions、text_emb 等

        返回:
            input_dict: 基础输入字典（不含 noisy_latents 和 targets）
        """
        B = batch_dict['latents'].shape[0]

        # 生成位置编码网格（视频）
        patch_f, patch_h, patch_w = self.patch_size
        latent_grid_id = get_mesh_id(
            batch_dict['latents'].shape[-3] // patch_f,
            batch_dict['latents'].shape[-2] // patch_h,
            batch_dict['latents'].shape[-1] // patch_w,
            t=0, f_w=1, f_shift=0, action=False,
        ).to(self.device)
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        # 生成位置编码网格（动作）
        action_grid_id = get_mesh_id(
            batch_dict['actions'].shape[-3],  # action patch = 1x1x1
            batch_dict['actions'].shape[-2],
            batch_dict['actions'].shape[-1],
            t=1, f_w=1, f_shift=0, action=True,
        ).to(self.device)
        action_grid_id = action_grid_id[None].repeat(B, 1, 1)

        # 条件时间步（蒸馏时为零）
        F_video = batch_dict['latents'].shape[2]
        F_action = batch_dict['actions'].shape[2]
        cond_timesteps_video = torch.zeros(B, F_video, device=self.device)
        cond_timesteps_action = torch.zeros(B, F_action, device=self.device)

        text_emb = batch_dict['text_emb']
        drop_text_ratio = float(getattr(self.config, 'drop_text_ratio', 0.0))
        if drop_text_ratio > 0 and hasattr(self, 'empty_emb'):
            drop_mask = torch.rand(B, device=self.device) < drop_text_ratio
            if drop_mask.any():
                empty_emb = self.empty_emb.to(device=text_emb.device, dtype=text_emb.dtype)
                text_emb = text_emb.clone()
                text_emb[drop_mask] = empty_emb.expand(B, -1, -1)[drop_mask]

        # 构建 latent_dict（不含 noisy_latents 和 targets）
        latent_dict = {
            'latent': batch_dict['latents'],  # 干净样本
            'cond_timesteps': cond_timesteps_video,
            'grid_id': latent_grid_id,
            'text_emb': text_emb,
        }

        # 构建 action_dict（不含 noisy_latents 和 targets）
        action_dict = {
            'latent': batch_dict['actions'],  # 干净样本
            'cond_timesteps': cond_timesteps_action,
            'grid_id': action_grid_id,
            'text_emb': text_emb,
            'actions_mask': batch_dict.get('actions_mask'),
        }

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': self.config.frame_chunk_size,
            'window_size': self.config.attn_window,
        }
        return input_dict

    # ==================================================================
    # _prepare_input_dict — 与 wan_va/train.py 完全一致
    # 返回 input_dict
    # ==================================================================
    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """
        准备模型输入字典。

        参数:
            batch_dict: 数据批次，包含 latents、actions、text_emb 等

        返回:
            input_dict: 包含以下字段的字典
              - latent_dict: 视频 latent 相关的字典（加噪、时间步、位置编码等）
              - action_dict: 动作 latent 相关的字典
              - chunk_size:  帧分块大小
              - window_size: 注意力窗口大小

        处理流程：
          1. 对视频 latent 添加噪声 → latent_dict
          2. 对动作 latent 添加噪声 → action_dict
          3. 将文本嵌入添加到两个字典中
          4. 组合成最终的 input_dict
        """
        # 对视频 latent 添加噪声
        latent_dict = self._add_noise(
            latent=batch_dict['latents'],
            train_scheduler=self.train_scheduler_latent,
            action_mask=None,
            action_mode=False,
            noisy_cond_prob=self.config.noisy_cond_prob,
        )

        # 对动作 latent 添加噪声
        action_dict = self._add_noise(
            latent=batch_dict['actions'],
            train_scheduler=self.train_scheduler_action,
            action_mask=batch_dict['actions_mask'],
            action_mode=True,
            noisy_cond_prob=0.0,
        )

        # 添加文本嵌入（用于条件生成）
        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': self.config.frame_chunk_size,
            'window_size': self.config.attn_window,
        }
        return input_dict

    def convert_input_format(self, input_dict):
        """
        将输入数据移动到正确的设备（GPU）。

        参数:
            input_dict: 数据批次字典

        返回:
            移动到 GPU 后的数据批次字典
        """
        def _move_value(value, key=None):
            # Raw policy inputs are consumed by the Cosmos inference adapter,
            # which may run in a separate process and expects CPU numpy data.
            if isinstance(key, str) and key.startswith("raw_"):
                return value
            if torch.is_tensor(value):
                return value.to(self.device, non_blocking=True)
            if isinstance(value, dict):
                return {inner_key: _move_value(inner_value, inner_key)
                        for inner_key, inner_value in value.items()}
            return value

        for key, value in input_dict.items():
            input_dict[key] = _move_value(value, key)
        return input_dict
