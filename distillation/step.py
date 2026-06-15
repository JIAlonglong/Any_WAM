"""
一致性蒸馏的训练步实现（StepMixin）。

这个文件包含了 LCM 蒸馏的核心逻辑：
  1. 教师模型通过 CFG Euler 步生成"伪 ground truth"
  2. 学生模型在 sigma_start 处学习一致性预测
  3. 目标学生（EMA）在 sigma_end 处生成训练目标
  4. 损失 = 学生预测 vs 目标预测 之间的一致性损失

FlowMatch 反演公式：
  pred_x0 = x_t - σ·v    （从 v-prediction 反推干净样本）
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from utils import data_seq_to_patch, logger
from consistency import scalings_for_boundary_conditions


class StepMixin:
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
            5D 视频张量 [B, C, F, H, W]，可以直接用于计算一致性损失
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
            sigma:        当前的噪声水平 σ
            sigma_data:   数据噪声水平（默认使用配置中的值）

        返回:
            一致性预测结果，形状与 noisy_latent 相同

        直觉理解：
            - v_pred 是模型预测的"去噪方向"
            - pred_x0 是沿这个方向推断出的干净样本
            - 一致性函数通过 c_skip 和 c_out 在输入和预测之间做加权
            - 边界条件保证 σ=0 时输出等于输入
        """
        if sigma_data is None:
            sigma_data = self.config.sigma_data
        # 将 sigma 扩展为 5D 以匹配 latent 的形状 [B, C, F, H, W]
        sigma_5d = sigma[None, None, :, None, None].to(v_pred.dtype).to(v_pred.device)
        # 计算边界条件缩放系数
        c_skip, c_out = scalings_for_boundary_conditions(
            sigma_5d, sigma_data=sigma_data)
        # FlowMatch 反演：从 v-prediction 推断干净样本
        pred_x0 = noisy_latent - sigma_5d * v_pred
        # 一致性函数：加权组合
        return c_skip * noisy_latent + c_out * pred_x0

    # ==================================================================
    # 一个完整的训练步
    # ==================================================================
    def _train_step(self, batch, batch_idx):
        """
        执行一步 LCM 蒸馏训练。

        训练流程概览：
          1. 准备输入（加噪、构建 input_dict）
          2. 计算 sigma_start 和 sigma_end（在 1000 步 schedule 中跳 k 步）
          3. 教师模型做 CFG Euler 步（有条件 + 无条件推理，然后加权组合）
          4. 学生模型在 sigma_start 处做一致性预测
          5. 目标学生（EMA）在 sigma_end 处生成训练目标
          6. 计算损失并反向传播

        参数:
            batch:     数据批次，包含 latents、actions、text_emb 等
            batch_idx: 当前批次在梯度累积中的索引

        返回:
            包含损失值和是否需要梯度同步的字典
        """
        batch = self.convert_input_format(batch)

        B = batch['latents'].shape[0]
        ref_shape = batch['latents'].shape     # [B, C, F, H, W]
        num_frames = ref_shape[2]
        actions_mask = batch.get('actions_mask')

        # ---- 步骤 1: 准备 input_dict（与原生训练完全一致）----
        # 对视频和动作 latent 分别加噪，构建模型输入
        input_dict = self._prepare_input_dict(batch)

        # ---- 步骤 2: 计算视频的 sigma_start 和 sigma_end ----
        # 从 1000 步 schedule 中找到最近的时间步索引
        video_timesteps = input_dict['latent_dict']['timesteps'][0]  # [F]
        sched_ts = self.train_scheduler_latent.timesteps  # [1000]
        video_ts_ids = torch.argmin(
            (sched_ts[:, None] - video_timesteps.cpu()).abs(), dim=0)  # [F]

        # sigma_start: 当前噪声水平
        sigma_start = self.train_scheduler_latent.sigmas[video_ts_ids].to(self.device)
        # sigma_end: 跳 k 步后的噪声水平（k=500，即跳到 schedule 的后半段）
        end_ids = (video_ts_ids + self.k).clamp(max=self.config.num_train_timesteps - 1)
        sigma_end = self.train_scheduler_latent.sigmas[end_ids].to(self.device)
        timesteps_end = self.train_scheduler_latent.timesteps[end_ids].to(self.device)

        # ---- 步骤 2b: 计算动作的 sigma 对 ----
        if self.distill_action:
            action_timesteps = input_dict['action_dict']['timesteps'][0]  # [F]
            sched_ts_a = self.train_scheduler_action.timesteps
            action_ts_ids = torch.argmin(
                (sched_ts_a[:, None] - action_timesteps.cpu()).abs(), dim=0)

            sigma_start_action = self.train_scheduler_action.sigmas[action_ts_ids].to(self.device)
            end_ids_action = (action_ts_ids + self.k_action).clamp(
                max=self.config.num_train_timesteps - 1)
            sigma_end_action = self.train_scheduler_action.sigmas[end_ids_action].to(self.device)
            timesteps_end_action = self.train_scheduler_action.timesteps[end_ids_action].to(self.device)

        # ---- 步骤 3: 教师 CFG Euler 步 ----
        # 随机采样 CFG 引导强度（在 [cfg_min, cfg_max] 范围内均匀采样）
        cfg_scale = self.config.cfg_min + torch.rand(1).item() * (
            self.config.cfg_max - self.config.cfg_min)

        with torch.no_grad():
            # 3a. 教师有条件推理（使用真实的文本嵌入）
            video_v_cond, action_v_cond = self.teacher(input_dict, train_mode=True)

            # 3b. 教师无条件推理（将文本嵌入替换为空嵌入）
            # 这是 CFG 的标准做法：比较有条件和无条件的输出差异
            B_emb = input_dict['latent_dict']['text_emb'].shape[0]
            empty_emb = self.empty_emb.expand(B_emb, -1, -1)
            input_dict_uncond = {
                'latent_dict': {**input_dict['latent_dict'], 'text_emb': empty_emb},
                'action_dict': {**input_dict['action_dict'], 'text_emb': empty_emb},
                'chunk_size': input_dict['chunk_size'],
                'window_size': input_dict['window_size'],
            }
            video_v_uncond, _ = self.teacher(input_dict_uncond, train_mode=True)

            # 3c. CFG 组合：v_cfg = v_uncond + scale * (v_cond - v_uncond)
            # 视频使用 CFG，动作不使用 CFG（action_guidance_scale=1）
            video_v_cfg = video_v_uncond + cfg_scale * (video_v_cond - video_v_uncond)

            # 3d. 视频 Euler 步：从 sigma_start 推进到 sigma_end
            # 公式：x_prev = x_t + v * (σ_end - σ_start)
            video_v_cfg_5d = self._extract_video_v(video_v_cfg, ref_shape, B)
            sigma_s = sigma_start[None, None, :, None, None].to(video_v_cfg_5d)
            sigma_e = sigma_end[None, None, :, None, None].to(video_v_cfg_5d)
            x_prev = input_dict['latent_dict']['noisy_latents'] + \
                     video_v_cfg_5d * (sigma_e - sigma_s)

            # 3e. 动作 Euler 步（如果启用动作蒸馏）
            if self.distill_action:
                action_v_5d = self._extract_action_v(action_v_cond, num_frames)
                sigma_s_a = sigma_start_action[None, None, :, None, None].to(action_v_5d)
                sigma_e_a = sigma_end_action[None, None, :, None, None].to(action_v_5d)
                x_prev_action = input_dict['action_dict']['noisy_latents'] + \
                                action_v_5d * (sigma_e_a - sigma_s_a)

        # ---- 步骤 4: 学生模型在 sigma_start 处做一致性预测 ----
        # 梯度累积优化：只在需要同步时才启用梯度同步
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        if not should_sync:
            self.student.set_requires_gradient_sync(False)
        else:
            self.student.set_requires_gradient_sync(True)

        student_video_v_seq, student_action_v_seq = self.student(input_dict, train_mode=True)

        # ---- 步骤 4a: 视频一致性预测 ----
        if self.distill_video:
            student_video_v = self._extract_video_v(student_video_v_seq, ref_shape, B)
            # 通过一致性函数将 v-prediction 转换为一致性预测
            student_video_pred = self._consistency_function(
                student_video_v,
                input_dict['latent_dict']['noisy_latents'],
                sigma_start,
            )

        # ---- 步骤 4b: 动作预测 ----
        if self.distill_action:
            student_action_v = self._extract_action_v(student_action_v_seq, num_frames)
            if self.action_distill_mode == "x0":
                # x0 参数化：直接预测干净样本
                # 公式：pred_x0 = x_σ - σ·v
                sigma_s_a_5d = sigma_start_action[None, None, :, None, None].to(student_action_v)
                student_action_pred = input_dict['action_dict']['noisy_latents'] - \
                                      sigma_s_a_5d * student_action_v
            else:
                # 使用标准一致性函数
                student_action_pred = self._consistency_function(
                    student_action_v,
                    input_dict['action_dict']['noisy_latents'],
                    sigma_start_action,
                )

        # ---- 步骤 5: 目标学生（EMA）在 sigma_end 处生成训练目标 ----
        # 构建 sigma_end 处的输入：使用教师 Euler 步推进后的 x_prev
        input_dict_end = {
            'latent_dict': {
                **input_dict['latent_dict'],
                'noisy_latents': x_prev.detach(),           # 教师推进后的潜在表示
                'timesteps': timesteps_end[None].repeat(B, 1),  # sigma_end 对应的时间步
            },
            'action_dict': {**input_dict['action_dict']},
            'chunk_size': input_dict['chunk_size'],
            'window_size': input_dict['window_size'],
        }
        if self.distill_action:
            input_dict_end['action_dict'] = {
                **input_dict['action_dict'],
                'noisy_latents': x_prev_action.detach(),
                'timesteps': timesteps_end_action[None].repeat(B, 1),
            }

        with torch.no_grad():
            # 目标学生（EMA 副本）在 sigma_end 处做前向
            target_video_v_seq, target_action_v_seq = self.target_student(
                input_dict_end, train_mode=True)

            if self.distill_video:
                target_video_v = self._extract_video_v(target_video_v_seq, ref_shape, B)
                # 目标的一致性预测
                target_video_pred = self._consistency_function(
                    target_video_v, x_prev, sigma_end,
                )

            if self.distill_action:
                target_action_v = self._extract_action_v(target_action_v_seq, num_frames)
                if self.action_distill_mode == "x0":
                    sigma_e_a_5d = sigma_end_action[None, None, :, None, None].to(target_action_v)
                    target_action_pred = x_prev_action - sigma_e_a_5d * target_action_v
                else:
                    target_action_pred = self._consistency_function(
                        target_action_v, x_prev_action, sigma_end_action,
                    )

        # ---- 步骤 6: 计算损失 ----
        video_loss = torch.tensor(0.0, device=self.device)
        if self.distill_video:
            if self.config.loss_type == "huber":
                # Huber 损失：对异常值更鲁棒
                c = self.config.huber_c
                video_diff = student_video_pred.float() - target_video_pred.detach().float()
                video_loss = torch.mean(torch.sqrt(video_diff ** 2 + c ** 2) - c)
            else:
                # MSE 损失
                video_loss = F.mse_loss(
                    student_video_pred.float(), target_video_pred.detach().float())

        action_loss = torch.tensor(0.0, device=self.device)
        if self.distill_action:
            mask = actions_mask.float()
            if self.config.loss_type == "huber":
                c = self.config.huber_c
                # 只对有效的动作位置计算损失（通过 mask 过滤）
                action_diff = (student_action_pred.float() * mask) - \
                              (target_action_pred.detach().float() * mask)
                action_loss = (torch.sqrt(action_diff ** 2 + c ** 2) - c).sum() / \
                              mask.sum().clamp(min=1)
            else:
                action_diff = (student_action_pred.float() * mask) - \
                              (target_action_pred.detach().float() * mask)
                action_loss = (action_diff ** 2).sum() / mask.sum().clamp(min=1)

        # ---- 步骤 6b: 动作感知正则化（原生 flow matching MSE）----
        # 这是一个辅助损失，让学生在动作预测上也保持与原始 flow matching 目标的一致性
        action_aware_loss = torch.tensor(0.0, device=self.device)
        if self.action_aware:
            student_action_v = self._extract_action_v(student_action_v_seq, num_frames)
            action_targets = input_dict['action_dict']['targets']
            mask = actions_mask.float()
            aa_diff = (student_action_v.float() - action_targets.float().detach()) * mask
            action_aware_loss = (aa_diff ** 2).sum() / mask.sum().clamp(min=1)

        # 总损失 = 视频一致性 + 动作一致性 + 动作感知正则
        loss = video_loss + self.config.action_loss_weight * action_loss \
               + getattr(self.config, 'action_aware_weight', 0.0) * action_aware_loss

        loss = loss / self.gradient_accumulation_steps

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
            "should_sync": should_sync,
        }
