"""
FlashWAMDistiller：模型/优化器/数据集初始化、训练循环、检查点管理。

这个文件是蒸馏训练的主类，负责：
  1. 初始化三个模型：教师（冻结）、在线学生（可训练）、目标学生（EMA）
  2. 配置优化器和学习率调度器
  3. 加载数据集
  4. 执行训练循环
  5. 定期保存检查点
"""

import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from safetensors.torch import save_file

from distributed.fsdp import shard_model, apply_ac
from distributed.util import _configure_model, dist_mean
from modules.utils import load_transformer
from utils import logger, warmup_constant_lambda, FlowMatchScheduler

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from data import DataMixin
from step import StepMixin
from ema import update_ema


class FlashWAMDistiller(DataMixin, StepMixin):
    """
    Flash-WAM 蒸馏训练器。

    继承自：
      - DataMixin: 数据加载和噪声准备功能
      - StepMixin: 训练步和一致性函数计算功能

    核心职责：
      - 管理三个模型的生命周期
      - 执行 LCM 蒸馏训练循环
      - 处理分布式训练和梯度累积
      - 定期保存检查点和记录日志
    """

    def __init__(self, config):
        """
        初始化蒸馏训练器。

        初始化流程：
          1. 设置基本配置（设备、精度、步长等）
          2. 初始化 WandB 日志（如果启用）
          3. 创建 FlowMatch 调度器（视频和动作各一个）
          4. 加载空文本嵌入（用于 CFG 无条件推理）
          5. 加载三个模型：教师、在线学生、目标学生
          6. 配置优化器和学习率调度器
          7. 加载数据集

        参数:
            config: 配置对象，包含所有训练参数
        """
        self.config = config
        self.step = 0                                          # 当前训练步数
        self.device = torch.device(f"cuda:{config.local_rank}")  # 当前 GPU 设备
        self.dtype = config.param_dtype                        # 模型参数精度（bf16）
        self.patch_size = config.patch_size                    # 视频 patch 大小
        self.gradient_accumulation_steps = config.gradient_accumulation_steps  # 梯度累积步数

        # k: 在 1000 步 schedule 中的跳步数
        # 例如：1000 / 25 = 40，意味着每次跳 40 步
        self.k = config.num_train_timesteps // config.num_ddim_timesteps
        # 蒸馏模式控制
        self.distill_video = getattr(config, 'distill_video', True)      # 是否蒸馏视频
        self.distill_action = getattr(config, 'distill_action', False)   # 是否蒸馏动作
        self.action_distill_mode = getattr(config, 'action_distill_mode', 'consistency')  # 动作参数化方式
        self.action_aware = getattr(config, 'action_aware', False)       # 是否使用动作感知正则
        # 动作的跳步数（可能与视频不同）
        self.k_action = config.num_train_timesteps // getattr(
            config, 'num_ddim_timesteps_action', config.num_ddim_timesteps)

        # WandB 日志初始化（仅主进程）
        if config.enable_wandb and HAS_WANDB and config.rank == 0:
            wandb.init(
                project="lcm_video_distill_lingbot_va",
                entity=getattr(config, "wandb_entity", None),
                config=dict(config),
            )

        # ==============================================================
        # 调度器初始化 — 与 wan_va/train.py 完全一致
        # ==============================================================
        # 视频调度器：SNR 偏移 5.0
        self.train_scheduler_latent = FlowMatchScheduler(
            shift=config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(config.num_train_timesteps, training=True)

        # 动作调度器：SNR 偏移 1.0（动作的噪声分布不同）
        self.train_scheduler_action = FlowMatchScheduler(
            shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(config.num_train_timesteps, training=True)

        # 空文本嵌入：用于 CFG 无条件推理时替换真实文本嵌入
        self.empty_emb = torch.load(config.empty_emb_path, map_location="cpu").to(self.device)

        if config.rank == 0:
            logger.info(f"LCM stride k = {self.k} "
                        f"(num_train_timesteps={config.num_train_timesteps}, "
                        f"num_ddim_timesteps={config.num_ddim_timesteps})")
            logger.info(f"Distill video: {self.distill_video}")
            logger.info(f"Distill action: {self.distill_action}")
            logger.info(f"Action aware: {self.action_aware}")
            if self.distill_action:
                logger.info(f"  k_action = {self.k_action}, "
                            f"action_loss_weight = {config.action_loss_weight}, "
                            f"mode = {self.action_distill_mode}")
            if self.action_aware:
                logger.info(f"  action_aware_weight = {config.action_aware_weight}")
            logger.info(f"Empty embedding shape: {self.empty_emb.shape}")

        # ==============================================================
        # 三个模型的初始化
        # ==============================================================
        teacher_path = os.path.join(config.teacher_model_path, "transformer")

        # 1. 教师模型（frozen）：预训练好的 LingBot-VA，不参与训练
        logger.info("Loading teacher (frozen) ...")
        self.teacher = load_transformer(teacher_path, torch_dtype=self.dtype, torch_device="cpu")
        self.teacher.requires_grad_(False)  # 冻结所有参数
        self.teacher.eval()                 # 评估模式
        self.teacher = self.teacher.to(self.dtype)
        # 使用 FSDP 分片和配置化
        self.teacher = _configure_model(
            model=self.teacher, shard_fn=shard_model,
            param_dtype=self.dtype, device=self.device, eval_mode=True,
        )

        # 确定学生模型的初始化路径（从检查点恢复或从教师初始化）
        resume_path = getattr(config, "resume_from_path", None)
        resume_step = getattr(config, "resume_from_step", None)
        if resume_path is not None:
            # 从指定路径恢复
            student_path = os.path.join(resume_path, "online_student", "transformer")
            target_path = os.path.join(resume_path, "target_student", "transformer")
            self.step = resume_step if resume_step is not None else 0
            if config.rank == 0:
                logger.info(f"Resuming from path: {resume_path}")
                logger.info(f"  Online student: {student_path}")
                logger.info(f"  Target student: {target_path}")
                logger.info(f"  Starting step: {self.step}")
        elif resume_step is not None:
            # 从输出目录的指定步数恢复
            student_path = os.path.join(
                config.output_dir, "checkpoints", f"step_{resume_step}",
                "online_student", "transformer")
            target_path = os.path.join(
                config.output_dir, "checkpoints", f"step_{resume_step}",
                "target_student", "transformer")
            self.step = resume_step
            if config.rank == 0:
                logger.info(f"Resuming from step {resume_step}")
                logger.info(f"  Online student: {student_path}")
                logger.info(f"  Target student: {target_path}")
        else:
            # 从教师权重初始化（标准 LCM 蒸馏初始化）
            student_path = teacher_path
            target_path = teacher_path

        # 2. 在线学生（trainable）：正在训练的模型
        logger.info("Loading online student (trainable) ...")
        self.student = load_transformer(student_path, torch_dtype=torch.float32, torch_device="cpu")
        apply_ac(self.student)  # 应用激活检查点（节省显存）
        self.student = self.student.to(self.dtype)
        self.student = _configure_model(
            model=self.student, shard_fn=shard_model,
            param_dtype=self.dtype, device=self.device, eval_mode=False,
        )
        self.student.train()                # 训练模式
        self.student.requires_grad_(True)   # 启用梯度计算

        # 3. 目标学生（EMA, frozen）：在线学生的指数移动平均副本
        logger.info("Loading target student (EMA, frozen) ...")
        self.target_student = load_transformer(target_path, torch_dtype=self.dtype, torch_device="cpu")
        self.target_student = self.target_student.to(self.dtype)
        self.target_student = _configure_model(
            model=self.target_student, shard_fn=shard_model,
            param_dtype=self.dtype, device=self.device, eval_mode=False,
        )
        self.target_student.requires_grad_(False)  # 冻结，仅用于推理
        self.target_student.eval()

        # ==============================================================
        # 优化器和学习率调度器
        # ==============================================================
        # AdamW 优化器：只优化在线学生的可训练参数
        self.optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,      # 使用融合实现（更快）
            foreach=False,
        )
        # 学习率调度器：warmup + 常数学习率
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps),
        )
        # 如果从检查点恢复，快速推进学习率调度器到正确位置
        if self.step > 0:
            for _ in range(self.step):
                self.lr_scheduler.step()
            if config.rank == 0:
                logger.info(f"LR scheduler fast-forwarded to step {self.step}, "
                            f"lr={self.lr_scheduler.get_last_lr()[0]:.2e}")

        # ==============================================================
        # 数据集加载 — 与原生训练相同
        # ==============================================================
        logger.info("Loading dataset ...")
        from patches import SafeMultiLatentLeRobotDataset as MultiLatentLeRobotDataset
        train_dataset = MultiLatentLeRobotDataset(config=config)
        # 分布式采样器：确保每个 GPU 看到不同的数据子集
        train_sampler = (
            DistributedSampler(train_dataset, num_replicas=config.world_size,
                               rank=config.rank, shuffle=True, seed=config.seed)
            if config.world_size > 1 else None
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.load_worker,
            sampler=train_sampler,
        )
        self.save_dir = Path(config.output_dir) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_loader_iter = None  # 数据迭代器（懒加载）

    # ==================================================================
    # 保存检查点
    # ==================================================================
    def _save_checkpoint(self, which="online_student"):
        """
        保存模型检查点。

        参数:
            which: 要保存的模型，"online_student" 或 "target_student"

        保存内容：
          - 模型权重（safetensors 格式，bf16 精度）
          - 模型配置（config.json）
        """
        model = self.student if which == "online_student" else self.target_student
        try:
            # 获取完整状态字典（支持分布式训练）
            state_dict = get_model_state_dict(
                model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

            if self.config.rank == 0:
                ckpt_dir = self.save_dir / f"step_{self.step}" / which / "transformer"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_file(state_dict_bf16, ckpt_dir / "diffusion_pytorch_model.safetensors")
                config_dict = dict(model.config)
                config_dict.pop("_name_or_path", None)
                with open(ckpt_dir / "config.json", "w") as f:
                    json.dump(config_dict, f, indent=2)
                logger.info(f"  Saved {which} → {ckpt_dir}")

            if dist.is_initialized():
                dist.barrier()  # 同步所有进程
        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save {which}: {e}")
                import traceback
                logger.error(traceback.format_exc())
            if dist.is_initialized():
                dist.barrier()

    # ==================================================================
    # 主训练循环
    # ==================================================================
    def train(self):
        """
        执行 LCM 蒸馏训练循环。

        训练流程：
          1. 遍历数据集，每个批次执行一步蒸馏训练
          2. 累积梯度，达到累积步数后更新参数
          3. 更新目标学生（EMA）
          4. 定期保存检查点和记录日志
          5. 训练完成后保存最终检查点

        关键设计：
          - 梯度累积：每 gradient_accumulation_steps 步才更新一次参数
          - EMA 更新：每次参数更新后，用在线学生的权重更新目标学生
          - 分布式训练：所有进程同步梯度和损失值
        """
        config = self.config
        mode = []
        if self.distill_video:
            mode.append("video")
        if self.distill_action:
            mode.append("action")
        if self.action_aware:
            mode.append("action_aware")
        mode_str = "+".join(mode) if mode else "none"
        logger.info(f"Starting LCM {mode_str} distillation for {config.max_train_steps} steps ...")
        if self.distill_video:
            logger.info(f"  k = {self.k} ({config.num_train_timesteps} / {config.num_ddim_timesteps})")
        if self.distill_action:
            logger.info(f"  k_action = {self.k_action} "
                        f"({config.num_train_timesteps} / {config.num_ddim_timesteps_action})")
            logger.info(f"  action_loss_weight = {config.action_loss_weight}")
            logger.info(f"  action_distill_mode = {self.action_distill_mode}")
        logger.info(f"  Teacher CFG: [{config.cfg_min}, {config.cfg_max}]")
        logger.info(f"  EMA decay: {config.ema_decay}")
        logger.info(f"  Loss: {config.loss_type}")

        self.optimizer.zero_grad()
        acc_losses = []              # 累积的总损失
        acc_video_losses = []        # 累积的视频损失
        acc_action_losses = []       # 累积的动作损失
        acc_action_aware_losses = [] # 累积的动作感知损失
        step_in_acc = 0              # 当前累积步数

        progress_bar = tqdm(
            total=config.max_train_steps,
            desc="Distill", disable=(config.rank != 0),
            leave=True, dynamic_ncols=True, initial=self.step,
        )

        while self.step < config.max_train_steps:
            # 获取下一个数据批次
            batch = self._get_next_batch()
            # 执行一步训练
            result = self._train_step(batch, step_in_acc)
            # 累积损失值
            acc_losses.append(result["loss"])
            acc_video_losses.append(result["video_loss"])
            acc_action_losses.append(result["action_loss"])
            acc_action_aware_losses.append(result["action_aware_loss"])
            step_in_acc += 1

            # 检查是否需要梯度同步（达到累积步数）
            if result["should_sync"]:
                # 梯度裁剪
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), config.max_grad_norm)

                if not torch.isfinite(total_norm):
                    # 梯度范数为 NaN/Inf，跳过这一步
                    if config.rank == 0:
                        logger.warning(f"[step {self.step}] NaN grad norm, skipping")
                    self.optimizer.zero_grad()
                else:
                    # 正常更新参数
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # 更新目标学生（EMA）
                # 使用线性插值：target = decay * target + (1-decay) * student
                update_ema(
                    self.target_student.parameters(),
                    self.student.parameters(),
                    rate=config.ema_decay,
                )

                # 计算平均损失（跨所有进程）
                lr = self.lr_scheduler.get_last_lr()[0]
                avg_loss = dist_mean(torch.stack(acc_losses).sum()).item()
                avg_video_loss = dist_mean(torch.stack(acc_video_losses).sum()).item()
                avg_action_loss = dist_mean(torch.stack(acc_action_losses).sum()).item()
                avg_action_aware_loss = dist_mean(torch.stack(acc_action_aware_losses).sum()).item()
                # 重置累积器
                acc_losses = []
                acc_video_losses = []
                acc_action_losses = []
                acc_action_aware_losses = []
                step_in_acc = 0

                # 定期清理显存
                torch.cuda.synchronize()
                if self.step % config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                # 记录日志（仅主进程）
                if config.rank == 0:
                    progress_bar.n = self.step + 1
                    postfix = {
                        "loss": f"{avg_loss:.4f}",
                        "norm": f"{total_norm.item():.2f}",
                        "lr": f"{lr:.2e}",
                    }
                    log_dict = {
                        "loss/total": avg_loss,
                        "train/grad_norm": total_norm.item(),
                        "train/lr": lr,
                    }
                    if self.distill_video:
                        postfix["v"] = f"{avg_video_loss:.4f}"
                        log_dict["loss/video_consistency"] = avg_video_loss
                    if self.distill_action:
                        postfix["a"] = f"{avg_action_loss:.4f}"
                        log_dict["loss/action_consistency"] = avg_action_loss
                    if self.action_aware:
                        postfix["aa"] = f"{avg_action_aware_loss:.4f}"
                        log_dict["loss/action_aware"] = avg_action_aware_loss
                    progress_bar.set_postfix(postfix)
                    if config.enable_wandb and HAS_WANDB:
                        wandb.log(log_dict, step=self.step)

                self.step += 1

                # 定期保存检查点
                if self.step % config.save_interval == 0:
                    self._save_checkpoint("online_student")
                    self._save_checkpoint("target_student")

            # 分布式同步屏障
            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        logger.info("Distillation completed!")
        # 保存最终检查点
        self._save_checkpoint("online_student")
        self._save_checkpoint("target_student")
