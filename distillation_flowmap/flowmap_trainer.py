"""
FlowMapDistiller：Flow Map 蒸馏训练主类。

这个文件是 Flow Map 蒸馏训练的核心入口，负责：
  1. 初始化三个模型：教师（冻结）、在线学生（可训练）、目标学生（EMA）
  2. 为学生模型和目标学生模型添加 Flow Map 能力（delta_embedder + patched forward）
  3. 配置优化器和学习率调度器
  4. 加载数据集
  5. 执行 Flow Map 蒸馏训练循环（混合采样：扩散目标 + 一致性目标 + 流映射目标）
  6. 定期保存检查点

与原始 FlashWAMDistiller 的核心区别：
  - 学生模型和目标学生模型通过 setup_flowmap_model 替换 condition_embedder，
    增加 delta_embedder 用于编码参考时间步 r
  - 教师模型保持原始结构（不添加 delta_embedder）
  - 训练步使用 FlowMapStepMixin（支持扩散/一致性/流映射三种目标的混合采样）
  - 日志输出增加 Flow Map 相关信息（diffusion_ratio, consistency_ratio, epsilon 等）
  - 损失日志增加 gt_regression_loss（GT 回归辅助损失）
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

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

from distillation.data import DataMixin
from distillation.ema import update_ema
from flowmap_step import FlowMapStepMixin
from model_flowmap import setup_flowmap_model, patch_model_forward

# DMD 判别器（仅在 use_dmd=True 时导入）
try:
    from discriminator import ActionDiscriminator
    HAS_DISCRIMINATOR = True
except ImportError:
    HAS_DISCRIMINATOR = False


class FlowMapDistiller(DataMixin, FlowMapStepMixin):
    """
    Flow Map 蒸馏训练器。

    继承自：
      - DataMixin:       数据加载和噪声准备功能（来自 distillation.data）
      - FlowMapStepMixin: Flow Map 训练步和混合目标计算功能（扩散/一致性/流映射）

    与原始 FlashWAMDistiller 的关键区别：
      1. 学生模型和目标学生模型经过 setup_flowmap_model 改造，
         condition_embedder 被替换为 WanTwoTimeTextImageEmbedding（包含 delta_embedder）
      2. 学生模型和目标学生模型经过 patch_model_forward 补丁，
         forward / forward_train / _time_embed 方法支持额外的 r_timestep 参数
      3. 教师模型保持原始结构，不做任何修改（教师不需要 Flow Map 能力）
      4. 训练步由 FlowMapStepMixin 提供，实现三种目标的随机混合采样
    """

    def __init__(self, config):
        """
        初始化 Flow Map 蒸馏训练器。

        初始化流程：
          1. 设置基本配置（设备、精度、步长等）
          2. 初始化 WandB 日志（如果启用）
          3. 创建 FlowMatch 调度器（视频和动作各一个）
          4. 加载空文本嵌入（用于 CFG 无条件推理）
          5. 加载三个模型：教师、在线学生、目标学生
          6. 为学生和目标学生添加 Flow Map 能力（setup_flowmap_model + patch_model_forward）
          7. 配置优化器和学习率调度器
          8. 加载数据集

        参数:
            config: 配置对象，包含所有训练参数（参见 config.py）
        """
        self.config = config
        self.step = 0                                          # 当前训练步数
        self.device = torch.device(f"cuda:{config.local_rank}")  # 当前 GPU 设备
        self.dtype = config.param_dtype                        # 模型参数精度（bf16）
        self.patch_size = config.patch_size                    # 视频 patch 大小
        self.gradient_accumulation_steps = config.gradient_accumulation_steps  # 梯度累积步数

        # k: 在 1000 步 schedule 中的跳步数
        # 例如：1000 / 2 = 500，意味着每次跳 500 步
        self.k = config.num_train_timesteps // config.num_ddim_timesteps
        # 蒸馏模式控制
        self.distill_video = getattr(config, 'distill_video', True)      # 是否蒸馏视频
        self.distill_action = getattr(config, 'distill_action', False)   # 是否蒸馏动作
        self.action_distill_mode = getattr(config, 'action_distill_mode', 'consistency')  # 动作参数化方式
        self.action_aware = getattr(config, 'action_aware', False)       # 是否使用动作感知正则
        # 动作的跳步数（可能与视频不同）
        self.k_action = config.num_train_timesteps // getattr(
            config, 'num_ddim_timesteps_action', config.num_ddim_timesteps)

        # ==============================================================
        # Flow Map 蒸馏特有参数
        # ==============================================================
        # 三种目标的混合比例（扩散目标 + 一致性目标 + 流映射目标 = 1.0）
        self.diffusion_ratio = getattr(config, 'diffusion_ratio', 0.5)     # 扩散目标占比
        self.consistency_ratio = getattr(config, 'consistency_ratio', 0.25) # 一致性目标占比
        self.flowmap_ratio = getattr(config, 'flowmap_ratio', 0.25)         # 流映射目标占比
        # 中心差分扰动步长（用于计算 flow map 的数值梯度）
        self.epsilon = getattr(config, 'epsilon', 1.0)
        # GT 回归 loss 权重（辅助损失，帮助稳定训练）
        self.gt_regression_weight = getattr(config, 'gt_regression_weight', 0.1)
        # LoRA 微调开关
        self.use_lora = getattr(config, 'use_lora', False)

        # ==============================================================
        # DMD（On-Policy Distribution Matching Distillation）参数
        # ==============================================================
        self.use_dmd = getattr(config, 'use_dmd', False)
        self.discriminator = None
        self.discriminator_optimizer = None

        # WandB 日志初始化（仅主进程）
        if config.enable_wandb and HAS_WANDB and config.rank == 0:
            wandb.init(
                project="flowmap_distill_lingbot_va",
                entity=getattr(config, "wandb_entity", None),
                config=dict(config),
            )

        # TensorBoard 日志初始化（仅主进程）
        self.tb_writer = None
        if HAS_TENSORBOARD and config.rank == 0:
            tb_dir = Path(config.output_dir) / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"[TensorBoard] Logging to {tb_dir}")

        # ==============================================================
        # 调度器初始化 — 与原始 FlashWAMDistiller 完全一致
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

        # ==============================================================
        # 打印 Flow Map 蒸馏特有配置信息
        # ==============================================================
        if config.rank == 0:
            logger.info(f"Flow Map Distiller 初始化")
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
            # Flow Map 特有信息
            logger.info(f"--- Flow Map 配置 ---")
            logger.info(f"  diffusion_ratio   = {self.diffusion_ratio} (扩散目标占比, r=t)")
            logger.info(f"  consistency_ratio  = {self.consistency_ratio} (一致性目标占比, r=0)")
            logger.info(f"  flowmap_ratio      = {self.flowmap_ratio} (流映射目标占比, r<t)")
            logger.info(f"  epsilon            = {self.epsilon} (中心差分扰动步长)")
            logger.info(f"  gate_value         = {config.gate_value} (delta_emb 门控初始值)")
            logger.info(f"  deltatime_type     = {config.deltatime_type} (delta 时间步类型)")
            logger.info(f"  weight_type        = {getattr(config, 'weight_type', 'uniform')} (时间步采样权重)")
            logger.info(f"  gt_regression_weight = {self.gt_regression_weight} (GT 回归 loss 权重)")
            logger.info(f"  use_lora           = {self.use_lora} (LoRA 微调)")
            if self.use_lora:
                logger.info(f"  lora_rank          = {config.lora_rank}")
                logger.info(f"  lora_alpha         = {config.lora_alpha}")
                logger.info(f"  lora_target_modules= {config.lora_target_modules}")

        # ==============================================================
        # 三个模型的初始化
        # ==============================================================
        teacher_path = os.path.join(config.teacher_model_path, "transformer")

        # 1. 教师模型（frozen）：预训练好的 LingBot-VA，不参与训练
        #    注意：教师模型保持原始结构，不添加 delta_embedder
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

        # torch.compile 加速教师模型（冻结模型，无 LoRA，可以安全编译）
        # 注意：教师始终 compile，与 use_torch_compile（控制学生）无关
        # 使用 mode="default" 而非 "reduce-overhead"，避免 CUDA graphs 与多次 teacher 前向冲突
        if not getattr(config, 'skip_teacher_compile', False):
            logger.info("Compiling teacher model with torch.compile ...")
            self.teacher = torch.compile(self.teacher, mode="default")
            logger.info("Teacher model compiled.")

        # 确定学生模型的初始化路径（从检查点恢复或从教师初始化）
        resume_path = getattr(config, "resume_from_path", None)
        resume_step = getattr(config, "resume_from_step", None)
        self._is_lora_resume = False  # 标记是否从 LoRA checkpoint 恢复

        if resume_path is not None:
            # 从指定路径恢复
            student_path = os.path.join(resume_path, "online_student", "transformer")
            target_path = os.path.join(resume_path, "target_student", "transformer")
            self.step = resume_step if resume_step is not None else 0
            # 检查是否是 LoRA checkpoint
            student_config_path = os.path.join(student_path, "config.json")
            if os.path.exists(student_config_path):
                import json
                with open(student_config_path) as f:
                    ckpt_config = json.load(f)
                if ckpt_config.get('use_lora', False):
                    self._is_lora_resume = True
                    if config.rank == 0:
                        logger.info(f"Detected LoRA checkpoint, will load base model from teacher and apply LoRA adapters")
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
            # 检查是否是 LoRA checkpoint
            student_config_path = os.path.join(student_path, "config.json")
            if os.path.exists(student_config_path):
                import json
                with open(student_config_path) as f:
                    ckpt_config = json.load(f)
                if ckpt_config.get('use_lora', False):
                    self._is_lora_resume = True
                    if config.rank == 0:
                        logger.info(f"Detected LoRA checkpoint, will load base model from teacher and apply LoRA adapters")
            if config.rank == 0:
                logger.info(f"Resuming from step {resume_step}")
                logger.info(f"  Online student: {student_path}")
                logger.info(f"  Target student: {target_path}")
        else:
            # 从教师权重初始化（标准 LCM 蒸馏初始化）
            student_path = teacher_path
            target_path = teacher_path

        # 2. 在线学生（trainable）：正在训练的模型
        # 如果从 LoRA checkpoint 恢复，需要先加载基座模型，再加载 LoRA adapter
        if self._is_lora_resume:
            logger.info("Loading online student from LoRA checkpoint ...")
            logger.info("  Step 1: Loading base model from teacher ...")
            self.student = load_transformer(teacher_path, torch_dtype=torch.float32, torch_device="cpu")
        else:
            logger.info("Loading online student (trainable) ...")
            self.student = load_transformer(student_path, torch_dtype=torch.float32, torch_device="cpu")
        apply_ac(self.student)  # 应用激活检查点（节省显存）
        self.student = self.student.to(self.dtype)

        # 为学生模型添加 Flow Map 能力
        # setup_flowmap_model: 将 condition_embedder 替换为 WanTwoTimeTextImageEmbedding（含 delta_embedder）
        # patch_model_forward: 给 forward / forward_train / _time_embed 添加 r_timestep 参数支持
        logger.info("Setting up Flow Map for online student ...")
        self.student = setup_flowmap_model(
            self.student, gate_value=config.gate_value, deltatime_type=config.deltatime_type)
        self.student = patch_model_forward(self.student)

        # ==============================================================
        # LoRA 可选：为学生模型添加 LoRA adapter（降低显存开销）
        # ==============================================================
        if self.use_lora:
            if not HAS_PEFT:
                raise ImportError("LoRA requires the 'peft' library. Install with: pip install peft")
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=getattr(config, 'lora_dropout', 0.0),
                bias='none',
            )
            self.student = get_peft_model(self.student, lora_config, adapter_name='default')
            if config.rank == 0:
                self.student.print_trainable_parameters()
            logger.info(f"LoRA enabled: rank={config.lora_rank}, alpha={config.lora_alpha}")

            # 如果从 LoRA checkpoint 恢复，加载 LoRA adapter 权重
            if self._is_lora_resume:
                logger.info("  Step 2: Loading LoRA adapter weights from checkpoint ...")
                from safetensors.torch import load_file as safetensors_load_file
                adapter_path = os.path.join(student_path, "diffusion_pytorch_model.safetensors")
                if os.path.exists(adapter_path):
                    adapter_state = safetensors_load_file(adapter_path)
                    # 加载 adapter 权重（包括 lora_A, lora_B, delta_emb_gate 等）
                    missing, unexpected = self.student.load_state_dict(adapter_state, strict=False)
                    if config.rank == 0:
                        logger.info(f"  Loaded {len(adapter_state)} adapter parameters")
                        if missing:
                            logger.warning(f"  Missing keys: {len(missing)}")
                        if unexpected:
                            logger.warning(f"  Unexpected keys: {len(unexpected)}")
                else:
                    logger.warning(f"  Adapter weights not found: {adapter_path}")

        # Cast entire model (including LoRA parameters) to uniform dtype
        # before FSDP wrapping. FSDP requires all original parameters to
        # have the same dtype; LoRA parameters default to float32, which
        # conflicts with the bfloat16 base model.
        self.student = self.student.to(self.dtype)

        self.student = _configure_model(
            model=self.student, shard_fn=shard_model,
            param_dtype=self.dtype, device=self.device, eval_mode=False,
        )
        self.student.train()                # 训练模式
        self.student.requires_grad_(True)   # 启用梯度计算

        # torch.compile 加速（可选）
        if getattr(config, 'use_torch_compile', False):
            logger.info("Compiling student model with torch.compile ...")
            self.student = torch.compile(self.student, mode="reduce-overhead")
            logger.info("Student model compiled.")

        # 3. 目标学生（EMA, frozen）：在线学生的指数移动平均副本
        # 如果从 LoRA checkpoint 恢复，需要先加载基座模型，再加载 LoRA adapter
        if self._is_lora_resume:
            logger.info("Loading target student from LoRA checkpoint ...")
            logger.info("  Step 1: Loading base model from teacher ...")
            self.target_student = load_transformer(teacher_path, torch_dtype=self.dtype, torch_device="cpu")
        else:
            logger.info("Loading target student (EMA, frozen) ...")
            self.target_student = load_transformer(target_path, torch_dtype=self.dtype, torch_device="cpu")
        self.target_student = self.target_student.to(self.dtype)

        # 为目标学生模型添加 Flow Map 能力（与学生模型相同的改造）
        logger.info("Setting up Flow Map for target student ...")
        self.target_student = setup_flowmap_model(
            self.target_student, gate_value=config.gate_value, deltatime_type=config.deltatime_type)
        self.target_student = patch_model_forward(self.target_student)

        # 如果从 LoRA checkpoint 恢复，为目标学生也添加 LoRA 并加载权重
        if self._is_lora_resume and self.use_lora:
            logger.info("  Step 2: Adding LoRA to target student ...")
            target_lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=getattr(config, 'lora_dropout', 0.0),
                bias='none',
            )
            self.target_student = get_peft_model(self.target_student, target_lora_config, adapter_name='default')

            # 加载目标学生的 LoRA adapter 权重
            target_adapter_path = os.path.join(target_path, "diffusion_pytorch_model.safetensors")
            if os.path.exists(target_adapter_path):
                from safetensors.torch import load_file as safetensors_load_file
                target_adapter_state = safetensors_load_file(target_adapter_path)
                missing, unexpected = self.target_student.load_state_dict(target_adapter_state, strict=False)
                if config.rank == 0:
                    logger.info(f"  Loaded {len(target_adapter_state)} target adapter parameters")
            else:
                logger.warning(f"  Target adapter weights not found: {target_adapter_path}")
        elif self.use_lora:
            # 非 LoRA 恢复模式：正常为目标学生添加 LoRA
            target_lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=getattr(config, 'lora_dropout', 0.0),
                bias='none',
            )
            self.target_student = get_peft_model(self.target_student, target_lora_config, adapter_name='default')

        # Cast entire model (including LoRA parameters) to uniform dtype
        # before FSDP wrapping (same reason as online student above).
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
        # DMD 判别器初始化（仅在 use_dmd=True 时）
        # ==============================================================
        if self.use_dmd:
            if not HAS_DISCRIMINATOR:
                raise ImportError("DMD requires discriminator.py. "
                                  "Ensure distillation_flowmap/discriminator.py exists.")
            # 从模型配置获取维度信息
            inner_dim = self.student.num_attention_heads * self.student.attention_head_dim
            text_dim = self.student.condition_embedder.text_embedder.linear_1.in_features

            # 注意：action_latent_channels 应该是 action_dim（动作维度），
            # 因为判别器接收的是原始动作 [B, C, F, N, 1]，其中 C = action_dim
            # 判别器保持 float32 精度（与学生模型的 bfloat16 分开）
            # 在 forward 方法中会将输入转换为 float32
            # 注意：video_latent_channels 应该是 VAE 的输出通道数（48），不是 transformer 的 inner_dim
            # 注意：max_text_tokens 应该是 512（T5 编码器的序列长度），不是默认的 77
            self.discriminator = ActionDiscriminator(
                action_dim=config.action_dim,
                action_latent_channels=config.action_dim,  # 使用 action_dim 而非 inner_dim
                video_latent_channels=48,  # VAE 输出通道数（in_channels）
                text_dim=text_dim,
                hidden_dim=getattr(config, 'dmd_hidden_dim', 256),
                num_layers=getattr(config, 'dmd_num_layers', 4),
                num_heads=getattr(config, 'dmd_num_heads', 8),
                dropout=getattr(config, 'dmd_dropout', 0.1),
                max_text_tokens=512,  # T5 编码器的序列长度
            ).to(self.device)  # 保持 float32，不转换 dtype
            self.discriminator.train()

            # 判别器不使用 FSDP（参数量小，6.79M）
            # DTensor→Tensor 转换在判别器 forward 入口处理

            self.discriminator_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=getattr(config, 'dmd_discriminator_lr', 1e-5),
                betas=(0.9, 0.999),
                weight_decay=0.0,
            )

            if config.rank == 0:
                logger.info(f"--- DMD 配置 ---")
                logger.info(f"  use_dmd              = {self.use_dmd}")
                logger.info(f"  dmd_weight           = {getattr(config, 'dmd_weight', 0.1)}")
                logger.info(f"  dmd_warmup_steps     = {getattr(config, 'dmd_warmup_steps', 0)}")
                logger.info(f"  dmd_rollout_steps    = [{getattr(config, 'dmd_rollout_steps_min', 2)}, "
                            f"{getattr(config, 'dmd_rollout_steps_max', 8)}]")
                logger.info(f"  dmd_discriminator_lr = {getattr(config, 'dmd_discriminator_lr', 1e-5)}")
                logger.info(f"  dmd_discriminator_warmup = {getattr(config, 'dmd_discriminator_warmup', 200)}")
                total_params = sum(p.numel() for p in self.discriminator.parameters())
                logger.info(f"  discriminator params = {total_params / 1e6:.2f}M")

        # ==============================================================
        # 数据集加载 — 与原始 FlashWAMDistiller 完全一致
        # ==============================================================
        logger.info("Loading dataset ...")
        from distillation.patches import SafeMultiLatentLeRobotDataset as MultiLatentLeRobotDataset
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
            pin_memory=getattr(config, 'pin_memory', False),
            prefetch_factor=getattr(config, 'prefetch_factor', 2) if config.load_worker > 0 else None,
            persistent_workers=True if config.load_worker > 0 else False,
        )
        self.save_dir = Path(config.output_dir) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_loader_iter = None  # 数据迭代器（懒加载）

        # ==============================================================
        # DMD 判别器 checkpoint 恢复（如果从 checkpoint 恢复）
        # ==============================================================
        if self.use_dmd and self.discriminator is not None and self.step > 0:
            # 确定判别器 checkpoint 路径
            if resume_path is not None:
                disc_dir = Path(resume_path) / "discriminator"
            elif resume_step is not None:
                disc_dir = self.save_dir / f"step_{resume_step}" / "discriminator"
            else:
                disc_dir = None

            if disc_dir is not None:
                disc_ckpt = disc_dir / "discriminator.safetensors"
                disc_opt_ckpt = disc_dir / "discriminator_optimizer.pt"
                if disc_ckpt.exists():
                    from safetensors.torch import load_file as safetensors_load_file
                    disc_state = safetensors_load_file(str(disc_ckpt))
                    self.discriminator.load_state_dict(disc_state)
                    if config.rank == 0:
                        logger.info(f"Loaded discriminator from {disc_ckpt}")
                if disc_opt_ckpt.exists() and self.discriminator_optimizer is not None:
                    opt_state = torch.load(disc_opt_ckpt, map_location=self.device)
                    self.discriminator_optimizer.load_state_dict(opt_state)
                    if config.rank == 0:
                        logger.info(f"Loaded discriminator optimizer from {disc_opt_ckpt}")

    # ==================================================================
    # 保存检查点（与原始 FlashWAMDistiller 完全相同）
    # ==================================================================
    def _save_checkpoint(self, which="online_student"):
        """
        保存模型检查点。

        参数:
            which: 要保存的模型，"online_student" 或 "target_student"

        保存内容：
          - 全量微调：完整模型权重（safetensors 格式，bf16 精度）
          - LoRA 微调：仅 LoRA adapter 权重（大幅减小文件大小）
          - 模型配置（config.json）

        注意：保存的模型已经包含 Flow Map 改造（delta_embedder 等），
        恢复时可以直接加载，无需再次调用 setup_flowmap_model。
        """
        model = self.student if which == "online_student" else self.target_student
        try:
            if self.use_lora:
                # LoRA 模式：只保存 adapter 权重（不含基座模型权重，文件更小）
                state_dict = get_model_state_dict(
                    model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
                # 只保留 LoRA adapter 参数（包含 lora_A, lora_B, delta_emb_gate 等）
                adapter_keys = [k for k in state_dict.keys()
                                if 'lora_' in k or 'delta_emb_gate' in k or 'delta_embedder' in k]
                adapter_state_dict = {k: state_dict[k] for k in adapter_keys}
                state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in adapter_state_dict.items()}
            else:
                # 全量微调：保存完整状态字典
                state_dict = get_model_state_dict(
                    model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
                state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

            if self.config.rank == 0:
                ckpt_dir = self.save_dir / f"step_{self.step}" / which / "transformer"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_file(state_dict_bf16, ckpt_dir / "diffusion_pytorch_model.safetensors")
                config_dict = dict(model.config)
                config_dict.pop("_name_or_path", None)
                # 保存 LoRA 元信息，方便恢复时重建 LoRA 结构
                if self.use_lora:
                    config_dict['use_lora'] = True
                    config_dict['lora_rank'] = self.config.lora_rank
                    config_dict['lora_alpha'] = self.config.lora_alpha
                    config_dict['lora_target_modules'] = self.config.lora_target_modules
                with open(ckpt_dir / "config.json", "w") as f:
                    json.dump(config_dict, f, indent=2)
                logger.info(f"  Saved {which} -> {ckpt_dir} ({'LoRA adapter' if self.use_lora else 'full model'})")

            # 保存判别器权重（仅在 use_dmd=True 且保存 online_student 时）
            if self.use_dmd and which == "online_student" and self.discriminator is not None:
                try:
                    disc_state = get_model_state_dict(
                        self.discriminator,
                        options=StateDictOptions(full_state_dict=True, cpu_offload=True))
                    disc_state_bf16 = {k: v.to(torch.bfloat16) for k, v in disc_state.items()}
                    if self.config.rank == 0:
                        disc_dir = self.save_dir / f"step_{self.step}" / "discriminator"
                        disc_dir.mkdir(parents=True, exist_ok=True)
                        save_file(disc_state_bf16, disc_dir / "discriminator.safetensors")
                        # 保存判别器优化器状态
                        torch.save(self.discriminator_optimizer.state_dict(),
                                   disc_dir / "discriminator_optimizer.pt")
                        logger.info(f"  Saved discriminator -> {disc_dir}")
                except Exception as e:
                    if self.config.rank == 0:
                        logger.warning(f"Failed to save discriminator: {e}")

            if dist.is_initialized():
                dist.barrier(device_ids=[torch.cuda.current_device()])  # 同步所有进程
        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save {which}: {e}")
                import traceback
                logger.error(traceback.format_exc())
            if dist.is_initialized():
                dist.barrier(device_ids=[torch.cuda.current_device()])

    # ==================================================================
    # 主训练循环
    # ==================================================================
    def train(self):
        """
        执行 Flow Map 蒸馏训练循环。

        与原始 FlashWAMDistiller.train() 的区别：
          1. 日志输出增加 Flow Map 相关信息（diffusion_ratio, consistency_ratio, epsilon 等）
          2. 损失日志增加 gt_regression_loss（GT 回归辅助损失）
          3. 训练步由 FlowMapStepMixin._train_step 提供（支持混合采样）

        训练流程：
          1. 遍历数据集，每个批次执行一步 Flow Map 蒸馏训练
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

        logger.info(f"Starting Flow Map {mode_str} distillation for {config.max_train_steps} steps ...")
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
        # Flow Map 特有日志
        logger.info(f"  diffusion_ratio  = {self.diffusion_ratio}")
        logger.info(f"  consistency_ratio = {self.consistency_ratio}")
        logger.info(f"  flowmap_ratio     = {self.flowmap_ratio}")
        logger.info(f"  epsilon           = {self.epsilon}")
        logger.info(f"  gt_regression_weight = {self.gt_regression_weight}")
        if self.use_dmd:
            logger.info(f"  DMD enabled: weight={getattr(config, 'dmd_weight', 0.1)}, "
                        f"rollout=[{getattr(config, 'dmd_rollout_steps_min', 2)}, "
                        f"{getattr(config, 'dmd_rollout_steps_max', 8)}]")

        self.optimizer.zero_grad()
        acc_losses = []              # 累积的总损失
        acc_video_losses = []        # 累积的视频损失
        acc_action_losses = []       # 累积的动作损失
        acc_action_aware_losses = [] # 累积的动作感知损失
        acc_gt_regression_losses = []  # 累积的 GT 回归损失（Flow Map 特有）
        acc_d_losses = []            # 累积的判别器损失（DMD 特有）
        acc_dmd_grad_norms = []      # 累积的 DMD 梯度范数（DMD 特有）
        step_in_acc = 0              # 当前累积步数
        # DMD 参数
        dmd_weight = getattr(config, 'dmd_weight', 0.1)
        dmd_warmup_steps = getattr(config, 'dmd_warmup_steps', 0)
        dmd_discriminator_warmup = getattr(config, 'dmd_discriminator_warmup', 200)

        progress_bar = tqdm(
            total=config.max_train_steps,
            desc="FlowMap", disable=(config.rank != 0),
            leave=True, dynamic_ncols=True, initial=self.step,
        )

        while self.step < config.max_train_steps:
            # 获取下一个数据批次
            batch = self._get_next_batch()

            # ---- 第一阶段：Flow Map 蒸馏（始终执行）----
            result = self._train_step(batch, step_in_acc)

            # 累积损失值
            acc_losses.append(result["loss"])
            acc_video_losses.append(result["video_loss"])
            acc_action_losses.append(result["action_loss"])
            acc_action_aware_losses.append(result["action_aware_loss"])
            acc_gt_regression_losses.append(result.get(
                "gt_regression_loss", torch.tensor(0.0, device=self.device)))
            step_in_acc += 1

            # ---- 第二阶段：DMD（条件满足时执行）----
            d_loss_val = torch.tensor(0.0, device=self.device)
            dmd_grad_norm = 0.0
            use_dmd_now = (self.use_dmd
                           and self.distill_action
                           and self.step >= dmd_warmup_steps
                           and result["should_sync"])

            if use_dmd_now:
                if self.step >= dmd_warmup_steps + dmd_discriminator_warmup:
                    # 判别器预热完成：执行完整 DMD 训练
                    # _dmd_train_step 内部已完成：student backward + discriminator update
                    d_loss_val, dmd_loss_val = self._dmd_train_step(batch)
                    dmd_grad_norm = dmd_loss_val.item()  # 用 dmd_loss 值代替 grad_norm
                else:
                    # 判别器预热期间：只训判别器，不注入 DMD 梯度
                    with torch.no_grad():
                        fake_action = self._on_policy_rollout(batch)
                    # 将 DTensor 转换为普通 Tensor（FSDP 包装的 student 输出是 DTensor）
                    from discriminator import train_discriminator_step
                    from flowmap_step import _to_regular_tensor
                    d_loss_val = train_discriminator_step(
                        self.discriminator,
                        real_actions=_to_regular_tensor(batch['actions']),
                        fake_actions=_to_regular_tensor(fake_action.detach()),
                        video_latent=_to_regular_tensor(batch['latents']),
                        text_emb=_to_regular_tensor(batch['text_emb']),
                    )
                    d_loss_val.backward()
                    self.discriminator_optimizer.step()
                    self.discriminator_optimizer.zero_grad()

            acc_d_losses.append(d_loss_val.detach() if isinstance(d_loss_val, torch.Tensor) else torch.tensor(d_loss_val, device=self.device))
            acc_dmd_grad_norms.append(torch.tensor(dmd_grad_norm, device=self.device))

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

                    # EMA 更新只在正常参数更新后执行（跳过 NaN/Inf 梯度时不更新 EMA，
                    # 避免将异常梯度产生的错误权重传播到目标学生）
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
                avg_gt_regression_loss = dist_mean(
                    torch.stack(acc_gt_regression_losses).sum()).item()
                avg_d_loss = dist_mean(torch.stack(acc_d_losses).sum()).item()
                avg_dmd_grad_norm = dist_mean(torch.stack(acc_dmd_grad_norms).sum()).item()
                # 重置累积器
                acc_losses = []
                acc_video_losses = []
                acc_action_losses = []
                acc_action_aware_losses = []
                acc_gt_regression_losses = []
                acc_d_losses = []
                acc_dmd_grad_norms = []
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
                    postfix["gt"] = f"{avg_gt_regression_loss:.4f}"
                    log_dict["loss/gt_regression"] = avg_gt_regression_loss
                    # DMD 日志
                    if self.use_dmd and self.step >= dmd_warmup_steps:
                        postfix["d"] = f"{avg_d_loss:.4f}"
                        log_dict["loss/discriminator"] = avg_d_loss
                        if avg_dmd_grad_norm > 0:
                            postfix["dmd_g"] = f"{avg_dmd_grad_norm:.4f}"
                            log_dict["dmd/grad_norm"] = avg_dmd_grad_norm
                    progress_bar.set_postfix(postfix)
                    if config.enable_wandb and HAS_WANDB:
                        wandb.log(log_dict, step=self.step)
                    if self.tb_writer is not None:
                        for key, value in log_dict.items():
                            self.tb_writer.add_scalar(key, value, self.step)

                self.step += 1

                # 定期保存检查点
                if self.step % config.save_interval == 0:
                    self._save_checkpoint("online_student")
                    self._save_checkpoint("target_student")

            # 分布式同步屏障
            if dist.is_initialized():
                dist.barrier(device_ids=[torch.cuda.current_device()])

        progress_bar.close()
        logger.info("Flow Map distillation completed!")
        # 保存最终检查点
        self._save_checkpoint("online_student")
        self._save_checkpoint("target_student")
        # 关闭 TensorBoard writer
        if self.tb_writer is not None:
            self.tb_writer.close()
