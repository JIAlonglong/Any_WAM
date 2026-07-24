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
import torch.nn as nn
import math
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import save_file

from distillation_flowmap.samplers import build_stage2_sampler
from distributed.fsdp import shard_model, apply_ac
from wan_va.distributed.fsdp import shard_model_fsdp1
from distributed.util import _configure_model, dist_mean
from modules.utils import WanVAEStreamingWrapper, load_transformer, load_vae
from utils import logger, warmup_constant_lambda, FlowMatchScheduler
from distillation_flowmap.cosmos_policy_adapter import CosmosPolicyActionTeacher
from distillation_flowmap.cosmos_progressive_opd import (
    select_progressive_training_objective,
    should_stop_training_at_step,
)
from distillation_flowmap.cosmos_deployment_rollout import (
    deployment_joint_step_for_update,
)
from distillation_flowmap.cosmos_teacher_roles import resolve_teacher_roles
from distillation_flowmap.ablation.robotwin_diagnostics import (
    classify_parameter_branch,
    opd_diagnostic_aliases,
)

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


def _select_progressive_training_objective(
    config,
    *,
    step,
    deployment_enabled,
    raw_auxiliary_enabled,
):
    return select_progressive_training_objective(
        step=step,
        deployment_enabled=deployment_enabled,
        deployment_interval=int(getattr(
            config, "deployment_joint_rollout_interval", 4
        )),
        raw_auxiliary_enabled=raw_auxiliary_enabled,
        raw_auxiliary_warmup=int(getattr(config, "opd_aux_warmup_steps", 0)),
        raw_auxiliary_interval=int(getattr(config, "opd_aux_interval", 8)),
        raw_auxiliary_phase=int(getattr(config, "opd_aux_phase", 2)),
    )

# DMD 判别器（仅在 use_dmd=True 时导入）
try:
    from discriminator import ActionDiscriminator
    HAS_DISCRIMINATOR = True
except ImportError:
    HAS_DISCRIMINATOR = False


def _iter_flowmap_checkpointing_targets(student):
    stack = [student]
    seen = set()
    while stack:
        module = stack.pop()
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        yield module
        for attr in ("module", "_fsdp_wrapped_module"):
            child = getattr(module, attr, None)
            if child is not None and child is not module:
                stack.append(child)


def _call_with_student_checkpointing(student, enabled, fn):
    old_states = []
    for module in _iter_flowmap_checkpointing_targets(student):
        has_attr = hasattr(module, "_flowmap_gradient_checkpointing")
        old_states.append((
            module,
            has_attr,
            getattr(module, "_flowmap_gradient_checkpointing", None),
        ))
        module._flowmap_gradient_checkpointing = bool(enabled)
    try:
        return fn()
    finally:
        for module, has_attr, old_enabled in reversed(old_states):
            if not has_attr:
                try:
                    delattr(module, "_flowmap_gradient_checkpointing")
                except AttributeError:
                    pass
            else:
                module._flowmap_gradient_checkpointing = old_enabled


def _resolve_use_fsdp1(config):
    use_fsdp1 = bool(getattr(config, 'use_fsdp1', False))
    use_onpolicy = bool(getattr(config, 'use_onpolicy_transition', False))
    risky_opd_checkpoint = (
        bool(getattr(config, 'use_opd_aux', False))
        and bool(getattr(config, 'gradient_checkpointing', False))
    )

    if use_onpolicy:
        use_fsdp1 = True
    if risky_opd_checkpoint and not use_fsdp1:
        logger.warning(
            "Forcing FSDP1 because use_opd_aux=True with "
            "gradient_checkpointing=True can mix Tensor/DTensor during "
            "checkpoint recompute under FSDP2."
        )
        use_fsdp1 = True

    config.use_fsdp1 = use_fsdp1
    return use_fsdp1


def _is_dtensor_param(param):
    try:
        from torch.distributed.tensor import DTensor
    except Exception:
        return False
    return isinstance(param, DTensor)


def _assert_no_dtensor_params(module, name):
    offenders = [
        param_name
        for param_name, param in module.named_parameters()
        if _is_dtensor_param(param)
    ]
    if offenders:
        preview = ", ".join(offenders[:8])
        extra = "" if len(offenders) <= 8 else f", ... (+{len(offenders) - 8})"
        raise RuntimeError(
            f"{name} is expected to use FSDP1 but still has DTensor "
            f"parameters: {preview}{extra}. Check use_fsdp1/FSDP wrapping."
        )


def _set_video_channel_config_from_heads(config_dict, model=None, state_dict=None):
    """Persist channel config from the actual video head shapes."""
    patch_size = config_dict.get("patch_size")
    if patch_size is None and model is not None:
        patch_size = getattr(model, "patch_size", None)
    if patch_size is None:
        return config_dict

    patch_size = list(patch_size)
    patch_volume = math.prod(patch_size)
    if patch_volume <= 0:
        return config_dict

    in_features = None
    out_features = None
    if state_dict is not None:
        in_weight = state_dict.get("patch_embedding_mlp.weight")
        out_weight = state_dict.get("proj_out.weight")
        if in_weight is not None and len(in_weight.shape) >= 2:
            in_features = int(in_weight.shape[1])
        if out_weight is not None and len(out_weight.shape) >= 1:
            out_features = int(out_weight.shape[0])

    if model is not None:
        patch_embedding_mlp = getattr(model, "patch_embedding_mlp", None)
        proj_out = getattr(model, "proj_out", None)
        if in_features is None and patch_embedding_mlp is not None:
            in_features = int(getattr(patch_embedding_mlp, "in_features", 0) or 0)
        if out_features is None and proj_out is not None:
            out_features = int(getattr(proj_out, "out_features", 0) or 0)

    config_dict["patch_size"] = patch_size
    if in_features and in_features % patch_volume == 0:
        config_dict["in_channels"] = in_features // patch_volume
    if out_features and out_features % patch_volume == 0:
        config_dict["out_channels"] = out_features // patch_volume
    return config_dict


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
        self.teacher_backend = str(getattr(config, 'teacher_backend', 'wanva')).lower()
        self.is_cosmos_policy_teacher = self.teacher_backend in (
            'cosmos', 'cosmos_policy', 'cosmos-policy')
        self.teacher_roles = resolve_teacher_roles(config)
        self.video_teacher = None
        self._teacher_nofsdp = None
        self._video_teacher_nofsdp = None
        self.cosmos_video_target = bool(getattr(config, 'cosmos_video_target', False))
        self.cosmos_latent_target = bool(getattr(config, 'cosmos_latent_target', False))
        self.cosmos_video_cdiff_aux = bool(
            getattr(config, 'cosmos_video_cdiff_aux', False)
        )
        self.cosmos_video_vae = None
        self.cosmos_video_streaming_vae = None
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
        # Action block 总权重：显式控制动作分支整体梯度强度
        self.action_block_weight = getattr(config, 'action_block_weight', 1.0)
        # LoRA 微调开关
        self.use_lora = getattr(config, 'use_lora', False)

        # ==============================================================
        # DMD（On-Policy Distribution Matching Distillation）参数
        # ==============================================================
        self.use_dmd = getattr(config, 'use_dmd', False)
        self.use_onpolicy_transition = getattr(config, 'use_onpolicy_transition', False)
        self.use_opd_aux = getattr(config, 'use_opd_aux', False)
        self.use_fsdp1 = _resolve_use_fsdp1(config)
        self.opd_aux_variant = str(getattr(config, 'opd_aux_variant', 'default')).lower()
        if self.opd_aux_variant not in ('default', 'kto_paopd', 'kto_paopd_norm_focal'):
            raise ValueError(
                f"Unsupported opd_aux_variant={self.opd_aux_variant!r}; "
                "expected 'default', 'kto_paopd', or 'kto_paopd_norm_focal'."
            )
        self.target_student = None
        self.offline_eval_skip_target_student = bool(
            getattr(config, 'offline_eval_skip_target_student', False)
        )
        self.offline_eval_use_fsdp_teacher = bool(
            getattr(config, 'offline_eval_use_fsdp_teacher', False)
        )
        self.skip_target_student = (
            bool(getattr(config, 'skip_target_student_for_cosmos_latent', False))
            or self.offline_eval_skip_target_student
        )
        cosmos_latent_opd_without_target_student = (
            self.cosmos_latent_target
            and self.use_opd_aux
            and str(getattr(
                config,
                'opd_teacher_target_mode',
                'student_state',
            )).lower() in (
                'cosmos_latent_student_state',
                'cosmos_latent_velocity',
                'cosmos_latent_full',
            )
            and not bool(getattr(config, 'opd_aux_action', False))
        )
        if self.skip_target_student and not self.offline_eval_skip_target_student:
            blockers = []
            if not self.cosmos_latent_target:
                blockers.append("cosmos_latent_target=False")
            if getattr(config, 'use_action_distill', True):
                blockers.append("use_action_distill=True")
            if getattr(config, 'action_use_flowmap', False):
                blockers.append("action_use_flowmap=True")
            if self.use_dmd:
                blockers.append("use_dmd=True")
            if self.use_onpolicy_transition:
                blockers.append("use_onpolicy_transition=True")
            if self.use_opd_aux and not cosmos_latent_opd_without_target_student:
                blockers.append("use_opd_aux=True")
            if getattr(config, 'opd_aux_use_nofsdp_rollout', False):
                blockers.append("opd_aux_use_nofsdp_rollout=True")
            if blockers:
                raise ValueError(
                    "skip_target_student_for_cosmos_latent is only valid for "
                    "the pure Cosmos latent target path without EMA/target-student "
                    f"losses; blockers: {', '.join(blockers)}"
                )
        self.discriminator = None
        self.discriminator_optimizer = None

        # WandB 日志初始化（仅主进程）
        if config.enable_wandb and HAS_WANDB and config.rank == 0:
            from datetime import datetime
            _wandb_name = getattr(config, "wandb_run_name", None) or getattr(config, "wandb_name_prefix", "stage1") + "_rank%d_%s" % (config.lora_rank, datetime.now().strftime("%Y%m%d_%H%M"))
            wandb.init(
                project="flowmap_distill_lingbot_va",
                entity=getattr(config, "wandb_entity", None),
                name=_wandb_name,
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
        self._last_ema_decay = float(getattr(config, 'ema_decay', 0.995))

        if config.rank == 0:
            logger.info(f"Flow Map Distiller 初始化")
            logger.info(f"LCM stride k = {self.k} "
                        f"(num_train_timesteps={config.num_train_timesteps}, "
                        f"num_ddim_timesteps={config.num_ddim_timesteps})")
            logger.info(f"Distill video: {self.distill_video}")
            logger.info(f"Distill action: {self.distill_action}")
            logger.info(f"Action aware: {self.action_aware}")
            logger.info(f"Action teacher backend: {self.teacher_roles.action_backend}")
            logger.info(f"Video teacher backend: {self.teacher_roles.video_backend}")
            logger.info(f"Video teacher path: {self.teacher_roles.video_model_path}")
            logger.info(f"Cosmos video target: {self.cosmos_video_target}")
            logger.info(f"Cosmos latent target: {self.cosmos_latent_target}")
            logger.info(f"Cosmos video central-diff aux: {self.cosmos_video_cdiff_aux}")
            logger.info(f"Skip target student: {self.skip_target_student}")
            if self.distill_action:
                logger.info(f"  k_action = {self.k_action}, "
                            f"action_loss_weight = {config.action_loss_weight}, "
                            f"mode = {self.action_distill_mode}")
                logger.info(f"  action_block_weight = {self.action_block_weight}")
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
            logger.info(f"  use_opd_aux        = {self.use_opd_aux} (teacher-transition auxiliary)")
            logger.info(f"  use_fsdp1          = {self.use_fsdp1} (FSDP backend guard)")
            if self.use_opd_aux:
                logger.info(f"  opd_aux_variant    = {self.opd_aux_variant}")
                logger.info(f"  opd_aux_weight     = {getattr(config, 'opd_aux_weight', 0.1)}")
                logger.info(f"  opd_aux_interval   = {getattr(config, 'opd_aux_interval', 1)}")
                logger.info(f"  flowmap_aux_weight = {getattr(config, 'flowmap_aux_weight', 1.0)}")
                logger.info(f"  endpoint_aux_weight= {getattr(config, 'opd_endpoint_aux_weight', 0.0)}")
                logger.info(f"  same_state_velocity_weight= {getattr(config, 'opd_same_state_velocity_weight', 0.0)}")
                logger.info(f"  opd_target_mode    = {getattr(config, 'opd_teacher_target_mode', 'student_state')}")
                logger.info(f"  opd_grad_mode      = {getattr(config, 'opd_rollout_grad_mode', 'endpoint')}")
                logger.info(f"  opd_aux_gradient_checkpointing = {getattr(config, 'opd_aux_gradient_checkpointing', False)}")
                logger.info(f"  opd_aux_empty_cache = {getattr(config, 'opd_aux_empty_cache', False)}")
                logger.info(f"  action_transition_block = {getattr(config, 'action_transition_block_weight', getattr(config, 'action_block_weight', 1.0))}")
                logger.info(f"  action_local_fm_block   = {getattr(config, 'action_local_fm_block_weight', 1.0)}")
                logger.info(f"  action_local_fm_weight  = {getattr(config, 'action_aware_weight', 0.0)}")
                logger.info(f"  opd_transition_group_weight = {getattr(config, 'opd_transition_group_weight', 1.0)}")
                logger.info(f"  opd_anchor_cap_ratio = {getattr(config, 'opd_anchor_cap_ratio', -1.0)}")
                logger.info(f"  opd_query_bias     = {getattr(config, 'opd_query_bias', 'none')}")
                logger.info(f"  opd_query_ratio    = {getattr(config, 'opd_query_bias_ratio', 0.0)}")
                logger.info(f"  opd_low_noise_max  = {getattr(config, 'opd_low_noise_max_sigma', 0.25)}")
            if self.use_lora:
                logger.info(f"  lora_rank          = {config.lora_rank}")
                logger.info(f"  lora_alpha         = {config.lora_alpha}")
                logger.info(f"  lora_target_modules= {config.lora_target_modules}")

        # ==============================================================
        # 三个模型的初始化
        # ==============================================================
        if self.is_cosmos_policy_teacher:
            student_base_model_path = getattr(config, 'student_base_model_path', None)
            if student_base_model_path is None:
                raise ValueError(
                    "teacher_backend='cosmos_policy' requires cfg.student_base_model_path "
                    "pointing to a WanVA teacher/checkpoint root for student initialization."
                )
            student_base_model_path = os.path.abspath(os.path.expanduser(student_base_model_path))
            if os.path.basename(student_base_model_path) == "transformer":
                teacher_path = student_base_model_path
            else:
                teacher_path = os.path.join(student_base_model_path, "transformer")
            if not os.path.isfile(os.path.join(teacher_path, "config.json")):
                raise FileNotFoundError(
                    "Invalid student_base_model_path for Cosmos Policy backend: "
                    f"expected {os.path.join(teacher_path, 'config.json')}"
                )
        else:
            teacher_path = os.path.join(config.teacher_model_path, "transformer")

        # 1. 教师模型（frozen）：预训练好的 LingBot-VA，不参与训练
        #    注意：教师模型保持原始结构，不添加 delta_embedder
        if bool(getattr(config, 'offline_eval_skip_teacher', False)):
            self.teacher = None
            self._teacher_nofsdp = None
            logger.info("Skipping teacher load for offline eval cache replay.")
        elif self.is_cosmos_policy_teacher:
            logger.info("Loading Cosmos Policy action teacher metadata ...")
            self.teacher = CosmosPolicyActionTeacher(
                config.teacher_model_path, dtype=self.dtype, device="cpu", config=config)
            if bool(getattr(config, 'cosmos_policy_validate_weights', False)):
                metadata = self.teacher.load_state_metadata()
                if config.rank == 0:
                    logger.info(
                        "Cosmos Policy checkpoint readable: %s keys, sample=%s",
                        metadata["num_keys"],
                        metadata["sample_keys"],
                    )
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self._teacher_nofsdp = self.teacher.to(f"cuda:{local_rank}")
            self.teacher.requires_grad_(False).eval()
            logger.info(
                "Cosmos Policy action teacher ready; WanVA student base: %s",
                teacher_path,
            )
            if self.cosmos_latent_target:
                if not bool(getattr(config, 'cosmos_policy_use_raw_inference', False)):
                    raise ValueError(
                        "cfg.cosmos_latent_target=True requires "
                        "cfg.cosmos_policy_use_raw_inference=True."
                    )
                logger.info(
                    "Cosmos latent central-diff target enabled; WanVA VAE video "
                    "encoding is skipped for this path."
                )
            if self.cosmos_video_target:
                if not bool(getattr(config, 'cosmos_policy_use_raw_inference', False)):
                    raise ValueError(
                        "cfg.cosmos_video_target=True requires "
                        "cfg.cosmos_policy_use_raw_inference=True."
                    )
                vae_root = getattr(config, 'cosmos_video_vae_model_path', None)
                if vae_root is None:
                    vae_root = getattr(config, 'student_base_model_path', None)
                vae_root = os.path.abspath(os.path.expanduser(vae_root))
                vae_path = (
                    vae_root if os.path.basename(vae_root) == "vae"
                    else os.path.join(vae_root, "vae")
                )
                if not os.path.isdir(vae_path):
                    raise FileNotFoundError(
                        "Invalid cosmos_video_vae_model_path: "
                        f"expected VAE directory at {vae_path}"
                    )
                logger.info("Loading WanVA VAE encoder for Cosmos future video targets ...")
                self.cosmos_video_vae = load_vae(
                    vae_path,
                    torch_dtype=self.dtype,
                    torch_device=f"cuda:{local_rank}",
                )
                self.cosmos_video_vae.requires_grad_(False)
                self.cosmos_video_vae.eval()
                self.cosmos_video_streaming_vae = WanVAEStreamingWrapper(self.cosmos_video_vae)
                logger.info("WanVA VAE encoder ready for Cosmos future video targets.")
                if self.cosmos_video_cdiff_aux:
                    video_root = getattr(
                        config, 'cosmos_video_cdiff_teacher_model_path', None)
                    if video_root is None:
                        video_root = getattr(config, 'student_base_model_path', None)
                    video_root = os.path.abspath(os.path.expanduser(video_root))
                    if os.path.basename(video_root) == "transformer":
                        video_teacher_path = video_root
                    else:
                        video_teacher_path = os.path.join(video_root, "transformer")
                    if not os.path.isfile(os.path.join(video_teacher_path, "config.json")):
                        raise FileNotFoundError(
                            "Invalid cosmos_video_cdiff_teacher_model_path: "
                            f"expected {os.path.join(video_teacher_path, 'config.json')}"
                        )
                    logger.info(
                        "Loading WanVA video teacher for Cosmos central-diff aux ..."
                    )
                    self.video_teacher = load_transformer(
                        video_teacher_path,
                        torch_dtype=self.dtype,
                        torch_device="cpu",
                    )
                    self.video_teacher.requires_grad_(False)
                    self.video_teacher.eval()
                    self.video_teacher = self.video_teacher.to(self.dtype)
                    self._video_teacher_nofsdp = self.video_teacher.to(
                        f"cuda:{local_rank}")
                    self.video_teacher = None
                    logger.info(
                        "WanVA video teacher ready for Cosmos central-diff aux."
                    )
            if self.teacher_roles.uses_separate_video_teacher:
                video_root = os.path.abspath(os.path.expanduser(self.teacher_roles.video_model_path))
                if os.path.basename(video_root) == "transformer":
                    video_teacher_path = video_root
                else:
                    video_teacher_path = os.path.join(video_root, "transformer")
                if not os.path.isfile(os.path.join(video_teacher_path, "config.json")):
                    raise FileNotFoundError(
                        "Invalid video_teacher_model_path for Cosmos dual-teacher mode: "
                        f"expected {os.path.join(video_teacher_path, 'config.json')}"
                    )
                logger.info("Loading WanVA video teacher for Cosmos dual-teacher mode ...")
                self.video_teacher = load_transformer(
                    video_teacher_path,
                    torch_dtype=self.dtype,
                    torch_device="cpu",
                )
                self.video_teacher.requires_grad_(False)
                self.video_teacher.eval()
                self.video_teacher = self.video_teacher.to(self.dtype)
                self._video_teacher_nofsdp = self.video_teacher.to(f"cuda:{local_rank}")
                self.video_teacher = None
                logger.info("WanVA video teacher ready for Cosmos dual-teacher mode.")
        else:
            logger.info("Loading teacher (frozen) ...")
            self.teacher = load_transformer(teacher_path, torch_dtype=self.dtype, torch_device="cpu")
            self.teacher.requires_grad_(False)  # 冻结所有参数
            self.teacher.eval()                 # 评估模式
            self.teacher = self.teacher.to(self.dtype)

            if self.offline_eval_use_fsdp_teacher:
                logger.info(
                    "Offline eval uses the FSDP teacher; skipping non-FSDP teacher copy."
                )
            else:
                # 创建非 FSDP 教师副本（用于纯推理 forward，避免 FSDP all-gather 通信开销）
                # 教师 frozen + eval，不需要梯度同步，非 FSDP forward 结果完全一致
                import copy
                logger.info("Creating non-FSDP teacher copy for inference ...")
                self._teacher_nofsdp = copy.deepcopy(self.teacher)
                self._teacher_nofsdp = self._teacher_nofsdp.to(self.dtype)
                self._teacher_nofsdp.eval()
                for p in self._teacher_nofsdp.parameters():
                    p.requires_grad_(False)
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                self._teacher_nofsdp = self._teacher_nofsdp.to(f"cuda:{local_rank}")
                logger.info(f"Non-FSDP teacher copy created (on cuda:{local_rank}).")
            if self.use_fsdp1:
                # FSDP1 avoids DTensor/checkpoint recompute type mixing.
                self.teacher = shard_model_fsdp1(self.teacher, param_dtype=self.dtype)
                logger.info("Teacher wrapped with FSDP1")
            else:
                # FSDP2 mode
                self.teacher = _configure_model(
                    model=self.teacher, shard_fn=shard_model,
                    param_dtype=self.dtype, device=self.device, eval_mode=True,
                )
                if not getattr(config, 'skip_teacher_compile', False):
                    logger.info("Compiling teacher model with torch.compile ...")
                    self.teacher = torch.compile(self.teacher, mode="default")
                    logger.info("Teacher model compiled.")
            # 释放 FSDP teacher（推理用 _teacher_nofsdp），省 ~10GB/卡
            if self._teacher_nofsdp is not None:
                del self.teacher
                self.teacher = None
                logger.info("FSDP teacher released (inference uses non-FSDP copy).")
            elif self.offline_eval_use_fsdp_teacher:
                logger.info("FSDP teacher retained for offline eval.")

        # 确定学生模型的初始化路径（从检查点恢复或从教师初始化）
        resume_path = getattr(config, "resume_from_path", None)
        resume_step = getattr(config, "resume_from_step", None)
        self._resume_ckpt_dir = None
        self._is_lora_resume = False  # 标记是否从 LoRA checkpoint 恢复
        self._nofsdp_synced = False  # dirty flag for _sync_student_nofsdp optimization
        self._recent_ckpts = []     # track last 3 checkpoint steps for cleanup
        self._best_loss = float('inf')
        self._best_step = 0

        if resume_path is not None:
            # 从指定路径恢复
            self._resume_ckpt_dir = resume_path
            student_path = os.path.join(resume_path, "online_student", "transformer")
            target_path = os.path.join(resume_path, "target_student", "transformer")
            self.step = resume_step if resume_step is not None else 0
            reset_resume_step = getattr(config, "reset_resume_step", False)
            # 检查是否是 LoRA checkpoint
            student_config_path = os.path.join(student_path, "config.json")
            if os.path.exists(student_config_path):
                with open(student_config_path) as f:
                    ckpt_config = json.load(f)
                ckpt_step = int(ckpt_config.get("checkpoint_step", 0))
                if (not reset_resume_step) and resume_step is not None and resume_step != ckpt_step:
                    raise ValueError(
                        f"resume_from_step={resume_step} does not match "
                        f"checkpoint_step={ckpt_step} in {student_config_path}"
                    )
                if reset_resume_step:
                    self.step = 0
                elif resume_step is None:
                    self.step = ckpt_step
                if ckpt_config.get('use_lora', False):
                    self._is_lora_resume = True
                    if config.rank == 0:
                        logger.info(f"Detected LoRA checkpoint, will load base model from teacher and apply LoRA adapters")
                if self._is_lora_resume and not self.use_lora:
                    raise ValueError(
                        "Checkpoint was saved with use_lora=True, but current config.use_lora=False. "
                        "Set use_lora=True to resume this checkpoint."
                    )
            else:
                raise FileNotFoundError(f"Missing checkpoint config: {student_config_path}")
            resume_online_from_target = bool(getattr(config, "resume_online_from_target", False))
            if resume_online_from_target:
                target_config_path = os.path.join(target_path, "config.json")
                if not os.path.exists(target_config_path):
                    raise FileNotFoundError(f"Missing target checkpoint config: {target_config_path}")
                student_path = target_path
            if config.rank == 0:
                logger.info(f"Resuming from path: {resume_path}")
                logger.info(f"  Online student: {student_path}")
                logger.info(f"  Target student: {target_path}")
                if resume_online_from_target:
                    logger.info("  Initializing online student from Stage-1 target/EMA checkpoint")
                if reset_resume_step:
                    logger.info(f"  Resetting resumed checkpoint step to 0 for a new training stage")
                logger.info(f"  Starting step: {self.step}")
        elif resume_step is not None:
            # 从输出目录的指定步数恢复
            self._resume_ckpt_dir = os.path.join(config.output_dir, "checkpoints", f"step_{resume_step}")
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
                with open(student_config_path) as f:
                    ckpt_config = json.load(f)
                ckpt_step = int(ckpt_config.get("checkpoint_step", resume_step))
                if ckpt_step != resume_step:
                    raise ValueError(
                        f"resume_from_step={resume_step} does not match "
                        f"checkpoint_step={ckpt_step} in {student_config_path}"
                    )
                if ckpt_config.get('use_lora', False):
                    self._is_lora_resume = True
                    if config.rank == 0:
                        logger.info(f"Detected LoRA checkpoint, will load base model from teacher and apply LoRA adapters")
                if self._is_lora_resume and not self.use_lora:
                    raise ValueError(
                        "Checkpoint was saved with use_lora=True, but current config.use_lora=False. "
                        "Set use_lora=True to resume this checkpoint."
                    )
            else:
                raise FileNotFoundError(f"Missing checkpoint config: {student_config_path}")
            resume_online_from_target = bool(getattr(config, "resume_online_from_target", False))
            if resume_online_from_target:
                target_config_path = os.path.join(target_path, "config.json")
                if not os.path.exists(target_config_path):
                    raise FileNotFoundError(f"Missing target checkpoint config: {target_config_path}")
                student_path = target_path
            if config.rank == 0:
                logger.info(f"Resuming from step {resume_step}")
                logger.info(f"  Online student: {student_path}")
                logger.info(f"  Target student: {target_path}")
                if resume_online_from_target:
                    logger.info("  Initializing online student from target/EMA checkpoint")
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
        # apply_ac disabled - checkpointing + batched CFG = mask mismatch during backward
        # apply_ac(self.student)
        self.student = self.student.to(self.dtype)

        # 为学生模型添加 Flow Map 能力
        # setup_flowmap_model: 将 condition_embedder 替换为 WanTwoTimeTextImageEmbedding（含 delta_embedder）
        # patch_model_forward: 给 forward / forward_train / _time_embed 添加 r_timestep 参数支持
        logger.info("Setting up Flow Map for online student ...")
        self.student = setup_flowmap_model(
            self.student, gate_value=config.gate_value, deltatime_type=config.deltatime_type)
        self.student = patch_model_forward(self.student)
        self.student._flowmap_gradient_checkpointing = bool(
            getattr(config, "gradient_checkpointing", True)
        )
        self.student._flowmap_force_gradient_checkpointing = bool(
            getattr(config, "offline_eval_force_gradient_checkpointing", False)
        )
        if config.rank == 0:
            logger.info(
                "Student gradient checkpointing: %s",
                self.student._flowmap_gradient_checkpointing,
            )
            if self.student._flowmap_force_gradient_checkpointing:
                logger.info("Student force gradient checkpointing enabled for offline eval.")

        def load_flowmap_delta_weights(model, transformer_dir, label):
            """Restore FlowMap-only weights dropped by the base Wan loader."""
            if transformer_dir == teacher_path:
                return
            weights_path = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors")
            if not os.path.exists(weights_path):
                return
            from safetensors import safe_open

            delta_state = {}
            with safe_open(weights_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if ".delta_embedder." in key:
                        delta_state[key] = f.get_tensor(key)
            if not delta_state:
                if config.rank == 0:
                    logger.warning("No FlowMap delta weights found for %s at %s", label, weights_path)
                return
            _, unexpected = model.load_state_dict(delta_state, strict=False)
            if config.rank == 0:
                logger.info("Loaded %d FlowMap delta weights for %s", len(delta_state), label)
                if unexpected:
                    logger.warning("Unexpected FlowMap delta keys for %s: %s", label, unexpected)

        if not self._is_lora_resume:
            load_flowmap_delta_weights(self.student, student_path, "online student")

        def adapt_cosmos_latent_video_heads(model, label):
            if not self.cosmos_latent_target:
                return model
            channels = int(getattr(config, "cosmos_latent_channels", 16))
            patch_dim = channels * math.prod(model.patch_size)
            old_in = model.patch_embedding_mlp
            old_out = model.proj_out
            if old_in.in_features == patch_dim and old_out.out_features == patch_dim:
                return model

            new_in = nn.Linear(
                patch_dim,
                old_in.out_features,
                bias=old_in.bias is not None,
                device=old_in.weight.device,
                dtype=old_in.weight.dtype,
            )
            new_out = nn.Linear(
                old_out.in_features,
                patch_dim,
                bias=old_out.bias is not None,
                device=old_out.weight.device,
                dtype=old_out.weight.dtype,
            )
            with torch.no_grad():
                new_in.weight.zero_()
                copy_cols = min(patch_dim, old_in.in_features)
                new_in.weight[:, :copy_cols].copy_(old_in.weight[:, :copy_cols])
                if old_in.bias is not None:
                    new_in.bias.copy_(old_in.bias)

                new_out.weight.zero_()
                copy_rows = min(patch_dim, old_out.out_features)
                new_out.weight[:copy_rows].copy_(old_out.weight[:copy_rows])
                if old_out.bias is not None:
                    new_out.bias.zero_()
                    new_out.bias[:copy_rows].copy_(old_out.bias[:copy_rows])

            model.patch_embedding_mlp = new_in
            model.proj_out = new_out
            if hasattr(model, "config"):
                try:
                    model.config.in_channels = channels
                    model.config.out_channels = channels
                except Exception:
                    pass
            if config.rank == 0:
                logger.info(
                    "Adapted %s video heads for Cosmos latent channels: "
                    "input %d->%d, output %d->%d",
                    label,
                    old_in.in_features,
                    patch_dim,
                    old_out.out_features,
                    patch_dim,
                )
            return model

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
        self.student = adapt_cosmos_latent_video_heads(self.student, "online student")

        # 创建非 FSDP 学生副本（用于 DMD rollout 和 on-policy transition rollout）
        if (self.use_dmd or self.use_onpolicy_transition or
                getattr(config, 'opd_aux_use_nofsdp_rollout', False)):
            import copy
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
            logger.info("Creating non-FSDP student copy for DMD rollout ...")
            self._student_nofsdp = copy.deepcopy(self.student)
            # 解包 CheckpointWrapper（避免分布式环境中触发 DTensor dispatch）
            _unwrapped = 0
            for i, block in enumerate(self._student_nofsdp.blocks):
                if isinstance(block, CheckpointWrapper):
                    self._student_nofsdp.blocks[i] = block._checkpoint_wrapped_module
                    _unwrapped += 1
            logger.info(f"Unwrapped {_unwrapped} CheckpointWrapper blocks in non-FSDP copy.")
            self._student_nofsdp.eval()
            for p in self._student_nofsdp.parameters():
                p.requires_grad_(False)
            # 对 nofsdp 副本也应用 patch_model_forward，否则闭包引用原始 FSDP 模型
            self._student_nofsdp = patch_model_forward(self._student_nofsdp)
            # 移动到正确的 CUDA 设备（deepcopy 时模型在 CPU 上）
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self._student_nofsdp = self._student_nofsdp.to(f"cuda:{local_rank}")
            logger.info(f"Non-FSDP student copy created (patched, on cuda:{local_rank}).")
        else:
            self._student_nofsdp = None

        if self.use_fsdp1:
            # FSDP1 shards parameters without DTensor dispatch in checkpoint recompute.
            # apply_ac disabled - checkpointing + batched CFG = mask mismatch during backward
            # apply_ac(self.student)
            self.student = shard_model_fsdp1(self.student, param_dtype=self.dtype)
            _assert_no_dtensor_params(self.student, "student")
            logger.info("Student wrapped with FSDP1")
        else:
            self.student = _configure_model(
                model=self.student, shard_fn=shard_model,
                param_dtype=self.dtype, device=self.device, eval_mode=False,
            )
        self.student.train()                # 训练模式
        self.student.requires_grad_(True)   # 启用梯度计算

        # torch.compile 加速（可选）
        self._student_blocks_compiled = False
        if getattr(config, 'use_torch_compile', False):
            logger.info("Compiling student transformer blocks with torch.compile ...")
            for i, block in enumerate(self.student.blocks):
                self.student.blocks[i] = torch.compile(block, mode="reduce-overhead")
            logger.info(f"Compiled {len(self.student.blocks)} blocks.")
            self._student_blocks_compiled = True
        # 3. 目标学生（EMA, frozen）：在线学生的指数移动平均副本
        # 如果从 LoRA checkpoint 恢复，需要先加载基座模型，再加载 LoRA adapter
        if self.skip_target_student:
            logger.info(
                "Skipping target student load/EMA."
            )
        else:
            if self._is_lora_resume:
                logger.info("Loading target student from LoRA checkpoint ...")
                logger.info("  Step 1: Loading base model from teacher ...")
                self.target_student = load_transformer(teacher_path, torch_dtype=self.dtype, torch_device="cpu")
            else:
                logger.info("Loading target student (EMA, frozen) ...")
                self.target_student = load_transformer(target_path, torch_dtype=self.dtype, torch_device="cpu")
            self.target_student = self.target_student.to(self.dtype)
            self.target_student = adapt_cosmos_latent_video_heads(self.target_student, "target student")

            # 为目标学生模型添加 Flow Map 能力（与学生模型相同的改造）
            logger.info("Setting up Flow Map for target student ...")
            self.target_student = setup_flowmap_model(
                self.target_student, gate_value=config.gate_value, deltatime_type=config.deltatime_type)
            self.target_student = patch_model_forward(self.target_student)
            if not self._is_lora_resume:
                load_flowmap_delta_weights(self.target_student, target_path, "target student")

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

            if self.use_fsdp1:
                # FSDP1 avoids DTensor/checkpoint recompute type mixing.
                self.target_student = shard_model_fsdp1(self.target_student, param_dtype=self.dtype)
                _assert_no_dtensor_params(self.target_student, "target_student")
                logger.info("Target student wrapped with FSDP1")
            else:
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
        # 8-bit Adam (省 ~50% optimizer 显存)
        use_8bit_optimizer = getattr(config, 'use_8bit_optimizer', False)
        if use_8bit_optimizer:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                [p for p in self.student.parameters() if p.requires_grad],
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=1e-8,
                weight_decay=config.weight_decay,
            )
            if config.rank == 0:
                logger.info("Using 8-bit AdamW optimizer (bitsandbytes)")
        else:
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
        # Cosine decay: warmup then decay to 10% of peak
        def cosine_decay_lambda(step):
            warmup = config.warmup_steps
            total = config.max_train_steps
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=cosine_decay_lambda,
        )
        optimizer_state_path = None
        scheduler_state_path = None
        resume_optimizer_state = getattr(config, "resume_optimizer_state", True)
        if self._resume_ckpt_dir is not None and resume_optimizer_state:
            optimizer_state_path = os.path.join(self._resume_ckpt_dir, "optimizer.pt")
            scheduler_state_path = os.path.join(self._resume_ckpt_dir, "lr_scheduler.pt")
        elif self._resume_ckpt_dir is not None and config.rank == 0:
            logger.info(
                "Skipping optimizer/LR scheduler state restore; "
                "model weights and checkpoint step are still restored."
            )

        if optimizer_state_path is not None and os.path.exists(optimizer_state_path):
            opt_state = torch.load(optimizer_state_path, map_location="cpu")
            self.optimizer.load_state_dict(opt_state)
            if config.rank == 0:
                logger.info(f"Loaded optimizer state from {optimizer_state_path}")

        if scheduler_state_path is not None and os.path.exists(scheduler_state_path):
            sched_state = torch.load(scheduler_state_path, map_location="cpu")
            self.lr_scheduler.load_state_dict(sched_state)
            if config.rank == 0:
                logger.info(f"Loaded LR scheduler state from {scheduler_state_path}, "
                            f"lr={self.lr_scheduler.get_last_lr()[0]:.2e}")
        # 如果从旧检查点恢复且没有单独的 scheduler 状态，则快速推进学习率调度器
        elif self.step > 0:
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
        train_sampler = build_stage2_sampler(train_dataset, config)
        if config.rank == 0:
            logger.info(
                "Stage2 sampler: %s (group_by=%s)",
                getattr(config, "stage2_sampler", "default"),
                getattr(config, "stage2_group_by", "task"),
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
        self._light_eval_batches = None
        self._stage1_start_eval_batches = None

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
        if model is None:
            if self.config.rank == 0:
                logger.info(f"  Skipped {which} checkpoint because target student is disabled.")
            if dist.is_initialized():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            return
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
                step_dir = self.save_dir / f"step_{self.step}"
                ckpt_dir = step_dir / which / "transformer"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_file(state_dict_bf16, ckpt_dir / "diffusion_pytorch_model.safetensors")
                config_dict = dict(model.config)
                config_dict.pop("_name_or_path", None)
                # Persist FlowMap/runtime metadata needed by inference and resume scripts.
                config_dict['gate_value'] = getattr(self.config, 'gate_value', 0.0)
                config_dict['deltatime_type'] = getattr(self.config, 'deltatime_type', 'r')
                config_dict['num_train_timesteps'] = getattr(self.config, 'num_train_timesteps', 1000)
                config_dict['snr_shift'] = getattr(self.config, 'snr_shift', 1.0)
                config_dict['action_snr_shift'] = getattr(self.config, 'action_snr_shift', 1.0)
                config_dict['patch_size'] = list(getattr(self.config, 'patch_size', (1, 2, 2)))
                config_dict['distill_mode'] = getattr(self.config, 'distill_mode', 'flashwam')
                config_dict['checkpoint_step'] = self.step
                _set_video_channel_config_from_heads(config_dict, model, state_dict_bf16)
                # 保存 LoRA 元信息，方便恢复时重建 LoRA 结构
                if self.use_lora:
                    config_dict['use_lora'] = True
                    config_dict['lora_rank'] = self.config.lora_rank
                    config_dict['lora_alpha'] = self.config.lora_alpha
                    config_dict['lora_target_modules'] = self.config.lora_target_modules
                with open(ckpt_dir / "config.json", "w") as f:
                    json.dump(config_dict, f, indent=2)
                logger.info(f"  Saved {which} -> {ckpt_dir} ({'LoRA adapter' if self.use_lora else 'full model'})")

                if which == "online_student":
                    torch.save(self.optimizer.state_dict(), step_dir / "optimizer.pt")
                    torch.save(self.lr_scheduler.state_dict(), step_dir / "lr_scheduler.pt")
                    logger.info(f"  Saved optimizer/scheduler -> {step_dir}")

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
            raise

    # ==================================================================
    # 轻量级固定样本 eval：不跑环境、不跑 teacher，只评估学生重建能力
    # ==================================================================
    def _light_eval_is_enabled(self):
        return bool(getattr(self.config, "enable_light_eval", False))

    def _compute_student_grad_branch_norms(self):
        sq_sums = {
            "video": torch.zeros((), device=self.device),
            "action": torch.zeros((), device=self.device),
            "shared": torch.zeros((), device=self.device),
        }
        for name, param in self.student.named_parameters():
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            if grad.is_sparse:
                grad = grad.coalesce().values()
            branch = classify_parameter_branch(name)
            sq_sums[branch] = sq_sums[branch] + grad.detach().float().pow(2).sum()

        packed = torch.stack([sq_sums["video"], sq_sums["action"], sq_sums["shared"]])
        if dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed = packed.clamp(min=0).sqrt()
        return {
            "video": packed[0].item(),
            "action": packed[1].item(),
            "shared": packed[2].item(),
        }

    def _get_light_eval_batches(self):
        if self._light_eval_batches is not None:
            return self._light_eval_batches

        from torch.utils.data._utils.collate import default_collate

        dataset = self.train_loader.dataset
        num_batches = max(1, int(getattr(self.config, "light_eval_num_batches", 1)))
        start_index = int(getattr(self.config, "light_eval_start_index", 0))
        batches = []
        for i in range(num_batches):
            idx = (start_index + i) % len(dataset)
            # batch_size is intentionally fixed to 1: the model/FlexAttention
            # path has several batch=1 assumptions elsewhere in training.
            batches.append(default_collate([dataset[idx]]))
        self._light_eval_batches = batches
        return batches

    def _get_stage1_start_eval_batches(self):
        if self._stage1_start_eval_batches is not None:
            return self._stage1_start_eval_batches

        from torch.utils.data._utils.collate import default_collate

        dataset = self.train_loader.dataset
        num_batches = max(1, int(getattr(self.config, "stage1_start_eval_num_batches", 1)))
        start_index = int(getattr(self.config, "stage1_start_eval_start_index", 0))
        batches = []
        for i in range(num_batches):
            idx = (start_index + i) % len(dataset)
            batches.append(default_collate([dataset[idx]]))
        self._stage1_start_eval_batches = batches
        return batches

    def _light_eval_pairs(self):
        pairs = getattr(self.config, "light_eval_pairs", None)
        if not pairs:
            pairs = [(1000, 1000), (1000, 0), (750, 250)]
        return [(float(t), float(r)) for t, r in pairs]

    def _rollout_eval_is_enabled(self):
        return bool(getattr(self.config, "enable_rollout_eval", False))

    def _rollout_eval_pairs(self):
        pairs = getattr(self.config, "rollout_eval_pairs", None)
        if not pairs:
            pairs = [(1000, 0), (1000, 500), (750, 250)]
        return [(float(t), float(r)) for t, r in pairs]

    def _rollout_eval_student_steps(self):
        steps = getattr(self.config, "rollout_eval_student_steps", None)
        if not steps:
            steps = [1, 2, 4]
        return [max(1, int(s)) for s in steps]

    @torch.no_grad()
    def _run_light_eval(self):
        """
        Deterministic validation on fixed train/val samples.

        Metrics:
          - *_v_mse: predicted FlowMatch velocity vs fixed-noise target.
          - *_xr_mse/l1: one Euler jump from x_t to x_r vs true x_r built
            from the same clean sample and fixed noise.

        This intentionally avoids teacher forward and environment rollout so it
        can run inside training without changing optimizer/EMA state.
        """
        if not self._light_eval_is_enabled():
            return {}

        was_training = self.student.training
        self.student.eval()

        metric_sums = {}
        metric_counts = {}
        seed_base = int(getattr(self.config, "light_eval_seed", 42))
        action_ds = getattr(self.config, "action_downsample_factor", 4)

        def add_metric(name, value):
            value = value.detach().float()
            metric_sums[name] = metric_sums.get(name, torch.zeros((), device=self.device)) + value
            metric_counts[name] = metric_counts.get(name, 0) + 1

        try:
            for batch_idx, batch in enumerate(self._get_light_eval_batches()):
                batch = self._move_eval_batch_to_device(batch)
                base_input = self._prepare_base_dict(batch)
                B = batch["latents"].shape[0]
                ref_shape = batch["latents"].shape
                num_frames = ref_shape[2]

                for pair_idx, (t_value, r_value) in enumerate(self._light_eval_pairs()):
                    gen = torch.Generator(device=self.device)
                    gen.manual_seed(seed_base + batch_idx * 1009 + pair_idx)

                    video_t = torch.full(
                        (B, num_frames), t_value, device=self.device, dtype=torch.float32)
                    video_r = torch.full(
                        (B, num_frames), r_value, device=self.device, dtype=torch.float32)
                    action_t = video_t
                    action_r = video_r

                    video_noise = torch.randn(
                        batch["latents"].shape, device=self.device,
                        dtype=batch["latents"].dtype, generator=gen)
                    action_noise = torch.randn(
                        batch["actions"].shape, device=self.device,
                        dtype=batch["actions"].dtype, generator=gen)

                    video_noisy_t = self.train_scheduler_latent.add_noise(
                        batch["latents"], video_noise, video_t, t_dim=2)
                    video_noisy_r = self.train_scheduler_latent.add_noise(
                        batch["latents"], video_noise, video_r, t_dim=2)
                    video_v_target = self.train_scheduler_latent.training_target(
                        batch["latents"], video_noise, video_t)

                    action_noisy_t = self.train_scheduler_action.add_noise(
                        batch["actions"], action_noise, action_t, t_dim=2)
                    action_noisy_r = self.train_scheduler_action.add_noise(
                        batch["actions"], action_noise, action_r, t_dim=2)
                    action_v_target = self.train_scheduler_action.training_target(
                        batch["actions"], action_noise, action_t)

                    eval_input = {
                        "latent_dict": {
                            **base_input["latent_dict"],
                            "noisy_latents": video_noisy_t,
                            "timesteps": video_t,
                            "targets": video_v_target,
                        },
                        "action_dict": {
                            "noisy_latents": action_noisy_t[:, :, ::action_ds],
                            "latent": batch["actions"][:, :, ::action_ds],
                            "timesteps": action_t[:, ::action_ds],
                            "cond_timesteps": base_input["action_dict"]["cond_timesteps"][:, ::action_ds],
                            "text_emb": base_input["action_dict"]["text_emb"],
                            "targets": action_v_target[:, :, ::action_ds],
                        },
                        "chunk_size": base_input["chunk_size"],
                        "window_size": base_input["window_size"],
                    }
                    if base_input["action_dict"].get("grid_id") is not None:
                        from flowmap_step import _downsample_action_grid_id
                        eval_input["action_dict"]["grid_id"] = _downsample_action_grid_id(
                            base_input["action_dict"]["grid_id"], batch["actions"], action_ds)
                    if base_input["action_dict"].get("actions_mask") is not None:
                        eval_input["action_dict"]["actions_mask"] = (
                            base_input["action_dict"]["actions_mask"][:, :, ::action_ds])

                    from modules.model import FlexAttnFunc
                    _ld = eval_input["latent_dict"]
                    _ad = eval_input["action_dict"]
                    _total_length = (
                        _ld["noisy_latents"].flatten(0, 1).shape[0] * 2 +
                        _ad["noisy_latents"].flatten(0, 1).shape[0] * 2
                    )
                    _padded_length = (128 - _total_length % 128) % 128
                    FlexAttnFunc.init_mask(
                        _ld["noisy_latents"].shape,
                        _ad["noisy_latents"].shape,
                        _padded_length,
                        eval_input["chunk_size"],
                        window_size=eval_input["window_size"],
                        patch_size=self.patch_size,
                        device=self.device,
                    )

                    student_video_seq, student_action_seq = self.student(
                        eval_input, train_mode=True,
                        r_timestep=video_r,
                        action_r_timestep=action_r[:, ::action_ds],
                    )

                    pair_name = f"t{int(t_value)}_r{int(r_value)}"
                    sigma_t = video_t[:, None, :, None, None] / self.config.num_train_timesteps
                    sigma_r = video_r[:, None, :, None, None] / self.config.num_train_timesteps

                    if self.distill_video:
                        student_video_v = self._extract_video_v(student_video_seq, ref_shape, B)
                        video_pred_xr = video_noisy_t + student_video_v * (sigma_r - sigma_t)
                        add_metric(f"light_eval/{pair_name}/video_v_mse",
                                   (student_video_v.float() - video_v_target.float()).pow(2).mean())
                        add_metric(f"light_eval/{pair_name}/video_xr_mse",
                                   (video_pred_xr.float() - video_noisy_r.float()).pow(2).mean())
                        add_metric(f"light_eval/{pair_name}/video_xr_l1",
                                   (video_pred_xr.float() - video_noisy_r.float()).abs().mean())

                    if self.distill_action or self.action_aware:
                        action_frames = batch["actions"].shape[2] // action_ds
                        student_action_v = self._extract_action_v(student_action_seq, action_frames)
                        action_v_target_ds = action_v_target[:, :, ::action_ds]
                        action_noisy_t_ds = action_noisy_t[:, :, ::action_ds]
                        action_noisy_r_ds = action_noisy_r[:, :, ::action_ds]
                        action_sigma_t = action_t[:, None, ::action_ds, None, None] / self.config.num_train_timesteps
                        action_sigma_r = action_r[:, None, ::action_ds, None, None] / self.config.num_train_timesteps
                        action_pred_xr = action_noisy_t_ds + student_action_v * (
                            action_sigma_r - action_sigma_t)

                        mask = batch.get("actions_mask")
                        if mask is not None:
                            mask = mask[:, :, ::action_ds].float()
                            denom = (mask.sum() * student_action_v.shape[1]).clamp(min=1)
                            action_v_mse = (
                                ((student_action_v.float() - action_v_target_ds.float()) * mask).pow(2).sum()
                                / denom
                            )
                            action_xr_mse = (
                                ((action_pred_xr.float() - action_noisy_r_ds.float()) * mask).pow(2).sum()
                                / denom
                            )
                            action_xr_l1 = (
                                ((action_pred_xr.float() - action_noisy_r_ds.float()).abs() * mask).sum()
                                / denom
                            )
                        else:
                            action_v_mse = (student_action_v.float() - action_v_target_ds.float()).pow(2).mean()
                            action_xr_mse = (action_pred_xr.float() - action_noisy_r_ds.float()).pow(2).mean()
                            action_xr_l1 = (action_pred_xr.float() - action_noisy_r_ds.float()).abs().mean()

                        add_metric(f"light_eval/{pair_name}/action_v_mse", action_v_mse)
                        add_metric(f"light_eval/{pair_name}/action_xr_mse", action_xr_mse)
                        add_metric(f"light_eval/{pair_name}/action_xr_l1", action_xr_l1)

        finally:
            if was_training:
                self.student.train()

        out = {}
        for name, total in metric_sums.items():
            avg = total / max(1, metric_counts[name])
            if dist.is_initialized():
                avg = dist_mean(avg)
            out[name] = avg.item()
        return out

    @torch.no_grad()
    def _run_rollout_eval(self):
        """
        Fixed-sample rollout eval for Stage 2.

        This is heavier than _run_light_eval because it queries teacher rollout.
        It measures whether a small number of student Euler steps lands near the
        teacher/reference trajectory. Use a coarse interval such as 250 or 500.
        """
        if not self._rollout_eval_is_enabled():
            return {}

        from modules.model import FlexAttnFunc
        from flowmap_step import _downsample_action_grid_id

        was_training = self.student.training
        self.student.eval()
        if self.target_student is not None:
            self.target_student.eval()
        if getattr(self, "_student_nofsdp", None) is not None:
            self._student_nofsdp.eval()

        metric_sums = {}
        metric_counts = {}
        summary_sums = {}
        summary_counts = {}
        seed_base = int(getattr(self.config, "rollout_eval_seed", 42))
        action_ds = getattr(self.config, "action_downsample_factor", 4)
        teacher_steps = max(1, int(getattr(self.config, "rollout_eval_teacher_steps", 4)))
        cfg_scale = float(getattr(self.config, "rollout_eval_cfg_scale", 5.0))
        pairs = self._rollout_eval_pairs()
        student_steps = self._rollout_eval_student_steps()
        max_batches = max(1, int(getattr(self.config, "rollout_eval_num_batches", 1)))
        save_videos = (
            self.config.rank == 0 and
            bool(getattr(self.config, "rollout_eval_save_videos", False))
        )
        video_dir = None
        video_vae = None
        video_processor = None
        video_pairs_saved = 0
        video_max_pairs = max(1, int(getattr(self.config, "rollout_eval_video_max_pairs", 1)))
        video_sample_index = max(0, int(getattr(self.config, "rollout_eval_video_sample_index", 0)))
        video_fps = max(1, int(getattr(self.config, "rollout_eval_video_fps", 10)))
        if save_videos:
            from diffusers.video_processor import VideoProcessor
            from modules.utils import load_vae

            video_dir = getattr(self.config, "rollout_eval_video_dir", None)
            if video_dir is None:
                video_dir = os.path.join(
                    self.config.output_dir,
                    "rollout_eval_videos",
                    f"step_{int(getattr(self, 'step', 0)):06d}",
                )
            os.makedirs(video_dir, exist_ok=True)
            video_vae = load_vae(
                os.path.join(self.config.teacher_model_path, "vae"),
                torch_dtype=torch.float16,
                torch_device=self.device,
            )
            video_vae.eval()
            video_processor = VideoProcessor(vae_scale_factor=1)

        def add_metric(name, value):
            value = value.detach().float()
            metric_sums[name] = metric_sums.get(name, torch.zeros((), device=self.device)) + value
            metric_counts[name] = metric_counts.get(name, 0) + 1

        def add_summary(k_steps, name, value):
            key = f"rollout_eval_summary/s{k_steps}/{name}"
            value = value.detach().float()
            summary_sums[key] = summary_sums.get(key, torch.zeros((), device=self.device)) + value
            summary_counts[key] = summary_counts.get(key, 0) + 1

        def init_eval_mask(input_dict):
            ld = input_dict["latent_dict"]
            ad = input_dict["action_dict"]
            total_length = (
                ld["noisy_latents"].flatten(0, 1).shape[0] * 2 +
                ad["noisy_latents"].flatten(0, 1).shape[0] * 2
            )
            padded_length = (128 - total_length % 128) % 128
            FlexAttnFunc.init_mask(
                ld["noisy_latents"].shape,
                ad["noisy_latents"].shape,
                padded_length,
                input_dict["chunk_size"],
                window_size=input_dict["window_size"],
                patch_size=self.patch_size,
                device=self.device,
            )

        def masked_mse_l1(diff, mask):
            if mask is None:
                return diff.float().pow(2).mean(), diff.float().abs().mean()
            mask = mask.float()
            denom = (mask.sum() * diff.shape[1]).clamp(min=1)
            return (
                (diff.float() * mask).pow(2).sum() / denom,
                (diff.float().abs() * mask).sum() / denom,
            )

        def decode_eval_latents(latents):
            latents = latents.detach().to(next(video_vae.parameters()).device, dtype=video_vae.dtype)
            latents_mean = (
                torch.tensor(video_vae.config.latents_mean)
                .view(1, video_vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std_inv = (
                1.0
                / torch.tensor(video_vae.config.latents_std)
                .view(1, video_vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = latents / latents_std_inv + latents_mean
            video = video_vae.decode(latents, return_dict=False)[0]
            return video_processor.postprocess_video(video, output_type="np")[0]

        def save_eval_contact_sheet(named_videos, path):
            if not named_videos:
                return
            import numpy as np
            from PIL import Image, ImageDraw

            def to_uint8(frame):
                arr = np.asarray(frame)
                if arr.dtype != np.uint8:
                    arr = np.clip(arr * 255.0 if arr.max() <= 1.5 else arr, 0, 255).astype(np.uint8)
                return arr

            first_video = next(iter(named_videos.values()))
            frame_count = len(first_video)
            frame_ids = sorted(set([0, frame_count // 2, frame_count - 1]))
            thumb_w = 192
            rows = []
            for name, video in named_videos.items():
                thumbs = []
                for frame_id in frame_ids:
                    img = Image.fromarray(to_uint8(video[frame_id])).convert("RGB")
                    scale = thumb_w / img.width
                    img = img.resize((thumb_w, max(1, int(img.height * scale))))
                    thumbs.append(img)
                rows.append((name, thumbs))

            label_h = 24
            gap = 6
            row_h = label_h + max(img.height for _, row in rows for img in row)
            sheet_w = len(frame_ids) * thumb_w + (len(frame_ids) - 1) * gap
            sheet_h = len(rows) * row_h + (len(rows) - 1) * gap
            sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
            draw = ImageDraw.Draw(sheet)
            y = 0
            for name, row in rows:
                draw.text((0, y + 4), name, fill=(0, 0, 0))
                x = 0
                for img in row:
                    sheet.paste(img, (x, y + label_h))
                    x += thumb_w + gap
                y += row_h + gap
            sheet.save(path)

        try:
            for batch_idx, batch in enumerate(self._get_light_eval_batches()[:max_batches]):
                batch = self._move_eval_batch_to_device(batch)
                base_input = self._prepare_base_dict(batch)
                B = batch["latents"].shape[0]
                ref_shape = batch["latents"].shape
                num_frames = ref_shape[2]
                empty_emb = self.empty_emb.expand(
                    base_input["latent_dict"]["text_emb"].shape[0], -1, -1)

                for pair_idx, (t_value, r_value) in enumerate(pairs):
                    gen = torch.Generator(device=self.device)
                    gen.manual_seed(seed_base + batch_idx * 1009 + pair_idx)

                    video_t = torch.full(
                        (B, num_frames), t_value, device=self.device, dtype=torch.float32)
                    video_r = torch.full(
                        (B, num_frames), r_value, device=self.device, dtype=torch.float32)
                    action_t = video_t
                    action_r = video_r

                    video_noise = torch.randn(
                        batch["latents"].shape, device=self.device,
                        dtype=batch["latents"].dtype, generator=gen)
                    action_noise = torch.randn(
                        batch["actions"].shape, device=self.device,
                        dtype=batch["actions"].dtype, generator=gen)

                    video_noisy_t = self.train_scheduler_latent.add_noise(
                        batch["latents"], video_noise, video_t, t_dim=2)
                    video_noisy_r = self.train_scheduler_latent.add_noise(
                        batch["latents"], video_noise, video_r, t_dim=2)
                    video_v_target_r = self.train_scheduler_latent.training_target(
                        batch["latents"], video_noise, video_r)

                    action_noisy_t = self.train_scheduler_action.add_noise(
                        batch["actions"], action_noise, action_t, t_dim=2)
                    action_noisy_r = self.train_scheduler_action.add_noise(
                        batch["actions"], action_noise, action_r, t_dim=2)
                    action_v_target_t = self.train_scheduler_action.training_target(
                        batch["actions"], action_noise, action_t)

                    input_dict = {
                        "latent_dict": {
                            **base_input["latent_dict"],
                            "noisy_latents": video_noisy_t,
                            "timesteps": video_t,
                            "targets": self.train_scheduler_latent.training_target(
                                batch["latents"], video_noise, video_t),
                        },
                        "action_dict": {
                            **base_input["action_dict"],
                            "noisy_latents": action_noisy_t,
                            "timesteps": action_t,
                            "targets": action_v_target_t,
                        },
                        "chunk_size": base_input["chunk_size"],
                        "window_size": base_input["window_size"],
                    }
                    student_input = {
                        "latent_dict": input_dict["latent_dict"],
                        "action_dict": {
                            "noisy_latents": action_noisy_t[:, :, ::action_ds],
                            "latent": batch["actions"][:, :, ::action_ds],
                            "timesteps": action_t[:, ::action_ds],
                            "cond_timesteps": input_dict["action_dict"]["cond_timesteps"][:, ::action_ds],
                            "text_emb": input_dict["action_dict"]["text_emb"],
                            "grid_id": _downsample_action_grid_id(
                                input_dict["action_dict"].get("grid_id"), batch["actions"], action_ds),
                            "actions_mask": (
                                input_dict["action_dict"]["actions_mask"][:, :, ::action_ds]
                                if input_dict["action_dict"].get("actions_mask") is not None else None
                            ),
                        },
                        "chunk_size": input_dict["chunk_size"],
                        "window_size": input_dict["window_size"],
                    }

                    pair_name = f"t{int(t_value)}_r{int(r_value)}"
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
                    videos_to_save = {}
                    should_save_pair_video = (
                        save_videos and batch_idx == 0 and video_pairs_saved < video_max_pairs
                    )
                    if should_save_pair_video:
                        sample_idx = min(video_sample_index, B - 1)
                        videos_to_save["gt_r"] = video_noisy_r[sample_idx:sample_idx + 1].detach().cpu()
                        videos_to_save[f"teacher_t{teacher_steps}"] = (
                            teacher_x_r[sample_idx:sample_idx + 1].detach().cpu()
                        )

                    for k_steps in student_steps:
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
                            K_steps=k_steps,
                            action_target_r=action_r,
                        )
                        if should_save_pair_video:
                            sample_idx = min(video_sample_index, B - 1)
                            videos_to_save[f"student_s{k_steps}"] = (
                                student_x_r[sample_idx:sample_idx + 1].detach().cpu()
                            )
                        prefix = f"rollout_eval/{pair_name}/s{k_steps}_t{teacher_steps}"

                        video_teacher_x_mse = (
                            student_x_r.float() - teacher_x_r.float()).pow(2).mean()
                        video_teacher_x_l1 = (
                            student_x_r.float() - teacher_x_r.float()).abs().mean()
                        video_teacher_v_mse = (
                            student_v_r.float() - teacher_v_r.float()).pow(2).mean()
                        video_teacher_v_l1 = (
                            student_v_r.float() - teacher_v_r.float()).abs().mean()
                        video_gt_x_mse = (
                            student_x_r.float() - video_noisy_r.float()).pow(2).mean()
                        video_gt_x_l1 = (
                            student_x_r.float() - video_noisy_r.float()).abs().mean()
                        video_gt_v_mse = (
                            student_v_r.float() - video_v_target_r.float()).pow(2).mean()

                        add_metric(prefix + "/video_teacher_x_mse", video_teacher_x_mse)
                        add_metric(prefix + "/video_teacher_x_l1", video_teacher_x_l1)
                        add_metric(prefix + "/video_teacher_v_mse", video_teacher_v_mse)
                        add_metric(prefix + "/video_teacher_v_l1", video_teacher_v_l1)
                        add_metric(prefix + "/video_gt_x_mse", video_gt_x_mse)
                        add_metric(prefix + "/video_gt_x_l1", video_gt_x_l1)
                        add_metric(prefix + "/video_gt_v_mse", video_gt_v_mse)
                        add_metric(prefix + "/video_latent_norm", student_x_r.float().pow(2).mean().sqrt())
                        add_summary(k_steps, "video_teacher_x_l1_mean", video_teacher_x_l1)
                        add_summary(k_steps, "video_gt_x_l1_mean", video_gt_x_l1)

                        action_input = {
                            "latent_dict": {
                                **input_dict["latent_dict"],
                                "noisy_latents": student_x_r.detach(),
                                "timesteps": video_r,
                            },
                            "action_dict": student_input["action_dict"],
                            "chunk_size": input_dict["chunk_size"],
                            "window_size": input_dict["window_size"],
                        }
                        init_eval_mask(action_input)
                        _, student_action_seq = self.student(
                            action_input, train_mode=True,
                            r_timestep=video_r,
                            action_r_timestep=action_r[:, ::action_ds],
                        )
                        action_frames = batch["actions"].shape[2] // action_ds
                        student_action_v = self._extract_action_v(student_action_seq, action_frames)
                        action_noisy_t_ds = action_noisy_t[:, :, ::action_ds]
                        action_noisy_r_ds = action_noisy_r[:, :, ::action_ds]
                        action_v_target_ds = action_v_target_t[:, :, ::action_ds]
                        action_sigma_t = (
                            action_t[:, None, ::action_ds, None, None] /
                            self.config.num_train_timesteps)
                        action_sigma_r = (
                            action_r[:, None, ::action_ds, None, None] /
                            self.config.num_train_timesteps)
                        action_pred_xr = action_noisy_t_ds + student_action_v * (
                            action_sigma_r.to(student_action_v) -
                            action_sigma_t.to(student_action_v))
                        mask = batch.get("actions_mask")
                        mask_ds = mask[:, :, ::action_ds] if mask is not None else None

                        action_gt_xr_mse, action_gt_xr_l1 = masked_mse_l1(
                            action_pred_xr - action_noisy_r_ds, mask_ds)
                        action_gt_v_mse, action_gt_v_l1 = masked_mse_l1(
                            student_action_v - action_v_target_ds, mask_ds)
                        add_metric(prefix + "/action_gt_xr_mse", action_gt_xr_mse)
                        add_metric(prefix + "/action_gt_xr_l1", action_gt_xr_l1)
                        add_metric(prefix + "/action_gt_v_mse", action_gt_v_mse)
                        add_metric(prefix + "/action_gt_v_l1", action_gt_v_l1)
                        add_metric(prefix + "/action_v_norm", student_action_v.float().pow(2).mean().sqrt())
                        if student_action_v.shape[2] > 1:
                            add_metric(
                                prefix + "/action_v_smoothness_l1",
                                (student_action_v[:, :, 1:] - student_action_v[:, :, :-1])
                                .float().abs().mean())
                        add_summary(k_steps, "action_gt_xr_l1_mean", action_gt_xr_l1)
                        add_summary(k_steps, "action_gt_v_l1_mean", action_gt_v_l1)

                    if should_save_pair_video:
                        from diffusers.utils import export_to_video

                        decoded = {}
                        for video_name, latent_cpu in videos_to_save.items():
                            video_np = decode_eval_latents(latent_cpu.to(self.device))
                            decoded[video_name] = video_np
                            path = os.path.join(video_dir, f"{pair_name}_{video_name}.mp4")
                            export_to_video(video_np, path, fps=video_fps)
                            logger.info("Saved rollout eval video: %s", path)
                        sheet_path = os.path.join(video_dir, f"{pair_name}_contact_sheet.png")
                        save_eval_contact_sheet(decoded, sheet_path)
                        logger.info("Saved rollout eval contact sheet: %s", sheet_path)
                        video_pairs_saved += 1
        finally:
            if video_vae is not None:
                del video_vae
                torch.cuda.empty_cache()
            if was_training:
                self.student.train()

        out = {}
        all_sums = {**metric_sums, **summary_sums}
        all_counts = {**metric_counts, **summary_counts}
        for name, total in all_sums.items():
            avg = total / max(1, all_counts[name])
            if dist.is_initialized():
                avg = dist_mean(avg)
            out[name] = avg.item()
        return out

    @torch.no_grad()
    def _run_stage1_start_eval(self, model=None, metric_prefix="stage1_start_eval",
                               supports_r_timestep=True, transition_prefix=None):
        """
        One-shot Stage-1 quality snapshot before Stage-2 training.

        This checks local Flow Matching and one-step x0 reconstruction on fixed
        samples. It is intentionally diagnostic only: callers should log the
        result and continue training regardless of metric values or eval errors.
        """
        if not bool(getattr(self.config, "enable_stage1_start_eval", False)):
            return {}

        model = model if model is not None else self.student
        was_training = model.training
        model.eval()

        metric_sums = {}
        metric_counts = {}
        seed_base = int(getattr(self.config, "stage1_start_eval_seed", 42))
        timesteps = getattr(self.config, "stage1_start_eval_timesteps",
                            [1000, 750, 500, 250])
        timesteps = [float(t) for t in timesteps]
        transition_pairs = getattr(
            self.config, "stage1_start_eval_transition_pairs",
            [(1000, 750), (1000, 500), (750, 250), (500, 0)]
        )
        transition_pairs = [(float(t), float(r)) for t, r in transition_pairs]
        action_ds = getattr(self.config, "action_downsample_factor", 4)

        def add_metric(name, value):
            value = value.detach().float()
            metric_sums[name] = metric_sums.get(name, torch.zeros((), device=self.device)) + value
            metric_counts[name] = metric_counts.get(name, 0) + 1

        try:
            for batch_idx, batch in enumerate(self._get_stage1_start_eval_batches()):
                batch = self._move_eval_batch_to_device(batch)
                base_input = self._prepare_base_dict(batch)
                B = batch["latents"].shape[0]
                ref_shape = batch["latents"].shape
                num_frames = ref_shape[2]

                for t_idx, t_value in enumerate(timesteps):
                    gen = torch.Generator(device=self.device)
                    gen.manual_seed(seed_base + batch_idx * 1009 + t_idx)

                    video_t = torch.full(
                        (B, num_frames), t_value, device=self.device, dtype=torch.float32)
                    video_r0 = torch.zeros_like(video_t)
                    action_t = video_t
                    action_r0 = video_r0

                    video_noise = torch.randn(
                        batch["latents"].shape, device=self.device,
                        dtype=batch["latents"].dtype, generator=gen)
                    action_noise = torch.randn(
                        batch["actions"].shape, device=self.device,
                        dtype=batch["actions"].dtype, generator=gen)

                    video_noisy_t = self.train_scheduler_latent.add_noise(
                        batch["latents"], video_noise, video_t, t_dim=2)
                    video_v_target = self.train_scheduler_latent.training_target(
                        batch["latents"], video_noise, video_t)

                    action_noisy_t = self.train_scheduler_action.add_noise(
                        batch["actions"], action_noise, action_t, t_dim=2)
                    action_v_target = self.train_scheduler_action.training_target(
                        batch["actions"], action_noise, action_t)

                    eval_input = {
                        "latent_dict": {
                            **base_input["latent_dict"],
                            "noisy_latents": video_noisy_t,
                            "timesteps": video_t,
                            "targets": video_v_target,
                        },
                        "action_dict": {
                            "noisy_latents": action_noisy_t[:, :, ::action_ds],
                            "latent": batch["actions"][:, :, ::action_ds],
                            "timesteps": action_t[:, ::action_ds],
                            "cond_timesteps": base_input["action_dict"]["cond_timesteps"][:, ::action_ds],
                            "text_emb": base_input["action_dict"]["text_emb"],
                            "targets": action_v_target[:, :, ::action_ds],
                        },
                        "chunk_size": base_input["chunk_size"],
                        "window_size": base_input["window_size"],
                    }
                    if base_input["action_dict"].get("grid_id") is not None:
                        from flowmap_step import _downsample_action_grid_id
                        eval_input["action_dict"]["grid_id"] = _downsample_action_grid_id(
                            base_input["action_dict"]["grid_id"], batch["actions"], action_ds)
                    if base_input["action_dict"].get("actions_mask") is not None:
                        eval_input["action_dict"]["actions_mask"] = (
                            base_input["action_dict"]["actions_mask"][:, :, ::action_ds])

                    from modules.model import FlexAttnFunc
                    _ld = eval_input["latent_dict"]
                    _ad = eval_input["action_dict"]
                    _total_length = (
                        _ld["noisy_latents"].flatten(0, 1).shape[0] * 2 +
                        _ad["noisy_latents"].flatten(0, 1).shape[0] * 2
                    )
                    _padded_length = (128 - _total_length % 128) % 128
                    FlexAttnFunc.init_mask(
                        _ld["noisy_latents"].shape,
                        _ad["noisy_latents"].shape,
                        _padded_length,
                        eval_input["chunk_size"],
                        window_size=eval_input["window_size"],
                        patch_size=self.patch_size,
                        device=self.device,
                    )

                    if supports_r_timestep:
                        student_video_seq, student_action_seq = model(
                            eval_input, train_mode=True,
                            r_timestep=video_r0,
                            action_r_timestep=action_r0[:, ::action_ds],
                        )
                    else:
                        student_video_seq, student_action_seq = model(
                            eval_input, train_mode=True,
                        )

                    t_name = f"t{int(t_value)}"
                    sigma_t = video_t[:, None, :, None, None] / self.config.num_train_timesteps
                    sigma_0 = torch.zeros_like(sigma_t)

                    if self.distill_video:
                        student_video_v = self._extract_video_v(student_video_seq, ref_shape, B)
                        video_pred_x0 = video_noisy_t + student_video_v * (sigma_0 - sigma_t)
                        add_metric(f"{metric_prefix}/{t_name}/video_v_mse",
                                   (student_video_v.float() - video_v_target.float()).pow(2).mean())
                        add_metric(f"{metric_prefix}/{t_name}/video_v_l1",
                                   (student_video_v.float() - video_v_target.float()).abs().mean())
                        add_metric(f"{metric_prefix}/{t_name}/video_x0_mse",
                                   (video_pred_x0.float() - batch["latents"].float()).pow(2).mean())
                        add_metric(f"{metric_prefix}/{t_name}/video_x0_l1",
                                   (video_pred_x0.float() - batch["latents"].float()).abs().mean())

                    if self.distill_action or self.action_aware:
                        action_frames = batch["actions"].shape[2] // action_ds
                        student_action_v = self._extract_action_v(student_action_seq, action_frames)
                        action_v_target_ds = action_v_target[:, :, ::action_ds]
                        action_noisy_t_ds = action_noisy_t[:, :, ::action_ds]
                        action_latent_ds = batch["actions"][:, :, ::action_ds]
                        action_sigma_t = action_t[:, None, ::action_ds, None, None] / self.config.num_train_timesteps
                        action_sigma_0 = torch.zeros_like(action_sigma_t)
                        action_pred_x0 = action_noisy_t_ds + student_action_v * (
                            action_sigma_0 - action_sigma_t)

                        mask = batch.get("actions_mask")
                        if mask is not None:
                            mask = mask[:, :, ::action_ds].float()
                            denom = (mask.sum() * student_action_v.shape[1]).clamp(min=1)
                            action_v_mse = (
                                ((student_action_v.float() - action_v_target_ds.float()) * mask).pow(2).sum()
                                / denom
                            )
                            action_v_l1 = (
                                ((student_action_v.float() - action_v_target_ds.float()).abs() * mask).sum()
                                / denom
                            )
                            action_x0_mse = (
                                ((action_pred_x0.float() - action_latent_ds.float()) * mask).pow(2).sum()
                                / denom
                            )
                            action_x0_l1 = (
                                ((action_pred_x0.float() - action_latent_ds.float()).abs() * mask).sum()
                                / denom
                            )
                        else:
                            action_v_mse = (student_action_v.float() - action_v_target_ds.float()).pow(2).mean()
                            action_v_l1 = (student_action_v.float() - action_v_target_ds.float()).abs().mean()
                            action_x0_mse = (action_pred_x0.float() - action_latent_ds.float()).pow(2).mean()
                            action_x0_l1 = (action_pred_x0.float() - action_latent_ds.float()).abs().mean()

                        add_metric(f"{metric_prefix}/{t_name}/action_v_mse", action_v_mse)
                        add_metric(f"{metric_prefix}/{t_name}/action_v_l1", action_v_l1)
                        add_metric(f"{metric_prefix}/{t_name}/action_x0_mse", action_x0_mse)
                        add_metric(f"{metric_prefix}/{t_name}/action_x0_l1", action_x0_l1)

                if transition_prefix is None:
                    continue

                for pair_idx, (t_value, r_value) in enumerate(transition_pairs):
                    gen = torch.Generator(device=self.device)
                    gen.manual_seed(seed_base + batch_idx * 1009 + 10000 + pair_idx)

                    video_t = torch.full(
                        (B, num_frames), t_value, device=self.device, dtype=torch.float32)
                    video_r = torch.full(
                        (B, num_frames), r_value, device=self.device, dtype=torch.float32)
                    action_t = video_t
                    action_r = video_r

                    video_noise = torch.randn(
                        batch["latents"].shape, device=self.device,
                        dtype=batch["latents"].dtype, generator=gen)
                    action_noise = torch.randn(
                        batch["actions"].shape, device=self.device,
                        dtype=batch["actions"].dtype, generator=gen)

                    video_noisy_t = self.train_scheduler_latent.add_noise(
                        batch["latents"], video_noise, video_t, t_dim=2)
                    video_noisy_r = self.train_scheduler_latent.add_noise(
                        batch["latents"], video_noise, video_r, t_dim=2)
                    video_v_target_t = self.train_scheduler_latent.training_target(
                        batch["latents"], video_noise, video_t)
                    video_v_target_eff = (
                        (video_noisy_r - video_noisy_t)
                        / ((video_r[:, None, :, None, None] - video_t[:, None, :, None, None])
                           / self.config.num_train_timesteps).clamp(min=-1.0, max=-1e-6)
                    )

                    action_noisy_t = self.train_scheduler_action.add_noise(
                        batch["actions"], action_noise, action_t, t_dim=2)
                    action_noisy_r = self.train_scheduler_action.add_noise(
                        batch["actions"], action_noise, action_r, t_dim=2)
                    action_v_target_t = self.train_scheduler_action.training_target(
                        batch["actions"], action_noise, action_t)
                    action_v_target_eff = (
                        (action_noisy_r - action_noisy_t)
                        / ((action_r[:, None, :, None, None] - action_t[:, None, :, None, None])
                           / self.config.num_train_timesteps).clamp(min=-1.0, max=-1e-6)
                    )

                    eval_input = {
                        "latent_dict": {
                            **base_input["latent_dict"],
                            "noisy_latents": video_noisy_t,
                            "timesteps": video_t,
                            "targets": video_v_target_t,
                        },
                        "action_dict": {
                            "noisy_latents": action_noisy_t[:, :, ::action_ds],
                            "latent": batch["actions"][:, :, ::action_ds],
                            "timesteps": action_t[:, ::action_ds],
                            "cond_timesteps": base_input["action_dict"]["cond_timesteps"][:, ::action_ds],
                            "text_emb": base_input["action_dict"]["text_emb"],
                            "targets": action_v_target_t[:, :, ::action_ds],
                        },
                        "chunk_size": base_input["chunk_size"],
                        "window_size": base_input["window_size"],
                    }
                    if base_input["action_dict"].get("grid_id") is not None:
                        from flowmap_step import _downsample_action_grid_id
                        eval_input["action_dict"]["grid_id"] = _downsample_action_grid_id(
                            base_input["action_dict"]["grid_id"], batch["actions"], action_ds)
                    if base_input["action_dict"].get("actions_mask") is not None:
                        eval_input["action_dict"]["actions_mask"] = (
                            base_input["action_dict"]["actions_mask"][:, :, ::action_ds])

                    from modules.model import FlexAttnFunc
                    _ld = eval_input["latent_dict"]
                    _ad = eval_input["action_dict"]
                    _total_length = (
                        _ld["noisy_latents"].flatten(0, 1).shape[0] * 2 +
                        _ad["noisy_latents"].flatten(0, 1).shape[0] * 2
                    )
                    _padded_length = (128 - _total_length % 128) % 128
                    FlexAttnFunc.init_mask(
                        _ld["noisy_latents"].shape,
                        _ad["noisy_latents"].shape,
                        _padded_length,
                        eval_input["chunk_size"],
                        window_size=eval_input["window_size"],
                        patch_size=self.patch_size,
                        device=self.device,
                    )

                    if supports_r_timestep:
                        student_video_seq, student_action_seq = model(
                            eval_input, train_mode=True,
                            r_timestep=video_r,
                            action_r_timestep=action_r[:, ::action_ds],
                        )
                    else:
                        student_video_seq, student_action_seq = model(
                            eval_input, train_mode=True,
                        )

                    pair_name = f"t{int(t_value)}_r{int(r_value)}"
                    sigma_t = video_t[:, None, :, None, None] / self.config.num_train_timesteps
                    sigma_r = video_r[:, None, :, None, None] / self.config.num_train_timesteps

                    if self.distill_video:
                        student_video_v = self._extract_video_v(student_video_seq, ref_shape, B)
                        video_pred_xr = video_noisy_t + student_video_v * (sigma_r - sigma_t)
                        add_metric(f"{transition_prefix}/{pair_name}/video_xr_mse",
                                   (video_pred_xr.float() - video_noisy_r.float()).pow(2).mean())
                        add_metric(f"{transition_prefix}/{pair_name}/video_xr_l1",
                                   (video_pred_xr.float() - video_noisy_r.float()).abs().mean())
                        add_metric(f"{transition_prefix}/{pair_name}/video_v_eff_mse",
                                   (student_video_v.float() - video_v_target_eff.float()).pow(2).mean())

                    if self.distill_action or self.action_aware:
                        action_frames = batch["actions"].shape[2] // action_ds
                        student_action_v = self._extract_action_v(student_action_seq, action_frames)
                        action_noisy_t_ds = action_noisy_t[:, :, ::action_ds]
                        action_noisy_r_ds = action_noisy_r[:, :, ::action_ds]
                        action_v_target_eff_ds = action_v_target_eff[:, :, ::action_ds]
                        action_sigma_t = action_t[:, None, ::action_ds, None, None] / self.config.num_train_timesteps
                        action_sigma_r = action_r[:, None, ::action_ds, None, None] / self.config.num_train_timesteps
                        action_pred_xr = action_noisy_t_ds + student_action_v * (
                            action_sigma_r - action_sigma_t)

                        mask = batch.get("actions_mask")
                        if mask is not None:
                            mask = mask[:, :, ::action_ds].float()
                            denom = (mask.sum() * student_action_v.shape[1]).clamp(min=1)
                            action_xr_mse = (
                                ((action_pred_xr.float() - action_noisy_r_ds.float()) * mask).pow(2).sum()
                                / denom
                            )
                            action_xr_l1 = (
                                ((action_pred_xr.float() - action_noisy_r_ds.float()).abs() * mask).sum()
                                / denom
                            )
                            action_v_eff_mse = (
                                ((student_action_v.float() - action_v_target_eff_ds.float()) * mask).pow(2).sum()
                                / denom
                            )
                        else:
                            action_xr_mse = (action_pred_xr.float() - action_noisy_r_ds.float()).pow(2).mean()
                            action_xr_l1 = (action_pred_xr.float() - action_noisy_r_ds.float()).abs().mean()
                            action_v_eff_mse = (
                                student_action_v.float() - action_v_target_eff_ds.float()).pow(2).mean()

                        add_metric(f"{transition_prefix}/{pair_name}/action_xr_mse", action_xr_mse)
                        add_metric(f"{transition_prefix}/{pair_name}/action_xr_l1", action_xr_l1)
                        add_metric(f"{transition_prefix}/{pair_name}/action_v_eff_mse", action_v_eff_mse)

        finally:
            if was_training:
                model.train()

        out = {}
        for name, total in metric_sums.items():
            avg = total / max(1, metric_counts[name])
            if dist.is_initialized():
                avg = dist_mean(avg)
            out[name] = avg.item()
        return out

    def _move_eval_batch_to_device(self, batch):
        # Keep cached eval batches on CPU; convert a shallow copy each eval run.
        return {
            key: value.to(self.device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

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
            logger.info(f"  action_block_weight = {self.action_block_weight}")
        logger.info(f"  Teacher CFG: [{config.cfg_min}, {config.cfg_max}]")
        logger.info(f"  EMA decay: {config.ema_decay}")
        logger.info(f"  EMA warmup steps: {getattr(config, 'ema_warmup_steps', 0)}")
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
        if self.use_opd_aux:
            logger.info(f"  OPD aux enabled: weight={getattr(config, 'opd_aux_weight', 0.1)}, "
                        f"variant={self.opd_aux_variant}, "
                        f"warmup={getattr(config, 'opd_aux_warmup_steps', 0)}, "
                        f"interval={getattr(config, 'opd_aux_interval', 1)}, "
                        f"prob={getattr(config, 'opd_aux_prob', 1.0)}, "
                        f"target={getattr(config, 'opd_teacher_target_mode', 'student_state')}, "
                        f"grad={getattr(config, 'opd_rollout_grad_mode', 'endpoint')}, "
                        f"flowmap_aux={getattr(config, 'flowmap_aux_weight', 1.0)}, "
                        f"endpoint_aux={getattr(config, 'opd_endpoint_aux_weight', 0.0)}, "
                        f"same_state_vel={getattr(config, 'opd_same_state_velocity_weight', 0.0)}, "
                        f"aux_gc={getattr(config, 'opd_aux_gradient_checkpointing', False)}, "
                        f"action_transition_block={getattr(config, 'action_transition_block_weight', getattr(config, 'action_block_weight', 1.0))}, "
                        f"action_local_fm_block={getattr(config, 'action_local_fm_block_weight', 1.0)}, "
                        f"action_local_fm_weight={getattr(config, 'action_aware_weight', 0.0)}, "
                        f"transition_group_weight={getattr(config, 'opd_transition_group_weight', 1.0)}, "
                        f"anchor_cap_ratio={getattr(config, 'opd_anchor_cap_ratio', -1.0)}, "
                        f"query_bias={getattr(config, 'opd_query_bias', 'none')}, "
                        f"query_ratio={getattr(config, 'opd_query_bias_ratio', 0.0)}, "
                        f"low_noise_max={getattr(config, 'opd_low_noise_max_sigma', 0.25)}")

        if bool(getattr(config, "enable_stage1_start_eval", False)):
            try:
                if config.rank == 0:
                    logger.info("Running Stage-1 start eval before Stage-2 training ...")
                stage1_eval_log_dict = self._run_stage1_start_eval(
                    model=self.student,
                    metric_prefix="stage1_start_eval/current",
                    transition_prefix="stage1_flowmap_eval/current",
                    supports_r_timestep=True,
                )
                if bool(getattr(config, "enable_stage1_start_eval_baseline", True)):
                    baseline_model = getattr(self, "_teacher_nofsdp", None)
                    if baseline_model is not None:
                        baseline_log_dict = self._run_stage1_start_eval(
                            model=baseline_model,
                            metric_prefix="stage1_start_eval/baseline",
                            transition_prefix="stage1_flowmap_eval/baseline",
                            supports_r_timestep=False,
                        )
                        stage1_eval_log_dict.update(baseline_log_dict)
                        eps = 1e-12
                        for b_key, b_val in baseline_log_dict.items():
                            if b_key.startswith("stage1_start_eval/baseline/"):
                                base_prefix = "stage1_start_eval"
                            elif b_key.startswith("stage1_flowmap_eval/baseline/"):
                                base_prefix = "stage1_flowmap_eval"
                            else:
                                continue
                            suffix = b_key.replace(f"{base_prefix}/baseline/", "", 1)
                            c_key = f"{base_prefix}/current/{suffix}"
                            if c_key in stage1_eval_log_dict:
                                rel = (b_val - stage1_eval_log_dict[c_key]) / max(abs(b_val), eps)
                                stage1_eval_log_dict[f"{base_prefix}/improve/{suffix}"] = rel
                    elif config.rank == 0:
                        logger.warning("Stage-1 baseline eval requested, but _teacher_nofsdp is unavailable.")
                if config.rank == 0 and stage1_eval_log_dict:
                    logger.info(
                        "Stage-1 start eval: " +
                        ", ".join(
                            f"{k}={v:.6f}" for k, v in sorted(stage1_eval_log_dict.items())
                        )
                    )
                    if config.enable_wandb and HAS_WANDB:
                        wandb.log(stage1_eval_log_dict, step=self.step)
                    if self.tb_writer is not None:
                        for key, value in stage1_eval_log_dict.items():
                            self.tb_writer.add_scalar(key, value, self.step)
                if dist.is_initialized():
                    dist.barrier(device_ids=[torch.cuda.current_device()])
            except Exception as e:
                if config.rank == 0:
                    logger.warning(f"Stage-1 start eval failed; continuing Stage-2 training: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
                if dist.is_initialized():
                    dist.barrier(device_ids=[torch.cuda.current_device()])

        self.optimizer.zero_grad()
        acc_losses = []              # 累积的总损失
        acc_video_losses = []        # 累积的视频损失
        acc_cosmos_video_endpoint_losses = []
        acc_cosmos_video_cdiff_losses = []
        acc_deployment_video_endpoint_losses = []
        acc_deployment_action_endpoint_losses = []
        acc_deployment_total_losses = []
        acc_deployment_student_steps = []
        acc_deployment_t_starts = []
        acc_deployment_t_ends = []
        acc_video_local_fm_losses = []  # 累积的视频 local FM 损失
        acc_action_losses = []       # 累积的动作损失
        acc_action_local_fm_losses = []  # 累积的动作 local FM 损失
        acc_action_aware_losses = [] # 累积的动作感知损失
        acc_gt_regression_losses = []  # 累积的 GT 回归损失（Flow Map 特有）
        acc_raw_teacher_gt_mses = []
        acc_raw_teacher_gt_l1s = []
        acc_raw_teacher_abs_means = []
        acc_raw_gt_abs_means = []
        acc_raw_teacher_enabled = []
        acc_d_losses = []            # 累积的判别器损失（DMD 特有）
        acc_dmd_grad_norms = []      # 累积的 DMD 梯度范数（DMD 特有）
        acc_opd_aux_losses = []
        acc_opd_video_transition_losses = []
        acc_opd_endpoint_aux_losses = []
        acc_opd_same_state_velocity_losses = []
        acc_opd_local_fm_losses = []
        acc_opd_action_transition_losses = []
        acc_opd_action_local_fm_losses = []
        acc_opd_video_transition_contribs = []
        acc_opd_endpoint_aux_contribs = []
        acc_opd_same_state_velocity_contribs = []
        acc_opd_local_fm_contribs = []
        acc_opd_action_transition_contribs = []
        acc_opd_action_local_fm_contribs = []
        acc_opd_video_transition_ratios = []
        acc_opd_endpoint_aux_ratios = []
        acc_opd_same_state_velocity_ratios = []
        acc_opd_local_fm_ratios = []
        acc_opd_action_transition_ratios = []
        acc_opd_action_local_fm_ratios = []
        acc_opd_transition_group_scaled = []
        acc_opd_anchor_group_scaled = []
        acc_opd_transition_scales = []
        acc_opd_anchor_scales = []
        acc_opd_transition_group_ratios = []
        acc_opd_anchor_group_ratios = []
        acc_kto_good_ratios = []
        acc_kto_weight_means = []
        acc_kto_weight_mins = []
        acc_kto_weight_maxs = []
        acc_kto_thresholds = []
        acc_kto_main_actives = []
        acc_kto_main_good_ratios = []
        acc_kto_main_weight_means = []
        acc_kto_main_weight_mins = []
        acc_kto_main_weight_maxs = []
        acc_kto_main_thresholds = []
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

        while (
            self.step < config.max_train_steps
            and not should_stop_training_at_step(
                step=self.step,
                stop_after_step=getattr(config, 'stop_after_step', None),
            )
        ):
            # 获取下一个数据批次
            batch = self._get_next_batch()

            # ---- 训练步：FlowMap 主目标始终优先；旧 on-policy replacement 仅作 legacy ablation ----
            use_onpolicy = self.use_onpolicy_transition
            onpolicy_warmup = getattr(self.config, 'onpolicy_warmup_steps', 0)
            use_onpolicy_now = (use_onpolicy and self.step >= onpolicy_warmup)
            zero_tensor = torch.tensor(0.0, device=self.device)
            opd_aux_result = None
            use_opd_aux_now = False
            standalone_opd = (
                bool(getattr(self.config, 'opd_aux_standalone_step', False))
                and self.use_opd_aux
                and not use_onpolicy_now
            )
            deployment_enabled = (
                bool(getattr(
                    self.config, "deployment_joint_rollout_enabled", False
                ))
                and not use_onpolicy_now
            )
            if (
                (standalone_opd or deployment_enabled)
                and self.gradient_accumulation_steps != 1
            ):
                raise ValueError(
                    'standalone progressive objectives require '
                    'gradient_accumulation_steps=1 so every scheduled update '
                    'starts from zero gradients.'
                )

            scheduled_kind = _select_progressive_training_objective(
                self.config,
                step=self.step,
                deployment_enabled=deployment_enabled,
                raw_auxiliary_enabled=standalone_opd,
            )
            use_opd_aux_now = scheduled_kind == "raw_auxiliary"
            if use_opd_aux_now:
                opd_aux_prob = float(getattr(self.config, 'opd_aux_prob', 1.0))
                if opd_aux_prob < 1.0:
                    if dist.is_initialized():
                        aux_draw = torch.rand(1, device=self.device)
                        dist.broadcast(aux_draw, src=0)
                        use_opd_aux_now = aux_draw.item() < opd_aux_prob
                    else:
                        use_opd_aux_now = torch.rand(1).item() < opd_aux_prob
                    if not use_opd_aux_now:
                        scheduled_kind = "main"

            def _run_opd_aux():
                if self.opd_aux_variant in ('kto_paopd', 'kto_paopd_norm_focal'):
                    return self._opd_aux_transition_step_kto_paopd(batch, step_in_acc)
                return self._opd_aux_transition_step(batch, step_in_acc)

            if scheduled_kind == "deployment":
                if hasattr(self.student, 'set_requires_gradient_sync'):
                    self.student.set_requires_gradient_sync(True)
                update_index = (
                    self.step
                    // int(self.config.deployment_joint_rollout_interval)
                )
                student_steps = deployment_joint_step_for_update(update_index)
                result = self._cosmos_deployment_joint_rollout_step(
                    batch, step_in_acc, student_steps=student_steps
                )
                for metric_name in (
                    'video_loss',
                    'cosmos_video_endpoint_loss',
                    'cosmos_video_cdiff_loss',
                    'local_fm_loss',
                    'action_loss',
                    'action_local_fm_loss',
                    'action_aware_loss',
                    'gt_regression_loss',
                    'raw_teacher_gt_mse',
                    'raw_teacher_gt_l1',
                    'raw_teacher_abs_mean',
                    'raw_gt_abs_mean',
                    'raw_teacher_enabled',
                ):
                    result.setdefault(metric_name, zero_tensor)
            elif scheduled_kind == "raw_auxiliary":
                if hasattr(self.student, 'set_requires_gradient_sync'):
                    self.student.set_requires_gradient_sync(True)
                if (bool(getattr(self.config, 'opd_aux_empty_cache', False))
                        and torch.cuda.is_available()):
                    torch.cuda.empty_cache()
                opd_aux_checkpointing = bool(getattr(
                    self.config, 'opd_aux_gradient_checkpointing', False))
                opd_aux_result = _call_with_student_checkpointing(
                    self.student, opd_aux_checkpointing, _run_opd_aux)
                result = dict(opd_aux_result)
                for metric_name in (
                    'video_loss',
                    'cosmos_video_endpoint_loss',
                    'cosmos_video_cdiff_loss',
                    'local_fm_loss',
                    'action_loss',
                    'action_local_fm_loss',
                    'action_aware_loss',
                    'gt_regression_loss',
                    'raw_teacher_gt_mse',
                    'raw_teacher_gt_l1',
                    'raw_teacher_abs_mean',
                    'raw_gt_abs_mean',
                    'raw_teacher_enabled',
                ):
                    result.setdefault(metric_name, zero_tensor)
                result['should_sync'] = True
            else:
                if use_onpolicy_now:
                    # Legacy ablation: replacement-style transition matching.
                    result = self._onpolicy_transition_step(batch, step_in_acc)
                else:
                    result = self._train_step(batch, step_in_acc)

                if (not standalone_opd and self.use_opd_aux and not use_onpolicy_now and
                        result.get("should_sync", False) and
                        self.step >= getattr(self.config, 'opd_aux_warmup_steps', 0) and
                        not result.get("skip_step", False)):
                    opd_aux_interval = max(1, int(getattr(self.config, 'opd_aux_interval', 1)))
                    use_opd_aux_now = (self.step % opd_aux_interval == 0)
                    opd_aux_prob = float(getattr(self.config, 'opd_aux_prob', 1.0))
                    if use_opd_aux_now and opd_aux_prob < 1.0:
                        if dist.is_initialized():
                            aux_draw = torch.rand(1, device=self.device)
                            dist.broadcast(aux_draw, src=0)
                            use_opd_aux_now = aux_draw.item() < opd_aux_prob
                        else:
                            use_opd_aux_now = torch.rand(1).item() < opd_aux_prob

                if use_opd_aux_now:
                    if (bool(getattr(self.config, 'opd_aux_empty_cache', False))
                            and torch.cuda.is_available()):
                        torch.cuda.empty_cache()
                    opd_aux_checkpointing = bool(getattr(
                        self.config, 'opd_aux_gradient_checkpointing', False))
                    opd_aux_result = _call_with_student_checkpointing(
                        self.student, opd_aux_checkpointing, _run_opd_aux)
                    result["loss"] = result["loss"] + opd_aux_result.get("loss", zero_tensor)
                    result["skip_step"] = (
                        result.get("skip_step", False) or
                        opd_aux_result.get("skip_step", False)
                    )

            # 累积损失值
            acc_losses.append(result["loss"])
            if use_onpolicy_now:
                acc_video_losses.append(result.get("video_transition_loss", torch.tensor(0.0, device=self.device)))
            else:
                acc_video_losses.append(result["video_loss"])
            acc_cosmos_video_endpoint_losses.append(result.get(
                "cosmos_video_endpoint_loss", zero_tensor))
            acc_cosmos_video_cdiff_losses.append(result.get(
                "cosmos_video_cdiff_loss", zero_tensor))
            acc_deployment_video_endpoint_losses.append(result.get(
                "deployment_video_endpoint_loss", zero_tensor))
            acc_deployment_action_endpoint_losses.append(result.get(
                "deployment_action_endpoint_loss", zero_tensor))
            acc_deployment_total_losses.append(result.get(
                "deployment_total_loss", zero_tensor))
            acc_deployment_student_steps.append(result.get(
                "deployment_student_steps", zero_tensor))
            acc_deployment_t_starts.append(result.get(
                "deployment_t_start", zero_tensor))
            acc_deployment_t_ends.append(result.get(
                "deployment_t_end", zero_tensor))
            acc_video_local_fm_losses.append(result.get(
                "local_fm_loss", zero_tensor))
            acc_action_losses.append(result["action_loss"])
            acc_action_local_fm_losses.append(result.get(
                "action_local_fm_loss", result["action_aware_loss"]))
            acc_action_aware_losses.append(result["action_aware_loss"])
            acc_gt_regression_losses.append(result.get(
                "gt_regression_loss", zero_tensor))
            acc_raw_teacher_gt_mses.append(result.get(
                "raw_teacher_gt_mse", zero_tensor))
            acc_raw_teacher_gt_l1s.append(result.get(
                "raw_teacher_gt_l1", zero_tensor))
            acc_raw_teacher_abs_means.append(result.get(
                "raw_teacher_abs_mean", zero_tensor))
            acc_raw_gt_abs_means.append(result.get(
                "raw_gt_abs_mean", zero_tensor))
            acc_raw_teacher_enabled.append(result.get(
                "raw_teacher_enabled", zero_tensor))
            acc_opd_aux_losses.append(
                opd_aux_result.get("opd_aux_loss", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_video_transition_losses.append(
                opd_aux_result.get("opd_video_transition_loss", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_endpoint_aux_losses.append(
                opd_aux_result.get("opd_endpoint_aux_loss", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_same_state_velocity_losses.append(
                opd_aux_result.get("opd_same_state_velocity_loss", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_local_fm_losses.append(
                opd_aux_result.get("opd_local_fm_loss", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_action_transition_losses.append(
                opd_aux_result.get("opd_action_transition_loss", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_action_local_fm_losses.append(
                opd_aux_result.get("opd_action_local_fm_loss", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_video_transition_contribs.append(
                opd_aux_result.get("opd_video_transition_contrib", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_endpoint_aux_contribs.append(
                opd_aux_result.get("opd_endpoint_aux_contrib", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_same_state_velocity_contribs.append(
                opd_aux_result.get("opd_same_state_velocity_contrib", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_local_fm_contribs.append(
                opd_aux_result.get("opd_local_fm_contrib", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_action_transition_contribs.append(
                opd_aux_result.get("opd_action_transition_contrib", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_action_local_fm_contribs.append(
                opd_aux_result.get("opd_action_local_fm_contrib", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_video_transition_ratios.append(
                opd_aux_result.get("opd_video_transition_ratio", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_endpoint_aux_ratios.append(
                opd_aux_result.get("opd_endpoint_aux_ratio", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_same_state_velocity_ratios.append(
                opd_aux_result.get("opd_same_state_velocity_ratio", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_local_fm_ratios.append(
                opd_aux_result.get("opd_local_fm_ratio", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_action_transition_ratios.append(
                opd_aux_result.get("opd_action_transition_ratio", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_action_local_fm_ratios.append(
                opd_aux_result.get("opd_action_local_fm_ratio", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_transition_group_scaled.append(
                opd_aux_result.get("opd_transition_group_scaled", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_anchor_group_scaled.append(
                opd_aux_result.get("opd_anchor_group_scaled", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_transition_scales.append(
                opd_aux_result.get("opd_transition_scale", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_anchor_scales.append(
                opd_aux_result.get("opd_anchor_scale", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_transition_group_ratios.append(
                opd_aux_result.get("opd_transition_group_ratio", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_opd_anchor_group_ratios.append(
                opd_aux_result.get("opd_anchor_group_ratio", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_kto_good_ratios.append(
                opd_aux_result.get("kto_good_ratio", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_kto_weight_means.append(
                opd_aux_result.get("kto_weight_mean", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_kto_weight_mins.append(
                opd_aux_result.get("kto_weight_min", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_kto_weight_maxs.append(
                opd_aux_result.get("kto_weight_max", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_kto_thresholds.append(
                opd_aux_result.get("kto_threshold", zero_tensor)
                if opd_aux_result is not None else zero_tensor)
            acc_kto_main_actives.append(result.get("kto_main_active", zero_tensor))
            acc_kto_main_good_ratios.append(result.get("kto_main_good_ratio", zero_tensor))
            acc_kto_main_weight_means.append(result.get("kto_main_weight_mean", zero_tensor))
            acc_kto_main_weight_mins.append(result.get("kto_main_weight_min", zero_tensor))
            acc_kto_main_weight_maxs.append(result.get("kto_main_weight_max", zero_tensor))
            acc_kto_main_thresholds.append(result.get("kto_main_threshold", zero_tensor))
            step_in_acc += 1

            # ---- 第二阶段：DMD（条件满足时执行）----
            d_loss_val = torch.tensor(0.0, device=self.device)
            dmd_grad_norm = 0.0
            skip_step = result.get("skip_step", False)
            use_dmd_now = (self.use_dmd
                           and self.distill_action
                           and self.step >= dmd_warmup_steps
                           and result["should_sync"]
                           and not skip_step
                           and not use_onpolicy_now
                           and scheduled_kind == "main")

            if use_dmd_now:
                if self.step >= dmd_warmup_steps + dmd_discriminator_warmup:
                    # 判别器预热完成：执行完整 DMD 训练
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
                log_interval = max(1, int(getattr(config, "log_interval", 1)))
                will_log_step = (
                    (self.step % log_interval == 0) or
                    (self.step + 1 >= config.max_train_steps)
                )
                grad_branch_norms = {}
                if skip_step:
                    total_norm = torch.tensor(float("nan"), device=self.device)
                    self.optimizer.zero_grad()
                else:
                    if (
                        bool(getattr(config, "enable_grad_branch_diagnostics", False))
                        and will_log_step
                    ):
                        grad_branch_norms = self._compute_student_grad_branch_norms()
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
                        self._nofsdp_synced = False  # invalidate dirty flag after weight update

                        # EMA 更新只在正常参数更新后执行（跳过 NaN/Inf 梯度时不更新 EMA，
                        # 避免将异常梯度产生的错误权重传播到目标学生）
                        # FSDP1 with use_orig_params=True preserves original parameter names,
                        # so model.parameters() works directly (no .module needed)
                        ema_warmup_steps = int(getattr(config, 'ema_warmup_steps', 0))
                        ema_decay = 0.0 if self.step < ema_warmup_steps else config.ema_decay
                        self._last_ema_decay = float(ema_decay)
                        if self.target_student is not None:
                            update_ema(
                                self.target_student.parameters(),
                                self.student.parameters(),
                                rate=ema_decay,
                            )

                # 计算平均损失（跨所有进程）
                lr = self.lr_scheduler.get_last_lr()[0]
                metric_tensors = [
                    torch.stack(acc_losses).sum(),
                    torch.stack(acc_video_losses).sum(),
                    torch.stack(acc_cosmos_video_endpoint_losses).sum(),
                    torch.stack(acc_cosmos_video_cdiff_losses).sum(),
                    torch.stack(acc_deployment_video_endpoint_losses).sum(),
                    torch.stack(acc_deployment_action_endpoint_losses).sum(),
                    torch.stack(acc_deployment_total_losses).sum(),
                    torch.stack(acc_deployment_student_steps).sum(),
                    torch.stack(acc_deployment_t_starts).sum(),
                    torch.stack(acc_deployment_t_ends).sum(),
                    torch.stack(acc_video_local_fm_losses).sum(),
                    torch.stack(acc_action_losses).sum(),
                    torch.stack(acc_action_local_fm_losses).sum(),
                    torch.stack(acc_action_aware_losses).sum(),
                    torch.stack(acc_gt_regression_losses).sum(),
                    torch.stack(acc_raw_teacher_gt_mses).sum(),
                    torch.stack(acc_raw_teacher_gt_l1s).sum(),
                    torch.stack(acc_raw_teacher_abs_means).sum(),
                    torch.stack(acc_raw_gt_abs_means).sum(),
                    torch.stack(acc_raw_teacher_enabled).sum(),
                    torch.stack(acc_d_losses).sum(),
                    torch.stack(acc_dmd_grad_norms).sum(),
                    torch.stack(acc_opd_aux_losses).sum(),
                    torch.stack(acc_opd_video_transition_losses).sum(),
                    torch.stack(acc_opd_endpoint_aux_losses).sum(),
                    torch.stack(acc_opd_same_state_velocity_losses).sum(),
                    torch.stack(acc_opd_local_fm_losses).sum(),
                    torch.stack(acc_opd_action_transition_losses).sum(),
                    torch.stack(acc_opd_action_local_fm_losses).sum(),
                    torch.stack(acc_opd_video_transition_contribs).sum(),
                    torch.stack(acc_opd_endpoint_aux_contribs).sum(),
                    torch.stack(acc_opd_same_state_velocity_contribs).sum(),
                    torch.stack(acc_opd_local_fm_contribs).sum(),
                    torch.stack(acc_opd_action_transition_contribs).sum(),
                    torch.stack(acc_opd_action_local_fm_contribs).sum(),
                    torch.stack(acc_opd_video_transition_ratios).sum(),
                    torch.stack(acc_opd_endpoint_aux_ratios).sum(),
                    torch.stack(acc_opd_same_state_velocity_ratios).sum(),
                    torch.stack(acc_opd_local_fm_ratios).sum(),
                    torch.stack(acc_opd_action_transition_ratios).sum(),
                    torch.stack(acc_opd_action_local_fm_ratios).sum(),
                    torch.stack(acc_opd_transition_group_scaled).sum(),
                    torch.stack(acc_opd_anchor_group_scaled).sum(),
                    torch.stack(acc_opd_transition_scales).sum(),
                    torch.stack(acc_opd_anchor_scales).sum(),
                    torch.stack(acc_opd_transition_group_ratios).sum(),
                    torch.stack(acc_opd_anchor_group_ratios).sum(),
                ]
                kto_main_enabled = bool(getattr(self.config, 'kto_main_video_reweight', False))
                if kto_main_enabled:
                    metric_tensors.extend([
                        torch.stack(acc_kto_main_actives).sum(),
                        torch.stack(acc_kto_main_good_ratios).sum(),
                        torch.stack(acc_kto_main_weight_means).sum(),
                        torch.stack(acc_kto_main_weight_mins).sum(),
                        torch.stack(acc_kto_main_weight_maxs).sum(),
                        torch.stack(acc_kto_main_thresholds).sum(),
                    ])
                if self.opd_aux_variant in ('kto_paopd', 'kto_paopd_norm_focal'):
                    metric_tensors.extend([
                        torch.stack(acc_kto_good_ratios).sum(),
                        torch.stack(acc_kto_weight_means).sum(),
                        torch.stack(acc_kto_weight_mins).sum(),
                        torch.stack(acc_kto_weight_maxs).sum(),
                        torch.stack(acc_kto_thresholds).sum(),
                    ])
                metric_values = torch.stack(metric_tensors).float()
                metric_results = dist_mean(metric_values).tolist()
                base_metric_count = 47
                (
                    avg_loss,
                    avg_video_loss,
                    avg_cosmos_video_endpoint_loss,
                    avg_cosmos_video_cdiff_loss,
                    avg_deployment_video_endpoint_loss,
                    avg_deployment_action_endpoint_loss,
                    avg_deployment_total_loss,
                    avg_deployment_student_steps,
                    avg_deployment_t_start,
                    avg_deployment_t_end,
                    avg_video_local_fm_loss,
                    avg_action_loss,
                    avg_action_local_fm_loss,
                    avg_action_aware_loss,
                    avg_gt_regression_loss,
                    avg_raw_teacher_gt_mse_sum,
                    avg_raw_teacher_gt_l1_sum,
                    avg_raw_teacher_abs_mean_sum,
                    avg_raw_gt_abs_mean_sum,
                    avg_raw_teacher_enabled,
                    avg_d_loss,
                    avg_dmd_grad_norm,
                    avg_opd_aux_loss,
                    avg_opd_video_transition_loss,
                    avg_opd_endpoint_aux_loss,
                    avg_opd_same_state_velocity_loss,
                    avg_opd_local_fm_loss,
                    avg_opd_action_transition_loss,
                    avg_opd_action_local_fm_loss,
                    avg_opd_video_transition_contrib,
                    avg_opd_endpoint_aux_contrib,
                    avg_opd_same_state_velocity_contrib,
                    avg_opd_local_fm_contrib,
                    avg_opd_action_transition_contrib,
                    avg_opd_action_local_fm_contrib,
                    avg_opd_video_transition_ratio,
                    avg_opd_endpoint_aux_ratio,
                    avg_opd_same_state_velocity_ratio,
                    avg_opd_local_fm_ratio,
                    avg_opd_action_transition_ratio,
                    avg_opd_action_local_fm_ratio,
                    avg_opd_transition_group_scaled,
                    avg_opd_anchor_group_scaled,
                    avg_opd_transition_scale,
                    avg_opd_anchor_scale,
                    avg_opd_transition_group_ratio,
                    avg_opd_anchor_group_ratio,
                ) = metric_results[:base_metric_count]
                metric_cursor = base_metric_count
                if kto_main_enabled:
                    (
                        avg_kto_main_active,
                        avg_kto_main_good_ratio,
                        avg_kto_main_weight_mean,
                        avg_kto_main_weight_min,
                        avg_kto_main_weight_max,
                        avg_kto_main_threshold,
                    ) = metric_results[metric_cursor:metric_cursor + 6]
                    metric_cursor += 6
                else:
                    avg_kto_main_active = 0.0
                    avg_kto_main_good_ratio = 0.0
                    avg_kto_main_weight_mean = 0.0
                    avg_kto_main_weight_min = 0.0
                    avg_kto_main_weight_max = 0.0
                    avg_kto_main_threshold = 0.0
                if self.opd_aux_variant in ('kto_paopd', 'kto_paopd_norm_focal'):
                    (
                        avg_kto_good_ratio,
                        avg_kto_weight_mean,
                        avg_kto_weight_min,
                        avg_kto_weight_max,
                        avg_kto_threshold,
                    ) = metric_results[metric_cursor:]
                else:
                    avg_kto_good_ratio = 0.0
                    avg_kto_weight_mean = 0.0
                    avg_kto_weight_min = 0.0
                    avg_kto_weight_max = 0.0
                    avg_kto_threshold = 0.0
                action_total_raw = (
                    avg_action_loss
                    + self.gt_regression_weight * avg_gt_regression_loss
                    + getattr(self.config, "action_aware_weight", 0.0) * avg_action_aware_loss
                )
                action_total = self.action_block_weight * action_total_raw
                raw_teacher_count = max(avg_raw_teacher_enabled, 1e-12)
                avg_raw_teacher_gt_mse = avg_raw_teacher_gt_mse_sum / raw_teacher_count
                avg_raw_teacher_gt_l1 = avg_raw_teacher_gt_l1_sum / raw_teacher_count
                avg_raw_teacher_abs_mean = avg_raw_teacher_abs_mean_sum / raw_teacher_count
                avg_raw_gt_abs_mean = avg_raw_gt_abs_mean_sum / raw_teacher_count
                # 重置累积器
                acc_losses = []
                acc_video_losses = []
                acc_cosmos_video_endpoint_losses = []
                acc_cosmos_video_cdiff_losses = []
                acc_deployment_video_endpoint_losses = []
                acc_deployment_action_endpoint_losses = []
                acc_deployment_total_losses = []
                acc_deployment_student_steps = []
                acc_deployment_t_starts = []
                acc_deployment_t_ends = []
                acc_video_local_fm_losses = []
                acc_action_losses = []
                acc_action_local_fm_losses = []
                acc_action_aware_losses = []
                acc_gt_regression_losses = []
                acc_raw_teacher_gt_mses = []
                acc_raw_teacher_gt_l1s = []
                acc_raw_teacher_abs_means = []
                acc_raw_gt_abs_means = []
                acc_raw_teacher_enabled = []
                acc_d_losses = []
                acc_dmd_grad_norms = []
                acc_opd_aux_losses = []
                acc_opd_video_transition_losses = []
                acc_opd_endpoint_aux_losses = []
                acc_opd_same_state_velocity_losses = []
                acc_opd_local_fm_losses = []
                acc_opd_action_transition_losses = []
                acc_opd_action_local_fm_losses = []
                acc_opd_video_transition_contribs = []
                acc_opd_endpoint_aux_contribs = []
                acc_opd_same_state_velocity_contribs = []
                acc_opd_local_fm_contribs = []
                acc_opd_action_transition_contribs = []
                acc_opd_action_local_fm_contribs = []
                acc_opd_video_transition_ratios = []
                acc_opd_endpoint_aux_ratios = []
                acc_opd_same_state_velocity_ratios = []
                acc_opd_local_fm_ratios = []
                acc_opd_action_transition_ratios = []
                acc_opd_action_local_fm_ratios = []
                acc_opd_transition_group_scaled = []
                acc_opd_anchor_group_scaled = []
                acc_opd_transition_scales = []
                acc_opd_anchor_scales = []
                acc_opd_transition_group_ratios = []
                acc_opd_anchor_group_ratios = []
                acc_kto_good_ratios = []
                acc_kto_weight_means = []
                acc_kto_weight_mins = []
                acc_kto_weight_maxs = []
                acc_kto_thresholds = []
                acc_kto_main_actives = []
                acc_kto_main_good_ratios = []
                acc_kto_main_weight_means = []
                acc_kto_main_weight_mins = []
                acc_kto_main_weight_maxs = []
                acc_kto_main_thresholds = []
                step_in_acc = 0

                # 定期清理显存；只在真正清理时同步，避免每个 optimizer step 强制等待 GPU。
                if self.step % config.gc_interval == 0:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()

                # 记录日志（仅主进程）
                should_log = will_log_step
                if config.rank == 0 and should_log:
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
                        "train/ema_decay": getattr(self, '_last_ema_decay', config.ema_decay),
                    }
                    if grad_branch_norms:
                        log_dict["grad_norm/video_branch"] = grad_branch_norms["video"]
                        log_dict["grad_norm/action_branch"] = grad_branch_norms["action"]
                        log_dict["grad_norm/shared_branch"] = grad_branch_norms["shared"]
                    if scheduled_kind == "deployment":
                        postfix["dep"] = f"{avg_deployment_total_loss:.4f}"
                        log_dict["deployment/video_endpoint_loss"] = (
                            avg_deployment_video_endpoint_loss
                        )
                        log_dict["deployment/action_endpoint_loss"] = (
                            avg_deployment_action_endpoint_loss
                        )
                        log_dict["deployment/total_loss"] = (
                            avg_deployment_total_loss
                        )
                        log_dict["deployment/student_steps"] = (
                            avg_deployment_student_steps
                        )
                        log_dict["deployment/t_start"] = avg_deployment_t_start
                        log_dict["deployment/t_end"] = avg_deployment_t_end
                    if self.distill_video:
                        if use_onpolicy_now:
                            postfix["vt"] = f"{avg_video_loss:.4f}"
                            postfix["vlfm"] = f"{avg_video_local_fm_loss:.4f}"
                            log_dict["loss/video_transition"] = avg_video_loss
                            log_dict["loss/video_local_fm"] = avg_video_local_fm_loss
                        else:
                            postfix["v"] = f"{avg_video_loss:.4f}"
                            log_dict["loss/video_consistency"] = avg_video_loss
                            if bool(getattr(self.config, 'cosmos_video_cdiff_aux', False)):
                                postfix["vep"] = f"{avg_cosmos_video_endpoint_loss:.4f}"
                                postfix["vcd"] = f"{avg_cosmos_video_cdiff_loss:.4f}"
                                log_dict["loss/cosmos_video_endpoint"] = avg_cosmos_video_endpoint_loss
                                log_dict["loss/cosmos_video_cdiff"] = avg_cosmos_video_cdiff_loss
                    if kto_main_enabled:
                        postfix["mkto"] = f"{avg_kto_main_good_ratio:.2f}/{avg_kto_main_weight_mean:.2f}"
                        log_dict["kto_main/active"] = avg_kto_main_active
                        log_dict["kto_main/good_ratio"] = avg_kto_main_good_ratio
                        log_dict["kto_main/adaptive_weight_mean"] = avg_kto_main_weight_mean
                        log_dict["kto_main/adaptive_weight_min"] = avg_kto_main_weight_min
                        log_dict["kto_main/adaptive_weight_max"] = avg_kto_main_weight_max
                        log_dict["kto_main/threshold"] = avg_kto_main_threshold
                    if self.distill_action:
                        postfix["a"] = f"{avg_action_loss:.4f}"
                        postfix["at"] = f"{action_total:.4f}"
                        log_dict["loss/action_consistency"] = avg_action_loss
                        log_dict["loss/action_total_raw"] = action_total_raw
                        log_dict["loss/action_total"] = action_total
                    if avg_raw_teacher_enabled > 0:
                        postfix["ctgt"] = f"{avg_raw_teacher_gt_mse:.3f}/{avg_raw_teacher_gt_l1:.3f}"
                        log_dict["cosmos_raw/teacher_gt_mse"] = avg_raw_teacher_gt_mse
                        log_dict["cosmos_raw/teacher_gt_l1"] = avg_raw_teacher_gt_l1
                        log_dict["cosmos_raw/teacher_abs_mean"] = avg_raw_teacher_abs_mean
                        log_dict["cosmos_raw/gt_abs_mean"] = avg_raw_gt_abs_mean
                        log_dict["cosmos_raw/enabled_microbatches"] = avg_raw_teacher_enabled
                    if self.action_aware:
                        postfix["alfm"] = f"{avg_action_local_fm_loss:.4f}"
                        postfix["aa"] = f"{avg_action_aware_loss:.4f}"
                        log_dict["loss/action_local_fm"] = avg_action_local_fm_loss
                        log_dict["loss/action_aware"] = avg_action_aware_loss
                    if avg_opd_aux_loss > 0:
                        postfix["opd"] = f"{avg_opd_aux_loss:.4f}"
                        postfix["ovt"] = f"{avg_opd_video_transition_loss:.2e}"
                        postfix["oep"] = f"{avg_opd_endpoint_aux_loss:.2e}"
                        postfix["ossv"] = f"{avg_opd_same_state_velocity_loss:.2e}"
                        postfix["ovlfm"] = f"{avg_opd_local_fm_loss:.2e}"
                        postfix["wovt"] = f"{avg_opd_video_transition_contrib:.2e}"
                        postfix["rat"] = f"{avg_opd_video_transition_ratio:.2f}/{avg_opd_action_transition_ratio:.2f}/{avg_opd_action_local_fm_ratio:.2f}"
                        postfix["gtr"] = f"{avg_opd_transition_group_ratio:.2f}/{avg_opd_anchor_group_ratio:.2f}"
                        postfix["gsc"] = f"{avg_opd_transition_scale:.1f}/{avg_opd_anchor_scale:.2f}"
                        log_dict["loss/opd_aux"] = avg_opd_aux_loss
                        log_dict["loss/opd_video_transition"] = avg_opd_video_transition_loss
                        log_dict["loss/opd_endpoint_aux"] = avg_opd_endpoint_aux_loss
                        log_dict["loss/opd_same_state_velocity"] = avg_opd_same_state_velocity_loss
                        log_dict["loss/opd_local_fm"] = avg_opd_local_fm_loss
                        log_dict["loss_weighted/opd_video_transition"] = avg_opd_video_transition_contrib
                        log_dict["loss_weighted/opd_endpoint_aux"] = avg_opd_endpoint_aux_contrib
                        log_dict["loss_weighted/opd_same_state_velocity"] = avg_opd_same_state_velocity_contrib
                        log_dict["loss_weighted/opd_local_fm"] = avg_opd_local_fm_contrib
                        log_dict["loss_weighted/opd_transition_group_scaled"] = avg_opd_transition_group_scaled
                        log_dict["loss_weighted/opd_anchor_group_scaled"] = avg_opd_anchor_group_scaled
                        log_dict["loss_scale/opd_transition"] = avg_opd_transition_scale
                        log_dict["loss_scale/opd_anchor"] = avg_opd_anchor_scale
                        log_dict["loss_ratio/opd_transition_total"] = avg_opd_transition_group_ratio
                        log_dict["loss_ratio/opd_anchor_total"] = avg_opd_anchor_group_ratio
                        log_dict["loss_ratio/opd_video_transition"] = avg_opd_video_transition_ratio
                        log_dict["loss_ratio/opd_endpoint_aux"] = avg_opd_endpoint_aux_ratio
                        log_dict["loss_ratio/opd_same_state_velocity"] = avg_opd_same_state_velocity_ratio
                        log_dict["loss_ratio/opd_local_fm"] = avg_opd_local_fm_ratio
                        log_dict.update(opd_diagnostic_aliases({
                            "opd_endpoint_aux_loss": avg_opd_endpoint_aux_loss,
                            "opd_same_state_velocity_loss": avg_opd_same_state_velocity_loss,
                            "opd_action_transition_loss": avg_opd_action_transition_loss,
                            "opd_endpoint_aux_ratio": avg_opd_endpoint_aux_ratio,
                            "opd_same_state_velocity_ratio": avg_opd_same_state_velocity_ratio,
                            "opd_action_transition_ratio": avg_opd_action_transition_ratio,
                        }, self.config))
                        if self.opd_aux_variant in ('kto_paopd', 'kto_paopd_norm_focal'):
                            postfix["kto"] = f"{avg_kto_good_ratio:.2f}/{avg_kto_weight_mean:.2f}"
                            log_dict["kto/good_ratio"] = avg_kto_good_ratio
                            log_dict["kto/adaptive_weight_mean"] = avg_kto_weight_mean
                            log_dict["kto/adaptive_weight_min"] = avg_kto_weight_min
                            log_dict["kto/adaptive_weight_max"] = avg_kto_weight_max
                            log_dict["kto/threshold"] = avg_kto_threshold
                        if self.distill_action:
                            postfix["oat"] = f"{avg_opd_action_transition_loss:.2e}"
                            postfix["woat"] = f"{avg_opd_action_transition_contrib:.2e}"
                            log_dict["loss/opd_action_transition"] = avg_opd_action_transition_loss
                            log_dict["loss_weighted/opd_action_transition"] = avg_opd_action_transition_contrib
                            log_dict["loss_ratio/opd_action_transition"] = avg_opd_action_transition_ratio
                        if self.action_aware:
                            postfix["oalfm"] = f"{avg_opd_action_local_fm_loss:.2e}"
                            postfix["woalfm"] = f"{avg_opd_action_local_fm_contrib:.2e}"
                            log_dict["loss/opd_action_local_fm"] = avg_opd_action_local_fm_loss
                            log_dict["loss_weighted/opd_action_local_fm"] = avg_opd_action_local_fm_contrib
                            log_dict["loss_ratio/opd_action_local_fm"] = avg_opd_action_local_fm_ratio
                    postfix["gt"] = f"{avg_gt_regression_loss:.4f}"
                    log_dict["loss/gt_regression"] = avg_gt_regression_loss
                    # DMD 日志
                    if self.use_dmd and not use_onpolicy and self.step >= dmd_warmup_steps:
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
                        tb_flush_interval = max(1, int(getattr(config, "tb_flush_interval", 50)))
                        if self.step % tb_flush_interval == 0:
                            self.tb_writer.flush()

                self.step += 1

                # Lightweight deterministic eval. All ranks run this because
                # the student may be FSDP-wrapped; only rank 0 logs results.
                light_eval_interval = int(getattr(config, "light_eval_interval", 0))
                if (self._light_eval_is_enabled() and light_eval_interval > 0 and
                        self.step % light_eval_interval == 0):
                    torch.cuda.empty_cache()
                    gc.collect()
                    eval_log_dict = self._run_light_eval()
                    torch.cuda.empty_cache()
                    gc.collect()
                    if config.rank == 0 and eval_log_dict:
                        logger.info(
                            "Light eval: " + ", ".join(
                                f"{k}={v:.6f}" for k, v in sorted(eval_log_dict.items())
                            )
                        )
                        if config.enable_wandb and HAS_WANDB:
                            wandb.log(eval_log_dict, step=self.step)
                        if self.tb_writer is not None:
                            for key, value in eval_log_dict.items():
                                self.tb_writer.add_scalar(key, value, self.step)

                rollout_eval_interval = int(getattr(config, "rollout_eval_interval", 0))
                if (self._rollout_eval_is_enabled() and rollout_eval_interval > 0 and
                        self.step % rollout_eval_interval == 0):
                    torch.cuda.empty_cache()
                    gc.collect()
                    rollout_eval_log_dict = self._run_rollout_eval()
                    torch.cuda.empty_cache()
                    gc.collect()
                    if config.rank == 0 and rollout_eval_log_dict:
                        logger.info(
                            "Rollout eval: " + ", ".join(
                                f"{k}={v:.6f}" for k, v in sorted(rollout_eval_log_dict.items())
                            )
                        )
                        if config.enable_wandb and HAS_WANDB:
                            wandb.log(rollout_eval_log_dict, step=self.step)
                        if self.tb_writer is not None:
                            for key, value in rollout_eval_log_dict.items():
                                self.tb_writer.add_scalar(key, value, self.step)

                # 定期保存检查点（保留最近 3 个 + 最佳 loss 的）
                if self.step % config.save_interval == 0:
                    self._save_checkpoint("online_student")
                    if self.target_student is not None:
                        self._save_checkpoint("target_student")


            # 常规训练不需要每个 microbatch barrier；仅保留可选 debug barrier。
            train_barrier_interval = int(getattr(config, "train_barrier_interval", 0))
            if (dist.is_initialized() and train_barrier_interval > 0 and
                    result.get("should_sync", False) and self.step % train_barrier_interval == 0):
                dist.barrier(device_ids=[torch.cuda.current_device()])

        progress_bar.close()
        logger.info("Flow Map distillation completed!")
        # 保存最终检查点
        self._save_checkpoint("online_student")
        if self.target_student is not None:
            self._save_checkpoint("target_student")
        # 关闭 TensorBoard writer
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
