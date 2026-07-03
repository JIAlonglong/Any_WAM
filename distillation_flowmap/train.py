"""
Flash-WAM Flow Map 蒸馏入口。
基于 AnyFlow 的流映射方法，支持 2~50 步灵活推理。

启动方式：
  bash distillation_flowmap/run.sh
"""

import argparse
import os
import sys

# 将 wan_va 目录添加到 Python 路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wan_va"))

# 将项目根目录添加到路径（用于 import distillation）
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# 安装 flash_attn 存根（必须在导入 wan_va 模块之前调用）
from distillation.patches import install_flash_attn_stub
install_flash_attn_stub()

# 禁用 dynamo 错误抑制（FlexAttention 已不再使用 torch.compile，但保留此设置以防其他 compile 问题）
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from distributed.util import init_distributed
from utils import init_logger, logger
from flowmap_trainer import FlowMapDistiller
from distillation_flowmap.cosmos_policy_adapter import resolve_cosmos_policy_assets


_LIBERO_TEACHER_PATH = "/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero"
_STALE_LIBERO_TEACHER_PATHS = {
    "/kpfs-intern/jialongliu/projects/Flash-WAM/checkpoints/lingbot-va-posttrain-libero",
    "/root/intern/jialongliu/projects/Flash-WAM/checkpoints/lingbot-va-posttrain-libero",
}


def _normalize_teacher_model_path(path, teacher_backend="wanva"):
    """Return the teacher root directory expected by FlowMapDistiller."""
    if path is None:
        return path

    path = os.path.abspath(os.path.expanduser(path))
    backend = str(teacher_backend or "wanva").lower()
    if backend in ("cosmos", "cosmos_policy", "cosmos-policy"):
        return resolve_cosmos_policy_assets(path)["root"]
    if backend not in ("wanva", "lingbot_va", "lingbot-va"):
        raise ValueError(
            f"Unsupported teacher_backend={teacher_backend!r}; "
            "expected 'wanva' or 'cosmos_policy'."
        )

    if path in _STALE_LIBERO_TEACHER_PATHS:
        logger.warning(
            "Teacher path %s is the old LIBERO location; using %s instead.",
            path,
            _LIBERO_TEACHER_PATH,
        )
        path = _LIBERO_TEACHER_PATH

    # Accept both the teacher root and the transformer subdirectory on CLI.
    if os.path.basename(path) == "transformer" and os.path.isfile(os.path.join(path, "config.json")):
        path = os.path.dirname(path)

    transformer_config = os.path.join(path, "transformer", "config.json")
    if not os.path.isfile(transformer_config):
        raise FileNotFoundError(
            "Invalid teacher_model_path: expected "
            f"{transformer_config}. For LIBERO use {_LIBERO_TEACHER_PATH}."
        )

    return path


def _resolve_empty_emb_path(dataset_path):
    return os.environ.get("EMPTY_EMB_PATH", os.path.join(dataset_path, "empty_emb.pt"))


def _maybe_auto_scale_gradient_accumulation(config, args, world_size, rank=0):
    """Scale single-node accumulation defaults for multi-GPU torchrun launches."""
    if args.gradient_accumulation_steps is not None:
        return False
    if world_size <= 1:
        return False
    if not bool(getattr(config, "auto_scale_gradient_accumulation", False)):
        return False

    reference = int(getattr(
        config,
        "gradient_accumulation_reference",
        getattr(config, "gradient_accumulation_steps", 1),
    ))
    scaled = max(1, reference // world_size)
    current = int(getattr(config, "gradient_accumulation_steps", scaled))
    if current == scaled:
        return False

    config.gradient_accumulation_steps = scaled
    if rank == 0:
        logger.info(
            "Auto-scaled gradient_accumulation_steps: %d -> %d "
            "(reference=%d, world_size=%d). Pass --gradient-accumulation-steps "
            "to override.",
            current,
            scaled,
            reference,
            world_size,
        )
    return True


# ---------------------------------------------------------------------------
# 入口函数
# ---------------------------------------------------------------------------
def run(args):
    """
    执行 Flow Map 蒸馏训练。

    参数:
        args: 命令行参数，包含 output_dir、teacher_model_path 等

    流程：
      1. 加载配置文件（通过 CONFIG_FILE 环境变量支持不同配置）
      2. 初始化分布式训练
      3. 用命令行参数覆盖配置
      4. 创建并运行蒸馏训练器
    """
    # 通过环境变量选择配置模块
    # 这样可以支持不同任务的配置（如 config_g1_alltasks.py）与默认配置共存
    import importlib
    _config_mod = os.environ.get("CONFIG_FILE", "distillation_flowmap.config")
    config = importlib.import_module(_config_mod).cfg

    # 获取分布式训练参数
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)

    # 将分布式参数添加到配置中
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    # 用命令行参数覆盖配置（如果提供）
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.teacher_model_path is not None:
        config.teacher_model_path = args.teacher_model_path
    config.teacher_backend = getattr(config, "teacher_backend", "wanva")
    config.teacher_model_path = _normalize_teacher_model_path(
        config.teacher_model_path,
        teacher_backend=config.teacher_backend,
    )
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
        config.empty_emb_path = _resolve_empty_emb_path(args.dataset_path)
    if args.resume_from_step is not None:
        config.resume_from_step = args.resume_from_step
    if args.resume_from_path is not None:
        config.resume_from_path = args.resume_from_path
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    _maybe_auto_scale_gradient_accumulation(config, args, world_size, rank)
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.load_worker is not None:
        config.load_worker = args.load_worker

    # 打印配置信息（仅主进程）
    if rank == 0:
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
        logger.info(f"Teacher: {config.teacher_model_path}")
        logger.info(f"Teacher backend: {config.teacher_backend}")
        logger.info(f"Dataset: {config.dataset_path}")
        logger.info(f"Output:  {config.output_dir}")

    # 创建蒸馏训练器并开始训练
    trainer = FlowMapDistiller(config)
    trainer.train()


def main():
    """
    主函数：解析命令行参数并启动训练。

    命令行参数：
      --output-dir:          输出目录（存放检查点和日志）
      --teacher-model-path:  教师模型路径（预训练的 LingBot-VA）
      --dataset-path:        数据集路径（包含 latent 表示的 LeRobot 数据）
      --resume-from-step:    从指定步数恢复训练
      --resume-from-path:    从指定路径恢复训练（优先于 --resume-from-step）
    """
    parser = argparse.ArgumentParser(description="Flash-WAM FlowMap Distillation")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--teacher-model-path", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--resume-from-step", type=int, default=None,
                        help="Resume training from this checkpoint step")
    parser.add_argument("--resume-from-path", type=str, default=None,
                        help="Resume from explicit checkpoint directory (overrides --resume-from-step path)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None,
                        help="Override gradient accumulation steps (auto-calculated for multi-GPU)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override per-GPU batch size")
    parser.add_argument("--load-worker", type=int, default=None,
                        help="Override DataLoader num_workers")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    # 初始化日志系统并启动训练
    init_logger()
    main()
