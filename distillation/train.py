"""
Flash-WAM 蒸馏训练入口：模态感知的 LCM 蒸馏（联合视频+动作）。

蒸馏模式选择（通过 DISTILL_MODE 环境变量）：
  flashwam           — Flash-WAM（论文方法）：模态感知联合蒸馏
                       = 视频一致性损失 + 动作一致性损失 + 动作 MSE 正则
  joint              — 天真联合 LCM：视频一致性 + 动作一致性（消融实验）
  video              — 仅视频 LCM 一致性损失（消融实验）
  video_action_aware — 仅视频 LCM + 小权重动作 MSE 正则（消融实验）
  action             — 仅动作一致性，student 初始化为 video-LCM 检查点

设计思路：
  - 训练流程与 wan_va/train.py 几乎完全一致
  - 所有噪声添加、时间步采样、chunk_size、window_size 采样都复用原生训练代码
  - 唯一的区别在于蒸馏的特殊处理

与原生训练的唯一区别：
  1. 三个模型：教师（frozen）、在线学生（可训练）、目标学生（EMA）
  2. 教师使用 CFG（从 [2, 10] 随机采样引导强度），需要两次前向
  3. 学生和目标学生不使用 CFG
  4. 损失函数：一致性函数输出的 Huber 损失，而非 v-prediction 的 MSE
  5. k = num_train_timesteps / num_ddim_timesteps = 1000 / 2 = 500
     （在 1000 步 schedule 中构建训练对的步长）
"""

import argparse
import os
import sys

# 将 wan_va 目录添加到 Python 路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wan_va"))

# 安装 flash_attn 存根（必须在导入 wan_va 模块之前调用）
from patches import install_flash_attn_stub
install_flash_attn_stub()

from distributed.util import init_distributed
from utils import init_logger, logger
from trainer import FlashWAMDistiller


# ---------------------------------------------------------------------------
# 入口函数
# ---------------------------------------------------------------------------
def run(args):
    """
    执行蒸馏训练。

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
    _config_mod = os.environ.get("CONFIG_FILE", "config")
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
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
        config.empty_emb_path = os.path.join(args.dataset_path, "empty_emb.pt")
    if args.resume_from_step is not None:
        config.resume_from_step = args.resume_from_step
    if args.resume_from_path is not None:
        config.resume_from_path = args.resume_from_path

    # 打印配置信息（仅主进程）
    if rank == 0:
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
        logger.info(f"Teacher: {config.teacher_model_path}")
        logger.info(f"Dataset: {config.dataset_path}")
        logger.info(f"Output:  {config.output_dir}")

    # 创建蒸馏训练器并开始训练
    trainer = FlashWAMDistiller(config)
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
    parser = argparse.ArgumentParser(description="LCM Video Distillation v2")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--teacher-model-path", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--resume-from-step", type=int, default=None,
                        help="Resume training from this checkpoint step")
    parser.add_argument("--resume-from-path", type=str, default=None,
                        help="Resume from explicit checkpoint directory (overrides --resume-from-step path)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    # 初始化日志系统并启动训练
    init_logger()
    main()
