"""
指数移动平均（EMA）更新：用于维护目标学生模型。

EMA 的作用：
  - 目标学生是在线学生的"平滑版本"
  - 每次在线学生更新后，目标学生用 EMA 方式跟随更新
  - 这种方式比直接复制权重更稳定，减少训练震荡
  - EMA 衰减系数（默认 0.995）控制更新速度

数学公式：
  target = decay * target + (1 - decay) * student
  其中 decay = 0.995，意味着目标学生主要保留历史信息，缓慢吸收新信息

分布式训练支持：
  - 处理 DTensor（分布式张量）的情况
  - 自动处理本地张量和全局张量的转换
  - 使用 fused kernel 加速 EMA 更新
"""

import torch
import torch.distributed as dist


_ema_first_call = True  # 标记是否是第一次调用（用于调试打印）
_is_distributed = None  # 缓存分布式状态


@torch.no_grad()
def update_ema(target_params, source_params, rate=0.95):
    """
    使用指数移动平均更新目标模型的参数。

    参数:
        target_params: 目标模型（EMA 模型）的参数列表
        source_params: 源模型（在线学生）的参数列表
        rate:          EMA 衰减系数（默认 0.95，配置中设为 0.995）

    实现细节：
      - 使用 lerp_（线性插值）操作：target = rate * target + (1-rate) * source
      - 自动处理 DTensor（FSDP 分布式训练中的分片张量）
      - 使用 torch._foreach_lerp_ 融合多个参数的更新操作
    """
    global _ema_first_call, _is_distributed

    # 缓存分布式状态
    if _is_distributed is None:
        _is_distributed = dist.is_initialized()

    # 将参数列表转换为列表（如果是生成器）
    target_params = list(target_params)
    source_params = list(source_params)

    if _is_distributed:
        # 分布式训练：需要处理 DTensor
        _update_ema_distributed(target_params, source_params, rate)
    else:
        # 非分布式训练：使用 fused kernel
        _update_ema_fused(target_params, source_params, rate)

    _ema_first_call = False


def _update_ema_fused(target_params, source_params, rate):
    """
    使用 fused kernel 更新 EMA（非分布式训练）。

    使用 torch._foreach_lerp_ 将多个参数的 lerp 操作融合为一次 kernel launch。
    """
    # 提取数据张量
    target_data = [p.data for p in target_params]
    source_data = [p.data for p in source_params]

    # 使用 fused foreach 操作
    torch._foreach_lerp_(target_data, source_data, 1.0 - rate)


def _update_ema_distributed(target_params, source_params, rate):
    """
    更新 EMA（分布式训练，处理 DTensor）。

    根据 DTensor 类型分组处理，每组使用 fused kernel。
    """
    # 按 DTensor 类型分组
    dt_target_dtensor = []  # 两者都是 DTensor
    dt_target_plain = []    # 只有目标是 DTensor
    plain_dt_source = []    # 只有源是 DTensor
    plain_plain = []        # 两者都不是 DTensor

    for targ, src in zip(target_params, source_params):
        td, sd = targ.data, src.data
        t_dt = type(td).__name__ == "DTensor"
        s_dt = type(sd).__name__ == "DTensor"

        # 第一次调用时打印形状信息（用于调试）
        if _ema_first_call and len(dt_target_dtensor) + len(dt_target_plain) < 3:
            t_shape = td._local_tensor.shape if t_dt else td.shape
            s_shape = sd._local_tensor.shape if s_dt else sd.shape
            print(f"[EMA] t_dt={t_dt} t={t_shape}  s_dt={s_dt} s={s_shape}", flush=True)

        if t_dt and s_dt:
            dt_target_dtensor.append((td._local_tensor, sd._local_tensor))
        elif s_dt:
            plain_dt_source.append((td, sd.full_tensor()))
        elif t_dt:
            dt_target_plain.append((td._local_tensor, sd))
        else:
            plain_plain.append((td, sd))

    # 对每组使用 fused 操作
    if dt_target_dtensor:
        t_list, s_list = zip(*dt_target_dtensor)
        torch._foreach_lerp_(list(t_list), list(s_list), 1.0 - rate)

    if plain_dt_source:
        t_list, s_list = zip(*plain_dt_source)
        torch._foreach_lerp_(list(t_list), list(s_list), 1.0 - rate)

    if dt_target_plain:
        t_list, s_list = zip(*dt_target_plain)
        torch._foreach_lerp_(list(t_list), list(s_list), 1.0 - rate)

    if plain_plain:
        t_list, s_list = zip(*plain_plain)
        torch._foreach_lerp_(list(t_list), list(s_list), 1.0 - rate)
