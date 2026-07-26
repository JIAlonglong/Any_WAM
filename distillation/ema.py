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
from collections.abc import Iterable, Mapping


_is_distributed = None  # 缓存分布式状态


def _validated_named_parameter_pairs(
    target_named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    source_named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> list[tuple[str, torch.nn.Parameter, torch.nn.Parameter]]:
    target_named_parameters = list(target_named_parameters)
    source_named_parameters = list(source_named_parameters)
    target_names = [name for name, _ in target_named_parameters]
    source_names = [name for name, _ in source_named_parameters]
    if target_names != source_names:
        raise ValueError(
            "EMA target/source parameter names must match in the same order"
        )

    pairs = []
    for (name, target), (_, source) in zip(
        target_named_parameters, source_named_parameters
    ):
        if target.shape != source.shape:
            raise ValueError(
                f"EMA parameter shape mismatch for {name}: "
                f"target={tuple(target.shape)} source={tuple(source.shape)}"
            )
        if not target.is_floating_point() or not source.is_floating_point():
            raise ValueError(f"EMA parameter {name} must be floating point")
        pairs.append((name, target, source))
    return pairs


class SelectiveFp32EMA:
    """Persistent FP32 EMA for selected parameters with legacy fallback."""

    STATE_VERSION = 1

    def __init__(
        self,
        target_named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        source_named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        selected_names: Iterable[str],
    ):
        self._pairs = _validated_named_parameter_pairs(
            target_named_parameters, source_named_parameters
        )
        available_names = {name for name, _, _ in self._pairs}
        requested_names = set(selected_names)
        unknown_names = requested_names - available_names
        if unknown_names:
            preview = ", ".join(sorted(unknown_names)[:8])
            raise ValueError(f"EMA selected names are not model parameters: {preview}")

        self._selected_names = tuple(
            name for name, _, _ in self._pairs if name in requested_names
        )
        if not self._selected_names:
            raise ValueError(
                "SelectiveFp32EMA requires at least one selected parameter"
            )
        self._selected_name_set = set(self._selected_names)
        self._masters = {
            name: target.detach().float().clone()
            for name, target, _ in self._pairs
            if name in self._selected_name_set
        }

    @property
    def selected_names(self) -> tuple[str, ...]:
        return self._selected_names

    @torch.no_grad()
    def update(self, rate: float) -> dict[str, float]:
        rate = float(rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError("EMA rate must lie in [0, 1]")

        nonselected_targets = []
        nonselected_sources = []
        moved_count = 0
        master_delta_max = 0.0
        target_source_max = 0.0
        lerp_weight = 1.0 - rate

        for name, target, source in self._pairs:
            if name not in self._selected_name_set:
                nonselected_targets.append(target)
                nonselected_sources.append(source)
                continue

            master = self._masters[name]
            if master.numel() == 0:
                continue
            source_fp32 = source.detach().float()
            gap_max = float((source_fp32 - master).abs().max().item())
            if not torch.isfinite(torch.tensor(gap_max)):
                raise RuntimeError(
                    f"Non-finite FP32 action EMA update for parameter {name}"
                )
            delta_max = gap_max * lerp_weight
            master.lerp_(source_fp32, lerp_weight)
            target.copy_(master.to(dtype=target.dtype))
            moved_count += int(delta_max > 0.0)
            master_delta_max = max(master_delta_max, delta_max)
            target_source_max = max(
                target_source_max,
                float((target.detach() - source.detach()).abs().max().item()),
            )

        if nonselected_targets:
            update_ema(
                nonselected_targets,
                nonselected_sources,
                rate=rate,
            )

        return {
            "selected_count": float(len(self._selected_names)),
            "selected_moved_count": float(moved_count),
            "master_delta_max": master_delta_max,
            "target_source_max": target_source_max,
        }

    def state_dict(self) -> dict:
        return {
            "version": self.STATE_VERSION,
            "selected_names": self._selected_names,
            "masters": {
                name: master.detach().cpu().clone()
                for name, master in self._masters.items()
            },
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping) -> None:
        version = int(state.get("version", -1))
        if version != self.STATE_VERSION:
            raise ValueError(
                f"Unsupported selective EMA state version {version}; "
                f"expected {self.STATE_VERSION}"
            )
        selected_names = tuple(state.get("selected_names", ()))
        if selected_names != self._selected_names:
            raise ValueError(
                "Selective EMA state selected names do not match the model"
            )
        masters = state.get("masters")
        if not isinstance(masters, Mapping):
            raise ValueError("Selective EMA state masters must be a mapping")
        if set(masters) != self._selected_name_set:
            raise ValueError("Selective EMA state master names do not match the model")

        target_by_name = {
            name: target
            for name, target, _ in self._pairs
            if name in self._selected_name_set
        }
        for name in self._selected_names:
            saved = masters[name]
            master = self._masters[name]
            if not isinstance(saved, torch.Tensor):
                raise ValueError(f"Selective EMA state for {name} must be a tensor")
            if saved.shape != master.shape:
                raise ValueError(
                    f"Selective EMA state shape mismatch for {name}: "
                    f"saved={tuple(saved.shape)} expected={tuple(master.shape)}"
                )
            saved_fp32 = saved.to(device=master.device, dtype=torch.float32)
            if not bool(torch.isfinite(saved_fp32).all().item()):
                raise ValueError(
                    f"Selective EMA state contains non-finite values for {name}"
                )
            master.copy_(saved_fp32)
            target_by_name[name].copy_(
                master.to(dtype=target_by_name[name].dtype)
            )


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
    global _is_distributed

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
