from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

import distillation.ema as ema_module
from distillation.ema import SelectiveFp32EMA


class TinyActionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action = torch.nn.Linear(4, 4, bias=False)
        self.action.weight.data.zero_()

    def forward(self, inputs):
        return self.action(inputs)


def _full_state(model):
    state = get_model_state_dict(
        model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )
    return {name: value.clone() for name, value in state.items()}


def main():
    dist.init_process_group("gloo")
    torch.cuda.set_device(0)
    source = FSDP(
        TinyActionModel().cuda().to(torch.bfloat16),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        device_id=0,
    )
    target = FSDP(
        TinyActionModel().cuda().to(torch.bfloat16),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        device_id=0,
    )
    inputs = torch.ones(2, 4, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        target(inputs)
        source_param = dict(source.named_parameters())[
            "_fsdp_wrapped_module.action.weight"
        ]
        if source_param.numel():
            source_param.fill_(1.0)
        selected_name = "_fsdp_wrapped_module.action.weight"
        action_ema = SelectiveFp32EMA(
            target.named_parameters(),
            source.named_parameters(),
            {selected_name},
        )
        action_ema.update(0.0)

        invalidate = getattr(
            ema_module,
            "invalidate_fsdp1_unsharded_parameter_cache",
            lambda _model: 0,
        )
        invalidated_handles = invalidate(target)
        target_output = target(inputs)
        source_output = source(inputs)

    target_state = _full_state(target)
    source_state = _full_state(source)
    if dist.get_rank() == 0:
        assert invalidated_handles == 1
        assert torch.equal(target_output, source_output)
        assert torch.equal(target_state["action.weight"], source_state["action.weight"])
        print("FSDP_ACTION_EMA_WRITEBACK_OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
