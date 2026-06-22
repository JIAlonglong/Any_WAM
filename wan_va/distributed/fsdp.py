# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc

import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

def apply_ac(model):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in enumerate(model.blocks):
        transformer_block = ptd_checkpoint_wrapper(
            transformer_block,
            preserve_rng_state=False,
        )
        model.blocks[layer_id] = transformer_block


def shard_model(model,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                activation_checkpointing=False):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper as ptd_checkpoint_wrapper,
    )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mp_policy": mp_policy, "reshard_after_forward": True}

    if activation_checkpointing:
        for block in model.blocks:
            block = ptd_checkpoint_wrapper(block)

    for block in model.blocks:
        fully_shard(block.attn1, **fsdp_config)
        fully_shard(block.attn2, **fsdp_config)
        fully_shard(block.ffn, **fsdp_config)
        fully_shard(block, **fsdp_config)

    fully_shard(model, **fsdp_config)
    return model


def shard_model_fsdp1(model,
                      param_dtype=torch.bfloat16,
                      reduce_dtype=torch.float32):
    """Shard model using PyTorch FSDP1 with auto_wrap_policy."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    mp_policy = MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
    )

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    model = model.to(device)

    auto_wrap_policy = ModuleWrapPolicy({type(model.blocks[0])})

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        use_orig_params=True,
        device_id=torch.cuda.current_device(),
    )

    return model


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
