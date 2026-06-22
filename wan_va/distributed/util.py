# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from datetime import timedelta

import torch
import torch.distributed as dist


def _configure_model(model, shard_fn, param_dtype, device, eval_mode=True, fsdp1=False):
    """
    Configure and shard a model for distributed inference/training.

    Args:
        model: The model to configure.
        shard_fn: Sharding function (e.g. shard_model or shard_model_fsdp1).
        param_dtype: Parameter dtype for mixed precision.
        device: Target device.
        eval_mode: If True, set model to eval mode with no gradients.
        fsdp1: If True, use FSDP1-based sharding via shard_model_fsdp1.
               When True, shard_fn is ignored and shard_model_fsdp1 is used directly.
    """
    if eval_mode:
        model.eval().requires_grad_(False)
    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])

    if dist.is_initialized() and dist.get_world_size() > 1:
        if fsdp1:
            from wan_va.distributed.fsdp import shard_model_fsdp1
            model = shard_model_fsdp1(model, param_dtype=param_dtype, reduce_dtype=torch.float32)
        else:
            model = shard_fn(model)
    else:
        model.to(param_dtype)
        model.to(device)

    return model


def init_distributed(world_size, local_rank, rank):
    # if world_size > 1:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            rank=rank,
                            world_size=world_size,
                            timeout=timedelta(minutes=30))

def dist_mean(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.AVG)
    return local_tensor

def dist_max(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.MAX)
    return local_tensor
