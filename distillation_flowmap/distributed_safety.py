"""Small distributed control-flow guards used by training steps."""

import torch
import torch.distributed as dist


def all_ranks_finite(local_finite, *, device, dist_module=dist):
    """Return true only when every participating rank reports a finite loss."""
    finite = torch.tensor(
        int(bool(local_finite)),
        device=device,
        dtype=torch.int32,
    )
    if dist_module.is_available() and dist_module.is_initialized():
        dist_module.all_reduce(finite, op=dist_module.ReduceOp.MIN)
    return bool(finite.item())
