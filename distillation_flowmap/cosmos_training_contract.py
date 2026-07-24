from __future__ import annotations

import torch

CONTRACT_VERSION = 2
ACTION_PACKING_SCHEMA = "downsample_survivor_v2"


def pack_actions_for_downsample(
    aligned: torch.Tensor,
    target_shape: tuple[int, int, int, int, int],
    *,
    downsample_factor: int,
    schema: str,
) -> torch.Tensor:
    if schema != ACTION_PACKING_SCHEMA:
        raise ValueError(
            f"action packing schema must be {ACTION_PACKING_SCHEMA!r}, got {schema!r}"
        )
    batch, channels, frames, per_frame, width = target_shape
    if width != 1 or downsample_factor <= 0:
        raise ValueError("action carrier requires width=1 and a positive downsample factor")
    if frames % downsample_factor != 0:
        raise ValueError("action carrier frames must be divisible by downsample_factor")
    compact_frames = frames // downsample_factor
    capacity = compact_frames * per_frame
    if aligned.shape != (batch, capacity, channels):
        raise ValueError(
            f"expected aligned actions {(batch, capacity, channels)}, got {tuple(aligned.shape)}"
        )

    packed = torch.zeros(
        batch, channels, frames, per_frame, width,
        device=aligned.device, dtype=aligned.dtype,
    )
    compact = aligned.reshape(batch, compact_frames, per_frame, channels)
    packed[:, :, ::downsample_factor, :, 0] = compact.permute(0, 3, 1, 2)
    return packed
