"""Mask helpers for offline rollout diagnostics."""

import torch


def _frame_mask_from_tensor(tensor, frame_dim=2, eps=0.0):
    if tensor is None:
        return None
    if tensor.ndim <= frame_dim:
        raise ValueError(
            f"Expected tensor with frame_dim={frame_dim}, got shape {tuple(tensor.shape)}"
        )
    if tensor.dtype == torch.bool:
        valid = tensor
    else:
        valid = tensor.detach().abs() > eps
    order = [0, frame_dim] + [dim for dim in range(tensor.ndim) if dim not in (0, frame_dim)]
    valid = valid.permute(order).reshape(tensor.shape[0], tensor.shape[frame_dim], -1)
    return valid.any(dim=-1)


def _clip_or_pad_frame_mask(frame_mask, num_frames):
    if num_frames is None:
        return frame_mask
    num_frames = int(num_frames)
    if frame_mask.shape[1] > num_frames:
        return frame_mask[:, :num_frames]
    if frame_mask.shape[1] < num_frames:
        pad = torch.zeros(
            frame_mask.shape[0],
            num_frames - frame_mask.shape[1],
            device=frame_mask.device,
            dtype=torch.bool,
        )
        return torch.cat([frame_mask, pad], dim=1)
    return frame_mask


def video_frame_mask_from_batch(batch, num_frames=None):
    """Return [B, F] valid-video-frame mask.

    RobotWin pads video latents to a fixed temporal length, but only action
    positions carry a mask today. The action mask uses the same latent frame
    axis, so it is the most reliable available source for offline diagnostics.
    """
    if batch.get("latent_mask") is not None:
        frame_mask = _frame_mask_from_tensor(batch["latent_mask"], frame_dim=2)
    elif batch.get("actions_mask") is not None:
        frame_mask = _frame_mask_from_tensor(batch["actions_mask"], frame_dim=2)
    else:
        frame_mask = _frame_mask_from_tensor(batch["latents"], frame_dim=2, eps=0.0)
    return _clip_or_pad_frame_mask(frame_mask.bool(), num_frames)


def masked_video_mse_l1(diff, frame_mask=None):
    diff = diff.float()
    if frame_mask is None:
        return diff.pow(2).mean(), diff.abs().mean()
    mask = frame_mask.to(device=diff.device, dtype=diff.dtype)[:, None, :, None, None]
    denom = (mask.sum() * diff.shape[1] * diff.shape[3] * diff.shape[4]).clamp(min=1.0)
    return (diff.pow(2).mul(mask).sum() / denom, diff.abs().mul(mask).sum() / denom)


def masked_video_rms(value, frame_mask=None):
    value = value.float()
    if frame_mask is None:
        return value.pow(2).mean().sqrt()
    mask = frame_mask.to(device=value.device, dtype=value.dtype)[:, None, :, None, None]
    denom = (mask.sum() * value.shape[1] * value.shape[3] * value.shape[4]).clamp(min=1.0)
    return (value.pow(2).mul(mask).sum() / denom).sqrt()


def crop_latent_video_to_valid_frames(latent, frame_mask, sample_idx=0):
    if frame_mask is None:
        return latent
    valid_count = int(frame_mask[int(sample_idx)].sum().item())
    valid_count = max(1, min(valid_count, latent.shape[2]))
    return latent[:, :, :valid_count].contiguous()
