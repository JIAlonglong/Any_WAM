from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class NormalizedFutureImages:
    primary: torch.Tensor
    wrist: torch.Tensor | None = None


def _as_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _to_btchw(value) -> torch.Tensor:
    tensor = _as_tensor(value)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 5:
        raise ValueError(
            f"Expected future image tensor with 5 dims, got shape={tuple(tensor.shape)}"
        )
    if tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(0, 1, 4, 2, 3)
    elif tensor.shape[2] in (1, 3):
        tensor = tensor
    else:
        raise ValueError(f"Cannot infer channel dimension from shape={tuple(tensor.shape)}")
    tensor = tensor.to(torch.float32)
    if tensor.numel() > 0 and tensor.max() > 2.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0).contiguous()


def _select(mapping: Mapping[str, object], key: str | None):
    if key and key in mapping:
        return mapping[key]
    if key:
        short_key = key.split(".")[-1]
        if short_key in mapping:
            return mapping[short_key]
    return None


def select_official_future_camera_images(
    prediction: Mapping[str, object],
    obs_cam_keys: list[str] | tuple[str, ...],
) -> list[object]:
    """Select official Cosmos future images in the WanVA observation-camera order."""
    out = []
    for key in obs_cam_keys:
        key_l = str(key).lower()
        if "eye_in_hand" in key_l or "wrist" in key_l:
            image = prediction.get("future_wrist_image")
            if image is None:
                image = prediction.get("future_wrist_image2")
        else:
            image = prediction.get("future_image")
            if image is None:
                image = prediction.get("future_image2")
        if image is None:
            raise KeyError(
                f"Cosmos future prediction does not contain an image for camera {key!r}; "
                f"available keys={sorted(prediction.keys())}"
            )
        out.append(image)
    return out


def normalize_future_images(
    future_image_predictions,
    *,
    primary_key: str | None = None,
    wrist_key: str | None = None,
) -> NormalizedFutureImages:
    if isinstance(future_image_predictions, torch.Tensor):
        return NormalizedFutureImages(primary=_to_btchw(future_image_predictions))

    if not isinstance(future_image_predictions, Mapping):
        raise TypeError(
            "future_image_predictions must be a tensor or mapping of camera name to tensor"
        )

    primary = _select(future_image_predictions, primary_key)
    if primary is None:
        for value in future_image_predictions.values():
            try:
                _to_btchw(value)
                primary = value
                break
            except (TypeError, ValueError):
                continue
    if primary is None:
        raise ValueError("No tensor found in future_image_predictions")

    wrist = _select(future_image_predictions, wrist_key)
    return NormalizedFutureImages(
        primary=_to_btchw(primary),
        wrist=_to_btchw(wrist) if wrist is not None else None,
    )


def select_future_frames(images: torch.Tensor, num_frames: int) -> torch.Tensor:
    if images.ndim != 5:
        raise ValueError(f"Expected [B,T,C,H,W], got {tuple(images.shape)}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if images.shape[1] >= num_frames:
        return images[:, :num_frames].contiguous()
    pad = images[:, -1:].expand(-1, num_frames - images.shape[1], -1, -1, -1)
    return torch.cat([images, pad], dim=1).contiguous()


def image_l1_mse(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if pred.shape != target.shape:
        raise ValueError(
            f"Image metric shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}"
        )
    diff = pred.to(torch.float32) - target.to(torch.float32)
    return {
        "l1": float(diff.abs().mean().item()),
        "mse": float((diff * diff).mean().item()),
    }
