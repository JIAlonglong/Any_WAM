import torch

from distillation_flowmap.rollout_masking import (
    crop_latent_video_to_valid_frames,
    masked_video_mse_l1,
    masked_video_rms,
    video_frame_mask_from_batch,
)


def test_video_frame_mask_uses_actions_mask_time_axis():
    actions_mask = torch.zeros(2, 3, 5, 4, 1, dtype=torch.bool)
    actions_mask[0, :, :2] = True
    actions_mask[1, :, :4] = True
    batch = {
        "latents": torch.ones(2, 8, 5, 2, 2),
        "actions_mask": actions_mask,
    }

    mask = video_frame_mask_from_batch(batch, num_frames=5)

    assert mask.tolist() == [
        [True, True, False, False, False],
        [True, True, True, True, False],
    ]


def test_masked_video_metrics_ignore_padded_frames():
    diff = torch.zeros(1, 1, 4, 1, 1)
    diff[:, :, :2] = 2.0
    diff[:, :, 2:] = 100.0
    frame_mask = torch.tensor([[True, True, False, False]])

    mse, l1 = masked_video_mse_l1(diff, frame_mask)
    rms = masked_video_rms(diff, frame_mask)

    assert torch.isclose(mse, torch.tensor(4.0))
    assert torch.isclose(l1, torch.tensor(2.0))
    assert torch.isclose(rms, torch.tensor(2.0))


def test_crop_latent_video_to_valid_frames():
    latent = torch.randn(1, 4, 6, 2, 2)
    frame_mask = torch.tensor([[True, True, True, False, False, False]])

    cropped = crop_latent_video_to_valid_frames(latent, frame_mask, sample_idx=0)

    assert cropped.shape == (1, 4, 3, 2, 2)
    assert torch.equal(cropped, latent[:, :, :3])
