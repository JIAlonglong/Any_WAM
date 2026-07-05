import torch

from distillation_flowmap.cosmos_future_aux import (
    image_l1_mse,
    normalize_future_images,
    select_future_frames,
)


def test_normalize_future_images_from_dict_uint8():
    primary = torch.zeros(2, 4, 16, 16, 3, dtype=torch.uint8)
    wrist = torch.full((2, 4, 16, 16, 3), 255, dtype=torch.uint8)

    result = normalize_future_images(
        {
            "observation.images.agentview_rgb": primary,
            "observation.images.eye_in_hand_rgb": wrist,
        },
        primary_key="observation.images.agentview_rgb",
        wrist_key="observation.images.eye_in_hand_rgb",
    )

    assert result.primary.shape == (2, 4, 3, 16, 16)
    assert result.wrist.shape == (2, 4, 3, 16, 16)
    assert result.primary.dtype == torch.float32
    assert result.primary.max().item() == 0.0
    assert result.wrist.min().item() == 1.0


def test_normalize_future_images_from_single_tensor():
    tensor = torch.rand(2, 4, 3, 16, 16)

    result = normalize_future_images(tensor)

    assert result.primary.shape == (2, 4, 3, 16, 16)
    assert result.wrist is None


def test_normalize_future_images_adds_batch_dim_for_single_sample_future():
    tensor = torch.zeros(4, 16, 16, 3, dtype=torch.uint8)

    result = normalize_future_images(tensor)

    assert result.primary.shape == (1, 4, 3, 16, 16)


def test_normalize_future_images_falls_back_to_numpy_array_values():
    array = torch.zeros(4, 16, 16, 3, dtype=torch.uint8).numpy()

    result = normalize_future_images({"primary_image": array})

    assert result.primary.shape == (1, 4, 3, 16, 16)


def test_normalize_future_images_accepts_single_hwc_image():
    image = torch.zeros(16, 16, 3, dtype=torch.uint8)

    result = normalize_future_images({"future_image": image})

    assert result.primary.shape == (1, 1, 3, 16, 16)


def test_select_future_frames_clamps_to_available_frames():
    tensor = torch.arange(1 * 3 * 1 * 1 * 1).reshape(1, 3, 1, 1, 1).float()

    selected = select_future_frames(tensor, num_frames=5)

    assert selected.shape[1] == 5
    assert selected[0, -1, 0, 0, 0].item() == tensor[0, -1, 0, 0, 0].item()


def test_image_l1_mse_uses_common_shape():
    a = torch.zeros(1, 2, 3, 8, 8)
    b = torch.ones(1, 2, 3, 8, 8)

    metrics = image_l1_mse(a, b)

    assert metrics["l1"] == 1.0
    assert metrics["mse"] == 1.0
