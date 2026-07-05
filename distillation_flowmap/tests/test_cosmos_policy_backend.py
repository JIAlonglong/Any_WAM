import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _write_minimal_cosmos_policy_checkpoint(root: Path):
    root.mkdir(parents=True)
    torch.save({"net.x_embedder.proj.1.weight": torch.zeros(1)}, root / "Cosmos-Policy-LIBERO-Predict2-2B.pt")
    (root / "config.json").write_text(json.dumps({"action_dim": 7, "action_horizon": 16}))
    (root / "libero_dataset_statistics.json").write_text(json.dumps({"actions_min": [0.0], "actions_max": [1.0]}))
    (root / "libero_t5_embeddings.pkl").write_bytes(b"placeholder")


def test_train_accepts_cosmos_policy_teacher_backend(tmp_path):
    sys.path.insert(0, os.path.join(os.getcwd(), "distillation_flowmap"))
    train = importlib.import_module("distillation_flowmap.train")
    ckpt = tmp_path / "cosmos_policy"
    _write_minimal_cosmos_policy_checkpoint(ckpt)

    resolved = train._normalize_teacher_model_path(str(ckpt), teacher_backend="cosmos_policy")

    assert resolved == str(ckpt.resolve())


def test_train_still_rejects_cosmos_policy_path_for_wanva_backend(tmp_path):
    sys.path.insert(0, os.path.join(os.getcwd(), "distillation_flowmap"))
    train = importlib.import_module("distillation_flowmap.train")
    ckpt = tmp_path / "cosmos_policy"
    _write_minimal_cosmos_policy_checkpoint(ckpt)

    try:
        train._normalize_teacher_model_path(str(ckpt), teacher_backend="wanva")
    except FileNotFoundError as exc:
        assert "transformer/config.json" in str(exc)
    else:
        raise AssertionError("expected WanVA backend to reject a Cosmos Policy checkpoint")


def test_cosmos_policy_action_teacher_returns_wanva_action_tokens(tmp_path):
    from distillation_flowmap.cosmos_policy_adapter import CosmosPolicyActionTeacher

    ckpt = tmp_path / "cosmos_policy"
    _write_minimal_cosmos_policy_checkpoint(ckpt)
    teacher = CosmosPolicyActionTeacher(str(ckpt), dtype=torch.float32)

    action_target = torch.randn(2, 3, 4, 5, 1)
    tokens = teacher.action_target_tokens({"targets": action_target})

    assert tokens.shape == (2, 20, 3)
    assert torch.allclose(tokens[:, 0], action_target[:, :, 0, 0, 0])


def test_libero_state_to_cosmos_proprio_uses_official_order():
    from distillation_flowmap.cosmos_policy_adapter import libero_state_to_cosmos_proprio

    state = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.25, -0.25])
    proprio = libero_state_to_cosmos_proprio(state)

    assert proprio.shape == (9,)
    assert torch.allclose(
        torch.as_tensor(proprio),
        torch.tensor([0.25, -0.25, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]),
        atol=1e-6,
    )


def test_cosmos_actions_map_to_flowmap_x0_with_quantile_norm():
    from distillation_flowmap.cosmos_policy_adapter import cosmos_actions_to_flowmap_x0

    actions = torch.tensor(
        [[[0.25, 0.50, 0.75, 0.00, 1.00, -0.50, 0.10],
          [0.50, 0.25, 0.00, 1.00, 0.50, 0.25, -0.10]]],
        dtype=torch.float32,
    )
    inverse_ids = list(range(7)) + [7] * 23
    x0 = cosmos_actions_to_flowmap_x0(
        actions,
        target_shape=(1, 30, 2, 4, 1),
        q01=[0.0] * 30,
        q99=[1.0] * 30,
        inverse_used_action_channel_ids=inverse_ids,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert x0.shape == (1, 30, 2, 4, 1)
    flat = x0.permute(0, 2, 3, 4, 1).reshape(1, 8, 30)
    assert torch.allclose(flat[0, 0, :7], actions[0, 0] * 2.0 - 1.0, atol=2e-6)
    assert torch.allclose(flat[0, 1, :7], actions[0, 1] * 2.0 - 1.0, atol=2e-6)
    assert torch.count_nonzero(flat[0, :, 7:]) == 0
    assert torch.count_nonzero(flat[0, 2:]) == 0


def test_raw_cosmos_teacher_uses_provider_action_x0(tmp_path):
    from distillation_flowmap.cosmos_policy_adapter import CosmosPolicyActionTeacher

    ckpt = tmp_path / "cosmos_policy"
    _write_minimal_cosmos_policy_checkpoint(ckpt)
    cfg = SimpleNamespace(
        cosmos_policy_use_raw_inference=True,
        norm_stat={"q01": [0.0] * 30, "q99": [1.0] * 30},
        inverse_used_action_channel_ids=list(range(7)) + [7] * 23,
    )
    teacher = CosmosPolicyActionTeacher(str(ckpt), dtype=torch.float32, config=cfg)
    teacher._raw_action_provider = lambda raw_batch: torch.ones(1, 2, 7) * 0.5

    x0 = teacher.action_target_x0(
        {"latent": torch.zeros(1, 30, 2, 4, 1)},
        raw_batch={
            "raw_primary_image": torch.zeros(1, 128, 128, 3, dtype=torch.uint8),
            "raw_wrist_image": torch.zeros(1, 128, 128, 3, dtype=torch.uint8),
            "raw_proprio": torch.zeros(1, 9),
            "raw_task": ["dummy task"],
        },
    )

    flat = x0.permute(0, 2, 3, 4, 1).reshape(1, 8, 30)
    assert torch.allclose(flat[0, :2, :7], torch.zeros(2, 7), atol=2e-6)
    assert torch.count_nonzero(flat[0, :, 7:]) == 0


def test_raw_cosmos_teacher_result_provider_preserves_future_predictions(tmp_path):
    from distillation_flowmap.cosmos_policy_adapter import CosmosPolicyActionTeacher

    ckpt = tmp_path / "cosmos_policy"
    _write_minimal_cosmos_policy_checkpoint(ckpt)
    teacher = CosmosPolicyActionTeacher(str(ckpt), dtype=torch.float32)
    future_image = np.full((4, 4, 3), 128, dtype=np.uint8)
    teacher._raw_action_provider = lambda raw_batch: {
        "actions": torch.ones(1, 2, 7) * 0.25,
        "future_image_predictions": [{"future_image": future_image}],
        "value_prediction": [0.75],
    }

    result = teacher.predict_raw_action_result(
        {
            "raw_primary_image": torch.zeros(1, 128, 128, 3, dtype=torch.uint8),
            "raw_wrist_image": torch.zeros(1, 128, 128, 3, dtype=torch.uint8),
            "raw_proprio": torch.zeros(1, 9),
            "raw_task": ["dummy task"],
        },
        include_future=True,
    )

    assert torch.allclose(result["actions"], torch.ones(1, 2, 7) * 0.25)
    assert result["future_image_predictions"][0]["future_image"].shape == (4, 4, 3)
    assert result["future_image_predictions"][0]["future_image"].dtype == np.uint8
    assert result["value_prediction"] == [0.75]


def test_official_cosmos_future_prediction_selector_accepts_dict_and_batch_list():
    from distillation_flowmap.cosmos_future_video import (
        OFFICIAL_COSMOS_FUTURE_VIDEO_SOURCE,
        select_first_future_prediction,
    )

    prediction = {"future_image": np.full((4, 4, 3), 11, dtype=np.uint8)}

    assert OFFICIAL_COSMOS_FUTURE_VIDEO_SOURCE == "official_cosmos_future_image_predictions"
    assert select_first_future_prediction({"future_image_predictions": prediction}) is prediction
    assert select_first_future_prediction({"future_image_predictions": [prediction]}) is prediction
    assert select_first_future_prediction({"future_image_predictions": []}) is None
    assert select_first_future_prediction({}) is None


def test_raw_cosmos_batch_for_official_future_video_requires_raw_fields():
    from distillation_flowmap.cosmos_future_video import build_raw_cosmos_batch

    batch = {
        "latents": torch.zeros(1),
        "raw_primary_image": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
        "raw_wrist_image": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
        "raw_task": ["open the drawer"],
    }

    try:
        build_raw_cosmos_batch(batch)
    except KeyError as exc:
        message = str(exc)
        assert "raw_proprio" in message
        assert "COSMOS_POLICY_USE_RAW_INFERENCE=1" in message
    else:
        raise AssertionError("expected official future video batch construction to require raw_proprio")


def test_raw_cosmos_batch_for_official_future_video_keeps_only_policy_inputs():
    from distillation_flowmap.cosmos_future_video import build_raw_cosmos_batch

    batch = {
        "latents": torch.ones(1),
        "actions": torch.ones(1),
        "raw_primary_image": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
        "raw_wrist_image": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
        "raw_proprio": torch.zeros(1, 9),
        "raw_task": ["open the drawer"],
    }

    raw_batch = build_raw_cosmos_batch(batch)

    assert set(raw_batch) == {"raw_primary_image", "raw_wrist_image", "raw_proprio", "raw_task"}
    assert raw_batch["raw_task"] == ["open the drawer"]


def test_predict_official_cosmos_future_uses_include_future_flag():
    from distillation_flowmap.cosmos_future_video import predict_official_future_prediction

    prediction = {"future_image": np.full((4, 4, 3), 88, dtype=np.uint8)}

    class FakeTeacher:
        def __init__(self):
            self.calls = []

        def predict_raw_action_result(self, raw_batch, include_future=False):
            self.calls.append((raw_batch, include_future))
            return {"actions": torch.zeros(1, 2, 7), "future_image_predictions": [prediction]}

    batch = {
        "latents": torch.ones(1),
        "raw_primary_image": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
        "raw_wrist_image": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
        "raw_proprio": torch.zeros(1, 9),
        "raw_task": ["open the drawer"],
    }
    teacher = FakeTeacher()

    selected = predict_official_future_prediction(teacher, batch)

    assert selected is prediction
    assert len(teacher.calls) == 1
    raw_batch, include_future = teacher.calls[0]
    assert include_future is True
    assert set(raw_batch) == {"raw_primary_image", "raw_wrist_image", "raw_proprio", "raw_task"}


def test_future_prediction_to_video_np_stacks_official_wrist_and_primary():
    from distillation_flowmap.rollout_eval_video_stage2 import future_prediction_to_video_np

    prediction = {
        "future_wrist_image": np.full((4, 6, 3), 10, dtype=np.uint8),
        "future_image": np.full((2, 3, 3), 20, dtype=np.uint8),
    }

    video = future_prediction_to_video_np(prediction)

    assert video.shape == (1, 4, 12, 3)
    assert video.dtype == np.uint8
    assert np.all(video[0, :, :6] == 10)
    assert np.all(video[0, :, 6:] == 20)


def test_future_prediction_to_video_np_repeats_to_reference_frame_count():
    from distillation_flowmap.rollout_eval_video_stage2 import future_prediction_to_video_np

    prediction = {"future_image": np.full((4, 6, 3), 20, dtype=np.uint8)}

    video = future_prediction_to_video_np(prediction, num_frames=5)

    assert video.shape == (5, 4, 6, 3)
    assert np.all(video == 20)


def test_future_predictions_to_video_np_resamples_multiple_predictions():
    from distillation_flowmap.rollout_eval_video_stage2 import future_predictions_to_video_np

    predictions = [
        {"future_image": np.full((4, 6, 3), 10, dtype=np.uint8)},
        {"future_image": np.full((4, 6, 3), 20, dtype=np.uint8)},
        {"future_image": np.full((4, 6, 3), 30, dtype=np.uint8)},
    ]

    video = future_predictions_to_video_np(predictions, num_frames=6)

    assert video.shape == (6, 4, 6, 3)
    assert [int(video[i, 0, 0, 0]) for i in range(6)] == [10, 10, 20, 20, 30, 30]


def test_predict_official_future_sequence_reads_multiple_raw_dataset_frames():
    from distillation_flowmap.rollout_eval_video_stage2 import predict_official_future_prediction_sequence

    class FakeDataset:
        new_metas = [{"episode_index": 0, "start_frame": 10, "end_frame": 14, "tasks": ["open"]}]

        def __len__(self):
            return 1

        def _get_raw_policy_observation(self, cur_meta, local_frame_index):
            return {
                "raw_primary_image": torch.zeros(8, 8, 3, dtype=torch.uint8),
                "raw_wrist_image": torch.zeros(8, 8, 3, dtype=torch.uint8),
                "raw_proprio": torch.full((9,), float(local_frame_index)),
                "raw_task": "open",
            }

    class FakeTeacher:
        def __init__(self):
            self.frame_ids = []

        def predict_raw_action_result(self, raw_batch, include_future=False):
            assert include_future is True
            frame_id = int(raw_batch["raw_proprio"][0].item())
            self.frame_ids.append(frame_id)
            return {
                "actions": torch.zeros(1, 2, 7),
                "future_image_predictions": {
                    "future_image": np.full((4, 4, 3), frame_id, dtype=np.uint8),
                },
            }

    teacher = FakeTeacher()

    predictions = predict_official_future_prediction_sequence(
        teacher,
        FakeDataset(),
        sample_index=0,
        num_predictions=3,
    )

    assert teacher.frame_ids == [10, 12, 13]
    assert [int(pred["future_image"][0, 0, 0]) for pred in predictions] == [10, 12, 13]


def test_pad_frames_to_min_duration_repeats_last_frame():
    from distillation_flowmap.cosmos_future_video import pad_frames_to_min_duration

    first = np.zeros((2, 2, 3), dtype=np.uint8)
    padded = pad_frames_to_min_duration([first], fps=10, min_seconds=1.0)

    assert len(padded) == 10
    assert np.all(padded[0] == 0)
    assert np.all(padded[-1] == 0)
    assert padded[-1] is not first


def test_save_cosmos_future_video_writes_prediction_comparison(tmp_path):
    from evaluation.libero.rollout_cosmos_policy import save_cosmos_future_video

    obs_frames = [
        {
            "observation.images.agentview_rgb": np.full((16, 16, 3), 10, dtype=np.uint8),
            "observation.images.eye_in_hand_rgb": np.full((16, 16, 3), 20, dtype=np.uint8),
        },
        {
            "observation.images.agentview_rgb": np.full((16, 16, 3), 30, dtype=np.uint8),
            "observation.images.eye_in_hand_rgb": np.full((16, 16, 3), 40, dtype=np.uint8),
        },
    ]
    future_predictions = [
        {
            "future_image": np.full((8, 8, 3), 50, dtype=np.uint8),
            "future_wrist_image": np.full((8, 8, 3), 60, dtype=np.uint8),
        },
        {
            "future_image": np.full((8, 8, 3), 70, dtype=np.uint8),
            "future_wrist_image": np.full((8, 8, 3), 80, dtype=np.uint8),
        },
    ]
    save_path = tmp_path / "future.mp4"

    save_cosmos_future_video(obs_frames, future_predictions, save_path, fps=5)

    assert save_path.is_file()
    assert save_path.stat().st_size > 0


def test_save_cosmos_future_chunk_video_writes_one_prediction_per_chunk(tmp_path):
    from evaluation.libero.rollout_cosmos_policy import save_cosmos_future_chunk_video

    future_predictions = [
        {
            "future_image": np.full((8, 8, 3), 50, dtype=np.uint8),
            "future_wrist_image": np.full((8, 8, 3), 60, dtype=np.uint8),
        },
        {
            "future_image": np.full((8, 8, 3), 70, dtype=np.uint8),
            "future_wrist_image": np.full((8, 8, 3), 80, dtype=np.uint8),
        },
    ]
    save_path = tmp_path / "future_chunks.mp4"

    save_cosmos_future_chunk_video(future_predictions, save_path, fps=2)

    assert save_path.is_file()
    assert save_path.stat().st_size > 0


def test_raw_teacher_stats_use_action_mask_and_channels():
    from distillation_flowmap.cosmos_policy_adapter import compute_masked_action_stats

    teacher = torch.tensor(
        [[[[[1.0], [2.0]], [[3.0], [4.0]]],
          [[[2.0], [4.0]], [[6.0], [8.0]]]]],
        dtype=torch.float32,
    )
    target = torch.ones_like(teacher)
    mask = torch.tensor([[[[[1.0], [0.0]], [[1.0], [0.0]]]]])

    stats = compute_masked_action_stats(teacher, target, mask)

    assert torch.allclose(stats["mse"], torch.tensor(7.5))
    assert torch.allclose(stats["l1"], torch.tensor(2.0))
    assert torch.allclose(stats["teacher_abs_mean"], torch.tensor(3.0))
    assert torch.allclose(stats["target_abs_mean"], torch.tensor(1.0))


def test_convert_input_format_preserves_raw_non_tensor_fields():
    from distillation.data import DataMixin

    class Dummy(DataMixin):
        pass

    mixin = Dummy()
    mixin.device = torch.device("cpu")
    batch = {
        "latents": torch.ones(1),
        "raw_task": ["open the drawer"],
        "raw_meta": {"episode": 3},
    }

    converted = mixin.convert_input_format(batch)

    assert torch.equal(converted["latents"], torch.ones(1))
    assert converted["raw_task"] == ["open the drawer"]
    assert converted["raw_meta"] == {"episode": 3}


def test_libero_cosmos_policy_configs_are_action_only(monkeypatch):
    monkeypatch.setenv("COSMOS_POLICY_PATH", "/tmp/cosmos-policy")
    monkeypatch.setenv("STUDENT_BASE_MODEL_PATH", "/tmp/wanva-base")
    sys.modules.pop("distillation_flowmap.config_libero_cosmos_policy_stage1", None)
    sys.modules.pop("distillation_flowmap.config_libero_cosmos_policy_stage2", None)

    stage1 = importlib.import_module("distillation_flowmap.config_libero_cosmos_policy_stage1").cfg
    stage2 = importlib.import_module("distillation_flowmap.config_libero_cosmos_policy_stage2").cfg

    for cfg in (stage1, stage2):
        assert cfg.teacher_backend == "cosmos_policy"
        assert cfg.teacher_model_path == "/tmp/cosmos-policy"
        assert cfg.student_base_model_path == "/tmp/wanva-base"
        assert cfg.distill_video is False
        assert cfg.distill_action is True
        assert cfg.enable_light_eval is False
        assert cfg.use_opd_aux is False
        assert cfg.cosmos_policy_use_raw_inference is False
        assert cfg.return_raw_observation is False


def test_cosmos_dual_teacher_stage1_config_imports(monkeypatch):
    monkeypatch.setenv("COSMOS_POLICY_PATH", "/tmp/cosmos-policy")
    monkeypatch.setenv("STUDENT_BASE_MODEL_PATH", "/tmp/wanva-base")
    sys.modules.pop(
        "distillation_flowmap.config_libero_cosmos_policy_stage1_dual_teacher", None)

    cfg = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage1_dual_teacher"
    ).cfg

    assert cfg.teacher_backend == "cosmos_policy"
    assert cfg.action_teacher_backend == "cosmos_policy"
    assert cfg.teacher_model_path == "/tmp/cosmos-policy"
    assert cfg.student_base_model_path == "/tmp/wanva-base"
    assert cfg.video_teacher_backend == "wanva"
    assert cfg.video_teacher_model_path == "/tmp/wanva-base"
    assert cfg.distill_video is True
    assert cfg.distill_action is True
    assert cfg.action_use_flowmap is True
    assert cfg.diffusion_ratio == 0.5
    assert cfg.consistency_ratio == 0.25
    assert cfg.flowmap_ratio == 0.25


def test_cosmos_all_cosmos_stage1_flowmap_config_imports(monkeypatch):
    monkeypatch.setenv("COSMOS_POLICY_PATH", "/tmp/cosmos-policy")
    monkeypatch.setenv("STUDENT_BASE_MODEL_PATH", "/tmp/wanva-base")
    sys.modules.pop(
        "distillation_flowmap.config_libero_cosmos_policy_stage1_all_cosmos_flowmap", None)

    cfg = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage1_all_cosmos_flowmap"
    ).cfg

    assert cfg.teacher_backend == "cosmos_policy"
    assert cfg.action_teacher_backend == "cosmos_policy"
    assert cfg.teacher_model_path == "/tmp/cosmos-policy"
    assert cfg.student_base_model_path == "/tmp/wanva-base"
    assert cfg.cosmos_video_target is True
    assert cfg.cosmos_video_vae_model_path == "/tmp/wanva-base"
    assert cfg.cosmos_policy_use_raw_inference is True
    assert cfg.return_raw_observation is True
    assert cfg.distill_video is True
    assert cfg.distill_action is True
    assert cfg.action_aware is False
    assert cfg.use_gt_regression is False
    assert cfg.use_central_diff is False
    assert cfg.action_use_flowmap is False
    assert cfg.diffusion_ratio == 0.5
    assert cfg.consistency_ratio == 0.25
    assert cfg.flowmap_ratio == 0.25


def test_cosmos_stage1_wanva_cdiff_config_imports(monkeypatch):
    monkeypatch.setenv("COSMOS_POLICY_PATH", "/tmp/cosmos-policy")
    monkeypatch.setenv("STUDENT_BASE_MODEL_PATH", "/tmp/wanva-base")
    sys.modules.pop(
        "distillation_flowmap.config_libero_cosmos_policy_stage1_wanva_cdiff", None)

    cfg = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage1_wanva_cdiff"
    ).cfg

    assert cfg.teacher_backend == "cosmos_policy"
    assert cfg.action_teacher_backend == "cosmos_policy"
    assert cfg.teacher_model_path == "/tmp/cosmos-policy"
    assert cfg.student_base_model_path == "/tmp/wanva-base"
    assert cfg.cosmos_video_target is True
    assert cfg.cosmos_policy_use_raw_inference is True
    assert cfg.distill_video is True
    assert cfg.distill_action is True
    assert cfg.cosmos_video_cdiff_aux is True
    assert cfg.cosmos_video_cdiff_teacher_model_path == "/tmp/wanva-base"
    assert cfg.cosmos_video_cdiff_mode == "primary"
    assert cfg.cosmos_video_cdiff_loss_weight == 1.0
    assert cfg.cosmos_video_endpoint_loss_weight == 1e-3
    assert cfg.use_central_diff is True
    assert cfg.cosmos_action_lingbotva_stage1 is True
    assert cfg.action_use_flowmap is True
    assert cfg.use_action_distill is True
    assert cfg.action_aware is True
    assert cfg.use_gt_regression is True
    assert cfg.gt_regression_weight == 0.15
    assert cfg.action_aware_weight == 0.1
    assert cfg.num_ddim_timesteps_action == 1
    assert cfg.use_opd_aux is False


def test_cosmos_stage1_wanva_cdiff_aux_mode_keeps_endpoint_primary(monkeypatch):
    monkeypatch.setenv("COSMOS_POLICY_PATH", "/tmp/cosmos-policy")
    monkeypatch.setenv("STUDENT_BASE_MODEL_PATH", "/tmp/wanva-base")
    monkeypatch.setenv("COSMOS_VIDEO_CDIFF_MODE", "aux")
    sys.modules.pop(
        "distillation_flowmap.config_libero_cosmos_policy_stage1_wanva_cdiff", None)

    cfg = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage1_wanva_cdiff"
    ).cfg

    assert cfg.cosmos_video_cdiff_mode == "aux"
    assert cfg.cosmos_video_endpoint_loss_weight == 1.0


def test_cosmos_dual_teacher_stage2_config_imports(monkeypatch):
    monkeypatch.setenv("COSMOS_POLICY_PATH", "/tmp/cosmos-policy")
    monkeypatch.setenv("STUDENT_BASE_MODEL_PATH", "/tmp/wanva-base")
    sys.modules.pop(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_dual_teacher", None)

    cfg = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_dual_teacher"
    ).cfg

    assert cfg.teacher_backend == "cosmos_policy"
    assert cfg.action_teacher_backend == "cosmos_policy"
    assert cfg.teacher_model_path == "/tmp/cosmos-policy"
    assert cfg.student_base_model_path == "/tmp/wanva-base"
    assert cfg.video_teacher_backend == "wanva"
    assert cfg.video_teacher_model_path == "/tmp/wanva-base"
    assert cfg.distill_video is True
    assert cfg.distill_action is True
    assert cfg.action_use_flowmap is True
    assert cfg.use_opd_aux is True


def test_cosmos_wanva_cdiff_stage2_lingbotva_config_imports(monkeypatch):
    monkeypatch.setenv("TEACHER_PATH", "/tmp/wanva-teacher")
    sys.modules.pop(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_wanva_cdiff", None)

    cfg = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_wanva_cdiff"
    ).cfg

    assert getattr(cfg, "teacher_backend", "wanva") == "wanva"
    assert cfg.teacher_model_path == "/tmp/wanva-teacher"
    assert cfg.distill_mode == "flashwam"
    assert cfg.distill_video is True
    assert cfg.distill_action is True
    assert cfg.use_central_diff is True
    assert cfg.action_use_flowmap is True
    assert cfg.use_action_distill is True
    assert cfg.action_aware is True
    assert cfg.use_gt_regression is True
    assert cfg.use_opd_aux is True
    assert cfg.opd_aux_action is False
    assert cfg.resume_online_from_target is True
    assert cfg.reset_resume_step is True
    assert cfg.resume_from_path.endswith(
        "output_libero_cosmos_policy_stage1_wanva_cdiff/checkpoints/step_5000"
    )


def test_rollout_video_conditioning_keeps_first_frame_clean():
    from distillation_flowmap.rollout_eval_video_stage2 import apply_first_frame_condition

    clean_latents = torch.randn(2, 3, 4, 5, 6)
    noisy_latents = torch.randn_like(clean_latents)
    timesteps = torch.full((2, 4), 1000.0)
    target_timesteps = torch.full((2, 4), 250.0)

    conditioned, conditioned_t, conditioned_r = apply_first_frame_condition(
        noisy_latents.clone(),
        timesteps.clone(),
        target_timesteps.clone(),
        clean_latents,
    )

    assert torch.allclose(conditioned[:, :, 0:1], clean_latents[:, :, 0:1])
    assert torch.allclose(conditioned[:, :, 1:], noisy_latents[:, :, 1:])
    assert torch.equal(conditioned_t[:, 0], torch.zeros(2))
    assert torch.equal(conditioned_r[:, 0], torch.zeros(2))
    assert torch.equal(conditioned_t[:, 1:], timesteps[:, 1:])
    assert torch.equal(conditioned_r[:, 1:], target_timesteps[:, 1:])


def test_cosmos_action_only_rollout_video_skips_student_latent_diagnostic_by_default():
    from distillation_flowmap.rollout_eval_video_stage2 import should_save_student_latent_video

    assert should_save_student_latent_video(
        is_cosmos_policy_teacher=True,
        distill_video=False,
        allow_cosmos_latent_diagnostic=False,
    ) is False
    assert should_save_student_latent_video(
        is_cosmos_policy_teacher=True,
        distill_video=False,
        allow_cosmos_latent_diagnostic=True,
    ) is True
    assert should_save_student_latent_video(
        is_cosmos_policy_teacher=True,
        distill_video=True,
        allow_cosmos_latent_diagnostic=False,
    ) is True
    assert should_save_student_latent_video(
        is_cosmos_policy_teacher=False,
        distill_video=False,
        allow_cosmos_latent_diagnostic=False,
    ) is True


def test_cosmos_teacher_official_eval_defaults_match_nvidia_libero_settings():
    from evaluation.libero.rollout_cosmos_policy import apply_eval_defaults

    args = SimpleNamespace(
        official_libero_eval=True,
        libero_benchmark="libero_10",
        cosmos_repo="/opt/cosmos",
        seed=None,
        env_seed=None,
        warmup_steps=None,
        warmup_gripper=None,
        max_env_steps=None,
        initial_states_json=None,
    )

    apply_eval_defaults(args)

    assert args.seed == 195
    assert args.env_seed == 0
    assert args.warmup_steps == 10
    assert args.warmup_gripper == -1.0
    assert args.max_env_steps == 530
    assert args.initial_states_json == (
        "/opt/cosmos/"
        "cosmos_predict2/_src/predict2/cosmos_policy/experiments/robot/libero/"
        "libero_10_metainfo.json"
    )


def test_cosmos_teacher_official_initial_state_loader_skips_failed_demos(tmp_path):
    from evaluation.libero.rollout_cosmos_policy import load_initial_state_from_metainfo

    metainfo = {
        "put_the_bowl_on_the_plate": {
            "demo_0": {"success": True, "initial_state": [1.0, 2.0, 3.0]},
            "demo_1": {"success": False, "initial_state": [4.0, 5.0, 6.0]},
        }
    }
    path = tmp_path / "libero_10_metainfo.json"
    path.write_text(json.dumps(metainfo))

    state, source, skipped = load_initial_state_from_metainfo(
        str(path),
        "put the bowl on the plate",
        episode_idx=0,
    )
    failed_state, failed_source, failed_skipped = load_initial_state_from_metainfo(
        str(path),
        "put the bowl on the plate",
        episode_idx=1,
    )

    assert state.tolist() == [1.0, 2.0, 3.0]
    assert source.endswith("libero_10_metainfo.json:put_the_bowl_on_the_plate/demo_0")
    assert skipped is False
    assert failed_state is None
    assert failed_source.endswith("libero_10_metainfo.json:put_the_bowl_on_the_plate/demo_1")
    assert failed_skipped is True
