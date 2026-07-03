import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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
