import importlib
import json
import os
import sys
from pathlib import Path

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
