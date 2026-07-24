import importlib
import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FLOWMAP_DIR = os.path.join(REPO_ROOT, "distillation_flowmap")
WANVA_DIR = os.path.join(REPO_ROOT, "wan_va")
for path in (FLOWMAP_DIR, WANVA_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from distillation_flowmap.flowmap_trainer import (
    FlowMapDistiller,
    _set_video_channel_config_from_heads,
    _write_json_atomic,
)


class _FakeModel:
    def __init__(self):
        self.patch_size = [1, 2, 2]
        self.patch_embedding_mlp = torch.nn.Linear(64, 3072)
        self.proj_out = torch.nn.Linear(3072, 64)


def test_set_video_channel_config_from_heads_uses_adapted_cosmos_latent_heads():
    config_dict = {
        "in_channels": 48,
        "out_channels": 48,
        "patch_size": [1, 2, 2],
    }

    _set_video_channel_config_from_heads(config_dict, _FakeModel())

    assert config_dict["in_channels"] == 16
    assert config_dict["out_channels"] == 16


def test_cosmos_latent_stage2_defaults_are_memory_safe(monkeypatch):
    for name in (
        "GRADIENT_CHECKPOINTING",
        "OPD_AUX_WARMUP_STEPS",
        "OPD_ROLLOUT_STEP_PAIRS",
        "OPD_TRANSITION_GROUP_WEIGHT",
    ):
        monkeypatch.delenv(name, raising=False)

    module = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_cosmos_latent_cdiff"
    )
    module = importlib.reload(module)

    assert module.cfg.gradient_checkpointing is True
    assert module.cfg.opd_aux_warmup_steps >= 8
    assert module.cfg.opd_rollout_step_pairs == [[1, 1]]
    assert module.cfg.opd_transition_group_weight <= 1e-2


def test_stage_configs_identify_their_persisted_training_contract():
    stage1 = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage1"
    )
    stage2 = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    )

    assert stage1.cfg.training_contract_stage == "raw_stage1"
    assert stage2.cfg.training_contract_stage == "progressive_stage2"


class _CheckpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = {"in_channels": 16, "out_channels": 16}


class _Stateful:
    def state_dict(self):
        return {}


def test_checkpoint_config_persists_stage2_contract_atomically(tmp_path, monkeypatch):
    import distillation_flowmap.flowmap_trainer as trainer_module

    trainer = FlowMapDistiller.__new__(FlowMapDistiller)
    trainer.student = _CheckpointModel()
    trainer.target_student = None
    trainer.use_lora = False
    trainer.use_dmd = False
    trainer.discriminator = None
    trainer.save_dir = tmp_path
    trainer.step = 17
    trainer.optimizer = _Stateful()
    trainer.lr_scheduler = _Stateful()
    trainer.config = SimpleNamespace(
        rank=0,
        training_contract_stage="progressive_stage2",
        contract_version=2,
        action_packing_schema="downsample_survivor_v2",
        action_downsample_factor=4,
        deployment_timestep_start=1000,
        deployment_timestep_end=0,
        deployment_joint_steps=(1, 2, 4),
        deployment_joint_rollout_interval=4,
        deployment_action_weight=1.0,
        raw_teacher_window_is_auxiliary=True,
    )
    monkeypatch.setattr(
        trainer_module,
        "get_model_state_dict",
        lambda model, options: model.state_dict(),
    )
    monkeypatch.setattr(trainer_module, "save_file", lambda state, path: None)
    monkeypatch.setattr(trainer_module.torch, "save", lambda state, path: None)
    replacements = []
    real_replace = os.replace

    def recording_replace(source, destination):
        replacements.append((os.fspath(source), os.fspath(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(trainer_module.os, "replace", recording_replace)

    trainer._save_checkpoint()

    config_path = (
        tmp_path / "step_17" / "online_student" / "transformer" / "config.json"
    )
    payload = json.loads(config_path.read_text())
    assert payload["training_contract_stage"] == "progressive_stage2"
    assert payload["joint_student_steps"] == [1, 2, 4]
    assert payload["deployment_action_weight"] == 1.0
    source, destination = replacements[-1]
    assert os.path.dirname(source) == os.fspath(config_path.parent)
    assert os.path.basename(source).startswith(".config.json.")
    assert os.path.basename(source).endswith(".tmp")
    assert destination == os.fspath(config_path)
    assert not os.path.exists(source)


def test_atomic_checkpoint_config_failure_preserves_prior_file(tmp_path, monkeypatch):
    import distillation_flowmap.flowmap_trainer as trainer_module

    config_path = tmp_path / "config.json"
    config_path.write_text('{"prior": true}\n')

    def fail_json_dump(payload, handle, **kwargs):
        handle.write('{"partial":')
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(trainer_module.json, "dump", fail_json_dump)

    with pytest.raises(RuntimeError, match="serialization failed"):
        _write_json_atomic(config_path, {"next": True})

    assert config_path.read_text() == '{"prior": true}\n'
    assert not (tmp_path / ".config.json.tmp").exists()


@pytest.mark.parametrize("stale_kind", ["file", "dangling_symlink"])
def test_atomic_checkpoint_config_ignores_fixed_stale_temp(
    tmp_path, stale_kind
):
    config_path = tmp_path / "config.json"
    stale_path = tmp_path / ".config.json.tmp"
    victim_path = tmp_path / "victim.json"
    if stale_kind == "file":
        stale_path.write_text("stale")
    else:
        stale_path.symlink_to(victim_path.name)

    _write_json_atomic(config_path, {"complete": True})

    assert json.loads(config_path.read_text()) == {"complete": True}
    if stale_kind == "file":
        assert stale_path.read_text() == "stale"
    else:
        assert stale_path.is_symlink()
        assert not victim_path.exists()


def test_invalid_contract_prevents_all_checkpoint_writes(tmp_path, monkeypatch):
    import distillation_flowmap.flowmap_trainer as trainer_module

    trainer = FlowMapDistiller.__new__(FlowMapDistiller)
    trainer.student = _CheckpointModel()
    trainer.target_student = None
    trainer.use_lora = False
    trainer.use_dmd = False
    trainer.discriminator = None
    trainer.save_dir = tmp_path
    trainer.step = 17
    trainer.optimizer = _Stateful()
    trainer.lr_scheduler = _Stateful()
    trainer.config = SimpleNamespace(
        rank=0,
        training_contract_stage="progressive_stage2",
        contract_version=2,
        action_packing_schema="downsample_survivor_v2",
        action_downsample_factor=4,
        deployment_timestep_start=1000,
        deployment_timestep_end=0,
        deployment_joint_steps=(1, 2, 4),
        deployment_joint_rollout_interval=4,
        deployment_action_weight=2.0,
        raw_teacher_window_is_auxiliary=True,
    )
    writes = []
    monkeypatch.setattr(
        trainer_module,
        "get_model_state_dict",
        lambda *args, **kwargs: writes.append("state_dict") or {},
    )
    monkeypatch.setattr(
        trainer_module,
        "save_file",
        lambda *args, **kwargs: writes.append("weights"),
    )
    monkeypatch.setattr(
        trainer_module.torch,
        "save",
        lambda *args, **kwargs: writes.append("torch_save"),
    )

    with pytest.raises(ValueError, match="deployment_action_weight"):
        trainer._save_checkpoint()

    assert writes == []
    assert not list(tmp_path.iterdir())
