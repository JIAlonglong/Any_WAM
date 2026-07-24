import importlib
import sys

import pytest


MODULE = "distillation_flowmap.config_libero_apm_lora_ablation"


def fresh_import():
    sys.modules.pop(MODULE, None)
    return importlib.import_module(MODULE)


def test_ablation_config_enables_lora_and_manifest(monkeypatch):
    monkeypatch.setenv("DATASET_SAMPLE_MANIFEST", "/tmp/train.json")
    monkeypatch.setenv("OPD_DANCEOPD_ENDPOINT_WEIGHT", "0")
    monkeypatch.setenv("OPD_DANCEOPD_VELOCITY_WEIGHT", "1")

    cfg = fresh_import().cfg

    assert cfg.use_lora
    assert cfg.lora_rank == 128
    assert cfg.lora_alpha == 64
    assert cfg.lora_dropout == 0.0
    assert cfg.learning_rate == 5e-6
    assert cfg.dataset_sample_manifest == "/tmp/train.json"
    assert cfg.attn_mode == "torch"
    assert not cfg.opd_aux_action
    assert not cfg.opd_joint_action_rollout
    assert cfg.opd_danceopd_action_velocity_weight == 0.0
    assert cfg.opd_danceopd_endpoint_weight == 0.0
    assert cfg.opd_danceopd_velocity_weight == 1.0


def test_ablation_config_rejects_missing_manifest(monkeypatch):
    monkeypatch.delenv("DATASET_SAMPLE_MANIFEST", raising=False)

    with pytest.raises(ValueError, match="DATASET_SAMPLE_MANIFEST"):
        fresh_import()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LORA_RANK", "0"),
        ("LORA_ALPHA", "0"),
        ("LORA_DROPOUT", "-0.1"),
        ("LEARNING_RATE", "0"),
    ],
)
def test_ablation_config_rejects_invalid_lora_settings(
    monkeypatch, name, value
):
    monkeypatch.setenv("DATASET_SAMPLE_MANIFEST", "/tmp/train.json")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match="LoRA"):
        fresh_import()
