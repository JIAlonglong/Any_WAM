import importlib
import sys


def test_robotwin_stage2_exposes_light_eval_env(monkeypatch):
    monkeypatch.setenv("ENABLE_LIGHT_EVAL", "1")
    monkeypatch.setenv("LIGHT_EVAL_INTERVAL", "7")
    monkeypatch.setenv("LIGHT_EVAL_NUM_BATCHES", "2")
    monkeypatch.setenv("LIGHT_EVAL_SEED", "123")
    monkeypatch.setenv("LIGHT_EVAL_START_INDEX", "4")
    sys.modules.pop("distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow", None)

    module = importlib.import_module("distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow")
    cfg = module.cfg

    assert cfg.enable_light_eval is True
    assert cfg.light_eval_interval == 7
    assert cfg.light_eval_num_batches == 2
    assert cfg.light_eval_seed == 123
    assert cfg.light_eval_start_index == 4
    assert cfg.light_eval_pairs == [(1000, 1000), (1000, 0), (750, 250)]


def test_robotwin_stage2_exposes_dataset_sample_manifest_env(monkeypatch):
    monkeypatch.setenv("DATASET_SAMPLE_MANIFEST", "/tmp/train_manifest.json")
    sys.modules.pop("distillation_flowmap.config_robotwin_fullfinetune_stage1_warmup", None)
    sys.modules.pop("distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow", None)

    module = importlib.import_module("distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow")

    assert module.cfg.dataset_sample_manifest == "/tmp/train_manifest.json"
