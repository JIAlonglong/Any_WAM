import importlib
import os
import sys


def test_config_uses_empty_emb_path_env(monkeypatch):
    monkeypatch.setenv("DATASET_PATH", "/data/robotwin_aug_500")
    monkeypatch.setenv("EMPTY_EMB_PATH", "/data/robotwin-empty/empty_emb.pt")
    sys.modules.pop("distillation_flowmap.config", None)
    sys.modules.pop("distillation_flowmap.config_robotwin_fullfinetune_stage1_warmup", None)

    module = importlib.import_module("distillation_flowmap.config_robotwin_fullfinetune_stage1_warmup")

    assert module.cfg.dataset_path == "/data/robotwin_aug_500"
    assert module.cfg.empty_emb_path == "/data/robotwin-empty/empty_emb.pt"


def test_train_dataset_override_respects_empty_emb_path_env(monkeypatch):
    sys.path.insert(0, os.path.join(os.getcwd(), "distillation_flowmap"))
    train = importlib.import_module("distillation_flowmap.train")

    monkeypatch.setenv("EMPTY_EMB_PATH", "/shared/empty_emb.pt")
    assert train._resolve_empty_emb_path("/data/robotwin_aug_500") == "/shared/empty_emb.pt"

    monkeypatch.delenv("EMPTY_EMB_PATH")
    assert train._resolve_empty_emb_path("/data/robotwin_aug_500") == (
        "/data/robotwin_aug_500/empty_emb.pt"
    )
