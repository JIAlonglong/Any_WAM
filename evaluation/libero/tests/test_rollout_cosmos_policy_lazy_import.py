import importlib

import pytest


def test_libero_runtime_dependency_error_is_deferred_and_actionable(monkeypatch):
    rollout = importlib.import_module("evaluation.libero.rollout_cosmos_policy")
    real_import_module = rollout.importlib.import_module

    def missing_runtime(name):
        if name.startswith("libero."):
            raise ModuleNotFoundError("No module named 'robosuite'")
        return real_import_module(name)

    monkeypatch.setattr(rollout.importlib, "import_module", missing_runtime)

    with pytest.raises(
        RuntimeError,
        match="LIBERO simulator dependencies are required to construct an evaluation environment",
    ):
        rollout._load_libero_runtime()
