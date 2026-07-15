import importlib
import os
import subprocess
from pathlib import Path

import pytest


def _repo_root():
    return Path(__file__).resolve().parents[2]


def test_sapien_test_propagates_renderer_initialization_error(monkeypatch):
    module = importlib.import_module("evaluation.robotwin.test_render")

    def fail_setup_scene(self):
        raise RuntimeError("failed to find a rendering device")

    monkeypatch.setattr(module.Sapien_TEST, "setup_scene", fail_setup_scene)

    with pytest.raises(
        RuntimeError, match="SAPIEN renderer initialization failed"
    ) as error:
        module.Sapien_TEST()

    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "failed to find a rendering device"


def test_closed_loop_dry_run_includes_renderer_preflight(tmp_path):
    repo_root = _repo_root()
    script = (
        repo_root
        / "distillation_flowmap"
        / "ablation"
        / "run_final_danceopd_closed_loop.sh"
    )
    env = os.environ.copy()
    env["ROOT"] = str(tmp_path / "final_danceopd")

    completed = subprocess.run(
        ["bash", str(script), "--dry-run", "--main-only"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    source = script.read_text(encoding="utf-8")
    assert "build_renderer_preflight_command" in source
    assert "run_renderer_preflight" in source
    assert "evaluation.robotwin.test_render" in completed.stdout
    assert "Sapien_TEST" in completed.stdout
    preflight_call = 'run_renderer_preflight "' + "$" + '{gpu}"'
    variant_call = 'run_variant "' + "$" + '{variant_index}"'
    assert source.index(preflight_call) < source.index(variant_call)
