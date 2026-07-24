from types import SimpleNamespace
from pathlib import Path

import pytest

from distillation_flowmap.runtime_metadata import (
    flowmap_runtime_metadata,
    resolve_action_downsample_factor,
)


def test_runtime_metadata_persists_continuous_action_factor():
    config = SimpleNamespace(
        num_train_timesteps=1000,
        snr_shift=5.0,
        action_snr_shift=0.05,
        action_downsample_factor=1,
    )

    metadata = flowmap_runtime_metadata(config)

    assert metadata == {
        "num_train_timesteps": 1000,
        "snr_shift": 5.0,
        "action_snr_shift": 0.05,
        "action_downsample_factor": 1,
    }


def test_checkpoint_action_factor_overrides_legacy_job_fallback():
    assert resolve_action_downsample_factor(
        {"action_downsample_factor": 1}, fallback=4
    ) == 1


def test_legacy_checkpoint_keeps_legacy_job_fallback():
    assert resolve_action_downsample_factor({}, fallback=4) == 4


@pytest.mark.parametrize("factor", [0, -1])
def test_non_positive_action_factor_is_rejected(factor):
    with pytest.raises(ValueError, match="must be positive"):
        resolve_action_downsample_factor(
            {"action_downsample_factor": factor}, fallback=4
        )


def test_checkpoint_saver_and_server_share_runtime_metadata_contract():
    root = Path(__file__).resolve().parents[2]
    trainer_source = (
        root / "distillation_flowmap" / "flowmap_trainer.py"
    ).read_text(encoding="utf-8")
    server_source = (
        root / "wan_va" / "wan_va_server.py"
    ).read_text(encoding="utf-8")

    assert "config_dict.update(flowmap_runtime_metadata(self.config))" in trainer_source
    assert "resolve_action_downsample_factor(" in server_source
    assert 'getattr(job_config, "action_downsample_factor", 1)' in server_source
