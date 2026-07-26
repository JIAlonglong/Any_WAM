from pathlib import Path

import pytest

from wan_va.native_teacher_contract import resolve_native_teacher_contract


def test_native_teacher_contract_accepts_matched_coarse_budget():
    assert resolve_native_teacher_contract(
        model_name="teacher_native",
        video_steps=2,
        action_steps=2,
    ) == {
        "backend": "native_teacher",
        "model": "teacher_native",
        "video_steps": 2,
        "action_steps": 2,
        "action_grid": "full",
        "video_action_bridge": "disabled",
    }


@pytest.mark.parametrize("steps", [0, -1])
def test_native_teacher_contract_rejects_non_positive_budget(steps):
    with pytest.raises(ValueError, match="positive"):
        resolve_native_teacher_contract(
            model_name="teacher_native",
            video_steps=steps,
            action_steps=steps,
        )


def test_native_teacher_contract_rejects_mismatched_budget():
    with pytest.raises(ValueError, match="matched"):
        resolve_native_teacher_contract(
            model_name="teacher_native",
            video_steps=1,
            action_steps=2,
        )


def test_native_teacher_contract_rejects_student_identity():
    with pytest.raises(ValueError, match="teacher_native"):
        resolve_native_teacher_contract(
            model_name="stage2",
            video_steps=1,
            action_steps=1,
        )


def test_native_teacher_server_uses_native_separate_loops():
    root = Path(__file__).resolve().parents[2]
    source = (
        root / "wan_va" / "wan_va_native_teacher_server.py"
    ).read_text(encoding="utf-8")

    assert "flowmap_inference" not in source
    assert "action_downsample_factor" not in source
    assert "# Native video denoising loop" in source
    assert "# Native action denoising loop" in source
    assert 'self.scheduler.set_timesteps(self.native_contract["video_steps"])' in source
    assert (
        "self.action_scheduler.set_timesteps("
        'self.native_contract["action_steps"]'
    ) in source
    assert "actions = self.action_scheduler.step(" in source
    assert "actions[:, :, ::" not in source


def test_native_teacher_server_exposes_deployment_cli():
    root = Path(__file__).resolve().parents[2]
    source = (
        root / "wan_va" / "wan_va_native_teacher_server.py"
    ).read_text(encoding="utf-8")

    for flag in (
        "--checkpoint-path",
        "--num-steps",
        "--action-num-steps",
        "--model-name",
        "--latency-jsonl",
        "--save-root",
    ):
        assert flag in source
    assert "append_sampler_latency_record" in source
    assert "eval_metadata" in source
