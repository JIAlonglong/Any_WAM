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
