import json
from pathlib import Path


EXPECTED_VARIANTS = [
    ("f0", "final_w_o_opd", "final_ablation"),
    ("f1", "final_endpoint_only_danceopd", "final_ablation"),
    ("f2", "final_danceopd_velocity_only", "final_ablation"),
    ("f3", "final_stepwam_danceopd", "final_ablation"),
    ("s1", "final_local_adjacent_only", "structural_control"),
    ("s2", "final_action_only", "structural_control"),
]


def _variants_payload():
    path = Path(__file__).resolve().parents[1] / "ablation" / "robotwin_stepwam_variants.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_variants_metadata_only_exposes_the_frozen_final_protocol():
    payload = _variants_payload()

    assert payload["defaults"] == {
        "teacher": "lingbot-va",
        "benchmark": "robotwin",
        "stage1_steps": 5000,
        "stage2_steps": 5000,
        "student_steps": 4,
        "stage1_config": "distillation_flowmap.config_robotwin_fullfinetune_stage1_warmup",
        "stage2_config": "distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow",
    }
    variants = payload["variants"]
    assert [
        (variant["id"], variant["name"], variant["priority"])
        for variant in variants
    ] == EXPECTED_VARIANTS
    assert all(variant["seeds"] == [0] for variant in variants)
    assert all(variant["requires_code_flag"] is False for variant in variants)
