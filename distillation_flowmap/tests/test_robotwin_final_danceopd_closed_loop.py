import os
import subprocess
from pathlib import Path


def _runner_path():
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "ablation" / "run_final_danceopd_closed_loop.sh"


def test_closed_loop_dry_run_freezes_final_main_protocol(tmp_path):
    script = _runner_path()

    assert script.exists()

    env = os.environ.copy()
    env["ROOT"] = str(tmp_path / "final_danceopd")
    completed = subprocess.run(
        ["bash", str(script), "--dry-run", "--main-only"],
        cwd=script.parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    output = completed.stdout
    for variant in (
        "final_w_o_opd",
        "final_endpoint_only_danceopd",
        "final_danceopd_velocity_only",
        "final_stepwam_danceopd",
    ):
        assert variant in output
    for task in (
        "place_a2b_right",
        "put_object_cabinet",
        "stack_bowls_three",
        "lift_pot",
        "place_can_basket",
        "handover_block",
        "open_microwave",
        "open_laptop",
        "pick_dual_bottles",
        "blocks_ranking_size",
        "place_burger_fries",
        "rotate_qrcode",
    ):
        assert task in output
    assert "final_local_adjacent_only" not in output
    assert "final_action_only" not in output
    assert "--checkpoint-path" in output
    assert "target_student/transformer" in output
    assert "--num-steps 4" in output
    assert "--test_num 50" in output
    assert "--nfe 4" in output
    assert "--no-save-visualization" in output
    assert "ROBOTWIN_ROOT=" in output
    assert "assets/objects/objaverse/list.json" in script.read_text(encoding="utf-8")
    assert "assets/embodiments" in script.read_text(encoding="utf-8")


def test_closed_loop_relative_root_is_made_absolute_before_client_launch(tmp_path):
    script = _runner_path()
    relative_root = "relative-final-danceopd"

    completed = subprocess.run(
        ["bash", str(script), "--dry-run", "--main-only", "--root", relative_root],
        cwd=script.parents[2],
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=True,
    )

    expected_root = script.parents[2] / relative_root
    assert (
        f"--checkpoint-path {expected_root}/final_w_o_opd/seed_0/stage2/"
        "checkpoints/step_5000/target_student/transformer"
    ) in completed.stdout
    assert (
        f"--save-root {expected_root}/final_w_o_opd/seed_0/stage2/closed_loop"
    ) in completed.stdout
