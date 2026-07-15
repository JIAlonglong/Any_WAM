import os
import subprocess
from pathlib import Path


def _runner_path():
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "ablation" / "run_final_danceopd_eval.sh"


def test_final_danceopd_eval_dry_run_freezes_main_protocol(tmp_path):
    script = _runner_path()

    assert script.exists()

    env = os.environ.copy()
    env["ROOT"] = str(tmp_path / "final_danceopd")
    completed = subprocess.run(
        ["bash", str(script), "--dry-run", "--main-only", "--skip-assets"],
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
    assert "final_local_adjacent_only" not in output
    assert "final_action_only" not in output
    assert "--cache-teacher-trajectories" in output
    assert "--trajectory-teacher-steps 1 2 4" in output
    assert "--rollout-drift" in output
    assert "--same-state-velocity" in output
    assert "--decoded-video-metrics" in output
    assert "--decoded-video-device cuda" in output
    assert "--decoded-video-lpips" in output
    assert "--require-complete-main" in output
    assert "train.py" not in output
