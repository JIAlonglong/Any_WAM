import subprocess
from pathlib import Path

from distillation_flowmap.tests.test_cosmos_stage1_stage2_eval_pipeline import (
    ROOT,
    WRAPPER,
    _pipeline_env,
)


def _read_only_pipeline(tmp_path: Path, *arguments: str):
    env, output_root = _pipeline_env(tmp_path)
    env.pop("OUTPUT_ROOT", None)
    before = sorted(
        str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")
    )
    result = subprocess.run(
        ["bash", str(WRAPPER), *arguments],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    after = sorted(
        str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")
    )
    return result, before, after, output_root, Path(env["CALLS_LOG"])


def _assert_complete_aligned_plan(stdout: str):
    for resolved in (
        "OPD_AUX_INTERVAL=4",
        "OPD_DANCEOPD_ROLLOUT_STEPS=2,4",
        "OPD_DANCEOPD_ANCHOR_TEACHER_STEPS=8",
        "OPD_DANCEOPD_ENDPOINT_WEIGHT=1.0",
        "OPD_DANCEOPD_VELOCITY_WEIGHT=1.0",
        "ACTION_DOWNSAMPLE_FACTOR=1",
        "VIDEO_ACTION_BRIDGE=0",
    ):
        assert resolved in stdout
    assert stdout.count("--nproc_per_node=8") == 1


def test_phase_check_prints_one_complete_eight_rank_plan_and_writes_nothing(
    tmp_path,
):
    result, before, after, output_root, calls = _read_only_pipeline(
        tmp_path,
        "--phase",
        "check",
        "--run-tag",
        "aligned-anchor-check",
    )
    assert result.returncode == 0, result.stderr
    assert before == after
    assert not output_root.exists()
    assert not calls.exists()
    _assert_complete_aligned_plan(result.stdout)


def test_all_phase_dry_run_accepts_smoke_aliases_and_writes_nothing(tmp_path):
    result, before, after, output_root, calls = _read_only_pipeline(
        tmp_path,
        "--phase",
        "all",
        "--steps",
        "1",
        "--save-interval",
        "1",
        "--episodes",
        "1",
        "--run-tag",
        "aligned-anchor-dry",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert before == after
    assert not output_root.exists()
    assert not calls.exists()
    assert "EVAL_EPISODES=1" in result.stdout
    _assert_complete_aligned_plan(result.stdout)
