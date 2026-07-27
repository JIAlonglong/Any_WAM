import subprocess
from pathlib import Path

from distillation_flowmap.tests.test_cosmos_stage1_stage2_eval_pipeline import (
    ROOT,
    WRAPPER,
    _pipeline_env,
    _write_executable,
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
        "ALIGNED_VIDEO_OPD_INTERVAL=4",
        "OPD_AUX_INTERVAL=4",
        "OPD_DANCEOPD_ROLLOUT_STEPS=2,4",
        "OPD_DANCEOPD_ANCHOR_TEACHER_STEPS=8",
        "OPD_DANCEOPD_ENDPOINT_WEIGHT=1.0",
        "OPD_DANCEOPD_VELOCITY_WEIGHT=1.0",
        "ACTION_DOWNSAMPLE_FACTOR=4",
        "VIDEO_ACTION_BRIDGE=0",
    ):
        assert resolved in stdout
    assert "ALIGNED_TRAIN_COMMAND=" not in stdout


def _command(stdout: str, label: str) -> str:
    prefix = f"{label}="
    lines = [line for line in stdout.splitlines() if line.startswith(prefix)]
    assert len(lines) == 1
    return lines[0]


def _assert_authentic_read_only_chain(stdout: str, *, episodes: int):
    stage1 = _command(stdout, "STAGE1_COMMAND")
    lock = _command(stdout, "LOCK_COMMAND")
    stage2 = _command(stdout, "STAGE2_COMMAND")
    evaluation = _command(stdout, "EVAL_COMMAND")
    assert "stage1.sh dry-run" in stage1
    assert "--steps" in stage1 and "--save-interval" in stage1
    assert "locks.sh" in lock and "--dry-run" in lock
    assert "stage2.sh universal-video-action" in stage2
    assert "--dry-run" in stage2
    assert f"S4_EPISODES_PER_TASK={episodes}" in evaluation
    assert "eval.sh dry-run" in evaluation

    stage1_source = (
        ROOT / "distillation_flowmap" / "run_cosmos_raw_stage1_8gpu.sh"
    ).read_text(encoding="utf-8")
    stage2_source = (
        ROOT / "distillation_flowmap" / "run_cosmos_libero_train_8gpu.sh"
    ).read_text(encoding="utf-8")
    assert stage1_source.count('"--nproc_per_node=8"') == 1
    assert stage2_source.count("--nproc_per_node=8") == 1


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
    _assert_authentic_read_only_chain(result.stdout, episodes=50)


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
    _assert_authentic_read_only_chain(result.stdout, episodes=1)


def test_live_pipeline_propagates_episode_count_to_evaluator(tmp_path):
    env, output_root = _pipeline_env(tmp_path)
    eval_script = _write_executable(
        tmp_path / "episode-eval.sh",
        'printf "eval:%s:%s\\n" "$S4_EPISODES_PER_TASK" "$*" >> "$CALLS_LOG"\n',
    )
    env["COSMOS_JOINT124_EVAL_LAUNCHER"] = str(eval_script)
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--phase",
            "all",
            "--output-root",
            str(output_root),
            "--run-tag",
            "episodes",
            "--episodes",
            "3",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    calls = Path(env["CALLS_LOG"]).read_text(encoding="utf-8").splitlines()
    assert calls[-1] == "eval:3:run"
