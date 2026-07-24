import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "evaluation" / "libero" / "run_lingbotva_task0_124_eval_4gpu.sh"


def _make_family(tmp_path):
    stage1 = tmp_path / "stage1" / "target_student" / "transformer"
    stage1.mkdir(parents=True)
    ablation = tmp_path / "ablation"
    for arm in ("stage1_only", "anchor_only", "field_only", "apm"):
        (
            ablation
            / arm
            / "seed_42"
            / "stage2"
            / "checkpoints"
            / "step_500"
            / "target_student"
            / "transformer"
        ).mkdir(parents=True)
    return stage1.parents[1], ablation


def test_check_only_plans_five_models_at_matched_124_budgets(tmp_path):
    stage1, ablation = _make_family(tmp_path)
    env = {**os.environ, "CHECK_ONLY": "1"}
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--stage1-checkpoint",
            str(stage1),
            "--ablation-root",
            str(ablation),
            "--train-steps",
            "500",
            "--episodes",
            "20",
            "--gpu-ids",
            "0,1,2,3",
            "--output-root",
            str(tmp_path / "eval"),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    jobs = [line for line in result.stdout.splitlines() if line.startswith("JOB ")]
    assert len(jobs) == 15
    for model in ("stage1", "stage1_only", "anchor_only", "field_only", "apm"):
        model_jobs = [line for line in jobs if f"model={model} " in line]
        assert len(model_jobs) == 3
        assert {line.split("steps=")[1].split()[0] for line in model_jobs} == {
            "1",
            "2",
            "4",
        }
    assert all("suite=libero_10 task=0:1" in line for line in jobs)
    assert all(
        f"video_steps={steps} action_steps={steps}" in line
        for steps in (1, 2, 4)
        for line in jobs
        if f"steps={steps} " in line
    )
    assert not (tmp_path / "eval").exists()


def test_requires_exactly_four_unique_gpus(tmp_path):
    stage1, ablation = _make_family(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--stage1-checkpoint",
            str(stage1),
            "--ablation-root",
            str(ablation),
            "--output-root",
            str(tmp_path / "eval"),
            "--gpu-ids",
            "0,1",
        ],
        cwd=ROOT,
        env={**os.environ, "CHECK_ONLY": "1"},
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "exactly four" in result.stderr
