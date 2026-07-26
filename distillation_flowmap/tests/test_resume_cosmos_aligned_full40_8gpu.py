from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = (
    ROOT / "distillation_flowmap" / "resume_cosmos_aligned_full40_8gpu.sh"
)


def _write_executable(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)
    return path


def _fixture(tmp_path: Path, *, checkpoint: bool = True):
    output_root = tmp_path / "output"
    run_tag = "aligned-anchor-field-full40-20260725"
    if checkpoint:
        transformer = (
            output_root
            / run_tag
            / "stage1"
            / "checkpoints"
            / "step_1000"
            / "target_student"
            / "transformer"
        )
        transformer.mkdir(parents=True)
        (transformer / "config.json").write_text("{}")
        (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"x")

    calls_log = tmp_path / "calls.log"
    launcher = _write_executable(
        tmp_path / "record-launcher.sh",
        'printf "%s\\n" "$*" >> "$CALLS_LOG"\n',
    )
    env = os.environ.copy()
    env.update(
        {
            "CALLS_LOG": str(calls_log),
            "COSMOS_PIPELINE_LAUNCHER": str(launcher),
            "OUTPUT_ROOT": str(output_root),
            "RUN_TAG": run_tag,
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "PYTHON_BIN": "/tmp/python",
            "COSMOS_PREDICT2_REPO": str(tmp_path / "cosmos"),
            "COSMOS_POLICY_PYTHON": "/tmp/cosmos-python",
            "COSMOS_POLICY_EXTRA_PYTHONPATH": str(tmp_path / "cosmos-extra"),
            "WAN_STUDENT_BASE_MODEL_PATH": str(tmp_path / "wan"),
            "COSMOS_POLICY_PATH": str(tmp_path / "teacher"),
            "DATASET_PATH": str(tmp_path / "dataset"),
            "EMPTY_EMB_PATH": str(tmp_path / "dataset" / "empty_emb.pt"),
        }
    )
    return env, calls_log


def _option(tokens: list[str], name: str) -> str:
    index = tokens.index(name)
    return tokens[index + 1]


def test_resume_wrapper_runs_stage1_stage2_eval_in_order(tmp_path):
    env, calls_log = _fixture(tmp_path)
    result = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    calls = [
        line.split() for line in calls_log.read_text().splitlines()
    ]
    assert [_option(call, "--phase") for call in calls] == [
        "stage1",
        "stage2",
        "eval",
    ]
    assert _option(calls[0], "--resume-stage") == "stage1"
    assert _option(calls[0], "--resume-step") == "1000"
    assert "--resume-stage" not in calls[1]
    assert "--resume-stage" not in calls[2]
    for call in calls:
        assert _option(call, "--run-tag") == (
            "aligned-anchor-field-full40-20260725"
        )
        assert _option(call, "--stage1-steps") == "5000"
        assert _option(call, "--stage2-steps") == "10000"
        assert _option(call, "--save-interval") == "1000"
        assert _option(call, "--episodes") == "50"


def test_resume_wrapper_dry_run_propagates_to_all_phases(tmp_path):
    env, calls_log = _fixture(tmp_path)
    result = subprocess.run(
        ["bash", str(WRAPPER), "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    calls = [
        line.split() for line in calls_log.read_text().splitlines()
    ]
    assert len(calls) == 3
    assert all("--dry-run" in call for call in calls)


def test_resume_wrapper_rejects_missing_checkpoint_before_launch(tmp_path):
    env, calls_log = _fixture(tmp_path, checkpoint=False)
    result = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Stage-1 resume checkpoint" in result.stderr
    assert not calls_log.exists()
