import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "evaluation" / "libero" / "run_cosmos_progressive_s4_suite.sh"


def _suite_env(tmp_path):
    marker = tmp_path / "python sentinel was invoked"
    sentinel = tmp_path / "python sentinel"
    sentinel.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('called', encoding='utf-8')\n"
        "raise SystemExit(99)\n",
        encoding="utf-8",
    )
    sentinel.chmod(sentinel.stat().st_mode | stat.S_IXUSR)
    env = os.environ.copy()
    env.update(
        {
            "SUITE_ROOT": str(tmp_path / "full suite output with spaces"),
            "S4_CKPT_ROOT": str(tmp_path / "public checkpoint with spaces"),
            "S4_DATASET_PATH": str(tmp_path / "latent dataset with spaces"),
            "S4_EMPTY_EMBEDDING": str(tmp_path / "empty embedding with spaces.pt"),
            "COSMOS_POLICY_PATH": str(tmp_path / "cosmos policy with spaces"),
            "COSMOS_POLICY_PYTHON": str(tmp_path / "official cosmos python"),
            "COSMOS_PREDICT2_REPO": str(tmp_path / "cosmos repo with spaces"),
            "PYTHON_BIN": str(sentinel),
            "SENTINEL_MARKER": str(marker),
        }
    )
    return env


def run_suite(mode, *, env):
    return subprocess.run(
        ["bash", str(SCRIPT), mode],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_success(result):
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def _command_tokens(output):
    return [
        shlex.split(line.partition("=")[2])
        for line in output.splitlines()
        if line.startswith("COMMAND=")
    ]


def test_suite_dry_run_plans_offline_and_formal_without_child_execution(tmp_path):
    env = _suite_env(tmp_path)
    result = run_suite("dry-run", env=env)
    _assert_success(result)

    output = result.stdout
    suite_root = Path(env["SUITE_ROOT"])
    protocol_root = suite_root / "protocol"
    assert "MODE=dry-run" in output
    assert "PHASE_PLAN=runtime_preflight" in output
    assert "PHASE_PLAN=protocol" in output
    assert "PHASE_PLAN=teacher_cache" in output
    assert "PHASE_PLAN=offline_paper" in output
    assert "PHASE_PLAN=closed_loop_formal" in output
    assert "OFFLINE_STUDENT_GPU=0" in output
    assert "OFFLINE_COSMOS_WORKER_GPU=1" in output
    assert "--selection-per-task 3" in output
    assert "--test-per-task 5" in output
    assert "--student-steps 4" in output
    assert "--teacher-steps 8" in output
    assert "FORMAL_SHARD_0_STUDENT_GPU=0" in output
    assert "FORMAL_SHARD_1_STUDENT_GPU=2" in output
    assert f"FORMAL_PROMPT_TABLE_PLAN={suite_root / 'closed_loop_formal' / 'prompt_embeddings.pt'}" in output
    commands = _command_tokens(output)
    assert any(
        "evaluation.libero.rollout_cosmos_progressive_s4" in command
        and "--preflight" in command
        and "CUDA_VISIBLE_DEVICES=0" in command
        and "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES=1" in command
        for command in commands
    )
    assert any(
        command[0] == str(Path(env["PYTHON_BIN"]))
        and "distillation_flowmap.prepare_cosmos_progressive_protocol" in command
        and str(protocol_root) in command
        and "1000,0" in command
        for command in commands
    )
    assert any(
        "distillation_flowmap.build_cosmos_progressive_teacher_cache" in command
        and str(protocol_root / "test_manifest.json") in command
        and str(protocol_root / "teacher_cache" / "test") in command
        and "CUDA_VISIBLE_DEVICES=0" in command
        and "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES=1" in command
        and command[command.index("--teacher-steps") + 1] == "8"
        for command in commands
    )
    assert any(
        "distillation_flowmap.eval_cosmos_progressive_s4_paper" in command
        and str(protocol_root / "test_manifest.json") in command
        and str(protocol_root / "teacher_cache" / "test") in command
        and "--student-steps" in command
        and "--teacher-steps" in command
        and command[command.index("--student-steps") + 1] == "4"
        and command[command.index("--teacher-steps") + 1] == "8"
        for command in commands
    )
    assert not suite_root.exists()
    assert not Path(env["SENTINEL_MARKER"]).exists()


def test_suite_run_rejects_existing_root_before_child_execution(tmp_path):
    env = _suite_env(tmp_path)
    Path(env["SUITE_ROOT"]).mkdir()

    result = run_suite("run", env=env)

    assert result.returncode == 2
    assert "SUITE_ROOT already exists; no-overwrite policy" in result.stderr
    assert not Path(env["SENTINEL_MARKER"]).exists()


def test_suite_rejects_invalid_live_checkpoint_before_root_or_child_execution(tmp_path):
    env = _suite_env(tmp_path)
    suite_root = Path(env["SUITE_ROOT"])

    result = run_suite("run", env=env)

    assert result.returncode == 2
    assert "S4_CKPT_ROOT is not a directory" in result.stderr
    assert not suite_root.exists()
    assert not Path(env["SENTINEL_MARKER"]).exists()


@pytest.mark.parametrize(
    "missing_name",
    [
        "SUITE_ROOT",
        "S4_CKPT_ROOT",
        "S4_DATASET_PATH",
        "S4_EMPTY_EMBEDDING",
        "COSMOS_POLICY_PATH",
        "COSMOS_POLICY_PYTHON",
        "COSMOS_PREDICT2_REPO",
    ],
)
def test_suite_rejects_missing_required_input_before_root_or_child(tmp_path, missing_name):
    env = _suite_env(tmp_path)
    suite_root = Path(env["SUITE_ROOT"])
    env.pop(missing_name)

    result = run_suite("run", env=env)

    assert result.returncode == 2
    assert f"set {missing_name}" in result.stderr
    assert not suite_root.exists()
    assert not Path(env["SENTINEL_MARKER"]).exists()


def test_suite_launcher_static_contract_uses_test_cache_and_existing_formal_launcher():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "distillation_flowmap.prepare_cosmos_progressive_protocol" in source
    assert "distillation_flowmap.build_cosmos_progressive_teacher_cache" in source
    assert "distillation_flowmap.eval_cosmos_progressive_s4_paper" in source
    assert "evaluation.libero.rollout_cosmos_progressive_s4" in source
    assert "run_cosmos_progressive_s4_eval.sh" in source
    assert '"${PROTOCOL_ROOT}/test_manifest.json"' in source
    assert '"${PROTOCOL_ROOT}/teacher_cache/test"' in source
    assert "--student-steps" in source
    assert "--teacher-steps" in source
    assert "S4_DRY_RUN=1" in source
    assert "S4_DRY_RUN=0" in source
