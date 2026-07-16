import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "evaluation" / "libero" / "run_cosmos_progressive_s4_eval.sh"


def _launcher_env(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "S4_CKPT_ROOT": str(tmp_path / "public checkpoint with spaces"),
            "EVAL_ROOT": str(tmp_path / "evaluation output with spaces"),
            "S4_PROMPT_TABLE": str(tmp_path / "prompt table with spaces.npz"),
            "S4_EMPTY_EMBEDDING": str(tmp_path / "empty embedding with spaces.pt"),
            "S4_SMOKE_DATASET_PATH": str(tmp_path / "latent dataset with spaces"),
            "S4_SMOKE_MANIFEST": str(tmp_path / "final test manifest with spaces.json"),
            "S4_SMOKE_PAIRS": str(tmp_path / "same prior pairs with spaces.json"),
            "S4_SMOKE_CACHE_DIR": str(tmp_path / "teacher cache with spaces"),
            "PYTHON_BIN": sys.executable,
            "S4_DRY_RUN": "1",
        }
    )
    return env


def run_launcher(mode, *, env):
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


def test_formal_dry_run_assigns_two_student_worker_pairs_and_task_shards(tmp_path):
    result = run_launcher("formal", env=_launcher_env(tmp_path))
    _assert_success(result)

    output = result.stdout
    assert "MODE=formal" in output
    assert "SHARD_0_STUDENT_GPU=0" in output
    assert "SHARD_0_COSMOS_WORKER_GPU=1" in output
    assert "SHARD_1_STUDENT_GPU=2" in output
    assert "SHARD_1_COSMOS_WORKER_GPU=3" in output
    assert "SHARD_0_TASK_RANGE=0,5" in output
    assert "SHARD_1_TASK_RANGE=5,10" in output
    assert "REQUESTED_SEEDS_PER_TASK=50" in output
    assert "REQUESTED_RECORDS=500" in output
    assert output.count("--env-seed") == 100


def test_gate_dry_run_uses_one_pair_and_five_shared_seeds_per_task(tmp_path):
    result = run_launcher("gate", env=_launcher_env(tmp_path))
    _assert_success(result)

    output = result.stdout
    assert "MODE=gate" in output
    assert "SHARD_0_STUDENT_GPU=0" in output
    assert "SHARD_0_COSMOS_WORKER_GPU=1" in output
    assert "SHARD_0_TASK_RANGE=0,10" in output
    assert "SHARD_1_" not in output
    assert "CUDA_VISIBLE_DEVICES=2" not in output
    assert "REQUESTED_SEEDS_PER_TASK=5" in output
    assert "REQUESTED_RECORDS=50" in output
    assert output.count("--env-seed") == 5
    assert output.count("--env-seed 0") == 1
    assert output.count("--env-seed 4") == 1


def test_dry_run_creates_no_output_and_never_executes_python_sentinel(tmp_path):
    env = _launcher_env(tmp_path)
    output_root = Path(env["EVAL_ROOT"])
    marker = tmp_path / "python-sentinel-was-invoked"
    sentinel = tmp_path / "python sentinel"
    sentinel.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('called', encoding='utf-8')\n"
        "raise SystemExit(99)\n",
        encoding="utf-8",
    )
    sentinel.chmod(sentinel.stat().st_mode | stat.S_IXUSR)
    env["PYTHON_BIN"] = str(sentinel)

    result = run_launcher("dry-run", env=env)
    _assert_success(result)

    assert "MODE=dry-run" in result.stdout
    assert "COMMAND=" in result.stdout
    assert not output_root.exists()
    assert not marker.exists()


def test_live_mode_rejects_existing_output_root_before_child_execution(tmp_path):
    env = _launcher_env(tmp_path)
    env["S4_DRY_RUN"] = "0"
    output_root = Path(env["EVAL_ROOT"])
    output_root.mkdir()
    marker = tmp_path / "python-sentinel-was-invoked"
    sentinel = tmp_path / "python sentinel"
    sentinel.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('called', encoding='utf-8')\n",
        encoding="utf-8",
    )
    sentinel.chmod(sentinel.stat().st_mode | stat.S_IXUSR)
    env["PYTHON_BIN"] = str(sentinel)

    result = run_launcher("gate", env=env)

    assert result.returncode == 2
    assert "EVAL_ROOT already exists; no-overwrite policy" in result.stderr
    assert not marker.exists()


def test_dry_run_commands_quote_paths_with_spaces(tmp_path):
    env = _launcher_env(tmp_path)
    result = run_launcher("formal", env=env)
    _assert_success(result)

    command_lines = [
        line.partition("=")[2]
        for line in result.stdout.splitlines()
        if line.startswith("COMMAND=") and "evaluation.libero.rollout_cosmos_progressive_s4" in line
    ]
    assert command_lines
    tokens = shlex.split(command_lines[0])
    assert env["S4_CKPT_ROOT"] in tokens
    assert env["S4_PROMPT_TABLE"] in tokens
    assert env["S4_EMPTY_EMBEDDING"] in tokens
    assert "-m" in tokens
    assert "evaluation.libero.rollout_cosmos_progressive_s4" in tokens


def test_smoke_dry_run_requires_prebuilt_cache_inputs_and_is_nonpaper(tmp_path):
    env = _launcher_env(tmp_path)
    result = run_launcher("smoke", env=env)
    _assert_success(result)

    output = result.stdout
    assert "MODE=smoke" in output
    assert f"SMOKE_MANIFEST={env['S4_SMOKE_MANIFEST']}" in output
    assert f"SMOKE_PAIRS={env['S4_SMOKE_PAIRS']}" in output
    assert f"SMOKE_CACHE_DIR={env['S4_SMOKE_CACHE_DIR']}" in output
    assert "--cache-only-smoke" in output
    assert "--skip-same-state-velocity" in output
    assert "SMOKE_RESULT=non-paper-cache-only" in output


def test_launcher_syntax_and_merge_guards_are_present():
    syntax = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    _assert_success(syntax)

    source = SCRIPT.read_text(encoding="utf-8")
    assert "-m evaluation.libero.rollout_cosmos_progressive_s4" in source
    assert "duplicate record" in source
    assert "checkpoint mismatch" in source
    assert "seed mismatch" in source
    assert "bootstrap_ci_95" in source
    assert "record_path.relative_to(root).parts" in source
    assert 'if "seed" in record and int(record["seed"]) != seed:' in source


def test_dry_run_plans_prompt_materialization_without_invoking_a_child(tmp_path):
    env = _launcher_env(tmp_path)
    env.pop("S4_PROMPT_TABLE")
    env["S4_DATASET_PATH"] = str(tmp_path / "training latent dataset with spaces")
    marker = tmp_path / "prompt-materializer-was-invoked"
    sentinel = tmp_path / "python sentinel"
    sentinel.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('called', encoding='utf-8')\n"
        "raise SystemExit(99)\n",
        encoding="utf-8",
    )
    sentinel.chmod(sentinel.stat().st_mode | stat.S_IXUSR)
    env["PYTHON_BIN"] = str(sentinel)

    result = run_launcher("formal", env=env)
    _assert_success(result)

    planned = Path(env["EVAL_ROOT"]) / "prompt_embeddings.pt"
    assert f"PROMPT_TABLE_PLAN={planned}" in result.stdout
    assert not Path(env["EVAL_ROOT"]).exists()
    assert not marker.exists()
