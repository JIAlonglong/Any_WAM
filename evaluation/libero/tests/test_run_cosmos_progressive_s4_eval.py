import json
import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest


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
    assert output.count("--student-steps 4") == 102
    assert output.count("--episode-index-offset") == 100
    assert "--save-video" not in output


def _shard_records(output):
    records = {}
    for line in output.splitlines():
        if not line.startswith("SHARD_"):
            continue
        key, value = line.split("=", 1)
        _, shard, field = key.split("_", 2)
        records.setdefault(int(shard), {})[field] = value
    return [
        (
            shard,
            int(values["STUDENT_GPU"]),
            int(values["COSMOS_WORKER_GPU"]),
            values["TASK_RANGE"],
        )
        for shard, values in sorted(records.items())
    ]


def test_formal_four_shards_use_all_eight_gpus_and_joint_k2(tmp_path):
    env = _launcher_env(tmp_path)
    env.update(
        {
            "S4_STUDENT_STEPS": "2",
            "S4_FORMAL_NUM_SHARDS": "4",
            "S4_VIDEO_SEEDS": "0,1",
        }
    )

    result = run_launcher("formal", env=env)
    _assert_success(result)

    assert _shard_records(result.stdout) == [
        (0, 0, 1, "0,3"),
        (1, 2, 3, "3,6"),
        (2, 4, 5, "6,8"),
        (3, 6, 7, "8,10"),
    ]
    assert result.stdout.count("--student-steps 2") == 204
    assert result.stdout.count("--save-video") == 8
    assert result.stdout.count("--episode-index-offset") == 200


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("S4_STUDENT_STEPS", "3", "S4_STUDENT_STEPS must be 1, 2, or 4"),
        ("S4_FORMAL_NUM_SHARDS", "3", "S4_FORMAL_NUM_SHARDS must be 2 or 4"),
        ("S4_VIDEO_SEEDS", "0,nope", "S4_VIDEO_SEEDS"),
        ("S4_VIDEO_SEEDS", "0,", "S4_VIDEO_SEEDS"),
        ("S4_VIDEO_SEEDS", "50", "S4_VIDEO_SEEDS"),
    ],
)
def test_formal_controls_are_strictly_validated(tmp_path, name, value, expected):
    env = _launcher_env(tmp_path)
    env[name] = value

    result = run_launcher("formal", env=env)

    assert result.returncode == 2
    assert expected in result.stderr


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


def _formal_merge_program():
    source = SCRIPT.read_text(encoding="utf-8")
    start_marker = (
        '    "${PYTHON_BIN}" - "${EVAL_ROOT}" "${S4_CKPT_ROOT}" "${seed_count}" '
        '"${S4_STUDENT_STEPS}" "${shard_plan}" <<'
        "'PY'\n"
    )
    end_marker = "\nPY\n}\n\n\nprepare_prompt_table"
    start = source.index(start_marker) + len(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def _run_formal_merge(root, checkpoint, *, steps=2, shard_plan="0:0:5;1:5:10"):
    return subprocess.run(
        [
            sys.executable,
            "-",
            str(root),
            str(checkpoint),
            "50",
            str(steps),
            shard_plan,
        ],
        cwd=ROOT,
        input=_formal_merge_program(),
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("payload_seed", "expected_error"),
    [
        (None, "seed mismatch: missing record seed"),
        (1, "seed mismatch: record seed=1 path seed=0"),
    ],
)
def test_formal_merge_rejects_missing_or_mismatched_payload_seed(
    tmp_path, payload_seed, expected_error
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output_root = tmp_path / "formal output"
    record_path = output_root / "shard_0" / "seed_0" / "records" / "task_0_episode_0.json"
    record_path.parent.mkdir(parents=True)
    record = {
        "task_idx": 0,
        "episode_idx": 0,
        "s4_checkpoint": str(checkpoint.resolve()),
        "student_steps": 2,
        "success": False,
    }
    if payload_seed is not None:
        record["seed"] = payload_seed
    record_path.write_text(json.dumps(record), encoding="utf-8")

    result = _run_formal_merge(output_root, checkpoint)

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_formal_merge_rejects_step_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output_root = tmp_path / "formal output"
    record_path = output_root / "shard_0" / "seed_0" / "records" / "task_0_episode_0.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps(
            {
                "task_idx": 0,
                "episode_idx": 0,
                "seed": 0,
                "s4_checkpoint": str(checkpoint.resolve()),
                "student_steps": 4,
                "success": False,
            }
        ),
        encoding="utf-8",
    )

    result = _run_formal_merge(output_root, checkpoint, steps=2)

    assert result.returncode != 0
    assert "step mismatch: expected=2 got=4" in result.stderr


def test_formal_merge_rejects_episode_index_that_does_not_match_path_seed(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output_root = tmp_path / "formal output"
    record_path = output_root / "shard_0" / "seed_7" / "records" / "task_0_episode_0.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps(
            {
                "task_idx": 0,
                "episode_idx": 0,
                "seed": 7,
                "s4_checkpoint": str(checkpoint.resolve()),
                "student_steps": 2,
                "success": False,
            }
        ),
        encoding="utf-8",
    )

    result = _run_formal_merge(output_root, checkpoint, steps=2)

    assert result.returncode != 0
    assert "episode mismatch: expected episode_idx=7 got=0" in result.stderr


def test_formal_merge_writes_joint_k_and_accepts_four_shard_plan(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output_root = tmp_path / "formal output"
    shard_ranges = ((0, 0, 3), (1, 3, 6), (2, 6, 8), (3, 8, 10))
    for shard, task_start, task_end in shard_ranges:
        for seed in range(50):
            records_dir = output_root / f"shard_{shard}" / f"seed_{seed}" / "records"
            records_dir.mkdir(parents=True, exist_ok=True)
            for task in range(task_start, task_end):
                (records_dir / f"task_{task}_episode_{seed}.json").write_text(
                    json.dumps(
                        {
                            "task_idx": task,
                            "episode_idx": seed,
                            "seed": seed,
                            "s4_checkpoint": str(checkpoint.resolve()),
                            "student_steps": 2,
                            "success": (task + seed) % 2 == 0,
                        }
                    ),
                    encoding="utf-8",
                )

    result = _run_formal_merge(
        output_root,
        checkpoint,
        steps=2,
        shard_plan="0:0:3;1:3:6;2:6:8;3:8:10",
    )

    _assert_success(result)
    summary = json.loads(
        (output_root / "formal_summary.json").read_text(encoding="utf-8")
    )
    assert summary["student_steps"] == 2
    assert summary["num_records"] == 500


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
    assert 'if "seed" not in record:' in source
    assert "record_seed != seed" in source


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
