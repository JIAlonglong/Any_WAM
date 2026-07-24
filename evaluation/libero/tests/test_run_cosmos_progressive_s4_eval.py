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
            "S4_EVAL_CLASSIFICATION": "formal_verified",
            "S4_EVAL_IS_FORMAL": "1",
            "S4_ALIGNMENT_VERIFIED": "1",
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

    rollout_commands = []
    for line in result.stdout.splitlines():
        if not line.startswith("COMMAND="):
            continue
        tokens = shlex.split(line.partition("=")[2])
        if "--env-seed" in tokens:
            rollout_commands.append(tokens)
    assert len(rollout_commands) == 200
    for tokens in rollout_commands:
        seed = int(tokens[tokens.index("--env-seed") + 1])
        assert ("--save-video" in tokens) is (seed in {0, 1})


def test_formal_joint_k1_reaches_every_preflight_and_rollout(tmp_path):
    env = _launcher_env(tmp_path)
    env["S4_STUDENT_STEPS"] = "1"

    result = run_launcher("formal", env=env)
    _assert_success(result)

    command_lines = [
        shlex.split(line.partition("=")[2])
        for line in result.stdout.splitlines()
        if line.startswith("COMMAND=")
        and "evaluation.libero.rollout_cosmos_progressive_s4" in line
    ]
    assert len(command_lines) == 102
    assert all(
        tokens[tokens.index("--student-steps") + 1] == "1"
        for tokens in command_lines
    )


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("S4_STUDENT_STEPS", "3", "S4_STUDENT_STEPS must be 1, 2, or 4"),
        ("S4_FORMAL_NUM_SHARDS", "3", "S4_FORMAL_NUM_SHARDS must be 2 or 4"),
        ("S4_VIDEO_SEEDS", "0,nope", "S4_VIDEO_SEEDS"),
        ("S4_VIDEO_SEEDS", "0,", "S4_VIDEO_SEEDS"),
        ("S4_VIDEO_SEEDS", "01", "S4_VIDEO_SEEDS"),
        ("S4_VIDEO_SEEDS", "08", "S4_VIDEO_SEEDS"),
        ("S4_VIDEO_SEEDS", "09", "S4_VIDEO_SEEDS"),
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
        '        "${S4_EVAL_IS_FORMAL}" <<\'PY\'\n'
    )
    end_marker = "\nPY\n}\n\n\nprepare_prompt_table"
    start = source.index(start_marker) + len(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def _run_formal_merge(
    root,
    checkpoint,
    *,
    steps=2,
    shard_plan="0:0:5;1:5:10",
    classification="formal_verified",
    is_formal="1",
):
    return subprocess.run(
        [
            sys.executable,
            "-",
            str(root),
            str(checkpoint),
            "50",
            str(steps),
            shard_plan,
            classification,
            is_formal,
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
    assert summary["evaluation_classification"] == "formal_verified"
    assert summary["is_formal"] is True
    assert not (output_root / ".formal_summary.json.tmp").exists()


def test_formal_merge_counts_real_failure_setup_and_skip_records_as_unsuccessful(
    tmp_path, monkeypatch
):
    import numpy as np

    from evaluation.libero.cosmos_progressive_s4_client import (
        CosmosProgressiveS4Client,
    )
    from evaluation.libero.tests.test_cosmos_progressive_s4_service import (
        OBS,
        _FailingService,
        _SuccessfulService,
        _WarmupEnv,
        _install_fake_libero,
    )

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    checkpoint_id = str(checkpoint.resolve())
    scratch = tmp_path / "constructed records"

    failure_client = CosmosProgressiveS4Client(
        _FailingService(),
        output_dir=scratch / "failure",
        student_steps=2,
        expected_s4_checkpoint=checkpoint_id,
        warmup_steps=0,
    )
    failure = failure_client.run_with_env(
        env=_WarmupEnv(),
        initial_state=np.zeros(1, dtype=np.float32),
        task_idx=0,
        episode_idx=0,
        prompt="open the drawer",
        max_env_steps=1,
        rollout_seed=0,
    )

    setup_client = CosmosProgressiveS4Client(
        _SuccessfulService(),
        output_dir=scratch / "setup",
        student_steps=2,
        expected_s4_checkpoint=checkpoint_id,
    )
    setup = setup_client.run_with_env(
        env=_WarmupEnv(),
        initial_state=np.zeros(1, dtype=np.float32),
        task_idx=0,
        episode_idx=1,
        prompt="open the drawer",
        max_env_steps=1,
        rollout_seed=1,
        init_env_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("setup failed")
        ),
    )

    _install_fake_libero(monkeypatch, skipped=True)
    skipped_client = CosmosProgressiveS4Client(
        _SuccessfulService(),
        output_dir=scratch / "skip",
        student_steps=2,
        expected_s4_checkpoint=checkpoint_id,
    )
    skipped = skipped_client.run_libero_task(
        libero_benchmark="libero_10",
        task_idx=0,
        episode_idx=2,
        camera_size=128,
        max_env_steps=1,
        env_seed=2,
    )

    output_root = tmp_path / "formal output"
    constructed = {(0, 0): failure, (0, 1): setup, (0, 2): skipped}
    for task in range(10):
        shard = 0 if task < 5 else 1
        for seed in range(50):
            record = constructed.get(
                (task, seed),
                {
                    "task_idx": task,
                    "episode_idx": seed,
                    "seed": seed,
                    "s4_checkpoint": checkpoint_id,
                    "student_steps": 2,
                    "success": True,
                },
            )
            record_path = (
                output_root
                / f"shard_{shard}"
                / f"seed_{seed}"
                / "records"
                / f"task_{task}_episode_{seed}.json"
            )
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(record), encoding="utf-8")

    result = _run_formal_merge(output_root, checkpoint)

    _assert_success(result)
    summary = json.loads(
        (output_root / "formal_summary.json").read_text(encoding="utf-8")
    )
    assert summary["num_records"] == 500
    assert summary["per_task_success"]["0"] == 47 / 50


def test_formal_merge_rejects_real_checkpoint_mismatch_record(tmp_path):
    import numpy as np

    from evaluation.libero.cosmos_progressive_s4_client import (
        CosmosProgressiveS4Client,
    )
    from evaluation.libero.tests.test_cosmos_progressive_s4_service import (
        OBS,
        _DoneEnv,
    )

    checkpoint = tmp_path / "expected checkpoint"
    checkpoint.mkdir()
    checkpoint_id = str(checkpoint.resolve())

    class MismatchService:
        checkpoint_identifier = checkpoint_id

        def infer(self, _request):
            return {
                "ok": True,
                "action": np.zeros((16, 7), dtype=np.float32),
                "student_steps": 2,
                "s4_checkpoint": "/observed/wrong-checkpoint",
            }

    output_root = tmp_path / "formal output"
    client = CosmosProgressiveS4Client(
        MismatchService(),
        output_dir=output_root / "shard_0" / "seed_0",
        student_steps=2,
        expected_s4_checkpoint=checkpoint_id,
    )
    record = client.run_with_env(
        env=_DoneEnv(),
        initial_state=np.zeros(1, dtype=np.float32),
        task_idx=0,
        episode_idx=0,
        prompt="open the drawer",
        max_env_steps=1,
        init_env_fn=lambda *_args, **_kwargs: OBS,
        rollout_seed=0,
    )

    assert record["s4_checkpoint"] == "/observed/wrong-checkpoint"
    assert record["expected_s4_checkpoint"] == checkpoint_id
    result = _run_formal_merge(output_root, checkpoint)
    assert result.returncode != 0
    assert "checkpoint mismatch" in result.stderr


@pytest.mark.parametrize(
    ("classification", "is_formal", "missing_control"),
    [
        ("formal_verified", "1", "S4_ALIGNMENT_VERIFIED=1"),
        (
            "diagnostic_known_alignment_mismatch",
            "0",
            "S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH=1",
        ),
        ("blocked_known_alignment_mismatch", "0", "blocked"),
    ],
)
def test_live_formal_launcher_rejects_classification_gate_bypass(
    tmp_path, classification, is_formal, missing_control
):
    env = _launcher_env(tmp_path)
    env.update(
        {
            "S4_DRY_RUN": "0",
            "S4_EVAL_CLASSIFICATION": classification,
            "S4_EVAL_IS_FORMAL": is_formal,
        }
    )
    env.pop("S4_ALIGNMENT_VERIFIED", None)
    env.pop("S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH", None)
    marker = tmp_path / "child-called"
    sentinel = tmp_path / "python sentinel"
    sentinel.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('called', encoding='utf-8')\n",
        encoding="utf-8",
    )
    sentinel.chmod(sentinel.stat().st_mode | stat.S_IXUSR)
    env["PYTHON_BIN"] = str(sentinel)

    result = run_launcher("formal", env=env)

    assert result.returncode == 2
    assert missing_control in result.stderr
    assert not marker.exists()


def test_formal_child_summary_is_published_atomically():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'temporary_path = root / ".formal_summary.json.tmp"' in source
    assert "os.replace(temporary_path, path)" in source


def test_formal_waits_for_every_shard_and_skips_merge_when_one_shard_fails(tmp_path):
    env = _launcher_env(tmp_path)
    env.update(
        {
            "S4_DRY_RUN": "0",
            "S4_FORMAL_NUM_SHARDS": "4",
            "S4_STUDENT_STEPS": "2",
        }
    )
    log = tmp_path / "sentinel.log"
    sentinel = tmp_path / "multi shard sentinel.py"
    sentinel.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        f"log = pathlib.Path({str(log)!r})\n"
        "with log.open('a', encoding='utf-8') as stream:\n"
        "    if '--preflight' in sys.argv:\n"
        "        stream.write(f'preflight:{os.environ.get(\"CUDA_VISIBLE_DEVICES\")}\\n')\n"
        "        raise SystemExit(0)\n"
        "    if sys.argv[1:2] == ['-']:\n"
        "        stream.write('MERGE\\n')\n"
        "        raise SystemExit(0)\n"
        "    seed = sys.argv[sys.argv.index('--env-seed') + 1]\n"
        "    gpu = os.environ['CUDA_VISIBLE_DEVICES']\n"
        "    stream.write(f'rollout:{gpu}:{seed}\\n')\n"
        "if gpu == '0' and seed == '0':\n"
        "    raise SystemExit(17)\n",
        encoding="utf-8",
    )
    sentinel.chmod(sentinel.stat().st_mode | stat.S_IXUSR)
    env["PYTHON_BIN"] = str(sentinel)

    result = run_launcher("formal", env=env)

    assert result.returncode != 0
    lines = log.read_text(encoding="utf-8").splitlines()
    assert "rollout:2:49" in lines
    assert "rollout:4:49" in lines
    assert "rollout:6:49" in lines
    assert "MERGE" not in lines


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
