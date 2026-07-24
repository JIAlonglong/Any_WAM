import json
import os
import stat
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "evaluation"
    / "libero"
    / "run_cosmos_progressive_joint_124_eval_8gpu.sh"
)


def _base_env(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    dataset = tmp_path / "dataset"
    checkpoint.mkdir()
    dataset.mkdir()
    empty_embedding = dataset / "empty emb.pt"
    empty_embedding.write_bytes(b"test")
    env = os.environ.copy()
    env.update(
        {
            "MATRIX_ROOT": str(tmp_path / "matrix output"),
            "S4_CKPT_ROOT": str(checkpoint),
            "S4_DATASET_PATH": str(dataset),
            "S4_EMPTY_EMBEDDING": str(empty_embedding),
            "PYTHON_BIN": sys.executable,
        }
    )
    env.pop("S4_PROMPT_TABLE", None)
    env.pop("S4_VIDEO_SEEDS", None)
    env.pop("S4_FORMAL_NUM_SHARDS", None)
    env.pop("S4_STUDENT_STEPS", None)
    env.pop("S4_DRY_RUN", None)
    return env


def _run(mode, *, env):
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


def test_matrix_dry_run_plans_three_full_sequential_evaluations(tmp_path):
    env = _base_env(tmp_path)
    root = Path(env["MATRIX_ROOT"])

    result = _run("dry-run", env=env)
    _assert_success(result)

    assert [
        int(line.split("=", 1)[1])
        for line in result.stdout.splitlines()
        if line.startswith("MATRIX_STEP=")
    ] == [1, 2, 4]
    assert result.stdout.count("REQUESTED_RECORDS=500") == 3
    assert result.stdout.count("FORMAL_NUM_SHARDS=4") == 3
    assert result.stdout.count("VIDEO_SEEDS=0,1") == 3
    assert result.stdout.count("--student-steps 1") == 204
    assert result.stdout.count("--student-steps 2") == 204
    assert result.stdout.count("--student-steps 4") == 204
    assert not root.exists()


def test_matrix_rejects_existing_root_before_starting_a_child(tmp_path):
    env = _base_env(tmp_path)
    root = Path(env["MATRIX_ROOT"])
    root.mkdir()
    marker = tmp_path / "child-called"
    child = tmp_path / "child launcher.sh"
    child.write_text(
        "#!/usr/bin/env bash\n"
        f"touch {str(marker)!r}\n",
        encoding="utf-8",
    )
    child.chmod(child.stat().st_mode | stat.S_IXUSR)
    env["S4_FORMAL_LAUNCHER"] = str(child)

    result = _run("run", env=env)

    assert result.returncode == 2
    assert "MATRIX_ROOT already exists" in result.stderr
    assert not marker.exists()


def _write_child_sentinel(path, *, fail_step=None, corrupt_step=None):
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "assert sys.argv[1:] == ['formal']\n"
        "root = pathlib.Path(os.environ['EVAL_ROOT'])\n"
        "step = int(os.environ['S4_STUDENT_STEPS'])\n"
        "with pathlib.Path(os.environ['SENTINEL_LOG']).open('a', encoding='utf-8') as f:\n"
        "    f.write(f'{step}\\n')\n"
        + (
            f"if step == {fail_step}:\n"
            "    raise SystemExit(17)\n"
            if fail_step is not None
            else ""
        )
        + "root.mkdir(parents=True)\n"
        "prompt = os.environ.get('S4_PROMPT_TABLE', '')\n"
        "if not prompt:\n"
        "    (root / 'prompt_embeddings.pt').write_bytes(b'prompt')\n"
        "reported_step = step\n"
        + (
            f"if step == {corrupt_step}:\n"
            "    reported_step = 99\n"
            if corrupt_step is not None
            else ""
        )
        + "summary = {\n"
        "    'checkpoint': str(pathlib.Path(os.environ['S4_CKPT_ROOT']).resolve()),\n"
        "    'student_steps': reported_step,\n"
        "    'num_records': 500,\n"
        "}\n"
        "if step == 2 and os.environ.get('CORRUPTION') == 'records':\n"
        "    summary['num_records'] = 499\n"
        "if step == 2 and os.environ.get('CORRUPTION') == 'checkpoint':\n"
        "    summary['checkpoint'] = '/wrong/checkpoint'\n"
        "(root / 'formal_summary.json').write_text(json.dumps(summary), encoding='utf-8')\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_live_matrix_runs_serially_reuses_k1_prompt_and_writes_summary(tmp_path):
    env = _base_env(tmp_path)
    log = tmp_path / "child-order.log"
    child = tmp_path / "child.py"
    _write_child_sentinel(child)
    env.update(
        {
            "S4_FORMAL_LAUNCHER": str(child),
            "SENTINEL_LOG": str(log),
        }
    )

    result = _run("run", env=env)
    _assert_success(result)

    root = Path(env["MATRIX_ROOT"])
    assert log.read_text(encoding="utf-8").splitlines() == ["1", "2", "4"]
    assert (root / "k1" / "prompt_embeddings.pt").is_file()
    payload = json.loads((root / "matrix_summary.json").read_text(encoding="utf-8"))
    assert payload == {
        "schema": "cosmos_progressive_joint_124_matrix_v1",
        "checkpoint": str(Path(env["S4_CKPT_ROOT"]).resolve()),
        "steps": [1, 2, 4],
        "summaries": {
            "1": str(root / "k1" / "formal_summary.json"),
            "2": str(root / "k2" / "formal_summary.json"),
            "4": str(root / "k4" / "formal_summary.json"),
        },
    }


def test_child_failure_stops_before_later_k_and_publishes_no_summary(tmp_path):
    env = _base_env(tmp_path)
    log = tmp_path / "child-order.log"
    child = tmp_path / "failing-child.py"
    _write_child_sentinel(child, fail_step=2)
    env.update(
        {
            "S4_FORMAL_LAUNCHER": str(child),
            "SENTINEL_LOG": str(log),
        }
    )

    result = _run("run", env=env)

    assert result.returncode == 17
    assert log.read_text(encoding="utf-8").splitlines() == ["1", "2"]
    root = Path(env["MATRIX_ROOT"])
    assert not (root / "k4").exists()
    assert not (root / "matrix_summary.json").exists()


def test_invalid_child_summary_is_rejected_before_matrix_publish(tmp_path):
    env = _base_env(tmp_path)
    log = tmp_path / "child-order.log"
    child = tmp_path / "corrupt-child.py"
    _write_child_sentinel(child, corrupt_step=2)
    env.update(
        {
            "S4_FORMAL_LAUNCHER": str(child),
            "SENTINEL_LOG": str(log),
        }
    )

    result = _run("run", env=env)

    assert result.returncode != 0
    assert "student_steps mismatch" in result.stderr
    assert not (Path(env["MATRIX_ROOT"]) / "matrix_summary.json").exists()


def test_record_count_and_checkpoint_must_match_before_publish(tmp_path):
    for corruption, message in (
        ("records", "num_records mismatch"),
        ("checkpoint", "checkpoint mismatch"),
    ):
        case = tmp_path / corruption
        case.mkdir()
        env = _base_env(case)
        child = case / "corrupt-child.py"
        _write_child_sentinel(child)
        env.update(
            {
                "S4_FORMAL_LAUNCHER": str(child),
                "SENTINEL_LOG": str(case / "child-order.log"),
                "CORRUPTION": corruption,
            }
        )

        result = _run("run", env=env)

        assert result.returncode != 0
        assert message in result.stderr
        assert not (Path(env["MATRIX_ROOT"]) / "matrix_summary.json").exists()


def test_caller_prompt_table_is_reused_for_all_children(tmp_path):
    env = _base_env(tmp_path)
    prompt = tmp_path / "caller prompt.pt"
    prompt.write_bytes(b"prompt")
    log = tmp_path / "child-order.log"
    child = tmp_path / "child.py"
    _write_child_sentinel(child)
    env.update(
        {
            "S4_PROMPT_TABLE": str(prompt),
            "S4_FORMAL_LAUNCHER": str(child),
            "SENTINEL_LOG": str(log),
        }
    )

    result = _run("run", env=env)
    _assert_success(result)

    assert log.read_text(encoding="utf-8").splitlines() == ["1", "2", "4"]
    assert not (Path(env["MATRIX_ROOT"]) / "k1" / "prompt_embeddings.pt").exists()
