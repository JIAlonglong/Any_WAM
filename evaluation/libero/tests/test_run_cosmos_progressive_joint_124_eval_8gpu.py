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
            "S4_ALIGNMENT_VERIFIED": "1",
        }
    )
    env.pop("S4_PROMPT_TABLE", None)
    env.pop("S4_VIDEO_SEEDS", None)
    env.pop("S4_FORMAL_NUM_SHARDS", None)
    env.pop("S4_STUDENT_STEPS", None)
    env.pop("S4_MODEL_ROLE", None)
    env.pop("S4_DRY_RUN", None)
    env.pop("S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH", None)
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
    for step in (1, 2, 4):
        assert result.stdout.count(f"--video-steps {step}") == 204
        assert result.stdout.count(f"--action-steps {step}") == 204
    assert "--student-steps" not in result.stdout
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
        "    'video_steps': step,\n"
        "    'action_steps': step,\n"
        "    'model_role': os.environ.get('S4_MODEL_ROLE', 'stage2_target'),\n"
        "    'checkpoint_contract_identity': 'contract-v1',\n"
        "    'num_records': 500,\n"
        "    'evaluation_classification': os.environ['S4_EVAL_CLASSIFICATION'],\n"
        "    'is_formal': os.environ['S4_EVAL_IS_FORMAL'] == '1',\n"
        "}\n"
        "if step == 2 and os.environ.get('CORRUPTION') == 'records':\n"
        "    summary['num_records'] = 499\n"
        "if step == 2 and os.environ.get('CORRUPTION') == 'checkpoint':\n"
        "    summary['checkpoint'] = '/wrong/checkpoint'\n"
        "if step == 2 and os.environ.get('CORRUPTION') == 'classification':\n"
        "    summary['evaluation_classification'] = 'wrong'\n"
        "if step == 2 and os.environ.get('CORRUPTION') == 'model_role':\n"
        "    summary['model_role'] = 'wrong-role'\n"
        "if step == 2 and os.environ.get('CORRUPTION') == 'video_steps':\n"
        "    summary['video_steps'] = 99\n"
        "if step == 2 and os.environ.get('CORRUPTION') == 'action_steps':\n"
        "    summary['action_steps'] = 99\n"
        "if step == 2 and os.environ.get('CORRUPTION') == 'identity':\n"
        "    summary['checkpoint_contract_identity'] = 'different-contract'\n"
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
    for step in (1, 2, 4):
        child_payload = json.loads(
            (root / f"k{step}" / "formal_summary.json").read_text(encoding="utf-8")
        )
        assert child_payload["student_steps"] == step
        assert child_payload["video_steps"] == step
        assert child_payload["action_steps"] == step
        assert child_payload["model_role"] == "stage2_target"
        assert child_payload["checkpoint_contract_identity"] == "contract-v1"
    payload = json.loads((root / "matrix_summary.json").read_text(encoding="utf-8"))
    assert payload == {
        "schema": "cosmos_progressive_joint_124_matrix_v1",
        "checkpoint": str(Path(env["S4_CKPT_ROOT"]).resolve()),
        "evaluation_classification": "formal_verified",
        "is_formal": True,
        "steps": [1, 2, 4],
        "model_role": "stage2_target",
        "checkpoint_contract_identity": "contract-v1",
        "summaries": {
            "1": str(root / "k1" / "formal_summary.json"),
            "2": str(root / "k2" / "formal_summary.json"),
            "4": str(root / "k4" / "formal_summary.json"),
        },
    }


def test_unverified_run_stops_before_any_child(tmp_path):
    env = _base_env(tmp_path)
    env.pop("S4_ALIGNMENT_VERIFIED")
    marker = tmp_path / "child-called"
    child = tmp_path / "child.sh"
    child.write_text(
        "#!/usr/bin/env bash\n"
        f"touch {str(marker)!r}\n",
        encoding="utf-8",
    )
    child.chmod(child.stat().st_mode | stat.S_IXUSR)
    env["S4_FORMAL_LAUNCHER"] = str(child)

    result = _run("run", env=env)

    assert result.returncode == 2
    assert "S4_ALIGNMENT_VERIFIED=1" in result.stderr
    assert "S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH=1" in result.stderr
    assert not marker.exists()
    assert not Path(env["MATRIX_ROOT"]).exists()


def test_unverified_dry_run_is_planned_as_blocked_and_creates_nothing(tmp_path):
    env = _base_env(tmp_path)
    env.pop("S4_ALIGNMENT_VERIFIED")

    result = _run("dry-run", env=env)
    _assert_success(result)

    assert "ALIGNMENT_BLOCKED=1" in result.stdout
    assert "EVALUATION_CLASSIFICATION=blocked_known_alignment_mismatch" in result.stdout
    assert "EVALUATION_IS_FORMAL=0" in result.stdout
    assert not Path(env["MATRIX_ROOT"]).exists()


def test_explicit_known_mismatch_override_marks_every_summary_nonformal(tmp_path):
    env = _base_env(tmp_path)
    env.pop("S4_ALIGNMENT_VERIFIED")
    env["S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH"] = "1"
    child = tmp_path / "diagnostic-child.py"
    log = tmp_path / "child.log"
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
    for step in (1, 2, 4):
        child_summary = json.loads(
            (root / f"k{step}" / "formal_summary.json").read_text(encoding="utf-8")
        )
        assert child_summary["evaluation_classification"] == (
            "diagnostic_known_alignment_mismatch"
        )
        assert child_summary["is_formal"] is False
    matrix = json.loads((root / "matrix_summary.json").read_text(encoding="utf-8"))
    assert matrix["evaluation_classification"] == (
        "diagnostic_known_alignment_mismatch"
    )
    assert matrix["is_formal"] is False


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


def test_child_classification_must_match_before_matrix_publish(tmp_path):
    env = _base_env(tmp_path)
    child = tmp_path / "corrupt-child.py"
    _write_child_sentinel(child)
    env.update(
        {
            "S4_FORMAL_LAUNCHER": str(child),
            "SENTINEL_LOG": str(tmp_path / "child.log"),
            "CORRUPTION": "classification",
        }
    )

    result = _run("run", env=env)

    assert result.returncode != 0
    assert "evaluation classification mismatch" in result.stderr
    assert not (Path(env["MATRIX_ROOT"]) / "matrix_summary.json").exists()


def test_child_contract_fields_must_match_before_matrix_publish(tmp_path):
    for corruption, message in (
        ("model_role", "model_role mismatch"),
        ("video_steps", "video/action step mismatch"),
        ("action_steps", "video/action step mismatch"),
        ("identity", "checkpoint_contract_identity mismatch across K"),
    ):
        case = tmp_path / corruption
        case.mkdir()
        env = _base_env(case)
        child = case / "corrupt-child.py"
        _write_child_sentinel(child)
        env.update(
            {
                "S4_FORMAL_LAUNCHER": str(child),
                "SENTINEL_LOG": str(case / "child.log"),
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
