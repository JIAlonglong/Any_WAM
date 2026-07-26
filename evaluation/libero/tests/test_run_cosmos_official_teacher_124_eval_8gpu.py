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
    / "run_cosmos_official_teacher_124_eval_8gpu.sh"
)


def _write_sentinel(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "payload = {\n"
        "    'argv': sys.argv[1:],\n"
        "    'S4_MATRIX_ROLES': os.environ.get('S4_MATRIX_ROLES'),\n"
        "    'S4_ALIGNMENT_VERIFIED': os.environ.get('S4_ALIGNMENT_VERIFIED'),\n"
        "    'S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH': os.environ.get('S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH'),\n"
        "    'S4_FORMAL_NUM_SHARDS': os.environ.get('S4_FORMAL_NUM_SHARDS'),\n"
        "    'S4_EPISODES_PER_TASK': os.environ.get('S4_EPISODES_PER_TASK'),\n"
        "    'S4_CKPT_ROOT': os.environ.get('S4_CKPT_ROOT'),\n"
        "    'MATRIX_ROOT': os.environ.get('MATRIX_ROOT'),\n"
        "}\n"
        "pathlib.Path(os.environ['SENTINEL_OUTPUT']).write_text(\n"
        "    json.dumps(payload), encoding='utf-8'\n"
        ")\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _env(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    teacher = tmp_path / "official teacher"
    teacher.mkdir()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    empty = dataset / "empty emb.pt"
    empty.write_bytes(b"empty")
    prompts = dataset / "all 40 prompts.pt"
    prompts.write_bytes(b"prompts")
    lock = tmp_path / "teacher.lock.json"
    lock.write_text("{}", encoding="utf-8")
    sentinel = tmp_path / "matrix sentinel.py"
    _write_sentinel(sentinel)
    capture = tmp_path / "captured.json"
    matrix_root = tmp_path / "teacher matrix"

    env = os.environ.copy()
    env.update(
        {
            "MATRIX_ROOT": str(matrix_root),
            "COSMOS_POLICY_PATH": str(teacher),
            "COSMOS_POLICY_TEACHER_LOCK": str(lock),
            "S4_DATASET_PATH": str(dataset),
            "S4_EMPTY_EMBEDDING": str(empty),
            "S4_PROMPT_TABLE": str(prompts),
            "PYTHON_BIN": sys.executable,
            "COSMOS_TEACHER_MATRIX_LAUNCHER": str(sentinel),
            "SENTINEL_OUTPUT": str(capture),
            # The dedicated wrapper must override all three.
            "S4_MATRIX_ROLES": "stage2_target",
            "S4_ALIGNMENT_VERIFIED": "0",
            "S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH": "1",
        }
    )
    env.pop("S4_EPISODES_PER_TASK", None)
    env.pop("S4_FORMAL_NUM_SHARDS", None)
    return env, capture, matrix_root


def _run(mode: str, *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), mode],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_teacher_wrapper_forces_verified_teacher_only_eight_gpu_contract(tmp_path):
    env, capture, matrix_root = _env(tmp_path)

    result = _run("dry-run", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(capture.read_text(encoding="utf-8"))
    assert payload == {
        "argv": ["dry-run"],
        "S4_MATRIX_ROLES": "official_teacher",
        "S4_ALIGNMENT_VERIFIED": "1",
        "S4_ALLOW_KNOWN_ALIGNMENT_MISMATCH": "0",
        "S4_FORMAL_NUM_SHARDS": "4",
        "S4_EPISODES_PER_TASK": "50",
        "S4_CKPT_ROOT": env["COSMOS_POLICY_PATH"],
        "MATRIX_ROOT": str(matrix_root),
    }
    assert not matrix_root.exists()


def test_teacher_wrapper_preserves_episode_override(tmp_path):
    env, capture, _matrix_root = _env(tmp_path)
    env["S4_EPISODES_PER_TASK"] = "1"

    result = _run("dry-run", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(capture.read_text(encoding="utf-8"))
    assert payload["S4_EPISODES_PER_TASK"] == "1"
    assert payload["S4_MATRIX_ROLES"] == "official_teacher"


def test_teacher_wrapper_rejects_existing_root_before_delegation(tmp_path):
    env, capture, matrix_root = _env(tmp_path)
    matrix_root.mkdir()

    result = _run("run", env=env)

    assert result.returncode == 2
    assert "MATRIX_ROOT already exists" in result.stderr
    assert not capture.exists()


def test_teacher_wrapper_rejects_unknown_mode(tmp_path):
    env, capture, _matrix_root = _env(tmp_path)

    result = _run("smoke", env=env)

    assert result.returncode == 2
    assert "run or dry-run" in result.stderr
    assert not capture.exists()
