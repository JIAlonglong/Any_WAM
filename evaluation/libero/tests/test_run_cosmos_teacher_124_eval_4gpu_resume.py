import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "evaluation"
    / "libero"
    / "run_cosmos_teacher_124_eval_4gpu_resume.sh"
)
SUITES = ("libero_10", "libero_spatial", "libero_object", "libero_goal")


def _write_fake_launcher(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import cosmos_cuda\n"
        "import json, os, pathlib, sys\n"
        "capture = pathlib.Path(os.environ['CAPTURE_PATH'])\n"
        "with capture.open('a', encoding='utf-8') as handle:\n"
        "    handle.write(json.dumps({\n"
        "        'mode': sys.argv[1],\n"
        "        'suite': os.environ['S4_LIBERO_BENCHMARK'],\n"
        "        'steps': int(os.environ['S4_STUDENT_STEPS']),\n"
        "        'eval_root': os.environ['EVAL_ROOT'],\n"
        "        'shards': int(os.environ['S4_FORMAL_NUM_SHARDS']),\n"
        "        'layout': os.environ['S4_FORMAL_GPU_LAYOUT'],\n"
        "        'video_seeds': os.environ['S4_VIDEO_SEEDS'],\n"
        "        'allocator': os.environ['PYTORCH_CUDA_ALLOC_CONF'],\n"
        "    }, sort_keys=True) + '\\n')\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _env(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path]:
    cosmos_repo = tmp_path / "cosmos repo"
    cosmos_cuda = (
        cosmos_repo / "packages" / "cosmos-cuda" / "cosmos_cuda"
    )
    cosmos_cuda.mkdir(parents=True)
    (cosmos_cuda / "__init__.py").write_text(
        "__version__ = 'test'\n", encoding="utf-8"
    )
    (cosmos_repo / "packages" / "cosmos-oss").mkdir(parents=True)
    teacher = tmp_path / "official teacher"
    teacher.mkdir()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    empty = dataset / "empty.pt"
    empty.write_bytes(b"empty")
    prompts = dataset / "all40.pt"
    prompts.write_bytes(b"prompts")
    lock = tmp_path / "teacher.lock.json"
    lock.write_text("{}", encoding="utf-8")
    launcher = tmp_path / "fake formal launcher.py"
    _write_fake_launcher(launcher)
    capture = tmp_path / "calls.jsonl"
    matrix = tmp_path / "existing matrix"

    env = os.environ.copy()
    env.update(
        {
            "COSMOS_POLICY_PATH": str(teacher),
            "COSMOS_POLICY_TEACHER_LOCK": str(lock),
            "COSMOS_PREDICT2_REPO": str(cosmos_repo),
            "COSMOS_POLICY_PYTHON": sys.executable,
            "S4_DATASET_PATH": str(dataset),
            "S4_EMPTY_EMBEDDING": str(empty),
            "S4_PROMPT_TABLE": str(prompts),
            "PYTHON_BIN": sys.executable,
            "COSMOS_TEACHER_FORMAL_LAUNCHER": str(launcher),
            "CAPTURE_PATH": str(capture),
            "S4_DRY_RUN": "1",
            "S4_EPISODES_PER_TASK": "50",
        }
    )
    return env, matrix, teacher, capture


def _write_complete_summary(
    matrix: Path,
    teacher: Path,
    *,
    suite: str = "libero_10",
    steps: int = 1,
    num_records: int = 500,
) -> Path:
    root = matrix / f"k{steps}" / suite
    root.mkdir(parents=True)
    summary = root / "formal_summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": "cosmos_progressive_s4_formal_eval_v2",
                "model_role": "official_teacher",
                "libero_benchmark": suite,
                "video_steps": steps,
                "action_steps": steps,
                "num_tasks": 10,
                "num_records": num_records,
                "seeds_per_task": 50,
                "is_formal": True,
                "evaluation_classification": "formal_verified",
                "checkpoint": str(teacher.resolve()),
                "checkpoint_contract_identity": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    return summary


def _run(env: dict[str, str], matrix: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), str(matrix)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _calls(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_resume_skips_complete_k1_libero10_and_plans_remaining_eleven(tmp_path):
    env, matrix, teacher, capture = _env(tmp_path)
    _write_complete_summary(matrix, teacher)

    result = _run(env, matrix)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "SKIP_COMPLETED=libero_10,K=1,records=500" in result.stdout
    calls = _calls(capture)
    assert [(call["steps"], call["suite"]) for call in calls] == [
        (steps, suite)
        for steps in (1, 2, 4)
        for suite in SUITES
        if (steps, suite) != (1, "libero_10")
    ]
    assert all(call["mode"] == "dry-run" for call in calls)
    assert all(call["shards"] == 4 for call in calls)
    assert all(call["layout"] == "colocated" for call in calls)
    assert all(call["video_seeds"] == "0" for call in calls)
    assert all(
        call["allocator"] == "max_split_size_mb:128" for call in calls
    )


def test_resume_rejects_existing_cell_without_complete_summary(tmp_path):
    env, matrix, _teacher, capture = _env(tmp_path)
    (matrix / "k1" / "libero_10").mkdir(parents=True)

    result = _run(env, matrix)

    assert result.returncode != 0
    assert "incomplete existing cell" in result.stderr
    assert _calls(capture) == []


@pytest.mark.parametrize(
    ("steps", "num_records", "message"),
    [(2, 500, "video/action step mismatch"), (1, 499, "record count mismatch")],
)
def test_resume_rejects_invalid_completed_summary(
    tmp_path, steps, num_records, message
):
    env, matrix, teacher, capture = _env(tmp_path)
    path = _write_complete_summary(matrix, teacher, num_records=num_records)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["video_steps"] = steps
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run(env, matrix)

    assert result.returncode != 0
    assert message in result.stderr
    assert _calls(capture) == []


def test_resume_dry_run_plans_twelve_cells_without_creating_matrix(tmp_path):
    env, matrix, _teacher, capture = _env(tmp_path)

    result = _run(env, matrix)

    assert result.returncode == 0, result.stdout + result.stderr
    assert len(_calls(capture)) == 12
    assert not matrix.exists()
    assert f"MATRIX_SUMMARY_PLAN={matrix / 'matrix_summary.json'}" in result.stdout
    assert f"MATRIX_SUMMARY_CSV_PLAN={matrix / 'matrix_summary.csv'}" in result.stdout


@pytest.mark.parametrize("steps", (1, 2, 4))
def test_resume_can_plan_only_one_requested_k_without_matrix_merge(tmp_path, steps):
    env, matrix, _teacher, capture = _env(tmp_path)
    env["COSMOS_TEACHER_STEPS"] = str(steps)

    result = _run(env, matrix)

    assert result.returncode == 0, result.stdout + result.stderr
    assert [(call["steps"], call["suite"]) for call in _calls(capture)] == [
        (steps, suite) for suite in SUITES
    ]
    assert "MATRIX_SUMMARY_PLAN=" not in result.stdout
    assert not matrix.exists()
