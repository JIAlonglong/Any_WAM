import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "evaluation" / "libero" / "merge_lingbotva_4suite_results.py"
SUITES = ("libero_10", "libero_spatial", "libero_object", "libero_goal")


def _write_results(root: Path, total: int = 10):
    for suite_idx, suite in enumerate(SUITES):
        worker = root / "workers" / f"{suite}_0_10" / "results" / "libero_eval"
        worker.mkdir(parents=True)
        for task_idx in range(10):
            succ = (suite_idx + task_idx) % (total + 1)
            (worker / f"{suite}_{task_idx}.json").write_text(
                json.dumps(
                    {
                        "succ_num": float(succ),
                        "total_num": float(total),
                        "succ_rate": succ / total,
                    }
                )
            )


def test_merges_all_40_tasks_and_validates_episode_budget(tmp_path):
    _write_results(tmp_path, total=10)
    output = tmp_path / "summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-root",
            str(tmp_path),
            "--expected-episodes",
            "10",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(output.read_text())
    assert summary["num_tasks"] == 40
    assert summary["total_episodes"] == 400
    assert set(summary["suites"]) == set(SUITES)
    assert all(v["num_tasks"] == 10 for v in summary["suites"].values())


def test_rejects_missing_task(tmp_path):
    _write_results(tmp_path, total=10)
    next(tmp_path.rglob("libero_goal_9.json")).unlink()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-root",
            str(tmp_path),
            "--expected-episodes",
            "10",
            "--output",
            str(tmp_path / "summary.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Missing result" in result.stderr
