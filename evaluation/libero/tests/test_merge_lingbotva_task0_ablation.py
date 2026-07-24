import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "evaluation" / "libero" / "merge_lingbotva_task0_ablation.py"
MODELS = ("stage1", "stage1_only", "anchor_only", "field_only", "apm")


def _write_family(root: Path, episodes: int = 20):
    for model_idx, model in enumerate(MODELS):
        for steps in (1, 2, 4):
            result_dir = root / model / f"steps_{steps}" / "results" / "libero_eval"
            result_dir.mkdir(parents=True)
            successes = min(episodes, model_idx + steps)
            (result_dir / "libero_10_0.json").write_text(
                json.dumps(
                    {
                        "succ_num": float(successes),
                        "total_num": float(episodes),
                        "succ_rate": successes / episodes,
                    }
                )
            )


def test_merges_15_jobs_and_computes_matched_stage1_deltas(tmp_path):
    _write_family(tmp_path)
    output = tmp_path / "summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-root",
            str(tmp_path),
            "--expected-episodes",
            "20",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(output.read_text())
    assert summary["benchmark"] == "libero_10"
    assert summary["task_idx"] == 0
    assert len(summary["rows"]) == 15
    assert summary["models"]["anchor_only"]["steps"]["2"]["delta_vs_stage1"] == 2 / 20
    assert summary["models"]["stage1"]["steps"]["4"]["delta_vs_stage1"] == 0.0


def test_rejects_missing_budget(tmp_path):
    _write_family(tmp_path)
    (
        tmp_path
        / "apm"
        / "steps_4"
        / "results"
        / "libero_eval"
        / "libero_10_0.json"
    ).unlink()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-root",
            str(tmp_path),
            "--expected-episodes",
            "20",
            "--output",
            str(tmp_path / "summary.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Missing result" in result.stderr
