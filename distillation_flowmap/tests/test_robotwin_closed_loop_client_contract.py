from pathlib import Path


def test_robotwin_client_supports_portable_closed_loop_metrics():
    repo_root = Path(__file__).resolve().parents[2]
    source = (
        repo_root / "evaluation" / "robotwin" / "eval_polict_client_openpi.py"
    ).read_text(encoding="utf-8")

    assert 'os.environ.get("ROBOTWIN_ROOT"' in source
    assert "build_closed_loop_task_metrics" in source
    assert "closed_loop_metrics.json" in source
    assert "--no-save-visualization" in source
    assert 'config["eval_video_log"] = False' in source
