import re
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


def test_robotwin_client_keeps_legacy_result_file_under_save_root():
    repo_root = Path(__file__).resolve().parents[2]
    source = (
        repo_root / "evaluation" / "robotwin" / "eval_polict_client_openpi.py"
    ).read_text(encoding="utf-8")

    assert re.search(
        r'Path\(args\["save_root"\]\)\s*/\s*"client_results"', source
    )
    assert 'Path(f"eval_result/' not in source


def test_robotwin_client_uses_loopback_endpoint_matching_runner_readiness():
    repo_root = Path(__file__).resolve().parents[2]
    source = (
        repo_root / "evaluation" / "robotwin" / "eval_polict_client_openpi.py"
    ).read_text(encoding="utf-8")

    assert 'WebsocketClientPolicy(host="127.0.0.1", port=usr_args["port"])' in source
