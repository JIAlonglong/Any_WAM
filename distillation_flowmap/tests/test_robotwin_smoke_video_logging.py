from pathlib import Path


def _run_smoke_source():
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "ablation" / "run_smoke.sh").read_text(encoding="utf-8")


def test_protocol_smoke_passes_video_dir_to_video_eval():
    source = _run_smoke_source()
    start = source.index("distillation_flowmap/rollout_eval_video_stage2.py")
    end = source.index('if [ "${RUN_ROBOTWIN_ENV_SMOKE:-0}"', start)
    video_eval_block = source[start:end]

    assert '--video-dir "${RUN_DIR}/videos"' in video_eval_block
    assert "--video-decode-device cpu" in video_eval_block
