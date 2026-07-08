from pathlib import Path


def _video_eval_source():
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "rollout_eval_video_stage2.py").read_text(encoding="utf-8")


def test_video_eval_defaults_to_cpu_video_decode():
    source = _video_eval_source()

    arg_index = source.index('"--video-decode-device"')
    next_arg = source.index('"--cosmos-future-max-predictions"', arg_index)
    arg_block = source[arg_index:next_arg]

    assert 'default="cpu"' in arg_block


def test_video_eval_forces_checkpoint_safe_rollout_path():
    source = _video_eval_source()

    assert "cfg.offline_eval_force_gradient_checkpointing = True" in source
    assert "torch.enable_grad()" in source
    assert "student_action_v = student_action_v.detach()" in source


def test_video_eval_does_not_eager_load_vae_before_student_rollout():
    source = _video_eval_source()

    first_load_vae = source.index("load_vae(")
    student_rollout = source.index("trainer._student_euler_integrate(")

    assert first_load_vae > student_rollout
