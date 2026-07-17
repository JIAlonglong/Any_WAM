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

    assert "cfg.offline_eval_force_gradient_checkpointing = not args.disable_eval_gradient_checkpointing" in source
    assert "--disable-eval-gradient-checkpointing" in source
    assert "torch.enable_grad()" in source
    assert "student_action_v = student_action_v.detach()" in source


def test_video_eval_does_not_eager_load_vae_before_student_rollout():
    source = _video_eval_source()

    first_load_vae = source.index("load_vae(")
    student_rollout = source.index("trainer._student_euler_integrate(")

    assert first_load_vae > student_rollout


def test_video_eval_teacher_action_metrics_are_opt_in_and_use_teacher_endpoint():
    source = _video_eval_source()

    flag_index = source.index('"--emit-teacher-action-gt"')
    next_arg = source.index("args = parser.parse_args()", flag_index)
    flag_block = source[flag_index:next_arg]
    teacher_rollout = source.index("trainer._teacher_integrate_to_r(")
    teacher_action_query = source.index("trainer._teacher_forward_at_student_state(")

    assert 'action="store_true"' in flag_block
    assert teacher_action_query > teacher_rollout
    assert "teacher_x_r," in source[teacher_action_query:teacher_action_query + 240]
    assert 'action_input_dict=student_input["action_dict"]' in source
    assert "action_frames=action_frames" in source
    assert "if args.emit_teacher_action_gt:" in source
    assert "for name, value in teacher_action_metrics.items()" in source
    assert "action_teacher_gt_xr_mse" in source
