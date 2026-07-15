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
    assert "torch.enable_grad()" in source
    assert "student_action_v = student_action_v.detach()" in source


def test_video_eval_does_not_eager_load_vae_before_student_rollout():
    source = _video_eval_source()

    first_load_vae = source.index("load_vae(")
    student_rollout = source.index("trainer._student_euler_integrate(")

    assert first_load_vae > student_rollout


def test_video_eval_does_not_limit_assets_to_first_eval_record():
    source = _video_eval_source()

    assert "and batch_idx == 0" not in source


def test_video_eval_names_assets_with_the_record_identity():
    source = _video_eval_source()

    assert 'record.get("sample_key"' in source
    assert 'f"{asset_prefix}_{name}.mp4"' in source
    assert 'f"{asset_prefix}_contact_sheet.png"' in source


def test_legacy_video_eval_wrapper_is_not_a_stable_entrypoint():
    repo_root = Path(__file__).resolve().parents[1]

    assert not (repo_root / "run_eval_video_stage2.sh").exists()


def test_video_eval_preserves_unmanifested_start_index_without_eager_dataset_cache():
    source = _video_eval_source()

    assert 'parser.add_argument("--start-index", type=int, default=0)' in source
    assert (
        "cfg.light_eval_start_index = (\n"
        "        args.start_index if args.eval_manifest is None else 0\n"
        "    )"
    ) in source
    assert source.index("cfg.cache_dataset_in_memory = False") < source.index(
        "is_cosmos_policy_teacher_cfg ="
    )
