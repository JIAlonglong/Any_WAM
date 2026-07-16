from pathlib import Path

import pytest

from distillation_flowmap.run_cosmos_mixed_step_policy import (
    DEFAULT_STAGE1_CHECKPOINT,
    LEGACY_PROGRESSIVE_ROOT,
    build_multibudget_eval_plans,
    build_policy_manifest_payload,
    build_policy_train_plan,
    build_torchrun_command,
    parse_device_list,
    validate_policy_root,
)


DEVICES = "0,1,2,3,4,5,6,7"


def _plan(tmp_path, **overrides):
    root = tmp_path / "policy"
    defaults = {
        "policy_name": "universe",
        "root": root,
        "dataset_path": Path("/tmp/libero"),
        "protocol_source_root": tmp_path / "shared_protocol",
        "stage1_checkpoint": Path("/tmp/stage1_step5000"),
        "current_step": 0,
        "chunk_size": 250,
        "max_train_steps": 5000,
        "save_interval": 250,
        "master_port": 29761,
        "train_seed": 20260716,
        "torchrun": Path("/tmp/torchrun"),
        "teacher_model_path": Path("/tmp/cosmos_policy"),
        "device_list": DEVICES,
        "world_size": 8,
    }
    defaults.update(overrides)
    return build_policy_train_plan(**defaults)


@pytest.mark.parametrize(
    ("policy_name", "expected_weights"),
    [
        ("universe", [0.50, 0.30, 0.20]),
        ("s2", [0.20, 0.60, 0.20]),
        ("s1", [0.70, 0.20, 0.10]),
    ],
)
def test_initial_independent_policy_plans_start_from_common_online_student(
    tmp_path, policy_name, expected_weights
):
    root = tmp_path / policy_name
    plan = _plan(tmp_path, policy_name=policy_name, root=root)

    assert plan["policy"]["name"] == policy_name
    assert plan["policy"]["rollout_step_pairs"] == [[4, 1], [4, 2], [8, 4]]
    assert plan["policy"]["weights"] == pytest.approx(expected_weights)
    assert plan["output_dir"] == root
    assert plan["checkpoint_dir"] == root / "checkpoints" / "step_250"
    assert plan["source_checkpoint"] == Path("/tmp/stage1_step5000")
    assert plan["max_train_steps"] == 5000
    assert plan["save_interval"] == 250
    assert plan["train_env"]["RESUME_FROM_PATH"] == "/tmp/stage1_step5000"
    assert plan["train_env"]["RESUME_ONLINE_FROM_TARGET"] == "0"
    assert plan["train_env"]["RESET_RESUME_STEP"] == "1"
    assert plan["train_env"]["RESUME_OPTIMIZER_STATE"] == "0"
    assert plan["train_env"]["SKIP_TARGET_STUDENT_FOR_COSMOS_LATENT"] == "1"
    assert plan["train_env"]["COSMOS_PROGRESSIVE_STAGE"] == "s4"
    assert plan["train_env"]["COSMOS_MIXED_STEP_POLICY"] == policy_name
    assert plan["train_env"]["MAX_TRAIN_STEPS"] == "5000"
    assert plan["train_env"]["SAVE_INTERVAL"] == "250"
    assert plan["train_env"]["CUDA_VISIBLE_DEVICES"] == DEVICES
    assert plan["train_env"]["COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES"] == DEVICES
    assert "--nproc_per_node=8" in plan["train_argv"]
    assert "--master_port=29761" in plan["train_argv"]
    assert plan["train_argv"][plan["train_argv"].index("--resume-from-path") + 1] == "/tmp/stage1_step5000"
    assert plan["train_argv"][plan["train_argv"].index("--gradient-accumulation-steps") + 1] == "1"
    assert all(eval_plan["env"]["COSMOS_PROGRESSIVE_STAGE"] == "s4" for eval_plan in plan["eval_plans"])
    assert all(eval_plan["env"]["COSMOS_MIXED_STEP_POLICY"] == policy_name for eval_plan in plan["eval_plans"])


def test_default_stage1_source_is_the_agreed_common_online_student_checkpoint():
    assert str(DEFAULT_STAGE1_CHECKPOINT).endswith(
        "output_libero_cosmos_policy_stage1_cosmos_latent_cdiff_8gpu_20260706_"
        "cosmos_latent_s1s2_8gpu/checkpoints/step_5000"
    )


def test_resume_plan_uses_only_its_own_online_checkpoint_and_optimizer_state(tmp_path):
    root = tmp_path / "universe"
    plan = _plan(tmp_path, root=root, current_step=250, chunk_size=250)

    assert plan["checkpoint_dir"] == root / "checkpoints" / "step_500"
    assert plan["train_env"]["RESUME_FROM_PATH"] == str(root / "checkpoints" / "step_250")
    assert plan["train_env"]["RESUME_ONLINE_FROM_TARGET"] == "0"
    assert plan["train_env"]["RESET_RESUME_STEP"] == "0"
    assert plan["train_env"]["RESUME_OPTIMIZER_STATE"] == "1"
    assert plan["train_env"]["COSMOS_MIXED_STEP_POLICY"] == "universe"


def test_root_guard_rejects_legacy_root_alias_and_existing_checkpoints(tmp_path):
    legacy_root = tmp_path / "legacy_s4"
    with pytest.raises(ValueError, match="legacy"):
        validate_policy_root(legacy_root, legacy_root=legacy_root)

    root = tmp_path / "new_policy"
    (root / "checkpoints" / "step_250").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="checkpoints"):
        validate_policy_root(root, legacy_root=legacy_root)

    assert validate_policy_root(root, legacy_root=legacy_root, resume=True) == root.resolve()


def test_policy_plan_rejects_nonempty_checkpoint_directory_without_resume(tmp_path):
    root = tmp_path / "policy"
    (root / "checkpoints" / "step_250").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="checkpoints"):
        _plan(tmp_path, root=root)


def test_device_parsing_requires_exactly_eight_unique_devices(tmp_path):
    assert parse_device_list(DEVICES, world_size=8) == tuple(str(index) for index in range(8))

    with pytest.raises(ValueError, match="8 devices"):
        parse_device_list("0,1", world_size=8)
    with pytest.raises(ValueError, match="unique"):
        parse_device_list("0,1,2,3,4,5,6,6", world_size=8)
    with pytest.raises(ValueError, match="exactly 8"):
        _plan(tmp_path, root=tmp_path / "bad_world", world_size=2, device_list="0,1")


def test_torchrun_command_is_an_eight_rank_train_invocation():
    command = build_torchrun_command(
        torchrun=Path("/tmp/torchrun"),
        world_size=8,
        master_port=29801,
        teacher_model_path=Path("/tmp/cosmos"),
        dataset_path=Path("/tmp/libero"),
        output_dir=Path("/tmp/output"),
        resume_from_path=Path("/tmp/stage1"),
    )

    assert command[:3] == ["/tmp/torchrun", "--nproc_per_node=8", "--master_port=29801"]
    assert command[3] == "distillation_flowmap/train.py"
    assert command[command.index("--gradient-accumulation-steps") + 1] == "1"


def test_multibudget_eval_plans_use_distinct_outputs_and_teacher_step_caches(tmp_path):
    root = tmp_path / "universe"
    protocol_root = tmp_path / "shared_protocol"
    plans = build_multibudget_eval_plans(
        checkpoint_dir=root / "checkpoints" / "step_500",
        dataset_path=Path("/tmp/libero"),
        selection_manifest=protocol_root / "selection_manifest.json",
        eval_pairs=protocol_root / "eval_pairs.json",
        shared_protocol_root=protocol_root,
        output_dir=root,
        teacher_model_path=Path("/tmp/cosmos"),
        python_executable="/tmp/python",
    )

    assert [plan["budget"] for plan in plans] == ["s1", "s2", "s4"]
    assert [(plan["teacher_steps"], plan["student_steps"]) for plan in plans] == [
        (4, 1),
        (4, 2),
        (8, 4),
    ]
    assert plans[0]["cache_dir"] == protocol_root / "teacher_cache" / "selection" / "t4"
    assert plans[1]["cache_dir"] == protocol_root / "teacher_cache" / "selection" / "t4"
    assert plans[2]["cache_dir"] == protocol_root / "teacher_cache" / "selection" / "t8"
    assert len({plan["output_json"] for plan in plans}) == 3
    assert all("--output-json" in plan["argv"] for plan in plans)
    assert all(plan["env"]["CUDA_VISIBLE_DEVICES"] == "0" for plan in plans)
    assert all(plan["env"]["COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES"] == "0" for plan in plans)


def test_manifest_payload_records_reproducibility_and_objective_contract(tmp_path):
    plan = _plan(tmp_path, force_sequence="s1,s2,s4")

    payload = build_policy_manifest_payload(
        plan,
        git_hash="abc123",
        dirty_diff_hash="def456",
        timestamp="2026-07-16T00:00:00+00:00",
    )

    assert payload["schema"] == "cosmos_mixed_step_policy_manifest_v1"
    assert payload["git_hash"] == "abc123"
    assert payload["dirty_diff_hash"] == "def456"
    assert payload["source_checkpoint"] == "/tmp/stage1_step5000"
    assert payload["policy"]["name"] == "universe"
    assert payload["seed"] == 20260716
    assert payload["visible_devices"] == list(range(8))
    assert payload["train_command"] == plan["train_argv"]
    assert payload["objective_settings"]["progressive_template_stage"] == "s4"
    assert payload["objective_settings"]["target_free_online_student"] is True
    assert payload["objective_settings"]["mixed_endpoint_sampling"] is True


def test_legacy_root_constant_is_not_the_new_policy_default_root():
    assert "stage2_progressive" in str(LEGACY_PROGRESSIVE_ROOT)
