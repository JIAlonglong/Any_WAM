import json
from pathlib import Path

import pytest

import distillation_flowmap.run_cosmos_mixed_step_policy as mixed_runner
from distillation_flowmap.run_cosmos_mixed_step_policy import (
    DEFAULT_STAGE1_CHECKPOINT,
    LEGACY_PROGRESSIVE_ROOT,
    build_multibudget_eval_plans,
    build_policy_manifest_payload,
    build_policy_train_plan,
    build_torchrun_command,
    copy_immutable_protocol_metadata,
    execute_policy_plan,
    parse_args,
    parse_device_list,
    validate_preflight_selection_evidence,
    validate_policy_eval_execution_contract,
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
        "stage1_checkpoint": DEFAULT_STAGE1_CHECKPOINT,
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


def _resume_manifest(policy_name: str = "universe", **overrides) -> dict:
    payload = {
        "policy": {"name": policy_name},
        "source_checkpoint": str(DEFAULT_STAGE1_CHECKPOINT),
    }
    payload.update(overrides)
    return payload


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
    assert plan["source_checkpoint"] == DEFAULT_STAGE1_CHECKPOINT
    assert plan["max_train_steps"] == 5000
    assert plan["save_interval"] == 250
    assert plan["train_env"]["RESUME_FROM_PATH"] == str(DEFAULT_STAGE1_CHECKPOINT)
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
    assert plan["train_argv"][plan["train_argv"].index("--resume-from-path") + 1] == str(DEFAULT_STAGE1_CHECKPOINT)
    assert plan["train_argv"][plan["train_argv"].index("--gradient-accumulation-steps") + 1] == "1"
    assert all(eval_plan["env"]["COSMOS_PROGRESSIVE_STAGE"] == "s4" for eval_plan in plan["eval_plans"])
    assert all(eval_plan["env"]["COSMOS_MIXED_STEP_POLICY"] == policy_name for eval_plan in plan["eval_plans"])


def test_default_stage1_source_is_the_agreed_common_online_student_checkpoint():
    assert str(DEFAULT_STAGE1_CHECKPOINT).endswith(
        "output_libero_cosmos_policy_stage1_cosmos_latent_cdiff_8gpu_20260706_"
        "cosmos_latent_s1s2_8gpu/checkpoints/step_5000"
    )


def test_resume_manifest_must_retain_the_common_stage1_provenance(tmp_path):
    root = tmp_path / "universe"
    root.mkdir()
    (root / "policy_manifest.json").write_text(
        json.dumps(
            _resume_manifest(source_checkpoint=str(tmp_path / "alternate_stage1"))
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="common Stage-1 source provenance"):
        _plan(tmp_path, root=root, current_step=250, chunk_size=250)


@pytest.mark.parametrize("nested_child", ["protocol", "metrics"])
def test_train_plan_rejects_nested_owned_artifact_symlink_before_any_write(
    tmp_path, nested_child
):
    root = tmp_path / "policy"
    protocol_source = tmp_path / "shared_protocol"
    root.mkdir()
    protocol_source.mkdir()
    (root / nested_child).symlink_to(protocol_source, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        _plan(tmp_path, root=root, protocol_source_root=protocol_source)


def test_train_plan_rejects_existing_target_selection_proxy_without_overwrite(tmp_path):
    root = tmp_path / "policy"
    proxy = root / "metrics" / "selection" / "step_250" / "selection_proxy.json"
    proxy.parent.mkdir(parents=True)
    proxy.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="selection proxy"):
        _plan(tmp_path, root=root)


def test_protocol_copy_rechecks_nested_symlink_before_mkdir_or_copy(tmp_path):
    plan = _plan(tmp_path)
    source_root = Path(plan["protocol_source_root"])
    source_root.mkdir()
    for filename in mixed_runner._METADATA_FILENAMES:
        (source_root / filename).write_text("{}", encoding="utf-8")
    root = Path(plan["root"])
    root.mkdir()
    (root / "protocol").symlink_to(source_root, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        copy_immutable_protocol_metadata(plan)


def test_train_eval_contract_allows_only_its_canonical_copied_protocol_metadata(tmp_path):
    plan = _plan(tmp_path)

    validate_policy_eval_execution_contract(
        plan,
        checkpoint_dir=plan["checkpoint_dir"],
        protocol_metadata_root=plan["root"] / "protocol",
    )
    with pytest.raises(ValueError, match="canonical root/protocol"):
        validate_policy_eval_execution_contract(
            plan,
            checkpoint_dir=plan["checkpoint_dir"],
            protocol_metadata_root=plan["protocol_source_root"],
        )
    with pytest.raises(ValueError, match="canonical root/protocol"):
        validate_policy_eval_execution_contract(
            plan,
            checkpoint_dir=plan["checkpoint_dir"],
            protocol_metadata_root=tmp_path / "arbitrary_protocol",
        )


def _preflight_record(label: str, selection_ordinal: int) -> dict:
    index, teacher_steps, student_steps = {
        "s1": (0, 4, 1),
        "s2": (1, 4, 2),
        "s4": (2, 8, 4),
    }[label]
    return {
        "policy_name": "universe",
        "forced": True,
        "pair_label": label,
        "pair_index": index,
        "teacher_steps": teacher_steps,
        "student_steps": student_steps,
        "selection_ordinal": selection_ordinal,
    }


def test_preflight_plan_sets_only_reviewed_auxiliary_overrides_and_forced_schedule(tmp_path):
    plan = _plan(
        tmp_path,
        max_train_steps=9,
        save_interval=9,
        stop_after_step=9,
        chunk_size=9,
        force_sequence="s1,s2,s4,s1,s2,s4,s1,s2,s4",
        preflight_evidence=True,
        opd_aux_warmup_steps=0,
        opd_aux_interval=1,
    )

    assert plan["preflight_evidence"] is True
    assert plan["forced_indices"] == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert plan["train_env"]["OPD_AUX_WARMUP_STEPS"] == "0"
    assert plan["train_env"]["OPD_AUX_INTERVAL"] == "1"
    assert "OPD_AUX_WARMUP_STEPS" not in _plan(tmp_path / "full")["train_env"]
    assert "OPD_AUX_INTERVAL" not in _plan(tmp_path / "full")["train_env"]


def test_preflight_selection_evidence_requires_exact_forced_rank_zero_sequence(tmp_path):
    path = tmp_path / "cosmos_mixed_step_opd.jsonl"
    labels = ["s1", "s2", "s4"] * 3
    records = [
        _preflight_record(label, selection_ordinal)
        for selection_ordinal, label in enumerate(labels)
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    assert validate_preflight_selection_evidence(path) == {"s1": 3, "s2": 3, "s4": 3}

    path.write_text(
        "\n".join(json.dumps(record) for record in records[:8]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly nine"):
        validate_preflight_selection_evidence(path)

    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        validate_preflight_selection_evidence(path)


@pytest.mark.parametrize(
    ("mutate_lines", "match"),
    [
        (lambda lines: lines.insert(3, ""), "blank"),
        (lambda lines: lines.__setitem__(1, lines[2]), "sequence"),
        (
            lambda lines: lines.append(
                json.dumps(_preflight_record("s1", 9), sort_keys=True)
            ),
            "exactly nine",
        ),
    ],
)
def test_preflight_selection_evidence_rejects_blank_repeated_or_extra_records(
    tmp_path, mutate_lines, match
):
    labels = ["s1", "s2", "s4"] * 3
    lines = [
        json.dumps(_preflight_record(label, ordinal), sort_keys=True)
        for ordinal, label in enumerate(labels)
    ]
    mutate_lines(lines)
    path = tmp_path / "cosmos_mixed_step_opd.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        validate_preflight_selection_evidence(path)


@pytest.mark.parametrize(
    ("existing_step", "transformer_only"),
    [(500, False), (750, True), (1000, False)],
)
def test_resume_plan_rejects_every_future_scheduled_checkpoint_destination(
    tmp_path, existing_step, transformer_only
):
    root = tmp_path / "universe"
    root.mkdir()
    (root / "policy_manifest.json").write_text(
        json.dumps(_resume_manifest()), encoding="utf-8"
    )
    existing = root / "checkpoints" / f"step_{existing_step}"
    if transformer_only:
        (existing / "online_student" / "transformer").mkdir(parents=True)
    else:
        existing.mkdir(parents=True)

    with pytest.raises(FileExistsError, match=f"step_{existing_step}"):
        _plan(
            tmp_path,
            root=root,
            current_step=250,
            chunk_size=750,
            max_train_steps=1000,
            save_interval=250,
        )


def test_fresh_policy_root_is_claimed_atomically_only_at_execution_boundary(tmp_path):
    plan = _plan(tmp_path)
    root = Path(plan["root"])

    assert not root.exists()
    assert mixed_runner.claim_fresh_policy_root(plan) == root
    assert root.is_dir()
    with pytest.raises(FileExistsError, match="fresh policy root"):
        mixed_runner.claim_fresh_policy_root(plan)


def test_resume_plan_uses_only_its_own_online_checkpoint_and_optimizer_state(tmp_path):
    root = tmp_path / "universe"
    (root / "policy_manifest.json").parent.mkdir(parents=True)
    (root / "policy_manifest.json").write_text(
        json.dumps(_resume_manifest()), encoding="utf-8"
    )
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


def test_root_guard_rejects_an_ancestor_that_contains_the_legacy_root(tmp_path):
    legacy_root = tmp_path / "legacy_s4" / "run"

    with pytest.raises(ValueError, match="legacy"):
        validate_policy_root(legacy_root.parent, legacy_root=legacy_root)


def test_policy_plan_rejects_output_overlap_with_protocol_source_and_alternate_stage1(tmp_path):
    protocol_source_root = tmp_path / "shared_protocol"
    with pytest.raises(ValueError, match="protocol source"):
        _plan(
            tmp_path,
            root=protocol_source_root / "new_policy",
            protocol_source_root=protocol_source_root,
        )

    stage1_checkpoint = tmp_path / "stage1" / "checkpoints" / "step_5000"
    with pytest.raises(ValueError, match="approved common Stage-1"):
        _plan(
            tmp_path,
            root=stage1_checkpoint.parents[1],
            protocol_source_root=tmp_path / "other_protocol",
            stage1_checkpoint=stage1_checkpoint,
        )


@pytest.mark.parametrize(
    ("input_name", "input_path", "root_contains_input", "match"),
    [
        ("teacher_model_path", "teacher", False, "teacher model"),
        ("teacher_model_path", "teacher", True, "teacher model"),
        ("dataset_path", "dataset", False, "dataset"),
        ("dataset_path", "dataset", True, "dataset"),
    ],
)
def test_policy_plan_rejects_output_overlap_with_every_read_only_input(
    tmp_path, input_name, input_path, root_contains_input, match
):
    root = tmp_path / "new_policy"
    read_only_source = root / input_path if root_contains_input else tmp_path / input_path
    if not root_contains_input:
        root = read_only_source / "new_policy"
    with pytest.raises(ValueError, match=match):
        _plan(tmp_path, root=root, **{input_name: read_only_source})


def test_fresh_policy_rejects_existing_root_manifest_even_without_checkpoints(tmp_path):
    root = tmp_path / "fresh_policy"
    root.mkdir(parents=True)

    for manifest_policy in ("universe", "s2"):
        (root / "policy_manifest.json").write_text(
            json.dumps({"policy": {"name": manifest_policy}}), encoding="utf-8"
        )
        with pytest.raises(FileExistsError, match="policy_manifest"):
            _plan(tmp_path, root=root, policy_name="universe")


def test_resume_requires_a_matching_policy_manifest(tmp_path):
    root = tmp_path / "universe"
    resume_kwargs = {"root": root, "current_step": 250, "chunk_size": 250}

    with pytest.raises(FileNotFoundError, match="policy_manifest"):
        _plan(tmp_path, **resume_kwargs)

    root.mkdir(parents=True)
    (root / "policy_manifest.json").write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        _plan(tmp_path, **resume_kwargs)

    (root / "policy_manifest.json").write_text(
        json.dumps({"policy": {"name": "s2"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="requested policy"):
        _plan(tmp_path, **resume_kwargs)

    (root / "policy_manifest.json").write_text(
        json.dumps(_resume_manifest()), encoding="utf-8"
    )
    assert _plan(tmp_path, **resume_kwargs)["policy"]["name"] == "universe"


def test_execute_revalidates_resume_manifest_before_any_subprocess(tmp_path):
    root = tmp_path / "universe"
    root.mkdir(parents=True)
    manifest_path = root / "policy_manifest.json"
    manifest_path.write_text(
        json.dumps(_resume_manifest()), encoding="utf-8"
    )
    plan = _plan(tmp_path, root=root, current_step=250, chunk_size=250)
    manifest_path.write_text(
        json.dumps({"policy": {"name": "s1"}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="requested policy"):
        execute_policy_plan(plan)


@pytest.mark.parametrize(
    ("input_key", "suffix", "match"),
    [
        ("teacher_model_path", "teacher", "teacher model"),
        ("dataset_path", "dataset", "dataset"),
    ],
)
def test_execute_revalidates_teacher_and_dataset_isolation_before_any_subprocess(
    tmp_path, input_key, suffix, match
):
    root = tmp_path / "universe"
    root.mkdir(parents=True)
    (root / "policy_manifest.json").write_text(
        json.dumps(_resume_manifest()), encoding="utf-8"
    )
    plan = _plan(tmp_path, root=root, current_step=250, chunk_size=250)
    plan[input_key] = root / "read_only" / suffix

    with pytest.raises(ValueError, match=match):
        execute_policy_plan(plan)


def test_policy_allowlist_rejects_unknown_registry_entries_in_builder_and_cli(tmp_path):
    with pytest.raises(ValueError, match="Unsupported Cosmos mixed-step policy"):
        _plan(tmp_path, policy_name="s4")

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--policy",
                "s4",
                "--root",
                str(tmp_path / "new_policy"),
                "--protocol-source-root",
                str(tmp_path / "protocol"),
            ]
        )


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
    assert payload["source_checkpoint"] == str(DEFAULT_STAGE1_CHECKPOINT)
    assert payload["policy"]["name"] == "universe"
    assert payload["seed"] == 20260716
    assert payload["visible_devices"] == list(range(8))
    assert payload["train_command"] == plan["train_argv"]
    assert payload["objective_settings"]["progressive_template_stage"] == "s4"
    assert payload["objective_settings"]["target_free_online_student"] is True
    assert payload["objective_settings"]["mixed_endpoint_sampling"] is True


def test_legacy_root_constant_is_not_the_new_policy_default_root():
    assert "stage2_progressive" in str(LEGACY_PROGRESSIVE_ROOT)
