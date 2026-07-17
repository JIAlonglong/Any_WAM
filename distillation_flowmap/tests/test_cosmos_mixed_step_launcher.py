import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

import distillation_flowmap.run_cosmos_mixed_step_policy as mixed_runner
from distillation_flowmap.run_cosmos_mixed_step_policy import (
    build_policy_eval_plan,
    execute_policy_eval_plan,
    parse_args,
    validate_owned_policy_checkpoint,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = REPO_ROOT / "distillation_flowmap" / "launch_cosmos_mixed_step_8gpu.sh"


def _write_policy_manifest(root: Path, policy_name: str = "universe") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "policy_manifest.json").write_text(
        json.dumps({"policy": {"name": policy_name}}), encoding="utf-8"
    )


def _eval_plan(tmp_path: Path, **overrides):
    root = tmp_path / "universe"
    _write_policy_manifest(root)
    defaults = {
        "policy_name": "universe",
        "root": root,
        "checkpoint_dir": root / "checkpoints" / "step_5000",
        "dataset_path": tmp_path / "dataset",
        "protocol_source_root": tmp_path / "shared_protocol",
        "teacher_model_path": tmp_path / "teacher",
        "stage1_checkpoint": tmp_path / "stage1",
        "python_executable": sys.executable,
        "eval_device_list": "6",
        "eval_worker_device_list": "7",
    }
    defaults.update(overrides)
    return build_policy_eval_plan(**defaults)


def _materialize_eval_execution_inputs(plan) -> None:
    checkpoint = Path(plan["checkpoint_dir"])
    transformer = checkpoint / "online_student" / "transformer"
    transformer.mkdir(parents=True, exist_ok=True)
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    protocol = Path(plan["protocol_source_root"])
    protocol.mkdir(parents=True, exist_ok=True)
    (protocol / "selection_manifest.json").write_text(
        json.dumps({"records": []}), encoding="utf-8"
    )
    (protocol / "eval_pairs.json").write_text(
        json.dumps({"pairs": [{"pair_id": "pair0"}]}), encoding="utf-8"
    )


def _launcher_source() -> str:
    assert LAUNCHER_PATH.is_file(), "Task 3 launcher has not been created"
    return LAUNCHER_PATH.read_text(encoding="utf-8")


def _run_pure_launcher_write_helper(
    tmp_path: Path,
    *,
    root: Path,
    protocol: Path,
    dataset: Path,
    teacher: Path,
    stage1: Path,
    invocation: str,
) -> subprocess.CompletedProcess[str]:
    """Exercise only a copied shell helper, never the launcher's main entrypoint."""
    footer = '\ntrap launcher_exit_cleanup EXIT\nmain "$@"\n'
    source = _launcher_source()
    assert source.endswith(footer)
    harness = tmp_path / "pure_launcher_write_guard.sh"
    harness.write_text(
        source.removesuffix(footer)
        + "\n"
        + f"ROOT_BASE={shlex.quote(str(root))}\n"
        + f"PROTOCOL_SOURCE_ROOT={shlex.quote(str(protocol))}\n"
        + f"DATASET_PATH={shlex.quote(str(dataset))}\n"
        + f"TEACHER_MODEL_PATH={shlex.quote(str(teacher))}\n"
        + f"STAGE1_CHECKPOINT={shlex.quote(str(stage1))}\n"
        + "mkdir() { printf 'mkdir reached\\n' >&2; exit 99; }\n"
        + invocation
        + "\n",
        encoding="utf-8",
    )
    return subprocess.run(
        ["bash", str(harness)],
        text=True,
        capture_output=True,
        check=False,
    )


def _function_body(source: str, name: str, next_name: str) -> str:
    start = source.index(f"\n{name}() {{")
    end = source.index(f"\n{next_name}() {{", start)
    return source[start:end]


def test_eval_only_plan_owns_its_checkpoint_and_contains_only_three_budget_evaluators(
    tmp_path,
):
    plan = _eval_plan(tmp_path)

    assert plan["schema"] == "cosmos_mixed_step_policy_eval_plan_v1"
    assert plan["policy"]["name"] == "universe"
    assert plan["checkpoint_dir"] == tmp_path / "universe" / "checkpoints" / "step_5000"
    assert plan["stage1_checkpoint"] == tmp_path / "stage1"
    assert plan["python_executable"] == Path(sys.executable).resolve()
    assert "train_argv" not in plan
    assert [item["budget"] for item in plan["eval_plans"]] == ["s1", "s2", "s4"]
    assert [item["cache_dir"].name for item in plan["eval_plans"]] == ["t4", "t4", "t8"]
    assert all(item["env"]["COSMOS_MIXED_STEP_POLICY"] == "universe" for item in plan["eval_plans"])


def test_eval_only_plan_rejects_checkpoint_outside_the_owned_policy_root(tmp_path):
    with pytest.raises(ValueError, match="owned checkpoints"):
        _eval_plan(tmp_path, checkpoint_dir=tmp_path / "other" / "step_5000")


def test_eval_plan_rejects_a_canonical_policy_symlink_to_the_stage1_source(tmp_path):
    stage1 = tmp_path / "stage1"
    stage1.mkdir()
    root_base = tmp_path / "output_base"
    root_base.mkdir()
    policy_root_link = root_base / "universe"
    policy_root_link.symlink_to(stage1, target_is_directory=True)

    with pytest.raises(ValueError, match="Stage-1"):
        _eval_plan(
            tmp_path,
            root=policy_root_link,
            stage1_checkpoint=stage1,
        )


def test_eval_plan_rejects_a_nested_metrics_symlink_to_the_stage1_source(tmp_path):
    root = tmp_path / "universe"
    _write_policy_manifest(root)
    stage1 = tmp_path / "stage1"
    stage1.mkdir()
    (root / "metrics").symlink_to(stage1, target_is_directory=True)

    with pytest.raises(ValueError, match="must resolve under policy root"):
        _eval_plan(tmp_path, root=root, stage1_checkpoint=stage1)


@pytest.mark.parametrize("checkpoint_name", ["5000", "step_+1", "step_-1", "step_1.0", "step_"])
def test_eval_only_checkpoint_validator_accepts_only_exact_step_decimal_names(
    tmp_path, checkpoint_name
):
    root = tmp_path / "universe"
    root.mkdir()

    with pytest.raises(ValueError, match="owned checkpoints/step_N"):
        validate_owned_policy_checkpoint(
            root=root, checkpoint_dir=root / "checkpoints" / checkpoint_name
        )


def test_eval_only_cli_keeps_evaluation_explicit_and_does_not_select_training_mode(tmp_path):
    root = tmp_path / "universe"
    args = parse_args(
        [
            "--policy",
            "universe",
            "--root",
            str(root),
            "--protocol-source-root",
            str(tmp_path / "protocol"),
            "--eval-checkpoint",
            str(root / "checkpoints" / "step_5000"),
        ]
    )

    assert args.eval_checkpoint == root / "checkpoints" / "step_5000"
    assert args.run is False


def test_eval_only_executor_invokes_only_evaluator_argvs_with_no_real_subprocess(
    tmp_path, monkeypatch
):
    root = tmp_path / "universe"
    checkpoint = root / "checkpoints" / "step_5000"
    _write_policy_manifest(root)
    (checkpoint / "online_student" / "transformer").mkdir(parents=True)
    (checkpoint / "online_student" / "transformer" / "config.json").write_text(
        "{}", encoding="utf-8"
    )
    protocol = tmp_path / "shared_protocol"
    protocol.mkdir()
    (protocol / "selection_manifest.json").write_text(
        json.dumps({"records": []}), encoding="utf-8"
    )
    (protocol / "eval_pairs.json").write_text(
        json.dumps({"pairs": [{"pair_id": "pair0"}]}), encoding="utf-8"
    )
    plan = _eval_plan(tmp_path, root=root, checkpoint_dir=checkpoint, protocol_source_root=protocol)
    calls = []
    monkeypatch.setattr(
        mixed_runner.subprocess,
        "run",
        lambda argv, **kwargs: calls.append((list(argv), kwargs)),
    )
    monkeypatch.setattr(
        mixed_runner,
        "_write_selection_proxy",
        lambda passed_plan: passed_plan["selection_proxy_path"],
    )

    execute_policy_eval_plan(plan)

    assert len(calls) == 3
    assert all(
        any(item.endswith("eval_cosmos_progressive_stage2.py") for item in call[0])
        for call in calls
    )
    assert all("distillation_flowmap/train.py" not in call[0] for call in calls)


@pytest.mark.parametrize(
    "mutate_plan",
    [
        lambda plan: plan["eval_plans"][0]["argv"].__setitem__(
            1, "distillation_flowmap/train.py"
        ),
        lambda plan: plan["eval_plans"][2].__setitem__(
            "cache_dir", plan["protocol_source_root"] / "teacher_cache" / "selection" / "t4"
        ),
        lambda plan: plan["eval_plans"][0]["argv"].__setitem__(0, "/tmp/other-python"),
        lambda plan: plan.__setitem__(
            "selection_proxy_path", plan["root"] / "metrics" / "other" / "selection_proxy.json"
        ),
        lambda plan: (
            plan.__setitem__("python_executable", Path("/tmp/fake-python")),
            [
                eval_plan["argv"].__setitem__(0, "/tmp/fake-python")
                for eval_plan in plan["eval_plans"]
            ],
        ),
    ],
)
def test_eval_only_executor_rejects_programmatic_nonapproved_plans_before_subprocess(
    tmp_path, monkeypatch, mutate_plan
):
    plan = _eval_plan(tmp_path)
    _materialize_eval_execution_inputs(plan)
    mutate_plan(plan)
    calls = []
    monkeypatch.setattr(
        mixed_runner.subprocess,
        "run",
        lambda argv, **kwargs: calls.append((list(argv), kwargs)),
    )
    monkeypatch.setattr(
        mixed_runner,
        "_write_selection_proxy",
        lambda passed_plan: passed_plan["selection_proxy_path"],
    )

    with pytest.raises(ValueError, match="approved evaluator-only contract"):
        execute_policy_eval_plan(plan)

    assert calls == []


def test_custom_python_eval_plan_can_be_printed_but_never_executes(tmp_path, monkeypatch):
    plan = _eval_plan(tmp_path, python_executable="/tmp/print-only-python")
    assert plan["python_executable"] == Path("/tmp/print-only-python")
    _materialize_eval_execution_inputs(plan)
    calls = []
    monkeypatch.setattr(
        mixed_runner.subprocess,
        "run",
        lambda argv, **kwargs: calls.append((list(argv), kwargs)),
    )
    monkeypatch.setattr(
        mixed_runner,
        "_write_selection_proxy",
        lambda passed_plan: passed_plan["selection_proxy_path"],
    )

    with pytest.raises(ValueError, match="intended Python executable"):
        execute_policy_eval_plan(plan)

    assert calls == []


def test_eval_only_executor_revalidates_stage1_source_before_subprocess(tmp_path, monkeypatch):
    plan = _eval_plan(tmp_path)
    plan["stage1_checkpoint"] = plan["root"]
    calls = []
    monkeypatch.setattr(
        mixed_runner.subprocess,
        "run",
        lambda argv, **kwargs: calls.append((list(argv), kwargs)),
    )

    with pytest.raises(ValueError, match="Stage-1"):
        execute_policy_eval_plan(plan)

    assert calls == []


def test_eval_only_executor_revalidates_nested_metrics_symlink_before_subprocess(
    tmp_path, monkeypatch
):
    plan = _eval_plan(tmp_path)
    _materialize_eval_execution_inputs(plan)
    stage1 = Path(plan["stage1_checkpoint"])
    stage1.mkdir()
    (Path(plan["root"]) / "metrics").symlink_to(stage1, target_is_directory=True)
    calls = []
    monkeypatch.setattr(
        mixed_runner.subprocess,
        "run",
        lambda argv, **kwargs: calls.append((list(argv), kwargs)),
    )
    monkeypatch.setattr(
        mixed_runner,
        "_write_selection_proxy",
        lambda passed_plan: passed_plan["selection_proxy_path"],
    )

    with pytest.raises(ValueError, match="must resolve under policy root"):
        execute_policy_eval_plan(plan)

    assert calls == []


def test_launcher_declares_static_safe_shell_contract_and_never_runs_in_this_test():
    source = _launcher_source()

    assert source.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in source
    assert 'SUPPORTED_POLICIES=("universe" "s2" "s1")' in source
    assert "rm -rf" not in source
    # These tests only read source and monkeypatch the pure runner executor;
    # they never invoke the launcher or create a tmux session.
    assert "tmux " + "new-session" not in Path(__file__).read_text(encoding="utf-8")
    assert "Po" + "pen(" not in Path(__file__).read_text(encoding="utf-8")
    assert "new-session" in source


def test_launcher_preflight_is_unique_eight_gpu_and_forces_all_modes_three_times():
    source = _launcher_source()

    assert 'PREFLIGHT_PORT=29860' in source
    assert 'PREFLIGHT_FORCE_SEQUENCE="s1,s2,s4,s1,s2,s4,s1,s2,s4"' in source
    assert 'PREFLIGHT_MINIMUM_PER_LABEL=3' in source
    assert '"--max-train-steps" "9"' in source
    assert '"--save-interval" "9"' in source
    assert '"--stop-after-step" "9"' in source
    assert '"--preflight-evidence"' in source
    assert '"--opd-aux-warmup-steps" "0"' in source
    assert '"--opd-aux-interval" "1"' in source
    assert 'preflight-universe-${run_tag}' in source
    assert 'cosmos-mixed-preflight-universe-${run_tag}' in source
    assert 'PREFLIGHT_COMPLETE' in source
    assert 'PREFLIGHT_FAILED' in source
    assert 'online_student/transformer/config.json' in source
    assert 'target_student' in source
    assert 'selection_proxy.json' in source
    assert 'metrics/cosmos_mixed_step_opd.jsonl' in source


def test_launcher_preflight_gate_validates_rank_zero_jsonl_before_completion_marker():
    source = _launcher_source()
    worker = _function_body(source, "write_worker_script", "launch_tmux_session")

    assert 'PREFLIGHT_SELECTION_LOG=%q' in worker
    assert 'PREFLIGHT_MINIMUM_PER_LABEL=%q' in worker
    assert "validate_preflight_selection_evidence" in worker
    assert '"$PREFLIGHT_SELECTION_LOG" "$PREFLIGHT_MINIMUM_PER_LABEL"' in worker
    assert worker.index("validate_preflight_selection_evidence") < worker.index(
        'if [[ -e "$SUCCESS_MARKER"'
    )
    assert 'write_failure_marker' in worker


def test_launcher_training_input_validation_pins_the_common_stage1_before_artifacts():
    source = _launcher_source()
    training_validation = _function_body(source, "validate_training_inputs", "validate_eval_inputs")

    assert 'canonical_directory "$DEFAULT_STAGE1_CHECKPOINT"' in training_validation
    assert '"$STAGE1_CHECKPOINT" == "$approved_stage1_checkpoint"' in training_validation
    assert "approved common Stage-1 checkpoint" in training_validation
    assert training_validation.index("approved_stage1_checkpoint") < training_validation.index(
        "assert_root_base_input_isolation"
    ) < training_validation.index("require_program")


def test_launcher_has_serial_gates_named_sessions_and_no_s4_training_policy():
    source = _launcher_source()

    assert 'POLICY_PORTS=( ["universe"]=29861 ["s2"]=29862 ["s1"]=29863 )' in source
    assert 'require_marker "$root_base/PREFLIGHT_COMPLETE" "Universe start"' in source
    assert 'require_marker "$root_base/universe/TRAINING_COMPLETE" "S2 start"' in source
    assert 'require_marker "$root_base/s2/TRAINING_COMPLETE" "S1 start"' in source
    assert 'cosmos-mixed-${policy}-${run_tag}' in source
    assert 'case "$policy" in\n        universe|s2|s1)' in source
    assert 'start s4' not in source
    assert 'POLICY_PORTS=( ["s4"]=' not in source


def test_launcher_tmux_command_status_eval_and_unknown_command_are_fail_safe():
    source = _launcher_source()

    assert '"$TMUX_BIN" new-session -d -s "$session_name" "$worker_script"' in source
    assert '"$TMUX_BIN" has-session -t "$session_name"' in source
    assert 'show_status "$ROOT_BASE"' in source
    assert '"--eval-checkpoint" "$checkpoint"' in source
    assert 'Refusing to overwrite existing selection proxy' in source
    assert 'die "Unknown command: $command"' in source
    assert 'Usage:' in source


def test_launcher_reserves_fixed_operation_identity_before_logs_workers_or_tmux():
    source = _launcher_source()
    preflight = _function_body(source, "start_preflight", "start_policy")
    policy = _function_body(source, "start_policy", "start_eval")
    evaluation = _function_body(source, "start_eval", "main")

    assert 'RESERVATION_ROOT_NAME=".cosmos_mixed_step_reservations"' in source
    assert 'if ! mkdir "$reservation_dir"; then' in source
    assert 'Operation already has an in-flight reservation' in source
    assert 'reserve_operation "$ROOT_BASE" "preflight"' in preflight
    assert 'reserve_operation "$root_base" "policy-${policy}"' in policy
    assert 'reserve_operation "$root_base" "eval-${policy}-${checkpoint_label}"' in evaluation
    for body in (preflight, policy, evaluation):
        assert body.index("reserve_operation") < body.index("prepare_artifact_paths")
        assert body.index("reserve_operation") < body.index("launch_tmux_session")
        assert body.index("launch_tmux_session") < body.index(
            "transfer_active_reservation_to_worker"
        )
    assert 'rmdir "$RESERVATION_DIR"' in source
    assert 'if (( exit_code != 0 )); then\n        write_failure_marker\n    fi\n    release_reservation' in source
    assert 'in-flight reservation:' in source


def test_launcher_rejects_bidirectional_input_overlap_before_artifact_creation():
    source = _launcher_source()
    training_validation = _function_body(source, "validate_training_inputs", "validate_eval_inputs")
    eval_validation = _function_body(source, "validate_eval_inputs", "new_run_tag")

    assert 'assert_root_base_input_isolation' in source
    assert '"$root_base" == "$input_path"/*' in source
    assert '"$input_path" == "$root_base"/*' in source
    assert training_validation.index("assert_root_base_input_isolation") < training_validation.index(
        "require_program"
    )
    assert eval_validation.index("assert_root_base_input_isolation") < eval_validation.index(
        "require_program"
    )
    assert '"$ROOT_BASE" "$PROTOCOL_SOURCE_ROOT" "$DATASET_PATH"' in training_validation
    assert '"$TEACHER_MODEL_PATH" "$STAGE1_CHECKPOINT"' in training_validation
    assert '"$ROOT_BASE" "$PROTOCOL_SOURCE_ROOT" "$DATASET_PATH"' in eval_validation
    assert '"$TEACHER_MODEL_PATH"' in eval_validation
    assert '"$STAGE1_CHECKPOINT"' in eval_validation

    for body in (
        _function_body(source, "start_preflight", "start_policy"),
        _function_body(source, "start_policy", "start_eval"),
        _function_body(source, "start_eval", "main"),
    ):
        validation_call = "validate_eval_inputs" if body.startswith("\nstart_eval") else "validate_training_inputs"
        assert body.index(validation_call) < body.index("prepare_artifact_paths")


def test_launcher_eval_rechecks_canonical_policy_root_and_forwards_stage1_source():
    source = _launcher_source()
    evaluation = _function_body(source, "start_eval", "main")

    assert 'policy_root="$(canonical_directory "$root_base/$policy")"' in evaluation
    assert 'assert_root_base_input_isolation \\' in evaluation
    assert '"$policy_root" "$PROTOCOL_SOURCE_ROOT" "$DATASET_PATH"' in evaluation
    assert '"$TEACHER_MODEL_PATH" "$STAGE1_CHECKPOINT"' in evaluation
    assert '"--stage1-checkpoint" "$STAGE1_CHECKPOINT"' in evaluation
    assert evaluation.index('policy_root="$(canonical_directory') < evaluation.index(
        "assert_root_base_input_isolation"
    ) < evaluation.index("prepare_artifact_paths")


def test_launcher_eval_rejects_external_policy_root_symlink_before_reservation_or_artifacts(
    tmp_path,
):
    root = tmp_path / "root"
    root.mkdir()
    immutable_sources = {
        "protocol": tmp_path / "protocol",
        "dataset": tmp_path / "dataset",
        "teacher": tmp_path / "teacher",
        "stage1": tmp_path / "stage1",
    }
    for source in immutable_sources.values():
        source.mkdir()
    external_policy_root = tmp_path / "external-output" / "universe"
    checkpoint = external_policy_root / "checkpoints" / "step_5000"
    checkpoint.mkdir(parents=True)
    _write_policy_manifest(external_policy_root)
    (root / "universe").symlink_to(external_policy_root, target_is_directory=True)
    (tmp_path / "run_cosmos_mixed_step_policy.py").write_text(
        "# pure shell-harness runner stub\n", encoding="utf-8"
    )

    result = _run_pure_launcher_write_helper(
        tmp_path,
        root=root,
        protocol=immutable_sources["protocol"],
        dataset=immutable_sources["dataset"],
        teacher=immutable_sources["teacher"],
        stage1=immutable_sources["stage1"],
        invocation=(
            f"PYTHON_BIN={shlex.quote(sys.executable)}\n"
            "reserve_operation() { printf 'reservation reached\\n' >&2; exit 99; }\n"
            "prepare_artifact_paths() { printf 'artifact setup reached\\n' >&2; exit 99; }\n"
            "write_worker_script() { printf 'worker write reached\\n' >&2; exit 99; }\n"
            'start_eval "universe" "$ROOT_BASE/universe/checkpoints/step_5000"'
        ),
    )

    assert result.returncode == 2
    assert "Launcher write destination must resolve under ROOT_BASE" in result.stderr
    assert "reservation reached" not in result.stderr
    assert "artifact setup reached" not in result.stderr
    assert "worker write reached" not in result.stderr
    assert not (root / ".cosmos_mixed_step_reservations").exists()
    assert not (root / "logs").exists()


def test_launcher_eval_checks_resolved_policy_root_ownership_before_checkpoint_work():
    source = _launcher_source()
    evaluation = _function_body(source, "start_eval", "main")

    policy_root_index = evaluation.index('policy_root="$(canonical_directory')
    assert "assert_launcher_owned_write_paths_isolated" in evaluation
    ownership_guard_index = evaluation.index(
        "assert_launcher_owned_write_paths_isolated", policy_root_index
    )
    checkpoint_index = evaluation.index('checkpoint="$(canonical_directory')
    assert '"$policy_root"' in evaluation[ownership_guard_index:checkpoint_index]
    assert policy_root_index < ownership_guard_index < checkpoint_index


def test_launcher_eval_guards_nested_metrics_outputs_and_terminal_markers_before_writes():
    source = _launcher_source()
    evaluation = _function_body(source, "start_eval", "main")

    assert "assert_eval_artifact_paths_isolated" in source
    assert '"$metrics_dir" "$selection_proxy"' in evaluation
    assert '"$eval_complete_marker" "$eval_failed_marker"' in evaluation
    assert evaluation.index("assert_eval_artifact_paths_isolated") < evaluation.index(
        "ensure_marker_absent"
    ) < evaluation.index("prepare_artifact_paths")


@pytest.mark.parametrize(
    ("symlink_name", "target_name", "invocation"),
    [
        (
            "logs",
            "stage1",
            'prepare_artifact_paths "$ROOT_BASE" "guard-session"',
        ),
        (
            ".cosmos_mixed_step_reservations",
            "protocol",
            'reserve_operation "$ROOT_BASE" "preflight" "$ROOT_BASE/PREFLIGHT_FAILED"',
        ),
    ],
)
def test_launcher_rejects_symlinked_owned_write_roots_before_any_mkdir(
    tmp_path, symlink_name, target_name, invocation
):
    root = tmp_path / "root"
    root.mkdir()
    immutable_sources = {
        "protocol": tmp_path / "protocol",
        "dataset": tmp_path / "dataset",
        "teacher": tmp_path / "teacher",
        "stage1": tmp_path / "stage1",
    }
    for source in immutable_sources.values():
        source.mkdir()
    target = immutable_sources[target_name]
    (root / symlink_name).symlink_to(target, target_is_directory=True)

    result = _run_pure_launcher_write_helper(
        tmp_path,
        root=root,
        protocol=immutable_sources["protocol"],
        dataset=immutable_sources["dataset"],
        teacher=immutable_sources["teacher"],
        stage1=immutable_sources["stage1"],
        invocation=invocation,
    )

    assert result.returncode == 2
    assert "Launcher write destination must resolve under ROOT_BASE" in result.stderr
    assert "mkdir reached" not in result.stderr
    assert not any(target.iterdir())


def test_launcher_rechecks_all_owned_write_destinations_before_any_mkdir_or_redirection():
    source = _launcher_source()
    training_validation = _function_body(source, "validate_training_inputs", "validate_eval_inputs")
    eval_validation = _function_body(source, "validate_eval_inputs", "write_marker_once")
    reservation = _function_body(source, "reserve_operation", "transfer_active_reservation_to_worker")
    artifacts = _function_body(source, "prepare_artifact_paths", "write_worker_script")
    worker = _function_body(source, "write_worker_script", "launch_tmux_session")

    assert "assert_launcher_owned_write_paths_isolated" in source
    for validation in (training_validation, eval_validation):
        assert validation.index("assert_launcher_owned_write_paths_isolated") < validation.index(
            'require_program "$PYTHON_BIN"'
        )
    assert '"$reservation_root"' in reservation
    assert '"$log_dir" "$LOG_FILE" "$WORKER_SCRIPT"' in artifacts
    assert reservation.index("assert_launcher_owned_write_paths_isolated") < reservation.index(
        'mkdir -p "$reservation_root"'
    )
    assert artifacts.index("assert_launcher_owned_write_paths_isolated") < artifacts.index(
        'mkdir -p "$log_dir"'
    )
    assert worker.index("assert_launcher_owned_write_paths_isolated") < worker.index(
        ': > "$log_file"'
    ) < worker.index('} > "$worker_script"')
