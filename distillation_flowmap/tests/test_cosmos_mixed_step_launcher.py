import json
from pathlib import Path

import pytest

import distillation_flowmap.run_cosmos_mixed_step_policy as mixed_runner
from distillation_flowmap.run_cosmos_mixed_step_policy import (
    build_policy_eval_plan,
    execute_policy_eval_plan,
    parse_args,
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
        "python_executable": "/tmp/python",
        "eval_device_list": "6",
        "eval_worker_device_list": "7",
    }
    defaults.update(overrides)
    return build_policy_eval_plan(**defaults)


def _launcher_source() -> str:
    assert LAUNCHER_PATH.is_file(), "Task 3 launcher has not been created"
    return LAUNCHER_PATH.read_text(encoding="utf-8")


def test_eval_only_plan_owns_its_checkpoint_and_contains_only_three_budget_evaluators(
    tmp_path,
):
    plan = _eval_plan(tmp_path)

    assert plan["schema"] == "cosmos_mixed_step_policy_eval_plan_v1"
    assert plan["policy"]["name"] == "universe"
    assert plan["checkpoint_dir"] == tmp_path / "universe" / "checkpoints" / "step_5000"
    assert "train_argv" not in plan
    assert [item["budget"] for item in plan["eval_plans"]] == ["s1", "s2", "s4"]
    assert [item["cache_dir"].name for item in plan["eval_plans"]] == ["t4", "t4", "t8"]
    assert all(item["env"]["COSMOS_MIXED_STEP_POLICY"] == "universe" for item in plan["eval_plans"])


def test_eval_only_plan_rejects_checkpoint_outside_the_owned_policy_root(tmp_path):
    with pytest.raises(ValueError, match="owned checkpoints"):
        _eval_plan(tmp_path, checkpoint_dir=tmp_path / "other" / "step_5000")


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
    assert '"--max-train-steps" "9"' in source
    assert '"--save-interval" "9"' in source
    assert '"--stop-after-step" "9"' in source
    assert 'preflight-universe-${run_tag}' in source
    assert 'cosmos-mixed-preflight-universe-${run_tag}' in source
    assert 'PREFLIGHT_COMPLETE' in source
    assert 'PREFLIGHT_FAILED' in source
    assert 'online_student/transformer/config.json' in source
    assert 'target_student' in source
    assert 'selection_proxy.json' in source


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
