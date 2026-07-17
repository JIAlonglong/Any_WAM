import csv
import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "distillation_flowmap" / "run_robotwin_teacher_student_offline.sh"


def test_launcher_is_executable_for_documented_direct_invocation():
    assert stat.S_IMODE(LAUNCHER.stat().st_mode) & 0o111


def _run_plan(tmp_path, *extra_args, check=True):
    output_root = tmp_path / "offline_matrix"
    command = [
        "bash",
        str(LAUNCHER),
        "--split",
        "core2",
        "--checkpoint",
        "danceopd",
        "--source",
        "target",
        "--k",
        "1",
        "--output-root",
        str(output_root),
        *extra_args,
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result, output_root


def _metric_payload(k=1, pair_id="t1000_r0_i0"):
    prefix = f"rollout_eval/{pair_id}/s{k}_t{k}/"
    values = {
        "action_teacher_gt_xr_mse": 0.30,
        "action_teacher_gt_xr_l1": 0.40,
        "action_teacher_gt_v_mse": 0.50,
        "action_teacher_gt_v_l1": 0.60,
        "action_gt_xr_mse": 0.20,
        "action_gt_xr_l1": 0.30,
        "action_gt_v_mse": 0.40,
        "action_gt_v_l1": 0.50,
        "video_teacher_gt_x_mse": 0.70,
        "video_teacher_gt_x_l1": 0.80,
        "video_teacher_gt_v_mse": 0.90,
        "video_gt_x_mse": 0.60,
        "video_gt_x_l1": 0.70,
        "video_gt_v_mse": 0.80,
    }
    return {prefix + name: value for name, value in values.items()}


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_pair_file(path, pair_ids=("t1000_r0_i0",)):
    _write_json(
        path,
        {
            "schema": "robotwin_stepwam_mini_eval_pairs_v1",
            "pairs": [
                {"pair_id": pair_id, "t": 1000.0, "r": 0.0, "pair_seed": index}
                for index, pair_id in enumerate(pair_ids)
            ],
        },
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result_plan(output_root, result_path, *, pair_ids=("t1000_r0_i0",), execution_requested=False):
    pair_path = output_root / "protocol" / "eval_pairs.json"
    pair_sha256 = _write_pair_file(pair_path, pair_ids)
    return {
        "schema": "robotwin_teacher_student_offline_plan_v1",
        "execution_requested": execution_requested,
        "output_root": str(output_root),
        "selection": {"split": "core2"},
        "entries": [
            {
                "split": "core2",
                "checkpoint_name": "danceopd",
                "rollout_source": "target",
                "student_steps": 1,
                "teacher_steps": 1,
                "pair_path": str(pair_path),
                "pair_sha256": pair_sha256,
                "expected_raw_result": str(result_path),
            }
        ],
    }


def test_launcher_defaults_to_plan_only_and_requires_run_flag(tmp_path):
    result, output_root = _run_plan(tmp_path)

    plan = json.loads(result.stdout)
    assert plan["schema"] == "robotwin_teacher_student_offline_plan_v1"
    assert plan["execution_requested"] is False
    assert not output_root.exists()
    assert "--run" in LAUNCHER.read_text(encoding="utf-8")
    assert "core2" in LAUNCHER.read_text(encoding="utf-8")
    assert "final12" in LAUNCHER.read_text(encoding="utf-8")


def test_launcher_plan_records_exact_live_teacher_video_command_and_provenance(tmp_path):
    result, output_root = _run_plan(tmp_path)

    plan = json.loads(result.stdout)
    assert len(plan["entries"]) == 1
    entry = plan["entries"][0]
    assert entry["checkpoint_name"] == "danceopd"
    assert entry["rollout_source"] == "target"
    assert entry["student_steps"] == entry["teacher_steps"] == 1
    assert entry["dataset_path"].endswith("lerobot_robotwin_eef_aug_500")
    assert entry["manifest_sha256"]
    assert entry["pair_sha256"]
    assert entry["checkpoint_inventory_sha256"]
    assert entry["expected_raw_result"] == str(output_root / "core2" / "danceopd" / "target" / "k1" / "result.json")
    command = entry["command"]
    assert "rollout_eval_video_stage2.py" in command
    assert "--emit-teacher-action-gt" in command
    assert "--rollout-student target" in command
    assert "--student-steps 1" in command
    assert "--teacher-steps 1" in command
    assert "RESUME_ONLINE_FROM_TARGET=1" in command
    assert "--video-dir" not in command
    assert "--teacher-cache-path" not in command


def test_launcher_rejects_non_protocol_dataset_and_existing_output_root(tmp_path):
    bad_dataset, _ = _run_plan(tmp_path, "--dataset-path", "/tmp/not-robotwin", check=False)
    assert bad_dataset.returncode != 0
    assert "dataset" in bad_dataset.stderr.lower()

    existing_root = tmp_path / "offline_matrix"
    existing_root.mkdir()
    existing, _ = _run_plan(tmp_path, check=False)
    assert existing.returncode != 0
    assert "fresh" in existing.stderr.lower()


def test_launcher_rejects_output_inside_existing_ablation_artifact_tree(tmp_path):
    protected_root = (
        "/root/nas/junjie/jj/Any_WAM/distillation_flowmap/"
        "output_robotwin_stepwam_ablation/forbidden_offline_matrix"
    )

    result, _ = _run_plan(tmp_path, "--output-root", protected_root, check=False)

    assert result.returncode != 0
    assert "immutable" in result.stderr.lower()


def test_summary_reports_negative_delta_as_student_better(tmp_path):
    from distillation_flowmap import summarize_robotwin_teacher_student as summary

    table = summary.summarize_metric_rows(
        [{"student": 0.2, "teacher": 0.3, "metric": "video_gt_x_mse"}]
    )

    assert table[0]["student_minus_teacher"] == pytest.approx(-0.1)
    assert table[0]["student_better"] is True


def test_summary_requires_all_teacher_student_action_video_pairs_and_writes_tables(tmp_path):
    from distillation_flowmap import summarize_robotwin_teacher_student as summary

    output_root = tmp_path / "matrix"
    result_path = output_root / "core2" / "danceopd" / "target" / "k1" / "result.json"
    payload = {
        "rollout_source_requested": "target",
        "rollout_source_resolved": "target",
        **_metric_payload(),
    }
    _write_json(result_path, payload)
    plan_path = output_root / "plan.json"
    _write_json(plan_path, _result_plan(output_root, result_path))

    out = output_root / "summary"
    rows = summary.summarize(plan_path, out)

    assert len(rows) == 7
    video_row = [row for row in rows if row["metric"] == "video_gt_x_mse"][0]
    assert video_row["student_minus_teacher"] == pytest.approx(-0.1)
    assert video_row["student_better"] is True
    with (out / "summary.csv").open("r", encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 7
    markdown = (out / "summary.md").read_text(encoding="utf-8")
    assert "student_minus_teacher" in markdown
    assert "negative means student better" in markdown


def test_summary_rejects_missing_teacher_action_key(tmp_path):
    from distillation_flowmap import summarize_robotwin_teacher_student as summary

    output_root = tmp_path / "matrix"
    result_path = output_root / "core2" / "danceopd" / "target" / "k1" / "result.json"
    payload = {
        "rollout_source_requested": "target",
        "rollout_source_resolved": "target",
        **_metric_payload(),
    }
    payload.pop("rollout_eval/t1000_r0_i0/s1_t1/action_teacher_gt_xr_mse")
    _write_json(result_path, payload)
    plan_path = output_root / "plan.json"
    _write_json(plan_path, _result_plan(output_root, result_path))

    with pytest.raises(ValueError, match="missing required metrics"):
        summary.summarize(plan_path, output_root / "summary")


def test_summary_rejects_result_missing_an_entire_fixed_pair(tmp_path):
    from distillation_flowmap import summarize_robotwin_teacher_student as summary

    output_root = tmp_path / "matrix"
    result_path = output_root / "core2" / "danceopd" / "target" / "k1" / "result.json"
    _write_json(
        result_path,
        {
            "rollout_source_requested": "target",
            "rollout_source_resolved": "target",
            **_metric_payload(pair_id="t1000_r0_i0"),
        },
    )
    plan_path = output_root / "plan.json"
    _write_json(
        plan_path,
        _result_plan(
            output_root,
            result_path,
            pair_ids=("t1000_r0_i0", "t1000_r500_i1"),
        ),
    )

    with pytest.raises(ValueError, match="missing fixed eval pair"):
        summary.summarize(plan_path, output_root / "summary")


def test_summary_rejects_result_with_unexpected_pair(tmp_path):
    from distillation_flowmap import summarize_robotwin_teacher_student as summary

    output_root = tmp_path / "matrix"
    result_path = output_root / "core2" / "danceopd" / "target" / "k1" / "result.json"
    _write_json(
        result_path,
        {
            "rollout_source_requested": "target",
            "rollout_source_resolved": "target",
            **_metric_payload(pair_id="t1000_r0_i0"),
            **_metric_payload(pair_id="unexpected_pair"),
        },
    )
    plan_path = output_root / "plan.json"
    _write_json(plan_path, _result_plan(output_root, result_path))

    with pytest.raises(ValueError, match="unexpected eval pair"):
        summary.summarize(plan_path, output_root / "summary")


def test_summary_rejects_pair_file_digest_mismatch(tmp_path):
    from distillation_flowmap import summarize_robotwin_teacher_student as summary

    output_root = tmp_path / "matrix"
    result_path = output_root / "core2" / "danceopd" / "target" / "k1" / "result.json"
    _write_json(
        result_path,
        {
            "rollout_source_requested": "target",
            "rollout_source_resolved": "target",
            **_metric_payload(),
        },
    )
    plan = _result_plan(output_root, result_path)
    plan["entries"][0]["pair_sha256"] = "0" * 64
    plan_path = output_root / "plan.json"
    _write_json(plan_path, plan)

    with pytest.raises(ValueError, match="pair SHA-256"):
        summary.summarize(plan_path, output_root / "summary")


def test_completed_core2_gate_requires_verified_matching_entry_coverage(tmp_path):
    from distillation_flowmap import summarize_robotwin_teacher_student as summary

    output_root = tmp_path / "core2_gate"
    result_path = output_root / "core2" / "danceopd" / "target" / "k1" / "result.json"
    _write_json(
        result_path,
        {
            "rollout_source_requested": "target",
            "rollout_source_resolved": "target",
            **_metric_payload(),
        },
    )
    gate_plan = _result_plan(output_root, result_path, execution_requested=True)
    gate_path = output_root / "plan.json"
    _write_json(gate_path, gate_plan)

    rows = summary.verify_completed_core2_gate(
        gate_path,
        required_entries={("danceopd", "target", 1)},
    )
    assert len(rows) == 7

    with pytest.raises(ValueError, match="does not cover required final12 entries"):
        summary.verify_completed_core2_gate(
            gate_path,
            required_entries={("full_stepwam", "target", 1)},
        )


def test_launcher_final12_run_requires_core2_gate_verifier():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "--core2-gate" in source
    assert "--verify-core2-gate" in source


def test_launcher_refuses_final12_execution_before_matching_core2_gate(tmp_path):
    result, output_root = _run_plan(
        tmp_path,
        "--run",
        "--split",
        "final12",
        check=False,
    )

    assert result.returncode != 0
    assert "--core2-gate" in result.stderr
    assert not output_root.exists()
