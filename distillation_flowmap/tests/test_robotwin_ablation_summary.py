import csv
import json
from pathlib import Path


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_run(root, variant, seed, heldout_video_error, train_video_error=None):
    run_dir = root / variant / f"seed_{seed}"
    manifest = {
        "variant": variant,
        "seed": seed,
        "run_dir": str(run_dir),
        "git_hash": f"hash{seed}",
        "stage1_ckpt": str(run_dir / "stage1" / "checkpoints" / "step_10"),
        "stage2_ckpt": str(run_dir / "stage2" / "checkpoints" / "step_20"),
        "selected_task_filter": ["place_a2b_right", "open_microwave"],
        "task_list": {"episodes_per_task": 50, "student_steps": 4},
        "protocol_seed": 7,
        "train_manifest_path": str(root / "protocol" / "train.json"),
        "heldout_eval_manifest_path": str(root / "protocol" / "heldout.json"),
        "eval_pairs_path": str(root / "protocol" / "pairs.json"),
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    offline_metric = "rollout_eval/t1000_r0_i0/s4_t4/video_teacher_x_mse"
    action_metric = "rollout_eval/t1000_r0_i0/s4_t4/action_gt_xr_mse"
    _write_json(
        run_dir / "metrics" / "offline_rollout.json",
        {
            offline_metric: heldout_video_error,
            action_metric: heldout_video_error / 10.0,
            "per_task": {
                "place_a2b_right": {
                    offline_metric: heldout_video_error * 0.5,
                    action_metric: heldout_video_error / 20.0,
                },
                "open_microwave": {
                    offline_metric: heldout_video_error * 1.5,
                    action_metric: heldout_video_error * 3.0 / 20.0,
                },
            },
        },
    )
    _write_json(
        run_dir / "metrics" / "video_mse.json",
        {
            offline_metric: heldout_video_error,
            "per_task": {
                "place_a2b_right": {offline_metric: heldout_video_error * 0.5},
                "open_microwave": {offline_metric: heldout_video_error * 1.5},
            },
        },
    )
    if train_video_error is not None:
        _write_json(
            run_dir / "metrics" / "train_rollout.json",
            {
                "rollout_eval/t1000_r0_i0/s4_t4/video_teacher_x_mse": train_video_error,
            },
        )
    video_dir = run_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "t1000_r0_i0_student_s4.mp4").write_bytes(b"fake")
    (video_dir / "t1000_r0_i0_contact_sheet.png").write_bytes(b"fake")
    return run_dir


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_robotwin_ablation_summary_writes_protocol_outputs(tmp_path):
    from distillation_flowmap.ablation import summarize_robotwin_ablation as summary

    root = tmp_path / "runs"
    out = tmp_path / "report"
    _write_run(root, "full_stepwam", 0, heldout_video_error=0.40, train_video_error=0.25)
    _write_run(root, "full_stepwam", 1, heldout_video_error=0.60, train_video_error=0.30)
    _write_run(root, "w_o_opd", 0, heldout_video_error=0.80, train_video_error=0.70)

    summary.summarize(root=root, out=out, baseline_variant="w_o_opd")

    per_seed = _read_csv(out / "mini_ablation_per_seed.csv")
    assert len(per_seed) == 3
    assert "heldout/offline_rollout/rollout_eval/t1000_r0_i0/s4_t4/video_teacher_x_mse" in per_seed[0]
    assert "gap/train_minus_heldout/rollout_eval/t1000_r0_i0/s4_t4/video_teacher_x_mse" in per_seed[0]

    summary_rows = _read_csv(out / "mini_ablation_summary.csv")
    target = [
        row for row in summary_rows
        if row["variant"] == "full_stepwam"
        and row["metric"] == "heldout/offline_rollout/rollout_eval/t1000_r0_i0/s4_t4/video_teacher_x_mse"
    ][0]
    assert target["n"] == "2"
    assert abs(float(target["mean"]) - 0.50) < 1e-9
    assert abs(float(target["std"]) - 0.14142135623730948) < 1e-9
    assert abs(float(target["baseline_mean"]) - 0.80) < 1e-9
    assert abs(float(target["delta_vs_w_o_opd"]) + 0.30) < 1e-9

    per_task = _read_csv(out / "mini_ablation_per_task.csv")
    assert {row["task"] for row in per_task} == {"place_a2b_right", "open_microwave"}
    task_metric = "heldout/offline_rollout/rollout_eval/t1000_r0_i0/s4_t4/video_teacher_x_mse"
    full_task_rows = {
        row["task"]: row
        for row in per_task
        if row["variant"] == "full_stepwam" and row["seed"] == "0"
    }
    assert float(full_task_rows["place_a2b_right"][task_metric]) == 0.20
    assert abs(float(full_task_rows["open_microwave"][task_metric]) - 0.60) < 1e-12
    assert float(full_task_rows["place_a2b_right"][task_metric]) != float(
        full_task_rows["open_microwave"][task_metric]
    )

    assets = [json.loads(line) for line in (out / "mini_ablation_video_assets.jsonl").read_text().splitlines()]
    assert {asset["kind"] for asset in assets} == {"mp4", "contact_sheet"}

    report = (out / "mini_ablation_report.md").read_text(encoding="utf-8")
    assert "Trend only" in report
    assert "delta_vs_w_o_opd" in report
