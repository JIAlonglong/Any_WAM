from pathlib import Path


def _parallel_train_source():
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "ablation" / "run_parallel_train.sh").read_text(encoding="utf-8")


def _final_train_source():
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "ablation" / "run_final_ablation.sh").read_text(encoding="utf-8")


def _final_eval_source():
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "ablation" / "run_final_eval.sh").read_text(encoding="utf-8")


def _opd_calibration_eval_source():
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "ablation" / "run_opd_mechanism_calibration_eval.sh").read_text(
        encoding="utf-8"
    )


def test_parallel_train_uses_protocol_controlled_shared_stage1():
    source = _parallel_train_source()

    assert 'VARIANTS="full_stepwam,w_o_opd,endpoint_only_opd,velocity_only_opd"' in source
    assert 'SEEDS="0,1,2"' in source
    assert 'TASK_PRESET="${TASK_PRESET:-core4}"' in source
    assert 'MAX_EPISODES_PER_TASK="${MAX_EPISODES_PER_TASK:-100}"' in source
    assert 'TRAIN_SAMPLES_PER_TASK="${TRAIN_SAMPLES_PER_TASK:-80}"' in source
    assert 'HELDOUT_SAMPLES_PER_TASK="${HELDOUT_SAMPLES_PER_TASK:-20}"' in source
    assert "--use-shared-stage1" in source
    assert 'launch_job stage1 full_stepwam "${seed}"' in source
    assert 'launch_job stage2 "${variant}" "${seed}"' in source
    assert '--stage "${phase}"' in source
    assert "--task-preset" in source
    assert "--train-samples-per-task" in source
    assert "--heldout-samples-per-task" in source


def test_parallel_train_records_batch_failures_before_exiting():
    source = _parallel_train_source()

    assert "failures=0" in source
    assert "wait_batch stage1" in source
    assert "wait_batch stage2" in source
    assert "set +e" in source
    assert "set -e" in source
    assert 'if [ "${stage1_failures}" -ne 0 ]; then' in source
    assert 'if [ "${failures}" -ne 0 ]; then' in source


def test_final_ablation_launcher_uses_representative_12_task_protocol():
    source = _final_train_source()

    assert 'TASK_PRESET="${TASK_PRESET:-representative}"' in source
    assert 'MAX_EPISODES_PER_TASK="${MAX_EPISODES_PER_TASK:-50}"' in source
    assert 'MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-50}"' in source
    assert 'TRAIN_SAMPLES_PER_TASK="${TRAIN_SAMPLES_PER_TASK:-40}"' in source
    assert 'HELDOUT_SAMPLES_PER_TASK="${HELDOUT_SAMPLES_PER_TASK:-10}"' in source
    assert 'STAGE1_STEPS="${STAGE1_STEPS:-5000}"' in source
    assert 'STAGE2_STEPS="${STAGE2_STEPS:-5000}"' in source
    assert 'SEEDS="${SEEDS:-0,1,2}"' in source
    assert "full_stepwam" in source
    assert "w_o_opd" in source
    assert "endpoint_only_opd" in source
    assert "velocity_only_opd" in source
    assert "local_adjacent_only" in source
    assert "action_only" in source
    assert 'run_parallel_train.sh' in source


def test_final_eval_launcher_reuses_heldout_cache_and_summarizes():
    source = _final_eval_source()

    assert 'ROOT="${ROOT:-${PROJECT_ROOT}/distillation_flowmap/output_robotwin_stepwam_ablation/protocol_final12_5000}"' in source
    assert 'VARIANTS="${VARIANTS:-full_stepwam,w_o_opd,endpoint_only_opd,velocity_only_opd,local_adjacent_only,action_only}"' in source
    assert 'SEEDS="${SEEDS:-0,1,2}"' in source
    assert 'TEACHER_CACHE="${TEACHER_CACHE:-${ROOT}/protocol/teacher_cache_heldout_equal_nfe.pt}"' in source
    assert "--teacher-cache-only" in source
    assert "--teacher-cache-path" in source
    assert "--disable-eval-gradient-checkpointing" in source
    assert "--eval-rollout-grad-mode" in source
    assert "--eval-empty-cache" in source
    assert "heldout_eval_manifest.json" in source
    assert "train_eval_manifest.json" in source
    assert "make_balanced_train_eval_manifest" in source
    assert "rollout_eval_stage2.py" in source
    assert "rollout_eval_video_stage2.py" in source
    assert "summarize_robotwin_ablation.py" in source


def test_final_eval_supports_calibrated_teacher_and_cache_source_overrides():
    source = _final_eval_source()

    assert 'STUDENT_STEPS="${STUDENT_STEPS:-1 2 4}"' in source
    assert 'TEACHER_STEPS="${TEACHER_STEPS:-1 2 4 8}"' in source
    assert 'TEACHER_CACHE_SOURCE_VARIANT="${TEACHER_CACHE_SOURCE_VARIANT:-full_stepwam}"' in source
    assert 'TEACHER_CACHE_SOURCE_SEED="${TEACHER_CACHE_SOURCE_SEED:-0}"' in source
    assert 'BASELINE_VARIANT="${BASELINE_VARIANT:-w_o_opd}"' in source
    assert 'stage2_ckpt "${TEACHER_CACHE_SOURCE_VARIANT}" "${TEACHER_CACHE_SOURCE_SEED}"' in source
    assert source.count('--teacher-steps "${teacher_step_list[@]}"') == 4
    assert 'read -r -a student_step_list <<< "${STUDENT_STEPS}"' in source
    assert source.count('--student-steps "${student_step_list[@]}"') == 4
    assert '--baseline-variant "${BASELINE_VARIANT}"' in source


def test_opd_calibration_eval_is_small_and_protocol_controlled():
    source = _opd_calibration_eval_source()

    assert 'STAGE2_STEPS="${STAGE2_STEPS:-750}"' in source
    assert 'TASK_PRESET="${TASK_PRESET:-core2}"' in source
    assert 'SEEDS="${SEEDS:-0}"' in source
    assert 'TEACHER_STEPS="${TEACHER_STEPS:-8}"' in source
    assert 'TEACHER_CACHE_SOURCE_VARIANT="${TEACHER_CACHE_SOURCE_VARIANT:-calib_w_o_opd}"' in source
    assert 'BASELINE_VARIANT="${BASELINE_VARIANT:-calib_w_o_opd}"' in source
    assert 'EVAL_GPUS="${EVAL_GPUS:-0,1,2,3,4}"' in source
    assert 'MAX_PARALLEL="${MAX_PARALLEL:-5}"' in source
    assert 'VARIANTS="${VARIANTS:-calib_w_o_opd,calib_endpoint_only,calib_velocity_only,calib_full_last_step,calib_full_suffix_grad}"' in source
    assert 'run_final_eval.sh' in source


def test_final_eval_runs_video_variants_in_bounded_parallel_batches():
    source = _final_eval_source()

    assert 'local active=0' in source
    assert 'wait_batch "video"' in source
    assert 'metrics/video_mse.log' in source
