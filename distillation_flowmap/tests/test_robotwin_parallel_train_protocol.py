from pathlib import Path


def _parallel_train_source():
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "ablation" / "run_parallel_train.sh").read_text(encoding="utf-8")


def _final_train_source():
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "ablation" / "run_final_ablation.sh").read_text(encoding="utf-8")


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
