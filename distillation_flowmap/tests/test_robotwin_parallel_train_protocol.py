from pathlib import Path


def _parallel_train_source():
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "ablation" / "run_parallel_train.sh").read_text(encoding="utf-8")


def test_parallel_train_uses_protocol_controlled_shared_stage1():
    source = _parallel_train_source()

    assert 'VARIANTS="full_stepwam,w_o_opd,endpoint_only_opd,velocity_only_opd"' in source
    assert 'SEEDS="0,1,2"' in source
    assert 'TASK_PRESET="${TASK_PRESET:-core4}"' in source
    assert 'TRAIN_SAMPLES_PER_TASK="${TRAIN_SAMPLES_PER_TASK:-80}"' in source
    assert 'HELDOUT_SAMPLES_PER_TASK="${HELDOUT_SAMPLES_PER_TASK:-20}"' in source
    assert "--use-shared-stage1" in source
    assert 'launch_job stage1 full_stepwam "${seed}"' in source
    assert 'launch_job stage2 "${variant}" "${seed}"' in source
    assert '--stage "${phase}"' in source
    assert "--task-preset" in source
    assert "--train-samples-per-task" in source
    assert "--heldout-samples-per-task" in source
