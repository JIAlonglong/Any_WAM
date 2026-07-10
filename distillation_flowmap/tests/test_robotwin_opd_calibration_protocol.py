import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    REPO_ROOT
    / "distillation_flowmap"
    / "ablation"
    / "run_opd_mechanism_calibration.sh"
)


def test_calibration_runner_has_valid_shell_syntax():
    result = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr


def test_calibration_runner_defaults_to_small_stage2_only_protocol():
    source = RUNNER.read_text(encoding="utf-8")

    assert 'TASK_PRESET="${TASK_PRESET:-core2}"' in source
    assert 'STAGE2_STEPS="${STAGE2_STEPS:-750}"' in source
    assert 'TRAIN_SAMPLES_PER_TASK="${TRAIN_SAMPLES_PER_TASK:-20}"' in source
    assert 'HELDOUT_SAMPLES_PER_TASK="${HELDOUT_SAMPLES_PER_TASK:-10}"' in source
    assert 'MAX_EPISODES_PER_TASK="${MAX_EPISODES_PER_TASK:-30}"' in source
    assert 'MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-30}"' in source
    assert '--stage "stage2"' in source
    assert '--stage1-ckpt "${SHARED_STAGE1_CKPT}"' in source
    assert "--stage1-steps" not in source


def test_calibration_runner_launches_only_mechanism_variants():
    source = RUNNER.read_text(encoding="utf-8")

    assert (
        'VARIANTS="${VARIANTS:-calib_w_o_opd,calib_endpoint_only,'
        'calib_velocity_only,calib_full_last_step,calib_full_full_grad}"'
        in source
    )
    assert 'SEED="${SEED:-0}"' in source
    assert 'MAX_PARALLEL="${MAX_PARALLEL:-5}"' in source


def test_calibration_runner_requires_explicit_opt_in_to_expand_protocol():
    source = RUNNER.read_text(encoding="utf-8")

    assert 'ALLOW_LARGE_CALIBRATION="${ALLOW_LARGE_CALIBRATION:-0}"' in source
    assert "Refusing expanded calibration" in source
    assert "STAGE2_STEPS > 1000" in source
