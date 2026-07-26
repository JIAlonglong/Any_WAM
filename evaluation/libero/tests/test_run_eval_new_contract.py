import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "evaluation" / "libero" / "run_eval_new.sh"


def _env(tmp_path, suite):
    output = tmp_path / "output"
    student = output / "checkpoints" / "step_1" / "target_student" / "transformer"
    student.mkdir(parents=True)
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "OUTPUT_ROOT": str(output),
            "TEACHER_CKPT": str(teacher),
            "LIBERO_BENCHMARK": suite,
            "CHECK_ONLY": "1",
            "EVAL_MODE": "success",
            "TEST_NUM": "2",
            "TASK_START": "0",
            "TASK_END": "5",
            "NUM_STEPS": "2",
            "ACTION_NUM_STEPS": "2",
            "CUDA_VISIBLE_DEVICES": "7",
            "SAVE_ROOT": str(tmp_path / "results"),
        }
    )
    return env


@pytest.mark.parametrize(
    "suite", ["libero_10", "libero_spatial", "libero_object", "libero_goal"]
)
def test_check_only_propagates_suite_joint_steps_and_gpu(tmp_path, suite):
    result = subprocess.run(
        ["bash", str(SCRIPT), "step_1", "target_student"],
        cwd=ROOT,
        env=_env(tmp_path, suite),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Benchmark:      {suite}" in result.stdout
    assert "Video steps:    2" in result.stdout
    assert "Action steps:   2" in result.stdout
    assert "Visible GPUs:   7" in result.stdout
    assert "Base model:" in result.stdout


def test_unknown_suite_is_rejected(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT), "step_1", "target_student"],
        cwd=ROOT,
        env=_env(tmp_path, "libero_unknown"),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Unsupported LIBERO_BENCHMARK" in result.stderr


def test_eval_contract_wires_model_and_sampler_latency_output(tmp_path):
    env = _env(tmp_path, "libero_10")
    env["MODEL_NAME"] = "stage2"
    env["LATENCY_JSONL"] = str(tmp_path / "latency" / "sampler_latency.jsonl")

    result = subprocess.run(
        ["bash", str(SCRIPT), "step_1", "target_student"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Model name:     stage2" in result.stdout
    assert f"Latency JSONL:  {env['LATENCY_JSONL']}" in result.stdout


def test_flowmap_backend_remains_default(tmp_path):
    env = _env(tmp_path, "libero_10")
    result = subprocess.run(
        ["bash", str(SCRIPT), "step_1", "target_student"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Server backend: flowmap" in result.stdout


def test_flowmap_backend_keeps_default_sampling_budgets(tmp_path):
    env = _env(tmp_path, "libero_10")
    env.pop("NUM_STEPS")
    env.pop("ACTION_NUM_STEPS")
    result = subprocess.run(
        ["bash", str(SCRIPT), "step_1", "target_student"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Video steps:    20" in result.stdout
    assert "Action steps:   50" in result.stdout


def test_native_teacher_backend_routes_to_dedicated_server(tmp_path):
    env = _env(tmp_path, "libero_10")
    (
        Path(env["OUTPUT_ROOT"])
        / "checkpoints"
        / "external"
        / "teacher_native"
        / "transformer"
    ).mkdir(parents=True)
    env.update(
        {
            "SERVER_BACKEND": "native_teacher",
            "MODEL_NAME": "teacher_native",
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "external", "teacher_native"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Server backend: native_teacher" in result.stdout
    assert "wan_va/wan_va_native_teacher_server.py" in result.stdout


def test_native_teacher_defaults_action_steps_to_video_steps(tmp_path):
    env = _env(tmp_path, "libero_10")
    env.pop("ACTION_NUM_STEPS")
    env.update(
        {
            "SERVER_BACKEND": "native_teacher",
            "MODEL_NAME": "teacher_native",
            "NUM_STEPS": "4",
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "step_1", "target_student"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Video steps:    4" in result.stdout
    assert "Action steps:   4" in result.stdout


def test_native_teacher_start_server_invokes_dedicated_entrypoint(tmp_path):
    env = _env(tmp_path, "libero_10")
    base_model = tmp_path / "base_model"
    for component in ("transformer", "vae", "tokenizer", "text_encoder"):
        (base_model / component).mkdir(parents=True)

    server_stub = tmp_path / "server_stub.sh"
    server_stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$SERVER_CAPTURE"\n')
    server_stub.chmod(0o755)
    client_stub = tmp_path / "client_stub.sh"
    client_stub.write_text("#!/bin/sh\nexit 0\n")
    client_stub.chmod(0o755)
    server_capture = tmp_path / "server_args.txt"

    env.pop("ACTION_NUM_STEPS")
    env.update(
        {
            "CHECK_ONLY": "0",
            "EVAL_MODE": "success",
            "SERVER_BACKEND": "native_teacher",
            "MODEL_NAME": "teacher_native",
            "NUM_STEPS": "4",
            "WAN22_PRETRAINED_PATH": str(base_model),
            "SERVER_PYTHON": str(server_stub),
            "CLIENT_PYTHON": str(client_stub),
            "SERVER_CAPTURE": str(server_capture),
            "SERVER_WAIT_SECONDS": "0",
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "step_1", "target_student"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    server_args = server_capture.read_text()
    assert "wan_va/wan_va_native_teacher_server.py" in server_args
    assert "--config-name\nlibero" in server_args
    assert "--port\n29057" in server_args
    expected_checkpoint = (
        Path(env["OUTPUT_ROOT"])
        / "checkpoints"
        / "step_1"
        / "target_student"
        / "transformer"
    )
    assert f"--checkpoint-path\n{expected_checkpoint}" in server_args
    assert "--num-steps\n4" in server_args
    assert "--action-num-steps\n4" in server_args
    assert "--model-name\nteacher_native" in server_args
    assert f"--save-root\n{Path(env['SAVE_ROOT']) / 'actions'}" in server_args


def test_unknown_server_backend_is_rejected(tmp_path):
    env = _env(tmp_path, "libero_10")
    env["SERVER_BACKEND"] = "unknown"
    result = subprocess.run(
        ["bash", str(SCRIPT), "step_1", "target_student"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Unsupported SERVER_BACKEND" in result.stderr


def test_client_and_server_preserve_episode_latency_provenance():
    client_source = (ROOT / "evaluation" / "libero" / "client.py").read_text()
    server_source = (ROOT / "wan_va" / "wan_va_server.py").read_text()

    assert '"eval_metadata": {' in client_source
    assert '"suite": libero_benchmark' in client_source
    assert '"task_idx": int(task_idx)' in client_source
    assert '"episode_idx": int(episode_idx)' in client_source
    assert "started_at = time.perf_counter()" in server_source
    assert "append_sampler_latency_record" in server_source
