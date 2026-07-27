import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "distillation_flowmap" / "run_cosmos_stage2_student_eval_8gpu.sh"
AUDITED_COSMOS_REPO = Path(
    os.environ.get(
        "COSMOS_AUDITED_REPO_SOURCE",
        "/kpfs-intern/jialongliu/projects/cosmos-predict2.5-formal-compat-441b897",
    )
)
RAW_CONTRACT = {
    "contract_version": 2,
    "training_contract_stage": "raw_stage1",
    "action_packing_schema": "downsample_survivor_v2",
    "action_downsample_factor": 4,
    "action_chunk_shape": [4, 4],
    "teacher_backend": "cosmos_policy",
    "student_backend": "wan_flowmap",
}
STAGE2_CONTRACT = {
    "contract_version": 2,
    "training_contract_stage": "progressive_stage2",
    "action_packing_schema": "downsample_survivor_v2",
    "action_downsample_factor": 4,
    "action_chunk_shape": [4, 4],
    "deployment_timestep_start": 1000,
    "deployment_timestep_end": 0,
    "joint_student_steps": [1, 2, 4],
    "deployment_joint_rollout_interval": 4,
    "deployment_action_weight": 1.0,
    "raw_teacher_window_is_auxiliary": True,
    "teacher_backend": "cosmos_policy",
    "student_backend": "wan_flowmap",
}


def _executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _plain(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _transformer(path: Path, payload: dict[str, object]) -> Path:
    _plain(path / "config.json", (json.dumps(payload) + "\n").encode())
    _plain(path / "diffusion_pytorch_model.safetensors", b"weights")
    return path


def _stage2_checkpoint(
    output_root: Path,
    *,
    stage1_checkpoint: Path,
    wan_base: Path,
    teacher: Path,
    step: int = 5000,
) -> Path:
    from distillation_flowmap.cosmos_stage2_lineage import validate_stage1_parent

    parent = validate_stage1_parent(stage1_checkpoint, expected_step=3000)
    checkpoint = (
        output_root
        / "student-eval"
        / "universal-video-action"
        / "checkpoints"
        / f"step_{step}"
    )
    payload = {
        **STAGE2_CONTRACT,
        "checkpoint_step": step,
        "teacher_model_path": str(teacher.resolve()),
        "parent_stage1_path": parent.canonical_path,
        "parent_stage1_contract_identity": parent.contract_identity,
        "parent_stage1_expected_step": 3000,
        "student_base_model_path": str(stage1_checkpoint / "target_student"),
        "wan_student_base_model_path": str(wan_base.resolve()),
    }
    for variant in ("online_student", "target_student"):
        _transformer(checkpoint / variant / "transformer", payload)
    _plain(checkpoint / "optimizer.pt")
    _plain(checkpoint / "lr_scheduler.pt")
    return checkpoint / "target_student" / "transformer"


def _audited_cosmos_repo(tmp_path: Path) -> Path:
    del tmp_path
    return AUDITED_COSMOS_REPO


def _sentinel(path: Path, kind: str) -> Path:
    return _executable(
        path,
        "python3 - \"$@\" <<'PY'\n"
        "import json, os, pathlib, sys\n"
        "record = {'kind': " + repr(kind) + ", 'argv': sys.argv[1:]}\n"
        "for name in ('COSMOS_STAGE1_EXPECTED_STEP', 'MATRIX_ROOT', 'S4_CKPT_ROOT', "
        "'S4_MATRIX_ROLES', 'S4_FORMAL_NUM_SHARDS', 'S4_FORMAL_GPU_LAYOUT', "
        "'S4_VIDEO_SEEDS', 'S4_EPISODES_PER_TASK', 'CUDA_VISIBLE_DEVICES', "
        "'PYTORCH_CUDA_ALLOC_CONF', 'PYTHON_BIN', 'WAN_STUDENT_BASE_MODEL_PATH', "
        "'DATASET_PATH', 'COSMOS_PREDICT2_REPO', 'COSMOS_PREDICT2_REPO_COMMIT', "
        "'COSMOS_POLICY_PYTHON', 'COSMOS_POLICY_EXTRA_PYTHONPATH', "
        "'COSMOS_WORKER_ENV_ROOT', 'COSMOS_WORKER_SITE_PACKAGES', "
        "'COSMOS_WORKER_CUDA_LIBRARY_PATH', 'LD_LIBRARY_PATH', "
        "'PARENT_STAGE1_PATH', 'PARENT_STAGE1_CONTRACT_IDENTITY', "
        "'STAGE2_LINEAGE_JSON'):\n"
        "    if name in os.environ: record[name.lower()] = os.environ[name]\n"
        "if '--dry-run' in sys.argv[1:]:\n"
        "    print('SENTINEL_' + record['kind'].upper() + '_DRY_RUN=1')\n"
        "    raise SystemExit(0)\n"
        "if record['kind'] == 'stage2':\n"
        "    record['parent_expected_step'] = os.environ['COSMOS_STAGE1_EXPECTED_STEP']\n"
        "    record['stage2_steps'] = sys.argv[sys.argv.index('--steps') + 1]\n"
        "    checkpoint = pathlib.Path(os.environ['PIPELINE_RUN_ROOT']) / 'universal-video-action' / 'checkpoints' / ('step_' + record['stage2_steps'])\n"
        "    payload = {'contract_version': 2, 'training_contract_stage': 'progressive_stage2', 'action_packing_schema': 'downsample_survivor_v2', 'action_downsample_factor': 4, 'action_chunk_shape': [4, 4], 'deployment_timestep_start': 1000, 'deployment_timestep_end': 0, 'joint_student_steps': [1, 2, 4], 'deployment_joint_rollout_interval': 4, 'deployment_action_weight': 1.0, 'raw_teacher_window_is_auxiliary': True, 'teacher_backend': 'cosmos_policy', 'student_backend': 'wan_flowmap', 'checkpoint_step': int(record['stage2_steps']), 'teacher_model_path': os.environ['COSMOS_POLICY_PATH'], 'parent_stage1_path': os.environ['PARENT_STAGE1_PATH'], 'parent_stage1_contract_identity': os.environ['PARENT_STAGE1_CONTRACT_IDENTITY'], 'parent_stage1_expected_step': int(os.environ['COSMOS_STAGE1_EXPECTED_STEP']), 'student_base_model_path': os.environ['STUDENT_BASE_MODEL_PATH'], 'wan_student_base_model_path': os.environ['WAN_STUDENT_BASE_MODEL_PATH']}\n"
        "    for variant in ('online_student', 'target_student'):\n"
        "        root = checkpoint / variant / 'transformer'; root.mkdir(parents=True); (root / 'config.json').write_text(json.dumps(payload) + '\\n'); (root / 'diffusion_pytorch_model.safetensors').write_bytes(b'x')\n"
        "    (checkpoint / 'optimizer.pt').write_bytes(b'x'); (checkpoint / 'lr_scheduler.pt').write_bytes(b'x')\n"
        "elif record['kind'] == 'provenance':\n"
        "    root = pathlib.Path(sys.argv[sys.argv.index('--output-root') + 1]); root.mkdir(parents=True)\n"
        "elif record['kind'] == 'prompt':\n"
        "    if '--validate-only' in sys.argv[1:]:\n"
        "        print('SENTINEL_PROMPT_VALIDATE_ONLY=1')\n"
        "    else:\n"
        "        output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1]); output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(b'prompt'); record['output'] = str(output)\n"
        "elif record['kind'] == 'eval':\n"
        "    record['roles'] = os.environ['S4_MATRIX_ROLES']\n"
        "    if sys.argv[1:] == ['dry-run']:\n"
        "        for k in (1, 2, 4):\n"
        "            for suite in ('libero_10', 'libero_spatial', 'libero_object', 'libero_goal'):\n"
        "                print(f'MATRIX_STEP={k} MATRIX_SUITE={suite} PREFLIGHT_SHARD=0 PREFLIGHT_SHARD=1 PREFLIGHT_SHARD=2 PREFLIGHT_SHARD=3 SHARD_0_TASK_RANGE=0,3 SHARD_3_TASK_RANGE=8,10 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7')\n"
        "        raise SystemExit(0)\n"
        "with pathlib.Path(os.environ['CALLS_LOG']).open('a', encoding='utf-8') as handle: handle.write(json.dumps(record) + '\\n')\n"
        "PY\n",
    )


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    run_root = tmp_path / "outputs"
    wan = tmp_path / "wan"
    _plain(
        wan / "transformer" / "config.json",
        b'{"_class_name":"WanTransformer3DModel"}\n',
    )
    _plain(wan / "transformer" / "diffusion_pytorch_model.safetensors")
    dataset = tmp_path / "dataset"
    _plain(dataset / "empty_emb.pt")
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    _plain(teacher / "config.json", b'{"model_type":"cosmos-policy"}\n')
    for name in (
        "Cosmos-Policy-LIBERO-Predict2-2B.pt",
        "libero_dataset_statistics.json",
        "libero_t5_embeddings.pkl",
    ):
        _plain(teacher / name)
    stage1 = tmp_path / "stage1"
    checkpoint = stage1 / "checkpoints" / "step_3000"
    raw_payload = {
        **RAW_CONTRACT,
        "checkpoint_step": 3000,
        "student_base_model_path": str(wan.resolve()),
        "teacher_model_path": str(teacher.resolve()),
    }
    for variant in ("online_student", "target_student"):
        _transformer(checkpoint / variant / "transformer", raw_payload)
    local_model = tmp_path / "local-model"; local_model.mkdir()
    worker_root = tmp_path / "cosmos-worker"
    worker_python = _executable(
        worker_root / "bin" / "python",
        "if [[ \"${1:-}\" == \"-c\" ]]; then\n"
        "  printf 'COSMOS_WORKER_RUNTIME_OK=12.8\\n'\n"
        "  exit 0\n"
        "fi\n"
        f"exec {str(Path(sys.executable).resolve())!r} \"$@\"\n",
    )
    worker_site = worker_root / "lib/python3.10/site-packages"
    cuda_dirs = []
    for package in (
        "cublas",
        "cuda_cupti",
        "cuda_nvrtc",
        "cuda_runtime",
        "cudnn",
        "cufft",
        "cufile",
        "curand",
        "cusolver",
        "cusparse",
        "nccl",
        "nvjitlink",
        "nvtx",
    ):
        cuda_dir = worker_site / "nvidia" / package / "lib"
        cuda_dir.mkdir(parents=True)
        cuda_dirs.append(str(cuda_dir))
    cosmos_cuda = tmp_path / "cosmos-pythonpath" / "cosmos-cuda"
    cosmos_oss = tmp_path / "cosmos-pythonpath" / "cosmos-oss"
    cosmos_cuda.mkdir(parents=True)
    cosmos_oss.mkdir(parents=True)
    cosmos_repo = _audited_cosmos_repo(tmp_path)
    calls = tmp_path / "calls.jsonl"
    env = os.environ.copy()
    env.update({
        "PYTHON_BIN": sys.executable,
        "WAN_STUDENT_BASE_MODEL_PATH": str(wan),
        "DATASET_PATH": str(dataset),
        "COSMOS_POLICY_PATH": str(teacher),
        "COSMOS_PREDICT25_LOCAL_MODEL_DIR": str(local_model),
        "COSMOS_PREDICT2_REPO": str(cosmos_repo),
        "COSMOS_POLICY_PYTHON": str(worker_python),
        "COSMOS_POLICY_EXTRA_PYTHONPATH": f"{cosmos_cuda}:{cosmos_oss}",
        "COSMOS_WORKER_ENV_ROOT": str(worker_root),
        "COSMOS_WORKER_SITE_PACKAGES": str(worker_site),
        "COSMOS_WORKER_CUDA_LIBRARY_PATH": ":".join(cuda_dirs),
        "COSMOS_LOCK_PREPARER": str(_sentinel(tmp_path / "provenance.sh", "provenance")),
        "COSMOS_STAGE2_LAUNCHER": str(_sentinel(tmp_path / "stage2.sh", "stage2")),
        "COSMOS_WAN_PROMPT_TABLE_BUILDER": str(_sentinel(tmp_path / "prompt.sh", "prompt")),
        "COSMOS_JOINT124_EVAL_LAUNCHER": str(_sentinel(tmp_path / "eval.sh", "eval")),
        "CALLS_LOG": str(calls),
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    })
    return env, run_root


def run_wrapper(tmp_path: Path, *, phase: str = "all", create_parent: bool = True):
    env, output_root = _environment(tmp_path)
    stage1_root = tmp_path / "stage1"
    if not create_parent:
        stage1_root = tmp_path / "missing-stage1"
    result = subprocess.run(
        ["bash", str(SCRIPT), "--phase", phase, "--stage1-root", str(stage1_root),
         "--output-root", str(output_root), "--run-tag", "student-eval"],
        cwd=ROOT, text=True, capture_output=True, env=env, check=False,
    )
    calls = []
    log = Path(env["CALLS_LOG"])
    if log.exists():
        calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    return result, calls


def _invoke(
    tmp_path: Path,
    env: dict[str, str],
    output_root: Path,
    *extra: str,
):
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--stage1-root",
            str(tmp_path / "stage1"),
            "--output-root",
            str(output_root),
            "--run-tag",
            "student-eval",
            *extra,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_all_phase_runs_stage2_then_student_only_matrix(tmp_path):
    result, calls = run_wrapper(tmp_path, phase="all")
    assert result.returncode == 0, result.stderr
    assert [call["kind"] for call in calls] == ["provenance", "stage2", "prompt", "eval"]
    assert calls[1]["parent_expected_step"] == "3000"
    assert calls[1]["stage2_steps"] == "5000"
    assert calls[2]["output"].endswith("libero_wan_prompt_embeddings_all40.pt")
    assert calls[3]["roles"] == "stage2_target"
    assert calls[1]["pytorch_cuda_alloc_conf"] == "max_split_size_mb:128"
    assert calls[3]["pytorch_cuda_alloc_conf"] == "max_split_size_mb:128"


def test_eval_uses_four_paired_shards_and_all_eight_gpus(tmp_path):
    env, output_root = _environment(tmp_path)
    _stage2_checkpoint(
        output_root,
        stage1_checkpoint=tmp_path / "stage1" / "checkpoints" / "step_3000",
        wan_base=Path(env["WAN_STUDENT_BASE_MODEL_PATH"]),
        teacher=Path(env["COSMOS_POLICY_PATH"]),
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "--phase", "eval", "--stage1-root", str(tmp_path / "stage1"), "--output-root", str(output_root), "--run-tag", "student-eval"],
        cwd=ROOT, text=True, capture_output=True, env=env, check=False,
    )
    calls = [json.loads(line) for line in Path(env["CALLS_LOG"]).read_text(encoding="utf-8").splitlines()]
    evaluation = calls[-1]
    assert result.returncode == 0, result.stderr
    assert evaluation["s4_formal_num_shards"] == "4"
    assert evaluation["s4_formal_gpu_layout"] == "paired"
    assert evaluation["cuda_visible_devices"] == "0,1,2,3,4,5,6,7"
    assert evaluation["s4_video_seeds"] == "0"


def test_wrapper_rejects_missing_stage1_step_3000(tmp_path):
    result, calls = run_wrapper(tmp_path, create_parent=False)
    assert result.returncode != 0
    assert "Stage-1" in result.stderr
    assert calls == []


def test_wrapper_rejects_existing_fresh_stage2_or_eval_root(tmp_path):
    env, output_root = _environment(tmp_path)
    stage2 = output_root / "student-eval" / "universal-video-action"
    stage2.mkdir(parents=True)
    result = subprocess.run(["bash", str(SCRIPT), "--phase", "stage2", "--stage1-root", str(tmp_path / "stage1"), "--output-root", str(output_root), "--run-tag", "student-eval"], cwd=ROOT, text=True, capture_output=True, env=env, check=False)
    assert result.returncode != 0 and not Path(env["CALLS_LOG"]).exists()
    stage2.rmdir()
    matrix = output_root / "student-eval" / "eval" / "student_only"; matrix.mkdir(parents=True)
    _stage2_checkpoint(
        output_root,
        stage1_checkpoint=tmp_path / "stage1" / "checkpoints" / "step_3000",
        wan_base=Path(env["WAN_STUDENT_BASE_MODEL_PATH"]),
        teacher=Path(env["COSMOS_POLICY_PATH"]),
    )
    result = subprocess.run(["bash", str(SCRIPT), "--phase", "eval", "--stage1-root", str(tmp_path / "stage1"), "--output-root", str(output_root), "--run-tag", "student-eval"], cwd=ROOT, text=True, capture_output=True, env=env, check=False)
    assert result.returncode != 0 and not Path(env["CALLS_LOG"]).exists()


def test_check_only_prints_full_plan_without_writes(tmp_path):
    env, output_root = _environment(tmp_path)
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    result = subprocess.run(["bash", str(SCRIPT), "--phase", "check", "--stage1-root", str(tmp_path / "stage1"), "--output-root", str(output_root), "--run-tag", "student-eval"], cwd=ROOT, text=True, capture_output=True, env=env, check=False)
    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert result.returncode == 0, result.stderr
    assert before == after
    assert "STAGE2_STEPS=5000" in result.stdout
    assert "MAX_TRAIN_STEPS=5000" in result.stdout
    assert "\nS4_MATRIX_ROLES=stage2_target\n" in result.stdout
    assert "stage2_target" in result.stdout
    assert "K=1,2,4" in result.stdout
    assert result.stdout.count("MATRIX_STEP=") == 12
    assert result.stdout.count("PREFLIGHT_SHARD=3") == 12
    assert "SHARD_0_TASK_RANGE=0,3" in result.stdout
    assert "SHARD_3_TASK_RANGE=8,10" in result.stdout


def test_dry_run_expands_the_matrix_without_writes(tmp_path):
    env, output_root = _environment(tmp_path)
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    result = subprocess.run(["bash", str(SCRIPT), "--dry-run", "--stage1-root", str(tmp_path / "stage1"), "--output-root", str(output_root), "--run-tag", "student-eval"], cwd=ROOT, text=True, capture_output=True, env=env, check=False)
    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert result.returncode == 0, result.stderr
    assert before == after
    assert result.stdout.count("MATRIX_STEP=") == 12


def test_wrapper_passes_its_default_python_to_matrix_without_ambient_override(tmp_path):
    env, output_root = _environment(tmp_path)
    env.pop("PYTHON_BIN")
    _stage2_checkpoint(
        output_root,
        stage1_checkpoint=tmp_path / "stage1" / "checkpoints" / "step_3000",
        wan_base=Path(env["WAN_STUDENT_BASE_MODEL_PATH"]),
        teacher=Path(env["COSMOS_POLICY_PATH"]),
    )
    result = subprocess.run(["bash", str(SCRIPT), "--phase", "eval", "--stage1-root", str(tmp_path / "stage1"), "--output-root", str(output_root), "--run-tag", "student-eval"], cwd=ROOT, text=True, capture_output=True, env=env, check=False)
    calls = [json.loads(line) for line in Path(env["CALLS_LOG"]).read_text(encoding="utf-8").splitlines()]
    assert result.returncode == 0, result.stderr
    assert calls[-1]["python_bin"] == "/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python"


def test_clean_shell_derives_wan_and_uses_shared_dataset_default(tmp_path):
    env, output_root = _environment(tmp_path)
    env.pop("WAN_STUDENT_BASE_MODEL_PATH")
    env.pop("DATASET_PATH")

    result = _invoke(tmp_path, env, output_root, "--phase", "all")

    assert result.returncode == 0, result.stderr
    calls = [
        json.loads(line)
        for line in Path(env["CALLS_LOG"]).read_text(encoding="utf-8").splitlines()
    ]
    stage2 = next(call for call in calls if call["kind"] == "stage2")
    evaluation = next(call for call in calls if call["kind"] == "eval")
    expected_wan = str((tmp_path / "wan").resolve())
    expected_dataset = (
        "/kpfs-intern/jialongliu/projects/Flash-WAM/"
        "training_data/libero-long-lerobot"
    )
    assert stage2["wan_student_base_model_path"] == expected_wan
    assert stage2["dataset_path"] == expected_dataset
    assert evaluation["wan_student_base_model_path"] == expected_wan


def test_complete_worker_runtime_is_resolved_once_and_passed_to_both_children(
    tmp_path,
):
    env, output_root = _environment(tmp_path)

    result = _invoke(tmp_path, env, output_root, "--phase", "all")

    assert result.returncode == 0, result.stderr
    calls = [
        json.loads(line)
        for line in Path(env["CALLS_LOG"]).read_text(encoding="utf-8").splitlines()
    ]
    stage2 = next(call for call in calls if call["kind"] == "stage2")
    evaluation = next(call for call in calls if call["kind"] == "eval")
    for name in (
        "cosmos_predict2_repo",
        "cosmos_predict2_repo_commit",
        "cosmos_policy_python",
        "cosmos_policy_extra_pythonpath",
        "cosmos_worker_env_root",
        "cosmos_worker_site_packages",
        "cosmos_worker_cuda_library_path",
        "ld_library_path",
    ):
        assert stage2[name] == evaluation[name]
    assert (
        stage2["cosmos_predict2_repo_commit"]
        == "1eb8457072b4a1adfe1f83c3076e4aa5452cbab2"
    )
    assert stage2["ld_library_path"].startswith(
        stage2["cosmos_worker_cuda_library_path"]
    )


def test_worker_runtime_import_probe_fails_before_any_write(tmp_path):
    env, output_root = _environment(tmp_path)
    broken_worker = _executable(
        tmp_path / "cosmos-worker" / "bin" / "broken-python",
        "printf 'worker import failed\\n' >&2\nexit 17\n",
    )
    env["COSMOS_POLICY_PYTHON"] = str(broken_worker)
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))

    result = _invoke(tmp_path, env, output_root, "--phase", "all")

    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert result.returncode != 0
    assert "worker runtime probe" in result.stderr.lower()
    assert before == after
    assert not output_root.exists()
    assert not Path(env["CALLS_LOG"]).exists()


def test_real_stage1_lineage_is_validated_before_any_output_or_child(tmp_path):
    env, output_root = _environment(tmp_path)
    config = (
        tmp_path
        / "stage1"
        / "checkpoints"
        / "step_3000"
        / "target_student"
        / "transformer"
        / "config.json"
    )
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["action_downsample_factor"] = 8
    config.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _invoke(tmp_path, env, output_root, "--phase", "check")

    assert result.returncode != 0
    assert "Stage-1" in result.stderr or "contract" in result.stderr
    assert not output_root.exists()
    assert not Path(env["CALLS_LOG"]).exists()


def test_check_only_validates_lock_and_prompt_inputs_without_writes(tmp_path):
    env, output_root = _environment(tmp_path)
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))

    result = _invoke(tmp_path, env, output_root, "--phase", "check")

    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert result.returncode == 0, result.stderr
    assert "SENTINEL_PROVENANCE_DRY_RUN=1" in result.stdout
    assert "SENTINEL_PROMPT_DRY_RUN=1" in result.stdout
    assert before == after
    assert not output_root.exists()
    assert not Path(env["CALLS_LOG"]).exists()


def test_eval_preflights_real_stage2_inference_before_prompt_or_matrix_write(
    tmp_path,
):
    env, output_root = _environment(tmp_path)
    transformer = _stage2_checkpoint(
        output_root,
        stage1_checkpoint=tmp_path / "stage1" / "checkpoints" / "step_3000",
        wan_base=Path(env["WAN_STUDENT_BASE_MODEL_PATH"]),
        teacher=Path(env["COSMOS_POLICY_PATH"]),
    )
    config = transformer / "config.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["parent_stage1_contract_identity"] = "0" * 64
    config.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _invoke(tmp_path, env, output_root, "--phase", "eval")

    assert result.returncode != 0
    assert "Stage-2" in result.stderr or "parent_stage1_contract_identity" in result.stderr
    assert not (
        output_root / "student-eval" / "libero_wan_prompt_embeddings_all40.pt"
    ).exists()
    assert not (output_root / "student-eval" / "eval" / "student_only").exists()
    assert not Path(env["CALLS_LOG"]).exists()


def test_wrapper_restricts_gpu_contract_to_exact_ordinals_zero_through_seven(
    tmp_path,
):
    env, output_root = _environment(tmp_path)
    env["CUDA_VISIBLE_DEVICES"] = "8,9,10,11,12,13,14,15"

    result = _invoke(tmp_path, env, output_root, "--phase", "check")

    assert result.returncode == 2
    assert "0,1,2,3,4,5,6,7" in result.stderr
    assert not output_root.exists()


def test_wrapper_rejects_symlinked_run_root_component_before_any_write(tmp_path):
    env, _output_root = _environment(tmp_path)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    output_root = linked_parent / "outputs"

    result = _invoke(tmp_path, env, output_root, "--phase", "check")

    assert result.returncode == 2
    assert "symlink" in result.stderr.lower()
    assert not output_root.exists()
    assert not Path(env["CALLS_LOG"]).exists()


def test_relative_symlinked_output_root_uses_the_callers_cwd_for_preflight(tmp_path):
    env, _output_root = _environment(tmp_path)
    caller = tmp_path / "caller"
    caller.mkdir()
    real_parent = tmp_path / "real-relative-parent"
    real_parent.mkdir()
    (caller / "linked-parent").symlink_to(real_parent, target_is_directory=True)

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--phase",
            "check",
            "--stage1-root",
            str(tmp_path / "stage1"),
            "--output-root",
            "linked-parent/outputs",
            "--run-tag",
            "student-eval",
        ],
        cwd=caller,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 2
    assert "symlink" in result.stderr.lower()
    assert not (real_parent / "outputs").exists()
    assert not Path(env["CALLS_LOG"]).exists()


@pytest.mark.parametrize(
    ("sent_signal", "expected_code", "expected_name"),
    [
        (signal.SIGINT, 130, "INT"),
        (signal.SIGTERM, 143, "TERM"),
    ],
)
def test_wrapper_forwards_signals_to_child_process_group_with_standard_exit_code(
    tmp_path, sent_signal, expected_code, expected_name
):
    env, output_root = _environment(tmp_path)
    ready = tmp_path / "descendant-ready"
    observed = tmp_path / "descendant-signal"
    completed = tmp_path / "descendant-completed"
    descendant = tmp_path / "hanging-descendant.py"
    descendant.write_text(
        f"#!{Path(sys.executable).resolve()}\n"
        "import pathlib, signal, sys, time\n"
        f"observed = pathlib.Path({str(observed)!r})\n"
        "def handle(signum, _frame):\n"
        "    observed.write_text(signal.Signals(signum).name.removeprefix('SIG'))\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGINT, handle)\n"
        "signal.signal(signal.SIGTERM, handle)\n"
        f"pathlib.Path({str(ready)!r}).touch()\n"
        "time.sleep(3)\n"
        f"pathlib.Path({str(completed)!r}).touch()\n",
        encoding="utf-8",
    )
    descendant.chmod(descendant.stat().st_mode | stat.S_IXUSR)
    hanging = _executable(
        tmp_path / "hanging-stage2.sh",
        f"{str(descendant)!r} &\n"
        "child=$!\n"
        "trap 'wait \"$child\"; exit 0' INT TERM\n"
        "wait \"$child\"\n",
    )
    env["COSMOS_STAGE2_LAUNCHER"] = str(hanging)
    command = [
        "bash",
        str(SCRIPT),
        "--phase",
        "stage2",
        "--stage1-root",
        str(tmp_path / "stage1"),
        "--output-root",
        str(output_root),
        "--run-tag",
        "student-eval",
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # The audited lineage and worker-runtime preflights intentionally complete
    # before the child process starts; allow them to finish on a loaded host.
    deadline = time.monotonic() + 60
    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ready.exists(), process.communicate(timeout=5)

    started = time.monotonic()
    process.send_signal(sent_signal)
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == expected_code, stdout + "\n" + stderr
    assert observed.read_text(encoding="utf-8") == expected_name
    assert not completed.exists()
    assert time.monotonic() - started < 2
