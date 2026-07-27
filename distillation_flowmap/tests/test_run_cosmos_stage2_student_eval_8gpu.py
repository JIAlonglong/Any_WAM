import json
import os
import stat
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "distillation_flowmap" / "run_cosmos_stage2_student_eval_8gpu.sh"


def _executable(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _plain(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _sentinel(path: Path, kind: str) -> Path:
    return _executable(
        path,
        "python3 - \"$@\" <<'PY'\n"
        "import json, os, pathlib, sys\n"
        "record = {'kind': " + repr(kind) + ", 'argv': sys.argv[1:]}\n"
        "for name in ('COSMOS_STAGE1_EXPECTED_STEP', 'MATRIX_ROOT', 'S4_CKPT_ROOT', "
        "'S4_MATRIX_ROLES', 'S4_FORMAL_NUM_SHARDS', 'S4_FORMAL_GPU_LAYOUT', "
        "'S4_VIDEO_SEEDS', 'S4_EPISODES_PER_TASK', 'CUDA_VISIBLE_DEVICES', "
        "'PYTORCH_CUDA_ALLOC_CONF', 'PYTHON_BIN'):\n"
        "    if name in os.environ: record[name.lower()] = os.environ[name]\n"
        "if record['kind'] == 'stage2':\n"
        "    record['parent_expected_step'] = os.environ['COSMOS_STAGE1_EXPECTED_STEP']\n"
        "    record['stage2_steps'] = sys.argv[sys.argv.index('--steps') + 1]\n"
        "    root = pathlib.Path(os.environ['PIPELINE_RUN_ROOT']) / 'universal-video-action' / 'checkpoints' / ('step_' + record['stage2_steps']) / 'target_student' / 'transformer'\n"
        "    root.mkdir(parents=True); (root / 'config.json').write_text('{}'); (root / 'diffusion_pytorch_model.safetensors').write_bytes(b'x')\n"
        "elif record['kind'] == 'provenance':\n"
        "    root = pathlib.Path(sys.argv[sys.argv.index('--output-root') + 1]); root.mkdir(parents=True)\n"
        "elif record['kind'] == 'prompt':\n"
        "    output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1]); output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(b'prompt'); record['output'] = str(output)\n"
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
    stage1 = tmp_path / "stage1"
    target = stage1 / "checkpoints" / "step_3000" / "target_student" / "transformer"
    _plain(target / "config.json", b"{}")
    _plain(target / "diffusion_pytorch_model.safetensors")
    wan = tmp_path / "wan"
    _plain(wan / "transformer" / "config.json", b"{}")
    _plain(wan / "transformer" / "diffusion_pytorch_model.safetensors")
    dataset = tmp_path / "dataset"
    _plain(dataset / "empty_emb.pt")
    teacher = tmp_path / "teacher"; teacher.mkdir()
    local_model = tmp_path / "local-model"; local_model.mkdir()
    calls = tmp_path / "calls.jsonl"
    env = os.environ.copy()
    env.update({
        "PYTHON_BIN": sys.executable,
        "WAN_STUDENT_BASE_MODEL_PATH": str(wan),
        "DATASET_PATH": str(dataset),
        "COSMOS_POLICY_PATH": str(teacher),
        "COSMOS_PREDICT25_LOCAL_MODEL_DIR": str(local_model),
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
    transformer = output_root / "student-eval" / "universal-video-action" / "checkpoints" / "step_5000" / "target_student" / "transformer"
    _plain(transformer / "config.json", b"{}")
    _plain(transformer / "diffusion_pytorch_model.safetensors")
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
    transformer = output_root / "student-eval" / "universal-video-action" / "checkpoints" / "step_5000" / "target_student" / "transformer"
    _plain(transformer / "config.json", b"{}")
    _plain(transformer / "diffusion_pytorch_model.safetensors")
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
    transformer = output_root / "student-eval" / "universal-video-action" / "checkpoints" / "step_5000" / "target_student" / "transformer"
    _plain(transformer / "config.json", b"{}")
    _plain(transformer / "diffusion_pytorch_model.safetensors")
    result = subprocess.run(["bash", str(SCRIPT), "--phase", "eval", "--stage1-root", str(tmp_path / "stage1"), "--output-root", str(output_root), "--run-tag", "student-eval"], cwd=ROOT, text=True, capture_output=True, env=env, check=False)
    calls = [json.loads(line) for line in Path(env["CALLS_LOG"]).read_text(encoding="utf-8").splitlines()]
    assert result.returncode == 0, result.stderr
    assert calls[-1]["python_bin"] == "/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python"
