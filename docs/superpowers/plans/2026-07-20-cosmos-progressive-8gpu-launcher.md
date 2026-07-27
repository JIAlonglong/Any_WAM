# Cosmos Progressive Eight-GPU Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one guarded eight-A800 launcher for the aligned Cosmos S4 -> S2 -> S1 training chain and prepare the released Stage-1 EMA for it.

**Architecture:** A single Bash launcher owns stage selection, immutable default paths, preflight, fresh/resume checkpoint routing, and the eight-rank `torchrun` invocation.  A Python/pytest test module treats the launcher as a subprocess and constructs only minimal fake checkpoint layouts, so all launch paths are verified without CUDA or Cosmos dependencies.  The released Stage-1 artifact is transferred once outside the launcher and adapted with a relative symlink rather than copying its 10.2GB transformer.

**Tech Stack:** Bash, PyTorch `torchrun`, pytest, FSDP1, ModelScope HTTP artifact endpoint, `rsync`.

## Global Constraints

- Work only in `/kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/cosmos-progressive-s4-eval` on branch `codex/cosmos-progressive-s4-eval`.
- Use `distillation_flowmap.config_libero_cosmos_policy_stage2_progressive` as the only source of S4/S2/S1 objective settings.
- Fresh S4 initializes from ModelScope revision `dbec074e378aafdf025d1eae9941871353f60c29`, artifact `raw_stage1_5000/target_student/transformer`, SHA-256 `13091097309a6579c938ea926fa4752f904f38f57c21f050e59b8b3cd70cc03a`.
- Use exactly eight unique CUDA ordinals and `torchrun --nproc_per_node=8`.
- Default output root is `/kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_cosmos_progressive_dance_aligned_8gpu_20260720`.
- Fixed fresh budgets are S4=5000, S2=3000, S1=3000; set `SAVE_INTERVAL=1000`.
- No GPU job, ModelScope download, output directory, or checkpoint may be created by `--dry-run`.
- Do not change the aligned Cosmos config, DanceOPD logic, or existing evaluation scripts in this work item.

---

## File Structure

- Create: `distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh`
  - Owns stage parsing, verified paths, rank-safe worker GPU mapping, fresh and same-stage continuation routing, and exact `torchrun` arguments.
- Create: `distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py`
  - Uses temporary fixture directories and subprocess dry-runs to test launcher behavior without CUDA.
- Existing verification: `distillation_flowmap/tests/test_cosmos_progressive_config.py`
  - Already asserts the aligned S4/S2/S1 Dance rollout and velocity settings; Task 1 runs it unchanged.
- Create at runtime, not tracked: `/kpfs-intern/jialongliu/models/modelscope/JIAlonglong/any-wam-cosmos-checkpoints/raw_stage1_5000/online_student`
  - Relative symlink to the already transferred `target_student` directory.

## Task 1: Add the launcher contract tests and guarded launcher

**Files:**
- Create: `distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh`
- Create: `distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py`

**Interfaces:**
- Consumes: `COSMOS_STAGE1_ROOT`, `OUTPUT_ROOT`, `DATASET_PATH`, `COSMOS_POLICY_PATH`, `COSMOS_POLICY_PYTHON`, `COSMOS_PREDICT2_REPO`, `COSMOS_PREDICT25_LOCAL_MODEL_DIR`, `CUDA_VISIBLE_DEVICES`, and `COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES` environment overrides.
- Produces: `bash distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh s4|s2|s1 [--master-port PORT] [--dry-run] [--resume-step STEP]`.
- Produces: a dry-run stdout record containing `COSMOS_PROGRESSIVE_STAGE`, `MAX_TRAIN_STEPS`, `RESUME_FROM_PATH`, `RESUME_ONLINE_FROM_TARGET`, `RESET_RESUME_STEP`, `RESUME_OPTIMIZER_STATE`, and the full `torchrun` argv.

- [ ] **Step 1: Write failing subprocess tests for the public launcher interface.**

Create `distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py` with these helpers and tests:

```python
import os
import stat
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "distillation_flowmap" / "run_cosmos_progressive_stage2_8gpu.sh"
DEVICES = "0,1,2,3,4,5,6,7"


def _transformer(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"checkpoint_step": 1000}\n', encoding="utf-8")
    (root / "diffusion_pytorch_model.safetensors").write_bytes(b"test")
    return root


def _layout(tmp_path: Path):
    stage1 = tmp_path / "stage1"
    _transformer(stage1 / "target_student" / "transformer")
    (stage1 / "online_student").symlink_to("target_student", target_is_directory=True)
    output = tmp_path / "output"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "empty_emb.pt").write_bytes(b"test")
    policy = tmp_path / "policy"
    policy.mkdir()
    repo = tmp_path / "cosmos-predict2.5"
    repo.mkdir()
    local_model = tmp_path / "local-model"
    local_model.mkdir()
    worker_python = tmp_path / "cosmos-python"
    worker_python.write_text("#!/usr/bin/env bash\\nexit 0\\n", encoding="utf-8")
    worker_python.chmod(worker_python.stat().st_mode | stat.S_IXUSR)
    return stage1, output, dataset, policy, repo, local_model, worker_python


def _env(tmp_path: Path):
    stage1, output, dataset, policy, repo, local_model, worker_python = _layout(tmp_path)
    env = os.environ.copy()
    env.update({
        "COSMOS_STAGE1_ROOT": str(stage1),
        "OUTPUT_ROOT": str(output),
        "DATASET_PATH": str(dataset),
        "COSMOS_POLICY_PATH": str(policy),
        "COSMOS_POLICY_PYTHON": str(worker_python),
        "COSMOS_PREDICT2_REPO": str(repo),
        "COSMOS_PREDICT25_LOCAL_MODEL_DIR": str(local_model),
        "CUDA_VISIBLE_DEVICES": DEVICES,
        "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES": DEVICES,
    })
    return env, output


def _run(tmp_path: Path, *args: str, env=None):
    run_env, _ = _env(tmp_path) if env is None else (env, None)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        text=True,
        capture_output=True,
        env=run_env,
        check=False,
    )


@pytest.mark.parametrize(
    ("stage", "steps", "resume_target"),
    [("s4", "5000", "1"), ("s2", "3000", "0"), ("s1", "3000", "0")],
)
def test_dry_run_prints_stage_contract(tmp_path, stage, steps, resume_target):
    env, output = _env(tmp_path)
    if stage == "s2":
        _transformer(output / "s4" / "checkpoints" / "step_5000" / "online_student" / "transformer")
    if stage == "s1":
        _transformer(output / "s2" / "checkpoints" / "step_3000" / "online_student" / "transformer")
    result = _run(tmp_path, stage, "--dry-run", env=env)
    assert result.returncode == 0, result.stderr
    assert f"COSMOS_PROGRESSIVE_STAGE={stage}" in result.stdout
    assert f"MAX_TRAIN_STEPS={steps}" in result.stdout
    assert f"RESUME_ONLINE_FROM_TARGET={resume_target}" in result.stdout
    assert "--nproc_per_node=8" in result.stdout
    assert not (output / stage).exists()


def test_dry_run_rejects_missing_stage1_compatibility_link(tmp_path):
    env, _ = _env(tmp_path)
    Path(env["COSMOS_STAGE1_ROOT"]).joinpath("online_student").unlink()
    result = _run(tmp_path, "s4", "--dry-run", env=env)
    assert result.returncode != 0
    assert "online_student/transformer" in result.stderr


def test_dry_run_rejects_non_eight_gpu_layout(tmp_path):
    env, _ = _env(tmp_path)
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    result = _run(tmp_path, "s4", "--dry-run", env=env)
    assert result.returncode != 0
    assert "exactly 8" in result.stderr


def test_dry_run_rejects_existing_fresh_checkpoint_root(tmp_path):
    env, output = _env(tmp_path)
    (output / "s4" / "checkpoints").mkdir(parents=True)
    result = _run(tmp_path, "s4", "--dry-run", env=env)
    assert result.returncode != 0
    assert "Refusing fresh run" in result.stderr


def test_resume_step_uses_own_online_checkpoint_and_optimizer(tmp_path):
    env, output = _env(tmp_path)
    resume = output / "s4" / "checkpoints" / "step_1000"
    _transformer(resume / "online_student" / "transformer")
    (resume / "optimizer.pt").write_bytes(b"test")
    result = _run(tmp_path, "s4", "--resume-step", "1000", "--dry-run", env=env)
    assert result.returncode == 0, result.stderr
    assert f"RESUME_FROM_PATH={resume}" in result.stdout
    assert "RESET_RESUME_STEP=0" in result.stdout
    assert "RESUME_OPTIMIZER_STATE=1" in result.stdout
```

- [ ] **Step 2: Run the tests and confirm they fail because the launcher does not exist.**

Run:

```bash
pytest -q distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py
```

Expected: collection or subprocess failure referring to missing
`run_cosmos_progressive_stage2_8gpu.sh`.

- [ ] **Step 3: Implement the guarded Bash launcher.**

Create `distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh` with this complete interface and core implementation:

```bash
#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\\n' "$*" >&2; exit 2; }
require_dir() { [[ -d "$2" ]] || die "$1 is not a directory: $2"; }
require_file() { [[ -f "$2" ]] || die "$1 is missing: $2"; }
require_transformer() {
    local label="$1" root="$2"
    require_dir "$label" "$root"
    require_file "$label/config.json" "$root/config.json"
    require_file "$label/diffusion_pytorch_model.safetensors" "$root/diffusion_pytorch_model.safetensors"
}
validate_port() { [[ "$1" =~ ^[0-9]+$ ]] && (( $1 >= 1 && $1 <= 65535 )) || die "invalid master port: $1"; }
validate_devices() {
    local label="$1" raw="$2" item
    local -a values
    local -A seen=()
    IFS=, read -r -a values <<< "$raw"
    (( ${#values[@]} == 8 )) || die "$label must contain exactly 8 comma-separated GPU ordinals: $raw"
    for item in "${values[@]}"; do
        [[ "$item" =~ ^[0-9]+$ ]] || die "$label contains a non-numeric GPU ordinal: $item"
        [[ -z "${seen[$item]:-}" ]] || die "$label contains duplicate GPU ordinal: $item"
        seen[$item]=1
    done
}
print_assignment() { printf '%s=%s\\n' "$1" "$2"; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
STAGE1_ROOT="${COSMOS_STAGE1_ROOT:-/kpfs-intern/jialongliu/models/modelscope/JIAlonglong/any-wam-cosmos-checkpoints/raw_stage1_5000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_cosmos_progressive_dance_aligned_8gpu_20260720}"
DATASET_PATH="${DATASET_PATH:-/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot}"
COSMOS_POLICY_PATH="${COSMOS_POLICY_PATH:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
COSMOS_POLICY_PYTHON="${COSMOS_POLICY_PYTHON:-/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/bin/python}"
COSMOS_PREDICT2_REPO="${COSMOS_PREDICT2_REPO:-/kpfs-intern/jialongliu/projects/cosmos-predict2.5}"
COSMOS_PREDICT25_LOCAL_MODEL_DIR="${COSMOS_PREDICT25_LOCAL_MODEL_DIR:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES="${COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

stage="${1:-}"; [[ $# -gt 0 ]] && shift || true
dry_run=0; resume_step=""; master_port=""
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --master-port) shift; (( $# )) || die "--master-port requires a value"; master_port="$1" ;;
        --resume-step) shift; (( $# )) || die "--resume-step requires a value"; resume_step="$1" ;;
        *) die "unknown option: $1" ;;
    esac
    shift
done

case "$stage" in
    s4) max_steps=5000; default_port=29661; predecessor="$STAGE1_ROOT"; fresh_target=1 ;;
    s2) max_steps=3000; default_port=29662; predecessor="$OUTPUT_ROOT/s4/checkpoints/step_5000"; fresh_target=0 ;;
    s1) max_steps=3000; default_port=29663; predecessor="$OUTPUT_ROOT/s2/checkpoints/step_3000"; fresh_target=0 ;;
    *) die "stage must be one of: s4, s2, s1" ;;
esac
master_port="${master_port:-$default_port}"; validate_port "$master_port"
output_dir="$OUTPUT_ROOT/$stage"
validate_devices CUDA_VISIBLE_DEVICES "$CUDA_VISIBLE_DEVICES"
validate_devices COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES "$COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES"
[[ "$COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES" == "$CUDA_VISIBLE_DEVICES" ]] || die "worker GPU list must equal CUDA_VISIBLE_DEVICES"
require_dir DATASET_PATH "$DATASET_PATH"; require_file empty_emb.pt "$DATASET_PATH/empty_emb.pt"
require_dir COSMOS_POLICY_PATH "$COSMOS_POLICY_PATH"; require_dir COSMOS_PREDICT2_REPO "$COSMOS_PREDICT2_REPO"
require_dir COSMOS_PREDICT25_LOCAL_MODEL_DIR "$COSMOS_PREDICT25_LOCAL_MODEL_DIR"; [[ -x "$COSMOS_POLICY_PYTHON" ]] || die "COSMOS_POLICY_PYTHON is not executable: $COSMOS_POLICY_PYTHON"

if [[ -n "$resume_step" ]]; then
    [[ "$resume_step" =~ ^[1-9][0-9]*$ ]] && (( resume_step < max_steps )) || die "resume step must be in [1, $((max_steps - 1))]"
    resume_from_path="$output_dir/checkpoints/step_$resume_step"
    require_transformer "resume online_student/transformer" "$resume_from_path/online_student/transformer"
    require_file resume_optimizer "$resume_from_path/optimizer.pt"
    resume_online_from_target=0; reset_resume_step=0; resume_optimizer_state=1
else
    [[ ! -d "$output_dir/checkpoints" ]] || die "Refusing fresh run with existing checkpoints: $output_dir/checkpoints"
    resume_from_path="$predecessor"; resume_online_from_target="$fresh_target"; reset_resume_step=1; resume_optimizer_state=0
    require_transformer "predecessor online_student/transformer" "$resume_from_path/online_student/transformer"
    if (( fresh_target )); then require_transformer "Stage-1 target_student/transformer" "$resume_from_path/target_student/transformer"; fi
fi

export CONFIG_FILE=distillation_flowmap.config_libero_cosmos_policy_stage2_progressive COSMOS_PROGRESSIVE_STAGE="$stage" COSMOS_PROGRESSIVE_OUTPUT_ROOT="$OUTPUT_ROOT" OUTPUT_DIR="$output_dir" MAX_TRAIN_STEPS="$max_steps" SAVE_INTERVAL=1000 RESUME_FROM_PATH="$resume_from_path" RESUME_ONLINE_FROM_TARGET="$resume_online_from_target" RESET_RESUME_STEP="$reset_resume_step" RESUME_OPTIMIZER_STATE="$resume_optimizer_state" USE_FSDP1=1 GRADIENT_CHECKPOINTING=1 OPD_AUX_GRADIENT_CHECKPOINTING=1 OPD_SERIAL_STUDENT_CFG=1 OPD_AUX_STANDALONE_STEP=1 OPD_COSMOS_SPATIAL_CROP_SIZE=28 ENABLE_WANDB=0 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES COSMOS_POLICY_PATH COSMOS_POLICY_PYTHON COSMOS_PREDICT2_REPO COSMOS_PREDICT25_LOCAL_MODEL_DIR
export COSMOS_POLICY_EXTRA_PYTHONPATH="${COSMOS_POLICY_EXTRA_PYTHONPATH:-$COSMOS_PREDICT2_REPO/packages/cosmos-cuda:$COSMOS_PREDICT2_REPO/packages/cosmos-oss}"
command=("$TORCHRUN_BIN" --nproc_per_node=8 "--master_port=$master_port" distillation_flowmap/train.py --teacher-model-path "$COSMOS_POLICY_PATH" --dataset-path "$DATASET_PATH" --output-dir "$output_dir" --resume-from-path "$resume_from_path" --gradient-accumulation-steps 1)
for key in COSMOS_PROGRESSIVE_STAGE MAX_TRAIN_STEPS RESUME_FROM_PATH RESUME_ONLINE_FROM_TARGET RESET_RESUME_STEP RESUME_OPTIMIZER_STATE SAVE_INTERVAL CUDA_VISIBLE_DEVICES COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES; do print_assignment "$key" "${!key}"; done
printf 'command='; printf '%q ' "${command[@]}"; printf '\\n'
(( dry_run )) && exit 0
mkdir -p "$output_dir"
cd "$PROJECT_ROOT"
exec "${command[@]}"
```

Ensure the file is executable:

```bash
chmod +x distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh
```

- [ ] **Step 4: Run syntax and targeted tests to verify the launcher passes.**

Run:

```bash
bash -n distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh
pytest -q distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py distillation_flowmap/tests/test_cosmos_progressive_config.py
```

Expected: Bash exits `0`; every launcher preflight test and all progressive
config regression tests pass without accessing a GPU.

- [ ] **Step 5: Commit the tested launcher and tests.**

```bash
git add distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh \
  distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py
git commit -m "feat: add guarded Cosmos progressive 8-GPU launcher"
```

## Task 2: Prepare the released Stage-1 source and verify real-path dry-runs

**Files:**
- Create at runtime: `/kpfs-intern/jialongliu/models/modelscope/JIAlonglong/any-wam-cosmos-checkpoints/raw_stage1_5000/target_student/transformer/config.json`
- Create at runtime: `/kpfs-intern/jialongliu/models/modelscope/JIAlonglong/any-wam-cosmos-checkpoints/raw_stage1_5000/target_student/transformer/diffusion_pytorch_model.safetensors`
- Create at runtime: `/kpfs-intern/jialongliu/models/modelscope/JIAlonglong/any-wam-cosmos-checkpoints/raw_stage1_5000/online_student` as a relative symbolic link.

**Interfaces:**
- Consumes: the pinned ModelScope Stage-1 artifact and the committed Task 1 launcher.
- Produces: a checksum-verified remote Stage-1 root accepted by `s4 --dry-run`.

- [ ] **Step 1: Download the two pinned Stage-1 files locally and validate the model checksum.**

Use the ModelScope API endpoint with the pinned revision, storing only under a
temporary local staging directory.  Then run:

```bash
shasum -a 256 raw_stage1_5000/target_student/transformer/diffusion_pytorch_model.safetensors
```

Expected: exactly
`13091097309a6579c938ea926fa4752f904f38f57c21f050e59b8b3cd70cc03a`.

- [ ] **Step 2: Transfer the verified Stage-1 tree to the exact ModelScope cache root on the training machine.**

Run a resumable transfer that preserves the directory structure:

```bash
rsync -a --partial --append-verify raw_stage1_5000/ \
  root@36.103.197.249:/kpfs-intern/jialongliu/models/modelscope/JIAlonglong/any-wam-cosmos-checkpoints/raw_stage1_5000/
```

Expected: remote tree contains the two target transformer files and no copied
online transformer data.

- [ ] **Step 3: Create and inspect the zero-copy compatibility link remotely.**

From the remote Stage-1 root, run:

```bash
ln -s target_student online_student
test -L online_student
test -f online_student/transformer/config.json
test -f online_student/transformer/diffusion_pytorch_model.safetensors
```

Expected: all checks pass.  If `online_student` already exists, verify it
resolves to `target_student` before leaving it untouched; never replace an
unrelated directory or link.

- [ ] **Step 4: Run actual-path S4 dry-run and config smoke, without GPUs.**

Run:

```bash
source /kpfs-intern/jialongliu/miniforge3/bin/activate
conda activate flashwam
cd /kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/cosmos-progressive-s4-eval
bash distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh s4 --dry-run
python -m pytest -q distillation_flowmap/tests/test_cosmos_progressive_config.py
```

Expected: S4 reports `MAX_TRAIN_STEPS=5000`,
`RESUME_ONLINE_FROM_TARGET=1`, and `--nproc_per_node=8`; config tests resolve
S4/S2/S1 Dance settings `(4, 1.0)`, `(2, 1.0)`, `(1, 0.0)`.

- [ ] **Step 5: Record the exact user launch commands in the handoff.**

Deliver these commands after the successful dry-run:

```bash
source /kpfs-intern/jialongliu/miniforge3/bin/activate && conda activate flashwam && cd /kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/cosmos-progressive-s4-eval && bash distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh s4 --master-port 29661
source /kpfs-intern/jialongliu/miniforge3/bin/activate && conda activate flashwam && cd /kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/cosmos-progressive-s4-eval && bash distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh s2 --master-port 29662
source /kpfs-intern/jialongliu/miniforge3/bin/activate && conda activate flashwam && cd /kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/cosmos-progressive-s4-eval && bash distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh s1 --master-port 29663
```

The S2 command is valid only after new S4 completes to
`$OUTPUT_ROOT/s4/checkpoints/step_5000`; S1 is valid only after new S2
completes to `$OUTPUT_ROOT/s2/checkpoints/step_3000`.

## Plan Self-Review

- **Spec coverage:** Task 1 implements every launcher safety, stage routing,
  eight-GPU mapping, fixed save cadence, dry-run, and non-GPU validation
  requirement.  Task 2 covers the pinned Stage-1 download, checksum,
  zero-copy compatibility layout, real-path smoke, and final handoff.
- **Placeholder scan:** The plan contains no unfinished implementation markers
  or unspecified paths; command variables are named interfaces with fixed
  defaults defined in Task 1.
- **Consistency:** `COSMOS_STAGE1_ROOT`, `OUTPUT_ROOT`, fresh-stage resume
  behavior, and the S4/S2/S1 maximum steps are identical in the spec, test
  fixture, launcher implementation, runtime preparation, and handoff commands.
