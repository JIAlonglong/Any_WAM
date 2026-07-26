# Native Teacher Coarse-Grid Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auditable eight-GPU LIBERO evaluation path that runs the original LingBotVA teacher with native, separate video and action denoising loops at matched `K=1/2/4` solver budgets.

**Architecture:** A dedicated native-teacher server vendors the upstream LingBotVA server semantics and adds only the deployment CLI, provenance, and latency hooks required by the existing LIBERO client. `run_eval_new.sh` selects this server only when `SERVER_BACKEND=native_teacher`; a separate eight-GPU launcher covers all 40 tasks and writes summaries under the `teacher_native` identity.

**Tech Stack:** Bash, Python 3.10, PyTorch, torch.distributed, Diffusers, WebSockets, pytest, LIBERO.

## Global Constraints

- The native server must not import or invoke `flowmap_inference`.
- The native server must not use `action_downsample_factor`.
- Native video and action denoising remain two separate loops.
- Formal budgets are exactly matched `1/1`, `2/2`, and `4/4`.
- Model identity is exactly `teacher_native`.
- The existing FlowMap student backend remains the default and retains its current behavior.
- Teacher-native and student evaluations use different output roots and ports.
- Formal coverage is four suites, 40 tasks, and 50 episodes per task.
- No unrelated dirty-worktree files may be staged or committed.

---

## File Structure

- Create `wan_va/native_teacher_contract.py`: pure validation and runtime-contract helpers.
- Create `wan_va/wan_va_native_teacher_server.py`: dedicated native server and original two-loop sampler.
- Modify `evaluation/libero/run_eval_new.sh`: explicit backend routing.
- Create `evaluation/libero/run_lingbotva_native_teacher_4suite_124_eval_8gpu.sh`: formal teacher-only launcher.
- Create `distillation_flowmap/tests/test_native_teacher_contract.py`: unit and source-contract tests.
- Modify `evaluation/libero/tests/test_run_eval_new_contract.py`: backend-routing tests.
- Create `evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py`: launcher contract tests.

### Task 1: Native Teacher Runtime Contract

**Files:**
- Create: `wan_va/native_teacher_contract.py`
- Create: `distillation_flowmap/tests/test_native_teacher_contract.py`

**Interfaces:**
- Produces: `resolve_native_teacher_contract(*, model_name: str, video_steps: int, action_steps: int) -> dict`
- Produces: dictionary keys `backend`, `model`, `video_steps`, `action_steps`, `action_grid`, and `video_action_bridge`
- Consumes: no repository runtime state; the helper remains importable without CUDA or model dependencies

- [ ] **Step 1: Write failing contract tests**

```python
import pytest

from wan_va.native_teacher_contract import resolve_native_teacher_contract


def test_native_teacher_contract_accepts_matched_coarse_budget():
    assert resolve_native_teacher_contract(
        model_name="teacher_native",
        video_steps=2,
        action_steps=2,
    ) == {
        "backend": "native_teacher",
        "model": "teacher_native",
        "video_steps": 2,
        "action_steps": 2,
        "action_grid": "full",
        "video_action_bridge": "disabled",
    }


@pytest.mark.parametrize("steps", [0, -1])
def test_native_teacher_contract_rejects_non_positive_budget(steps):
    with pytest.raises(ValueError, match="positive"):
        resolve_native_teacher_contract(
            model_name="teacher_native",
            video_steps=steps,
            action_steps=steps,
        )


def test_native_teacher_contract_rejects_mismatched_budget():
    with pytest.raises(ValueError, match="matched"):
        resolve_native_teacher_contract(
            model_name="teacher_native",
            video_steps=1,
            action_steps=2,
        )


def test_native_teacher_contract_rejects_student_identity():
    with pytest.raises(ValueError, match="teacher_native"):
        resolve_native_teacher_contract(
            model_name="stage2",
            video_steps=1,
            action_steps=1,
        )
```

- [ ] **Step 2: Run the tests and verify the import fails**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest distillation_flowmap/tests/test_native_teacher_contract.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'wan_va.native_teacher_contract'`.

- [ ] **Step 3: Implement the pure contract helper**

```python
"""Runtime contract for native LingBotVA teacher evaluation."""


def _positive_steps(value, field):
    steps = int(value)
    if steps <= 0:
        raise ValueError(f"{field} must be positive")
    return steps


def resolve_native_teacher_contract(*, model_name, video_steps, action_steps):
    if model_name != "teacher_native":
        raise ValueError("native teacher backend requires model_name=teacher_native")
    video_steps = _positive_steps(video_steps, "video_steps")
    action_steps = _positive_steps(action_steps, "action_steps")
    if video_steps != action_steps:
        raise ValueError("native teacher requires matched video/action budgets")
    return {
        "backend": "native_teacher",
        "model": "teacher_native",
        "video_steps": video_steps,
        "action_steps": action_steps,
        "action_grid": "full",
        "video_action_bridge": "disabled",
    }
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest distillation_flowmap/tests/test_native_teacher_contract.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the contract**

```bash
git add wan_va/native_teacher_contract.py \
  distillation_flowmap/tests/test_native_teacher_contract.py
git commit -m "feat: define native teacher evaluation contract"
```

### Task 2: Dedicated Native Teacher Server

**Files:**
- Create: `wan_va/wan_va_native_teacher_server.py`
- Modify: `distillation_flowmap/tests/test_native_teacher_contract.py`

**Interfaces:**
- Consumes: `resolve_native_teacher_contract(...)` from Task 1
- Consumes: the existing WebSocket request schema from `evaluation/libero/client.py`
- Produces: CLI flags `--config-name`, `--port`, `--checkpoint-path`, `--num-steps`, `--action-num-steps`, `--model-name`, `--latency-jsonl`, and `--save-root`
- Produces: WebSocket responses containing `{"action": np.ndarray}` with the full continuous action grid

- [ ] **Step 1: Add failing source-contract tests**

```python
from pathlib import Path


def test_native_teacher_server_uses_native_separate_loops():
    root = Path(__file__).resolve().parents[2]
    source = (
        root / "wan_va" / "wan_va_native_teacher_server.py"
    ).read_text(encoding="utf-8")

    assert "flowmap_inference" not in source
    assert "action_downsample_factor" not in source
    assert "# Native video denoising loop" in source
    assert "# Native action denoising loop" in source
    assert "self.scheduler.set_timesteps(self.native_contract[\"video_steps\"])" in source
    assert (
        "self.action_scheduler.set_timesteps("
        "self.native_contract[\"action_steps\"]"
    ) in source
    assert "actions = self.action_scheduler.step(" in source
    assert "actions[:, :, ::" not in source


def test_native_teacher_server_exposes_deployment_cli():
    root = Path(__file__).resolve().parents[2]
    source = (
        root / "wan_va" / "wan_va_native_teacher_server.py"
    ).read_text(encoding="utf-8")

    for flag in (
        "--checkpoint-path",
        "--num-steps",
        "--action-num-steps",
        "--model-name",
        "--latency-jsonl",
        "--save-root",
    ):
        assert flag in source
    assert "append_sampler_latency_record" in source
    assert "eval_metadata" in source
```

- [ ] **Step 2: Run the source tests and verify the missing file failure**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest \
  distillation_flowmap/tests/test_native_teacher_contract.py::test_native_teacher_server_uses_native_separate_loops \
  distillation_flowmap/tests/test_native_teacher_contract.py::test_native_teacher_server_exposes_deployment_cli \
  -q
```

Expected: both tests fail with `FileNotFoundError`.

- [ ] **Step 3: Vendor the upstream native server semantics**

Copy the current upstream implementation from:

```text
/kpfs-intern/jialongliu/projects/lingbot-va/wan_va/wan_va_server.py
```

to:

```text
wan_va/wan_va_native_teacher_server.py
```

Preserve these upstream behaviors exactly:

```python
# Native video denoising loop
for i, t in enumerate(timesteps):
    ...
    video_noise_pred = self.transformer(..., action_mode=False)
    latents = self.scheduler.step(
        video_noise_pred, t, latents, return_dict=False
    )

# Native action denoising loop
for i, t in enumerate(action_timesteps):
    ...
    action_noise_pred = self.transformer(..., action_mode=True)
    actions = self.action_scheduler.step(
        action_noise_pred, t, actions, return_dict=False
    )
```

Do not copy upstream generated artifacts, checkpoint files, or unrelated
working-tree modifications.

- [ ] **Step 4: Add deployment configuration and contract resolution**

Add arguments:

```python
parser.add_argument("--checkpoint-path", required=True)
parser.add_argument("--num-steps", type=int, required=True)
parser.add_argument("--action-num-steps", type=int, required=True)
parser.add_argument("--model-name", default="teacher_native")
parser.add_argument("--latency-jsonl", default=None)
parser.add_argument("--save-root", required=True)
```

Resolve and log the contract before constructing `VA_Server`:

```python
native_contract = resolve_native_teacher_contract(
    model_name=args.model_name,
    video_steps=args.num_steps,
    action_steps=args.action_num_steps,
)
logger.info("NATIVE_TEACHER_CONTRACT %s", native_contract)

config.wan22_pretrained_model_name_or_path = str(
    Path(args.checkpoint_path).resolve().parent
)
config.num_inference_steps = native_contract["video_steps"]
config.action_num_inference_steps = native_contract["action_steps"]
config.eval_model_name = native_contract["model"]
config.latency_jsonl = args.latency_jsonl
config.save_root = args.save_root
```

Validate that `args.checkpoint_path` is the `transformer` directory under the
base model root and contains either
`diffusion_pytorch_model.safetensors` or
`diffusion_pytorch_model.safetensors.index.json`; otherwise raise
`FileNotFoundError` before distributed model construction.

- [ ] **Step 5: Add provenance and latency handling without changing sampling**

On reset, store the client-provided metadata:

```python
self.eval_metadata = obs.get("eval_metadata", {})
```

Around `_infer`, record sampler-only latency with the existing helpers:

```python
started_at = time.perf_counter()
action, _ = self._infer(obs, frame_st_id=self.frame_st_id)
elapsed_ms = (time.perf_counter() - started_at) * 1000.0
if self.latency_jsonl:
    record = build_sampler_latency_record(
        model="teacher_native",
        suite=self.eval_metadata["suite"],
        task_idx=self.eval_metadata["task_idx"],
        episode_idx=self.eval_metadata["episode_idx"],
        video_steps=self.native_contract["video_steps"],
        action_steps=self.native_contract["action_steps"],
        call_index=self.sampler_call_index,
        elapsed_ms=elapsed_ms,
    )
    append_sampler_latency_record(self.latency_jsonl, record)
    self.sampler_call_index += 1
```

This uses the existing `build_sampler_latency_record` and
`append_sampler_latency_record` signatures without introducing a second
latency schema.

- [ ] **Step 6: Run server contract and syntax tests**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest distillation_flowmap/tests/test_native_teacher_contract.py -q
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m py_compile wan_va/wan_va_native_teacher_server.py
```

Expected: all tests pass and compilation exits zero.

- [ ] **Step 7: Commit the native server**

```bash
git add wan_va/wan_va_native_teacher_server.py \
  distillation_flowmap/tests/test_native_teacher_contract.py
git commit -m "feat: add native LingBotVA teacher server"
```

### Task 3: Evaluator Backend Routing

**Files:**
- Modify: `evaluation/libero/run_eval_new.sh`
- Modify: `evaluation/libero/tests/test_run_eval_new_contract.py`

**Interfaces:**
- Consumes: `SERVER_BACKEND` with allowed values `flowmap` and `native_teacher`
- Consumes: the native server CLI from Task 2
- Produces: check-only header line `Server backend: <backend>`
- Preserves: existing FlowMap command when `SERVER_BACKEND` is unset

- [ ] **Step 1: Add failing routing tests**

```python
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


def test_native_teacher_backend_routes_to_dedicated_server(tmp_path):
    env = _env(tmp_path, "libero_10")
    env.update({
        "SERVER_BACKEND": "native_teacher",
        "MODEL_NAME": "teacher_native",
    })
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
```

- [ ] **Step 2: Run routing tests and verify they fail**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest evaluation/libero/tests/test_run_eval_new_contract.py -q
```

Expected: new backend assertions fail because routing is absent.

- [ ] **Step 3: Implement explicit backend selection**

Add near the existing environment defaults:

```bash
SERVER_BACKEND="${SERVER_BACKEND:-flowmap}"
case "$SERVER_BACKEND" in
    flowmap)
        SERVER_ENTRYPOINT="wan_va/wan_va_server.py"
        ;;
    native_teacher)
        SERVER_ENTRYPOINT="wan_va/wan_va_native_teacher_server.py"
        [ "$MODEL_NAME" = "teacher_native" ] || {
            echo "native_teacher requires MODEL_NAME=teacher_native" >&2
            exit 2
        }
        ;;
    *)
        echo "Unsupported SERVER_BACKEND: $SERVER_BACKEND" >&2
        exit 2
        ;;
esac
```

Print both values in `log_header`:

```bash
echo "  Server backend: ${SERVER_BACKEND}"
echo "  Server entry:   ${SERVER_ENTRYPOINT}"
```

Replace only the hard-coded server entrypoint in `start_server`:

```bash
"$SERVER_ENTRYPOINT" \
```

Keep all existing arguments and the default FlowMap behavior unchanged.

- [ ] **Step 4: Run routing and existing evaluator tests**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest \
  evaluation/libero/tests/test_run_eval_new_contract.py \
  distillation_flowmap/tests/test_runtime_metadata.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit backend routing**

```bash
git add evaluation/libero/run_eval_new.sh \
  evaluation/libero/tests/test_run_eval_new_contract.py
git commit -m "feat: route LIBERO native teacher evaluation"
```

### Task 4: Formal Eight-GPU Native Teacher Launcher

**Files:**
- Create: `evaluation/libero/run_lingbotva_native_teacher_4suite_124_eval_8gpu.sh`
- Create: `evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py`

**Interfaces:**
- Consumes: `run_eval_new.sh` with `SERVER_BACKEND=native_teacher`
- Produces: CLI `--checkpoint`, `--output-root`, `--episodes`, `--gpu-ids`, `--master-port-base`, and `--ws-port-base`
- Produces: summaries at `<output-root>/teacher_native/steps_<K>/summary.json`

- [ ] **Step 1: Add failing launcher tests**

```python
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "evaluation"
    / "libero"
    / "run_lingbotva_native_teacher_4suite_124_eval_8gpu.sh"
)


def test_check_only_plans_native_teacher_40_tasks_at_matched_124(tmp_path):
    checkpoint = tmp_path / "base" / "transformer"
    checkpoint.mkdir(parents=True)
    env = os.environ.copy()
    env["CHECK_ONLY"] = "1"
    result = subprocess.run(
        [
            "bash", str(SCRIPT),
            "--checkpoint", str(checkpoint),
            "--output-root", str(tmp_path / "results"),
            "--episodes", "50",
            "--gpu-ids", "0,1,2,3,4,5,6,7",
            "--master-port-base", "30680",
            "--ws-port-base", "30780",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("WORKER ") == 24
    assert "model=teacher_native" in result.stdout
    assert "backend=native_teacher" in result.stdout
    assert "video_steps=1 action_steps=1" in result.stdout
    assert "video_steps=2 action_steps=2" in result.stdout
    assert "video_steps=4 action_steps=4" in result.stdout
    assert "tasks=0:5" in result.stdout
    assert "tasks=5:10" in result.stdout


def test_launcher_rejects_duplicate_gpus(tmp_path):
    env = os.environ.copy()
    env["CHECK_ONLY"] = "1"
    result = subprocess.run(
        [
            "bash", str(SCRIPT),
            "--checkpoint", str(tmp_path / "transformer"),
            "--output-root", str(tmp_path / "results"),
            "--gpu-ids", "0,1,2,3,4,5,6,6",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "8 unique GPUs" in result.stderr
```

- [ ] **Step 2: Run launcher tests and verify the missing script failure**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py \
  -q
```

Expected: tests fail because the launcher does not exist.

- [ ] **Step 3: Implement the launcher**

Base its worker partition and merge behavior on
`evaluation/libero/run_lingbotva_4suite_124_eval_8gpu.sh`, with these fixed
values:

```bash
MODEL_NAME="teacher_native"
BUDGETS=(1 2 4)
SUITES=(libero_10 libero_10 libero_spatial libero_spatial \
        libero_object libero_object libero_goal libero_goal)
STARTS=(0 5 0 5 0 5 0 5)
ENDS=(5 10 5 10 5 10 5 10)
```

Every worker invocation must set:

```bash
SERVER_BACKEND=native_teacher
MODEL_NAME=teacher_native
STUDENT_CKPT="$CHECKPOINT"
TEACHER_CKPT="$CHECKPOINT"
NUM_STEPS="$steps"
ACTION_NUM_STEPS="$steps"
```

Print:

```text
WORKER model=teacher_native backend=native_teacher steps=K ...
```

Call `merge_lingbotva_4suite_results.py` after all eight workers for a budget
exit successfully, with expected model `teacher_native`, expected episodes,
and matched video/action steps.

- [ ] **Step 4: Run launcher and existing partition tests**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py \
  evaluation/libero/tests/test_run_lingbotva_4suite_124_eval_8gpu.py \
  -q
bash -n evaluation/libero/run_lingbotva_native_teacher_4suite_124_eval_8gpu.sh
```

Expected: all tests pass and `bash -n` exits zero.

- [ ] **Step 5: Commit the launcher**

```bash
git add \
  evaluation/libero/run_lingbotva_native_teacher_4suite_124_eval_8gpu.sh \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py
git commit -m "feat: add eight-GPU native teacher evaluator"
```

### Task 5: End-to-End Verification and Smoke Gate

**Files:**
- No production file changes expected
- Runtime artifacts: `/kpfs-intern/jialongliu/projects/Flash-WAM/evaluation/results/libero_teacher_native_coarse_smoke_20260726`

**Interfaces:**
- Consumes: native server, backend routing, and launcher from Tasks 1–4
- Produces: an auditable go/no-go result for the formal eight-GPU run

- [ ] **Step 1: Run the complete relevant test suite**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m pytest \
  distillation_flowmap/tests/test_native_teacher_contract.py \
  evaluation/libero/tests/test_run_eval_new_contract.py \
  evaluation/libero/tests/test_run_lingbotva_native_teacher_4suite_124_eval_8gpu.py \
  evaluation/libero/tests/test_run_lingbotva_4suite_124_eval_8gpu.py \
  distillation_flowmap/tests/test_runtime_metadata.py \
  -q
git diff --check
```

Expected: all tests pass and `git diff --check` exits zero.

- [ ] **Step 2: Run check-only formal planning**

Run:

```bash
CHECK_ONLY=1 \
SERVER_PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
CLIENT_PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python \
bash evaluation/libero/run_lingbotva_native_teacher_4suite_124_eval_8gpu.sh \
  --checkpoint /kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero/transformer \
  --episodes 50 \
  --output-root /kpfs-intern/jialongliu/projects/Flash-WAM/evaluation/results/libero_teacher_native_coarse_20260726 \
  --master-port-base 30680 \
  --ws-port-base 30780
```

Expected: exactly 24 worker lines and no model loading.

- [ ] **Step 3: Run one-task smoke jobs sequentially**

For each `K` in `1 2 4`, run:

```bash
CUDA_VISIBLE_DEVICES=0 \
SERVER_PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
CLIENT_PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python \
SERVER_BACKEND=native_teacher \
STUDENT_CKPT=/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero/transformer \
TEACHER_CKPT=/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero/transformer \
WAN22_PRETRAINED_PATH=/kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero \
LIBERO_BENCHMARK=libero_10 \
EVAL_MODE=success \
NUM_STEPS="$K" \
ACTION_NUM_STEPS="$K" \
TEST_NUM=1 \
TASK_START=0 \
TASK_END=1 \
PORT="$((31880 + K))" \
MASTER_PORT="$((31780 + K))" \
SAVE_ROOT="/kpfs-intern/jialongliu/projects/Flash-WAM/evaluation/results/libero_teacher_native_coarse_smoke_20260726/steps_${K}" \
MODEL_NAME=teacher_native \
bash evaluation/libero/run_eval_new.sh external teacher_native
```

Expected for every `K`: exit zero, one task JSON with `total_num=1`, and a
startup line beginning `NATIVE_TEACHER_CONTRACT`.

- [ ] **Step 4: Audit saved action tensors**

For the first two saved action chunks of each budget, load the tensor and
verify:

```python
assert actions.shape == (1, 30, 4, 4, 1)
assert torch.isfinite(actions).all()
assert actions[:, :7, 1:, :, :].std() > 0
```

Also inspect the native-server log and verify that no line contains
`FlowMap joint inference` or `action_downsample_factor`.

- [ ] **Step 5: Report the formal launch command**

After the smoke gate passes, hand off:

```bash
cd /kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/libero-stage2-deployment-alignment && \
SERVER_PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
CLIENT_PYTHON=/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python \
ATTN_MODE=torch \
VIDEO_ACTION_BRIDGE=0 \
bash evaluation/libero/run_lingbotva_native_teacher_4suite_124_eval_8gpu.sh \
  --checkpoint /kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero/transformer \
  --episodes 50 \
  --output-root /kpfs-intern/jialongliu/projects/Flash-WAM/evaluation/results/libero_teacher_native_coarse_20260726 \
  --master-port-base 30680 \
  --ws-port-base 30780
```

Do not launch the formal run automatically; the user controls the eight-GPU
job allocation.
