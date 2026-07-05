# Cosmos Video FlowMap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two parallel Cosmos variants: a dual-teacher FlowMap path using Cosmos for actions and WanVA for video, plus an opt-in Cosmos future-image auxiliary path.

**Architecture:** Add a role resolver so training knows which frozen model supplies action targets and which supplies video latent velocity targets. Keep current LingbotVA and action-only Cosmos configs unchanged; new behavior is enabled only by new configs or explicit env vars. The future-image path is a separate auxiliary signal built from official `future_image_predictions`, not a replacement for WanVA v-prediction.

**Tech Stack:** Python, PyTorch, WanVA transformer, Cosmos Policy adapter, pytest, existing FlowMap training/eval scripts.

---

## File Structure

- Create `distillation_flowmap/cosmos_teacher_roles.py`
  - validates and normalizes action/video teacher role settings
  - provides simple dataclass used by trainer/tests

- Modify `distillation_flowmap/flowmap_trainer.py`
  - load a separate WanVA video teacher when `teacher_backend="cosmos_policy"` and `distill_video=True`
  - expose `_video_teacher_nofsdp` without changing normal WanVA behavior

- Modify `distillation_flowmap/flowmap_step.py`
  - add `_action_teacher_model` and `_video_teacher_model`
  - route video teacher calls to `_video_teacher_model`
  - route Cosmos action target calls to `_action_teacher_model`

- Create `distillation_flowmap/config_libero_cosmos_policy_stage1_dual_teacher.py`
  - Stage 1 config for Cosmos action teacher + WanVA video teacher

- Create `distillation_flowmap/config_libero_cosmos_policy_stage2_dual_teacher.py`
  - Stage 2 config for the same roles, continuing from dual-teacher Stage 1

- Create `distillation_flowmap/cosmos_future_aux.py`
  - normalize official future image predictions
  - prepare deterministic diagnostics
  - later create latent auxiliary targets

- Create `distillation_flowmap/sweep_cosmos_checkpoints.py`
  - evaluate checkpoint lists with existing offline metrics and optional video generation
  - write JSONL and summary CSV

- Create tests:
  - `distillation_flowmap/tests/test_cosmos_teacher_roles.py`
  - `distillation_flowmap/tests/test_cosmos_future_aux.py`
  - extend `distillation_flowmap/tests/test_cosmos_policy_backend.py` only if existing config import coverage needs updating

## Task 1: Teacher Role Resolver

**Files:**
- Create: `distillation_flowmap/cosmos_teacher_roles.py`
- Test: `distillation_flowmap/tests/test_cosmos_teacher_roles.py`

- [ ] **Step 1: Write failing tests**

Create `distillation_flowmap/tests/test_cosmos_teacher_roles.py`:

```python
from types import SimpleNamespace

import pytest

from distillation_flowmap.cosmos_teacher_roles import resolve_teacher_roles


def test_action_only_cosmos_does_not_require_video_teacher():
    cfg = SimpleNamespace(
        teacher_backend="cosmos_policy",
        teacher_model_path="/ckpts/cosmos",
        distill_video=False,
    )

    roles = resolve_teacher_roles(cfg)

    assert roles.action_backend == "cosmos_policy"
    assert roles.action_model_path == "/ckpts/cosmos"
    assert roles.video_backend is None
    assert roles.video_model_path is None
    assert roles.uses_separate_video_teacher is False


def test_dual_teacher_requires_wanva_video_model_path():
    cfg = SimpleNamespace(
        teacher_backend="cosmos_policy",
        teacher_model_path="/ckpts/cosmos",
        distill_video=True,
    )

    with pytest.raises(ValueError, match="video_teacher_model_path"):
        resolve_teacher_roles(cfg)


def test_dual_teacher_defaults_video_backend_to_wanva():
    cfg = SimpleNamespace(
        teacher_backend="cosmos_policy",
        teacher_model_path="/ckpts/cosmos",
        distill_video=True,
        video_teacher_model_path="/ckpts/wanva",
    )

    roles = resolve_teacher_roles(cfg)

    assert roles.action_backend == "cosmos_policy"
    assert roles.action_model_path == "/ckpts/cosmos"
    assert roles.video_backend == "wanva"
    assert roles.video_model_path == "/ckpts/wanva"
    assert roles.uses_separate_video_teacher is True


def test_rejects_cosmos_as_video_teacher_for_latent_flowmap():
    cfg = SimpleNamespace(
        teacher_backend="cosmos_policy",
        teacher_model_path="/ckpts/cosmos",
        distill_video=True,
        video_teacher_backend="cosmos_policy",
        video_teacher_model_path="/ckpts/cosmos",
    )

    with pytest.raises(ValueError, match="Cosmos Policy cannot be used as the FlowMap video teacher"):
        resolve_teacher_roles(cfg)


def test_normal_wanva_config_keeps_single_teacher():
    cfg = SimpleNamespace(
        teacher_backend="wanva",
        teacher_model_path="/ckpts/wanva",
        distill_video=True,
    )

    roles = resolve_teacher_roles(cfg)

    assert roles.action_backend == "wanva"
    assert roles.action_model_path == "/ckpts/wanva"
    assert roles.video_backend == "wanva"
    assert roles.video_model_path == "/ckpts/wanva"
    assert roles.uses_separate_video_teacher is False
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest distillation_flowmap/tests/test_cosmos_teacher_roles.py -q
```

Expected: import error because `distillation_flowmap.cosmos_teacher_roles` does not exist.

- [ ] **Step 3: Implement resolver**

Create `distillation_flowmap/cosmos_teacher_roles.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_COSMOS_BACKENDS = {"cosmos", "cosmos_policy", "cosmos-policy"}


@dataclass(frozen=True)
class TeacherRoles:
    action_backend: str
    action_model_path: str | None
    video_backend: str | None
    video_model_path: str | None
    uses_separate_video_teacher: bool


def _get(config: Any, name: str, default: Any = None) -> Any:
    return getattr(config, name, default)


def _normalize_backend(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    return str(value).strip().lower()


def _is_cosmos_backend(value: str | None) -> bool:
    return value in _COSMOS_BACKENDS


def resolve_teacher_roles(config: Any) -> TeacherRoles:
    teacher_backend = _normalize_backend(_get(config, "teacher_backend", "wanva"), "wanva")
    action_backend = _normalize_backend(
        _get(config, "action_teacher_backend", teacher_backend),
        teacher_backend,
    )
    distill_video = bool(_get(config, "distill_video", True))

    action_model_path = _get(config, "action_teacher_model_path", None)
    if action_model_path is None:
        action_model_path = _get(config, "teacher_model_path", None)

    video_backend = _normalize_backend(_get(config, "video_teacher_backend", None), None)
    video_model_path = _get(config, "video_teacher_model_path", None)

    if not distill_video:
        return TeacherRoles(
            action_backend=action_backend,
            action_model_path=action_model_path,
            video_backend=None,
            video_model_path=None,
            uses_separate_video_teacher=False,
        )

    if _is_cosmos_backend(action_backend):
        if video_backend is None:
            video_backend = "wanva"
        if _is_cosmos_backend(video_backend):
            raise ValueError(
                "Cosmos Policy cannot be used as the FlowMap video teacher; "
                "provide a WanVA video_teacher_model_path."
            )
        if not video_model_path:
            raise ValueError(
                "distill_video=True with teacher_backend='cosmos_policy' requires "
                "cfg.video_teacher_model_path pointing to a WanVA/LingbotVA checkpoint."
            )
        return TeacherRoles(
            action_backend=action_backend,
            action_model_path=action_model_path,
            video_backend=video_backend,
            video_model_path=video_model_path,
            uses_separate_video_teacher=True,
        )

    if video_backend is None:
        video_backend = teacher_backend
    if video_model_path is None:
        video_model_path = _get(config, "teacher_model_path", None)
    return TeacherRoles(
        action_backend=action_backend,
        action_model_path=action_model_path,
        video_backend=video_backend,
        video_model_path=video_model_path,
        uses_separate_video_teacher=False,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest distillation_flowmap/tests/test_cosmos_teacher_roles.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add distillation_flowmap/cosmos_teacher_roles.py distillation_flowmap/tests/test_cosmos_teacher_roles.py
git commit -m "feat: add cosmos teacher role resolver"
```

## Task 2: Dual-Teacher Loading and Accessors

**Files:**
- Modify: `distillation_flowmap/flowmap_trainer.py`
- Modify: `distillation_flowmap/flowmap_step.py`
- Test: `distillation_flowmap/tests/test_cosmos_teacher_roles.py`

- [ ] **Step 1: Add tests for trainer-safe role behavior**

Append to `distillation_flowmap/tests/test_cosmos_teacher_roles.py`:

```python
def test_dual_teacher_uses_student_base_as_default_video_teacher_when_requested():
    cfg = SimpleNamespace(
        teacher_backend="cosmos_policy",
        teacher_model_path="/ckpts/cosmos",
        student_base_model_path="/ckpts/wanva",
        distill_video=True,
        use_student_base_as_video_teacher=True,
    )

    roles = resolve_teacher_roles(cfg)

    assert roles.video_backend == "wanva"
    assert roles.video_model_path == "/ckpts/wanva"
    assert roles.uses_separate_video_teacher is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest distillation_flowmap/tests/test_cosmos_teacher_roles.py::test_dual_teacher_uses_student_base_as_default_video_teacher_when_requested -q
```

Expected: failure because the resolver does not read `use_student_base_as_video_teacher`.

- [ ] **Step 3: Extend resolver for the default video teacher path**

In `distillation_flowmap/cosmos_teacher_roles.py`, before checking `not video_model_path`, add:

```python
        if not video_model_path and bool(_get(config, "use_student_base_as_video_teacher", False)):
            video_model_path = _get(config, "student_base_model_path", None)
```

- [ ] **Step 4: Run resolver tests**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest distillation_flowmap/tests/test_cosmos_teacher_roles.py -q
```

Expected: all resolver tests pass.

- [ ] **Step 5: Import resolver in trainer**

In `distillation_flowmap/flowmap_trainer.py`, near the other imports, add:

```python
from distillation_flowmap.cosmos_teacher_roles import resolve_teacher_roles
```

In `FlowMapDistiller.__init__`, after `self.is_cosmos_policy_teacher`, add:

```python
        self.teacher_roles = resolve_teacher_roles(config)
        self._video_teacher_nofsdp = None
        self.video_teacher = None
```

Add rank-0 logging after the existing teacher-backend logging:

```python
                logger.info(f"Action teacher backend: {self.teacher_roles.action_backend}")
                logger.info(f"Video teacher backend: {self.teacher_roles.video_backend}")
                logger.info(f"Video teacher path: {self.teacher_roles.video_model_path}")
```

- [ ] **Step 6: Load separate WanVA video teacher for Cosmos dual-teacher**

In `distillation_flowmap/flowmap_trainer.py`, after the Cosmos action teacher block that creates `self._teacher_nofsdp`, insert:

```python
            if self.teacher_roles.uses_separate_video_teacher:
                video_root = os.path.abspath(os.path.expanduser(self.teacher_roles.video_model_path))
                if os.path.basename(video_root) == "transformer":
                    video_teacher_path = video_root
                else:
                    video_teacher_path = os.path.join(video_root, "transformer")
                if not os.path.isfile(os.path.join(video_teacher_path, "config.json")):
                    raise FileNotFoundError(
                        "Invalid video_teacher_model_path for Cosmos dual-teacher mode: "
                        f"expected {os.path.join(video_teacher_path, 'config.json')}"
                    )
                logger.info("Loading WanVA video teacher for Cosmos dual-teacher mode ...")
                self.video_teacher = load_transformer(
                    video_teacher_path,
                    torch_dtype=self.dtype,
                    torch_device="cpu",
                )
                self.video_teacher.requires_grad_(False)
                self.video_teacher.eval()
                self.video_teacher = self.video_teacher.to(self.dtype)
                import copy
                self._video_teacher_nofsdp = copy.deepcopy(self.video_teacher)
                self._video_teacher_nofsdp = self._video_teacher_nofsdp.to(self.dtype)
                self._video_teacher_nofsdp.eval()
                for p in self._video_teacher_nofsdp.parameters():
                    p.requires_grad_(False)
                self._video_teacher_nofsdp = self._video_teacher_nofsdp.to(f"cuda:{local_rank}")
                del self.video_teacher
                self.video_teacher = None
                logger.info("WanVA video teacher ready for Cosmos dual-teacher mode.")
```

- [ ] **Step 7: Add role-specific accessors**

In `distillation_flowmap/flowmap_step.py`, replace the existing `_teacher_model` property area with:

```python
    @property
    def _teacher_model(self):
        """Return the default frozen teacher for backward compatibility."""
        return getattr(self, '_teacher_nofsdp', self.teacher)

    @property
    def _action_teacher_model(self):
        """Return the teacher that supplies action targets."""
        return getattr(self, '_teacher_nofsdp', self.teacher)

    @property
    def _video_teacher_model(self):
        """Return the teacher that supplies WanVA latent video targets."""
        video_teacher = getattr(self, '_video_teacher_nofsdp', None)
        if video_teacher is not None:
            return video_teacher
        return self._teacher_model
```

- [ ] **Step 8: Route video teacher calls**

In `distillation_flowmap/flowmap_step.py`, replace video-teacher forward calls:

```python
self._teacher_model(input_plus, train_mode=True)
self._teacher_model(input_minus, train_mode=True)
self._teacher_model(cond_input, train_mode=True)
self._teacher_model(uncond_input, train_mode=True)
self._teacher_model(input_dict, train_mode=True)
self._teacher_model(doubled_input, train_mode=True)
```

with:

```python
self._video_teacher_model(input_plus, train_mode=True)
self._video_teacher_model(input_minus, train_mode=True)
self._video_teacher_model(cond_input, train_mode=True)
self._video_teacher_model(uncond_input, train_mode=True)
self._video_teacher_model(input_dict, train_mode=True)
self._video_teacher_model(doubled_input, train_mode=True)
```

Do not change the Cosmos action-only block yet.

- [ ] **Step 9: Route action teacher call**

In `distillation_flowmap/flowmap_step.py`, change:

```python
        teacher = self._teacher_model
```

to:

```python
        teacher = self._action_teacher_model
```

- [ ] **Step 10: Run focused tests**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest distillation_flowmap/tests/test_cosmos_teacher_roles.py distillation_flowmap/tests/test_cosmos_policy_backend.py -q
```

Expected: all tests pass.

- [ ] **Step 11: Commit**

```bash
git add distillation_flowmap/cosmos_teacher_roles.py distillation_flowmap/flowmap_trainer.py distillation_flowmap/flowmap_step.py distillation_flowmap/tests/test_cosmos_teacher_roles.py
git commit -m "feat: wire cosmos dual teacher roles"
```

## Task 3: Dual-Teacher Configs

**Files:**
- Create: `distillation_flowmap/config_libero_cosmos_policy_stage1_dual_teacher.py`
- Create: `distillation_flowmap/config_libero_cosmos_policy_stage2_dual_teacher.py`
- Test: `distillation_flowmap/tests/test_cosmos_policy_backend.py`

- [ ] **Step 1: Add config import tests**

Append to `distillation_flowmap/tests/test_cosmos_policy_backend.py`:

```python
def test_cosmos_dual_teacher_stage1_config_imports():
    import importlib

    cfg = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage1_dual_teacher"
    ).cfg

    assert cfg.teacher_backend == "cosmos_policy"
    assert cfg.action_teacher_backend == "cosmos_policy"
    assert cfg.video_teacher_backend == "wanva"
    assert cfg.distill_video is True
    assert cfg.distill_action is True
    assert cfg.action_use_flowmap is True
    assert cfg.diffusion_ratio == 0.5
    assert cfg.consistency_ratio == 0.25
    assert cfg.flowmap_ratio == 0.25


def test_cosmos_dual_teacher_stage2_config_imports():
    import importlib

    cfg = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_dual_teacher"
    ).cfg

    assert cfg.teacher_backend == "cosmos_policy"
    assert cfg.action_teacher_backend == "cosmos_policy"
    assert cfg.video_teacher_backend == "wanva"
    assert cfg.distill_video is True
    assert cfg.distill_action is True
    assert cfg.action_use_flowmap is True
    assert cfg.use_opd_aux is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest distillation_flowmap/tests/test_cosmos_policy_backend.py::test_cosmos_dual_teacher_stage1_config_imports distillation_flowmap/tests/test_cosmos_policy_backend.py::test_cosmos_dual_teacher_stage2_config_imports -q
```

Expected: import errors because config modules do not exist.

- [ ] **Step 3: Add Stage 1 dual-teacher config**

Create `distillation_flowmap/config_libero_cosmos_policy_stage1_dual_teacher.py`:

```python
"""Stage 1 LIBERO FlowMap using Cosmos for actions and WanVA for video."""
import copy
import os

from distillation_flowmap.config_libero_fullfinetune_stage1_warmup import cfg as _base_cfg

cfg = copy.deepcopy(_base_cfg)
_this_dir = os.path.dirname(os.path.abspath(__file__))


def _env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


cfg.teacher_backend = "cosmos_policy"
cfg.action_teacher_backend = "cosmos_policy"
cfg.teacher_model_path = os.environ.get(
    "COSMOS_POLICY_PATH",
    "/root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
)
cfg.student_base_model_path = os.environ.get(
    "STUDENT_BASE_MODEL_PATH",
    "/root/nas/junjie/jj/Any_WAM/checkpoints/lingbot-va-posttrain-libero",
)
cfg.video_teacher_backend = "wanva"
cfg.use_student_base_as_video_teacher = _env_bool("USE_STUDENT_BASE_AS_VIDEO_TEACHER", True)
cfg.video_teacher_model_path = os.environ.get(
    "VIDEO_TEACHER_MODEL_PATH",
    cfg.student_base_model_path,
)

cfg.cosmos_policy_validate_weights = _env_bool("COSMOS_POLICY_VALIDATE_WEIGHTS", False)
cfg.cosmos_policy_use_raw_inference = _env_bool("COSMOS_POLICY_USE_RAW_INFERENCE", True)
cfg.return_raw_observation = cfg.cosmos_policy_use_raw_inference
if cfg.cosmos_policy_use_raw_inference:
    cfg.cache_dataset_in_memory = _env_bool("CACHE_DATASET_IN_MEMORY", False)
cfg.raw_primary_image_key = os.environ.get(
    "COSMOS_POLICY_PRIMARY_IMAGE_KEY", "observation.images.agentview_rgb")
cfg.raw_wrist_image_key = os.environ.get(
    "COSMOS_POLICY_WRIST_IMAGE_KEY", "observation.images.eye_in_hand_rgb")
cfg.cosmos_policy_inference_mode = os.environ.get("COSMOS_POLICY_INFERENCE_MODE", "subprocess")
cfg.cosmos_policy_repo = os.environ.get(
    "COSMOS_PREDICT2_REPO",
    "/root/nas/junjie/cosmos_predict2_5/repos/cosmos-predict2.5",
)
cfg.cosmos_policy_python = os.environ.get(
    "COSMOS_POLICY_PYTHON",
    "/root/nas/junjie/cosmos_predict2_5/envs/predict2_py310/bin/python",
)
cfg.cosmos_policy_extra_pythonpath = os.environ.get(
    "COSMOS_POLICY_EXTRA_PYTHONPATH",
    "/root/nas/junjie/conda_envs/any_wam/lib/python3.10/site-packages",
)
cfg.cosmos_policy_local_model_dir = os.environ.get(
    "COSMOS_PREDICT25_LOCAL_MODEL_DIR",
    "/root/nas/junjie/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World",
)
cfg.cosmos_policy_config_name = os.environ.get(
    "COSMOS_POLICY_CONFIG_NAME", "cosmos_predict2_2b_480p_libero__inference_only")
cfg.cosmos_policy_config_file = os.environ.get(
    "COSMOS_POLICY_CONFIG_FILE",
    "cosmos_predict2/_src/predict2/cosmos_policy/config/config.py",
)
cfg.cosmos_policy_num_denoising_steps_action = int(
    os.environ.get("COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION", 5))
cfg.cosmos_policy_seed = int(os.environ.get("COSMOS_POLICY_SEED", 1))

cfg.output_dir = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_this_dir, "output_libero_cosmos_policy_stage1_dual_teacher"),
)
cfg.wandb_name_prefix = "stage1_cosmos_dual_teacher"
cfg.enable_wandb = _env_bool("ENABLE_WANDB", False)

cfg.distill_mode = "flashwam"
cfg.distill_video = True
cfg.distill_action = True
cfg.action_aware = True
cfg.use_action_distill = True
cfg.use_gt_regression = True
cfg.use_central_diff = True
cfg.action_use_flowmap = True

cfg.diffusion_ratio = float(os.environ.get("DIFFUSION_RATIO", 0.5))
cfg.consistency_ratio = float(os.environ.get("CONSISTENCY_RATIO", 0.25))
cfg.flowmap_ratio = float(os.environ.get("FLOWMAP_RATIO", 0.25))

cfg.action_loss_weight = float(os.environ.get("ACTION_LOSS_WEIGHT", cfg.action_loss_weight))
cfg.action_block_weight = float(os.environ.get("ACTION_BLOCK_WEIGHT", cfg.action_block_weight))
cfg.action_aware_weight = float(os.environ.get("ACTION_AWARE_WEIGHT", cfg.action_aware_weight))
cfg.gt_regression_weight = float(os.environ.get("GT_REGRESSION_WEIGHT", cfg.gt_regression_weight))
cfg.learning_rate = float(os.environ.get("LEARNING_RATE", 1e-6))
cfg.max_train_steps = int(os.environ.get("MAX_TRAIN_STEPS", 5000))
cfg.save_interval = int(os.environ.get("SAVE_INTERVAL", 1000))
cfg.skip_teacher_compile = _env_bool("SKIP_TEACHER_COMPILE", True)
cfg.gradient_checkpointing = _env_bool("GRADIENT_CHECKPOINTING", False)
```

- [ ] **Step 4: Add Stage 2 dual-teacher config**

Create `distillation_flowmap/config_libero_cosmos_policy_stage2_dual_teacher.py`:

```python
"""Stage 2 LIBERO FlowMap using Cosmos for actions and WanVA for video."""
import copy
import os

from distillation_flowmap.config_libero_fullfinetune_stage2_anyflow import cfg as _base_cfg

cfg = copy.deepcopy(_base_cfg)
_this_dir = os.path.dirname(os.path.abspath(__file__))


def _env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


_stage1_ckpt = os.path.join(
    _this_dir,
    "output_libero_cosmos_policy_stage1_dual_teacher",
    "checkpoints",
    "step_5000",
)

cfg.teacher_backend = "cosmos_policy"
cfg.action_teacher_backend = "cosmos_policy"
cfg.teacher_model_path = os.environ.get(
    "COSMOS_POLICY_PATH",
    "/root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
)
cfg.student_base_model_path = os.environ.get(
    "STUDENT_BASE_MODEL_PATH",
    "/root/nas/junjie/jj/Any_WAM/checkpoints/lingbot-va-posttrain-libero",
)
cfg.video_teacher_backend = "wanva"
cfg.use_student_base_as_video_teacher = _env_bool("USE_STUDENT_BASE_AS_VIDEO_TEACHER", True)
cfg.video_teacher_model_path = os.environ.get(
    "VIDEO_TEACHER_MODEL_PATH",
    cfg.student_base_model_path,
)

cfg.cosmos_policy_validate_weights = _env_bool("COSMOS_POLICY_VALIDATE_WEIGHTS", False)
cfg.cosmos_policy_use_raw_inference = _env_bool("COSMOS_POLICY_USE_RAW_INFERENCE", True)
cfg.return_raw_observation = cfg.cosmos_policy_use_raw_inference
if cfg.cosmos_policy_use_raw_inference:
    cfg.cache_dataset_in_memory = _env_bool("CACHE_DATASET_IN_MEMORY", False)
cfg.raw_primary_image_key = os.environ.get(
    "COSMOS_POLICY_PRIMARY_IMAGE_KEY", "observation.images.agentview_rgb")
cfg.raw_wrist_image_key = os.environ.get(
    "COSMOS_POLICY_WRIST_IMAGE_KEY", "observation.images.eye_in_hand_rgb")
cfg.cosmos_policy_inference_mode = os.environ.get("COSMOS_POLICY_INFERENCE_MODE", "subprocess")
cfg.cosmos_policy_repo = os.environ.get(
    "COSMOS_PREDICT2_REPO",
    "/root/nas/junjie/cosmos_predict2_5/repos/cosmos-predict2.5",
)
cfg.cosmos_policy_python = os.environ.get(
    "COSMOS_POLICY_PYTHON",
    "/root/nas/junjie/cosmos_predict2_5/envs/predict2_py310/bin/python",
)
cfg.cosmos_policy_extra_pythonpath = os.environ.get(
    "COSMOS_POLICY_EXTRA_PYTHONPATH",
    "/root/nas/junjie/conda_envs/any_wam/lib/python3.10/site-packages",
)
cfg.cosmos_policy_local_model_dir = os.environ.get(
    "COSMOS_PREDICT25_LOCAL_MODEL_DIR",
    "/root/nas/junjie/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World",
)
cfg.cosmos_policy_config_name = os.environ.get(
    "COSMOS_POLICY_CONFIG_NAME", "cosmos_predict2_2b_480p_libero__inference_only")
cfg.cosmos_policy_config_file = os.environ.get(
    "COSMOS_POLICY_CONFIG_FILE",
    "cosmos_predict2/_src/predict2/cosmos_policy/config/config.py",
)
cfg.cosmos_policy_num_denoising_steps_action = int(
    os.environ.get("COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION", 5))
cfg.cosmos_policy_seed = int(os.environ.get("COSMOS_POLICY_SEED", 1))

cfg.resume_from_path = os.environ.get("RESUME_FROM_PATH", _stage1_ckpt)
cfg.resume_online_from_target = _env_bool("RESUME_ONLINE_FROM_TARGET", True)
cfg.reset_resume_step = _env_bool("RESET_RESUME_STEP", True)
cfg.output_dir = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_this_dir, "output_libero_cosmos_policy_stage2_dual_teacher"),
)
cfg.wandb_name_prefix = "stage2_cosmos_dual_teacher"
cfg.enable_wandb = _env_bool("ENABLE_WANDB", False)

cfg.distill_mode = "flashwam"
cfg.distill_video = True
cfg.distill_action = True
cfg.action_aware = True
cfg.use_action_distill = True
cfg.use_gt_regression = True
cfg.use_central_diff = True
cfg.action_use_flowmap = True

cfg.diffusion_ratio = float(os.environ.get("DIFFUSION_RATIO", cfg.diffusion_ratio))
cfg.consistency_ratio = float(os.environ.get("CONSISTENCY_RATIO", cfg.consistency_ratio))
cfg.flowmap_ratio = float(os.environ.get("FLOWMAP_RATIO", cfg.flowmap_ratio))
cfg.use_opd_aux = _env_bool("USE_OPD_AUX", True)
cfg.opd_aux_action = _env_bool("OPD_AUX_ACTION", False)
cfg.learning_rate = float(os.environ.get("LEARNING_RATE", 5e-7))
cfg.max_train_steps = int(os.environ.get("MAX_TRAIN_STEPS", 5000))
cfg.save_interval = int(os.environ.get("SAVE_INTERVAL", 1000))
cfg.skip_teacher_compile = _env_bool("SKIP_TEACHER_COMPILE", True)
cfg.gradient_checkpointing = _env_bool("GRADIENT_CHECKPOINTING", False)
```

- [ ] **Step 5: Run config tests**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest distillation_flowmap/tests/test_cosmos_policy_backend.py -q
```

Expected: all config/backend tests pass.

- [ ] **Step 6: Commit**

```bash
git add distillation_flowmap/config_libero_cosmos_policy_stage1_dual_teacher.py distillation_flowmap/config_libero_cosmos_policy_stage2_dual_teacher.py distillation_flowmap/tests/test_cosmos_policy_backend.py
git commit -m "feat: add cosmos dual teacher configs"
```

## Task 4: Checkpoint Sweep Script

**Files:**
- Create: `distillation_flowmap/sweep_cosmos_checkpoints.py`
- Test: `distillation_flowmap/tests/test_cosmos_checkpoint_sweep.py`

- [ ] **Step 1: Write parser tests**

Create `distillation_flowmap/tests/test_cosmos_checkpoint_sweep.py`:

```python
from pathlib import Path

from distillation_flowmap.sweep_cosmos_checkpoints import (
    discover_checkpoints,
    parse_step,
)


def test_parse_step_from_checkpoint_dir():
    assert parse_step(Path("/tmp/run/checkpoints/step_5000")) == 5000
    assert parse_step(Path("/tmp/run/checkpoints/latest")) is None


def test_discover_checkpoints_sorted_by_step(tmp_path):
    for name in ["step_1000", "step_50", "step_5000"]:
        (tmp_path / "checkpoints" / name).mkdir(parents=True)

    paths = discover_checkpoints(tmp_path)

    assert [p.name for p in paths] == ["step_50", "step_1000", "step_5000"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest distillation_flowmap/tests/test_cosmos_checkpoint_sweep.py -q
```

Expected: import error because script does not exist.

- [ ] **Step 3: Implement script skeleton**

Create `distillation_flowmap/sweep_cosmos_checkpoints.py`:

```python
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


_STEP_RE = re.compile(r"^step_(\d+)$")


def parse_step(path: Path) -> int | None:
    match = _STEP_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def discover_checkpoints(run_dir: Path) -> list[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    paths = [p for p in ckpt_dir.iterdir() if p.is_dir() and parse_step(p) is not None]
    return sorted(paths, key=lambda p: parse_step(p) or -1)


def _load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_metrics(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten_metrics(f"{prefix}/{key}" if prefix else str(key), child, out)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        out[prefix] = value


def write_summary(rows: Iterable[dict], output_csv: Path) -> None:
    rows = list(rows)
    keys = sorted({key for row in rows for key in row})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_offline_eval(args, checkpoint: Path, output_json: Path) -> None:
    cmd = [
        sys.executable,
        "distillation_flowmap/eval_cosmos_policy_stage1_metrics.py",
        "--checkpoint",
        str(checkpoint),
        "--output-json",
        str(output_json),
        "--num-batches",
        str(args.num_batches),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.config:
        cmd.extend(["--config", args.config])
    env = os.environ.copy()
    if args.dry_run:
        print("DRY_RUN", " ".join(cmd), flush=True)
        return
    subprocess.run(cmd, check=True, env=env)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir_s in args.run_dir:
        run_dir = Path(run_dir_s)
        for checkpoint in discover_checkpoints(run_dir):
            step = parse_step(checkpoint)
            metrics_json = output_dir / f"{run_dir.name}_step_{step}.json"
            run_offline_eval(args, checkpoint, metrics_json)
            row = {
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "step": step,
            }
            if metrics_json.exists():
                flat = {}
                _flatten_metrics("", _load_metrics(metrics_json), flat)
                row.update(flat)
            rows.append(row)
    write_summary(rows, output_dir / "summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run parser tests**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest distillation_flowmap/tests/test_cosmos_checkpoint_sweep.py -q
```

Expected: parser/discovery tests pass.

- [ ] **Step 5: Dry-run existing checkpoints**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python distillation_flowmap/sweep_cosmos_checkpoints.py \
  --run-dir distillation_flowmap/output_libero_cosmos_policy_stage1 \
  --output-dir distillation_flowmap/logs/cosmos_sweep_dryrun \
  --num-batches 1 \
  --batch-size 1 \
  --dry-run
```

Expected: prints one eval command per discovered checkpoint and writes `summary.csv`.

- [ ] **Step 6: Commit**

```bash
git add distillation_flowmap/sweep_cosmos_checkpoints.py distillation_flowmap/tests/test_cosmos_checkpoint_sweep.py
git commit -m "feat: add cosmos checkpoint sweep script"
```

## Task 5: Cosmos Future-Image Normalizer

**Files:**
- Create: `distillation_flowmap/cosmos_future_aux.py`
- Test: `distillation_flowmap/tests/test_cosmos_future_aux.py`

- [ ] **Step 1: Write normalizer tests**

Create `distillation_flowmap/tests/test_cosmos_future_aux.py`:

```python
import torch

from distillation_flowmap.cosmos_future_aux import normalize_future_images


def test_normalize_future_images_from_dict_uint8():
    primary = torch.zeros(2, 4, 16, 16, 3, dtype=torch.uint8)
    wrist = torch.full((2, 4, 16, 16, 3), 255, dtype=torch.uint8)
    result = normalize_future_images(
        {
            "observation.images.agentview_rgb": primary,
            "observation.images.eye_in_hand_rgb": wrist,
        },
        primary_key="observation.images.agentview_rgb",
        wrist_key="observation.images.eye_in_hand_rgb",
    )

    assert result.primary.shape == (2, 4, 3, 16, 16)
    assert result.wrist.shape == (2, 4, 3, 16, 16)
    assert result.primary.dtype == torch.float32
    assert result.primary.max().item() == 0.0
    assert result.wrist.min().item() == 1.0


def test_normalize_future_images_from_single_tensor():
    tensor = torch.rand(2, 4, 3, 16, 16)

    result = normalize_future_images(tensor)

    assert result.primary.shape == (2, 4, 3, 16, 16)
    assert result.wrist is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest distillation_flowmap/tests/test_cosmos_future_aux.py -q
```

Expected: import error because module does not exist.

- [ ] **Step 3: Implement normalizer**

Create `distillation_flowmap/cosmos_future_aux.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class NormalizedFutureImages:
    primary: torch.Tensor
    wrist: torch.Tensor | None = None


def _to_btc_hw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 5:
        raise ValueError(f"Expected future image tensor with 5 dims, got shape={tuple(tensor.shape)}")
    if tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(0, 1, 4, 2, 3)
    elif tensor.shape[2] in (1, 3):
        tensor = tensor
    else:
        raise ValueError(f"Cannot infer channel dimension from shape={tuple(tensor.shape)}")
    tensor = tensor.to(torch.float32)
    if tensor.numel() > 0 and tensor.max() > 2.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0).contiguous()


def _select(mapping: Mapping[str, torch.Tensor], key: str | None) -> torch.Tensor | None:
    if key and key in mapping:
        return mapping[key]
    if key:
        short_key = key.split(".")[-1]
        if short_key in mapping:
            return mapping[short_key]
    return None


def normalize_future_images(
    future_image_predictions,
    *,
    primary_key: str | None = None,
    wrist_key: str | None = None,
) -> NormalizedFutureImages:
    if isinstance(future_image_predictions, torch.Tensor):
        return NormalizedFutureImages(primary=_to_btc_hw(future_image_predictions))

    if not isinstance(future_image_predictions, Mapping):
        raise TypeError(
            "future_image_predictions must be a tensor or mapping of camera name to tensor"
        )

    primary = _select(future_image_predictions, primary_key)
    if primary is None:
        for value in future_image_predictions.values():
            if isinstance(value, torch.Tensor):
                primary = value
                break
    if primary is None:
        raise ValueError("No tensor found in future_image_predictions")

    wrist = _select(future_image_predictions, wrist_key)
    return NormalizedFutureImages(
        primary=_to_btc_hw(primary),
        wrist=_to_btc_hw(wrist) if wrist is not None else None,
    )
```

- [ ] **Step 4: Run future aux tests**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest distillation_flowmap/tests/test_cosmos_future_aux.py -q
```

Expected: normalizer tests pass.

- [ ] **Step 5: Commit**

```bash
git add distillation_flowmap/cosmos_future_aux.py distillation_flowmap/tests/test_cosmos_future_aux.py
git commit -m "feat: normalize cosmos future image predictions"
```

## Task 6: Future-Image Diagnostic Metrics

**Files:**
- Modify: `distillation_flowmap/eval_cosmos_policy_stage1_metrics.py`
- Modify: `distillation_flowmap/cosmos_future_aux.py`
- Test: `distillation_flowmap/tests/test_cosmos_future_aux.py`

- [ ] **Step 1: Add metric helper test**

Append to `distillation_flowmap/tests/test_cosmos_future_aux.py`:

```python
from distillation_flowmap.cosmos_future_aux import image_l1_mse


def test_image_l1_mse_uses_common_shape():
    a = torch.zeros(1, 2, 3, 8, 8)
    b = torch.ones(1, 2, 3, 8, 8)

    metrics = image_l1_mse(a, b)

    assert metrics["l1"] == 1.0
    assert metrics["mse"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest distillation_flowmap/tests/test_cosmos_future_aux.py::test_image_l1_mse_uses_common_shape -q
```

Expected: import error for `image_l1_mse`.

- [ ] **Step 3: Implement image metrics helper**

Append to `distillation_flowmap/cosmos_future_aux.py`:

```python
def image_l1_mse(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if pred.shape != target.shape:
        raise ValueError(f"Image metric shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    diff = pred.to(torch.float32) - target.to(torch.float32)
    return {
        "l1": float(diff.abs().mean().item()),
        "mse": float((diff * diff).mean().item()),
    }
```

- [ ] **Step 4: Wire diagnostic metrics in eval script**

In `distillation_flowmap/eval_cosmos_policy_stage1_metrics.py`, import:

```python
from distillation_flowmap.cosmos_future_aux import (
    image_l1_mse,
    normalize_future_images,
)
```

When the teacher result is requested with `include_future=True`, add:

```python
future = result.get("future_image_predictions") if isinstance(result, dict) else None
if future is not None:
    normalized_future = normalize_future_images(
        future,
        primary_key=getattr(cfg, "raw_primary_image_key", None),
        wrist_key=getattr(cfg, "raw_wrist_image_key", None),
    )
    metrics["cosmos_future/primary_abs_mean"] = float(
        normalized_future.primary.abs().mean().item()
    )
    if normalized_future.wrist is not None:
        metrics["cosmos_future/wrist_abs_mean"] = float(
            normalized_future.wrist.abs().mean().item()
        )
```

Keep this diagnostic independent of the action metric loop. If raw future predictions are unavailable, do not fail the eval.

- [ ] **Step 5: Run tests**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest distillation_flowmap/tests/test_cosmos_future_aux.py distillation_flowmap/tests/test_cosmos_policy_backend.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add distillation_flowmap/cosmos_future_aux.py distillation_flowmap/eval_cosmos_policy_stage1_metrics.py distillation_flowmap/tests/test_cosmos_future_aux.py
git commit -m "feat: add cosmos future image diagnostics"
```

## Task 7: Opt-In Future Latent Auxiliary

**Files:**
- Modify: `distillation_flowmap/flowmap_trainer.py`
- Modify: `distillation_flowmap/flowmap_step.py`
- Modify: `distillation_flowmap/cosmos_future_aux.py`
- Test: `distillation_flowmap/tests/test_cosmos_future_aux.py`

- [ ] **Step 1: Add config guards**

In `distillation_flowmap/flowmap_trainer.py`, after teacher role setup, add:

```python
        self.cosmos_future_aux_weight = float(getattr(config, "cosmos_future_aux_weight", 0.0))
        self.cosmos_future_aux_interval = int(getattr(config, "cosmos_future_aux_interval", 16))
        if self.cosmos_future_aux_weight > 0.0:
            if not self.is_cosmos_policy_teacher:
                raise ValueError("cosmos_future_aux_weight>0 requires teacher_backend='cosmos_policy'.")
            if not bool(getattr(config, "cosmos_policy_use_raw_inference", False)):
                raise ValueError("cosmos_future_aux_weight>0 requires cosmos_policy_use_raw_inference=True.")
            if not self.distill_video:
                raise ValueError("cosmos_future_aux_weight>0 requires distill_video=True.")
```

- [ ] **Step 2: Implement latent target helper as a pure shape helper first**

Add tests that validate frame selection and camera concatenation. Do not call the heavy VAE in unit tests.

```python
from distillation_flowmap.cosmos_future_aux import select_future_frames


def test_select_future_frames_clamps_to_available_frames():
    tensor = torch.arange(1 * 3 * 1 * 1 * 1).reshape(1, 3, 1, 1, 1).float()

    selected = select_future_frames(tensor, num_frames=5)

    assert selected.shape[1] == 5
    assert selected[0, -1, 0, 0, 0].item() == tensor[0, -1, 0, 0, 0].item()
```

Implement:

```python
def select_future_frames(images: torch.Tensor, num_frames: int) -> torch.Tensor:
    if images.ndim != 5:
        raise ValueError(f"Expected [B,T,C,H,W], got {tuple(images.shape)}")
    if images.shape[1] >= num_frames:
        return images[:, :num_frames].contiguous()
    pad = images[:, -1:].expand(-1, num_frames - images.shape[1], -1, -1, -1)
    return torch.cat([images, pad], dim=1).contiguous()
```

- [ ] **Step 3: Add training loss with default weight zero**

In `flowmap_step.py`, after the student clean latent estimate is available in the main video path, add a guarded call:

```python
        future_aux_loss = torch.tensor(0.0, device=self.device)
        if (
            getattr(self, "cosmos_future_aux_weight", 0.0) > 0.0
            and batch_idx % max(getattr(self, "cosmos_future_aux_interval", 1), 1) == 0
        ):
            future_aux_loss = self._compute_cosmos_future_aux_loss(
                batch=batch,
                student_x0=student_video_pred_x0,
                ref_shape=video_latents.shape,
            )
            total_loss = total_loss + self.cosmos_future_aux_weight * future_aux_loss
            metrics["cosmos_future_aux_loss"] = future_aux_loss.detach()
```

The helper `_compute_cosmos_future_aux_loss` should:

- call `self._action_teacher_model.predict_raw_action_result(..., include_future=True)`
- normalize official future images
- select configured future frames
- encode to WanVA latents using the existing VAE helper already used by video eval/training
- compare to `student_x0` with L1

If there is no reusable VAE encode helper, stop this task after adding the guards and shape helper, then extract the existing encode path in a separate commit. Do not duplicate large VAE encode code inside `flowmap_step.py`.

- [ ] **Step 4: Run smoke with weight zero**

Run a 1-step config import/training smoke with:

```bash
CONFIG_FILE=distillation_flowmap.config_libero_cosmos_policy_stage1_dual_teacher \
MAX_TRAIN_STEPS=1 \
SAVE_INTERVAL=1 \
COSMOS_FUTURE_AUX_WEIGHT=0 \
/root/nas/junjie/conda_envs/any_wam/bin/python -m torch.distributed.run --nproc_per_node=1 distillation_flowmap/train.py \
  --output-dir distillation_flowmap/output_smoke_cosmos_dual_teacher
```

Expected: the smoke reaches `step_1` and saves a checkpoint.

- [ ] **Step 5: Commit**

```bash
git add distillation_flowmap/flowmap_trainer.py distillation_flowmap/flowmap_step.py distillation_flowmap/cosmos_future_aux.py distillation_flowmap/tests/test_cosmos_future_aux.py
git commit -m "feat: add opt-in cosmos future latent auxiliary"
```

## Task 8: Smoke Runs and First Comparison

**Files:**
- No code unless a smoke exposes a bug
- Outputs: `distillation_flowmap/logs/`

- [ ] **Step 1: Run unit test suite subset**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest \
  distillation_flowmap/tests/test_cosmos_teacher_roles.py \
  distillation_flowmap/tests/test_cosmos_future_aux.py \
  distillation_flowmap/tests/test_cosmos_checkpoint_sweep.py \
  distillation_flowmap/tests/test_cosmos_policy_backend.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 2: Run dual-teacher Stage 1 one-step smoke**

Run in tmux:

```bash
tmux new -d -s cosmos_dual_stage1_smoke \
  'cd /root/nas/junjie/jj/Any_WAM && \
   CONFIG_FILE=distillation_flowmap.config_libero_cosmos_policy_stage1_dual_teacher \
   MAX_TRAIN_STEPS=1 SAVE_INTERVAL=1 ENABLE_WANDB=0 \
   /root/nas/junjie/conda_envs/any_wam/bin/python -m torch.distributed.run --nproc_per_node=1 distillation_flowmap/train.py \
     --output-dir distillation_flowmap/output_smoke_cosmos_dual_teacher \
   2>&1 | tee distillation_flowmap/logs/cosmos_dual_stage1_smoke_$(date +%Y%m%d_%H%M%S).log'
```

Expected: checkpoint `step_1` is produced and logs mention both `Action teacher backend: cosmos_policy` and `Video teacher backend: wanva`.

- [ ] **Step 3: Run dry-run sweep**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python distillation_flowmap/sweep_cosmos_checkpoints.py \
  --run-dir distillation_flowmap/output_libero_cosmos_policy_stage1 \
  --run-dir distillation_flowmap/output_smoke_cosmos_dual_teacher \
  --output-dir distillation_flowmap/logs/cosmos_sweep_smoke \
  --num-batches 1 \
  --batch-size 1 \
  --dry-run
```

Expected: `distillation_flowmap/logs/cosmos_sweep_smoke/summary.csv` exists.

- [ ] **Step 4: Run real offline sweep on tiny settings**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python distillation_flowmap/sweep_cosmos_checkpoints.py \
  --run-dir distillation_flowmap/output_libero_cosmos_policy_stage1 \
  --run-dir distillation_flowmap/output_smoke_cosmos_dual_teacher \
  --output-dir distillation_flowmap/logs/cosmos_sweep_tiny \
  --num-batches 1 \
  --batch-size 1
```

Expected: JSON metrics and `summary.csv` are written.

- [ ] **Step 5: Commit only if smoke fixes were needed**

If a bugfix was required, commit the exact touched files:

```bash
git add <fixed-files>
git commit -m "fix: stabilize cosmos dual teacher smoke"
```

## Task 9: Longer Experiment Gate

**Files:**
- No code unless the short comparison exposes a defect

- [ ] **Step 1: Start 500-1000 step dual-teacher Stage 1**

Use the available GPUs but keep the first run shorter than 5000 steps:

```bash
tmux new -d -s cosmos_dual_stage1_1000 \
  'cd /root/nas/junjie/jj/Any_WAM && \
   CONFIG_FILE=distillation_flowmap.config_libero_cosmos_policy_stage1_dual_teacher \
   MAX_TRAIN_STEPS=1000 SAVE_INTERVAL=250 ENABLE_WANDB=0 \
   /root/nas/junjie/conda_envs/any_wam/bin/python -m torch.distributed.run --nproc_per_node=8 distillation_flowmap/train.py \
     --output-dir distillation_flowmap/output_libero_cosmos_policy_stage1_dual_teacher_1000 \
   2>&1 | tee distillation_flowmap/logs/cosmos_dual_stage1_1000_$(date +%Y%m%d_%H%M%S).log'
```

- [ ] **Step 2: Sweep every saved checkpoint**

Run:

```bash
/root/nas/junjie/conda_envs/any_wam/bin/python distillation_flowmap/sweep_cosmos_checkpoints.py \
  --run-dir distillation_flowmap/output_libero_cosmos_policy_stage1_dual_teacher_1000 \
  --output-dir distillation_flowmap/logs/cosmos_dual_stage1_1000_sweep \
  --num-batches 8 \
  --batch-size 4
```

- [ ] **Step 3: Decide whether to launch 5000-step Stage 1**

Proceed only if the 1000-step run is at least not worse than action-only Cosmos on fixed offline action metrics and video losses are stable.

- [ ] **Step 4: Record experiment result**

Update the experiment notes with:

- command
- git commit
- checkpoint path
- summary metrics
- generated video paths
- whether the variant should continue

Commit notes only:

```bash
git add docs/superpowers/specs/2026-07-05-cosmos-video-flowmap-design.md docs/superpowers/plans/2026-07-05-cosmos-video-flowmap.md
git commit -m "docs: record cosmos dual teacher experiment result"
```

## Self-Review

- Spec coverage: dual-teacher role split, future-image auxiliary, checkpoint sweep, smoke tests, and longer-run gate are each mapped to tasks.
- Placeholder scan: no unresolved placeholder markers; the only conditional stop is explicit in Task 7 to prevent duplicated VAE encode logic.
- Type consistency: resolver names are `TeacherRoles` and `resolve_teacher_roles`; accessor names are `_action_teacher_model` and `_video_teacher_model`; config fields are consistent across tasks.
