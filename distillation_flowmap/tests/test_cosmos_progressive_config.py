import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from distillation_flowmap.cosmos_stage2_lineage import validate_stage1_parent
from distillation_flowmap.cosmos_libero_variants import resolve_variant


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FLOWMAP_DIR = os.path.join(REPO_ROOT, "distillation_flowmap")
WANVA_DIR = os.path.join(REPO_ROOT, "wan_va")
for path in (FLOWMAP_DIR, WANVA_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(autouse=True)
def _explicit_cosmos_paths(monkeypatch, tmp_path):
    stage1 = tmp_path / "stage1"
    payload = {
        "contract_version": 2,
        "training_contract_stage": "raw_stage1",
        "action_packing_schema": "downsample_survivor_v2",
        "action_downsample_factor": 4,
        "action_chunk_shape": [4, 4],
        "checkpoint_step": 5000,
        "teacher_backend": "cosmos_policy",
    }
    for variant in ("online_student", "target_student"):
        transformer = stage1 / variant / "transformer"
        transformer.mkdir(parents=True)
        (transformer / "config.json").write_text(json.dumps(payload))
        (transformer / "diffusion_pytorch_model.safetensors").write_bytes(
            b"weights"
        )
    parent = validate_stage1_parent(stage1)
    lineage = json.dumps(
        {
            "parent_stage1_contract_identity": parent.contract_identity,
            "parent_stage1_path": parent.canonical_path,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    values = {
        "STUDENT_BASE_MODEL_PATH": str(stage1 / "target_student"),
        "RESUME_FROM_PATH": str(stage1),
        "PARENT_STAGE1_PATH": parent.canonical_path,
        "PARENT_STAGE1_CONTRACT_IDENTITY": parent.contract_identity,
        "STAGE2_LINEAGE_JSON": lineage,
        "OUTPUT_DIR": str(tmp_path / "stage2-arm"),
        "RESUME_ONLINE_FROM_TARGET": "1",
        "RESET_RESUME_STEP": "1",
        "RESUME_OPTIMIZER_STATE": "0",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return {"stage1": stage1, "parent": parent, "env": values}


def _import_progressive(env):
    process_env = os.environ.copy()
    process_env.update(env)
    process_env["PYTHONPATH"] = REPO_ROOT
    return subprocess.run(
        [
            "/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python",
            "-c",
            "import distillation_flowmap.config_libero_cosmos_policy_stage2_progressive",
        ],
        cwd=REPO_ROOT,
        env=process_env,
        text=True,
        capture_output=True,
        check=False,
    )


def _read_progressive_fields(env, fields):
    process_env = os.environ.copy()
    process_env.update(env)
    process_env["PYTHONPATH"] = REPO_ROOT
    expression = repr(tuple(fields))
    return subprocess.run(
        [
            "/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python",
            "-c",
            (
                "import json\n"
                "from distillation_flowmap.config_libero_cosmos_policy_stage2_progressive "
                "import cfg\n"
                f"print('MECHANISM_JSON=' + json.dumps({{name: getattr(cfg, name) "
                f"for name in {expression}}}, sort_keys=True))"
            ),
        ],
        cwd=REPO_ROOT,
        env=process_env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_progressive_config_exposes_cosmos_mechanism_defaults(
    _explicit_cosmos_paths,
):
    fields = (
        "mechanism_diagnostics",
        "mechanism_diagnostic_interval",
        "mechanism_diagnostic_seed",
        "mechanism_diagnostic_r",
        "mechanism_diagnostic_s",
        "mechanism_diagnostic_teacher_steps",
        "mechanism_cosmos_t_min",
        "mechanism_cosmos_t_max",
        "mechanism_teacher_joint_available",
    )
    result = _read_progressive_fields(_explicit_cosmos_paths["env"], fields)
    assert result.returncode == 0, result.stderr
    payload_line = next(
        line for line in result.stdout.splitlines() if line.startswith("MECHANISM_JSON=")
    )
    payload = json.loads(payload_line.removeprefix("MECHANISM_JSON="))
    assert payload == {
        "mechanism_diagnostics": True,
        "mechanism_diagnostic_interval": 100,
        "mechanism_diagnostic_seed": 42,
        "mechanism_diagnostic_r": 500.0,
        "mechanism_diagnostic_s": 250.0,
        "mechanism_diagnostic_teacher_steps": 8,
        "mechanism_cosmos_t_min": pytest.approx(4.0 / 5.0),
        "mechanism_cosmos_t_max": pytest.approx(80.0 / 81.0),
        "mechanism_teacher_joint_available": False,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MECHANISM_DIAGNOSTIC_INTERVAL", "0"),
        ("MECHANISM_DIAGNOSTIC_TEACHER_STEPS", "0"),
        ("MECHANISM_DIAGNOSTIC_R", "1001"),
        ("MECHANISM_DIAGNOSTIC_S", "-1"),
        ("MECHANISM_COSMOS_T_MIN", "0.5"),
        ("MECHANISM_COSMOS_T_MAX", "1.0"),
    ],
)
def test_progressive_config_rejects_invalid_mechanism_controls(
    _explicit_cosmos_paths, name, value
):
    env = dict(_explicit_cosmos_paths["env"])
    env[name] = value
    result = _import_progressive(env)
    assert result.returncode != 0
    assert name in result.stderr


@pytest.mark.parametrize(
    "missing", ["STUDENT_BASE_MODEL_PATH", "RESUME_FROM_PATH"]
)
def test_progressive_config_requires_explicit_cosmos_paths(missing):
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": REPO_ROOT,
            "STUDENT_BASE_MODEL_PATH": "/explicit/cosmos-base",
            "RESUME_FROM_PATH": "/explicit/cosmos-stage1",
        }
    )
    env.pop(missing)

    result = subprocess.run(
        [
            "/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python",
            "-c",
            "import distillation_flowmap.config_libero_cosmos_policy_stage2_progressive",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert missing in result.stderr


def test_progressive_config_validates_lineage_outside_launcher(
    _explicit_cosmos_paths,
):
    result = _import_progressive(_explicit_cosmos_paths["env"])

    assert result.returncode == 0, result.stderr


def _write_stage2_resume(
    root, *, step, parent, missing=None, parent_identity=None
):
    checkpoint = root / "checkpoints" / f"step_{step}"
    payload = {
        "contract_version": 2,
        "training_contract_stage": "progressive_stage2",
        "action_packing_schema": "downsample_survivor_v2",
        "action_downsample_factor": 4,
        "action_chunk_shape": [4, 4],
        "checkpoint_step": step,
        "deployment_timestep_start": 1000,
        "deployment_timestep_end": 0,
        "joint_student_steps": [1, 2, 4],
        "deployment_joint_rollout_interval": 4,
        "deployment_action_weight": 1.0,
        "raw_teacher_window_is_auxiliary": True,
        "parent_stage1_path": parent.canonical_path,
        "parent_stage1_contract_identity": (
            parent_identity or parent.contract_identity
        ),
    }
    for variant in ("online_student", "target_student"):
        if missing == variant:
            continue
        transformer = checkpoint / variant / "transformer"
        transformer.mkdir(parents=True)
        (transformer / "config.json").write_text(json.dumps(payload))
        (transformer / "diffusion_pytorch_model.safetensors").write_bytes(
            b"weights"
        )
    if missing != "optimizer.pt":
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    if missing != "lr_scheduler.pt":
        (checkpoint / "lr_scheduler.pt").write_bytes(b"scheduler")
    return checkpoint


def _resume_env(paths, checkpoint, output_dir):
    env = dict(paths["env"])
    env.update(
        {
            "OUTPUT_DIR": str(output_dir),
            "RESUME_FROM_PATH": str(checkpoint),
            "RESUME_ONLINE_FROM_TARGET": "0",
            "RESET_RESUME_STEP": "0",
            "RESUME_OPTIMIZER_STATE": "1",
        }
    )
    return env


def test_progressive_config_rejects_foreign_fresh_resume(
    _explicit_cosmos_paths, tmp_path
):
    foreign = tmp_path / "foreign-stage1"
    foreign.mkdir()
    env = dict(_explicit_cosmos_paths["env"])
    env["RESUME_FROM_PATH"] = str(foreign)

    result = _import_progressive(env)

    assert result.returncode != 0
    assert "fresh RESUME_FROM_PATH" in result.stderr


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("RESUME_ONLINE_FROM_TARGET", "0"),
        ("RESET_RESUME_STEP", "0"),
        ("RESUME_OPTIMIZER_STATE", "1"),
    ],
)
def test_progressive_config_rejects_inconsistent_fresh_flags(
    _explicit_cosmos_paths, flag, value
):
    env = dict(_explicit_cosmos_paths["env"])
    env[flag] = value

    result = _import_progressive(env)

    assert result.returncode != 0
    assert flag in result.stderr


def test_progressive_config_accepts_valid_stage2_resume(
    _explicit_cosmos_paths, tmp_path
):
    arm = tmp_path / "resume-arm"
    checkpoint = _write_stage2_resume(
        arm,
        step=1000,
        parent=_explicit_cosmos_paths["parent"],
    )
    result = _import_progressive(
        _resume_env(_explicit_cosmos_paths, checkpoint, arm)
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "missing", ["target_student", "optimizer.pt", "lr_scheduler.pt"]
)
def test_progressive_config_rejects_incomplete_stage2_resume(
    _explicit_cosmos_paths, tmp_path, missing
):
    arm = tmp_path / f"resume-arm-{missing}"
    checkpoint = _write_stage2_resume(
        arm,
        step=1000,
        parent=_explicit_cosmos_paths["parent"],
        missing=missing,
    )

    result = _import_progressive(
        _resume_env(_explicit_cosmos_paths, checkpoint, arm)
    )

    assert result.returncode != 0
    assert missing in result.stderr


def test_progressive_config_rejects_resume_from_foreign_arm(
    _explicit_cosmos_paths, tmp_path
):
    own_arm = tmp_path / "own-arm"
    foreign_arm = tmp_path / "foreign-arm"
    checkpoint = _write_stage2_resume(
        foreign_arm,
        step=1000,
        parent=_explicit_cosmos_paths["parent"],
    )

    result = _import_progressive(
        _resume_env(_explicit_cosmos_paths, checkpoint, own_arm)
    )

    assert result.returncode != 0
    assert "checkpoints" in result.stderr


def test_progressive_config_rejects_resume_parent_identity_mismatch(
    _explicit_cosmos_paths, tmp_path
):
    arm = tmp_path / "resume-arm"
    checkpoint = _write_stage2_resume(
        arm,
        step=1000,
        parent=_explicit_cosmos_paths["parent"],
        parent_identity="0" * 64,
    )

    result = _import_progressive(
        _resume_env(_explicit_cosmos_paths, checkpoint, arm)
    )

    assert result.returncode != 0
    assert "parent_stage1_contract_identity" in result.stderr


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("backend", "cosmos_policy"),
        ("student_base", "STUDENT_BASE_MODEL_PATH"),
        ("identity", "PARENT_STAGE1_CONTRACT_IDENTITY"),
        ("lineage_json", "STAGE2_LINEAGE_JSON"),
    ],
)
def test_progressive_config_rejects_invalid_lineage_outside_launcher(
    _explicit_cosmos_paths, mutation, message
):
    env = dict(_explicit_cosmos_paths["env"])
    stage1 = _explicit_cosmos_paths["stage1"]
    if mutation == "backend":
        for variant in ("online_student", "target_student"):
            config = stage1 / variant / "transformer" / "config.json"
            payload = json.loads(config.read_text())
            payload["teacher_backend"] = "wanva"
            config.write_text(json.dumps(payload))
    elif mutation == "student_base":
        env["STUDENT_BASE_MODEL_PATH"] = str(stage1 / "online_student")
    elif mutation == "identity":
        env["PARENT_STAGE1_CONTRACT_IDENTITY"] = "0" * 64
    else:
        env["STAGE2_LINEAGE_JSON"] = json.dumps(
            {"parent_stage1_path": "/wrong"}
        )

    result = _import_progressive(env)

    assert result.returncode != 0
    assert message in result.stderr


def test_progressive_sources_have_no_forbidden_legacy_path_fallbacks():
    config = (
        Path(__file__).resolve().parents[1]
        / "config_libero_cosmos_policy_stage2_progressive.py"
    ).read_text(encoding="utf-8")
    launcher = (
        Path(__file__).resolve().parents[1]
        / "run_cosmos_progressive_stage2_8gpu.sh"
    ).read_text(encoding="utf-8")

    for source in (config, launcher):
        assert "/root/nas" not in source
        assert "lingbot-va/checkpoints" not in source.lower()


def test_progressive_s4_defaults_to_full_cosmos_opd(monkeypatch):
    for name in (
        "COSMOS_PROGRESSIVE_STAGE",
        "OPD_ENDPOINT_FOCUS_PROB",
        "OPD_DANCEOPD_VELOCITY_WEIGHT",
        "OPD_DANCEOPD_ENDPOINT_WEIGHT",
        "OPD_DANCEOPD_ROLLOUT_STEPS",
        "OPD_AUX_INTERVAL",
        "OPD_AUX_PHASE",
    ):
        monkeypatch.delenv(name, raising=False)

    module = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    )
    module = importlib.reload(module)

    assert module.cfg.opd_teacher_target_mode == "cosmos_latent_full"
    assert module.cfg.opd_rollout_step_pairs == [[8, 4]]
    assert module.cfg.diffusion_ratio == 0.5
    assert module.cfg.consistency_ratio == 0.25
    assert module.cfg.flowmap_ratio == 0.25
    assert module.cfg.opd_endpoint_focus_prob > 0
    assert module.cfg.opd_danceopd_rollout_steps == (2, 4)
    assert module.cfg.opd_danceopd_anchor_teacher_steps == 8
    assert module.cfg.opd_danceopd_endpoint_weight == 1.0
    assert module.cfg.opd_danceopd_velocity_weight == 1.0
    assert module.cfg.aligned_video_opd_enabled is True
    assert module.cfg.aligned_video_opd_interval == 4
    assert module.cfg.action_downsample_factor == 4
    assert module.cfg.video_action_bridge == 0
    assert module.cfg.opd_serial_student_cfg is True
    assert module.cfg.opd_aux_standalone_step is True
    assert module.cfg.opd_aux_gradient_checkpointing is True
    assert module.cfg.opd_cosmos_spatial_crop_size == 28
    assert module.cfg.cosmos_use_teacher_action_anchor is True
    assert module.cfg.opd_joint_action_rollout is True
    assert module.cfg.opd_danceopd_action_endpoint_weight > 0
    assert module.cfg.deployment_joint_rollout_enabled is True
    assert module.cfg.deployment_joint_rollout_interval == 4
    assert module.cfg.deployment_joint_steps == (1, 2, 4)
    assert module.cfg.deployment_timestep_start == 1000
    assert module.cfg.deployment_timestep_end == 0
    assert module.cfg.deployment_action_weight == 1.0
    assert module.cfg.raw_teacher_window_is_auxiliary is True
    assert module.cfg.opd_aux_interval == 4
    assert module.cfg.opd_aux_phase == 2


def test_progressive_auxiliary_schedule_accepts_a_positive_integer_interval(
    monkeypatch,
):
    monkeypatch.setenv("COSMOS_PROGRESSIVE_STAGE", "s4")
    monkeypatch.setenv("OPD_AUX_INTERVAL", "7")
    module = importlib.reload(importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    ))

    assert module.cfg.opd_aux_interval == 7
    assert module.cfg.opd_aux_phase == 2


def test_progressive_k1_is_endpoint_only(monkeypatch):
    monkeypatch.setenv("COSMOS_PROGRESSIVE_STAGE", "s1")
    monkeypatch.delenv("OPD_DANCEOPD_ROLLOUT_STEPS", raising=False)
    monkeypatch.delenv("OPD_DANCEOPD_VELOCITY_WEIGHT", raising=False)
    module = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    )
    module = importlib.reload(module)

    assert module.cfg.opd_rollout_step_pairs == [[4, 1]]
    assert module.cfg.opd_endpoint_focus_prob > 0
    assert module.cfg.opd_danceopd_rollout_steps == (2, 4)
    assert module.cfg.opd_danceopd_velocity_weight == 1.0


def test_progressive_k2_uses_a_two_step_danceopd_query_grid(monkeypatch):
    monkeypatch.setenv("COSMOS_PROGRESSIVE_STAGE", "s2")
    monkeypatch.delenv("OPD_DANCEOPD_ROLLOUT_STEPS", raising=False)
    monkeypatch.delenv("OPD_DANCEOPD_VELOCITY_WEIGHT", raising=False)
    module = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    )
    module = importlib.reload(module)

    assert module.cfg.opd_rollout_step_pairs == [[4, 2]]
    assert module.cfg.opd_danceopd_rollout_steps == (2, 4)
    assert module.cfg.opd_danceopd_velocity_weight == 1.0


def test_progressive_universal_retains_original_lingbotva_definition(monkeypatch):
    monkeypatch.setenv("COSMOS_PROGRESSIVE_STAGE", "universal")
    monkeypatch.delenv("OPD_DANCEOPD_ROLLOUT_STEPS", raising=False)
    module = importlib.reload(importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    ))
    OPD_ROLLOUT_STEP_PAIRS = tuple(
        tuple(pair) for pair in module.cfg.opd_rollout_step_pairs
    )
    assert OPD_ROLLOUT_STEP_PAIRS == ((8, 1), (8, 2), (8, 4))
    assert module.cfg.opd_danceopd_rollout_step_choices == (2, 4)
    assert module.cfg.opd_danceopd_rollout_steps == (2, 4)
    assert module.cfg.opd_danceopd_velocity_weight == 1.0


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("OPD_DANCEOPD_ROLLOUT_STEPS", "2,3", "OPD_DANCEOPD_ROLLOUT_STEPS"),
        (
            "OPD_DANCEOPD_ANCHOR_TEACHER_STEPS",
            "7",
            "OPD_DANCEOPD_ANCHOR_TEACHER_STEPS",
        ),
        ("OPD_DANCEOPD_ENDPOINT_WEIGHT", "0", "OPD_DANCEOPD_ENDPOINT_WEIGHT"),
        ("OPD_DANCEOPD_VELOCITY_WEIGHT", "-1", "OPD_DANCEOPD_VELOCITY_WEIGHT"),
        ("OPD_AUX_INTERVAL", "0", "OPD_AUX_INTERVAL"),
        (
            "ALIGNED_VIDEO_OPD_INTERVAL",
            "7",
            "ALIGNED_VIDEO_OPD_INTERVAL",
        ),
        ("VIDEO_ACTION_BRIDGE", "1", "VIDEO_ACTION_BRIDGE"),
    ],
)
def test_progressive_config_rejects_invalid_aligned_video_opd_controls(
    _explicit_cosmos_paths, name, value, message
):
    env = dict(_explicit_cosmos_paths["env"])
    env[name] = value
    result = _import_progressive(env)
    assert result.returncode != 0
    assert message in result.stderr


def test_aligned_video_config_rejects_a_grid_without_teacher_band_query(
    monkeypatch,
):
    monkeypatch.setenv("COSMOS_PROGRESSIVE_STAGE", "universal")
    module = importlib.reload(importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    ))

    with pytest.raises(ValueError, match="Teacher-band query"):
        module._validate_aligned_query_grids((2, 4), shift=1.0)


def test_universal_launcher_variant_resolves_aligned_video_interval(tmp_path):
    resolved = resolve_variant(
        "universal-video-action",
        output_root=tmp_path,
        run_tag="aligned",
        steps=1,
        save_interval=1,
        master_port=29672,
    )
    assert resolved["opd_aux_interval"] == 4


def test_main_anyflow_branch_boundaries_preserve_arbitrary_intervals():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_step.py"
    ).read_text(encoding="utf-8")
    sampling_block = source.split("def sample_timestep_mixed(")[1].split(
        "def sample_cosmos_latent_timestep_mixed("
    )[0]

    assert "diffusion_ratio = self.diffusion_ratio / total_ratio" in sampling_block
    assert "consistency_ratio = self.consistency_ratio / total_ratio" in sampling_block
    assert "mode_rand < diffusion_ratio" in sampling_block
    assert "mode_rand < diffusion_ratio + consistency_ratio" in sampling_block
    assert "r = torch.where(is_diffusion, t, r)" in sampling_block
    assert "r = torch.where(is_consistency, torch.zeros_like(r), r)" in sampling_block

    def branch_for(probability):
        if probability < 0.5:
            return "diffusion"
        if probability < 0.75:
            return "endpoint"
        return "arbitrary"

    assert branch_for(0.00) == "diffusion"
    assert branch_for(0.499999) == "diffusion"
    assert branch_for(0.50) == "endpoint"
    assert branch_for(0.749999) == "endpoint"
    assert branch_for(0.75) == "arbitrary"
    t = max(0.8, 0.2)
    arbitrary_r = min(0.8, 0.2)
    assert arbitrary_r > 0
    assert arbitrary_r < t


def test_full_cosmos_endpoint_path_uses_uniform_pair_sampler():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_step.py"
    ).read_text(encoding="utf-8")
    full_cosmos_block = source.split(
        "def _cosmos_latent_full_opd_aux_transition_step("
    )[1].split("def _cosmos_latent_opd_aux_transition_step(")[0]

    assert "sample_uniform_rollout_step_pair(" in full_cosmos_block
    assert "rollout_step_pairs[0]" not in full_cosmos_block


def test_cosmos_danceopd_uses_pre_update_compositional_states_and_skips_s1_velocity():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_step.py"
    ).read_text(encoding="utf-8")
    cosmos_dance_block = source.split(
        "def _cosmos_danceopd_velocity_loss("
    )[1].split("def _cosmos_latent_full_opd_aux_transition_step(")[0]
    cosmos_full_block = source.split(
        "def _cosmos_latent_full_opd_aux_transition_step("
    )[1].split("def _cosmos_latent_opd_aux_transition_step(")[0]

    assert "sample_low_noise_query_indices(" in cosmos_dance_block
    assert "sample_semantic_query_indices(" not in cosmos_dance_block
    assert cosmos_dance_block.count(
        "video_states.append(current_video.detach().clone())"
    ) == 1
    update_position = cosmos_dance_block.index(
        "current_video = current_video + video_velocity"
    )
    capture_position = cosmos_dance_block.index(
        "video_states.append(current_video.detach().clone())"
    )
    query_position = cosmos_dance_block.index(
        "query_indices = sample_low_noise_query_indices("
    )
    assert capture_position < update_position < query_position
    assert "require_action=False" in cosmos_dance_block
    assert "predict_raw_joint_latent_velocity(" in cosmos_dance_block
    assert "masked_video_velocity_mse(" in cosmos_dance_block
    assert "if velocity_weight > 0:" in cosmos_full_block


def test_full_cosmos_anchor_separates_low_noise_student_from_supported_teacher():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_step.py"
    ).read_text(encoding="utf-8")
    full_cosmos_block = source.split(
        "def _cosmos_latent_full_opd_aux_transition_step("
    )[1].split("def _cosmos_latent_opd_aux_transition_step(")[0]

    assert "sample_endpoint_sigmas(" in full_cosmos_block
    assert "teacher_target_r" in full_cosmos_block
    assert "student_x0 = student_x_r - sigma_r" in full_cosmos_block
    assert "teacher_v_r" not in full_cosmos_block


def test_progressive_exposes_chunk_stop_and_training_manifest(monkeypatch):
    monkeypatch.setenv("DATASET_SAMPLE_MANIFEST", "/tmp/cosmos_train.json")
    monkeypatch.setenv("STOP_AFTER_STEP", "250")
    module = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    )
    module = importlib.reload(module)

    assert module.cfg.dataset_sample_manifest == "/tmp/cosmos_train.json"
    assert module.cfg.stop_after_step == 250
