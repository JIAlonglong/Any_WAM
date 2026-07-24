import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FLOWMAP_DIR = os.path.join(REPO_ROOT, "distillation_flowmap")
WANVA_DIR = os.path.join(REPO_ROOT, "wan_va")
for path in (FLOWMAP_DIR, WANVA_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(autouse=True)
def _explicit_cosmos_paths(monkeypatch):
    monkeypatch.setenv("STUDENT_BASE_MODEL_PATH", "/explicit/cosmos-base")
    monkeypatch.setenv("RESUME_FROM_PATH", "/explicit/cosmos-stage1")


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
    assert module.cfg.opd_danceopd_rollout_steps == 4
    assert module.cfg.opd_danceopd_endpoint_weight > 0
    assert module.cfg.opd_danceopd_velocity_weight == 1.0
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
    assert module.cfg.opd_aux_interval == 8
    assert module.cfg.opd_aux_phase == 2


def test_progressive_raw_auxiliary_schedule_ignores_conflicting_environment(
    monkeypatch,
):
    monkeypatch.setenv("COSMOS_PROGRESSIVE_STAGE", "s4")
    monkeypatch.setenv("OPD_AUX_INTERVAL", "4")
    monkeypatch.setenv("OPD_AUX_PHASE", "0")
    module = importlib.reload(importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    ))

    assert module.cfg.opd_aux_interval == 8
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
    assert module.cfg.opd_danceopd_rollout_steps == 1
    assert module.cfg.opd_danceopd_velocity_weight == 0.0


def test_progressive_k2_uses_a_two_step_danceopd_query_grid(monkeypatch):
    monkeypatch.setenv("COSMOS_PROGRESSIVE_STAGE", "s2")
    monkeypatch.delenv("OPD_DANCEOPD_ROLLOUT_STEPS", raising=False)
    monkeypatch.delenv("OPD_DANCEOPD_VELOCITY_WEIGHT", raising=False)
    module = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    )
    module = importlib.reload(module)

    assert module.cfg.opd_rollout_step_pairs == [[4, 2]]
    assert module.cfg.opd_danceopd_rollout_steps == 2
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
    assert module.cfg.opd_danceopd_rollout_steps == 2
    assert module.cfg.opd_danceopd_velocity_weight == 1.0


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
    assert "return_action=False" in cosmos_dance_block
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
