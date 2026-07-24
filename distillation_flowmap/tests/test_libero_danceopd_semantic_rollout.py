import ast
from pathlib import Path


FLOWMAP_STEP = Path(__file__).resolve().parents[1] / "flowmap_step.py"
FLOWMAP_TRAINER = Path(__file__).resolve().parents[1] / "flowmap_trainer.py"


def _danceopd_function():
    tree = ast.parse(FLOWMAP_STEP.read_text(encoding="utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_danceopd_aux_transition_step"
    )


def test_danceopd_queries_only_post_update_semantic_states():
    danceopd = _danceopd_function()
    called_names = {
        node.func.id
        for node in ast.walk(danceopd)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "sample_semantic_query_indices" in called_names
    assert "sample_low_noise_query_indices" not in called_names


def test_danceopd_appends_terminal_state_after_rollout_loop():
    source = FLOWMAP_STEP.read_text(encoding="utf-8")
    block = source.split("def _danceopd_aux_transition_step(", 1)[1].split(
        "def _opd_aux_transition_step(", 1
    )[0]
    loop_end_marker = "current_video, current_action = self._joint_euler_update("
    terminal_append = "video_states.append(current_video.detach().clone())"

    first_append = block.index(terminal_append)
    update_position = block.index(loop_end_marker)
    final_append = block.rindex(terminal_append)
    assert "query_indices = sample_semantic_query_indices(" in block
    query_position = block.index("query_indices = sample_semantic_query_indices(")

    assert first_append < update_position < final_append < query_position
    assert "video_timesteps.append(video_path[-1].detach().clone())" in block
    assert "action_timesteps.append(action_path[-1].detach().clone())" in block


def test_danceopd_releases_full_rollout_before_trainable_query():
    source = FLOWMAP_STEP.read_text(encoding="utf-8")
    block = source.split("def _danceopd_aux_transition_step(", 1)[1].split(
        "def _opd_aux_transition_step(", 1
    )[0]

    release_marker = (
        "del video_states, action_states, video_timesteps, action_timesteps"
    )
    assert release_marker in block
    release_position = block.index(release_marker)
    query_position = block.index("student_query_input = self._build_joint_input(")

    assert release_position < query_position


def test_danceopd_conditions_action_on_detached_generated_video():
    source = FLOWMAP_STEP.read_text(encoding="utf-8")
    block = source.split("def _danceopd_aux_transition_step(", 1)[1].split(
        "def _opd_aux_transition_step(", 1
    )[0]

    assert "condition_video=current_video.detach()" in block
    assert "condition_video=query_video.detach()" in block
    assert "condition_action=action_clean" in block
    assert "masked_action_teacher_forcing_loss(" in block
    assert "self.train_scheduler_action.training_target(" in block
    assert "'video_action_bridge_loss': bridge_loss.detach()" in block


def test_video_action_bridge_does_not_reenable_action_opd():
    source = FLOWMAP_STEP.read_text(encoding="utf-8")
    block = source.split("def _danceopd_aux_transition_step(", 1)[1].split(
        "def _opd_aux_transition_step(", 1
    )[0]

    assert "'opd_action_transition_loss': zero" in block
    assert "'opd_action_local_fm_loss': zero" in block


def test_video_action_bridge_metrics_are_forwarded_to_training_logs():
    source = FLOWMAP_TRAINER.read_text(encoding="utf-8")

    assert 'opd_aux_result.get("video_action_bridge_loss"' in source
    assert 'log_dict["loss/video_action_bridge"]' in source
    assert 'log_dict["loss_weighted/video_action_bridge"]' in source
    assert 'log_dict["video_action_bridge/active"]' in source
    assert 'log_dict["video_action_bridge/probability"]' in source
