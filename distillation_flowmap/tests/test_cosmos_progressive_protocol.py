import ast
from pathlib import Path
from distillation_flowmap.cosmos_progressive_protocol import (
    aligned_teacher_path_indices,
    build_dataset_manifest,
    build_task_splits,
    cosmos_latent_shape,
)
from types import SimpleNamespace


def test_task_splits_are_disjoint_and_reproducible():
    records = [
        {"index": index, "task": task}
        for task in ("drawer", "stack")
        for index in (range(0, 10) if task == "drawer" else range(10, 20))
    ]

    first = build_task_splits(
        records,
        selection_per_task=2,
        test_per_task=3,
        protocol_seed=17,
    )
    second = build_task_splits(
        records,
        selection_per_task=2,
        test_per_task=3,
        protocol_seed=17,
    )

    assert first == second
    train = {record["index"] for record in first["train"]}
    selection = {record["index"] for record in first["selection"]}
    test = {record["index"] for record in first["test"]}
    assert len(train) == 10
    assert len(selection) == 4
    assert len(test) == 6
    assert not train & selection
    assert not train & test
    assert not selection & test


def test_dataset_manifest_preserves_record_task_labels():
    records = [
        {"index": 4, "task": "drawer", "episode_index": 2},
        {"index": 9, "task": "stack", "episode_index": 7},
    ]

    manifest = build_dataset_manifest(
        records,
        split="selection",
        root_task="libero",
        protocol_seed=123,
    )

    assert manifest["tasks"] == [{"task": "libero", "indices": [4, 9]}]
    assert manifest["records"] == records
    assert manifest["split"] == "selection"


def test_aligned_teacher_path_indices_require_integer_compression_ratio():
    assert aligned_teacher_path_indices(teacher_steps=8, student_steps=4) == [0, 2, 4, 6, 8]

    try:
        aligned_teacher_path_indices(teacher_steps=3, student_steps=2)
    except ValueError as exc:
        assert "integer multiple" in str(exc)
    else:
        raise AssertionError("expected invalid teacher/student ratio to fail")


def test_cosmos_latent_shape_uses_official_prediction_horizon_not_dataset_cache():
    config = SimpleNamespace(
        cosmos_latent_channels=16,
        cosmos_latent_frames=9,
        cosmos_latent_height=28,
        cosmos_latent_width=28,
    )

    assert cosmos_latent_shape(config, batch_size=2) == (2, 16, 9, 28, 28)


def test_trainer_runs_deployment_and_raw_aux_on_disjoint_steps():
    from distillation_flowmap.cosmos_progressive_opd import (
        select_progressive_training_objective,
    )

    def scheduled_kind(step):
        return select_progressive_training_objective(
            step=step,
            deployment_enabled=True,
            deployment_interval=4,
            raw_auxiliary_enabled=True,
            raw_auxiliary_warmup=0,
            raw_auxiliary_interval=8,
            raw_auxiliary_phase=2,
        )

    assert scheduled_kind(step=8) == "deployment"
    assert scheduled_kind(step=10) == "raw_auxiliary"
    assert scheduled_kind(step=12) == "deployment"
    assert scheduled_kind(step=11) == "main"


def test_trainer_call_site_delegates_schedule_to_the_shared_selector():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_trainer.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_select_progressive_training_objective"
    )
    calls = []

    def selector(**kwargs):
        calls.append(kwargs)
        return "sentinel"

    namespace = {"select_progressive_training_objective": selector}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            "<trainer-selector>",
            "exec",
        ),
        namespace,
    )
    config = SimpleNamespace(
        deployment_joint_rollout_interval=4,
        opd_aux_warmup_steps=0,
        opd_aux_interval=8,
        opd_aux_phase=2,
    )

    assert namespace["_select_progressive_training_objective"](
        config,
        step=12,
        deployment_enabled=True,
        raw_auxiliary_enabled=True,
    ) == "sentinel"
    assert calls == [{
        "step": 12,
        "deployment_enabled": True,
        "deployment_interval": 4,
        "raw_auxiliary_enabled": True,
        "raw_auxiliary_warmup": 0,
        "raw_auxiliary_interval": 8,
        "raw_auxiliary_phase": 2,
    }]


def test_deployment_student_path_is_not_clamped_to_the_raw_teacher_window():
    root = Path(__file__).resolve().parents[1]
    step_source = (root / "flowmap_step.py").read_text(encoding="utf-8")
    trainer_source = (root / "flowmap_trainer.py").read_text(encoding="utf-8")
    deployment_method_source = step_source.split(
        "def _cosmos_deployment_joint_rollout_step("
    )[1].split("def _cosmos_latent_full_opd_aux_transition_step(")[0]

    assert "constrain_cosmos_teacher_timestep_pair" not in deployment_method_source
    assert "predict_raw_latent_velocity" not in deployment_method_source
    assert "_cosmos_deployment_joint_rollout_step" in trainer_source
