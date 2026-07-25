import ast
from pathlib import Path
import torch
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


def test_trainer_schedules_aligned_video_action_and_anyflow_disjointly():
    from distillation_flowmap.cosmos_progressive_opd import (
        select_progressive_training_objective,
    )

    source = (
        Path(__file__).resolve().parents[1] / "flowmap_trainer.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_select_progressive_training_objective"
    )
    namespace = {
        "select_progressive_training_objective":
            select_progressive_training_objective,
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[])
            ),
            "<trainer-selector>",
            "exec",
        ),
        namespace,
    )
    config = SimpleNamespace(
        opd_aux_interval=4,
        opd_aux_warmup_steps=8,
        opd_aux_phase=2,
    )

    def scheduled_kind(step, *, action_enabled=True):
        return namespace["_select_progressive_training_objective"](
            config,
            step=step,
            deployment_enabled=True,
            raw_auxiliary_enabled=action_enabled,
        )

    objectives = [scheduled_kind(step) for step in range(1, 17)]
    assert objectives[3] == "aligned_video_opd"
    assert objectives[7] == "aligned_video_opd"
    assert objectives[0] == "main_anyflow"
    assert objectives[9] == "action_opd"
    assert objectives[13] == "action_opd"
    assert set(objectives) == {
        "aligned_video_opd",
        "action_opd",
        "main_anyflow",
    }
    assert scheduled_kind(10, action_enabled=False) == "main_anyflow"


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
        opd_aux_interval=4,
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
        "raw_auxiliary_interval": 4,
        "raw_auxiliary_phase": 2,
    }]


def test_trainer_selector_supports_legacy_configs_without_deployment_fields():
    from distillation_flowmap.cosmos_progressive_opd import (
        select_progressive_training_objective,
    )

    source = (
        Path(__file__).resolve().parents[1] / "flowmap_trainer.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_select_progressive_training_objective"
    )
    namespace = {
        "select_progressive_training_objective":
            select_progressive_training_objective,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            "<trainer-selector>",
            "exec",
        ),
        namespace,
    )
    legacy_config = SimpleNamespace()
    selector = namespace["_select_progressive_training_objective"]

    assert selector(
        legacy_config,
        step=11,
        deployment_enabled=False,
        raw_auxiliary_enabled=False,
    ) == "main_anyflow"
    assert selector(
        legacy_config,
        step=10,
        deployment_enabled=False,
        raw_auxiliary_enabled=True,
    ) == "action_opd"


def test_real_train_loop_calls_shared_schedule_adapter_once_before_branches():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_trainer.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    trainer = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FlowMapDistiller"
    )
    train = next(
        node for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    adapter_calls = [
        node
        for node in ast.walk(train)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_select_progressive_training_objective"
    ]
    objective_branches = [
        node
        for node in ast.walk(train)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "scheduled_kind"
    ]

    assert len(adapter_calls) == 1
    assert len(objective_branches) >= 2
    assert adapter_calls[0].lineno < min(
        branch.lineno for branch in objective_branches
    )


def test_aligned_video_objective_is_the_only_scheduled_video_opd_route():
    root = Path(__file__).resolve().parents[1]
    step_source = (root / "flowmap_step.py").read_text(encoding="utf-8")
    trainer_source = (root / "flowmap_trainer.py").read_text(encoding="utf-8")
    deployment_method_source = step_source.split(
        "def _cosmos_deployment_joint_rollout_step("
    )[1].split("def _cosmos_latent_full_opd_aux_transition_step(")[0]

    assert "constrain_cosmos_teacher_timestep_pair" not in deployment_method_source
    assert "predict_raw_latent_velocity" not in deployment_method_source
    assert "_cosmos_aligned_video_opd_step" in trainer_source
    assert "_cosmos_deployment_joint_rollout_step" not in trainer_source


def test_aligned_video_branch_never_merges_the_action_opd_objective():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_trainer.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    trainer = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FlowMapDistiller"
    )
    train = next(
        node for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    aligned_branch = next(
        node for node in ast.walk(train)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "scheduled_kind"
        and any(
            isinstance(comparator, ast.Constant)
            and comparator.value == "aligned_video_opd"
            for comparator in node.test.comparators
        )
    )
    calls = {
        node.func.attr
        for node in ast.walk(
            ast.Module(body=aligned_branch.body, type_ignores=[])
        )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert calls & {"_cosmos_aligned_video_opd_step"}
    assert not calls & {
        "_cosmos_deployment_joint_rollout_step",
        "_cosmos_latent_full_opd_aux_transition_step",
        "_opd_aux_transition_step",
    }


def test_aligned_video_metric_reduction_order_is_stable_and_complete():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_trainer.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_ALIGNED_VIDEO_OPD_METRIC_KEYS"
            for target in node.targets
        )
    )
    assert ast.literal_eval(assignment.value) == (
        "opd_endpoint_loss",
        "opd_same_state_velocity_loss",
        "opd_endpoint_contrib",
        "opd_same_state_velocity_contrib",
        "opd_endpoint_ratio",
        "opd_same_state_velocity_ratio",
        "opd_query_sigma",
        "opd_query_index",
        "opd_student_steps",
        "opd_teacher_steps",
        "opd_valid_video_frames",
        "opd_same_prior_verified",
        "opd_canonical_state_verified",
    )


def test_aligned_video_metric_ratios_use_a_finite_clamped_denominator():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_trainer.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_aligned_video_opd_metrics"
    )
    namespace = {
        "torch": torch,
        "_ALIGNED_VIDEO_OPD_METRIC_KEYS": (
            "opd_endpoint_loss",
            "opd_same_state_velocity_loss",
            "opd_endpoint_contrib",
            "opd_same_state_velocity_contrib",
            "opd_endpoint_ratio",
            "opd_same_state_velocity_ratio",
            "opd_query_sigma",
            "opd_query_index",
            "opd_student_steps",
            "opd_teacher_steps",
            "opd_valid_video_frames",
            "opd_same_prior_verified",
            "opd_canonical_state_verified",
        ),
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[])
            ),
            "<aligned-video-opd-metrics>",
            "exec",
        ),
        namespace,
    )
    zero = torch.tensor(0.0)
    metrics = namespace["_aligned_video_opd_metrics"](
        {
            "opd_endpoint_loss": torch.tensor(2.0),
            "opd_same_state_velocity_loss": torch.tensor(3.0),
            "opd_endpoint_contrib": torch.tensor(1.0),
            "opd_same_state_velocity_contrib": torch.tensor(3.0),
        },
        zero,
    )
    assert metrics["opd_endpoint_ratio"].item() == 0.25
    assert metrics["opd_same_state_velocity_ratio"].item() == 0.75
    zero_metrics = namespace["_aligned_video_opd_metrics"]({}, zero)
    assert torch.isfinite(zero_metrics["opd_endpoint_ratio"])
    assert torch.isfinite(zero_metrics["opd_same_state_velocity_ratio"])


def test_aligned_video_metrics_map_to_complete_public_log_schema():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_trainer.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_aligned_video_opd_log_values"
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[])
            ),
            "<aligned-video-opd-log-values>",
            "exec",
        ),
        namespace,
    )
    ordered_values = {
        key: float(index)
        for index, key in enumerate(
            (
                "opd_endpoint_loss",
                "opd_same_state_velocity_loss",
                "opd_endpoint_contrib",
                "opd_same_state_velocity_contrib",
                "opd_endpoint_ratio",
                "opd_same_state_velocity_ratio",
                "opd_query_sigma",
                "opd_query_index",
                "opd_student_steps",
                "opd_teacher_steps",
                "opd_valid_video_frames",
                "opd_same_prior_verified",
                "opd_canonical_state_verified",
            ),
            start=1,
        )
    }
    assert namespace["_aligned_video_opd_log_values"](ordered_values) == {
        "loss/opd_endpoint": 1.0,
        "loss/opd_same_state_velocity": 2.0,
        "loss_weighted/opd_endpoint": 3.0,
        "loss_weighted/opd_same_state_velocity": 4.0,
        "loss_ratio/opd_endpoint": 5.0,
        "loss_ratio/opd_same_state_velocity": 6.0,
        "opd/query_sigma": 7.0,
        "opd/query_index": 8.0,
        "opd/student_steps": 9.0,
        "opd/teacher_steps": 10.0,
        "opd/valid_video_frames": 11.0,
        "opd/same_prior_verified": 12.0,
        "opd/canonical_state_verified": 13.0,
    }
