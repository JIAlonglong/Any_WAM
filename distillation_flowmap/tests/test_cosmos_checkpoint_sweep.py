from pathlib import Path

from distillation_flowmap.sweep_cosmos_checkpoints import (
    discover_checkpoints,
    parse_step,
    resolve_checkpoint_transformer,
)


def test_parse_step_from_checkpoint_dir():
    assert parse_step(Path("/tmp/run/checkpoints/step_5000")) == 5000
    assert parse_step(Path("/tmp/run/checkpoints/latest")) is None


def test_discover_checkpoints_sorted_by_step(tmp_path):
    for name in ["step_1000", "step_50", "step_5000"]:
        (tmp_path / "checkpoints" / name).mkdir(parents=True)

    paths = discover_checkpoints(tmp_path)

    assert [p.name for p in paths] == ["step_50", "step_1000", "step_5000"]


def test_resolve_checkpoint_transformer_prefers_online_student(tmp_path):
    step_dir = tmp_path / "checkpoints" / "step_1000"
    online = step_dir / "online_student" / "transformer"
    target = step_dir / "target_student" / "transformer"
    online.mkdir(parents=True)
    target.mkdir(parents=True)

    assert resolve_checkpoint_transformer(step_dir) == online
    assert resolve_checkpoint_transformer(step_dir, kind="target_student") == target


def test_resolve_checkpoint_transformer_supports_plain_transformer(tmp_path):
    step_dir = tmp_path / "checkpoints" / "step_1000"
    transformer = step_dir / "transformer"
    transformer.mkdir(parents=True)

    assert resolve_checkpoint_transformer(step_dir) == transformer
