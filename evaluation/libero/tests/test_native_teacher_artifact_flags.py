"""Focused contracts for opt-out evaluation artifacts.

These tests parse the entry points rather than importing their CUDA/LIBERO
dependencies, so they remain runnable in a lightweight test environment.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CLIENT = ROOT / "evaluation" / "libero" / "client.py"
SERVER = ROOT / "wan_va" / "wan_va_native_teacher_server.py"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _is_boolean_optional_argument(call: ast.Call, flag: str) -> bool:
    if not (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "add_argument"
        and any(isinstance(arg, ast.Constant) and arg.value == flag for arg in call.args)
    ):
        return False
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    action = keywords.get("action")
    default = keywords.get("default")
    return (
        isinstance(action, ast.Attribute)
        and isinstance(action.value, ast.Name)
        and action.value.id == "argparse"
        and action.attr == "BooleanOptionalAction"
        and isinstance(default, ast.Constant)
        and default.value is True
    )


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _is_save_video_enabled_test(test: ast.expr) -> bool:
    return isinstance(test, ast.Name) and test.id == "save_video_enabled"


def _is_debug_tensor_test(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Attribute)
        and isinstance(test.value, ast.Name)
        and test.value.id == "self"
        and test.attr == "save_debug_tensors"
    )


def _calls_named(nodes: list[ast.stmt], name: str) -> list[ast.Call]:
    return [
        node
        for statement in nodes
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def test_client_no_save_video_keeps_cache_key_frames_without_writing_mp4():
    tree = _parse(CLIENT)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(_is_boolean_optional_argument(call, "--save-video") for call in calls)

    run_one = _function(tree, "run_one")
    assert "save_video_enabled" in [arg.arg for arg in run_one.args.args]

    guarded_blocks = [
        node
        for node in ast.walk(run_one)
        if isinstance(node, ast.If) and _is_save_video_enabled_test(node.test)
    ]
    assert any(
        any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "full_obs_list"
            and call.func.attr == "append"
            for statement in block.body
            for call in ast.walk(statement)
        )
        for block in guarded_blocks
    )

    # Cache key frames must still be collected for the next server request.
    assert any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "key_frame_list"
        and call.func.attr == "append"
        for call in ast.walk(run_one)
        if isinstance(call, ast.Call)
    )
    assert any(_calls_named(block.body, "save_video") for block in guarded_blocks)


def test_native_server_no_save_debug_tensors_guards_only_debug_writes():
    tree = _parse(SERVER)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(
        _is_boolean_optional_argument(call, "--save-debug-tensors")
        for call in calls
    )

    guarded_save_calls = [
        call
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and _is_debug_tensor_test(node.test)
        for call in _calls_named(node.body, "save_async")
    ]
    assert len(guarded_save_calls) == 3

    latency_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "append_sampler_latency_record"
    ]
    assert len(latency_calls) == 1
