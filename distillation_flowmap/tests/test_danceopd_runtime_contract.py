import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FLOWMAP_STEP = ROOT / "distillation_flowmap" / "flowmap_step.py"


def test_danceopd_runtime_imports_every_query_helper_it_calls():
    tree = ast.parse(FLOWMAP_STEP.read_text())
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "distillation_flowmap.danceopd_query"
        for alias in node.names
    }

    required_helpers = {
        name
        for name in called_names
        if name in {
            "denoised_endpoint_mse",
            "direct_velocity_mse",
            "masked_velocity_mse",
            "sample_low_noise_query_indices",
            "sample_semantic_query_indices",
            "select_per_sample_trajectory_state",
        }
    }
    assert required_helpers <= imported_names
