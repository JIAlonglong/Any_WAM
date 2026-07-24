"""Exact environment-read contract for the Cosmos progressive config chain."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


CONFIG_CHAIN = (
    "distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py",
    "distillation_flowmap/config_libero_cosmos_policy_stage2_cosmos_latent_cdiff.py",
    "distillation_flowmap/config_libero_cosmos_policy_stage2_all_cosmos_flowmap.py",
    "distillation_flowmap/config_libero_cosmos_policy_stage1_all_cosmos_flowmap.py",
    "distillation_flowmap/config_libero_fullfinetune_stage1_warmup.py",
    "distillation_flowmap/config_libero_optimized.py",
)


CLEARED_LEGACY_ENV = frozenset(
    {
        "COSMOS_PROGRESSIVE_RUN_ID",
        "DATASET_SAMPLE_MANIFEST",
        "DISTILL_MODE",
        "LOCAL_FM_WEIGHT",
        "OPD_ANCHOR_CAP_RATIO",
        "OPD_AUX_USE_NOFSDP_ROLLOUT",
        "OPD_AUX_VARIANT",
        "OPD_ENDPOINT_AUX_WEIGHT",
        "OPD_TEACHER_TARGET_MODE",
        "OPD_TRANSITION_GROUP_WEIGHT",
        "ROLLOUT_STEP_PAIRS",
        "STAGE1_CKPT_NAME",
        "TEACHER_PATH",
        "VIDEO_TRANSITION_PARAM",
        "VIDEO_TRANSITION_WEIGHT",
    }
)

PINNED_OPERATIONAL_ENV = frozenset(
    {
        "CACHE_DATASET_IN_MEMORY",
        "COSMOS_POLICY_VALIDATE_WEIGHTS",
        "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES",
        "COSMOS_TRAIN_STEP_PROFILE",
        "ENABLE_LIGHT_EVAL",
        "ENABLE_ROLLOUT_EVAL",
        "ENABLE_STAGE1_START_EVAL",
        "ENABLE_STAGE1_START_EVAL_BASELINE",
        "ENABLE_TENSORBOARD",
        "ENABLE_WANDB",
        "GRADIENT_CHECKPOINTING",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_OFFLINE",
        "LIGHT_EVAL_INTERVAL",
        "LIGHT_EVAL_NUM_BATCHES",
        "LIGHT_EVAL_SEED",
        "LIGHT_EVAL_START_INDEX",
        "OPD_AUX_EMPTY_CACHE",
        "OPD_AUX_GRADIENT_CHECKPOINTING",
        "OPD_PROFILE",
        "OPD_SERIAL_STUDENT_CFG",
        "SKIP_TEACHER_COMPILE",
        "STOP_AFTER_STEP",
        "TRANSFORMERS_OFFLINE",
        "USE_FSDP1",
        "WANDB_MODE",
    }
)

_ALL_CURRENT_ENV = frozenset(
    """
ACTION_BLOCK_WEIGHT ACTION_LOSS_WEIGHT ATTN_MODE BETA1 BETA2
CACHE_DATASET_IN_MEMORY CFG_MAX CFG_MIN CONSISTENCY_RATIO
COSMOS_LATENT_CDIFF_INTERVAL COSMOS_LATENT_CDIFF_LOSS_WEIGHT
COSMOS_LATENT_CENTER_VELOCITY_MODE COSMOS_LATENT_CHANNELS
COSMOS_LATENT_ENDPOINT_LOSS_WEIGHT COSMOS_LATENT_EPSILON COSMOS_LATENT_FRAMES
COSMOS_LATENT_HEIGHT COSMOS_LATENT_TARGET_MODE COSMOS_LATENT_T_MAX
COSMOS_LATENT_T_MIN COSMOS_LATENT_WIDTH COSMOS_LIBERO_PROVENANCE_JSON
COSMOS_LIBERO_VARIANT_JSON
COSMOS_POLICY_CONFIG_FILE COSMOS_POLICY_CONFIG_NAME
COSMOS_POLICY_EXTRA_PYTHONPATH COSMOS_POLICY_INFERENCE_MODE
COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION COSMOS_POLICY_PATH
COSMOS_POLICY_PRIMARY_IMAGE_KEY COSMOS_POLICY_PYTHON COSMOS_POLICY_SEED
COSMOS_POLICY_USE_RAW_INFERENCE COSMOS_POLICY_VALIDATE_WEIGHTS
COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES COSMOS_POLICY_WRIST_IMAGE_KEY
COSMOS_PREDICT25_LOCAL_MODEL_DIR COSMOS_PREDICT2_REPO
COSMOS_PROGRESSIVE_OUTPUT_ROOT COSMOS_PROGRESSIVE_RUN_ID
COSMOS_PROGRESSIVE_STAGE COSMOS_TRAIN_STEP_PROFILE
COSMOS_USE_TEACHER_ACTION_ANCHOR COSMOS_VIDEO_VAE_MODEL_PATH DATASET_PATH
DATASET_SAMPLE_MANIFEST DIFFUSION_RATIO DISTILL_MODE DROP_TEXT_RATIO EMA_DECAY
EMA_WARMUP_STEPS ENABLE_LIGHT_EVAL ENABLE_ROLLOUT_EVAL
ENABLE_STAGE1_START_EVAL ENABLE_STAGE1_START_EVAL_BASELINE ENABLE_TENSORBOARD
ENABLE_WANDB FLOWMAP_RATIO FUSE_GUIDANCE_SCALE GRADIENT_CHECKPOINTING
HF_DATASETS_OFFLINE HF_HUB_OFFLINE LEARNING_RATE LIGHT_EVAL_INTERVAL
LIGHT_EVAL_NUM_BATCHES LIGHT_EVAL_SEED LIGHT_EVAL_START_INDEX LOCAL_FM_WEIGHT
MASTER_PORT MAX_GRAD_NORM MAX_TRAIN_STEPS MECHANISM_COSMOS_T_MAX
MECHANISM_COSMOS_T_MIN MECHANISM_DIAGNOSTICS MECHANISM_DIAGNOSTIC_INTERVAL
MECHANISM_DIAGNOSTIC_R MECHANISM_DIAGNOSTIC_S MECHANISM_DIAGNOSTIC_SEED
MECHANISM_DIAGNOSTIC_TEACHER_STEPS NUM_DDIM_TIMESTEPS_ACTION
OPD_ACTION_ROLLOUT_GRAD_MODE OPD_ANCHOR_CAP_RATIO OPD_AUX_ACTION
OPD_AUX_EMPTY_CACHE OPD_AUX_GRADIENT_CHECKPOINTING OPD_AUX_INTERVAL
OPD_AUX_PROB OPD_AUX_STANDALONE_STEP OPD_AUX_USE_NOFSDP_ROLLOUT
OPD_AUX_VARIANT OPD_AUX_WARMUP_STEPS OPD_AUX_WEIGHT
OPD_COSMOS_SPATIAL_CROP_SIZE OPD_DANCEOPD_ACTION_ENDPOINT_WEIGHT
OPD_DANCEOPD_ENDPOINT_WEIGHT OPD_DANCEOPD_QUERY_ALPHA
OPD_DANCEOPD_QUERY_BETA OPD_DANCEOPD_ROLLOUT_STEPS
OPD_DANCEOPD_TERMINAL_PRIOR_TOLERANCE
OPD_DANCEOPD_TERMINAL_PRIOR_WARN_FACTOR OPD_DANCEOPD_VELOCITY_WEIGHT
OPD_DANCEOPD_VERIFY_TERMINAL_PRIOR OPD_ENDPOINT_AUX_WEIGHT
OPD_ENDPOINT_FOCUS_PROB OPD_JOINT_ACTION_ROLLOUT OPD_PROFILE
OPD_ROLLOUT_GRAD_MODE OPD_ROLLOUT_GRAD_STEPS OPD_ROLLOUT_STEP_PAIRS
OPD_SERIAL_STUDENT_CFG OPD_TEACHER_TARGET_MODE OPD_TRANSITION_GROUP_WEIGHT
OUTPUT_DIR PARENT_STAGE1_CONTRACT_IDENTITY PARENT_STAGE1_PATH RESET_RESUME_STEP
RESUME_FROM_PATH RESUME_ONLINE_FROM_TARGET RESUME_OPTIMIZER_STATE
ROLLOUT_STEP_PAIRS RUN_TAG SAVE_INTERVAL SKIP_TARGET_STUDENT_FOR_COSMOS_LATENT
SKIP_TEACHER_COMPILE STAGE1_CKPT_NAME STAGE2_LINEAGE_JSON STOP_AFTER_STEP
STUDENT_BASE_MODEL_PATH TEACHER_PATH TRAIN_SEED TRANSFORMERS_OFFLINE USE_FSDP1
USE_OPD_AUX VARIANT_NAME VIDEO_LOSS_WEIGHT VIDEO_TRANSITION_PARAM
VIDEO_TRANSITION_WEIGHT WANDB_MODE WARMUP_STEPS
""".split()
)

CANONICAL_ENV = frozenset(
    _ALL_CURRENT_ENV - PINNED_OPERATIONAL_ENV - CLEARED_LEGACY_ENV
)


def _identity_field(name: str) -> str:
    special = {
        "ATTN_MODE": "variant.attention_mode",
        "DATASET_PATH": "variant.provenance.dataset.root",
        "COSMOS_POLICY_PATH": "variant.provenance.teacher.root",
        "COSMOS_VIDEO_VAE_MODEL_PATH": "variant.provenance.video_vae.root",
        "COSMOS_PREDICT25_LOCAL_MODEL_DIR": (
            "variant.provenance.local_model.root"
        ),
        "COSMOS_PREDICT2_REPO": "variant.provenance.cosmos_repo.root",
        "COSMOS_POLICY_PYTHON": "variant.provenance.worker.python_realpath",
        "COSMOS_POLICY_EXTRA_PYTHONPATH": (
            "variant.provenance.worker.extra_pythonpath"
        ),
        "COSMOS_LIBERO_VARIANT_JSON": "variant",
        "COSMOS_LIBERO_PROVENANCE_JSON": "variant.provenance",
        "COSMOS_PROGRESSIVE_STAGE": "variant.progressive_stage",
        "OPD_AUX_ACTION": "variant.action_opd_enabled",
        "OPD_DANCEOPD_ACTION_ENDPOINT_WEIGHT": (
            "variant.action_endpoint_weight"
        ),
        "OPD_DANCEOPD_ENDPOINT_WEIGHT": "variant.video_endpoint_weight",
        "OPD_DANCEOPD_VELOCITY_WEIGHT": "variant.video_velocity_weight",
        "OPD_DANCEOPD_ROLLOUT_STEPS": "variant.danceopd_rollout_steps",
        "OPD_ROLLOUT_STEP_PAIRS": "variant.rollout_step_pairs",
        "USE_OPD_AUX": "variant.use_opd_aux",
        "VARIANT_NAME": "variant.name",
        "OUTPUT_DIR": "recovery.output_dir",
        "COSMOS_PROGRESSIVE_OUTPUT_ROOT": "recovery.output_root",
        "RESUME_FROM_PATH": "recovery.resume_from_path",
        "RESUME_ONLINE_FROM_TARGET": "recovery.resume_online_from_target",
        "RESET_RESUME_STEP": "recovery.reset_resume_step",
        "RESUME_OPTIMIZER_STATE": "recovery.resume_optimizer_state",
        "PARENT_STAGE1_PATH": "recovery.parent_stage1_path",
        "PARENT_STAGE1_CONTRACT_IDENTITY": (
            "recovery.parent_stage1_contract_identity"
        ),
        "STAGE2_LINEAGE_JSON": "recovery.stage2_lineage",
        "STUDENT_BASE_MODEL_PATH": "recovery.student_base_model_path",
    }
    return special.get(name, "variant." + name.lower())


CANONICAL_ENV_TO_JSON_FIELDS = {
    name: (_identity_field(name),) for name in sorted(CANONICAL_ENV)
}

LAUNCHER_ENV_ACTION = {
    **{name: "set_canonical" for name in CANONICAL_ENV},
    **{name: "set_pinned" for name in PINNED_OPERATIONAL_ENV},
    **{name: "unset_legacy" for name in CLEARED_LEGACY_ENV},
}

DYNAMIC_ENV_SITES: Mapping[str, str] = {}


class EnvSchemaError(ValueError):
    """Raised when the audited config environment contract drifts."""


@dataclass(frozen=True, order=True)
class EnvRead:
    name: str
    file: str
    line: int
    access: str
    via_helper: str | None = None


@dataclass(frozen=True, order=True)
class DynamicEnvRead:
    file: str
    line: int
    access: str
    expression: str
    fingerprint: str


def _aliases(tree: ast.AST) -> set[str]:
    names = {"os"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "os":
                    names.add(item.asname or "os")
    return names


def _string_values(
    node: ast.AST,
    *,
    bindings: Mapping[str, frozenset[str]],
    constants: Mapping[str, frozenset[str]],
) -> frozenset[str] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return frozenset({node.value})
    if isinstance(node, ast.Name):
        return bindings.get(node.id) or constants.get(node.id)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        result = set()
        for item in node.elts:
            values = _string_values(item, bindings=bindings, constants=constants)
            if values is None:
                return None
            result.update(values)
        return frozenset(result)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_values(node.left, bindings=bindings, constants=constants)
        right = _string_values(node.right, bindings=bindings, constants=constants)
        if left is None or right is None:
            return None
        return frozenset(a + b for a in left for b in right)
    return None


def _env_call(node: ast.Call, aliases: set[str]):
    function = node.func
    if (
        isinstance(function, ast.Attribute)
        and function.attr == "get"
        and isinstance(function.value, ast.Attribute)
        and function.value.attr == "environ"
        and isinstance(function.value.value, ast.Name)
        and function.value.value.id in aliases
    ):
        return "os.environ.get", node.args[0] if node.args else None
    if (
        isinstance(function, ast.Attribute)
        and function.attr == "getenv"
        and isinstance(function.value, ast.Name)
        and function.value.id in aliases
    ):
        return "os.getenv", node.args[0] if node.args else None
    return None


def _helper_functions(tree: ast.Module, aliases: set[str]) -> dict[str, int]:
    helpers = {}
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        parameter_positions = {
            argument.arg: index for index, argument in enumerate(function.args.args)
        }
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            match = _env_call(node, aliases)
            if match is None or not isinstance(match[1], ast.Name):
                continue
            parameter = match[1].id
            if parameter in parameter_positions:
                helpers[function.name] = parameter_positions[parameter]
    return helpers


def _fingerprint(access: str, expression: ast.AST) -> str:
    source = access + ":" + ast.dump(expression, include_attributes=False)
    return hashlib.sha256(source.encode()).hexdigest()


def extract_config_env_reads(
    files: Iterable[str | Path],
) -> tuple[tuple[EnvRead, ...], tuple[DynamicEnvRead, ...]]:
    reads: set[EnvRead] = set()
    dynamic: set[DynamicEnvRead] = set()
    for raw_path in files:
        path = Path(raw_path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        aliases = _aliases(tree)
        helpers = _helper_functions(tree, aliases)
        constants: dict[str, frozenset[str]] = {}
        for statement in tree.body:
            if (
                isinstance(statement, (ast.Assign, ast.AnnAssign))
                and isinstance(
                    statement.value if isinstance(statement, ast.Assign) else statement.value,
                    ast.AST,
                )
            ):
                value = statement.value
                names = (
                    [target.id for target in statement.targets if isinstance(target, ast.Name)]
                    if isinstance(statement, ast.Assign)
                    else [statement.target.id]
                    if isinstance(statement.target, ast.Name)
                    else []
                )
                resolved = _string_values(value, bindings={}, constants=constants)
                if resolved is not None:
                    for name in names:
                        constants[name] = resolved

        def record(
            expression: ast.AST | None,
            *,
            access: str,
            line: int,
            bindings: Mapping[str, frozenset[str]],
            via_helper: str | None = None,
        ) -> None:
            if expression is None:
                return
            values = _string_values(
                expression, bindings=bindings, constants=constants
            )
            if values is None:
                fingerprint = _fingerprint(access, expression)
                dynamic.add(
                    DynamicEnvRead(
                        file=path.name,
                        line=line,
                        access=access,
                        expression=ast.dump(expression, include_attributes=False),
                        fingerprint=fingerprint,
                    )
                )
                return
            for name in values:
                reads.add(
                    EnvRead(
                        name=name,
                        file=path.name,
                        line=line,
                        access=access,
                        via_helper=via_helper,
                    )
                )

        def scan(node: ast.AST, bindings: Mapping[str, frozenset[str]]) -> None:
            if isinstance(node, ast.FunctionDef) and node.name in helpers:
                return
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                values = _string_values(
                    node.iter, bindings=bindings, constants=constants
                )
                if values is not None:
                    for value in values:
                        nested = dict(bindings)
                        nested[node.target.id] = frozenset({value})
                        for statement in node.body:
                            scan(statement, nested)
                    for statement in node.orelse:
                        scan(statement, bindings)
                    return
            if isinstance(node, ast.Call):
                match = _env_call(node, aliases)
                if match is not None:
                    record(
                        match[1],
                        access=match[0],
                        line=node.lineno,
                        bindings=bindings,
                    )
                elif isinstance(node.func, ast.Name) and node.func.id in helpers:
                    position = helpers[node.func.id]
                    expression = (
                        node.args[position] if len(node.args) > position else None
                    )
                    record(
                        expression,
                        access="helper",
                        line=node.lineno,
                        bindings=bindings,
                        via_helper=node.func.id,
                    )
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "environ"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id in aliases
            ):
                record(
                    node.slice,
                    access="os.environ[]",
                    line=node.lineno,
                    bindings=bindings,
                )
            for child in ast.iter_child_nodes(node):
                scan(child, bindings)

        for statement in tree.body:
            scan(statement, {})
    return tuple(sorted(reads)), tuple(sorted(dynamic))


def validate_config_chain(repo_root: str | Path) -> None:
    root = Path(repo_root)
    for index, relative in enumerate(CONFIG_CHAIN[:-1]):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        expected_module = (
            "distillation_flowmap." + Path(CONFIG_CHAIN[index + 1]).stem
        )
        matches = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == expected_module
            and any(
                item.name == "cfg" and item.asname == "_base_cfg"
                for item in node.names
            )
        ]
        if len(matches) != 1:
            raise EnvSchemaError(
                f"config chain edge must be exactly {expected_module} -> {relative}"
            )


def validate_environment_contract(
    repo_root: str | Path,
    *,
    files: Iterable[str | Path] | None = None,
    check_chain: bool = True,
) -> None:
    root = Path(repo_root)
    if check_chain:
        validate_config_chain(root)
    selected = list(files) if files is not None else [root / item for item in CONFIG_CHAIN]
    reads, dynamic = extract_config_env_reads(selected)
    actual = {read.name for read in reads}
    classified = CANONICAL_ENV | PINNED_OPERATIONAL_ENV | CLEARED_LEGACY_ENV
    missing = classified - actual
    extra = actual - classified
    if missing or extra:
        raise EnvSchemaError(
            f"environment schema drift: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    actual_dynamic = {site.fingerprint for site in dynamic}
    if actual_dynamic != set(DYNAMIC_ENV_SITES):
        raise EnvSchemaError(
            "dynamic environment schema drift: "
            f"expected={sorted(DYNAMIC_ENV_SITES)}, actual={sorted(actual_dynamic)}"
        )


def launcher_action_manifest() -> dict[str, str]:
    return {name: LAUNCHER_ENV_ACTION[name] for name in sorted(LAUNCHER_ENV_ACTION)}


def validate_launcher_environment(environment: Mapping[str, str]) -> None:
    required = CANONICAL_ENV | PINNED_OPERATIONAL_ENV
    missing = {name for name in required if name not in environment}
    if missing:
        raise EnvSchemaError(
            f"launcher environment is missing sealed values: {sorted(missing)}"
        )
    leaked = {name for name in CLEARED_LEGACY_ENV if name in environment}
    if leaked:
        raise EnvSchemaError(
            f"launcher environment retained cleared legacy values: {sorted(leaked)}"
        )
