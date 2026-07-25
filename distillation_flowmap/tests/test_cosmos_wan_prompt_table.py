import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from distillation_flowmap.cosmos_wan_prompt_table import (
    PROMPT_SHAPE,
    build_prompt_embeddings,
    canonical_manifest_digest,
    inspect_wan_base,
    load_canonical_tasks,
    save_prompt_table_atomic,
    validate_prompt_table,
)


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "distillation_flowmap" / "build_cosmos_wan_prompt_table.py"


class _TokenBatch:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class _FakeTokenizer:
    def __call__(self, prompts, **kwargs):
        assert kwargs == {
            "padding": "max_length",
            "max_length": 512,
            "truncation": True,
            "add_special_tokens": True,
            "return_attention_mask": True,
            "return_tensors": "pt",
        }
        rows = []
        masks = []
        for index, _ in enumerate(prompts):
            row = torch.zeros(512, dtype=torch.long)
            mask = torch.zeros(512, dtype=torch.long)
            row[:2] = torch.tensor([index + 1, index + 2])
            mask[:2] = 1
            rows.append(row)
            masks.append(mask)
        return _TokenBatch(torch.stack(rows), torch.stack(masks))


class _EncoderOutput:
    def __init__(self, value):
        self.last_hidden_state = value


class _FakeEncoder:
    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        batch, sequence = input_ids.shape
        assert sequence == 512
        assert torch.equal(attention_mask[:, :2], torch.ones(batch, 2, dtype=torch.long))
        value = torch.ones(batch, sequence, 4096, dtype=torch.float16)
        value[:, 2:] = 7
        return _EncoderOutput(value)


def _wan_base(tmp_path: Path) -> Path:
    root = tmp_path / "wan-base"
    for relative, payload in (
        ("tokenizer/tokenizer_config.json", {"model_max_length": 512}),
        ("text_encoder/config.json", {"d_model": 4096}),
        ("transformer/config.json", {"_class_name": "WanTransformer3DModel"}),
        ("model_index.json", {"_class_name": "WanPipeline"}),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    (root / "text_encoder/model.safetensors").write_bytes(b"encoder")
    (root / "transformer/diffusion_pytorch_model.safetensors").write_bytes(b"student")
    return root


def _payload(tasks, base_identity):
    return {
        "format": "flash_wam.libero_wan_prompt_table.v1",
        "embeddings": {
            task: torch.zeros(PROMPT_SHAPE, dtype=torch.float16) for task in tasks
        },
        "metadata": {
            "task_manifest_sha256": canonical_manifest_digest(tasks),
            "wan_base_identity": base_identity,
        },
    }


def test_checked_in_manifest_has_exactly_40_unique_official_task_strings():
    tasks = load_canonical_tasks()
    assert len(tasks) == 40
    assert len(set(tasks)) == 40
    assert tasks[0] == "open the top drawer and put the bowl inside"
    assert tasks[-1] == "put both the alphabet soup and the tomato sauce in the basket"


def test_builder_uses_wan_contract_and_zero_pads_tokens_after_attention_mask():
    tasks = ("task alpha", "task beta")
    embeddings = build_prompt_embeddings(
        tasks,
        tokenizer=_FakeTokenizer(),
        text_encoder=_FakeEncoder(),
        device="cpu",
        output_dtype=torch.float16,
        prompt_cleaner=lambda value: value,
        batch_size=1,
    )
    assert set(embeddings) == set(tasks)
    for value in embeddings.values():
        assert value.shape == PROMPT_SHAPE
        assert value.dtype == torch.float16
        assert torch.equal(value[:, :2], torch.ones(1, 2, 4096, dtype=torch.float16))
        assert torch.count_nonzero(value[:, 2:]) == 0


def test_wan_base_identity_rejects_missing_components_and_changes_with_artifacts(tmp_path):
    root = _wan_base(tmp_path)
    (root / "model_index.json").unlink()
    first = inspect_wan_base(root)
    assert len(first["identity_sha256"]) == 64
    assert first["resolved_path"] == str(root.resolve())

    (root / "text_encoder/model.safetensors").write_bytes(b"changed-size")
    second = inspect_wan_base(root)
    assert second["identity_sha256"] != first["identity_sha256"]

    (root / "tokenizer/tokenizer_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="tokenizer/tokenizer_config.json"):
        inspect_wan_base(root)


def test_atomic_save_refuses_overwrite_and_validator_rejects_missing_or_wrong_shape(tmp_path):
    tasks = ("task alpha", "task beta")
    output = tmp_path / "table.pt"
    payload = _payload(tasks, "base-id")
    save_prompt_table_atomic(output, payload)
    assert output.is_file()
    validate_prompt_table(output, tasks=tasks, expected_wan_base_identity="base-id")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_prompt_table_atomic(output, payload)

    missing = tmp_path / "missing.pt"
    bad_missing = _payload(tasks, "base-id")
    bad_missing["embeddings"].pop("task beta")
    torch.save(bad_missing, missing)
    with pytest.raises(ValueError, match="missing.*task beta"):
        validate_prompt_table(missing, tasks=tasks, expected_wan_base_identity="base-id")

    wrong_shape = tmp_path / "wrong-shape.pt"
    bad_shape = _payload(tasks, "base-id")
    bad_shape["embeddings"]["task beta"] = torch.zeros(1, 512, 1024)
    torch.save(bad_shape, wrong_shape)
    with pytest.raises(ValueError, match=r"\[1, 512, 4096\]"):
        validate_prompt_table(
            wrong_shape, tasks=tasks, expected_wan_base_identity="base-id"
        )


def test_validator_rejects_wrong_manifest_or_wan_base_identity(tmp_path):
    tasks = ("task alpha",)
    path = tmp_path / "table.pt"
    payload = _payload(tasks, "base-a")
    payload["metadata"]["task_manifest_sha256"] = "0" * 64
    torch.save(payload, path)
    with pytest.raises(ValueError, match="task manifest"):
        validate_prompt_table(path, tasks=tasks, expected_wan_base_identity="base-a")

    payload = _payload(tasks, "base-a")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Wan base identity"):
        validate_prompt_table(path, tasks=tasks, expected_wan_base_identity="base-b")


def test_cli_dry_run_prints_plan_and_writes_nothing(tmp_path):
    base = _wan_base(tmp_path)
    output = tmp_path / "table.pt"
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--wan-base-model",
            str(base),
            "--output",
            str(output),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "TASK_COUNT=40" in result.stdout
    assert "PROMPT_SHAPE=1,512,4096" in result.stdout
    assert f"OUTPUT={output}" in result.stdout
    assert "ACTION=build" in result.stdout
    assert not output.exists()


def test_cli_validate_only_checks_explicit_all_40_table_without_loading_model(tmp_path):
    base = _wan_base(tmp_path)
    tasks = load_canonical_tasks()
    identity = inspect_wan_base(base)["identity_sha256"]
    table = tmp_path / "table.pt"
    torch.save(_payload(tasks, identity), table)

    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--wan-base-model",
            str(base),
            "--output",
            str(table),
            "--validate-only",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "VALID=1" in result.stdout
    assert "TASK_COUNT=40" in result.stdout
