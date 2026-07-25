#!/usr/bin/env python3
"""Build or preflight the canonical all-40 Wan LIBERO prompt table."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from distillation_flowmap.cosmos_wan_prompt_table import (
    PROMPT_SHAPE,
    build_prompt_embeddings,
    inspect_wan_base,
    load_canonical_tasks,
    prompt_table_payload,
    save_prompt_table_atomic,
    validate_prompt_table,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan-base-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _emit(key: str, value: object) -> None:
    print(f"{key}={value}", flush=True)


def main() -> None:
    args = _parse_args()
    tasks = load_canonical_tasks(args.manifest) if args.manifest else load_canonical_tasks()
    wan_base = inspect_wan_base(args.wan_base_model)
    output = Path(args.output).expanduser()

    _emit("ACTION", "validate" if args.validate_only else "build")
    _emit("TASK_COUNT", len(tasks))
    _emit("PROMPT_SHAPE", ",".join(str(value) for value in PROMPT_SHAPE))
    _emit("WAN_BASE_MODEL", wan_base["resolved_path"])
    _emit("WAN_BASE_IDENTITY", wan_base["identity_sha256"])
    _emit("OUTPUT", output)

    if args.validate_only:
        validate_prompt_table(
            output,
            tasks=tasks,
            expected_wan_base_identity=wan_base["identity_sha256"],
        )
        _emit("VALID", 1)
        return
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite prompt table: {output}")
    if args.dry_run:
        _emit("DRY_RUN", 1)
        return

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import T5TokenizerFast, UMT5EncoderModel

    root = Path(wan_base["resolved_path"])
    tokenizer = T5TokenizerFast.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        root / "text_encoder",
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    embeddings = build_prompt_embeddings(
        tasks,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=device,
        output_dtype=dtype,
        prompt_cleaner=prompt_clean,
        batch_size=args.batch_size,
    )
    payload = prompt_table_payload(embeddings, tasks=tasks, wan_base=wan_base)
    save_prompt_table_atomic(output, payload)
    validate_prompt_table(
        output,
        tasks=tasks,
        expected_wan_base_identity=wan_base["identity_sha256"],
    )
    _emit("VALID", 1)


if __name__ == "__main__":
    main()
