"""Shared packed-sequence padding policy for FlexAttention training."""

import os


FLEX_ATTENTION_ALIGNMENT = 128
FLEX_ATTENTION_STATIC_SEQ_LEN_ENV = "FLEX_ATTN_STATIC_SEQ_LEN"
DEFAULT_FLEX_ATTENTION_STATIC_SEQ_LEN = 8704


def resolve_padded_length(total_length, *, attn_mode=None):
    """Return tail padding for the selected attention backend.

    FlexAttention is compiled with static shapes in RobotWin training.  When a
    ceiling is configured, every Flex call is therefore padded to that physical
    length while the existing ``-1`` mask tail preserves the logical length.
    Other backends retain the historical nearest-128 padding behavior.
    """
    total_length = int(total_length)
    if total_length < 0:
        raise ValueError("packed sequence length must be non-negative")

    nearest_alignment_padding = (-total_length) % FLEX_ATTENTION_ALIGNMENT
    if str(attn_mode).lower() != "flex":
        return nearest_alignment_padding

    configured = os.environ.get(FLEX_ATTENTION_STATIC_SEQ_LEN_ENV)
    if configured is None or not configured.strip():
        configured = str(DEFAULT_FLEX_ATTENTION_STATIC_SEQ_LEN)

    try:
        ceiling = int(configured)
    except ValueError as exc:
        raise ValueError(
            f"{FLEX_ATTENTION_STATIC_SEQ_LEN_ENV} must be a positive integer"
        ) from exc
    if ceiling <= 0:
        raise ValueError(
            f"{FLEX_ATTENTION_STATIC_SEQ_LEN_ENV} must be a positive integer"
        )
    if ceiling % FLEX_ATTENTION_ALIGNMENT:
        raise ValueError(
            f"{FLEX_ATTENTION_STATIC_SEQ_LEN_ENV} must be a multiple of "
            f"{FLEX_ATTENTION_ALIGNMENT}"
        )
    if total_length > ceiling:
        raise RuntimeError(
            "packed sequence length {} exceeds {}={}; increase the ceiling".format(
                total_length, FLEX_ATTENTION_STATIC_SEQ_LEN_ENV, ceiling
            )
        )
    return ceiling - total_length
