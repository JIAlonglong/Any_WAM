"""Small validation helpers shared by offline rollout evaluators."""

from collections.abc import Sequence


def normalize_rollout_steps(values: Sequence[int]) -> list[int]:
    """Return sorted unique positive rollout step counts."""
    steps = [int(value) for value in values]
    if not steps or any(value <= 0 for value in steps):
        raise ValueError(
            "rollout steps must be a non-empty sequence of positive integers"
        )
    if len(set(steps)) != len(steps):
        raise ValueError("rollout steps must not contain duplicates")
    return sorted(steps)
