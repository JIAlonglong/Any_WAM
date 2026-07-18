"""Pure, testable specifications for Cosmos mixed OPD rollout budgets.

The policies in this module affect only scheduled standalone full-OPD updates.
Keeping the distribution and sampling logic independent of the trainer makes
the eight-GPU runner and tests share the exact same immutable definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Callable, Mapping, MutableMapping, Sequence


RolloutStepPair = tuple[int, int]


def rollout_pair_label(pair: RolloutStepPair) -> str:
    """Return the stable inference-budget label for one (teacher, student) pair."""
    teacher_steps, student_steps = pair
    return f"s{int(student_steps)}" if int(teacher_steps) in (4, 8) else (
        f"t{int(teacher_steps)}_s{int(student_steps)}"
    )


@dataclass(frozen=True)
class CosmosMixedStepPolicySpec:
    """An immutable rollout-pair distribution used by a mixed OPD policy."""

    name: str
    rollout_step_pairs: tuple[RolloutStepPair, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        name = str(self.name).strip().lower()
        if not name:
            raise ValueError("Cosmos mixed-step policy name must be non-empty")
        pairs = tuple((int(teacher), int(student)) for teacher, student in self.rollout_step_pairs)
        if not pairs:
            raise ValueError("Cosmos mixed-step policy needs at least one rollout pair")
        if any(teacher <= 0 or student <= 0 for teacher, student in pairs):
            raise ValueError("Cosmos rollout steps must be positive integers")
        raw_weights = tuple(float(weight) for weight in self.weights)
        if len(raw_weights) != len(pairs):
            raise ValueError("Cosmos rollout weights must match rollout-pair count")
        if any(not math.isfinite(weight) or weight <= 0.0 for weight in raw_weights):
            raise ValueError("Cosmos rollout weights must be finite and positive")
        total = math.fsum(raw_weights)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("Cosmos rollout weights must have a finite positive sum")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "rollout_step_pairs", pairs)
        object.__setattr__(self, "weights", tuple(weight / total for weight in raw_weights))


_MIXED_POLICY_SPECS: Mapping[str, CosmosMixedStepPolicySpec] = {
    "universe": CosmosMixedStepPolicySpec(
        name="universe",
        rollout_step_pairs=((4, 1), (4, 2), (8, 4)),
        weights=(0.50, 0.30, 0.20),
    ),
    "s2": CosmosMixedStepPolicySpec(
        name="s2",
        rollout_step_pairs=((4, 1), (4, 2), (8, 4)),
        weights=(0.20, 0.60, 0.20),
    ),
    "s1": CosmosMixedStepPolicySpec(
        name="s1",
        rollout_step_pairs=((4, 1), (4, 2), (8, 4)),
        weights=(0.70, 0.20, 0.10),
    ),
}


def get_mixed_step_policy_spec(name: str) -> CosmosMixedStepPolicySpec:
    """Return the immutable spec for a supported mixed-step policy name."""
    normalized_name = str(name).strip().lower()
    try:
        return _MIXED_POLICY_SPECS[normalized_name]
    except KeyError as exc:
        raise ValueError(
            "Unknown Cosmos mixed-step policy "
            f"{normalized_name!r}; expected one of {sorted(_MIXED_POLICY_SPECS)}"
        ) from exc


def mixed_step_policy_names() -> tuple[str, ...]:
    """Return supported policy names in stable order."""
    return tuple(_MIXED_POLICY_SPECS)


def parse_forced_indices(
    value: str | Sequence[int | str] | None,
    spec: CosmosMixedStepPolicySpec,
) -> tuple[int, ...]:
    """Parse an optional deterministic preflight sequence into pair indices.

    The launcher may use readable labels (``s1,s2,s4``) or explicit indices
    (``0,1,2``).  The sequence is deliberately cyclic; a nine-update preflight
    can exercise all three modes exactly three times without relying on random
    draws.
    """
    if value is None or value == "":
        return ()
    tokens = (
        [part.strip() for part in value.split(",")]
        if isinstance(value, str)
        else [str(part).strip() for part in value]
    )
    if not tokens or any(not token for token in tokens):
        raise ValueError("Forced Cosmos mixed-step sequence must not contain empty values")
    labels = {
        rollout_pair_label(pair): index
        for index, pair in enumerate(spec.rollout_step_pairs)
    }
    labels.update(
        {
            f"t{teacher}_s{student}": index
            for index, (teacher, student) in enumerate(spec.rollout_step_pairs)
        }
    )
    indices = []
    for token in tokens:
        normalized = token.lower()
        if normalized in labels:
            index = labels[normalized]
        else:
            try:
                index = int(normalized)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown forced Cosmos mixed-step label {token!r}; "
                    f"expected one of {sorted(labels)} or an index"
                ) from exc
        if not 0 <= index < len(spec.rollout_step_pairs):
            raise ValueError(
                f"Forced Cosmos mixed-step index {index} is outside "
                f"[0, {len(spec.rollout_step_pairs)})"
            )
        indices.append(index)
    return tuple(indices)


def sample_weighted_index(
    weights: Sequence[float],
    *,
    generator: random.Random | None = None,
    seed: int | None = None,
) -> int:
    """Sample a normalized weighted index without importing model code."""
    normalized = CosmosMixedStepPolicySpec(
        name="sample",
        rollout_step_pairs=tuple((1, index + 1) for index in range(len(weights))),
        weights=tuple(weights),
    ).weights
    if generator is not None and seed is not None:
        raise ValueError("Pass either generator or seed, not both")
    rng = generator if generator is not None else random.Random(seed)
    threshold = rng.random()
    cumulative = 0.0
    for index, weight in enumerate(normalized):
        cumulative += weight
        if threshold < cumulative:
            return index
    return len(normalized) - 1


@dataclass(frozen=True)
class MixedStepSelection:
    """A single rank-synchronized full-OPD endpoint rollout selection."""

    policy_name: str
    index: int
    rollout_step_pair: RolloutStepPair
    label: str
    forced: bool

    @property
    def teacher_steps(self) -> int:
        return self.rollout_step_pair[0]

    @property
    def student_steps(self) -> int:
        return self.rollout_step_pair[1]


@dataclass(frozen=True)
class CosmosMixedDanceSchedule:
    rollout_steps: int
    velocity_weight: float


_MIXED_DANCEOPD_SCHEDULES = {
    (4, 1): CosmosMixedDanceSchedule(rollout_steps=1, velocity_weight=0.25),
    (4, 2): CosmosMixedDanceSchedule(rollout_steps=2, velocity_weight=0.50),
    (8, 4): CosmosMixedDanceSchedule(rollout_steps=4, velocity_weight=1.00),
}


def get_mixed_danceopd_schedule(selection: MixedStepSelection) -> CosmosMixedDanceSchedule:
    try:
        return _MIXED_DANCEOPD_SCHEDULES[selection.rollout_step_pair]
    except KeyError as exc:
        raise ValueError(
            "Unsupported Cosmos mixed DanceOPD pair "
            f"{selection.rollout_step_pair!r}"
        ) from exc


def _forced_index(
    forced_indices: Sequence[int] | None,
    *,
    selection_ordinal: int,
    pair_count: int,
) -> int | None:
    if not forced_indices:
        return None
    if selection_ordinal < 0:
        raise ValueError("selection_ordinal must be non-negative")
    index = int(forced_indices[selection_ordinal % len(forced_indices)])
    if not 0 <= index < pair_count:
        raise ValueError(
            f"Forced Cosmos mixed-step index {index} is outside [0, {pair_count})"
        )
    return index


def select_rank_synchronized_pair(
    spec: CosmosMixedStepPolicySpec,
    *,
    rank: int = 0,
    broadcast_index: Callable[[int], int] | None = None,
    generator: random.Random | None = None,
    seed: int | None = None,
    forced_indices: Sequence[int] | None = None,
    selection_ordinal: int = 0,
) -> MixedStepSelection:
    """Select once on rank zero and return the broadcast endpoint pair everywhere.

    ``broadcast_index`` intentionally accepts and returns an integer so this
    helper stays pure and can be tested without initializing torch.distributed.
    The trainer supplies the tiny tensor-broadcast adapter at runtime.
    """
    if rank == 0:
        forced_index = _forced_index(
            forced_indices,
            selection_ordinal=selection_ordinal,
            pair_count=len(spec.rollout_step_pairs),
        )
        index = (
            forced_index
            if forced_index is not None
            else sample_weighted_index(spec.weights, generator=generator, seed=seed)
        )
    else:
        index = 0
        forced_index = None
    if broadcast_index is not None:
        index = int(broadcast_index(index))
    if not 0 <= index < len(spec.rollout_step_pairs):
        raise ValueError(
            f"Broadcast Cosmos mixed-step index {index} is outside "
            f"[0, {len(spec.rollout_step_pairs)})"
        )
    pair = spec.rollout_step_pairs[index]
    return MixedStepSelection(
        policy_name=spec.name,
        index=index,
        rollout_step_pair=pair,
        label=rollout_pair_label(pair),
        forced=bool(forced_indices),
    )


def record_selection(
    histogram: MutableMapping[str, int],
    selection: MixedStepSelection,
) -> dict[str, int]:
    """Add one selected endpoint mode to the rank-zero aggregate histogram."""
    histogram[selection.label] = int(histogram.get(selection.label, 0)) + 1
    return {label: int(count) for label, count in sorted(histogram.items())}


def build_selection_record(
    *,
    spec: CosmosMixedStepPolicySpec,
    selection: MixedStepSelection,
    histogram: Mapping[str, int],
    seed: int,
    global_step: int,
    selection_ordinal: int,
) -> dict[str, object]:
    """Build a JSON-safe provenance record for one standalone full-OPD update."""
    return {
        "policy_name": spec.name,
        "rollout_step_pairs": [list(pair) for pair in spec.rollout_step_pairs],
        "weights": list(spec.weights),
        "seed": int(seed),
        "global_step": int(global_step),
        "selection_ordinal": int(selection_ordinal),
        "pair_index": int(selection.index),
        "pair_label": selection.label,
        "teacher_steps": int(selection.teacher_steps),
        "student_steps": int(selection.student_steps),
        "forced": bool(selection.forced),
        "histogram": {label: int(count) for label, count in sorted(histogram.items())},
    }


def append_selection_jsonl(path: str | Path, record: Mapping[str, object]) -> None:
    """Append one rank-zero selection record without pulling in trainer state."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
