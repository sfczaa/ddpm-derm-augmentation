"""Torch-free sampling policy and weight calculations for DDPM training."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

NATURAL = "natural"
SQRT_BALANCED = "sqrt_balanced"
SAMPLER_STRATEGIES = (NATURAL, SQRT_BALANCED)
DEFAULT_SAMPLER_STRATEGY = NATURAL


def validate_sampler_strategy(strategy: str) -> str:
    if strategy not in SAMPLER_STRATEGIES:
        raise ValueError(
            f"unknown sampler strategy {strategy!r}; expected one of "
            f"{SAMPLER_STRATEGIES}"
        )
    return strategy


def sampling_plan(strategy: str) -> dict[str, bool]:
    """Return mutually exclusive DataLoader shuffle/sampler settings."""
    strategy = validate_sampler_strategy(strategy)
    return {
        "shuffle": strategy == NATURAL,
        "use_weighted_sampler": strategy == SQRT_BALANCED,
    }


def per_sample_weights(labels: Sequence[int], strategy: str) -> list[float]:
    """Return one raw weight per row.

    sqrt_balanced assigns each sample in class c weight 1/sqrt(n_c).
    Natural sampling uses uniform row weights.
    """
    strategy = validate_sampler_strategy(strategy)
    labels = [int(label) for label in labels]
    if not labels:
        raise ValueError("cannot build sampler weights for an empty label sequence")
    counts = Counter(labels)
    if strategy == NATURAL:
        return [1.0] * len(labels)
    return [1.0 / math.sqrt(counts[label]) for label in labels]


def sampler_summary(
    labels: Sequence[int],
    strategy: str,
    class_names: Mapping[int, str] | None = None,
) -> dict[str, dict[str, float | int]]:
    """Summarize counts, row weights, and expected sampling proportions."""
    strategy = validate_sampler_strategy(strategy)
    labels = [int(label) for label in labels]
    weights = per_sample_weights(labels, strategy)
    counts = Counter(labels)
    class_weight = {label: weights[labels.index(label)] for label in counts}
    total_weight = sum(counts[label] * class_weight[label] for label in counts)
    total_samples = len(labels)

    result = {}
    for label in sorted(counts):
        name = class_names[label] if class_names is not None else str(label)
        proportion = counts[label] * class_weight[label] / total_weight
        result[name] = {
            "count": counts[label],
            "per_sample_weight": class_weight[label],
            "expected_sampling_proportion": proportion,
            "expected_samples_per_epoch": proportion * total_samples,
        }
    return result


def checkpoint_sampler_strategy(checkpoint: Mapping) -> str:
    """Read a saved strategy; legacy checkpoints are known-natural."""
    saved = checkpoint.get("sampler_strategy")
    if saved is None:
        saved = checkpoint.get("config", {}).get(
            "sampler_strategy", DEFAULT_SAMPLER_STRATEGY
        )
    return validate_sampler_strategy(saved)


def require_matching_checkpoint_strategy(
    checkpoint: Mapping, current_strategy: str
) -> str:
    current_strategy = validate_sampler_strategy(current_strategy)
    saved_strategy = checkpoint_sampler_strategy(checkpoint)
    if saved_strategy != current_strategy:
        raise ValueError(
            "sampler strategy mismatch: "
            f"checkpoint={saved_strategy!r} current={current_strategy!r}"
        )
    return saved_strategy
