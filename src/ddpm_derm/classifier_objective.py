"""Class-weighted classifier objective definitions derived from train only."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from . import config

CLASS_WEIGHTING_NONE = "none"
CLASS_WEIGHTING_INVERSE_SQRT = "inverse_sqrt"
CLASS_WEIGHTING_INVERSE_FREQUENCY = "inverse_frequency"
# CLASS_WEIGHT_FORMULA names the inverse-sqrt formula (v2); it is frozen so the
# v2 objective/identity stay byte-identical. inverse-frequency (v3) has its own.
CLASS_WEIGHT_FORMULA = "(1/sqrt(n_c))/mean_j(1/sqrt(n_j))"
INVERSE_FREQUENCY_FORMULA = "(1/n_c)/mean_j(1/n_j)"
CLASS_ORDER = tuple(config.CLASS_NAMES)


def ordered_class_counts(frame) -> dict[str, int]:
    """Return strict canonical counts from one complete training frame."""
    if "label_idx" not in frame.columns:
        raise ValueError("training frame is missing label_idx")
    labels = np.asarray(frame["label_idx"])
    if not np.issubdtype(labels.dtype, np.number):
        raise ValueError("training labels must be numeric class indices")
    numeric = labels.astype(float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("training labels must be finite integer class indices")
    indices = numeric.astype(int)
    unknown = sorted(set(indices) - set(range(config.NUM_CLASSES)))
    if unknown:
        raise ValueError(f"training frame contains unknown class indices: {unknown}")
    counts = {
        name: int(np.count_nonzero(indices == class_idx))
        for class_idx, name in enumerate(CLASS_ORDER)
    }
    missing = [name for name, count in counts.items() if count <= 0]
    if missing:
        raise ValueError(f"training frame has missing or zero-count classes: {missing}")
    if sum(counts.values()) != len(frame):
        raise ValueError("training class counts do not cover the full frame")
    return counts


def inverse_sqrt_weights(counts: Mapping[str, int]) -> np.ndarray:
    """Compute canonical inverse-sqrt weights normalized to mean one."""
    if tuple(counts) != CLASS_ORDER:
        raise ValueError(
            f"class count order must be {list(CLASS_ORDER)}, got {list(counts)}"
        )
    values = np.asarray(list(counts.values()), dtype=float)
    if (
        not np.isfinite(values).all()
        or np.any(values <= 0)
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError(
            "class counts must be finite positive integer values"
        )
    raw = 1.0 / np.sqrt(values)
    weights = raw / raw.mean()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("computed class weights must be finite and positive")
    return weights


def inverse_frequency_weights(counts: Mapping[str, int]) -> np.ndarray:
    """Compute canonical inverse-frequency weights normalized to mean one.

    Mirrors ``inverse_sqrt_weights`` but without the square root, so each class
    contributes equally to the weighted loss: ``n_c * weight_c`` is the same
    constant for every class.
    """
    if tuple(counts) != CLASS_ORDER:
        raise ValueError(
            f"class count order must be {list(CLASS_ORDER)}, got {list(counts)}"
        )
    values = np.asarray(list(counts.values()), dtype=float)
    if (
        not np.isfinite(values).all()
        or np.any(values <= 0)
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError(
            "class counts must be finite positive integer values"
        )
    raw = 1.0 / values
    weights = raw / raw.mean()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("computed class weights must be finite and positive")
    return weights


def _weighted_objective(counts, weights, *, class_weighting: str, formula: str):
    """Assemble the immutable objective dict shared by every weighting mode."""
    return {
        "loss_name": "cross_entropy",
        "class_weighting": class_weighting,
        "class_weight_formula": formula,
        "class_weight_normalization": "mean_one",
        "class_weight_count_source": (
            "full_post_variant_train_frame_before_limit"
        ),
        "class_order": list(CLASS_ORDER),
        "class_counts": counts,
        "class_weights": [float(value) for value in weights],
    }


def build_training_objective(mode: str, full_train_frame):
    """Return an optional immutable objective and its ordered weight vector."""
    if mode == CLASS_WEIGHTING_NONE:
        return None, None
    if mode == CLASS_WEIGHTING_INVERSE_SQRT:
        counts = ordered_class_counts(full_train_frame)
        weights = inverse_sqrt_weights(counts)
        objective = _weighted_objective(
            counts,
            weights,
            class_weighting="inverse_sqrt_train_frequency",
            formula=CLASS_WEIGHT_FORMULA,
        )
        return objective, weights
    if mode == CLASS_WEIGHTING_INVERSE_FREQUENCY:
        counts = ordered_class_counts(full_train_frame)
        weights = inverse_frequency_weights(counts)
        objective = _weighted_objective(
            counts,
            weights,
            class_weighting="inverse_train_frequency",
            formula=INVERSE_FREQUENCY_FORMULA,
        )
        return objective, weights
    raise ValueError(f"unsupported class weighting mode: {mode!r}")
