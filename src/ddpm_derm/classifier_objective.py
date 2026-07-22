"""Class-weighted classifier objective definitions derived from train only."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from . import config

CLASS_WEIGHTING_NONE = "none"
CLASS_WEIGHTING_INVERSE_SQRT = "inverse_sqrt"
CLASS_WEIGHTING_INVERSE_FREQUENCY = "inverse_frequency"
LOSS_CROSS_ENTROPY = "cross_entropy"
LOSS_FOCAL_CROSS_ENTROPY = "focal_cross_entropy"
FOCAL_FORMULA = (
    "sum_i[-alpha_yi*(1-p_ti)^gamma*log(p_ti)]/sum_i(alpha_yi)"
)
FOCAL_REDUCTION = "weighted_mean_by_target_alpha"
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


def validate_loss_configuration(
    mode: str,
    *,
    loss_name: str = LOSS_CROSS_ENTROPY,
    focal_gamma: float | None = None,
) -> float | None:
    """Validate loss/weighting combinations without silently ignoring options."""
    if loss_name == LOSS_CROSS_ENTROPY:
        if focal_gamma is not None:
            raise ValueError("cross_entropy does not accept focal_gamma")
        return None
    if loss_name != LOSS_FOCAL_CROSS_ENTROPY:
        raise ValueError(f"unsupported loss name: {loss_name!r}")
    if mode != CLASS_WEIGHTING_INVERSE_FREQUENCY:
        raise ValueError(
            "focal_cross_entropy requires class_weighting='inverse_frequency'"
        )
    if focal_gamma is None:
        raise ValueError("focal_cross_entropy requires an explicit focal_gamma")
    gamma = float(focal_gamma)
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("focal_gamma must be finite and >= 0")
    return gamma


def build_training_objective(
    mode: str,
    full_train_frame,
    *,
    loss_name: str = LOSS_CROSS_ENTROPY,
    focal_gamma: float | None = None,
):
    """Return an optional immutable objective and its ordered weight vector."""
    gamma = validate_loss_configuration(
        mode, loss_name=loss_name, focal_gamma=focal_gamma
    )
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
        if loss_name == LOSS_FOCAL_CROSS_ENTROPY:
            objective.update(
                {
                    "loss_name": LOSS_FOCAL_CROSS_ENTROPY,
                    "focal_gamma": gamma,
                    "focal_formula": FOCAL_FORMULA,
                    "focal_reduction": FOCAL_REDUCTION,
                }
            )
        return objective, weights
    raise ValueError(f"unsupported class weighting mode: {mode!r}")
