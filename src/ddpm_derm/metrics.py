"""Classification metrics in pure numpy (no sklearn dependency).

Primary metric for this project is df F1. We also report per-class recall,
macro-F1 and the confusion matrix. Accuracy is reported but never treated as the
headline (it is inflated by the majority class under heavy imbalance).
"""

from __future__ import annotations

import numpy as np

from . import config


def confusion_matrix(y_true, y_pred, num_classes: int = config.NUM_CLASSES) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _precision_recall_f1(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)          # true per class
    predicted = cm.sum(axis=0).astype(float)        # predicted per class
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    return precision, recall, f1, support


def classification_summary(
    y_true, y_pred, target_class: str = config.TARGET_CLASS
) -> dict:
    """Return the metric bundle used throughout the project.

    Keys: target_f1 (primary), target_recall, macro_f1, accuracy,
    per_class_recall, per_class_f1, per_class_precision, confusion_matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, support = _precision_recall_f1(cm)
    target_idx = config.CLASS_TO_IDX[target_class]
    total = cm.sum()
    accuracy = float(np.diag(cm).sum() / total) if total else 0.0

    names = config.CLASS_NAMES
    return {
        "target_class": target_class,
        "target_f1": float(f1[target_idx]),
        "target_recall": float(recall[target_idx]),
        "macro_f1": float(f1.mean()),
        "accuracy": accuracy,
        "per_class_recall": {names[i]: float(recall[i]) for i in range(len(names))},
        "per_class_precision": {names[i]: float(precision[i]) for i in range(len(names))},
        "per_class_f1": {names[i]: float(f1[i]) for i in range(len(names))},
        "support": {names[i]: int(support[i]) for i in range(len(names))},
        "confusion_matrix": cm.tolist(),
    }
