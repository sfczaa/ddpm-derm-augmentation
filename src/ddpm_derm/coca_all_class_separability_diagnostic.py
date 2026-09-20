"""Frozen-CoCa all-class embedding separability diagnostic.

Descriptive post-mixture diagnostic. It asks whether the frozen CoCa image
representation keeps enough signal to separate real ``df`` from the other six
classes, using only the fixed ``train`` (6995) and ``validation`` (1510) splits
- never the test split, C1 duplication, or any synthetic image/candidate, and
without training an image-level classifier head.

Four fixed analyses are reported through the project's
``metrics.classification_summary`` so the canonical class order never drifts:

  A. seven-class nearest-centroid (train centroids, cosine, canonical tie-break),
  B. cosine k-NN at the fixed k = (1, 5, 10) (majority -> summed-similarity ->
     canonical-index tie-break, deterministic stable ordering),
  C. validation-df representation margin (nearest train non-df minus nearest
     train df cosine distance) plus df neighbour purity per k, and
  D. one fixed balanced multinomial logistic-regression probe fit on train only.

Validation labels are used only once, after every fit has finished: the pipeline
runs Phase A (train-only fitting, logistic first), Phase B (predictions from
validation embeddings only), then Phase C (one-shot evaluation). One immutable
diagnostic identity is built once and shared by every artifact; the three formal
records are byte-identical and certified by a non-circular SHA-256 sidecar; and
the eight-key embedding NPZ is validated for exact dtype/shape/value before and
after writing. The module reuses the frozen-embedding helpers (``l2_normalize``,
``_distance_summary``, ``_write_npz_atomic``, ``prepare_output_dir``,
``encode_group``, ``sha256_file``) and ``coca_run.write_json_atomic``; it never
selects a condition, trains the deployed classifier, or touches test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

from . import coca_run, config, manifests, metrics
from .coca_embedding_diagnostic import (
    _distance_summary,
    _write_npz_atomic,
    encode_group,
    l2_normalize,
    prepare_output_dir,
    sha256_file,
)


DIAGNOSTIC_VERSION = "v1_all_class_separability_safe_v2"
FEATURE_DIM = 512
KNN_K_VALUES = (1, 5, 10)
INPUT_RESOLUTION = (224, 224)

# The fixed group-to-manifest mapping deliberately never resolves the test
# split; ``load_all_class_groups`` reads only these two.
GROUP_TO_MANIFEST_SPLIT = {"train": "train", "validation": "val"}

EXPECTED_TRAIN_CLASS_COUNTS = {
    "akiec": 226,
    "bcc": 348,
    "bkl": 778,
    "df": 85,
    "mel": 782,
    "nv": 4684,
    "vasc": 92,
}
EXPECTED_VALIDATION_CLASS_COUNTS = {
    "akiec": 45,
    "bcc": 77,
    "bkl": 166,
    "df": 14,
    "mel": 169,
    "nv": 1016,
    "vasc": 23,
}
EXPECTED_TRAIN_ROWS = sum(EXPECTED_TRAIN_CLASS_COUNTS.values())            # 6995
EXPECTED_VALIDATION_ROWS = sum(EXPECTED_VALIDATION_CLASS_COUNTS.values())  # 1510

# One fixed configuration only: no grid, CV, trial loop, or validation tuning.
LOGISTIC_REGRESSION_CONFIG = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "class_weight": "balanced",
    "fit_intercept": True,
    "tol": 1e-6,
    "max_iter": 5000,
    "random_state": 0,
}

ALGORITHM_IDENTITIES = {
    "nearest_centroid": (
        "train_class_mean_l2_renormalized_cosine_argmax_canonical_tiebreak_v1"
    ),
    "cosine_knn": (
        "cosine_topk_stable_desc_majority_then_simsum_then_canonical_index_v1"
    ),
    "validation_df_margin": (
        "nearest_train_non_df_minus_df_cosine_distance_margin_v1"
    ),
    "logistic_regression_probe": (
        "sklearn_lbfgs_l2_balanced_multinomial_fixed_v1"
    ),
}

INTERPRETATION_SCOPE = (
    "descriptive_representation_diagnostic_not_model_or_candidate_selection"
)

NPZ_KEYS = (
    "train_embeddings",
    "train_labels",
    "train_image_ids",
    "train_lesion_ids",
    "validation_embeddings",
    "validation_labels",
    "validation_image_ids",
    "validation_lesion_ids",
)

RECORD_INTEGRITY_SCHEMA = "all_class_separability_record_integrity_v1"
RECORD_INTEGRITY_ENTRIES = ("diagnostic_record", "completed_record", "latest_record")

# The full immutable diagnostic identity: every field is built once and shared.
IDENTITY_REQUIRED_FIELDS = (
    "diagnostic_version",
    "git_commit",
    "train_manifest_sha256",
    "validation_manifest_sha256",
    "fixed_split_identity",
    "shared_root_uuid",
    "diagnostic_output_identity",
    "model_identity",
    "dependency_versions",
    "algorithm_identities",
    "class_mapping",
    "expected_train_class_counts",
    "expected_validation_class_counts",
    "expected_row_counts",
    "embedding_dimension",
    "fixed_knn_k_values",
    "fixed_logistic_regression_config",
    "evaluation_scope",
    "interpretation_scope",
    "formal_training_started",
    "test_data_accessed",
    "condition_selected",
)
# Every record top-level field that duplicates an identity field, mapped to its
# identity key. All three copies (record top level, nested diagnostic_identity,
# identity file) must agree, so validate checks each of these -- not just a few.
IDENTITY_DUPLICATED_TOP_LEVEL = {
    "diagnostic_version": "diagnostic_version",
    "git_commit": "git_commit",
    "shared_root_uuid": "shared_root_uuid",
    "diagnostic_output_identity": "diagnostic_output_identity",
    "train_manifest_sha256": "train_manifest_sha256",
    "validation_manifest_sha256": "validation_manifest_sha256",
    "fixed_split_identity": "fixed_split_identity",
    "model_identity": "model_identity",
    "dependency_versions": "dependency_versions",
    "algorithm_identities": "algorithm_identities",
    "feature_dimension": "embedding_dimension",
    "fixed_knn_k_values": "fixed_knn_k_values",
    "fixed_logistic_regression_config": "fixed_logistic_regression_config",
    "interpretation_scope": "interpretation_scope",
    "formal_training_started": "formal_training_started",
    "test_data_accessed": "test_data_accessed",
    "condition_selected": "condition_selected",
}


# --- split loading and integrity (train + validation only) -------------------
def _expected_counts(split: str) -> dict[str, int]:
    if split == "train":
        return EXPECTED_TRAIN_CLASS_COUNTS
    return EXPECTED_VALIDATION_CLASS_COUNTS


def _validate_split_frame(split: str, frame) -> None:
    expected = _expected_counts(split)
    expected_total = sum(expected.values())
    if len(frame) != expected_total:
        raise ValueError(
            f"{split} split has {len(frame)} rows; expected {expected_total}"
        )
    if "source" in frame.columns and frame["source"].astype(str).eq(
        manifests.GENERATED_SOURCE
    ).any():
        raise ValueError(
            f"{split} split contains synthetic-source rows; this diagnostic is real-only"
        )
    for field in ("image_id", "lesion_id"):
        column = frame[field]
        if column.isna().any() or column.astype(str).str.strip().eq("").any():
            raise ValueError(f"{split} split contains missing {field} values")
    if frame["image_id"].astype(str).duplicated().any():
        raise ValueError(f"{split} split contains duplicated image_id values")
    if not frame["dx"].isin(config.CLASS_NAMES).all():
        raise ValueError(f"{split} split contains unknown dx labels")
    mapped = frame["dx"].map(config.CLASS_TO_IDX).to_numpy()
    if not np.array_equal(mapped, frame["label_idx"].to_numpy()):
        raise ValueError(f"{split} split label_idx does not match dx")
    counts = manifests.class_counts(frame)
    if counts != expected:
        raise ValueError(f"{split} class counts {counts} != expected {expected}")


def _validate_no_cross_split_overlap(train, validation) -> None:
    for field in ("image_id", "lesion_id"):
        overlap = set(train[field].astype(str)) & set(validation[field].astype(str))
        if overlap:
            raise ValueError(
                f"train/validation share {len(overlap)} {field} value(s); the fixed "
                f"split must not leak, e.g. {sorted(overlap)[:3]}"
            )


def load_all_class_groups() -> dict[str, object]:
    """Load only the fixed train and validation splits (never test)."""
    train = manifests.load_split(GROUP_TO_MANIFEST_SPLIT["train"]).reset_index(drop=True)
    validation = manifests.load_split(
        GROUP_TO_MANIFEST_SPLIT["validation"]
    ).reset_index(drop=True)
    _validate_split_frame("train", train)
    _validate_split_frame("validation", validation)
    _validate_no_cross_split_overlap(train, validation)
    return {"train": train, "validation": validation}


def _frame_labels_ids(frame):
    """Return int64 labels and 1-D unicode (never object) id arrays in frame order."""
    labels = frame["label_idx"].to_numpy(dtype=np.int64)
    image_ids = np.asarray(frame["image_id"].astype(str).tolist(), dtype=np.str_)
    lesion_ids = np.asarray(frame["lesion_id"].astype(str).tolist(), dtype=np.str_)
    return labels, image_ids, lesion_ids


# --- shared numeric guards ---------------------------------------------------
def _validate_labels(labels, count: int, *, name: str) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.shape != (count,):
        raise ValueError(f"{name} labels must have shape ({count},), got {labels.shape}")
    labels = labels.astype(np.int64)
    if labels.size and (labels.min() < 0 or labels.max() >= config.NUM_CLASSES):
        raise ValueError(
            f"{name} labels must lie in [0, {config.NUM_CLASSES}); got "
            f"[{int(labels.min())}, {int(labels.max())}]"
        )
    return labels


def _require_matching_dim(train_norm: np.ndarray, val_norm: np.ndarray) -> None:
    if train_norm.shape[1] != val_norm.shape[1]:
        raise ValueError(
            "train and validation embeddings have different feature dimensions"
        )


def _prediction_counts(predictions) -> dict[str, int]:
    predictions = np.asarray(predictions)
    return {
        config.IDX_TO_CLASS[idx]: int((predictions == idx).sum())
        for idx in range(config.NUM_CLASSES)
    }


def _cosine_similarity_matrix(query_norm: np.ndarray, reference_norm: np.ndarray) -> np.ndarray:
    return np.clip(query_norm @ reference_norm.T, -1.0, 1.0)


def _evaluate_predictions(val_labels, predictions) -> dict[str, object]:
    """Phase C only: score already-computed predictions against validation labels."""
    return {
        "classification_summary": metrics.classification_summary(val_labels, predictions),
        "prediction_counts": _prediction_counts(predictions),
    }


# --- A. seven-class nearest centroid (fit=train, predict, then evaluate) ------
def _class_centroids(train_norm: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
    centroids = np.empty((config.NUM_CLASSES, train_norm.shape[1]), dtype=np.float64)
    for class_idx in range(config.NUM_CLASSES):
        rows = train_norm[train_labels == class_idx]
        if rows.shape[0] == 0:
            raise ValueError(
                f"class index {class_idx} has no train embeddings to form a centroid"
            )
        centroids[class_idx] = l2_normalize(rows.mean(axis=0, keepdims=True))[0]
    return centroids


def _nearest_centroid_predict(centroids: np.ndarray, query_norm: np.ndarray) -> np.ndarray:
    similarities = np.clip(query_norm @ centroids.T, -1.0, 1.0)
    # np.argmax returns the first (lowest) index on an exact tie, i.e. the
    # canonical class index wins -- the fixed tie-break for this diagnostic.
    return np.argmax(similarities, axis=1).astype(np.int64)


def nearest_centroid_analysis(
    train_embeddings, train_labels, val_embeddings, val_labels
) -> dict[str, object]:
    train_norm = l2_normalize(train_embeddings)
    val_norm = l2_normalize(val_embeddings)
    _require_matching_dim(train_norm, val_norm)
    train_labels = _validate_labels(train_labels, train_norm.shape[0], name="train")
    centroids = _class_centroids(train_norm, train_labels)              # fit (train only)
    predictions = _nearest_centroid_predict(centroids, val_norm)        # predict (no labels)
    val_labels = _validate_labels(val_labels, val_norm.shape[0], name="validation")
    return {
        "algorithm": ALGORITHM_IDENTITIES["nearest_centroid"],
        **_evaluate_predictions(val_labels, predictions),
    }


# --- B. cosine k-NN at fixed k = (1, 5, 10) ----------------------------------
def _knn_vote(neighbor_labels, neighbor_similarities) -> int:
    votes = np.zeros(config.NUM_CLASSES, dtype=np.float64)
    similarity_sum = np.zeros(config.NUM_CLASSES, dtype=np.float64)
    for label, similarity in zip(neighbor_labels, neighbor_similarities):
        votes[int(label)] += 1.0
        similarity_sum[int(label)] += float(similarity)
    best_class = 0
    best_key = (votes[0], similarity_sum[0])
    for class_idx in range(1, config.NUM_CLASSES):
        key = (votes[class_idx], similarity_sum[class_idx])
        # strictly greater keeps the lower class index on a full tie: the fixed
        # majority -> summed-similarity -> canonical-index rule.
        if key > best_key:
            best_key = key
            best_class = class_idx
    return best_class


def _neighbor_order(similarities: np.ndarray) -> np.ndarray:
    # descending similarity; the stable sort breaks equal-similarity ties by
    # ascending reference index for a deterministic neighbour order.
    return np.argsort(-similarities, axis=1, kind="stable")


def _knn_predict(train_norm, train_labels, val_norm, k_values) -> dict[int, np.ndarray]:
    """Predict-only: cosine k-NN votes from validation embeddings (no val labels)."""
    if max(k_values) > train_norm.shape[0]:
        raise ValueError(
            f"k={max(k_values)} exceeds the {train_norm.shape[0]} reference embeddings"
        )
    similarities = _cosine_similarity_matrix(val_norm, train_norm)
    order = _neighbor_order(similarities)
    predictions: dict[int, np.ndarray] = {}
    for k in k_values:
        top_k = order[:, :k]
        pred = np.empty(val_norm.shape[0], dtype=np.int64)
        for row in range(val_norm.shape[0]):
            neighbor_idx = top_k[row]
            pred[row] = _knn_vote(
                train_labels[neighbor_idx], similarities[row, neighbor_idx]
            )
        predictions[int(k)] = pred
    return predictions


def cosine_knn_analysis(
    train_embeddings, train_labels, val_embeddings, val_labels, k_values=KNN_K_VALUES
) -> dict[str, object]:
    train_norm = l2_normalize(train_embeddings)
    val_norm = l2_normalize(val_embeddings)
    _require_matching_dim(train_norm, val_norm)
    train_labels = _validate_labels(train_labels, train_norm.shape[0], name="train")
    predictions = _knn_predict(train_norm, train_labels, val_norm, k_values)   # predict
    val_labels = _validate_labels(val_labels, val_norm.shape[0], name="validation")
    return {
        "algorithm": ALGORITHM_IDENTITIES["cosine_knn"],
        "k_values": [int(k) for k in k_values],
        "by_k": {
            f"k{k}": {"k": int(k), **_evaluate_predictions(val_labels, predictions[int(k)])}
            for k in k_values
        },
    }


# --- C. validation-df representation margin ----------------------------------
def validation_df_margin_analysis(
    train_embeddings, train_labels, val_embeddings, val_labels, k_values=KNN_K_VALUES
) -> dict[str, object]:
    train_norm = l2_normalize(train_embeddings)
    val_norm = l2_normalize(val_embeddings)
    _require_matching_dim(train_norm, val_norm)
    train_labels = _validate_labels(train_labels, train_norm.shape[0], name="train")
    val_labels = _validate_labels(val_labels, val_norm.shape[0], name="validation")
    df_idx = config.TARGET_CLASS_IDX
    train_df_mask = train_labels == df_idx
    if not train_df_mask.any() or train_df_mask.all():
        raise ValueError("train split needs both df and non-df rows for the margin")
    val_df_rows = np.where(val_labels == df_idx)[0]
    if val_df_rows.size == 0:
        raise ValueError("validation split has no df rows for the representation margin")

    val_df_norm = val_norm[val_df_rows]
    df_similarities = _cosine_similarity_matrix(val_df_norm, train_norm[train_df_mask])
    non_df_similarities = _cosine_similarity_matrix(val_df_norm, train_norm[~train_df_mask])
    nearest_df_distance = 1.0 - df_similarities.max(axis=1)
    nearest_non_df_distance = 1.0 - non_df_similarities.max(axis=1)
    margin = nearest_non_df_distance - nearest_df_distance
    positive = margin > 0.0

    full_similarities = _cosine_similarity_matrix(val_df_norm, train_norm)
    order = _neighbor_order(full_similarities)
    purity_by_k: dict[str, object] = {}
    for k in k_values:
        neighbor_labels = train_labels[order[:, :k]]
        df_fraction = (neighbor_labels == df_idx).mean(axis=1)
        purity_by_k[f"k{k}"] = {
            "k": int(k),
            "df_neighbor_fraction_summary": _distance_summary(df_fraction),
            "queries_with_df_majority": int((df_fraction > 0.5).sum()),
            "queries_with_df_majority_fraction": float((df_fraction > 0.5).mean()),
        }
    return {
        "algorithm": ALGORITHM_IDENTITIES["validation_df_margin"],
        "validation_df_count": int(val_df_rows.size),
        "margin_summary": _distance_summary(margin),
        "positive_margin_count": int(positive.sum()),
        "positive_margin_fraction": float(positive.mean()),
        "nearest_train_df_cosine_distance_summary": _distance_summary(nearest_df_distance),
        "nearest_train_non_df_cosine_distance_summary": _distance_summary(nearest_non_df_distance),
        "per_query_margin": [float(value) for value in margin],
        "df_neighbor_purity_by_k": purity_by_k,
    }


# --- D. one fixed balanced multinomial logistic-regression probe -------------
def fit_logistic_probe(train_embeddings, train_labels):
    """Fit the single fixed logistic probe on train labels only; fail loud on
    non-convergence. Validation is never seen here."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    train_norm = l2_normalize(train_embeddings)
    train_labels = _validate_labels(train_labels, train_norm.shape[0], name="train")
    model = LogisticRegression(**LOGISTIC_REGRESSION_CONFIG)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(train_norm, train_labels)
    convergence_warnings = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    n_iter = [int(value) for value in np.atleast_1d(model.n_iter_)]
    converged = not convergence_warnings and all(
        value < LOGISTIC_REGRESSION_CONFIG["max_iter"] for value in n_iter
    )
    if not converged:
        raise RuntimeError(
            f"fixed logistic probe did not converge (n_iter={n_iter}); "
            f"warnings={convergence_warnings}"
        )
    return model, {
        "n_iter": n_iter,
        "classes": [int(value) for value in model.classes_],
        "converged": True,
    }


def _logistic_predict(model, val_norm) -> np.ndarray:
    return model.predict(val_norm).astype(np.int64)


def logistic_probe_analysis(
    train_embeddings, train_labels, val_embeddings, val_labels
) -> dict[str, object]:
    train_arr = np.asarray(train_embeddings)
    val_norm = l2_normalize(val_embeddings)
    if train_arr.ndim != 2 or train_arr.shape[1] != val_norm.shape[1]:
        raise ValueError(
            "train and validation embeddings have different feature dimensions"
        )
    model, fit_info = fit_logistic_probe(train_arr, train_labels)     # fit (train only)
    predictions = _logistic_predict(model, val_norm)                  # predict (no labels)
    val_labels = _validate_labels(val_labels, val_norm.shape[0], name="validation")
    return {
        "algorithm": ALGORITHM_IDENTITIES["logistic_regression_probe"],
        "config": dict(LOGISTIC_REGRESSION_CONFIG),
        "n_iter": fit_info["n_iter"],
        "classes": fit_info["classes"],
        "converged": fit_info["converged"],
        **_evaluate_predictions(val_labels, predictions),
    }


def run_all_analyses(
    train_embeddings, train_labels, val_embeddings, val_labels, k_values=KNN_K_VALUES
) -> dict[str, object]:
    """Production pipeline with a strict fit -> predict -> evaluate ordering.

    Phase A fits everything on train only (the logistic probe first, so it is the
    first fit event and completes before any validation label is read). Phase B
    produces every prediction from validation *embeddings* only. Phase C is the
    single point where validation *labels* are consumed for metrics.
    """
    train_norm = l2_normalize(train_embeddings)
    val_norm = l2_normalize(val_embeddings)
    _require_matching_dim(train_norm, val_norm)
    train_labels_checked = _validate_labels(train_labels, train_norm.shape[0], name="train")
    if max(k_values) > train_norm.shape[0]:
        raise ValueError(
            f"k={max(k_values)} exceeds the {train_norm.shape[0]} reference embeddings"
        )

    # Phase A -- train-only fitting/preparation (validation labels never touched)
    logistic_model, logistic_fit_info = fit_logistic_probe(train_embeddings, train_labels_checked)
    centroids = _class_centroids(train_norm, train_labels_checked)
    # the k-NN "reference" is the train embeddings/labels themselves; no fitting.

    # Phase B -- predictions from validation embeddings only (no validation labels)
    centroid_pred = _nearest_centroid_predict(centroids, val_norm)
    knn_pred = _knn_predict(train_norm, train_labels_checked, val_norm, k_values)
    logistic_pred = _logistic_predict(logistic_model, val_norm)

    # Phase C -- one-shot evaluation; validation labels are consumed only here
    val_labels_checked = _validate_labels(val_labels, val_norm.shape[0], name="validation")
    nearest_centroid = {
        "algorithm": ALGORITHM_IDENTITIES["nearest_centroid"],
        **_evaluate_predictions(val_labels_checked, centroid_pred),
    }
    cosine_knn = {
        "algorithm": ALGORITHM_IDENTITIES["cosine_knn"],
        "k_values": [int(k) for k in k_values],
        "by_k": {
            f"k{k}": {"k": int(k), **_evaluate_predictions(val_labels_checked, knn_pred[int(k)])}
            for k in k_values
        },
    }
    logistic = {
        "algorithm": ALGORITHM_IDENTITIES["logistic_regression_probe"],
        "config": dict(LOGISTIC_REGRESSION_CONFIG),
        "n_iter": logistic_fit_info["n_iter"],
        "classes": logistic_fit_info["classes"],
        "converged": logistic_fit_info["converged"],
        **_evaluate_predictions(val_labels_checked, logistic_pred),
    }
    margin = validation_df_margin_analysis(
        train_embeddings, train_labels_checked, val_embeddings, val_labels_checked, k_values=k_values
    )
    return {
        "nearest_centroid": nearest_centroid,
        "cosine_knn": cosine_knn,
        "validation_df_margin": margin,
        "logistic_regression_probe": logistic,
    }


# --- immutable diagnostic identity (built once, shared by every artifact) -----
def build_identity(
    *,
    git_commit: str,
    train_manifest_sha256: str,
    validation_manifest_sha256: str,
    shared_root_uuid: str,
    diagnostic_output_identity: str,
    model_identity,
    dependency_versions,
    algorithm_identities,
) -> dict[str, object]:
    """Assemble the single immutable identity shared by record, sidecar, and file."""
    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "git_commit": git_commit,
        "train_manifest_sha256": train_manifest_sha256,
        "validation_manifest_sha256": validation_manifest_sha256,
        "fixed_split_identity": train_manifest_sha256,
        "shared_root_uuid": shared_root_uuid,
        "diagnostic_output_identity": diagnostic_output_identity,
        "model_identity": model_identity,
        "dependency_versions": dependency_versions,
        "algorithm_identities": algorithm_identities,
        "class_mapping": dict(config.CLASS_TO_IDX),
        "expected_train_class_counts": dict(EXPECTED_TRAIN_CLASS_COUNTS),
        "expected_validation_class_counts": dict(EXPECTED_VALIDATION_CLASS_COUNTS),
        "expected_row_counts": {
            "train": int(EXPECTED_TRAIN_ROWS),
            "validation": int(EXPECTED_VALIDATION_ROWS),
        },
        "embedding_dimension": FEATURE_DIM,
        "fixed_knn_k_values": [int(k) for k in KNN_K_VALUES],
        "fixed_logistic_regression_config": dict(LOGISTIC_REGRESSION_CONFIG),
        "evaluation_scope": "train_and_validation_representation_only",
        "interpretation_scope": INTERPRETATION_SCOPE,
        "formal_training_started": False,
        "test_data_accessed": False,
        "condition_selected": False,
    }


def _is_empty_identity_value(value) -> bool:
    return value is None or (isinstance(value, (str, dict, list, tuple)) and len(value) == 0)


def _require_complete_identity(identity, *, where: str) -> None:
    if not isinstance(identity, dict):
        raise ValueError(f"{where} identity must be a mapping")
    for field in IDENTITY_REQUIRED_FIELDS:
        if field not in identity:
            raise ValueError(f"{where} identity is missing field: {field}")
        if _is_empty_identity_value(identity[field]):
            raise ValueError(f"{where} identity field is null/empty: {field}")


def validate_diagnostic_identity(record, identity, expected_identity) -> bool:
    """Fail loud unless the record, the identity file, and the runtime-expected
    identity all agree on a complete immutable identity.

    Rejects missing/empty fields, a record whose ``diagnostic_identity`` differs
    from the identity file, a top-level duplicate that drifts from the nested
    identity, and an identity that is internally consistent but disagrees with
    the runtime-expected identity (the all-consistently-wrong case).
    """
    _require_complete_identity(identity, where="diagnostic")
    record_identity = record.get("diagnostic_identity")
    if record_identity is None:
        raise ValueError("record is missing diagnostic_identity")
    if record_identity != identity:
        raise ValueError("record diagnostic_identity does not match the identity file")
    for record_key, identity_key in IDENTITY_DUPLICATED_TOP_LEVEL.items():
        if record_key not in record:
            raise ValueError(
                f"record is missing top-level duplicate: {record_key}"
            )
        if record[record_key] != identity[identity_key]:
            raise ValueError(
                f"record top-level {record_key} disagrees with diagnostic_identity"
            )
    _require_complete_identity(expected_identity, where="expected")
    mismatches = [
        field
        for field in IDENTITY_REQUIRED_FIELDS
        if identity[field] != expected_identity[field]
    ]
    if mismatches:
        raise ValueError(
            f"diagnostic identity disagrees with expected runtime identity: {mismatches}"
        )
    return True


# --- record assembly ---------------------------------------------------------
def build_record(
    *,
    identity,
    row_counts,
    class_counts,
    embedding_artifact,
    analyses,
) -> dict[str, object]:
    """Assemble the record from the single shared identity (no drifting sources)."""
    _require_complete_identity(identity, where="record-input")
    return {
        "diagnostic_status": "COMPLETED",
        "diagnostic_version": identity["diagnostic_version"],
        "git_commit": identity["git_commit"],
        "shared_root_uuid": identity["shared_root_uuid"],
        "diagnostic_output_identity": identity["diagnostic_output_identity"],
        "train_manifest_sha256": identity["train_manifest_sha256"],
        "validation_manifest_sha256": identity["validation_manifest_sha256"],
        "fixed_split_identity": identity["fixed_split_identity"],
        "row_counts": row_counts,
        "class_counts": class_counts,
        "feature_dimension": identity["embedding_dimension"],
        "model_identity": identity["model_identity"],
        "embedding_extractor": {
            "feature_source": "frozen_image_encoder_output_before_linear_head",
            "linear_head_used": False,
            "native_eval_preprocessing": True,
            "input_resolution": list(INPUT_RESOLUTION),
            "train_time_augmentation": False,
            "l2_normalized": True,
        },
        "dependency_versions": identity["dependency_versions"],
        "algorithm_identities": identity["algorithm_identities"],
        "fixed_knn_k_values": identity["fixed_knn_k_values"],
        "fixed_logistic_regression_config": identity["fixed_logistic_regression_config"],
        "embedding_artifact": embedding_artifact,
        "analyses": analyses,
        "diagnostic_identity": identity,
        "formal_training_started": False,
        "test_data_accessed": False,
        "condition_selected": False,
        "interpretation_scope": identity["interpretation_scope"],
    }


# --- eight-key embedding NPZ dtype/shape/value contract ----------------------
def _npz_expectations(n_train: int, n_val: int) -> dict[str, dict]:
    return {
        "train_embeddings": {"dtype": np.float32, "shape": (n_train, FEATURE_DIM), "kind": "embedding"},
        "validation_embeddings": {"dtype": np.float32, "shape": (n_val, FEATURE_DIM), "kind": "embedding"},
        "train_labels": {"dtype": np.int64, "shape": (n_train,), "kind": "label"},
        "validation_labels": {"dtype": np.int64, "shape": (n_val,), "kind": "label"},
        "train_image_ids": {"dtype": "U", "shape": (n_train,), "kind": "id_unique"},
        "validation_image_ids": {"dtype": "U", "shape": (n_val,), "kind": "id_unique"},
        "train_lesion_ids": {"dtype": "U", "shape": (n_train,), "kind": "id"},
        "validation_lesion_ids": {"dtype": "U", "shape": (n_val,), "kind": "id"},
    }


def _check_dtype_shape(name: str, array: np.ndarray, spec: dict) -> None:
    if array.dtype == object or array.dtype.kind == "O":
        raise ValueError(f"{name} must not be object dtype")
    if spec["dtype"] == "U":
        if array.dtype.kind != "U":
            raise ValueError(f"{name} must be a unicode string array, got {array.dtype}")
    elif array.dtype != np.dtype(spec["dtype"]):
        raise ValueError(f"{name} must be {np.dtype(spec['dtype'])}, got {array.dtype}")
    if tuple(array.shape) != spec["shape"]:
        raise ValueError(f"{name} must have shape {spec['shape']}, got {tuple(array.shape)}")


def _check_embedding_values(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    norms = np.linalg.norm(array.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError(f"{name} rows are not unit norm")


def _validate_npz_arrays(arrays, *, n_train: int, n_val: int) -> None:
    """Pre-write schema validation of the exact eight-key embedding bundle."""
    if set(arrays) != set(NPZ_KEYS):
        raise ValueError(f"NPZ keys must be exactly {sorted(NPZ_KEYS)}; got {sorted(arrays)}")
    for name, spec in _npz_expectations(n_train, n_val).items():
        array = arrays[name]
        if not isinstance(array, np.ndarray):
            raise ValueError(f"{name} must be a numpy array")
        _check_dtype_shape(name, array, spec)
        if spec["kind"] == "embedding":
            _check_embedding_values(name, array)
        elif spec["kind"] == "label":
            if array.min() < 0 or array.max() >= config.NUM_CLASSES:
                raise ValueError(f"{name} labels out of range [0, {config.NUM_CLASSES})")
        else:
            if (np.char.strip(array) == "").any():
                raise ValueError(f"{name} contains empty/whitespace ids")
            if spec["kind"] == "id_unique" and np.unique(array).shape[0] != array.shape[0]:
                raise ValueError(f"{name} contains duplicate ids")
    for train_key, val_key in (
        ("train_image_ids", "validation_image_ids"),
        ("train_lesion_ids", "validation_lesion_ids"),
    ):
        overlap = set(arrays[train_key].tolist()) & set(arrays[val_key].tolist())
        if overlap:
            raise ValueError(
                f"{train_key}/{val_key} overlap across splits, e.g. {sorted(overlap)[:3]}"
            )


def _validate_npz_file(path, *, n_train: int, n_val: int) -> None:
    """Post-write validation: reopen with allow_pickle=False and re-check dtype/shape/values."""
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(NPZ_KEYS):
            raise ValueError(
                f"reopened NPZ keys {sorted(saved.files)} != {sorted(NPZ_KEYS)}"
            )
        for name, spec in _npz_expectations(n_train, n_val).items():
            array = saved[name]
            _check_dtype_shape(name, array, spec)
            if spec["kind"] == "embedding":
                _check_embedding_values(name, array)


def write_all_class_embeddings_npz(path, arrays, *, n_train: int, n_val: int):
    """Validate the schema, atomically write, then reopen and re-validate dtype/values."""
    path = Path(path)
    _validate_npz_arrays(arrays, n_train=n_train, n_val=n_val)
    _write_npz_atomic(path, arrays)
    _validate_npz_file(path, n_train=n_train, n_val=n_val)
    return path


# --- byte-identical formal records + non-circular integrity sidecar ----------
def _formal_record_paths(output_dir: Path, latest_record_path: Path):
    return {
        "diagnostic_record": output_dir / "all_class_separability_diagnostic.json",
        "completed_record": output_dir / "_COMPLETED.json",
        "latest_record": latest_record_path,
    }


def _build_record_integrity(paths: dict[str, Path]) -> dict[str, object]:
    reference = paths["diagnostic_record"].read_bytes()
    entries: dict[str, object] = {"schema": RECORD_INTEGRITY_SCHEMA, "algorithm": "sha256"}
    hashes = set()
    for key in RECORD_INTEGRITY_ENTRIES:
        data = paths[key].read_bytes()
        if data != reference:
            raise ValueError(f"formal record raw bytes diverged: {paths[key]}")
        digest = hashlib.sha256(data).hexdigest()
        entries[key] = {"filename": paths[key].name, "bytes": len(data), "sha256": digest}
        hashes.add(digest)
    if len(hashes) != 1:
        raise ValueError("formal records have diverging SHA-256")
    entries["raw_bytes_equal"] = True
    return entries


def validate_record_integrity(*, output_dir, latest_record_path) -> dict[str, object]:
    """Reopen the three formal records and the sidecar, recompute, and cross-check."""
    output_dir = Path(output_dir)
    latest_record_path = Path(latest_record_path)
    paths = _formal_record_paths(output_dir, latest_record_path)
    integrity_path = output_dir / "record_integrity.json"
    for path in list(paths.values()) + [integrity_path]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing record-integrity artifact: {path}")
    sidecar = json.loads(integrity_path.read_text(encoding="utf-8"))
    if sidecar.get("schema") != RECORD_INTEGRITY_SCHEMA or sidecar.get("algorithm") != "sha256":
        raise ValueError("record_integrity.json has an unexpected schema/algorithm")
    reference = paths["diagnostic_record"].read_bytes()
    hashes = set()
    for key in RECORD_INTEGRITY_ENTRIES:
        data = paths[key].read_bytes()
        if data != reference:
            raise ValueError(f"formal record raw bytes diverged: {paths[key]}")
        digest = hashlib.sha256(data).hexdigest()
        entry = sidecar.get(key)
        if (
            not isinstance(entry, dict)
            or entry.get("filename") != paths[key].name
            or entry.get("bytes") != len(data)
            or entry.get("sha256") != digest
        ):
            raise ValueError(f"record_integrity.json mismatch for {key}")
        hashes.add(digest)
    if len(hashes) != 1 or sidecar.get("raw_bytes_equal") is not True:
        raise ValueError("record_integrity.json does not certify identical records")
    return sidecar


def write_formal_records(*, output_dir, latest_record_path, record, identity) -> dict[str, str]:
    """Write the identity, three byte-identical formal records, and a non-circular
    SHA-256 integrity sidecar; refuse to overwrite an existing completed latest record."""
    output_dir = Path(output_dir)
    latest_record_path = Path(latest_record_path)
    if latest_record_path.exists():
        raise FileExistsError(
            f"completed latest diagnostic record exists; refusing to overwrite: "
            f"{latest_record_path}"
        )
    coca_run.write_json_atomic(output_dir / "diagnostic_identity.json", identity)
    paths = _formal_record_paths(output_dir, latest_record_path)
    for path in paths.values():
        coca_run.write_json_atomic(path, record)
    reference = paths["diagnostic_record"].read_bytes()
    for path in paths.values():
        if path.read_bytes() != reference:
            raise ValueError(f"formal record raw bytes diverged: {path}")
    integrity = _build_record_integrity(paths)
    integrity_path = output_dir / "record_integrity.json"
    coca_run.write_json_atomic(integrity_path, integrity)
    validate_record_integrity(output_dir=output_dir, latest_record_path=latest_record_path)
    return {
        "diagnostic_record": str(paths["diagnostic_record"]),
        "completed_marker": str(paths["completed_record"]),
        "latest_record": str(latest_record_path),
        "record_integrity": str(integrity_path),
    }


def _require_shapes(embeddings: np.ndarray, expected_rows: int) -> None:
    if embeddings.shape != (expected_rows, FEATURE_DIM):
        raise ValueError(
            f"embeddings shape {embeddings.shape} != ({expected_rows}, {FEATURE_DIM})"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain non-finite values")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Frozen-CoCa all-class separability diagnostic")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--shared-root-uuid", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    import torch
    from importlib.metadata import version

    from .model import build_model, model_identity

    args = parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    output_dir = prepare_output_dir(args.output_dir)
    latest_record_path = output_dir.parent / "latest_diagnostic_record.json"
    if latest_record_path.exists():
        raise FileExistsError(
            f"completed latest diagnostic record exists; refusing to overwrite: "
            f"{latest_record_path}"
        )

    groups = load_all_class_groups()
    train_labels, train_image_ids, train_lesion_ids = _frame_labels_ids(groups["train"])
    val_labels, val_image_ids, val_lesion_ids = _frame_labels_ids(groups["validation"])

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real CoCa diagnostic")
    model = build_model(
        arch="coca_vit_b32",
        freeze_backbone=True,
        coca_pretrained="laion2b_s13b_b90k",
    ).to(device)
    model.eval()
    if model.encoder.training or any(
        parameter.requires_grad for parameter in model.encoder.parameters()
    ):
        raise ValueError("CoCa encoder is not frozen in eval mode")

    print(f"START encode train rows={EXPECTED_TRAIN_ROWS}", flush=True)
    train_embeddings = encode_group(
        model,
        groups["train"],
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"DONE encode train shape={train_embeddings.shape}", flush=True)
    print(f"START encode validation rows={EXPECTED_VALIDATION_ROWS}", flush=True)
    val_embeddings = encode_group(
        model,
        groups["validation"],
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"DONE encode validation shape={val_embeddings.shape}", flush=True)

    _require_shapes(train_embeddings, EXPECTED_TRAIN_ROWS)
    _require_shapes(val_embeddings, EXPECTED_VALIDATION_ROWS)

    analyses = run_all_analyses(train_embeddings, train_labels, val_embeddings, val_labels)

    arrays = {
        "train_embeddings": train_embeddings.astype(np.float32),
        "train_labels": train_labels,
        "train_image_ids": train_image_ids,
        "train_lesion_ids": train_lesion_ids,
        "validation_embeddings": val_embeddings.astype(np.float32),
        "validation_labels": val_labels,
        "validation_image_ids": val_image_ids,
        "validation_lesion_ids": val_lesion_ids,
    }
    embedding_path = output_dir / "all_class_embeddings.npz"
    write_all_class_embeddings_npz(
        embedding_path, arrays, n_train=EXPECTED_TRAIN_ROWS, n_val=EXPECTED_VALIDATION_ROWS
    )

    diagnostic_output_identity = (
        f"{args.shared_root_uuid}:sqrt_balanced_seed0_v1:coca_classifier:"
        f"representation_diagnostics:{DIAGNOSTIC_VERSION}:{output_dir.name}"
    )
    identity = build_identity(
        git_commit=args.git_commit,
        train_manifest_sha256=sha256_file(config.MANIFESTS_DIR / "train.csv"),
        validation_manifest_sha256=sha256_file(config.MANIFESTS_DIR / "val.csv"),
        shared_root_uuid=args.shared_root_uuid,
        diagnostic_output_identity=diagnostic_output_identity,
        model_identity=model_identity(model, "coca_vit_b32", INPUT_RESOLUTION[0]),
        dependency_versions={
            "open_clip_torch": version("open_clip_torch"),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scikit_learn": version("scikit-learn"),
        },
        algorithm_identities=dict(ALGORITHM_IDENTITIES),
    )
    row_counts = dict(identity["expected_row_counts"])
    class_counts = {
        "train": manifests.class_counts(groups["train"]),
        "validation": manifests.class_counts(groups["validation"]),
    }
    embedding_artifact = {
        "path": str(embedding_path),
        "size_bytes": int(embedding_path.stat().st_size),
        "sha256": sha256_file(embedding_path),
        "npz_keys": sorted(NPZ_KEYS),
    }
    record = build_record(
        identity=identity,
        row_counts=row_counts,
        class_counts=class_counts,
        embedding_artifact=embedding_artifact,
        analyses=analyses,
    )
    # self-check: the record must carry a complete, non-drifting identity.
    validate_diagnostic_identity(record, identity, identity)
    written = write_formal_records(
        output_dir=output_dir,
        latest_record_path=latest_record_path,
        record=record,
        identity=identity,
    )
    print(written["diagnostic_record"])
    print(written["record_integrity"])
    print("ALL-CLASS COCA SEPARABILITY DIAGNOSTIC COMPLETED")
    print("formal_training_started=false")
    print("test_data_accessed=false")
    print("condition_selected=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
