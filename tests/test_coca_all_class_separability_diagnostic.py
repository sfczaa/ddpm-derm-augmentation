"""Local tests for the frozen-CoCa all-class separability diagnostic.

These exercise the pure-numpy / sklearn analyses, the fit -> predict -> evaluate
phase ordering, the shared immutable identity, the record-integrity sidecar, and
the eight-key NPZ dtype/value contract without torch or real CoCa weights. The
count tests read the fixed local manifests (train 6995 / validation 1510); the
test split is never loaded.
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm import coca_all_class_separability_diagnostic as diagnostic  # noqa: E402
from ddpm_derm import config, manifests  # noqa: E402

MODULE_SOURCE = Path(diagnostic.__file__).read_text(encoding="utf-8")


def separable_embeddings(counts, *, dim=32, seed=0, spread=0.03):
    """Seven well-separated class blobs so the fixed analyses recover signal."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(config.NUM_CLASSES, dim))
    blocks, labels = [], []
    for class_idx in range(config.NUM_CLASSES):
        n = counts[class_idx]
        blocks.append(centers[class_idx] + spread * rng.normal(size=(n, dim)))
        labels += [class_idx] * n
    return np.concatenate(blocks), np.asarray(labels, dtype=np.int64)


def real_frame(split: str) -> pd.DataFrame:
    return manifests.load_split(split)


def _unit_embeddings(n, *, dim=diagnostic.FEATURE_DIM, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, dim))
    return (values / np.linalg.norm(values, axis=1, keepdims=True)).astype(np.float32)


def _ids(n, prefix):
    return np.asarray([f"{prefix}{i}" for i in range(n)], dtype=np.str_)


def _valid_npz_arrays(n_train=6, n_val=4):
    return {
        "train_embeddings": _unit_embeddings(n_train, seed=1),
        "train_labels": np.arange(n_train, dtype=np.int64) % config.NUM_CLASSES,
        "train_image_ids": _ids(n_train, "tr-img-"),
        "train_lesion_ids": _ids(n_train, "tr-les-"),
        "validation_embeddings": _unit_embeddings(n_val, seed=2),
        "validation_labels": np.array([3, 3, 5, 1][:n_val], dtype=np.int64),
        "validation_image_ids": _ids(n_val, "va-img-"),
        "validation_lesion_ids": _ids(n_val, "va-les-"),
    }


def _fake_identity(**overrides):
    identity = diagnostic.build_identity(
        git_commit="a" * 40,
        train_manifest_sha256="t" * 64,
        validation_manifest_sha256="v" * 64,
        shared_root_uuid="root-uuid",
        diagnostic_output_identity="root-uuid:...:attempt",
        model_identity={
            "arch": "coca_vit_b32",
            "model_name": "coca_ViT-B-32",
            "pretrained_tag": "laion2b_s13b_b90k",
            "freeze_mode": "frozen_image_encoder_linear_head",
            "input_resolution": [224, 224],
        },
        dependency_versions={
            "open_clip_torch": "3.3.0",
            "torch": "2.0",
            "numpy": "2.1",
            "scikit_learn": "1.8.0",
        },
        algorithm_identities=dict(diagnostic.ALGORITHM_IDENTITIES),
    )
    identity.update(overrides)
    return identity


def _distinct_value(value):
    """Return a value guaranteed to differ from ``value`` (for drift tests)."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 100000
    if isinstance(value, str):
        return value + "-TAMPERED"
    if isinstance(value, list):
        return list(value) + ["TAMPERED"]
    if isinstance(value, dict):
        return {**value, "__tampered__": True}
    return "TAMPERED-SENTINEL"


# --- 1. train/val only; test access is never requested -----------------------
class SplitIsolationTests(unittest.TestCase):
    def test_load_groups_requests_only_train_and_val(self):
        calls = []
        original = manifests.load_split

        def spy(split):
            calls.append(split)
            return original(split)

        with mock.patch.object(diagnostic.manifests, "load_split", side_effect=spy):
            groups = diagnostic.load_all_class_groups()
        self.assertEqual(calls, ["train", "val"])
        self.assertNotIn("test", calls)
        self.assertEqual(set(groups), {"train", "validation"})

    def test_any_test_split_access_would_fail(self):
        original = manifests.load_split

        def guard(split):
            if split == "test":
                raise AssertionError("the diagnostic must never load the test split")
            return original(split)

        with mock.patch.object(diagnostic.manifests, "load_split", side_effect=guard):
            groups = diagnostic.load_all_class_groups()  # must not raise
        self.assertEqual(len(groups["train"]), diagnostic.EXPECTED_TRAIN_ROWS)


# --- 2. exact fixed counts, seven-class counts -------------------------------
class FixedCountTests(unittest.TestCase):
    def test_exact_split_and_class_counts(self):
        groups = diagnostic.load_all_class_groups()
        self.assertEqual(len(groups["train"]), 6995)
        self.assertEqual(len(groups["validation"]), 1510)
        self.assertEqual(
            manifests.class_counts(groups["train"]),
            diagnostic.EXPECTED_TRAIN_CLASS_COUNTS,
        )
        self.assertEqual(
            manifests.class_counts(groups["validation"]),
            diagnostic.EXPECTED_VALIDATION_CLASS_COUNTS,
        )
        self.assertEqual(diagnostic.EXPECTED_TRAIN_ROWS, 6995)
        self.assertEqual(diagnostic.EXPECTED_VALIDATION_ROWS, 1510)

    def test_frame_labels_ids_are_int64_and_unicode(self):
        groups = diagnostic.load_all_class_groups()
        labels, image_ids, lesion_ids = diagnostic._frame_labels_ids(groups["validation"])
        self.assertEqual(labels.dtype, np.int64)
        self.assertEqual(image_ids.dtype.kind, "U")
        self.assertEqual(lesion_ids.dtype.kind, "U")
        self.assertNotEqual(image_ids.dtype, object)


# --- 3. reject synthetic rows, duplicated/missing IDs, cross-split overlap ----
class IntegrityGuardTests(unittest.TestCase):
    def test_rejects_synthetic_source_rows(self):
        frame = real_frame("val").copy()
        frame["source"] = "real"
        frame.loc[frame.index[0], "source"] = manifests.GENERATED_SOURCE
        with self.assertRaisesRegex(ValueError, "synthetic-source"):
            diagnostic._validate_split_frame("validation", frame)

    def test_rejects_duplicated_image_id(self):
        frame = real_frame("val").copy()
        frame.loc[frame.index[1], "image_id"] = frame.loc[frame.index[0], "image_id"]
        with self.assertRaisesRegex(ValueError, "duplicated image_id"):
            diagnostic._validate_split_frame("validation", frame)

    def test_rejects_missing_image_id(self):
        frame = real_frame("val").copy()
        frame.loc[frame.index[0], "image_id"] = ""
        with self.assertRaisesRegex(ValueError, "missing image_id"):
            diagnostic._validate_split_frame("validation", frame)

    def test_rejects_wrong_class_counts(self):
        frame = real_frame("val").iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "rows; expected"):
            diagnostic._validate_split_frame("validation", frame)

    def test_rejects_cross_split_image_id_and_lesion_id_overlap(self):
        train = real_frame("train")
        for field in ("image_id", "lesion_id"):
            leaked = real_frame("val").copy()
            leaked.loc[leaked.index[0], field] = train.iloc[0][field]
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    diagnostic._validate_no_cross_split_overlap(train, leaked)


# --- 4/5. nearest-centroid math and scaling invariance -----------------------
class NearestCentroidTests(unittest.TestCase):
    def test_centroid_is_mean_of_normalized_rows_renormalized(self):
        train_norm = np.zeros((config.NUM_CLASSES, 2), dtype=np.float64)
        for idx in range(config.NUM_CLASSES):
            train_norm[idx] = [np.cos(idx), np.sin(idx)]
        train_norm = diagnostic.l2_normalize(train_norm)
        train_norm = np.vstack([train_norm, [[0.0, 1.0]]])
        labels = np.array(list(range(config.NUM_CLASSES)) + [config.TARGET_CLASS_IDX])
        centroids = diagnostic._class_centroids(train_norm, labels)
        df_rows = diagnostic.l2_normalize(train_norm[labels == config.TARGET_CLASS_IDX])
        expected = diagnostic.l2_normalize(df_rows.mean(axis=0, keepdims=True))[0]
        np.testing.assert_allclose(centroids[config.TARGET_CLASS_IDX], expected, atol=1e-12)
        self.assertAlmostEqual(
            float(np.linalg.norm(centroids[config.TARGET_CLASS_IDX])), 1.0, places=12
        )

    def test_prediction_breaks_exact_ties_by_canonical_index(self):
        centroids = np.zeros((config.NUM_CLASSES, 2))
        centroids[0] = [1.0, 0.0]
        centroids[1] = [0.0, 1.0]
        query = np.array([[1.0, 1.0]]) / np.sqrt(2.0)
        self.assertEqual(int(diagnostic._nearest_centroid_predict(centroids, query)[0]), 0)

    def test_analysis_is_invariant_to_positive_row_scaling(self):
        train, train_labels = separable_embeddings([6] * 7, seed=1)
        val, val_labels = separable_embeddings([2, 2, 2, 3, 2, 2, 2], seed=2)
        base = diagnostic.nearest_centroid_analysis(train, train_labels, val, val_labels)
        scale = np.arange(1, len(train) + 1)[:, None] * 3.0
        val_scale = np.arange(1, len(val) + 1)[:, None] * 5.0
        scaled = diagnostic.nearest_centroid_analysis(
            train * scale, train_labels, val * val_scale, val_labels
        )
        self.assertEqual(base["classification_summary"], scaled["classification_summary"])
        self.assertEqual(base["prediction_counts"], scaled["prediction_counts"])


# --- 6/7. k-NN deterministic order and two-layer tie-break -------------------
class CosineKnnTests(unittest.TestCase):
    def test_neighbor_order_is_descending_and_stable(self):
        similarities = np.array([[0.5, 0.9, 0.9, 0.1]])
        order = diagnostic._neighbor_order(similarities)
        self.assertEqual(order[0].tolist(), [1, 2, 0, 3])

    def test_analysis_reports_all_fixed_k_and_is_deterministic(self):
        train, train_labels = separable_embeddings([8] * 7, seed=3)
        val, val_labels = separable_embeddings([3, 3, 3, 4, 3, 3, 3], seed=4)
        first = diagnostic.cosine_knn_analysis(train, train_labels, val, val_labels)
        second = diagnostic.cosine_knn_analysis(train, train_labels, val, val_labels)
        self.assertEqual(first["k_values"], [1, 5, 10])
        self.assertEqual(set(first["by_k"]), {"k1", "k5", "k10"})
        self.assertEqual(first, second)

    def test_vote_tie_breaks_by_similarity_sum_then_canonical_index(self):
        self.assertEqual(diagnostic._knn_vote(np.array([5, 2]), np.array([0.4, 0.4])), 2)
        self.assertEqual(diagnostic._knn_vote(np.array([5, 2]), np.array([0.9, 0.4])), 5)
        self.assertEqual(
            diagnostic._knn_vote(np.array([2, 2, 5]), np.array([0.3, 0.3, 0.99])), 2
        )

    def test_k_exceeding_reference_fails_loud(self):
        train, train_labels = separable_embeddings([1] * 7, seed=5)
        val, val_labels = separable_embeddings([1] * 7, seed=6)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            diagnostic.cosine_knn_analysis(
                train, train_labels, val, val_labels, k_values=(1, 5, 10)
            )


# --- 8/9. validation-df margin direction, summary, and purity ----------------
class ValidationDfMarginTests(unittest.TestCase):
    def test_margin_direction_and_purity_hand_case(self):
        train = np.array([[1.0, 0.0], [0.0, 1.0]])
        train_labels = np.array([config.TARGET_CLASS_IDX, config.CLASS_TO_IDX["nv"]])
        val = np.array([[1.0, 0.0]])
        val_labels = np.array([config.TARGET_CLASS_IDX])
        result = diagnostic.validation_df_margin_analysis(
            train, train_labels, val, val_labels, k_values=(1,)
        )
        self.assertEqual(result["validation_df_count"], 1)
        self.assertAlmostEqual(result["margin_summary"]["mean"], 1.0)
        self.assertEqual(result["positive_margin_count"], 1)
        self.assertAlmostEqual(result["positive_margin_fraction"], 1.0)
        self.assertAlmostEqual(result["per_query_margin"][0], 1.0)
        purity = result["df_neighbor_purity_by_k"]["k1"]
        self.assertEqual(purity["queries_with_df_majority"], 1)
        self.assertAlmostEqual(purity["df_neighbor_fraction_summary"]["mean"], 1.0)

    def test_margin_summary_reports_fixed_quantile_keys(self):
        train, train_labels = separable_embeddings([8] * 7, seed=7)
        val, val_labels = separable_embeddings([3, 3, 3, 5, 3, 3, 3], seed=8)
        result = diagnostic.validation_df_margin_analysis(train, train_labels, val, val_labels)
        self.assertEqual(
            set(result["margin_summary"]),
            {"count", "min", "p10", "median", "mean", "p90", "max"},
        )
        self.assertEqual(result["validation_df_count"], 5)
        self.assertEqual(set(result["df_neighbor_purity_by_k"]), {"k1", "k5", "k10"})


# --- 10. classification_summary canonical class order ------------------------
class ClassOrderTests(unittest.TestCase):
    def test_summary_uses_canonical_class_order_and_df_target(self):
        train, train_labels = separable_embeddings([6] * 7, seed=9)
        val, val_labels = separable_embeddings([2, 2, 2, 3, 2, 2, 2], seed=10)
        summary = diagnostic.nearest_centroid_analysis(
            train, train_labels, val, val_labels
        )["classification_summary"]
        self.assertEqual(list(summary["per_class_f1"].keys()), config.CLASS_NAMES)
        self.assertEqual(list(summary["per_class_recall"].keys()), config.CLASS_NAMES)
        self.assertEqual(summary["target_class"], "df")


# --- 11/12/13. logistic probe: fixed config, train-only, fail-loud -----------
class LogisticProbeTests(unittest.TestCase):
    def test_config_is_exactly_the_fixed_specification(self):
        self.assertEqual(
            diagnostic.LOGISTIC_REGRESSION_CONFIG,
            {
                "penalty": "l2",
                "C": 1.0,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "fit_intercept": True,
                "tol": 1e-6,
                "max_iter": 5000,
                "random_state": 0,
            },
        )

    def test_fit_signature_only_takes_train(self):
        params = list(inspect.signature(diagnostic.fit_logistic_probe).parameters)
        self.assertEqual(params, ["train_embeddings", "train_labels"])

    def test_module_uses_no_grid_or_cross_validation(self):
        for banned in ("GridSearchCV", "RandomizedSearchCV", "cross_val", "GridSearch"):
            self.assertNotIn(banned, MODULE_SOURCE)

    def test_fit_is_deterministic_and_reports_convergence(self):
        train, train_labels = separable_embeddings([12] * 7, dim=24, seed=11)
        model_a, info_a = diagnostic.fit_logistic_probe(train, train_labels)
        model_b, info_b = diagnostic.fit_logistic_probe(train, train_labels)
        self.assertTrue(info_a["converged"])
        self.assertEqual(info_a["classes"], list(range(config.NUM_CLASSES)))
        np.testing.assert_allclose(model_a.coef_, model_b.coef_)

    def test_non_convergence_fails_loud(self):
        train, train_labels = separable_embeddings([30] * 7, dim=64, seed=12, spread=1.5)
        with mock.patch.dict(diagnostic.LOGISTIC_REGRESSION_CONFIG, {"max_iter": 1}):
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                diagnostic.fit_logistic_probe(train, train_labels)

    def test_analysis_reports_config_iters_and_prediction_counts(self):
        train, train_labels = separable_embeddings([12] * 7, dim=24, seed=13)
        val, val_labels = separable_embeddings([3, 3, 3, 4, 3, 3, 3], dim=24, seed=14)
        result = diagnostic.logistic_probe_analysis(train, train_labels, val, val_labels)
        self.assertEqual(result["config"], diagnostic.LOGISTIC_REGRESSION_CONFIG)
        self.assertTrue(result["converged"])
        self.assertEqual(sum(result["prediction_counts"].values()), len(val_labels))
        self.assertEqual(
            list(result["classification_summary"]["support"].keys()), config.CLASS_NAMES
        )


# --- Blocker 2: validation labels are used only after every fit --------------
class PhaseOrderingTests(unittest.TestCase):
    def _trace(self):
        train, train_labels = separable_embeddings([10] * 7, dim=24, seed=21)
        # a distinct validation length so a fit that saw val labels is detectable
        val, val_labels = separable_embeddings([2, 2, 2, 3, 2, 2, 2], dim=24, seed=22)
        events = []
        real_fit = diagnostic.fit_logistic_probe
        real_centroids = diagnostic._class_centroids
        real_summary = diagnostic.metrics.classification_summary
        fit_label_args = []

        def fit_spy(embeddings, labels):
            fit_label_args.append(np.asarray(labels))
            events.append("logistic_fit")
            return real_fit(embeddings, labels)

        def centroid_spy(train_norm, labels):
            fit_label_args.append(np.asarray(labels))
            events.append("centroid_fit")
            return real_centroids(train_norm, labels)

        def summary_spy(y_true, y_pred):
            events.append("validation_metric")
            return real_summary(y_true, y_pred)

        with mock.patch.object(diagnostic, "fit_logistic_probe", fit_spy), mock.patch.object(
            diagnostic, "_class_centroids", centroid_spy
        ), mock.patch.object(diagnostic.metrics, "classification_summary", summary_spy):
            diagnostic.run_all_analyses(train, train_labels, val, val_labels)
        return events, fit_label_args, val_labels

    def test_logistic_fit_is_first_and_precedes_all_validation_metrics(self):
        events, _, _ = self._trace()
        self.assertEqual(events[0], "logistic_fit")
        self.assertIn("validation_metric", events)
        first_metric = events.index("validation_metric")
        # every fit event happens strictly before the first validation-label metric
        self.assertEqual(events[:first_metric].count("validation_metric"), 0)
        self.assertLess(events.index("logistic_fit"), events.index("centroid_fit"))
        self.assertLess(events.index("centroid_fit"), first_metric)

    def test_validation_labels_never_reach_any_fit(self):
        _, fit_label_args, val_labels = self._trace()
        self.assertTrue(fit_label_args)  # both fits were observed
        for labels in fit_label_args:
            self.assertFalse(np.array_equal(labels, np.asarray(val_labels)))


# --- Blocker 1: complete, shared, validated immutable identity ---------------
class DiagnosticIdentityTests(unittest.TestCase):
    def test_build_identity_accepts_and_stores_the_three_required_objects(self):
        params = list(inspect.signature(diagnostic.build_identity).parameters)
        for required in ("model_identity", "dependency_versions", "algorithm_identities"):
            self.assertIn(required, params)
        identity = _fake_identity()
        for field in diagnostic.IDENTITY_REQUIRED_FIELDS:
            self.assertIn(field, identity)
            self.assertFalse(diagnostic._is_empty_identity_value(identity[field]))
        self.assertEqual(identity["fixed_split_identity"], identity["train_manifest_sha256"])
        self.assertEqual(identity["class_mapping"], config.CLASS_TO_IDX)
        self.assertEqual(identity["embedding_dimension"], 512)

    def test_record_carries_and_does_not_drift_from_identity(self):
        identity = _fake_identity()
        record = diagnostic.build_record(
            identity=identity,
            row_counts={"train": 6995, "validation": 1510},
            class_counts={"train": {}},
            embedding_artifact={"path": "e.npz"},
            analyses={"nearest_centroid": {}},
        )
        self.assertEqual(record["diagnostic_identity"], identity)
        for record_key, identity_key in diagnostic.IDENTITY_DUPLICATED_TOP_LEVEL.items():
            self.assertIn(record_key, record)
            self.assertEqual(record[record_key], identity[identity_key])
        self.assertTrue(diagnostic.validate_diagnostic_identity(record, identity, identity))

    def _record_for(self, identity):
        return diagnostic.build_record(
            identity=identity,
            row_counts={"train": 6995, "validation": 1510},
            class_counts={"train": {}},
            embedding_artifact={"path": "e.npz"},
            analyses={"nearest_centroid": {}},
        )

    def test_missing_identity_fields_fail_loud(self):
        for field in ("model_identity", "dependency_versions", "algorithm_identities"):
            with self.subTest(field=field):
                broken = _fake_identity()
                broken.pop(field)
                record = {**self._record_for(_fake_identity()), "diagnostic_identity": broken}
                with self.assertRaisesRegex(ValueError, "missing field"):
                    diagnostic.validate_diagnostic_identity(record, broken, _fake_identity())

    def test_null_or_empty_identity_field_fails_loud(self):
        for value in (None, {}, ""):
            with self.subTest(value=value):
                broken = _fake_identity(model_identity=value)
                record = {**self._record_for(_fake_identity()), "diagnostic_identity": broken}
                with self.assertRaisesRegex(ValueError, "null/empty"):
                    diagnostic.validate_diagnostic_identity(record, broken, _fake_identity())

    def test_record_identity_disagreeing_with_file_fails_loud(self):
        identity = _fake_identity()
        record = self._record_for(_fake_identity(git_commit="b" * 40))
        with self.assertRaisesRegex(ValueError, "does not match the identity file"):
            diagnostic.validate_diagnostic_identity(record, identity, identity)

    def test_top_level_duplicate_drift_fails_loud(self):
        # 17/17 duplicate-drift matrix: every mapped top-level field must reject.
        rejected, accepted = [], []
        for record_key, identity_key in diagnostic.IDENTITY_DUPLICATED_TOP_LEVEL.items():
            with self.subTest(record_key=record_key):
                identity = _fake_identity()
                record = self._record_for(identity)
                record[record_key] = _distinct_value(record[record_key])
                self.assertNotEqual(record[record_key], identity[identity_key])
                try:
                    diagnostic.validate_diagnostic_identity(record, identity, identity)
                    accepted.append(record_key)
                except ValueError as exc:
                    self.assertIn("disagrees", str(exc))
                    rejected.append(record_key)
                self.assertIn(record_key, rejected)
        self.assertEqual(len(rejected), len(diagnostic.IDENTITY_DUPLICATED_TOP_LEVEL))
        self.assertEqual(len(rejected), 17)
        self.assertEqual(accepted, [])

    def test_top_level_duplicate_missing_fails_loud(self):
        # 17/17 duplicate-missing matrix: deleting any mapped top-level field rejects.
        rejected, accepted = [], []
        for record_key in diagnostic.IDENTITY_DUPLICATED_TOP_LEVEL:
            with self.subTest(record_key=record_key):
                identity = _fake_identity()
                record = self._record_for(identity)
                del record[record_key]
                try:
                    diagnostic.validate_diagnostic_identity(record, identity, identity)
                    accepted.append(record_key)
                except ValueError as exc:
                    self.assertIn("missing top-level duplicate", str(exc))
                    rejected.append(record_key)
                self.assertIn(record_key, rejected)
        self.assertEqual(len(rejected), 17)
        self.assertEqual(accepted, [])

    def test_expected_identity_missing_field_fails_loud(self):
        # 22/22 expected-identity deletion matrix: an incomplete expected rejects.
        rejected, accepted = [], []
        for field in diagnostic.IDENTITY_REQUIRED_FIELDS:
            with self.subTest(field=field):
                identity = _fake_identity()
                record = self._record_for(identity)
                expected = _fake_identity()
                del expected[field]
                try:
                    diagnostic.validate_diagnostic_identity(record, identity, expected)
                    accepted.append(field)
                except ValueError as exc:
                    self.assertIn(field, str(exc))
                    rejected.append(field)
                self.assertIn(field, rejected)
        self.assertEqual(len(rejected), len(diagnostic.IDENTITY_REQUIRED_FIELDS))
        self.assertEqual(len(rejected), 22)
        self.assertEqual(accepted, [])

    def test_notebook_style_expected_rejects_consistently_wrong_model_identity(self):
        # mirrors the notebook: expected identity is rebuilt with the freshly
        # recomputed (correct) model_identity, so a record whose model_identity is
        # wrong-but-consistent across nested, top-level, and file is rejected.
        correct_model = {
            "arch": "coca_vit_b32",
            "model_name": "coca_ViT-B-32",
            "pretrained_tag": "laion2b_s13b_b90k",
            "freeze_mode": "frozen_image_encoder_linear_head",
            "input_resolution": [224, 224],
            "preprocessing_identity": {"train": "T", "eval": "RIGHT-EVAL-PREPROCESS"},
        }
        wrong_model = {**correct_model, "preprocessing_identity": {"train": "T", "eval": "WRONG-EVAL-PREPROCESS"}}
        wrong_identity = _fake_identity(model_identity=wrong_model)
        record = self._record_for(wrong_identity)
        self.assertEqual(record["model_identity"], wrong_model)
        self.assertEqual(record["diagnostic_identity"]["model_identity"], wrong_model)
        expected_identity = _fake_identity(model_identity=correct_model)
        with self.assertRaisesRegex(ValueError, "expected runtime identity"):
            diagnostic.validate_diagnostic_identity(record, wrong_identity, expected_identity)

    def test_internally_consistent_but_all_wrong_vs_expected_fails_loud(self):
        # record and identity file agree, but every copy is consistently wrong
        for wrong_overrides, correct_overrides in (
            ({"git_commit": "c" * 40}, {"git_commit": "a" * 40}),
            (
                {"model_identity": {"arch": "coca_vit_b32", "preprocessing_identity": {"eval": "WRONG-EVAL"}}},
                {"model_identity": {"arch": "coca_vit_b32", "preprocessing_identity": {"eval": "RIGHT-EVAL"}}},
            ),
        ):
            with self.subTest(field=next(iter(wrong_overrides))):
                wrong = _fake_identity(**wrong_overrides)
                record = self._record_for(wrong)
                expected = _fake_identity(**correct_overrides)
                with self.assertRaisesRegex(ValueError, "expected runtime identity"):
                    diagnostic.validate_diagnostic_identity(record, wrong, expected)


# --- Blocker 3B: NPZ dtype / shape / value contract --------------------------
class NpzContractTests(unittest.TestCase):
    def test_valid_bundle_writes_and_reopens_with_exact_dtypes(self):
        arrays = _valid_npz_arrays()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "all_class_embeddings.npz"
            diagnostic.write_all_class_embeddings_npz(path, arrays, n_train=6, n_val=4)
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), set(diagnostic.NPZ_KEYS))
                self.assertEqual(saved["train_embeddings"].dtype, np.float32)
                self.assertEqual(saved["train_labels"].dtype, np.int64)
                self.assertEqual(saved["train_image_ids"].dtype.kind, "U")
                for name, array in arrays.items():
                    self.assertTrue(np.array_equal(saved[name], array))

    def test_float64_embeddings_rejected(self):
        arrays = _valid_npz_arrays()
        arrays["train_embeddings"] = arrays["train_embeddings"].astype(np.float64)
        with self.assertRaisesRegex(ValueError, "train_embeddings must be float32"):
            diagnostic._validate_npz_arrays(arrays, n_train=6, n_val=4)

    def test_non_int64_labels_rejected(self):
        arrays = _valid_npz_arrays()
        arrays["train_labels"] = arrays["train_labels"].astype(np.int32)
        with self.assertRaisesRegex(ValueError, "train_labels must be int64"):
            diagnostic._validate_npz_arrays(arrays, n_train=6, n_val=4)

    def test_object_dtype_ids_rejected(self):
        arrays = _valid_npz_arrays()
        arrays["train_image_ids"] = np.array(list(arrays["train_image_ids"]), dtype=object)
        with self.assertRaisesRegex(ValueError, "must not be object dtype"):
            diagnostic._validate_npz_arrays(arrays, n_train=6, n_val=4)

    def test_missing_key_rejected(self):
        arrays = _valid_npz_arrays()
        del arrays["validation_lesion_ids"]
        with self.assertRaisesRegex(ValueError, "NPZ keys must be exactly"):
            diagnostic._validate_npz_arrays(arrays, n_train=6, n_val=4)

    def test_duplicate_and_empty_and_overlapping_ids_rejected(self):
        dup = _valid_npz_arrays()
        dup["train_image_ids"] = dup["train_image_ids"].copy()
        dup["train_image_ids"][1] = dup["train_image_ids"][0]
        with self.assertRaisesRegex(ValueError, "duplicate ids"):
            diagnostic._validate_npz_arrays(dup, n_train=6, n_val=4)

        empty = _valid_npz_arrays()
        empty["train_lesion_ids"] = empty["train_lesion_ids"].copy()
        empty["train_lesion_ids"][0] = "   "
        with self.assertRaisesRegex(ValueError, "empty/whitespace"):
            diagnostic._validate_npz_arrays(empty, n_train=6, n_val=4)

        overlap = _valid_npz_arrays()
        overlap["validation_image_ids"] = overlap["train_image_ids"][:4].copy()
        with self.assertRaisesRegex(ValueError, "overlap across splits"):
            diagnostic._validate_npz_arrays(overlap, n_train=6, n_val=4)

    def test_reopened_wrong_dtype_rejected(self):
        # write a file whose labels are int32, then the reopen validator must reject
        arrays = _valid_npz_arrays()
        arrays["train_labels"] = arrays["train_labels"].astype(np.int32)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.npz"
            with path.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            with self.assertRaisesRegex(ValueError, "train_labels must be int64"):
                diagnostic._validate_npz_file(path, n_train=6, n_val=4)

    def test_non_unit_norm_embeddings_rejected(self):
        arrays = _valid_npz_arrays()
        arrays["validation_embeddings"] = (arrays["validation_embeddings"] * 2.0).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "not unit norm"):
            diagnostic._validate_npz_arrays(arrays, n_train=6, n_val=4)


# --- Blocker 3A: record-integrity sidecar + byte-identical formal records ----
class RecordIntegrityTests(unittest.TestCase):
    def _write(self, temp):
        identity = _fake_identity()
        record = diagnostic.build_record(
            identity=identity,
            row_counts={"train": 6995, "validation": 1510},
            class_counts={"train": {}},
            embedding_artifact={"path": "e.npz", "size_bytes": 1, "sha256": "z" * 64},
            analyses={"nearest_centroid": {}},
        )
        out = Path(temp) / "attempt"
        out.mkdir()
        latest = Path(temp) / "latest_diagnostic_record.json"
        diagnostic.write_formal_records(
            output_dir=out, latest_record_path=latest, record=record, identity=identity
        )
        return out, latest

    def test_three_formal_records_byte_identical_and_sidecar_certifies(self):
        with tempfile.TemporaryDirectory() as temp:
            out, latest = self._write(temp)
            diagnostic_bytes = (out / "all_class_separability_diagnostic.json").read_bytes()
            completed_bytes = (out / "_COMPLETED.json").read_bytes()
            self.assertEqual(diagnostic_bytes, completed_bytes)
            self.assertEqual(diagnostic_bytes, latest.read_bytes())
            sidecar = json.loads((out / "record_integrity.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["schema"], diagnostic.RECORD_INTEGRITY_SCHEMA)
            self.assertEqual(sidecar["algorithm"], "sha256")
            self.assertIs(sidecar["raw_bytes_equal"], True)
            hashes = {sidecar[k]["sha256"] for k in diagnostic.RECORD_INTEGRITY_ENTRIES}
            self.assertEqual(len(hashes), 1)
            self.assertTrue(
                diagnostic.validate_record_integrity(output_dir=out, latest_record_path=latest)
            )
            # the record must not embed its own hash (no circular hash)
            self.assertNotIn(next(iter(hashes)), diagnostic_bytes.decode("utf-8"))

    def test_missing_sidecar_fails_loud(self):
        with tempfile.TemporaryDirectory() as temp:
            out, latest = self._write(temp)
            (out / "record_integrity.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "record-integrity artifact"):
                diagnostic.validate_record_integrity(output_dir=out, latest_record_path=latest)

    def test_corrupt_sidecar_hash_and_bytes_fail_loud(self):
        with tempfile.TemporaryDirectory() as temp:
            out, latest = self._write(temp)
            sidecar_path = out / "record_integrity.json"
            broken = json.loads(sidecar_path.read_text(encoding="utf-8"))
            broken["diagnostic_record"]["sha256"] = "0" * 64
            sidecar_path.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch for diagnostic_record"):
                diagnostic.validate_record_integrity(output_dir=out, latest_record_path=latest)

    def test_diverging_record_bytes_fail_loud(self):
        with tempfile.TemporaryDirectory() as temp:
            out, latest = self._write(temp)
            completed = out / "_COMPLETED.json"
            completed.write_text(completed.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw bytes diverged"):
                diagnostic.validate_record_integrity(output_dir=out, latest_record_path=latest)

    def test_write_formal_records_refuses_existing_latest(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "attempt"
            out.mkdir()
            latest = Path(temp) / "latest_diagnostic_record.json"
            latest.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                diagnostic.write_formal_records(
                    output_dir=out, latest_record_path=latest, record={"a": 1}, identity={"b": 2}
                )

    def test_formal_write_leaves_prior_record_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            prior = Path(temp) / "latest_validation_failure.json"
            prior.write_text('{"validation_status": "VALIDATION FAILED"}', encoding="utf-8")
            before = (diagnostic.sha256_file(prior), prior.stat().st_mtime_ns)
            self._write(temp)
            after = (diagnostic.sha256_file(prior), prior.stat().st_mtime_ns)
            self.assertEqual(before, after)


# --- prepare_output_dir + numeric fail-loud ----------------------------------
class GuardTests(unittest.TestCase):
    def test_l2_normalize_rejects_bad_matrices(self):
        for bad in (
            np.zeros((2, 4)),
            np.array([[np.nan, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]),
            np.array([[np.inf, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]),
            np.ones((4,)),
        ):
            with self.subTest(shape=bad.shape):
                with self.assertRaises(ValueError):
                    diagnostic.l2_normalize(bad)

    def test_dimension_mismatch_fails_loud(self):
        train, train_labels = separable_embeddings([4] * 7, dim=8, seed=15)
        val, val_labels = separable_embeddings([2] * 7, dim=6, seed=16)
        with self.assertRaisesRegex(ValueError, "different feature dimensions"):
            diagnostic.nearest_centroid_analysis(train, train_labels, val, val_labels)

    def test_label_out_of_range_fails_loud(self):
        train, train_labels = separable_embeddings([4] * 7, dim=8, seed=17)
        bad_labels = train_labels.copy()
        bad_labels[0] = config.NUM_CLASSES
        with self.assertRaisesRegex(ValueError, "labels must lie"):
            diagnostic._validate_labels(bad_labels, len(bad_labels), name="train")

    def test_prepare_output_dir_rejects_non_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            empty = Path(temp) / "empty"
            empty.mkdir()
            self.assertEqual(diagnostic.prepare_output_dir(empty), empty)
            occupied = Path(temp) / "occupied"
            occupied.mkdir()
            (occupied / "prior.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                diagnostic.prepare_output_dir(occupied)


if __name__ == "__main__":
    unittest.main()
