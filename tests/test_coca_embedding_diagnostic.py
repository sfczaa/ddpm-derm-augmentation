"""Local tests for the descriptive frozen-CoCa embedding diagnostic."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm import coca_embedding_diagnostic as diagnostic  # noqa: E402


def frame(count, *, prefix, source="real"):
    return pd.DataFrame(
        {
            "image_path": [f"{prefix}-{index}.jpg" for index in range(count)],
            "label_idx": [3] * count,
            "dx": ["df"] * count,
            "lesion_id": [f"{prefix}-lesion-{index}" for index in range(count)],
            "image_id": [f"{prefix}-image-{index}" for index in range(count)],
            "source": [source] * count,
        }
    )


class CoCaEmbeddingDiagnosticTests(unittest.TestCase):
    def test_build_groups_uses_train_and_validation_but_never_test(self):
        train = pd.concat([frame(85, prefix="train"), frame(2, prefix="other").assign(dx="nv", label_idx=5)])
        validation = frame(14, prefix="validation")
        synthetic = frame(500, prefix="synthetic", source="synthetic")

        def load_split(name):
            self.assertIn(name, {"train", "val"})
            return train if name == "train" else validation

        with patch.object(diagnostic.manifests, "load_split", side_effect=load_split) as split_loader, patch.object(
            diagnostic.manifests,
            "load_generated_manifest",
            return_value=synthetic,
        ):
            groups = diagnostic.build_df_groups("candidate.csv")
        self.assertEqual(split_loader.call_args_list[0].args, ("train",))
        self.assertEqual(split_loader.call_args_list[1].args, ("val",))
        self.assertEqual({name: len(value) for name, value in groups.items()}, diagnostic.EXPECTED_GROUP_COUNTS)

    def test_build_groups_rejects_duplicate_real_image_identity(self):
        train = frame(85, prefix="train")
        train.loc[1, "image_id"] = train.loc[0, "image_id"]
        with patch.object(diagnostic.manifests, "load_split", side_effect=[train, frame(14, prefix="validation")]), patch.object(
            diagnostic.manifests,
            "load_generated_manifest",
            return_value=frame(500, prefix="synthetic", source="synthetic"),
        ):
            with self.assertRaisesRegex(ValueError, "duplicated image_id"):
                diagnostic.build_df_groups("candidate.csv")

    def test_normalization_rejects_zero_and_non_finite_rows(self):
        for values in (
            np.array([[0.0, 0.0]]),
            np.array([[np.nan, 1.0]]),
            np.array([[np.inf, 1.0]]),
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    diagnostic.l2_normalize(values)

    def test_analysis_reports_centroid_and_directional_nearest_neighbors(self):
        embeddings = {
            "real_train_df": np.array([[2.0, 0.0], [1.0, 0.0]]),
            "synthetic_df": np.array([[0.0, 3.0]]),
            "validation_df": np.array([[1.0, 1.0]]),
        }
        result = diagnostic.analyze_embedding_groups(embeddings)
        self.assertEqual(result["feature_dimension"], 2)
        self.assertEqual(result["group_counts"], {"real_train_df": 2, "synthetic_df": 1, "validation_df": 1})
        self.assertAlmostEqual(
            result["centroid_cosine_distances"]["real_train_df__synthetic_df"]["cosine_distance"],
            1.0,
        )
        self.assertAlmostEqual(
            result["cross_group_nearest_neighbor_cosine_distances"]["real_train_df_to_synthetic_df"]["mean"],
            1.0,
        )
        self.assertAlmostEqual(
            result["centroid_cosine_distances"]["real_train_df__validation_df"]["cosine_distance"],
            1.0 - 1.0 / np.sqrt(2.0),
        )

    def test_analysis_is_invariant_to_positive_row_scaling(self):
        base = {
            "real_train_df": np.array([[1.0, 0.0], [1.0, 1.0]]),
            "synthetic_df": np.array([[0.0, 1.0], [1.0, 2.0]]),
            "validation_df": np.array([[1.0, -1.0]]),
        }
        scaled = {name: values * np.arange(1, len(values) + 1)[:, None] * 7 for name, values in base.items()}
        def numeric_leaves(value):
            if isinstance(value, dict):
                return [item for key in sorted(value) for item in numeric_leaves(value[key])]
            return [float(value)]

        np.testing.assert_allclose(
            numeric_leaves(diagnostic.analyze_embedding_groups(base)),
            numeric_leaves(diagnostic.analyze_embedding_groups(scaled)),
            rtol=0,
            atol=1e-15,
        )

    def test_analysis_rejects_wrong_order_and_dimension_mismatch(self):
        wrong_order = {
            "synthetic_df": np.ones((1, 2)),
            "real_train_df": np.ones((1, 2)),
            "validation_df": np.ones((1, 2)),
        }
        with self.assertRaisesRegex(ValueError, "ordered"):
            diagnostic.analyze_embedding_groups(wrong_order)
        mismatch = {
            "real_train_df": np.ones((1, 2)),
            "synthetic_df": np.ones((1, 3)),
            "validation_df": np.ones((1, 2)),
        }
        with self.assertRaisesRegex(ValueError, "different feature dimensions"):
            diagnostic.analyze_embedding_groups(mismatch)

    def test_output_dir_accepts_drive_probed_empty_directory_but_not_artifacts(self):
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
