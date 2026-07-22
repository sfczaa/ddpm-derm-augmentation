"""Regression tests for the post-v4 synthetic mixture diagnostic."""

from __future__ import annotations

import copy
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm import classifier_run, config, manifests  # noqa: E402
from ddpm_derm.train_classifier import parse_args  # noqa: E402


def real_frame() -> pd.DataFrame:
    rows = []
    for index in range(85):
        rows.append({
            "image_path": f"real/{index}.jpg",
            "label_idx": config.TARGET_CLASS_IDX,
            "dx": config.TARGET_CLASS,
            "lesion_id": f"real-lesion-{index}",
            "image_id": f"real-{index}",
        })
    rows.append({
        "image_path": "real/nv.jpg", "label_idx": config.CLASS_TO_IDX["nv"],
        "dx": "nv", "lesion_id": "nv-lesion", "image_id": "nv-0",
    })
    return pd.DataFrame(rows)


def synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "image_path": f"synthetic/{index}.png",
        "label_idx": config.TARGET_CLASS_IDX,
        "dx": config.TARGET_CLASS,
        "lesion_id": "synthetic",
        "image_id": f"synthetic-{index:03d}",
        "source": "synthetic",
    } for index in range(500)])


class SyntheticMixtureDiagnosticTests(unittest.TestCase):
    def build(self, count: int, candidate: pd.DataFrame | None = None):
        candidate = synthetic_frame() if candidate is None else candidate
        with mock.patch.object(manifests, "load_split", return_value=real_frame()), \
             mock.patch.object(manifests, "load_generated_manifest", return_value=candidate):
            return manifests.build_classifier_mixture_frame(
                df_target_count=585,
                synthetic_count=count,
                seed=0,
                generated_manifest="candidate.csv",
            )

    def test_fixed_df_support_and_nested_synthetic_prefixes(self):
        prior_ids: set[str] = set()
        prior_real_duplicates: Counter[str] | None = None
        order_hash = None
        for count in (0, 125, 250, 375, 500):
            with self.subTest(count=count):
                frame, identity = self.build(count)
                selected = set(frame.loc[frame["source"] == "synthetic", "image_id"])
                self.assertEqual(len(frame), 586)
                self.assertEqual(
                    int((frame["label_idx"] == config.TARGET_CLASS_IDX).sum()), 585
                )
                self.assertEqual(len(selected), count)
                self.assertTrue(prior_ids <= selected)
                self.assertEqual(identity["duplicated_real_df_count"], 500 - count)
                self.assertEqual(identity["total_df_count"], 585)
                real_counts = Counter(
                    frame.loc[
                        (frame["source"] == "real")
                        & (frame["label_idx"] == config.TARGET_CLASS_IDX),
                        "image_id",
                    ]
                )
                real_duplicates = Counter({key: value - 1 for key, value in real_counts.items()})
                if prior_real_duplicates is not None:
                    self.assertTrue(all(
                        prior_real_duplicates[key] >= value
                        for key, value in real_duplicates.items()
                    ))
                if order_hash is None:
                    order_hash = identity["candidate_order_sha256"]
                self.assertEqual(identity["candidate_order_sha256"], order_hash)
                prior_ids = selected
                prior_real_duplicates = real_duplicates

    def test_selection_is_independent_of_candidate_row_order(self):
        first_frame, first_identity = self.build(125)
        shuffled = synthetic_frame().sample(frac=1, random_state=9).reset_index(drop=True)
        second_frame, second_identity = self.build(125, shuffled)
        first_ids = set(first_frame.loc[first_frame["source"] == "synthetic", "image_id"])
        second_ids = set(second_frame.loc[second_frame["source"] == "synthetic", "image_id"])
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(first_identity, second_identity)

    def test_shared_synthetic_lesion_marker_is_allowed_for_train_only_rows(self):
        candidate = synthetic_frame()
        self.assertEqual(candidate["lesion_id"].nunique(), 1)
        frame, identity = self.build(125, candidate)
        self.assertEqual(int((frame["source"] == "synthetic").sum()), 125)
        self.assertEqual(identity["selected_synthetic_count"], 125)

    def test_invalid_counts_and_image_ids_fail_loud(self):
        with self.assertRaisesRegex(ValueError, "exceeds candidate size"):
            self.build(501)
        duplicate = synthetic_frame()
        duplicate.loc[1, "image_id"] = duplicate.loc[0, "image_id"]
        with self.assertRaisesRegex(ValueError, "must be unique"):
            self.build(125, duplicate)

    def test_cli_confines_mixture_to_isolated_validation_only_c4(self):
        valid = [
            "--variant", "C4", "--generated-manifest", "candidate.csv",
            "--mixture-synthetic-count", "125", "--evaluation-scope", "validation_only",
            "--run-label", "mixture_s125", "--output-dir", "isolated",
        ]
        self.assertEqual(parse_args(valid).mixture_synthetic_count, 125)
        invalid = (
            ["--variant", "C1", "--mixture-synthetic-count", "125"],
            ["--variant", "C4", "--generated-manifest", "candidate.csv", "--mixture-synthetic-count", "125"],
            ["--variant", "C4", "--generated-manifest", "candidate.csv", "--mixture-synthetic-count", "125", "--evaluation-scope", "validation_only"],
        )
        for argv in invalid:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_resume_rejects_mixture_identity_drift(self):
        saved = {
            "schema_version": 2,
            "run_label": "mixture_s125", "variant": "C4", "seed": 0,
            "fixed_config": {}, "candidate_manifest_sha256": "a" * 64,
            "source_split": "train", "source_manifest_sha256": "b" * 64,
            "git_commit": "c" * 40, "run_version": "mixture_v1",
            "model_identity": {}, "fixed_split_identity": "split",
            "shared_root_uuid": "root", "formal_output_identity": "diagnostic",
            "model_selection_metric": "validation_df_f1_strict_improvement",
            "class_mapping": {}, "checkpoint_format": "head_only",
            "data_intervention": {"selected_synthetic_count": 125},
        }
        current = copy.deepcopy(saved)
        current["data_intervention"]["selected_synthetic_count"] = 250
        with self.assertRaisesRegex(ValueError, "data_intervention"):
            classifier_run.require_matching_resume_identity(saved, current)


if __name__ == "__main__":
    unittest.main()
