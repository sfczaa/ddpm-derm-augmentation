"""The C4-filtered condition, checked against its pre-registration.

These tests exist because the value of this experiment is entirely in the rules
having been fixed before any outcome was seen. A filtering step that picks its
threshold after looking at df F1 is not evidence of anything, so what is
asserted here is the *rule*, not merely that some code runs:

  - the threshold is derived from the val set every run, never hardcoded;
  - acceptance is at-or-below that threshold, so an image exactly as far away
    as the most unusual genuinely-new real df is kept;
  - every accepted image is used, because a top-N would silently change meaning
    with pool size;
  - below the pre-registered minimum the condition does not run at all, and the
    accepted manifest is not written, so nothing downstream can train through
    the shortfall by accident;
  - df totals df_target_count either way, so C4-filtered differs from C1 only
    in where the df rows come from.

The judge itself is stubbed: these tests are about the selection rule, and a
134 MB checkpoint would make them a different kind of test.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ddpm_derm import config, manifests  # noqa: E402

GREY_BASE = 128
CANDIDATE_COLUMNS = ["image_path", "label_idx", "dx", "lesion_id", "image_id", "source"]


def solid(grey: int, size: int = 8) -> Image.Image:
    return Image.new("RGB", (size, size), (grey, grey, grey))


def write_candidate_pool(directory: Path, greys: list[int]) -> Path:
    """A candidate manifest plus its images, shaped like the real one."""
    images_dir = directory / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = directory / "synthetic_df.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CANDIDATE_COLUMNS)
        for index, grey in enumerate(greys):
            image_id = f"synthetic_df_{index:04d}"
            solid(grey).save(images_dir / f"{image_id}.png")
            writer.writerow([
                f"images/{image_id}.png",
                config.TARGET_CLASS_IDX,
                config.TARGET_CLASS,
                "synthetic",
                image_id,
                manifests.GENERATED_SOURCE,
            ])
    return manifest


def grey_features(images, model=None, img_size=None, batch=64) -> np.ndarray:
    """Stand-in for the judge: one feature, the image's grey level.

    Distances become plain differences in grey level, so every threshold in
    these tests is exact rather than approximate.
    """
    return np.array([[np.asarray(im, dtype=float).mean() / 255.0] for im in images])


class FilteredFrameTests(unittest.TestCase):
    """Frame composition. Uses the real fixed train split, which is read-only."""

    @classmethod
    def setUpClass(cls):
        frame = manifests.load_split("train")
        cls.n_real = int((frame["label_idx"] == config.TARGET_CLASS_IDX).sum())

    def build(self, accepted_count: int, df_target_count: int = 585):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_candidate_pool(
                Path(tmp), [GREY_BASE] * accepted_count
            )
            return manifests.build_classifier_filtered_frame(
                df_target_count=df_target_count,
                seed=0,
                accepted_manifest=manifest,
            )

    def test_df_total_is_the_target_regardless_of_how_many_were_accepted(self):
        # C1 and C4-filtered must differ only in the source of the df rows. If
        # the total fell with the accepted count, the comparison would confound
        # "filtering" with "less oversampling" and answer neither question.
        for accepted in (50, 137, 500):
            with self.subTest(accepted=accepted):
                frame, _ = self.build(accepted)
                df_rows = int((frame["label_idx"] == config.TARGET_CLASS_IDX).sum())
                self.assertEqual(df_rows, 585)

    def test_every_accepted_image_is_used(self):
        # No top-N: the count is whatever cleared the threshold.
        frame, intervention = self.build(137)
        synthetic = frame[frame["source"] == manifests.GENERATED_SOURCE]
        self.assertEqual(len(synthetic), 137)
        self.assertEqual(intervention["accepted_synthetic_count"], 137)
        self.assertEqual(synthetic["image_id"].nunique(), 137)

    def test_remaining_df_slots_are_filled_by_real_duplication(self):
        frame, intervention = self.build(137)
        self.assertEqual(intervention["original_real_df_count"], self.n_real)
        self.assertEqual(
            intervention["duplicated_real_df_count"], 585 - 137 - self.n_real
        )
        self.assertEqual(
            intervention["accepted_synthetic_count"]
            + intervention["original_real_df_count"]
            + intervention["duplicated_real_df_count"],
            intervention["total_df_count"],
        )

    def test_below_the_pre_registered_minimum_the_condition_does_not_run(self):
        # Section 4: the shortfall is the result. Training a condition on a
        # handful of images would produce noise, not an answer.
        self.assertEqual(manifests.FILTERED_MINIMUM_ACCEPTED, 50)
        with self.assertRaises(ValueError) as caught:
            self.build(manifests.FILTERED_MINIMUM_ACCEPTED - 1)
        self.assertIn("Report the shortfall", str(caught.exception))

    def test_more_accepted_rows_than_fit_is_refused(self):
        with self.assertRaises(ValueError):
            self.build(501)

    def test_composition_is_deterministic_for_a_seed(self):
        first, _ = self.build(137)
        second, _ = self.build(137)
        self.assertEqual(
            first["image_id"].tolist(), second["image_id"].tolist()
        )


class SelectionRuleTests(unittest.TestCase):
    """The acceptance rule itself, with the judge stubbed out."""

    def run_selection(self, val_offsets: list[int], candidate_offsets: list[int]):
        import c4_filtered_select

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: None)
        manifest = write_candidate_pool(
            tmp, [GREY_BASE + off for off in candidate_offsets]
        )
        accepted_out = tmp / "accepted.csv"
        record_out = tmp / "record.json"

        def fake_load_real_df(split):
            self.assertIn(split, ("train", "val"))
            if split == "train":
                return [solid(GREY_BASE) for _ in range(3)]
            return [solid(GREY_BASE + off) for off in val_offsets]

        with mock.patch.object(c4_filtered_select, "load_real_df", fake_load_real_df), \
             mock.patch.object(c4_filtered_select, "embedding_features", grey_features), \
             mock.patch.object(
                 c4_filtered_select, "load_real_data_judge",
                 lambda path: (None, {"img_size": 8, "variant": "C1", "seed": 2}),
             ), \
             mock.patch.object(
                 c4_filtered_select, "synthetic_provenance", lambda directory: {}
             ), \
             mock.patch.object(sys, "argv", [
                 "c4_filtered_select.py",
                 "--candidate-manifest", str(manifest),
                 "--judge-checkpoint", "unused",
                 "--accepted-out", str(accepted_out),
                 "--record-out", str(record_out),
             ]):
            c4_filtered_select.main()

        record = json.loads(record_out.read_text(encoding="utf-8"))
        return record, accepted_out

    def test_threshold_is_the_farthest_val_image_not_a_constant(self):
        record, _ = self.run_selection(
            val_offsets=[13, 26, 51],
            candidate_offsets=[25] * 60,
        )
        self.assertAlmostEqual(record["threshold"], 51 / 255.0, places=12)

        # Move the val set and the threshold must move with it.
        moved, _ = self.run_selection(
            val_offsets=[13, 26, 80],
            candidate_offsets=[25] * 60,
        )
        self.assertAlmostEqual(moved["threshold"], 80 / 255.0, places=12)

    def test_an_image_exactly_at_the_threshold_is_accepted(self):
        # "no farther from real df than the most unusual genuinely new real df"
        # is inclusive; an off-by-one here would silently drop boundary images.
        record, accepted_out = self.run_selection(
            val_offsets=[13, 26, 51],
            candidate_offsets=[25] * 55 + [51] + [80] * 5,
        )
        self.assertEqual(record["accepted_count"], 56)
        self.assertTrue(record["condition_runnable"])
        rows = list(csv.DictReader(accepted_out.open(encoding="utf-8")))
        self.assertEqual(len(rows), 56)

    def test_accepted_manifest_is_a_subset_of_the_candidate(self):
        record, accepted_out = self.run_selection(
            val_offsets=[13, 26, 51],
            candidate_offsets=[25] * 55 + [80] * 5,
        )
        rows = list(csv.DictReader(accepted_out.open(encoding="utf-8")))
        self.assertEqual(sorted(rows[0].keys()), sorted(CANDIDATE_COLUMNS))
        self.assertEqual(len(rows), record["accepted_count"])
        # Relative image_path is what keeps the manifest+images folder portable.
        for row in rows:
            self.assertFalse(Path(row["image_path"]).is_absolute())

    def test_shortfall_records_the_result_and_writes_no_manifest(self):
        record, accepted_out = self.run_selection(
            val_offsets=[13, 26, 51],
            candidate_offsets=[25] * 10 + [80] * 20,
        )
        self.assertEqual(record["accepted_count"], 10)
        self.assertFalse(record["condition_runnable"])
        self.assertEqual(record["minimum_accepted"], 50)
        self.assertFalse(
            accepted_out.exists(),
            "a shortfall must not leave a manifest a training run could pick up",
        )

    def test_record_reports_diversity_of_the_accepted_subset(self):
        # A subset that is closer but far less varied is a different animal
        # from one that is closer and equally varied. Section 5 requires this
        # to be visible rather than inferred.
        record, _ = self.run_selection(
            val_offsets=[13, 26, 51],
            candidate_offsets=list(range(0, 50)) + [10] * 10,
        )
        self.assertIn("diversity", record)
        self.assertIn("within_accepted", record["diversity"])
        self.assertIn("within_set_median_ratio", record["diversity"])

    def test_the_test_split_is_never_opened(self):
        record, _ = self.run_selection(
            val_offsets=[13, 26, 51],
            candidate_offsets=[25] * 60,
        )
        self.assertFalse(record["test_split_accessed"])
        self.assertTrue(record["flip_aware"])


class TrainerArgumentTests(unittest.TestCase):
    """The CLI cannot be talked into training the wrong thing."""

    def parse(self, *argv):
        from ddpm_derm import train_classifier

        return train_classifier.parse_args(list(argv))

    def test_filtered_variant_requires_the_accepted_manifest(self):
        with self.assertRaises(SystemExit):
            self.parse("--variant", "C4_FILTERED")

    def test_accepted_manifest_is_rejected_for_other_variants(self):
        for variant in ("C0", "C1", "C4"):
            with self.subTest(variant=variant), self.assertRaises(SystemExit):
                self.parse(
                    "--variant", variant,
                    "--generated-manifest", "x.csv",
                    "--accepted-manifest", "y.csv",
                )

    def test_accepted_manifest_and_mixture_count_are_not_combined(self):
        # Two different selection rules; combining them would make the
        # resulting condition unattributable to either.
        with self.assertRaises(SystemExit):
            self.parse(
                "--variant", "C4_FILTERED",
                "--accepted-manifest", "y.csv",
                "--mixture-synthetic-count", "10",
            )

    def test_filtered_variant_parses_with_its_manifest(self):
        args = self.parse(
            "--variant", "C4_FILTERED", "--accepted-manifest", "y.csv",
        )
        self.assertEqual(args.variant, "C4_FILTERED")
        self.assertEqual(args.accepted_manifest, "y.csv")


if __name__ == "__main__":
    unittest.main()
