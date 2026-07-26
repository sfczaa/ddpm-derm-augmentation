"""Torch-free tests for the exploratory DDPM sampler policy."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm import config, ddpm_sampler, manifests  # noqa: E402


class DDPMSamplerTests(unittest.TestCase):
    def test_natural_is_default_and_keeps_shuffle(self):
        self.assertEqual(
            ddpm_sampler.DEFAULT_SAMPLER_STRATEGY, ddpm_sampler.NATURAL
        )
        plan = ddpm_sampler.sampling_plan(ddpm_sampler.NATURAL)
        self.assertEqual(
            plan, {"shuffle": True, "use_weighted_sampler": False}
        )
        self.assertEqual(
            ddpm_sampler.per_sample_weights([0, 0, 1], "natural"),
            [1.0, 1.0, 1.0],
        )

    def test_unknown_strategy_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "unknown sampler strategy"):
            ddpm_sampler.sampling_plan("fully_balanced")

    def test_sqrt_plan_makes_sampler_and_shuffle_mutually_exclusive(self):
        plan = ddpm_sampler.sampling_plan("sqrt_balanced")
        self.assertFalse(plan["shuffle"])
        self.assertTrue(plan["use_weighted_sampler"])
        self.assertFalse(plan["shuffle"] and plan["use_weighted_sampler"])

    def test_sqrt_formula_is_per_sample_inverse_sqrt_class_count(self):
        labels = [0, 0, 0, 0, 1]
        weights = ddpm_sampler.per_sample_weights(labels, "sqrt_balanced")
        self.assertEqual(weights[:4], [0.5] * 4)
        self.assertEqual(weights[4], 1.0)

    def test_real_train_rows_receive_their_class_weight(self):
        train = manifests.load_ddpm_train_frame()
        labels = train["label_idx"].tolist()
        weights = ddpm_sampler.per_sample_weights(labels, "sqrt_balanced")
        counts = train["label_idx"].value_counts().to_dict()
        self.assertEqual(len(weights), len(train))
        for label, weight in zip(labels, weights):
            self.assertAlmostEqual(weight, 1.0 / math.sqrt(counts[label]))

    def test_expected_class_proportion_uses_sqrt_counts(self):
        train = manifests.load_ddpm_train_frame()
        summary = ddpm_sampler.sampler_summary(
            train["label_idx"].tolist(),
            "sqrt_balanced",
            config.IDX_TO_CLASS,
        )
        denominator = sum(
            math.sqrt(values["count"]) for values in summary.values()
        )
        for values in summary.values():
            expected = math.sqrt(values["count"]) / denominator
            self.assertAlmostEqual(
                values["expected_sampling_proportion"], expected
            )
        self.assertAlmostEqual(
            sum(v["expected_sampling_proportion"] for v in summary.values()),
            1.0,
        )

    def test_ddpm_frame_is_train_only_and_seeded_limit_repeats(self):
        with mock.patch.object(
            manifests,
            "load_split",
            wraps=manifests.load_split,
        ) as load_split:
            frame_a = manifests.load_ddpm_train_frame(limit=64, seed=7)
            frame_b = manifests.load_ddpm_train_frame(limit=64, seed=7)

        self.assertEqual(
            load_split.call_args_list,
            [mock.call("train"), mock.call("train")],
        )
        self.assertEqual(frame_a["image_id"].tolist(), frame_b["image_id"].tolist())

    def test_checkpoint_strategy_guard_rejects_mismatch(self):
        self.assertEqual(
            ddpm_sampler.require_matching_checkpoint_strategy({}, "natural"),
            "natural",
        )
        self.assertEqual(
            ddpm_sampler.require_matching_checkpoint_strategy(
                {"sampler_strategy": "sqrt_balanced"}, "sqrt_balanced"
            ),
            "sqrt_balanced",
        )
        with self.assertRaisesRegex(ValueError, "sampler strategy mismatch"):
            ddpm_sampler.require_matching_checkpoint_strategy(
                {"sampler_strategy": "natural"}, "sqrt_balanced"
            )
        with self.assertRaisesRegex(ValueError, "unknown sampler strategy"):
            ddpm_sampler.checkpoint_sampler_strategy(
                {"sampler_strategy": "mystery"}
            )


if __name__ == "__main__":
    unittest.main()
