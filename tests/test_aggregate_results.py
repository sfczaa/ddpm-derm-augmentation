"""Classifier aggregation must stay within one complete model identity."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import aggregate_results  # noqa: E402


class AggregateResultsTests(unittest.TestCase):
    def result(self, arch, tag="laion2b_s13b_b90k"):
        return {
            "variant": "C1",
            "seed": 0,
            "run_identity": {"checkpoint_format": "frozen_backbone_head_only_v1", "model_identity": {
                "arch": arch,
                "model_name": "coca_ViT-B-32" if arch == "coca_vit_b32" else "resnet18",
                "pretrained_tag": tag,
                "freeze_mode": "frozen_image_encoder_linear_head",
                "preprocessing_identity": {"train": "native", "eval": "native"},
                "input_resolution": [224, 224],
                "open_clip_torch_version": "3.3.0",
            }},
            "test_metrics": {"target_f1": 0.5, "macro_f1": 0.4, "target_recall": 0.3},
        }

    def test_expected_arch_is_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results_C1_seed0.json"
            path.write_text(json.dumps(self.result("resnet18")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected"):
                aggregate_results.load_results(Path(temp), "coca_vit_b32")

    def test_model_tag_mixing_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "results_C1_seed0.json").write_text(
                json.dumps(self.result("coca_vit_b32")), encoding="utf-8"
            )
            second = self.result("coca_vit_b32", tag="other")
            second["seed"] = 1
            (root / "results_C1_seed1.json").write_text(
                json.dumps(second), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "mixed"):
                aggregate_results.load_results(root, "coca_vit_b32")

    def test_checkpoint_format_mixing_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.result("coca_vit_b32")
            second = self.result("coca_vit_b32")
            second["seed"] = 1
            second["run_identity"]["checkpoint_format"] = "full_model_v1"
            for name, value in (("a.json", first), ("b.json", second)):
                (root / f"results_{name}").write_text(
                    json.dumps(value), encoding="utf-8"
                )
            with self.assertRaisesRegex(ValueError, "mixed"):
                aggregate_results.load_results(root, "coca_vit_b32")


if __name__ == "__main__":
    unittest.main()
