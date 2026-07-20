"""Local behavioral verification for the CoCa v2 weighted objective."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm import classifier_objective, classifier_run, coca_run  # noqa: E402
from ddpm_derm.train_classifier import (  # noqa: E402
    _load_trusted_checkpoint,
    build_criterion,
    evaluate_test_scope,
    parse_args,
    save_checkpoint,
)


EXPECTED_COUNTS = {
    "akiec": 226,
    "bcc": 348,
    "bkl": 778,
    "df": 585,
    "mel": 782,
    "nv": 4684,
    "vasc": 92,
}
EXPECTED_WEIGHTS = np.asarray([
    1.323009871451,
    1.066173005077,
    0.713062349324,
    0.822317194099,
    0.711236322385,
    0.290608866819,
    2.073592390846,
])


def matched_frame():
    labels = []
    for class_idx, name in enumerate(classifier_objective.CLASS_ORDER):
        labels.extend([class_idx] * EXPECTED_COUNTS[name])
    return pd.DataFrame({"label_idx": labels})


def weighted_objective():
    return classifier_objective.build_training_objective(
        "inverse_sqrt", matched_frame()
    )[0]


def model_identity():
    return {
        "arch": coca_run.ARCH,
        "model_name": coca_run.MODEL_NAME,
        "pretrained_tag": coca_run.PRETRAINED_TAG,
        "freeze_mode": "frozen_image_encoder_linear_head",
        "preprocessing_identity": {"train": "native", "eval": "native"},
        "input_resolution": [224, 224],
        "open_clip_torch_version": "3.3.0",
    }


def aggregate_runs():
    runs = []
    objective = weighted_objective()
    for variant, offset in (("C1", 0.0), ("C4", 0.1)):
        for seed in (0, 1, 2):
            runs.append({
                "variant": variant,
                "seed": seed,
                "evaluation_scope": "full",
                "run_identity": {
                    "model_identity": model_identity(),
                    "checkpoint_format": coca_run.CHECKPOINT_FORMAT,
                    "run_version": "v2_weighted_ce",
                    "training_objective": copy.deepcopy(objective),
                    "evaluation_scope": "full",
                },
                "test_metrics": {
                    "target_f1": 0.5 + offset,
                    "macro_f1": 0.4 + offset,
                    "target_recall": 0.3 + offset,
                    "per_class_recall": {"df": 0.3 + offset},
                },
            })
    return runs


def build_weighted_identity(evaluation_scope="full"):
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / "train.csv"
        source.write_text("split,train\n", encoding="utf-8")
        return classifier_run.build_run_identity(
            run_label="coca_v2", variant="C1", seed=0, epochs=20,
            img_size=128, batch_size=32, learning_rate=3e-4,
            weight_decay=1e-4, df_target_count=585, pretrained=True,
            limit=None, candidate_manifest=None, source_split="train",
            source_manifest=source, source_git_commit="abc",
            model_identity=model_identity(), fixed_split_identity="split",
            shared_root_uuid="root", formal_output_identity="formal",
            run_version="v2_weighted_ce",
            class_mapping={
                name: index
                for index, name in enumerate(classifier_objective.CLASS_ORDER)
            },
            training_objective=weighted_objective(),
            evaluation_scope=evaluation_scope,
        )


def resume_identity():
    return build_weighted_identity("validation_only")


class TinyWeightedCoCa(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.head = nn.Linear(2, 7)


class CoCaV2WeightedTests(unittest.TestCase):
    def test_inverse_sqrt_formula_order_and_expected_values(self):
        objective, weights = classifier_objective.build_training_objective(
            "inverse_sqrt", matched_frame()
        )
        self.assertEqual(objective["class_order"], list(EXPECTED_COUNTS))
        self.assertEqual(objective["class_counts"], EXPECTED_COUNTS)
        np.testing.assert_allclose(weights, EXPECTED_WEIGHTS, rtol=0, atol=5e-12)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=14)
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue((weights > 0).all())

    def test_c1_c4_and_smoke_limit_use_the_same_full_frame_weights(self):
        c1 = matched_frame()
        c4 = matched_frame().sample(frac=1.0, random_state=4).reset_index(drop=True)
        c1_objective, c1_weights = classifier_objective.build_training_objective(
            "inverse_sqrt", c1
        )
        c4_objective, c4_weights = classifier_objective.build_training_objective(
            "inverse_sqrt", c4
        )
        limited = c1.sample(n=64, random_state=0)
        self.assertEqual(len(limited), 64)
        self.assertEqual(sum(c1_objective["class_counts"].values()), len(c1))
        self.assertEqual(c1_objective, c4_objective)
        np.testing.assert_array_equal(c1_weights, c4_weights)

    def test_missing_zero_unknown_or_wrong_order_fails_loudly(self):
        missing = matched_frame().query("label_idx != 6")
        with self.assertRaisesRegex(ValueError, "missing or zero"):
            classifier_objective.ordered_class_counts(missing)
        unknown = pd.concat(
            [matched_frame(), pd.DataFrame({"label_idx": [7]})], ignore_index=True
        )
        with self.assertRaisesRegex(ValueError, "unknown"):
            classifier_objective.ordered_class_counts(unknown)
        reversed_counts = dict(reversed(list(EXPECTED_COUNTS.items())))
        with self.assertRaisesRegex(ValueError, "order"):
            classifier_objective.inverse_sqrt_weights(reversed_counts)
        bad_counts = dict(EXPECTED_COUNTS); bad_counts["df"] = 0
        with self.assertRaisesRegex(ValueError, "positive integer"):
            classifier_objective.inverse_sqrt_weights(bad_counts)
        bad_counts = dict(EXPECTED_COUNTS); bad_counts["df"] = 1.5
        with self.assertRaisesRegex(ValueError, "positive integer"):
            classifier_objective.inverse_sqrt_weights(bad_counts)

    def test_criterion_none_and_weighted_tensor_dtype_device(self):
        unweighted = build_criterion("none", None, torch.device("cpu"))
        self.assertIsNone(unweighted.weight)
        _, weights = classifier_objective.build_training_objective(
            "inverse_sqrt", matched_frame()
        )
        weighted = build_criterion("inverse_sqrt", weights, torch.device("cpu"))
        self.assertEqual(weighted.weight.dtype, torch.float32)
        self.assertEqual(weighted.weight.device.type, "cpu")
        np.testing.assert_allclose(weighted.weight.numpy(), weights, rtol=1e-6)

    def test_default_cli_and_identity_remain_v1_compatible(self):
        args = parse_args([])
        self.assertEqual(args.class_weighting, "none")
        self.assertEqual(args.evaluation_scope, "full")
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "train.csv"
            source.write_text("split,train\n", encoding="utf-8")
            identity = classifier_run.build_run_identity(
                run_label=None, variant="C1", seed=0, epochs=20,
                img_size=128, batch_size=32, learning_rate=3e-4,
                weight_decay=1e-4, df_target_count=585, pretrained=True,
                limit=None, candidate_manifest=None, source_split="train",
                source_manifest=source, source_git_commit="abc",
                model_identity=model_identity(), run_version="v1",
            )
        self.assertNotIn("training_objective", identity)
        self.assertNotIn("evaluation_scope", identity)
        classifier_run.require_matching_resume_identity(
            identity, copy.deepcopy(identity)
        )

    def test_build_identity_records_weighted_full_and_validation_scope(self):
        full = build_weighted_identity("full")
        validation = build_weighted_identity("validation_only")
        self.assertEqual(full["evaluation_scope"], "full")
        self.assertEqual(validation["evaluation_scope"], "validation_only")

    def test_weighted_full_scope_survives_checkpoint_and_result_round_trip(self):
        identity = build_weighted_identity("full")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint_path = root / "last.pt"
            result_path = root / "result.json"
            model = TinyWeightedCoCa()
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=3e-4)
            save_checkpoint(
                checkpoint_path, model, optimizer, 1, 0.0, [],
                argparse.Namespace(run_label=None), identity,
            )
            checkpoint = _load_trusted_checkpoint(
                checkpoint_path, torch.device("cpu")
            )
            result_path.write_text(
                json.dumps({"run_identity": identity}), encoding="utf-8"
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(
            checkpoint["run_identity"]["evaluation_scope"], "full"
        )
        self.assertEqual(result["run_identity"]["evaluation_scope"], "full")

    def test_weighted_identity_rejects_every_objective_or_scope_mismatch(self):
        saved = resume_identity()
        for field, value in (
            ("class_weighting", "none"),
            ("class_weight_formula", "other"),
            ("class_counts", {**EXPECTED_COUNTS, "df": 584}),
            ("class_weights", [1.0] * 7),
        ):
            current = copy.deepcopy(saved)
            current["training_objective"][field] = value
            with self.assertRaisesRegex(ValueError, "training_objective"):
                classifier_run.require_matching_resume_identity(saved, current)
        current = copy.deepcopy(saved); current["evaluation_scope"] = "full"
        with self.assertRaisesRegex(ValueError, "evaluation_scope"):
            classifier_run.require_matching_resume_identity(saved, current)

    def test_mismatch_guard_does_not_mutate_checkpoint_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "last.pt"
            path.write_bytes(b"immutable checkpoint fixture")
            before = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            saved = resume_identity()
            current = copy.deepcopy(saved)
            current["evaluation_scope"] = "full"
            with self.assertRaises(ValueError):
                classifier_run.require_matching_resume_identity(saved, current)
            after = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            self.assertEqual(before, after)

    def test_validation_only_never_calls_test_evaluator(self):
        with patch("ddpm_derm.train_classifier.evaluate") as evaluator:
            self.assertIsNone(
                evaluate_test_scope("validation_only", object(), None, "cpu")
            )
            evaluator.assert_not_called()
        with patch(
            "ddpm_derm.train_classifier.evaluate", return_value={"ok": True}
        ) as evaluator:
            self.assertEqual(
                evaluate_test_scope("full", "model", "loader", "cpu"),
                {"ok": True},
            )
            evaluator.assert_called_once()

    def test_aggregate_rejects_validation_only_v1_v2_and_weight_mixing(self):
        validation_only = aggregate_runs()
        validation_only[0]["evaluation_scope"] = "validation_only"
        validation_only[0]["test_metrics"] = None
        with self.assertRaisesRegex(ValueError, "full evaluation"):
            coca_run.aggregate_results(validation_only)

        inconsistent_scope = aggregate_runs()
        inconsistent_scope[-1]["run_identity"][
            "evaluation_scope"
        ] = "validation_only"
        with self.assertRaisesRegex(ValueError, "scope"):
            coca_run.aggregate_results(inconsistent_scope)

        mixed_version = aggregate_runs()
        mixed_version[-1]["run_identity"]["run_version"] = "v1"
        mixed_version[-1]["run_identity"].pop("training_objective")
        with self.assertRaisesRegex(ValueError, "run versions"):
            coca_run.aggregate_results(mixed_version)

        mixed_weights = aggregate_runs()
        mixed_weights[-1]["run_identity"]["training_objective"][
            "class_weights"
        ][0] += 0.1
        with self.assertRaisesRegex(ValueError, "training objectives"):
            coca_run.aggregate_results(mixed_weights)

        none_and_weighted = aggregate_runs()
        none_and_weighted[-1]["run_identity"].pop("training_objective")
        with self.assertRaisesRegex(ValueError, "training objectives"):
            coca_run.aggregate_results(none_and_weighted)


if __name__ == "__main__":
    unittest.main()
