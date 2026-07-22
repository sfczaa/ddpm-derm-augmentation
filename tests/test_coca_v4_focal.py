"""Local verification for the CoCa v4 focal inverse-frequency objective."""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm import classifier_objective, classifier_run, coca_run  # noqa: E402
from ddpm_derm.train_classifier import (  # noqa: E402
    FocalCrossEntropyLoss,
    _load_trusted_checkpoint,
    build_criterion,
    parse_args,
    save_checkpoint,
)


RUN_VERSION = "v4_focal_inverse_frequency"
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
    1.3671842524688507,
    0.8878840260286214,
    0.39715120958606714,
    0.5281771642016414,
    0.3951197455984147,
    0.0659657645298805,
    3.3585178375865246,
])


def matched_frame():
    labels = []
    for class_idx, name in enumerate(classifier_objective.CLASS_ORDER):
        labels.extend([class_idx] * EXPECTED_COUNTS[name])
    return pd.DataFrame({"label_idx": labels})


def focal_objective():
    return classifier_objective.build_training_objective(
        "inverse_frequency",
        matched_frame(),
        loss_name="focal_cross_entropy",
        focal_gamma=2.0,
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


def build_identity(evaluation_scope="validation_only"):
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / "train.csv"
        source.write_text("split,train\n", encoding="utf-8")
        return classifier_run.build_run_identity(
            run_label="coca_v4", variant="C1", seed=0, epochs=5,
            img_size=128, batch_size=32, learning_rate=3e-4,
            weight_decay=1e-4, df_target_count=585, pretrained=True,
            limit=None, candidate_manifest=None, source_split="train",
            source_manifest=source, source_git_commit="abc",
            model_identity=model_identity(), fixed_split_identity="split",
            shared_root_uuid="root", formal_output_identity="formal",
            run_version=RUN_VERSION,
            class_mapping={
                name: index
                for index, name in enumerate(classifier_objective.CLASS_ORDER)
            },
            training_objective=focal_objective(),
            evaluation_scope=evaluation_scope,
        )


def aggregate_runs():
    runs = []
    for variant in ("C1", "C4"):
        for seed in (0, 1, 2):
            runs.append({
                "variant": variant,
                "seed": seed,
                "evaluation_scope": "full",
                "run_identity": {
                    "model_identity": model_identity(),
                    "checkpoint_format": coca_run.CHECKPOINT_FORMAT,
                    "run_version": RUN_VERSION,
                    "training_objective": copy.deepcopy(focal_objective()),
                    "evaluation_scope": "full",
                },
                "test_metrics": {
                    "target_f1": 0.5,
                    "macro_f1": 0.4,
                    "target_recall": 0.3,
                    "per_class_recall": {"df": 0.3},
                },
            })
    return runs


class TinyFrozenClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 3, bias=False)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.head = nn.Linear(3, 7)

    def forward(self, inputs):
        with torch.no_grad():
            features = self.encoder(inputs)
        return self.head(features)


class CoCaV4FocalTests(unittest.TestCase):
    def test_default_cross_entropy_cli_and_identity_are_unchanged(self):
        args = parse_args([])
        self.assertEqual(args.loss_name, "cross_entropy")
        self.assertIsNone(args.focal_gamma)
        self.assertEqual(args.class_weighting, "none")
        self.assertEqual(
            classifier_objective.build_training_objective("none", matched_frame()),
            (None, None),
        )
        v3, _ = classifier_objective.build_training_objective(
            "inverse_frequency", matched_frame()
        )
        self.assertEqual(v3["loss_name"], "cross_entropy")
        self.assertNotIn("focal_gamma", v3)
        self.assertNotIn("focal_formula", v3)
        self.assertNotIn("focal_reduction", v3)

    def test_cli_accepts_only_the_registered_focal_combination(self):
        args = parse_args([
            "--loss-name", "focal_cross_entropy",
            "--class-weighting", "inverse_frequency",
            "--focal-gamma", "2.0",
        ])
        self.assertEqual(args.focal_gamma, 2.0)
        invalid = (
            ["--loss-name", "unsupported"],
            ["--focal-gamma", "2"],
            ["--loss-name", "focal_cross_entropy"],
            ["--loss-name", "focal_cross_entropy", "--focal-gamma", "2"],
            ["--loss-name", "focal_cross_entropy", "--class-weighting", "inverse_sqrt", "--focal-gamma", "2"],
            ["--loss-name", "focal_cross_entropy", "--class-weighting", "inverse_frequency", "--focal-gamma", "-1"],
            ["--loss-name", "focal_cross_entropy", "--class-weighting", "inverse_frequency", "--focal-gamma", "nan"],
            ["--loss-name", "focal_cross_entropy", "--class-weighting", "inverse_frequency", "--focal-gamma", "inf"],
        )
        for argv in invalid:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_objective_records_exact_focal_identity_and_weights(self):
        objective = focal_objective()
        self.assertEqual(objective["loss_name"], "focal_cross_entropy")
        self.assertEqual(objective["focal_gamma"], 2.0)
        self.assertEqual(objective["focal_formula"], classifier_objective.FOCAL_FORMULA)
        self.assertEqual(objective["focal_reduction"], "weighted_mean_by_target_alpha")
        self.assertEqual(objective["class_weighting"], "inverse_train_frequency")
        self.assertEqual(objective["class_counts"], EXPECTED_COUNTS)
        np.testing.assert_allclose(
            objective["class_weights"], EXPECTED_WEIGHTS, rtol=0, atol=5e-12
        )

    def test_gamma_two_matches_the_registered_formula_and_denominator(self):
        logits = torch.tensor(
            [
                [1.2, -0.4, 0.3, 0.1, -0.2, 0.5, -0.7],
                [-0.5, 0.7, 1.1, -0.3, 0.2, 0.4, -0.9],
                [0.2, 0.1, -0.8, 0.6, -0.1, 0.3, 0.9],
            ],
            dtype=torch.float64,
        )
        targets = torch.tensor([0, 2, 6])
        alpha = torch.tensor(
            [0.5, 1.25, 3.0, 0.7, 1.1, 2.2, 4.0], dtype=torch.float64
        )
        criterion = FocalCrossEntropyLoss(alpha, 2.0, torch.device("cpu")).double()
        log_probs = torch.log_softmax(logits, dim=1)
        log_pt = log_probs[torch.arange(len(targets)), targets]
        alpha_t = alpha[targets]
        expected = (-alpha_t * (1 - log_pt.exp()).pow(2) * log_pt).sum() / alpha_t.sum()
        torch.testing.assert_close(criterion(logits, targets), expected, rtol=0, atol=1e-15)

    def test_gamma_zero_matches_weighted_ce_for_unbalanced_targets(self):
        torch.manual_seed(7)
        logits = torch.randn(6, 7, dtype=torch.float32, requires_grad=True)
        targets = torch.tensor([0, 0, 0, 3, 6, 6])
        alpha = torch.tensor([0.2, 0.5, 0.7, 1.3, 1.9, 2.2, 4.1])
        focal = FocalCrossEntropyLoss(alpha, 0.0, torch.device("cpu"))
        expected = nn.CrossEntropyLoss(weight=focal.weight)(logits, targets)
        torch.testing.assert_close(focal(logits, targets), expected, rtol=0, atol=0)

    def test_alpha_scale_invariance_uses_target_alpha_weighted_mean(self):
        logits = torch.tensor([
            [0.2, 0.9, -0.4, 0.1, -0.5, 0.3, 0.7],
            [1.2, -0.2, 0.1, 0.4, -0.8, 0.6, -0.1],
        ])
        targets = torch.tensor([1, 6])
        alpha = torch.tensor([0.4, 2.0, 5.0, 0.8, 1.3, 2.1, 3.4])
        first = FocalCrossEntropyLoss(alpha, 2.0, "cpu")(logits, targets)
        second = FocalCrossEntropyLoss(alpha * 17, 2.0, "cpu")(logits, targets)
        torch.testing.assert_close(first, second)

    def test_focal_loss_is_finite_scalar_and_head_gradient_only(self):
        torch.manual_seed(3)
        model = TinyFrozenClassifier()
        inputs = torch.randn(5, 3)
        targets = torch.tensor([0, 1, 3, 3, 6])
        criterion = build_criterion(
            "inverse_frequency", EXPECTED_WEIGHTS, torch.device("cpu"),
            loss_name="focal_cross_entropy", focal_gamma=2.0,
        )
        loss = criterion(model(inputs), targets)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.encoder.parameters()))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.head.parameters()))

    def test_criterion_alpha_device_dtype_and_validation(self):
        criterion = build_criterion(
            "inverse_frequency", EXPECTED_WEIGHTS, torch.device("cpu"),
            loss_name="focal_cross_entropy", focal_gamma=2.0,
        )
        self.assertEqual(criterion.weight.device.type, "cpu")
        self.assertEqual(criterion.weight.dtype, torch.float32)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            FocalCrossEntropyLoss([1, 1, 1, 1, 1, 1, float("nan")], 2, "cpu")
        with self.assertRaisesRegex(ValueError, "shape"):
            FocalCrossEntropyLoss([1, 2], 2, "cpu")

    def test_full_frame_counts_are_unchanged_by_limit_or_variant_order(self):
        c1 = matched_frame()
        c4 = c1.sample(frac=1, random_state=4).reset_index(drop=True)
        c1_objective = classifier_objective.build_training_objective(
            "inverse_frequency", c1,
            loss_name="focal_cross_entropy", focal_gamma=2.0,
        )[0]
        c4_objective = classifier_objective.build_training_objective(
            "inverse_frequency", c4,
            loss_name="focal_cross_entropy", focal_gamma=2.0,
        )[0]
        self.assertEqual(c1_objective, c4_objective)
        self.assertEqual(sum(c1_objective["class_counts"].values()), 7495)
        self.assertEqual(len(c1.sample(n=64, random_state=0)), 64)

    def test_focal_objective_round_trips_through_head_only_checkpoint(self):
        identity = build_identity("full")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "last.pt"
            model = TinyFrozenClassifier()
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=3e-4)
            save_checkpoint(
                path, model, optimizer, 1, 0.0, [],
                argparse.Namespace(run_label=None), identity,
            )
            checkpoint = _load_trusted_checkpoint(path, torch.device("cpu"))
        self.assertEqual(checkpoint["run_identity"]["training_objective"], focal_objective())
        self.assertIn("head_state_dict", checkpoint)
        self.assertNotIn("model_state_dict", checkpoint)
        self.assertNotIn("encoder_state_dict", checkpoint)

    def test_resume_rejects_every_focal_identity_mismatch_without_mutation(self):
        saved = build_identity()
        changes = (
            ("loss_name", "cross_entropy"),
            ("focal_gamma", 1.0),
            ("focal_formula", "other"),
            ("focal_reduction", "batch_mean"),
            ("class_counts", {**EXPECTED_COUNTS, "df": 584}),
            ("class_weights", [1.0] * 7),
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "last.pt"
            path.write_bytes(b"immutable")
            before = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for field, value in changes:
                current = copy.deepcopy(saved)
                current["training_objective"][field] = value
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "training_objective"):
                    classifier_run.require_matching_resume_identity(saved, current)
            current = copy.deepcopy(saved)
            current["evaluation_scope"] = "full"
            with self.assertRaisesRegex(ValueError, "evaluation_scope"):
                classifier_run.require_matching_resume_identity(saved, current)
            after = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        self.assertEqual(before, after)

    def test_aggregate_rejects_validation_only_version_loss_gamma_and_reduction_mixing(self):
        validation = aggregate_runs()
        validation[0]["evaluation_scope"] = "validation_only"
        validation[0]["test_metrics"] = None
        with self.assertRaisesRegex(ValueError, "full evaluation"):
            coca_run.aggregate_results(validation)
        for field, value in (
            ("run_version", "v3_inverse_frequency_ce"),
            ("loss_name", "cross_entropy"),
            ("focal_gamma", 1.0),
            ("focal_formula", "other"),
            ("focal_reduction", "batch_mean"),
        ):
            runs = aggregate_runs()
            if field == "run_version":
                runs[-1]["run_identity"][field] = value
                pattern = "run versions"
            else:
                runs[-1]["run_identity"]["training_objective"][field] = value
                pattern = "training objectives"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, pattern):
                coca_run.aggregate_results(runs)


if __name__ == "__main__":
    unittest.main()
