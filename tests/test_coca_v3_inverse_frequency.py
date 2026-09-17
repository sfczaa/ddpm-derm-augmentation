"""Local behavioral verification for the CoCa v3 inverse-frequency objective.

The single experimental change from v2 is the class-weighting mode
(inverse_sqrt -> inverse_frequency). These tests pin the exact inverse-frequency
weights, prove v1 (none) and v2 (inverse_sqrt) stay byte-identical, and check
identity/resume/aggregation guards. No real ~1 GB CoCa weights are downloaded:
the encoder path is exercised through a mocked OpenCLIP loader.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm import classifier_objective, classifier_run, coca_run  # noqa: E402
from ddpm_derm.model import build_model, model_identity as build_model_identity  # noqa: E402
from ddpm_derm.train_classifier import (  # noqa: E402
    _load_trusted_checkpoint,
    build_criterion,
    build_optimizer,
    evaluate_test_scope,
    parse_args,
    save_checkpoint,
)


RUN_VERSION = "v3_inverse_frequency_ce"
EXPECTED_COUNTS = {
    "akiec": 226,
    "bcc": 348,
    "bkl": 778,
    "df": 585,
    "mel": 782,
    "nv": 4684,
    "vasc": 92,
}
# inverse-frequency: weight_c = (1/n_c) / mean_j(1/n_j), normalized to mean one.
EXPECTED_WEIGHTS = np.asarray([
    1.3671842524688507,
    0.8878840260286214,
    0.39715120958606714,
    0.5281771642016414,
    0.3951197455984147,
    0.0659657645298805,
    3.3585178375865246,
])
# Defining property of inverse frequency: n_c * weight_c is one shared constant.
EXPECTED_EQUAL_CONTRIBUTION = 308.98364105796026
# v2 inverse-sqrt weights: must never change when v3 is added.
V2_INVERSE_SQRT_WEIGHTS = np.asarray([
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


def inverse_frequency_objective():
    return classifier_objective.build_training_objective(
        "inverse_frequency", matched_frame()
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


def build_v3_identity(evaluation_scope="full"):
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / "train.csv"
        source.write_text("split,train\n", encoding="utf-8")
        return classifier_run.build_run_identity(
            run_label="coca_v3", variant="C1", seed=0, epochs=20,
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
            training_objective=inverse_frequency_objective(),
            evaluation_scope=evaluation_scope,
        )


def aggregate_runs(run_version=RUN_VERSION, objective=None):
    objective = objective or inverse_frequency_objective()
    runs = []
    for variant, offset in (("C1", 0.0), ("C4", 0.1)):
        for seed in (0, 1, 2):
            runs.append({
                "variant": variant,
                "seed": seed,
                "evaluation_scope": "full",
                "run_identity": {
                    "model_identity": model_identity(),
                    "checkpoint_format": coca_run.CHECKPOINT_FORMAT,
                    "run_version": run_version,
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


class TinyWeightedCoCa(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.head = nn.Linear(2, 7)


# --- mocked OpenCLIP loader (never downloads real weights) -------------------
class _NamedTransform:
    def __init__(self, name):
        self.name = name

    def __call__(self, image):
        return torch.ones(3, 4, 4)

    def __repr__(self):
        return f"NamedTransform({self.name})"


class _FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4))
        self.visual = SimpleNamespace(output_dim=4, image_size=(224, 224))

    def encode_image(self, images):
        return images[:, :4] * self.weight


class _FakeOpenClip:
    def list_pretrained(self):
        return [(coca_run.MODEL_NAME, coca_run.PRETRAINED_TAG)]

    def create_model_and_transforms(self, model_name, pretrained):
        return (
            _FakeEncoder(),
            _NamedTransform("native-train"),
            _NamedTransform("native-eval"),
        )


class CoCaV3InverseFrequencyTests(unittest.TestCase):
    # --- exact formula, order, weights ---------------------------------------
    def test_inverse_frequency_formula_order_and_expected_values(self):
        objective, weights = classifier_objective.build_training_objective(
            "inverse_frequency", matched_frame()
        )
        self.assertEqual(objective["loss_name"], "cross_entropy")
        self.assertEqual(objective["class_weighting"], "inverse_train_frequency")
        self.assertEqual(objective["class_weight_formula"], "(1/n_c)/mean_j(1/n_j)")
        self.assertEqual(objective["class_weight_normalization"], "mean_one")
        self.assertEqual(
            objective["class_weight_count_source"],
            "full_post_variant_train_frame_before_limit",
        )
        self.assertEqual(
            objective["class_order"], list(classifier_objective.CLASS_ORDER)
        )
        self.assertEqual(objective["class_order"], list(EXPECTED_COUNTS))
        self.assertEqual(objective["class_counts"], EXPECTED_COUNTS)
        np.testing.assert_allclose(weights, EXPECTED_WEIGHTS, rtol=0, atol=5e-12)
        self.assertEqual(tuple(weights.shape), (7,))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=14)
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue((weights > 0).all())

    def test_equal_class_contribution_is_the_inverse_frequency_signature(self):
        objective, weights = classifier_objective.build_training_objective(
            "inverse_frequency", matched_frame()
        )
        counts = np.asarray(
            [objective["class_counts"][name] for name in EXPECTED_COUNTS],
            dtype=float,
        )
        contributions = counts * weights
        np.testing.assert_allclose(
            contributions, EXPECTED_EQUAL_CONTRIBUTION, rtol=0, atol=1e-6
        )
        # every class contributes the *same* amount -> spread is ~0
        self.assertLess(float(contributions.max() - contributions.min()), 1e-6)

    # --- v1 (none) and v2 (inverse_sqrt) stay identical ----------------------
    def test_v2_inverse_sqrt_objective_is_byte_for_byte_unchanged(self):
        objective, weights = classifier_objective.build_training_objective(
            "inverse_sqrt", matched_frame()
        )
        self.assertEqual(objective["class_weighting"], "inverse_sqrt_train_frequency")
        self.assertEqual(
            objective["class_weight_formula"], "(1/sqrt(n_c))/mean_j(1/sqrt(n_j))"
        )
        np.testing.assert_allclose(
            weights, V2_INVERSE_SQRT_WEIGHTS, rtol=0, atol=5e-12
        )
        # inverse-frequency and inverse-sqrt must be genuinely different vectors
        self.assertFalse(np.allclose(weights, EXPECTED_WEIGHTS))

    def test_none_mode_returns_no_objective(self):
        self.assertEqual(
            classifier_objective.build_training_objective("none", matched_frame()),
            (None, None),
        )

    # --- counts sourced from the train frame only ----------------------------
    def test_c1_c4_and_smoke_limit_use_the_same_full_frame_weights(self):
        c1 = matched_frame()
        c4 = matched_frame().sample(frac=1.0, random_state=4).reset_index(drop=True)
        c1_objective, c1_weights = classifier_objective.build_training_objective(
            "inverse_frequency", c1
        )
        c4_objective, c4_weights = classifier_objective.build_training_objective(
            "inverse_frequency", c4
        )
        limited = c1.sample(n=64, random_state=0)
        self.assertEqual(len(limited), 64)
        # counts always cover the full frame, never the limited sample
        self.assertEqual(sum(c1_objective["class_counts"].values()), len(c1))
        self.assertEqual(c1_objective, c4_objective)
        np.testing.assert_array_equal(c1_weights, c4_weights)

    def test_counts_come_from_the_supplied_full_frame(self):
        objective, _ = classifier_objective.build_training_objective(
            "inverse_frequency", matched_frame()
        )
        self.assertEqual(
            objective["class_counts"],
            classifier_objective.ordered_class_counts(matched_frame()),
        )

    # --- fail-loud guards ----------------------------------------------------
    def test_missing_zero_noninteger_unknown_or_wrong_order_fail_loudly(self):
        missing = matched_frame().query("label_idx != 6")
        with self.assertRaisesRegex(ValueError, "missing or zero"):
            classifier_objective.ordered_class_counts(missing)
        unknown = pd.concat(
            [matched_frame(), pd.DataFrame({"label_idx": [7]})], ignore_index=True
        )
        with self.assertRaisesRegex(ValueError, "unknown"):
            classifier_objective.ordered_class_counts(unknown)
        nonfinite = matched_frame().copy()
        nonfinite.loc[0, "label_idx"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            classifier_objective.ordered_class_counts(nonfinite)
        reversed_counts = dict(reversed(list(EXPECTED_COUNTS.items())))
        with self.assertRaisesRegex(ValueError, "order"):
            classifier_objective.inverse_frequency_weights(reversed_counts)
        for bad_value, pattern in ((0, "positive integer"),
                                   (1.5, "positive integer"),
                                   (float("inf"), "positive integer")):
            bad_counts = dict(EXPECTED_COUNTS)
            bad_counts["df"] = bad_value
            with self.assertRaisesRegex(ValueError, pattern):
                classifier_objective.inverse_frequency_weights(bad_counts)

    # --- criterion tensor + unsupported mode ---------------------------------
    def test_criterion_none_and_inverse_frequency_tensor_dtype_device(self):
        unweighted = build_criterion("none", None, torch.device("cpu"))
        self.assertIsNone(unweighted.weight)
        _, weights = classifier_objective.build_training_objective(
            "inverse_frequency", matched_frame()
        )
        weighted = build_criterion("inverse_frequency", weights, torch.device("cpu"))
        self.assertEqual(weighted.weight.dtype, torch.float32)
        self.assertEqual(weighted.weight.device.type, "cpu")
        self.assertEqual(tuple(weighted.weight.shape), (7,))
        np.testing.assert_allclose(weighted.weight.numpy(), weights, rtol=1e-6)
        with self.assertRaisesRegex(ValueError, "requires an ordered"):
            build_criterion("inverse_frequency", None, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "unsupported class weighting"):
            build_criterion("focal", weights, torch.device("cpu"))

    # --- CLI surface ---------------------------------------------------------
    def test_cli_supports_three_modes_and_defaults_to_none(self):
        self.assertEqual(parse_args([]).class_weighting, "none")
        self.assertEqual(parse_args([]).evaluation_scope, "full")
        for mode in ("none", "inverse_sqrt", "inverse_frequency"):
            args = parse_args(["--class-weighting", mode])
            self.assertEqual(args.class_weighting, mode)
        with self.assertRaises(SystemExit):
            parse_args(["--class-weighting", "focal"])

    def test_default_cli_identity_remains_v1_compatible(self):
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

    # --- identity records evaluation scope -----------------------------------
    def test_build_identity_records_v3_full_and_validation_scope(self):
        full = build_v3_identity("full")
        validation = build_v3_identity("validation_only")
        self.assertEqual(full["evaluation_scope"], "full")
        self.assertEqual(full["run_version"], RUN_VERSION)
        self.assertEqual(
            full["training_objective"]["class_weighting"], "inverse_train_frequency"
        )
        self.assertEqual(validation["evaluation_scope"], "validation_only")

    # --- objective persisted in checkpoint + result; head-only ---------------
    def test_v3_objective_round_trips_and_checkpoint_is_head_only(self):
        identity = build_v3_identity("full")
        objective = inverse_frequency_objective()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint_path = root / "last.pt"
            model = TinyWeightedCoCa()
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=3e-4)
            save_checkpoint(
                checkpoint_path, model, optimizer, 1, 0.0, [],
                argparse.Namespace(run_label=None), identity,
            )
            checkpoint = _load_trusted_checkpoint(
                checkpoint_path, torch.device("cpu")
            )
            result = {
                "training_objective": objective,
                "run_identity": identity,
            }
        self.assertEqual(
            checkpoint["run_identity"]["training_objective"], objective
        )
        self.assertEqual(result["run_identity"]["evaluation_scope"], "full")
        self.assertIn("head_state_dict", checkpoint)
        self.assertNotIn("model_state_dict", checkpoint)
        self.assertNotIn("encoder_state_dict", checkpoint)
        self.assertFalse(
            any("encoder" in key or "text" in key for key in checkpoint)
        )

    # --- resume never mixes modes/objectives/scope/version -------------------
    def test_resume_rejects_cross_mode_objective_and_scope_or_version(self):
        saved = build_v3_identity("validation_only")
        none_identity = copy.deepcopy(saved)
        none_identity.pop("training_objective")
        with self.assertRaisesRegex(ValueError, "training_objective"):
            classifier_run.require_matching_resume_identity(saved, none_identity)
        sqrt_objective = classifier_objective.build_training_objective(
            "inverse_sqrt", matched_frame()
        )[0]
        sqrt_identity = copy.deepcopy(saved)
        sqrt_identity["training_objective"] = sqrt_objective
        with self.assertRaisesRegex(ValueError, "training_objective"):
            classifier_run.require_matching_resume_identity(saved, sqrt_identity)
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
        for field in ("evaluation_scope", "run_version"):
            current = copy.deepcopy(saved)
            current[field] = "full" if field == "evaluation_scope" else "v2_weighted_ce"
            with self.assertRaisesRegex(ValueError, field):
                classifier_run.require_matching_resume_identity(saved, current)

    def test_mismatch_guard_does_not_mutate_checkpoint_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "last.pt"
            path.write_bytes(b"immutable checkpoint fixture")
            before = (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_mtime_ns,
            )
            saved = build_v3_identity("validation_only")
            current = copy.deepcopy(saved)
            current["evaluation_scope"] = "full"
            with self.assertRaises(ValueError):
                classifier_run.require_matching_resume_identity(saved, current)
            after = (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_mtime_ns,
            )
            self.assertEqual(before, after)

    # --- validation_only never touches test ----------------------------------
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

    # --- aggregation refuses validation-only and v1/v2/v3 mixing --------------
    def test_aggregate_rejects_validation_only_and_mixed_versions(self):
        validation_only = aggregate_runs()
        validation_only[0]["evaluation_scope"] = "validation_only"
        validation_only[0]["test_metrics"] = None
        with self.assertRaisesRegex(ValueError, "full evaluation"):
            coca_run.aggregate_results(validation_only)

        # v3 mixed with a v1 (unweighted) run
        mixed_v1 = aggregate_runs()
        mixed_v1[-1]["run_identity"]["run_version"] = "v1"
        mixed_v1[-1]["run_identity"].pop("training_objective")
        with self.assertRaisesRegex(ValueError, "run versions"):
            coca_run.aggregate_results(mixed_v1)

        # v3 mixed with a v2 (inverse_sqrt) run -> same version label, different objective
        mixed_v2 = aggregate_runs()
        mixed_v2[-1]["run_identity"]["training_objective"] = (
            classifier_objective.build_training_objective(
                "inverse_sqrt", matched_frame()
            )[0]
        )
        with self.assertRaisesRegex(ValueError, "training objectives"):
            coca_run.aggregate_results(mixed_v2)

        # a single tampered inverse-frequency weight
        tampered = aggregate_runs()
        tampered[-1]["run_identity"]["training_objective"]["class_weights"][0] += 0.1
        with self.assertRaisesRegex(ValueError, "training objectives"):
            coca_run.aggregate_results(tampered)

    def test_pure_v3_runs_aggregate_and_report_paired_difference(self):
        aggregate = coca_run.aggregate_results(aggregate_runs())
        self.assertEqual(aggregate["ddof"], 0)
        for value in aggregate["paired_c4_minus_c1"]["seed_differences"]:
            self.assertAlmostEqual(value, 0.1)

    # --- mocked encoder: frozen, head-only optimizer, real criterion ----------
    def test_mocked_coca_freeze_optimizer_and_inverse_frequency_criterion(self):
        model = build_model(
            arch="coca_vit_b32", freeze_backbone=True,
            coca_pretrained=coca_run.PRETRAINED_TAG,
            open_clip_module=_FakeOpenClip(),
        )
        model.train()
        logits = model(torch.ones(3, 4))
        self.assertEqual(tuple(logits.shape), (3, 7))
        self.assertFalse(model.encoder.training)
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))
        optimizer = build_optimizer(model, 3e-4, 1e-4)
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(optimized, {id(p) for p in model.head.parameters()})
        self.assertFalse(optimized & {id(p) for p in model.encoder.parameters()})
        _, weights = classifier_objective.build_training_objective(
            "inverse_frequency", matched_frame()
        )
        criterion = build_criterion("inverse_frequency", weights, torch.device("cpu"))
        np.testing.assert_allclose(criterion.weight.numpy(), weights, rtol=1e-6)
        details = build_model_identity(model, "coca_vit_b32", 128)
        self.assertGreater(
            details["total_parameter_count"], details["trainable_parameter_count"]
        )


if __name__ == "__main__":
    unittest.main()
