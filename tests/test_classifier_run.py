"""Torch-free tests for classifier identity, hashing, and resume guards."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm import classifier_run  # noqa: E402


MANIFEST_BYTES = (
    b"image_path,label_idx,dx,source\n"
    b"images/a.png,3,df,synthetic\n"
)
MANIFEST_SHA256 = "e2cceeec9bef2a407f178cff5069e52719829ea6f504742cfbb94a8ed661090f"


class ClassifierRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "train.csv"
        self.source.write_bytes(b"split,train\n")
        self.candidate = self.root / "synthetic_df.csv"
        self.candidate.write_bytes(MANIFEST_BYTES)

    def tearDown(self):
        self.temp.cleanup()

    def identity(self, **overrides):
        values = {
            "run_label": "c4_sqrt_balanced_v1",
            "variant": "C4",
            "seed": 0,
            "epochs": 20,
            "img_size": 128,
            "batch_size": 32,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "df_target_count": 585,
            "pretrained": True,
            "limit": None,
            "candidate_manifest": self.candidate,
            "source_split": "train",
            "source_manifest": self.source,
            "source_git_commit": "5274433",
            "model_identity": {
                "arch": "coca_vit_b32",
                "model_name": "coca_ViT-B-32",
                "pretrained_tag": "laion2b_s13b_b90k",
                "freeze_mode": "frozen_image_encoder_linear_head",
                "preprocessing_identity": {"train": "native", "eval": "native"},
                "input_resolution": [224, 224],
                "total_parameter_count": 100,
                "trainable_parameter_count": 35,
                "open_clip_torch_version": "3.3.0",
                "torch_version": "test",
            },
            "fixed_split_identity": "fixed-lesion-split",
            "shared_root_uuid": "shared-root",
            "formal_output_identity": "coca-v1-formal",
            "run_version": "v1",
            "class_mapping": {"df": 3},
        }
        values.update(overrides)
        return classifier_run.build_run_identity(**values)

    def test_natural_identity_keeps_candidate_hash_empty(self):
        identity = self.identity(
            run_label=None,
            variant="C0",
            candidate_manifest=None,
        )
        self.assertEqual(identity["variant"], "C0")
        self.assertIsNone(identity["candidate_manifest_sha256"])

    def test_candidate_hash_is_fixed_and_portable(self):
        self.assertEqual(classifier_run.sha256_file(self.candidate), MANIFEST_SHA256)
        moved = self.root / "different_mount" / "synthetic_df.csv"
        moved.parent.mkdir()
        moved.write_bytes(MANIFEST_BYTES)
        self.assertEqual(classifier_run.sha256_file(moved), MANIFEST_SHA256)

    def test_identical_identity_allows_resume(self):
        identity = self.identity()
        classifier_run.require_matching_resume_identity(identity, copy.deepcopy(identity))

    def test_variant_and_seed_mismatches_are_rejected(self):
        saved = self.identity()
        for key, value in (("variant", "C1"), ("seed", 1)):
            with self.subTest(key=key):
                current = copy.deepcopy(saved)
                current[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    classifier_run.require_matching_resume_identity(saved, current)

    def test_config_and_candidate_hash_mismatches_are_rejected(self):
        saved = self.identity()
        current = copy.deepcopy(saved)
        current["fixed_config"]["batch_size"] = 16
        with self.assertRaisesRegex(ValueError, "fixed_config"):
            classifier_run.require_matching_resume_identity(saved, current)
        current = copy.deepcopy(saved)
        current["candidate_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "candidate_manifest_sha256"):
            classifier_run.require_matching_resume_identity(saved, current)

    def test_source_split_or_manifest_change_is_rejected(self):
        saved = self.identity()
        for key, value in (
            ("source_split", "val"),
            ("source_manifest_sha256", "1" * 64),
        ):
            with self.subTest(key=key):
                current = copy.deepcopy(saved)
                current[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    classifier_run.require_matching_resume_identity(saved, current)

    def test_coca_model_and_shared_root_mismatches_are_rejected(self):
        saved = self.identity()
        changes = (
            ("arch", "resnet18"),
            ("pretrained_tag", "other"),
            ("freeze_mode", "trainable"),
            ("preprocessing_identity", {"train": "other", "eval": "other"}),
        )
        for key, value in changes:
            with self.subTest(key=key):
                current = copy.deepcopy(saved)
                current["model_identity"][key] = value
                with self.assertRaisesRegex(ValueError, "model_identity"):
                    classifier_run.require_matching_resume_identity(saved, current)
        current = copy.deepcopy(saved)
        current["shared_root_uuid"] = "other"
        with self.assertRaisesRegex(ValueError, "shared_root_uuid"):
            classifier_run.require_matching_resume_identity(saved, current)

    def test_operational_metadata_does_not_affect_resume(self):
        saved = self.identity()
        current = copy.deepcopy(saved)
        saved["account_label"] = "A"
        saved["hostname"] = "host-a"
        current["account_label"] = "B"
        current["hostname"] = "host-b"
        classifier_run.require_matching_resume_identity(saved, current)

    def test_checkpoint_format_is_identity_and_mismatch_is_rejected(self):
        saved = self.identity()
        self.assertEqual(
            saved["checkpoint_format"],
            classifier_run.FROZEN_COCA_CHECKPOINT_FORMAT,
        )
        current = copy.deepcopy(saved)
        current["checkpoint_format"] = classifier_run.FULL_MODEL_CHECKPOINT_FORMAT
        with self.assertRaisesRegex(ValueError, "checkpoint_format"):
            classifier_run.require_matching_resume_identity(saved, current)

    def test_training_identity_is_train_only(self):
        with self.assertRaisesRegex(ValueError, "source_split"):
            self.identity(source_split="val")

    def test_exploratory_output_must_be_isolated(self):
        exploratory = self.root / "outputs" / "exploratory_balanced_ddpm"
        allowed = exploratory / "version" / "downstream_classifier" / "run"
        self.assertEqual(
            classifier_run.require_isolated_output_dir(allowed, exploratory),
            allowed.resolve(),
        )
        for rejected in (exploratory, self.root / "outputs" / "classifier"):
            with self.subTest(path=rejected):
                with self.assertRaisesRegex(ValueError, "must be a child"):
                    classifier_run.require_isolated_output_dir(rejected, exploratory)

    def test_legacy_c0_c1_config_guard_preserves_matching_resume(self):
        saved = {
            "variant": "C1",
            "seed": 0,
            "epochs": 20,
            "batch_size": 32,
            "img_size": 128,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "df_target_count": 585,
            "no_pretrained": False,
            "limit": None,
        }
        classifier_run.require_matching_legacy_config(saved, copy.deepcopy(saved))
        current = copy.deepcopy(saved)
        current["lr"] = 1e-3
        with self.assertRaisesRegex(ValueError, "lr"):
            classifier_run.require_matching_legacy_config(saved, current)


if __name__ == "__main__":
    unittest.main()
