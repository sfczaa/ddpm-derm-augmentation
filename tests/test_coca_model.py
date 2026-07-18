"""Mocked local verification for the frozen CoCa classifier path."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from PIL import Image
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm import dataset  # noqa: E402
from ddpm_derm.model import (  # noqa: E402
    COCA_MODEL_NAME,
    COCA_PRETRAINED,
    build_model,
    model_identity,
)
from ddpm_derm.train_classifier import (  # noqa: E402
    build_optimizer,
    classifier_output_paths,
)


class NamedTransform:
    def __init__(self, name):
        self.name = name

    def __call__(self, image):
        return torch.ones(3, 4, 4)

    def __repr__(self):
        return f"NamedTransform({self.name})"


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4))
        self.visual = SimpleNamespace(output_dim=4, image_size=(224, 224))
        self.grad_enabled = None
        self.training_during_forward = None

    def encode_image(self, images):
        self.grad_enabled = torch.is_grad_enabled()
        self.training_during_forward = self.training
        return images[:, :4] * self.weight


class FakeOpenClip:
    def __init__(self):
        self.encoder = None

    def list_pretrained(self):
        return [(COCA_MODEL_NAME, COCA_PRETRAINED)]

    def create_model_and_transforms(self, model_name, pretrained):
        self.encoder = FakeEncoder()
        return self.encoder, NamedTransform("native-train"), NamedTransform("native-eval")


class CoCaModelTests(unittest.TestCase):
    def build_coca(self):
        loader = FakeOpenClip()
        model = build_model(
            arch="coca_vit_b32",
            freeze_backbone=True,
            coca_pretrained=COCA_PRETRAINED,
            open_clip_module=loader,
        )
        return model, loader

    def test_resnet_default_behavior_is_unchanged(self):
        model = build_model(pretrained=False)
        self.assertEqual(model.fc.out_features, 7)

    def test_unsupported_arch_and_missing_freeze_fail_loud(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            build_model(arch="unknown", pretrained=False)
        with self.assertRaisesRegex(ValueError, "freeze"):
            build_model(arch="coca_vit_b32", open_clip_module=FakeOpenClip())
        invalid = FakeOpenClip()
        invalid.list_pretrained = lambda: []
        with self.assertRaisesRegex(ValueError, "model/tag"):
            build_model(
                arch="coca_vit_b32", freeze_backbone=True,
                open_clip_module=invalid,
            )

    def test_forward_freeze_train_mode_and_no_grad(self):
        model, loader = self.build_coca()
        model.train()
        logits = model(torch.ones(3, 4))
        self.assertEqual(tuple(logits.shape), (3, 7))
        self.assertFalse(loader.encoder.training)
        self.assertFalse(loader.encoder.training_during_forward)
        self.assertFalse(loader.encoder.grad_enabled)
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))

    def test_optimizer_contains_only_linear_head(self):
        model, _ = self.build_coca()
        optimizer = build_optimizer(model, 3e-4, 1e-4)
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(optimized, {id(p) for p in model.head.parameters()})
        self.assertFalse(optimized & {id(p) for p in model.encoder.parameters()})

    def test_native_transforms_and_actual_resolution_are_recorded(self):
        model, _ = self.build_coca()
        identity = model_identity(model, "coca_vit_b32", 128)
        self.assertEqual(identity["input_resolution"], [224, 224])
        self.assertIn("native-train", identity["preprocessing_identity"]["train"])
        self.assertIn("native-eval", identity["preprocessing_identity"]["eval"])
        self.assertGreater(identity["total_parameter_count"], identity["trainable_parameter_count"])

    def test_c1_c4_preprocessing_identity_is_identical(self):
        c1, _ = self.build_coca()
        c4, _ = self.build_coca()
        self.assertEqual(
            model_identity(c1, "coca_vit_b32", 128)["preprocessing_identity"],
            model_identity(c4, "coca_vit_b32", 128)["preprocessing_identity"],
        )

    def test_coca_outputs_are_arch_isolated_without_changing_resnet_paths(self):
        base = Path("outputs") / "classifier"
        resnet_ckpt, resnet_results = classifier_output_paths(base, "resnet18", "C1", 0)
        coca_ckpt, coca_results = classifier_output_paths(base, "coca_vit_b32", "C1", 0)
        self.assertEqual(resnet_ckpt, base / "checkpoints" / "C1_seed0")
        self.assertEqual(resnet_results, base / "results")
        self.assertEqual(coca_ckpt, base / "checkpoints" / "coca_vit_b32" / "C1_seed0")
        self.assertEqual(coca_results, base / "results" / "coca_vit_b32")

    def test_explicit_transform_is_passed_to_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "image.png"
            Image.new("RGB", (8, 8)).save(image_path)
            frame = pd.DataFrame([{"image_path": str(image_path), "label_idx": 3}])
            transform = NamedTransform("explicit-native")
            loader = dataset.build_dataloader(
                frame, batch_size=1, train=False, num_workers=0, transform=transform
            )
            self.assertIs(loader.dataset.transform, transform)

    def test_omitted_transform_keeps_legacy_builder(self):
        frame = pd.DataFrame([{"image_path": "unused", "label_idx": 0}])
        sentinel = NamedTransform("legacy")
        with patch("ddpm_derm.dataset.build_transforms", return_value=sentinel):
            loader = dataset.build_dataloader(frame, train=False, num_workers=0)
        self.assertIs(loader.dataset.transform, sentinel)


if __name__ == "__main__":
    unittest.main()
