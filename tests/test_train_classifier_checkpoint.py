"""Regression test for full-state classifier checkpoint loading."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm.train_classifier import (  # noqa: E402
    _ensure_durable_directory,
    _load_trusted_checkpoint,
    restore_checkpoint_state,
    save_checkpoint,
    validate_checkpoint_payload,
)
from ddpm_derm import classifier_run, coca_run  # noqa: E402


class TinyCoCa(nn.Module):
    def __init__(self, encoder_value=1.0):
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        nn.init.constant_(self.encoder.weight, encoder_value)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()
        self.head = nn.Linear(2, 3)


def identity(arch="coca_vit_b32"):
    return {
        "checkpoint_format": (
            classifier_run.FROZEN_COCA_CHECKPOINT_FORMAT
            if arch == "coca_vit_b32"
            else classifier_run.FULL_MODEL_CHECKPOINT_FORMAT
        ),
        "model_identity": {"arch": arch},
    }


class ClassifierCheckpointTests(unittest.TestCase):
    def test_trainer_checkpoint_loads_numpy_rng_state(self):
        """PyTorch 2.6+ safe-only loading cannot restore this full RNG state."""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "last.pt"
            torch.save({"rng_state": {"numpy": np.random.get_state()}}, path)

            checkpoint = _load_trusted_checkpoint(path, torch.device("cpu"))

        self.assertEqual(checkpoint["rng_state"]["numpy"][0], "MT19937")

    def test_durable_directory_is_marked_and_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp) / "drive"
            parent.mkdir()
            directory = parent / "run"

            self.assertEqual(_ensure_durable_directory(directory), directory)
            self.assertEqual(
                (directory / ".directory_ready").read_text(encoding="utf-8"),
                "ready\n",
            )
            self.assertEqual(_ensure_durable_directory(directory), directory)

    def test_checkpoint_publishes_into_prepared_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp) / "drive"
            parent.mkdir()
            checkpoint_dir = _ensure_durable_directory(parent / "checkpoints")
            destination = checkpoint_dir / "last.pt"
            model = Mock()
            model.state_dict.return_value = {"weight": torch.tensor([1.0])}
            optimizer = Mock()
            optimizer.state_dict.return_value = {"state": {}}

            def fake_save(payload, path):
                Path(path).write_bytes(b"serialized checkpoint")

            with patch(
                "ddpm_derm.train_classifier.torch.save", side_effect=fake_save
            ), patch(
                "ddpm_derm.train_classifier._load_trusted_checkpoint",
                return_value={
                    "epoch": 1,
                    "history": [],
                    "run_identity": identity("resnet18"),
                },
            ), patch(
                "ddpm_derm.train_classifier.validate_checkpoint_payload"
            ), patch(
                "ddpm_derm.train_classifier.coca_run.checkpoint_size"
            ):
                save_checkpoint(
                    destination,
                    model,
                    optimizer,
                    epoch=1,
                    best_val_f1=0.0,
                    history=[],
                    args=argparse.Namespace(
                        seed=0, run_label="validation_smoke"
                    ),
                    run_identity=identity("resnet18"),
                )

            self.assertEqual(destination.read_bytes(), b"serialized checkpoint")
            self.assertFalse(destination.with_suffix(".pt.tmp").exists())

    def test_frozen_coca_round_trip_is_head_only_and_preserves_rebuilt_encoder(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "last.pt"
            source = TinyCoCa(encoder_value=7.0)
            source_optimizer = torch.optim.AdamW(source.head.parameters(), lr=1e-3)
            loss = source.head(torch.ones(1, 2)).sum()
            loss.backward(); source_optimizer.step()
            expected_head = {k: v.detach().clone() for k, v in source.head.state_dict().items()}
            run_identity = identity()
            save_checkpoint(
                path, source, source_optimizer, 4, 0.75,
                [{"epoch": index} for index in range(1, 5)],
                argparse.Namespace(seed=0, run_label=None), run_identity,
            )
            checkpoint = _load_trusted_checkpoint(path, torch.device("cpu"))

            self.assertEqual(
                checkpoint["checkpoint_format"],
                classifier_run.FROZEN_COCA_CHECKPOINT_FORMAT,
            )
            self.assertNotIn("model_state_dict", checkpoint)
            self.assertNotIn("encoder_state_dict", checkpoint)
            self.assertEqual(set(checkpoint["head_state_dict"]), {"weight", "bias"})
            self.assertFalse(any("encoder" in key or "text" in key or "caption" in key for key in checkpoint))

            rebuilt = TinyCoCa(encoder_value=19.0)
            rebuilt_optimizer = torch.optim.AdamW(rebuilt.head.parameters(), lr=1e-3)
            encoder_before = rebuilt.encoder.weight.detach().clone()
            start, best, history = restore_checkpoint_state(
                checkpoint, rebuilt, rebuilt_optimizer, run_identity, "coca_vit_b32"
            )
            self.assertEqual((start, best, len(history)), (5, 0.75, 4))
            for key, value in rebuilt.head.state_dict().items():
                torch.testing.assert_close(value, expected_head[key], rtol=0, atol=0)
            torch.testing.assert_close(rebuilt.encoder.weight, encoder_before, rtol=0, atol=0)
            self.assertFalse(rebuilt.encoder.training)
            self.assertTrue(all(not p.requires_grad for p in rebuilt.encoder.parameters()))
            self.assertEqual(
                sum(len(group["params"]) for group in rebuilt_optimizer.param_groups),
                len(list(rebuilt.head.parameters())),
            )

    def test_frozen_coca_strict_format_and_head_key_guards(self):
        model = TinyCoCa()
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
        base = {
            "checkpoint_schema_version": 1,
            "checkpoint_format": classifier_run.FROZEN_COCA_CHECKPOINT_FORMAT,
            "head_state_dict": model.head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 1, "best_val_df_f1": 0.1, "history": [],
            "config": {}, "class_to_idx": {}, "rng_state": {},
            "run_identity": identity(),
        }
        validate_checkpoint_payload(base, model, identity(), "coca_vit_b32")
        for change, pattern in (
            ({"checkpoint_format": "full_model_v1"}, "unsupported checkpoint format"),
            ({"model_state_dict": model.state_dict()}, "unexpected"),
        ):
            bad = {**base, **change}
            with self.assertRaisesRegex(ValueError, pattern):
                validate_checkpoint_payload(bad, model, identity(), "coca_vit_b32")
        old_full = dict(base)
        old_full.pop("checkpoint_format")
        old_full["model_state_dict"] = model.state_dict()
        with self.assertRaisesRegex(ValueError, "unsupported checkpoint format"):
            validate_checkpoint_payload(old_full, model, identity(), "coca_vit_b32")
        missing = {**base, "head_state_dict": {"weight": base["head_state_dict"]["weight"]}}
        with self.assertRaisesRegex(ValueError, "missing=.*bias"):
            validate_checkpoint_payload(missing, model, identity(), "coca_vit_b32")
        unexpected = {**base, "head_state_dict": {**base["head_state_dict"], "encoder.weight": torch.ones(1)}}
        with self.assertRaisesRegex(ValueError, "unexpected=.*encoder"):
            validate_checkpoint_payload(unexpected, model, identity(), "coca_vit_b32")

    def test_resnet_style_checkpoint_keeps_full_state_and_legacy_restore(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "resnet.pt"
            model = nn.Linear(2, 3)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            run_identity = identity("resnet18")
            save_checkpoint(
                path, model, optimizer, 1, 0.2, [],
                argparse.Namespace(seed=0, run_label=None), run_identity,
            )
            saved = _load_trusted_checkpoint(path, torch.device("cpu"))
            self.assertIn("model_state_dict", saved)
            self.assertNotIn("head_state_dict", saved)
            restored = nn.Linear(2, 3)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            legacy = dict(saved); legacy.pop("checkpoint_format"); legacy.pop("checkpoint_schema_version")
            restore_checkpoint_state(
                legacy, restored, restored_optimizer, run_identity, "resnet18"
            )
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[key])

    def test_checkpoint_size_limit_is_coca_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "large.pt"
            with path.open("wb") as handle:
                handle.truncate(coca_run.MAX_COCA_CHECKPOINT_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "bytes.*MiB"):
                coca_run.checkpoint_size(path, arch=coca_run.ARCH)
            self.assertEqual(
                coca_run.checkpoint_size(path, arch="resnet18"),
                coca_run.MAX_COCA_CHECKPOINT_BYTES + 1,
            )

    def test_exploratory_checkpoint_refuses_to_recreate_missing_parent(self):
        """Recursive mkdir can fork one Drive path into duplicate folders."""
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "drive" / "checkpoints" / "last.pt"
            destination.parent.mkdir(parents=True)
            serialized_to = []

            def fake_save(payload, path):
                serialized_to.append(Path(path))
                Path(path).write_bytes(b"serialized checkpoint")
                shutil.rmtree(destination.parent)

            model = Mock()
            model.state_dict.return_value = {"weight": torch.tensor([1.0])}
            optimizer = Mock()
            optimizer.state_dict.return_value = {"state": {}}
            with patch(
                "ddpm_derm.train_classifier.torch.save", side_effect=fake_save
            ):
                with self.assertRaisesRegex(FileNotFoundError, "refusing"):
                    save_checkpoint(
                        destination,
                        model,
                        optimizer,
                        epoch=1,
                        best_val_f1=0.0,
                        history=[],
                        args=argparse.Namespace(
                            seed=0, run_label="validation_smoke"
                        ),
                        run_identity=identity("resnet18"),
                    )

            self.assertFalse(destination.parent.exists())
            self.assertNotEqual(serialized_to[0].parent, destination.parent)
            self.assertFalse(serialized_to[0].exists())
            self.assertFalse(destination.with_suffix(".pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
