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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm.train_classifier import (  # noqa: E402
    _ensure_durable_directory,
    _load_trusted_checkpoint,
    save_checkpoint,
)


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
                    run_identity={"git_commit": "test"},
                )

            self.assertEqual(destination.read_bytes(), b"serialized checkpoint")
            self.assertFalse(destination.with_suffix(".pt.tmp").exists())

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
                        run_identity={"git_commit": "test"},
                    )

            self.assertFalse(destination.parent.exists())
            self.assertNotEqual(serialized_to[0].parent, destination.parent)
            self.assertFalse(serialized_to[0].exists())
            self.assertFalse(destination.with_suffix(".pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
