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

    def test_checkpoint_stages_locally_and_recreates_publish_directory(self):
        """Drive shortcuts can drop an empty directory before the first save."""
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
                save_checkpoint(
                    destination,
                    model,
                    optimizer,
                    epoch=1,
                    best_val_f1=0.0,
                    history=[],
                    args=argparse.Namespace(seed=0),
                    run_identity={"git_commit": "test"},
                )

            self.assertEqual(destination.read_bytes(), b"serialized checkpoint")
            self.assertNotEqual(serialized_to[0].parent, destination.parent)
            self.assertFalse(serialized_to[0].exists())
            self.assertFalse(destination.with_suffix(".pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
