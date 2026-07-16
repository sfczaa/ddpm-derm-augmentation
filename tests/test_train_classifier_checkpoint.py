"""Regression test for full-state classifier checkpoint loading."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm.train_classifier import _load_trusted_checkpoint  # noqa: E402


class ClassifierCheckpointTests(unittest.TestCase):
    def test_trainer_checkpoint_loads_numpy_rng_state(self):
        """PyTorch 2.6+ safe-only loading cannot restore this full RNG state."""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "last.pt"
            torch.save({"rng_state": {"numpy": np.random.get_state()}}, path)

            checkpoint = _load_trusted_checkpoint(path, torch.device("cpu"))

        self.assertEqual(checkpoint["rng_state"]["numpy"][0], "MT19937")


if __name__ == "__main__":
    unittest.main()
