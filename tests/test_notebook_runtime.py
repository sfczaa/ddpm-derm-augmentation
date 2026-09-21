"""Reject notebook runtimes below the project's dependency minimums."""

import unittest
from unittest.mock import patch

from ddpm_derm.notebook_runtime import require_training_runtime


class NotebookRuntimeTests(unittest.TestCase):
    def test_supported_runtime_accepts_local_build_suffix(self):
        installed = {"torch": "2.13.0+cu130", "Pillow": "12.3.0"}
        with patch("ddpm_derm.notebook_runtime.version", side_effect=installed.__getitem__):
            require_training_runtime()

    def test_old_or_prerelease_runtime_is_rejected(self):
        for package, old in (("torch", "2.8.0"), ("torch", "2.13.0rc1"), ("Pillow", "12.2.0")):
            with self.subTest(package=package, version=old):
                installed = {"torch": "2.13.0", "Pillow": "12.3.0", package: old}
                with patch("ddpm_derm.notebook_runtime.version", side_effect=installed.__getitem__):
                    with self.assertRaisesRegex(RuntimeError, package):
                        require_training_runtime()
