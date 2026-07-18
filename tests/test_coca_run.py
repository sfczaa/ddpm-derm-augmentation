"""Torch-free tests for shared-root, validation, resume, and aggregation guards."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm import coca_run  # noqa: E402


def model_identity(tag=coca_run.PRETRAINED_TAG):
    return {
        "arch": coca_run.ARCH,
        "model_name": coca_run.MODEL_NAME,
        "pretrained_tag": tag,
        "freeze_mode": "frozen_image_encoder_linear_head",
        "preprocessing_identity": {"train": "native", "eval": "native"},
        "input_resolution": [224, 224],
        "open_clip_torch_version": "3.3.0",
    }


def fake_runs():
    runs = []
    for variant, offset in (("C1", 0.0), ("C4", 0.1)):
        for seed in (0, 1, 2):
            runs.append({
                "variant": variant,
                "seed": seed,
                "run_identity": {
                    "model_identity": model_identity(),
                    "checkpoint_format": coca_run.CHECKPOINT_FORMAT,
                },
                "test_metrics": {
                    "target_f1": 0.5 + offset + seed * 0.1,
                    "macro_f1": 0.4 + offset + seed * 0.1,
                    "target_recall": 0.3 + offset + seed * 0.1,
                    "per_class_recall": {"df": 0.3 + offset + seed * 0.1},
                },
            })
    return runs


class CoCaRunTests(unittest.TestCase):
    def test_missing_shared_root_is_not_created(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "ddpm-derm-coca-runs"
            with self.assertRaises(FileNotFoundError):
                coca_run.require_existing_shared_root(missing)
            self.assertFalse(missing.exists())

    def test_drive_probes_and_nested_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            result = coca_run.probe_shared_drive(temp)
            self.assertEqual(result["status"], "passed")
            self.assertFalse(any(Path(temp).glob(".coca_*probe*")))

    def test_sentinel_is_immutable_after_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / ".coca_shared_root.json"
            first = coca_run.create_or_validate_sentinel(
                path, shortcut_alias="ddpm-derm-coca-runs",
                resolved_path=temp, drive_folder_id="folder-1",
            )
            second = coca_run.create_or_validate_sentinel(
                path, shortcut_alias="changed", resolved_path="changed",
                drive_folder_id="folder-2",
            )
            self.assertEqual(first, second)

    def test_concurrent_marker_is_retained_and_not_auto_cleared(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "_RUNNING.json"
            path.write_text(json.dumps({"session_id": "old"}), encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "old"):
                coca_run.create_running_marker(path, {"session_id": "new"})
            self.assertTrue(path.exists())
            with self.assertRaises(ValueError):
                coca_run.clear_stale_marker(path, "yes")
            self.assertTrue(path.exists())
            coca_run.clear_stale_marker(path, "CLEAR STALE MARKER")
            self.assertFalse(path.exists())

    def test_validation_record_requires_pass_and_exact_identity(self):
        expected = {"git_commit": "abc", "shared_root_uuid": "root"}
        record = {
            "validation_status": "VALIDATION PASSED",
            "formal_training_started": False,
            **expected,
        }
        coca_run.require_validation_record(record, expected)
        for key, value in (("validation_status", "failed"),
                           ("formal_training_started", True),
                           ("git_commit", "different")):
            bad = copy.deepcopy(record)
            bad[key] = value
            with self.assertRaises(ValueError):
                coca_run.require_validation_record(bad, expected)

    def test_account_and_hostname_are_not_resume_identity(self):
        identity = {
            "git_commit": "abc", "run_version": "v1",
            "candidate_manifest_sha256": "1" * 64,
            "fixed_split_identity": "split", "shared_root_uuid": "root",
            "formal_output_identity": "formal", "model_identity": model_identity(),
            "checkpoint_format": coca_run.CHECKPOINT_FORMAT,
        }
        saved = {**identity, "account_label": "A", "hostname": "host-a"}
        current = {**identity, "account_label": "B", "hostname": "host-b"}
        coca_run.require_resume_identity(saved, current)
        current["shared_root_uuid"] = "other"
        with self.assertRaisesRegex(ValueError, "shared_root_uuid"):
            coca_run.require_resume_identity(saved, current)

    def test_population_std_and_paired_c4_minus_c1(self):
        aggregate = coca_run.aggregate_results(fake_runs())
        self.assertEqual(aggregate["ddof"], 0)
        self.assertAlmostEqual(
            aggregate["variants"]["C1"]["df_f1"]["population_std"],
            0.0816496580927726,
        )
        for value in aggregate["paired_c4_minus_c1"]["seed_differences"]:
            self.assertAlmostEqual(value, 0.1)

    def test_aggregate_rejects_mixed_model_identity(self):
        runs = fake_runs()
        runs[-1]["run_identity"]["model_identity"] = model_identity("other")
        with self.assertRaisesRegex(ValueError, "mixed"):
            coca_run.aggregate_results(runs)


if __name__ == "__main__":
    unittest.main()
