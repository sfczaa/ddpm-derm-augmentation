"""Regression coverage for PanDerm archive integrity and session recovery."""

from __future__ import annotations

import copy
import ast
import contextlib
import csv
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from tests.test_panderm_base_c1_finetune import (
    AllowDurableWriteGuard,
    build_mock_model,
)

from ddpm_derm import panderm, panderm_run, train_panderm


def identity(**overrides):
    value = panderm_run.build_run_identity(
        git_commit="c" * 40,
        seed=0,
        epochs=5,
        evaluation_scope="validation_only",
        checkpoint_sha256="a" * 64,
        model_identity={
            "arch": panderm_run.ARCH,
            "drop_path": panderm_run.DROP_PATH,
        },
        manifest_sha256={"train": "t", "val": "v"},
        fixed_split_identity="t",
        shared_root_uuid="uuid",
        formal_output_identity="validation-output",
        dependency_versions={"torch": "2.8.0"},
        warmup_epochs=5,
        drop_path=0.2,
        amp_requested=True,
        amp_effective=False,
        device_type="cpu",
    )
    value.update(overrides)
    return value


def completed_artifacts(run_identity):
    result = {
        "run_identity": copy.deepcopy(run_identity),
        "run_version": run_identity["run_version"],
        **copy.deepcopy(run_identity),
    }
    best = {"run_identity": copy.deepcopy(run_identity)}
    last = {"run_identity": copy.deepcopy(run_identity)}
    return result, best, last


class ContaminationGateMatrixTests(unittest.TestCase):
    def test_formal_and_test_matrix_rejects_all_provenance_variants(self):
        reviews = {
            "unknown_private_corpus": {
                **panderm_run.CONTAMINATION_REVIEW,
                "image_level_ham10000_overlap": "unknown_private_corpus",
            },
            "author_assertion": {
                **panderm_run.CONTAMINATION_REVIEW,
                "image_level_ham10000_overlap":
                    "excluded_on_primary_source_author_statement",
            },
            "independent_audit_false": {
                **panderm_run.CONTAMINATION_REVIEW,
                "independent_audit_possible": False,
            },
            "patient_overlap_not_excludable": {
                **panderm_run.CONTAMINATION_REVIEW,
                "patient_level_overlap": "not_excludable",
            },
        }
        rejected = 0
        for review in reviews.values():
            for purpose in (
                panderm_run.FORMAL_TRAINING,
                panderm_run.TEST_ACCESS,
            ):
                with self.assertRaisesRegex(ValueError, "independently unauditable"):
                    panderm_run.require_provenance_clearance(
                        upstream_commit=panderm_run.UPSTREAM_COMMIT,
                        checkpoint_sha256="a" * 64,
                        expected_checkpoint_sha256="a" * 64,
                        contamination_review=review,
                        purpose=purpose,
                    )
                rejected += 1
        self.assertEqual(rejected, 8)

    def test_validation_only_exploratory_path_passes_required_checks(self):
        cleared = panderm_run.require_provenance_clearance(
            upstream_commit=panderm_run.UPSTREAM_COMMIT,
            checkpoint_sha256="a" * 64,
            expected_checkpoint_sha256="a" * 64,
            purpose=panderm_run.VALIDATION_ONLY,
        )
        self.assertEqual(cleared["cleared_for"], "validation_only")
        self.assertFalse(cleared["formal_training_allowed"])
        self.assertFalse(cleared["test_access_allowed"])


class TestIsolationTimingTests(unittest.TestCase):
    def test_full_scope_rejected_before_manifest_loader(self):
        argv = [
            "--checkpoint", "weights.pth",
            "--upstream-dir", "upstream",
            "--output-dir", "out",
            "--evaluation-scope", "full",
        ]
        with mock.patch.object(
            train_panderm.manifests,
            "load_split",
            side_effect=AssertionError("manifest loader reached"),
        ) as loader:
            with self.assertRaises(SystemExit):
                train_panderm.main(argv)
        loader.assert_not_called()

    def test_test_access_unreachable_at_all_timing_probes(self):
        rejected = 0
        for _scenario in (
            "before_validation_match",
            "before_three_seed_completion",
            "forged_validation_pass",
        ):
            with self.assertRaisesRegex(ValueError, "independently unauditable"):
                panderm_run.require_provenance_clearance(
                    upstream_commit=panderm_run.UPSTREAM_COMMIT,
                    checkpoint_sha256="a" * 64,
                    expected_checkpoint_sha256="a" * 64,
                    purpose=panderm_run.TEST_ACCESS,
                )
            rejected += 1
        self.assertEqual(rejected, 3)

    def test_source_and_notebooks_have_no_executable_test_path(self):
        root = Path(__file__).resolve().parents[1]
        trainer = (
            root / "src" / "ddpm_derm" / "train_panderm.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('load_split("test")', trainer)
        self.assertNotIn('evaluation_scope == "full"', trainer)
        for name in (
            "colab_panderm_base_c1_finetune_validation.ipynb",
            "colab_panderm_base_c1_finetune_classifier.ipynb",
        ):
            notebook = json.loads(
                (root / "notebooks" / name).read_text(encoding="utf-8")
            )
            code = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
            tree = ast.parse(code)
            executable_test_loads = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "load_split"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "test"
            ]
            self.assertEqual(executable_test_loads, [])
            if name.endswith("_classifier.ipynb"):
                self.assertNotIn('"--evaluation-scope", "full"', code)


class ValidationDataStagingTests(unittest.TestCase):
    def _write_manifest(self, path, image_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_path"])
            writer.writeheader()
            for image_path in image_paths:
                writer.writerow({"image_path": image_path})

    def _approved_content_identity(
        self, shared, *, train_rows=1, val_rows=1, unique_images=2
    ):
        """Approved expected content digest, derived from the source tree.

        Production pins a Git-reviewed constant
        (``EXPECTED_VALIDATION_CONTENT_IDENTITY_SHA256``). Fixtures cannot use a
        fixed literal because each temporary tree is new, so they derive theirs
        from the *source*, which is still an input independent of the archive,
        identity file, READY marker and extraction being validated.
        """
        inventory = panderm_run.validation_source_inventory(
            shared,
            expected_train_rows=train_rows,
            expected_val_rows=val_rows,
            expected_unique_images=unique_images,
        )
        return inventory["file_content_identity"]["sha256"]

    def _fixture(self, root):
        shared = root / "shared_data"
        mixed = shared / "raw" / "mixed"
        mixed.mkdir(parents=True)
        for name in ("train_a.jpg", "val_a.jpg", "test_only.jpg"):
            (mixed / name).write_bytes(name.encode("ascii").ljust(16, b"_"))
        manifests = shared / "manifests"
        self._write_manifest(manifests / "train.csv", ["raw/mixed/train_a.jpg"])
        self._write_manifest(manifests / "val.csv", ["raw/mixed/val_a.jpg"])
        self._write_manifest(manifests / "test.csv", ["raw/mixed/test_only.jpg"])
        (manifests / "class_to_idx.json").write_text(
            json.dumps(panderm_run.EXPECTED_CLASS_TO_IDX) + "\n",
            encoding="utf-8",
        )
        return shared

    def test_allowlist_staging_excludes_test_manifest_and_mixed_test_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = self._fixture(root)
            local = root / "local_data"
            report = panderm_run.stage_validation_data(shared, local)

            self.assertTrue((local / "raw" / "mixed" / "train_a.jpg").is_file())
            self.assertTrue((local / "raw" / "mixed" / "val_a.jpg").is_file())
            self.assertFalse((local / "raw" / "mixed" / "test_only.jpg").exists())
            self.assertTrue((local / "manifests" / "train.csv").is_file())
            self.assertTrue((local / "manifests" / "val.csv").is_file())
            self.assertTrue(
                (local / "manifests" / "class_to_idx.json").is_file()
            )
            self.assertFalse((local / "manifests" / "test.csv").exists())
            self.assertEqual(report["manifest_files_copied"], 2)
            self.assertEqual(report["class_mapping_files_copied"], 1)
            self.assertEqual(report["images_copied"], 2)
            self.assertEqual(
                set(report["image_relative_paths"]),
                {"raw/mixed/train_a.jpg", "raw/mixed/val_a.jpg"},
            )

    def test_staging_reports_live_progress_and_time_based_heartbeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = self._fixture(root)
            local = root / "local_data"
            clock = [0.0]
            real_copy2 = panderm_run.shutil.copy2
            real_canonical_path = panderm_run._canonical_manifest_image_path

            def slow_canonical_path(raw):
                result = real_canonical_path(raw)
                clock[0] += 31.0
                return result

            def slow_copy(source, destination):
                result = real_copy2(source, destination)
                clock[0] += 31.0
                return result

            with (
                mock.patch.object(
                    panderm_run.shutil, "copy2", side_effect=slow_copy
                ),
                mock.patch.object(
                    panderm_run.time,
                    "perf_counter",
                    side_effect=lambda: clock[0],
                ),
                mock.patch.object(
                    panderm_run,
                    "_canonical_manifest_image_path",
                    side_effect=slow_canonical_path,
                ),
                mock.patch("builtins.print") as print_mock,
            ):
                report = panderm_run.stage_validation_data(shared, local)

            self.assertEqual(report["images_copied"], 2)
            messages = "\n".join(
                str(call.args[0]) for call in print_mock.call_args_list
            )
            for required in (
                "[Phase 1] START validation staging:",
                "[Phase 1] scan train: START",
                "[Phase 1] scan train: rows=1",
                "[Phase 1] scan train: DONE rows=1",
                "[Phase 1] scan val: START",
                "[Phase 1] scan val: rows=1",
                "[Phase 1] scan val: DONE rows=1",
                "[Phase 1] copy images: START total=2",
                "[Phase 1] copy images: 1/2",
                "[Phase 1] copy images: 2/2",
                "[Phase 1] copy images: DONE total=2",
                "[Phase 1] verify images: START expected=2",
                "[Phase 1] verify images: 2/2",
                "[Phase 1] verify images: DONE actual=2",
                "[Phase 1] DONE validation staging: copied=2",
            ):
                self.assertIn(required, messages)
            self.assertTrue(
                all(
                    call.kwargs.get("flush") is True
                    for call in print_mock.call_args_list
                )
            )

    def test_staging_rejects_unsafe_or_ambiguous_manifest_paths(self):
        cases = {
            "absolute": ["C:/outside.jpg"],
            "traversal": ["../outside.jpg"],
            "duplicate_destination": [
                "raw/mixed/train_a.jpg",
                "raw/mixed/train_a.jpg",
            ],
            "missing_source": ["raw/mixed/missing.jpg"],
        }
        for name, paths in cases.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    shared = self._fixture(root)
                    self._write_manifest(
                        shared / "manifests" / "train.csv", paths
                    )
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        panderm_run.stage_validation_data(
                            shared, root / "local_data"
                        )

    def test_static_validator_rejects_legacy_whole_data_copy(self):
        legacy = (
            "import shutil\n"
            'shutil.copytree(SHARED_PROJECT_DIR / "data", LOCAL_DATA_DIR)\n'
        )
        with self.assertRaisesRegex(ValueError, "recursive|copy"):
            panderm_run.validate_validation_notebook_source(legacy)

    def test_four_image_smoke_sample_is_deterministic_and_train_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = self._fixture(root)
            mixed = shared / "raw" / "mixed"
            train_paths = []
            for name in ("train_d.jpg", "train_c.jpg", "train_b.jpg"):
                (mixed / name).write_bytes(name.encode("ascii"))
                train_paths.append(f"raw/mixed/{name}")
            train_paths.append("raw/mixed/train_a.jpg")
            self._write_manifest(
                shared / "manifests" / "train.csv", train_paths
            )
            report = panderm_run.stage_train_smoke_sample(
                shared, root / "smoke"
            )
            self.assertEqual(report["source_manifest"], "manifests/train.csv")
            self.assertEqual(report["sample_count"], 4)
            self.assertEqual(report["relative_paths"], sorted(train_paths))
            self.assertFalse(report["test_manifest_read"])
            self.assertFalse(
                (root / "smoke" / "raw" / "mixed" / "test_only.jpg").exists()
            )

    def test_archive_build_and_single_tar_reuse_are_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = self._fixture(root)
            inventory = panderm_run.validation_source_inventory(
                shared,
                expected_train_rows=1,
                expected_val_rows=1,
                expected_unique_images=2,
            )
            cache_parent = root / "cache_parent"
            cache_parent.mkdir()
            cache = cache_parent / "cache"
            runtime = root / "runtime"
            runtime.mkdir()
            identity_record = panderm_run.build_validation_archive_cache(
                shared,
                cache,
                runtime,
                expected_file_content_identity_sha256=self._approved_content_identity(shared),
                source_fixed_split_identity="fixed-split",
                expected_train_rows=1,
                expected_val_rows=1,
                expected_unique_images=2,
                write_guard=AllowDurableWriteGuard().require,
            )
            self.assertTrue(
                (cache / panderm_run.VALIDATION_ARCHIVE_READY_FILENAME).is_file()
            )
            self.assertEqual(identity_record["unique_images"], 2)
            self.assertFalse(identity_record["test_manifest_included"])
            self.assertFalse(identity_record["whole_data_copy_used"])
            members = panderm_run._validated_tar_members(
                cache / panderm_run.VALIDATION_ARCHIVE_FILENAME,
                expected_members=inventory["member_names"],
            )
            member_names = {member.name for member in members}
            self.assertNotIn("manifests/test.csv", member_names)
            self.assertNotIn("raw/mixed/test_only.jpg", member_names)
            content_paths = {
                row["path"] for row in identity_record["file_content_rows"]
            }
            self.assertEqual(content_paths, member_names)
            self.assertNotIn("manifests/test.csv", content_paths)
            self.assertNotIn("raw/mixed/test_only.jpg", content_paths)

            real_copy = panderm_run._copy_file_with_progress
            with mock.patch.object(
                panderm_run,
                "_copy_file_with_progress",
                wraps=real_copy,
            ) as copy_mock:
                report = panderm_run.reuse_validation_archive_cache(
                    cache,
                    runtime,
                    runtime / "data",
                    expected_file_content_identity_sha256=self._approved_content_identity(shared),
                    expected_fixed_split_identity="fixed-split",
                    expected_manifest_sha256=inventory["manifest_sha256"],
                    expected_class_mapping_sha256=
                        inventory["class_mapping_sha256"],
                    write_guard=AllowDurableWriteGuard().require,
                )
            self.assertEqual(copy_mock.call_count, 1)
            self.assertEqual(report["archive_files_copied"], 1)
            self.assertEqual(report["images_staged"], 2)
            self.assertFalse((runtime / "data" / "manifests" / "test.csv").exists())
            self.assertFalse(
                (runtime / "data" / "raw" / "mixed" / "test_only.jpg").exists()
            )
            extracted_identity = panderm_run._file_content_identity_from_sources(
                {
                    name: runtime / "data" / Path(*name.split("/"))
                    for name in inventory["member_names"]
                },
                phase="test-extracted-content",
            )
            tar_identity = panderm_run._file_content_identity_from_tar(
                cache / panderm_run.VALIDATION_ARCHIVE_FILENAME,
                expected_members=inventory["member_names"],
                phase="test-tar-content",
            )
            self.assertEqual(inventory["file_content_identity"], tar_identity)
            self.assertEqual(tar_identity, extracted_identity)
            self.assertEqual(
                identity_record["file_content_identity_sha256"],
                inventory["file_content_identity"]["sha256"],
            )
            self.assertEqual(
                identity_record["file_content_rows"],
                inventory["file_content_identity"]["rows"],
            )
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                panderm_run.build_validation_archive_cache(
                    shared,
                    cache,
                    runtime,
                    expected_file_content_identity_sha256=self._approved_content_identity(shared),
                    source_fixed_split_identity="fixed-split",
                    expected_train_rows=1,
                    expected_val_rows=1,
                    expected_unique_images=2,
                    write_guard=AllowDurableWriteGuard().require,
                )

    def test_archive_build_fence_loss_never_publishes_final_cache(self):
        failure_phases = (
            "archive runtime build progress 5/5",
            "JSON temporary write archive_identity.json",
            "archive cache directory publish",
        )
        for failure_phase in failure_phases:
            with self.subTest(failure_phase=failure_phase):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    shared = self._fixture(root)
                    cache_parent = root / "cache_parent"
                    cache_parent.mkdir()
                    cache = cache_parent / "cache"
                    runtime = root / "runtime"
                    runtime.mkdir()
                    calls = []

                    def reject(phase):
                        calls.append(phase)
                        if failure_phase in phase:
                            raise RuntimeError("stale archive fence")

                    with self.assertRaisesRegex(
                        RuntimeError, "stale archive fence"
                    ):
                        panderm_run.build_validation_archive_cache(
                            shared,
                            cache,
                            runtime,
                            expected_file_content_identity_sha256=(
                                self._approved_content_identity(shared)
                            ),
                            source_fixed_split_identity="fixed-split",
                            expected_train_rows=1,
                            expected_val_rows=1,
                            expected_unique_images=2,
                            write_guard=reject,
                        )
                    self.assertFalse(cache.exists())
                    self.assertEqual(list(runtime.iterdir()), [])
                    self.assertTrue(
                        any(failure_phase in phase for phase in calls)
                    )

    def test_archive_reuse_fence_loss_cleans_runtime_and_never_publishes(self):
        failure_phases = (
            "archive cache validation completion",
            "archive reuse extraction complete",
            "archive reuse local publish",
        )
        for failure_phase in failure_phases:
            with self.subTest(failure_phase=failure_phase):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    shared = self._fixture(root)
                    inventory = panderm_run.validation_source_inventory(
                        shared,
                        expected_train_rows=1,
                        expected_val_rows=1,
                        expected_unique_images=2,
                    )
                    cache_parent = root / "cache_parent"
                    cache_parent.mkdir()
                    cache = cache_parent / "cache"
                    runtime = root / "runtime"
                    runtime.mkdir()
                    panderm_run.build_validation_archive_cache(
                        shared,
                        cache,
                        runtime,
                        expected_file_content_identity_sha256=(
                            self._approved_content_identity(shared)
                        ),
                        source_fixed_split_identity="fixed-split",
                        expected_train_rows=1,
                        expected_val_rows=1,
                        expected_unique_images=2,
                        write_guard=AllowDurableWriteGuard().require,
                    )
                    local = runtime / "data"
                    calls = []

                    def reject(phase):
                        calls.append(phase)
                        if failure_phase == phase:
                            raise RuntimeError("stale archive fence")

                    with self.assertRaisesRegex(
                        RuntimeError, "stale archive fence"
                    ):
                        panderm_run.reuse_validation_archive_cache(
                            cache,
                            runtime,
                            local,
                            expected_file_content_identity_sha256=(
                                self._approved_content_identity(shared)
                            ),
                            expected_fixed_split_identity="fixed-split",
                            expected_manifest_sha256=inventory["manifest_sha256"],
                            expected_class_mapping_sha256=(
                                inventory["class_mapping_sha256"]
                            ),
                            write_guard=reject,
                        )
                    self.assertFalse(local.exists())
                    self.assertEqual(list(runtime.iterdir()), [])
                    self.assertIn(failure_phase, calls)

    def test_extracted_image_content_tamper_matrix_is_rejected(self):
        def one_byte(train, _val):
            payload = train.read_bytes()
            train.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])

        def same_size(train, _val):
            train.write_bytes(b"x" * train.stat().st_size)

        def swapped(train, val):
            train_payload = train.read_bytes()
            val_payload = val.read_bytes()
            self.assertEqual(len(train_payload), len(val_payload))
            train.write_bytes(val_payload)
            val.write_bytes(train_payload)

        for name, mutate in {
            "one_byte": one_byte,
            "same_size": same_size,
            "swapped": swapped,
        }.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    shared = self._fixture(root)
                    runtime = root / "runtime"
                    runtime.mkdir()
                    cache_parent = root / "cache_parent"
                    cache_parent.mkdir()
                    cache = cache_parent / "cache"
                    identity_record = panderm_run.build_validation_archive_cache(
                        shared,
                        cache,
                        runtime,
                        expected_file_content_identity_sha256=self._approved_content_identity(shared),
                        source_fixed_split_identity="fixed-split",
                        expected_train_rows=1,
                        expected_val_rows=1,
                        expected_unique_images=2,
                        write_guard=AllowDurableWriteGuard().require,
                    )
                    local = runtime / "data"
                    panderm_run.reuse_validation_archive_cache(
                        cache,
                        runtime,
                        local,
                        expected_file_content_identity_sha256=self._approved_content_identity(shared),
                        expected_fixed_split_identity="fixed-split",
                        expected_manifest_sha256={
                            "train": identity_record["train_manifest_sha256"],
                            "val": identity_record["val_manifest_sha256"],
                        },
                        expected_class_mapping_sha256=
                            identity_record["class_mapping_sha256"],
                        write_guard=AllowDurableWriteGuard().require,
                    )
                    mutate(
                        local / "raw" / "mixed" / "train_a.jpg",
                        local / "raw" / "mixed" / "val_a.jpg",
                    )
                    with self.assertRaisesRegex(ValueError, "content"):
                        panderm_run._validate_extracted_validation_data(
                            local,
                            identity_record,
                            approved_file_content_identity_sha256=(
                                self._approved_content_identity(shared)
                            ),
                        )

    def test_extracted_image_file_set_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = self._fixture(root)
            runtime = root / "runtime"
            runtime.mkdir()
            cache_parent = root / "cache_parent"
            cache_parent.mkdir()
            cache = cache_parent / "cache"
            identity_record = panderm_run.build_validation_archive_cache(
                shared,
                cache,
                runtime,
                expected_file_content_identity_sha256=self._approved_content_identity(shared),
                source_fixed_split_identity="fixed-split",
                expected_train_rows=1,
                expected_val_rows=1,
                expected_unique_images=2,
                write_guard=AllowDurableWriteGuard().require,
            )
            local = runtime / "data"
            panderm_run.reuse_validation_archive_cache(
                cache,
                runtime,
                local,
                expected_file_content_identity_sha256=self._approved_content_identity(shared),
                expected_fixed_split_identity="fixed-split",
                expected_manifest_sha256={
                    "train": identity_record["train_manifest_sha256"],
                    "val": identity_record["val_manifest_sha256"],
                },
                expected_class_mapping_sha256=
                    identity_record["class_mapping_sha256"],
                write_guard=AllowDurableWriteGuard().require,
            )
            for name, mutate in {
                "missing": lambda candidate: (
                    candidate / "raw" / "mixed" / "train_a.jpg"
                ).unlink(),
                "extra": lambda candidate: (
                    candidate / "raw" / "mixed" / "extra.jpg"
                ).write_bytes(b"extra"),
                "path_drift": lambda candidate: (
                    candidate / "raw" / "mixed" / "train_a.jpg"
                ).rename(candidate / "raw" / "mixed" / "renamed.jpg"),
            }.items():
                with self.subTest(name=name):
                    candidate = runtime / f"data_{name}"
                    shutil.copytree(local, candidate)
                    mutate(candidate)
                    with self.assertRaises(
                        (ValueError, FileNotFoundError)
                    ):
                        panderm_run._validate_extracted_validation_data(
                            candidate,
                            identity_record,
                            approved_file_content_identity_sha256=(
                                self._approved_content_identity(shared)
                            ),
                        )

    def test_reuse_rejects_tamper_before_publishing_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = self._fixture(root)
            runtime = root / "runtime"
            runtime.mkdir()
            cache_parent = root / "cache_parent"
            cache_parent.mkdir()
            cache = cache_parent / "cache"
            inventory = panderm_run.validation_source_inventory(
                shared,
                expected_train_rows=1,
                expected_val_rows=1,
                expected_unique_images=2,
            )
            panderm_run.build_validation_archive_cache(
                shared,
                cache,
                runtime,
                expected_file_content_identity_sha256=self._approved_content_identity(shared),
                source_fixed_split_identity="fixed-split",
                expected_train_rows=1,
                expected_val_rows=1,
                expected_unique_images=2,
                write_guard=AllowDurableWriteGuard().require,
            )
            real_extract = panderm_run._safe_extract_validation_archive

            def extract_then_tamper(*args, **kwargs):
                real_extract(*args, **kwargs)
                destination = Path(args[1])
                image = destination / "raw" / "mixed" / "train_a.jpg"
                payload = image.read_bytes()
                image.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])

            local = runtime / "data"
            runner = mock.Mock()

            def reuse_then_invoke_runner():
                report = panderm_run.reuse_validation_archive_cache(
                    cache,
                    runtime,
                    local,
                    expected_file_content_identity_sha256=self._approved_content_identity(shared),
                    expected_fixed_split_identity="fixed-split",
                    expected_manifest_sha256=inventory["manifest_sha256"],
                    expected_class_mapping_sha256=
                        inventory["class_mapping_sha256"],
                    write_guard=AllowDurableWriteGuard().require,
                )
                runner(report)

            with (
                mock.patch.object(
                    panderm_run,
                    "_safe_extract_validation_archive",
                    side_effect=extract_then_tamper,
                ),
                self.assertRaisesRegex(ValueError, "content"),
            ):
                reuse_then_invoke_runner()
            runner.assert_not_called()
            self.assertFalse(local.exists())

    def test_cache_build_rejects_tar_payload_drift_before_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = self._fixture(root)
            runtime = root / "runtime"
            runtime.mkdir()
            cache_parent = root / "cache_parent"
            cache_parent.mkdir()
            cache = cache_parent / "cache"
            real_addfile = tarfile.TarFile.addfile

            def addfile_with_tampered_image(archive, tarinfo, fileobj=None):
                if tarinfo.name == "raw/mixed/train_a.jpg":
                    fileobj = io.BytesIO(b"x" * tarinfo.size)
                return real_addfile(archive, tarinfo, fileobj)

            with (
                mock.patch.object(
                    tarfile.TarFile,
                    "addfile",
                    new=addfile_with_tampered_image,
                ),
                self.assertRaisesRegex(ValueError, "source.*tar|content"),
            ):
                panderm_run.build_validation_archive_cache(
                    shared,
                    cache,
                    runtime,
                    expected_file_content_identity_sha256=self._approved_content_identity(shared),
                    source_fixed_split_identity="fixed-split",
                    expected_train_rows=1,
                    expected_val_rows=1,
                    expected_unique_images=2,
                    write_guard=AllowDurableWriteGuard().require,
                )
            self.assertFalse(cache.exists())

    def test_adversarial_tar_member_matrix_is_rejected(self):
        cases = {
            "absolute": [("/absolute.jpg", tarfile.REGTYPE)],
            "windows_absolute": [("C:/absolute.jpg", tarfile.REGTYPE)],
            "traversal": [("../escape.jpg", tarfile.REGTYPE)],
            "symlink": [("image.jpg", tarfile.SYMTYPE)],
            "hardlink": [("image.jpg", tarfile.LNKTYPE)],
            "duplicate": [
                ("image.jpg", tarfile.REGTYPE),
                ("image.jpg", tarfile.REGTYPE),
            ],
        }
        for name, specifications in cases.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary:
                    archive_path = Path(temporary) / "bad.tar"
                    with tarfile.open(archive_path, mode="w") as archive:
                        for member_name, member_type in specifications:
                            info = tarfile.TarInfo(member_name)
                            info.type = member_type
                            if member_type == tarfile.REGTYPE:
                                info.size = 1
                                archive.addfile(info, io.BytesIO(b"x"))
                            else:
                                info.linkname = "target"
                                archive.addfile(info)
                    with self.assertRaises(ValueError):
                        panderm_run._validated_tar_members(archive_path)

    def test_archive_identity_tamper_and_missing_ready_matrix_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = self._fixture(root)
            inventory = panderm_run.validation_source_inventory(
                shared,
                expected_train_rows=1,
                expected_val_rows=1,
                expected_unique_images=2,
            )
            runtime = root / "runtime"
            runtime.mkdir()
            cache_parent = root / "cache_parent"
            cache_parent.mkdir()
            base = cache_parent / "base"
            panderm_run.build_validation_archive_cache(
                shared,
                base,
                runtime,
                expected_file_content_identity_sha256=self._approved_content_identity(shared),
                source_fixed_split_identity="fixed-split",
                expected_train_rows=1,
                expected_val_rows=1,
                expected_unique_images=2,
                write_guard=AllowDurableWriteGuard().require,
            )

            mutations = {
                "archive_sha256": lambda value: value.__setitem__(
                    "archive_sha256", "0" * 64
                ),
                "byte_size": lambda value: value.__setitem__(
                    "byte_size", value["byte_size"] + 1
                ),
                "member_list": lambda value: value.__setitem__(
                    "canonical_sorted_member_list_sha256", "0" * 64
                ),
                "manifest_hash": lambda value: value.__setitem__(
                    "train_manifest_sha256", "0" * 64
                ),
                "class_mapping_hash": lambda value: value.__setitem__(
                    "class_mapping_sha256", "0" * 64
                ),
                "file_content_hash": lambda value: value.__setitem__(
                    "file_content_identity_sha256", "0" * 64
                ),
                "file_content_rows": lambda value: value["file_content_rows"][0]
                    .__setitem__("sha256", "0" * 64),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    candidate = cache_parent / name
                    shutil.copytree(base, candidate)
                    identity_path = (
                        candidate
                        / panderm_run.VALIDATION_ARCHIVE_IDENTITY_FILENAME
                    )
                    value = json.loads(identity_path.read_text(encoding="utf-8"))
                    mutate(value)
                    identity_path.write_text(
                        json.dumps(value), encoding="utf-8"
                    )
                    ready = {
                        "schema_version":
                            panderm_run.VALIDATION_ARCHIVE_SCHEMA_VERSION,
                        "cache_format_identity":
                            panderm_run.VALIDATION_ARCHIVE_CACHE_FORMAT,
                        "archive_filename":
                            panderm_run.VALIDATION_ARCHIVE_FILENAME,
                        "archive_sha256": value["archive_sha256"],
                        "file_content_identity_sha256":
                            value["file_content_identity_sha256"],
                        "archive_identity_sha256":
                            panderm_run._canonical_mapping_sha256(value),
                    }
                    (
                        candidate / panderm_run.VALIDATION_ARCHIVE_READY_FILENAME
                    ).write_text(json.dumps(ready), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        panderm_run.validate_validation_archive_cache(
                            candidate,
                            expected_file_content_identity_sha256=self._approved_content_identity(shared),
                            expected_fixed_split_identity="fixed-split",
                            expected_manifest_sha256=inventory["manifest_sha256"],
                            expected_class_mapping_sha256=
                                inventory["class_mapping_sha256"],
                        )

            for missing_field in (
                "file_content_identity_sha256",
                "file_content_rows",
            ):
                with self.subTest(missing_field=missing_field):
                    candidate = cache_parent / f"missing_{missing_field}"
                    shutil.copytree(base, candidate)
                    identity_path = (
                        candidate
                        / panderm_run.VALIDATION_ARCHIVE_IDENTITY_FILENAME
                    )
                    value = json.loads(identity_path.read_text(encoding="utf-8"))
                    del value[missing_field]
                    identity_path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "schema"):
                        panderm_run.validate_validation_archive_cache(
                            candidate,
                            expected_file_content_identity_sha256=self._approved_content_identity(shared),
                            expected_fixed_split_identity="fixed-split",
                            expected_manifest_sha256=inventory["manifest_sha256"],
                            expected_class_mapping_sha256=
                                inventory["class_mapping_sha256"],
                        )

            common_wrong = cache_parent / "common_wrong_content"
            shutil.copytree(base, common_wrong)
            identity_path = (
                common_wrong
                / panderm_run.VALIDATION_ARCHIVE_IDENTITY_FILENAME
            )
            value = json.loads(identity_path.read_text(encoding="utf-8"))
            value["file_content_rows"][0]["sha256"] = "0" * 64
            value["file_content_identity_sha256"] = (
                panderm_run._canonical_file_content_rows_sha256(
                    value["file_content_rows"]
                )
            )
            identity_path.write_text(json.dumps(value), encoding="utf-8")
            ready = {
                "schema_version": panderm_run.VALIDATION_ARCHIVE_SCHEMA_VERSION,
                "cache_format_identity":
                    panderm_run.VALIDATION_ARCHIVE_CACHE_FORMAT,
                "archive_filename": panderm_run.VALIDATION_ARCHIVE_FILENAME,
                "archive_sha256": value["archive_sha256"],
                "file_content_identity_sha256":
                    value["file_content_identity_sha256"],
                "archive_identity_sha256":
                    panderm_run._canonical_mapping_sha256(value),
            }
            (
                common_wrong / panderm_run.VALIDATION_ARCHIVE_READY_FILENAME
            ).write_text(json.dumps(ready), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content"):
                panderm_run.validate_validation_archive_cache(
                    common_wrong,
                    expected_file_content_identity_sha256=self._approved_content_identity(shared),
                    expected_fixed_split_identity="fixed-split",
                    expected_manifest_sha256=inventory["manifest_sha256"],
                    expected_class_mapping_sha256=
                        inventory["class_mapping_sha256"],
                )

            missing_ready_content = cache_parent / "missing_ready_content"
            shutil.copytree(base, missing_ready_content)
            ready_path = (
                missing_ready_content
                / panderm_run.VALIDATION_ARCHIVE_READY_FILENAME
            )
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            del ready["file_content_identity_sha256"]
            ready_path.write_text(json.dumps(ready), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "READY"):
                panderm_run.validate_validation_archive_cache(
                    missing_ready_content,
                    expected_file_content_identity_sha256=self._approved_content_identity(shared),
                    expected_fixed_split_identity="fixed-split",
                    expected_manifest_sha256=inventory["manifest_sha256"],
                    expected_class_mapping_sha256=
                        inventory["class_mapping_sha256"],
                )

            missing_ready = cache_parent / "missing_ready"
            shutil.copytree(base, missing_ready)
            (
                missing_ready / panderm_run.VALIDATION_ARCHIVE_READY_FILENAME
            ).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "inspect manually"):
                panderm_run.validate_validation_archive_cache(
                    missing_ready,
                    expected_file_content_identity_sha256=self._approved_content_identity(shared),
                    expected_fixed_split_identity="fixed-split",
                    expected_manifest_sha256=inventory["manifest_sha256"],
                    expected_class_mapping_sha256=
                        inventory["class_mapping_sha256"],
                )

    def test_failed_preflight_never_calls_staging_or_mutates_durable_files(self):
        valid_smoke = {
            key: True
            for key in {
                "official_preprocessing", "official_checkpoint_loaded",
                "cuda_forward", "backward", "all_12_blocks_have_gradients",
                "head_has_gradients", "fixed_pos_embed_has_no_gradient",
                "optimizer_coverage", "optimizer_step", "backbone_updated",
                "post_step_checkpoint_round_trip", "smoke_model_discarded",
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            durable = Path(temporary) / "existing.json"
            durable.write_text('{"existing": true}\n', encoding="utf-8")
            before = (
                durable.read_bytes(),
                durable.stat().st_mtime_ns,
            )
            staging = mock.Mock()
            with self.assertRaisesRegex(RuntimeError, "checkpoint"):
                panderm_run.stage_after_validation_preflights(
                    checkpoint_preflight={"weights_only_round_trip": False},
                    gpu_smoke=valid_smoke,
                    staging=staging,
                )
            staging.assert_not_called()
            self.assertEqual(
                (durable.read_bytes(), durable.stat().st_mtime_ns), before
            )
            self.assertFalse(
                (Path(temporary) / panderm_run.VALIDATION_ARCHIVE_READY_FILENAME)
                .exists()
            )

    def test_interrupted_archive_publish_never_writes_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = self._fixture(root)
            runtime = root / "runtime"
            runtime.mkdir()
            cache_parent = root / "cache_parent"
            cache_parent.mkdir()
            cache = cache_parent / "cache"
            with mock.patch.object(
                panderm_run,
                "_copy_file_with_progress",
                side_effect=RuntimeError("interrupted"),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    panderm_run.build_validation_archive_cache(
                        shared,
                        cache,
                        runtime,
                        expected_file_content_identity_sha256=self._approved_content_identity(shared),
                        source_fixed_split_identity="fixed-split",
                        expected_train_rows=1,
                        expected_val_rows=1,
                        expected_unique_images=2,
                        write_guard=AllowDurableWriteGuard().require,
                    )
            self.assertFalse(cache.exists())
            staging = list(cache_parent.glob(".cache.*.staging"))
            self.assertEqual(len(staging), 1)
            self.assertFalse(
                (
                    staging[0]
                    / panderm_run.VALIDATION_ARCHIVE_READY_FILENAME
                ).exists()
            )
            self.assertEqual(list(runtime.iterdir()), [])

    def test_archive_copy_heartbeat_reports_real_bytes_and_elapsed_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.tar"
            destination = root / "destination.tar"
            source.write_bytes(b"x" * (8 * 1024 * 1024 + 1))
            with (
                mock.patch.object(
                    panderm_run.time,
                    "perf_counter",
                    side_effect=[0.0, 61.0, 62.0],
                ),
                mock.patch("builtins.print") as print_mock,
            ):
                panderm_run._copy_file_with_progress(
                    source, destination, phase="archive-heartbeat-test"
                )
            messages = "\n".join(
                str(call.args[0]) for call in print_mock.call_args_list
            )
            self.assertIn("bytes=8388608/8388609", messages)
            self.assertIn("elapsed=61.0s", messages)
            self.assertIn("file=source.tar", messages)
            self.assertEqual(destination.read_bytes(), source.read_bytes())


class ApprovedContentIdentityTests(unittest.TestCase):
    """The archive must be judged against an anchor it cannot rewrite.

    Every content witness inside the cache -- the tar bytes, the per-file rows,
    the aggregate in ``archive_identity.json``, the archive SHA-256 and
    ``_READY.json`` -- can be regenerated together from tampered source, and the
    result is perfectly self-consistent. Validating them only against each other
    therefore proves nothing, which is exactly the hole these tests close: the
    approved digest is a separate reviewed input and is never read back from the
    artifact under validation.
    """

    def _fixture(self, root, *, mutate=None, swap=False):
        shared = root / "shared_data"
        mixed = shared / "raw" / "mixed"
        manifests = shared / "manifests"
        mixed.mkdir(parents=True)
        manifests.mkdir(parents=True)
        manifests.joinpath("class_to_idx.json").write_text(
            json.dumps(panderm_run.EXPECTED_CLASS_TO_IDX), encoding="utf-8"
        )
        payloads = {"train_a.jpg": b"train-payload-a", "val_a.jpg": b"val-payload-a"}
        if swap:
            payloads["train_a.jpg"], payloads["val_a.jpg"] = (
                payloads["val_a.jpg"], payloads["train_a.jpg"]
            )
        if mutate:
            payloads["train_a.jpg"] = mutate(payloads["train_a.jpg"])
        for name, body in payloads.items():
            mixed.joinpath(name).write_bytes(body)
        for split, image in (("train", "train_a.jpg"), ("val", "val_a.jpg")):
            with manifests.joinpath(f"{split}.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_path", "label_idx", "dx", "lesion_id", "image_id"
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "image_path": f"raw/mixed/{image}", "label_idx": 3, "dx": "df",
                    "lesion_id": f"{split}-lesion", "image_id": f"{split}-image",
                })
        return shared

    def _approved(self, shared):
        return panderm_run.validation_source_inventory(
            shared,
            expected_train_rows=1, expected_val_rows=1, expected_unique_images=2,
        )["file_content_identity"]["sha256"]

    def _build(self, shared, cache, runtime, approved):
        return panderm_run.build_validation_archive_cache(
            shared, cache, runtime,
            expected_file_content_identity_sha256=approved,
            source_fixed_split_identity="fixed-split",
            expected_train_rows=1, expected_val_rows=1, expected_unique_images=2,
            write_guard=AllowDurableWriteGuard().require,
        )

    def _validate(self, cache, identity, approved):
        return panderm_run.validate_validation_archive_cache(
            cache,
            expected_file_content_identity_sha256=approved,
            expected_fixed_split_identity="fixed-split",
            expected_manifest_sha256={
                "train": identity["train_manifest_sha256"],
                "val": identity["val_manifest_sha256"],
            },
            expected_class_mapping_sha256=identity["class_mapping_sha256"],
        )

    def _honest_cache(self, root, name="cache", **fixture):
        shared = self._fixture(root / name, **fixture)
        approved = self._approved(shared)
        parent = root / f"{name}_parent"
        parent.mkdir()
        runtime = root / f"{name}_runtime"
        runtime.mkdir()
        cache = parent / "cache"
        identity = self._build(shared, cache, runtime, approved)
        return shared, cache, runtime, identity, approved

    # --- the approved digest itself -----------------------------------------
    def test_reviewed_constant_is_a_real_64_hex_digest(self):
        value = panderm_run.EXPECTED_VALIDATION_CONTENT_IDENTITY_SHA256
        self.assertIs(type(value), str)
        self.assertEqual(len(value), 64)
        self.assertRegex(value, r"^[0-9a-f]{64}$")
        self.assertNotEqual(
            value, panderm_run.VALIDATION_CONTENT_IDENTITY_PLACEHOLDER
        )
        self.assertIs(
            type(panderm_run.EXPECTED_VALIDATION_CONTENT_MEMBER_COUNT), int
        )

    def test_missing_placeholder_and_malformed_expected_are_refused(self):
        for bad in (
            None,
            panderm_run.VALIDATION_CONTENT_IDENTITY_PLACEHOLDER,
            "",
            "abc",
            "A" * 64,
            "g" * 64,
            b"a" * 64,
            189,
        ):
            with self.subTest(expected=repr(bad)):
                with self.assertRaises(ValueError):
                    panderm_run.require_approved_content_identity(bad)

    def test_callers_cannot_omit_the_expected_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared, cache, runtime, identity, _ = self._honest_cache(root)
            with self.assertRaises(TypeError):
                panderm_run.validate_validation_archive_cache(
                    cache,
                    expected_fixed_split_identity="fixed-split",
                    expected_manifest_sha256={
                        "train": identity["train_manifest_sha256"],
                        "val": identity["val_manifest_sha256"],
                    },
                    expected_class_mapping_sha256=identity["class_mapping_sha256"],
                )
            with self.assertRaises(TypeError):
                panderm_run.build_validation_archive_cache(
                    shared, root / "other", runtime,
                    source_fixed_split_identity="fixed-split",
                )
            with self.assertRaises(TypeError):
                panderm_run.reuse_validation_archive_cache(
                    cache, runtime, runtime / "data",
                    expected_fixed_split_identity="fixed-split",
                    expected_manifest_sha256={
                        "train": identity["train_manifest_sha256"],
                        "val": identity["val_manifest_sha256"],
                    },
                    expected_class_mapping_sha256=identity["class_mapping_sha256"],
                )

    # --- honest archive ------------------------------------------------------
    def test_valid_source_tar_extraction_and_expected_all_agree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared, cache, runtime, identity, approved = self._honest_cache(root)
            self.assertEqual(identity["file_content_identity_sha256"], approved)
            self.assertEqual(self._validate(cache, identity, approved), identity)
            report = panderm_run.reuse_validation_archive_cache(
                cache, runtime, runtime / "data",
                expected_file_content_identity_sha256=approved,
                expected_fixed_split_identity="fixed-split",
                expected_manifest_sha256={
                    "train": identity["train_manifest_sha256"],
                    "val": identity["val_manifest_sha256"],
                },
                expected_class_mapping_sha256=identity["class_mapping_sha256"],
                write_guard=AllowDurableWriteGuard().require,
            )
            self.assertEqual(report["archive_files_copied"], 1)
            self.assertEqual(
                report["approved_file_content_identity_sha256"], approved
            )
            paths = [row["path"] for row in identity["file_content_rows"]]
            self.assertNotIn("manifests/test.csv", paths)

    def test_wrong_approved_expected_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, cache, _, identity, _ = self._honest_cache(root)
            with self.assertRaisesRegex(ValueError, "approved expected"):
                self._validate(cache, identity, "a" * 64)

    # --- coordinated archive and metadata rewrite ---------------------------
    def test_coordinated_tar_rows_aggregate_sha_and_ready_rewrite_is_rejected(self):
        """Case 15: everything inside the cache rewritten consistently."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, _, approved = self._honest_cache(root, name="honest")
            evil_shared, evil_cache, _, evil_identity, evil_approved = (
                self._honest_cache(
                    root, name="evil", mutate=lambda b: bytes([b[0] ^ 0xFF]) + b[1:]
                )
            )
            # The evil cache is fully self-consistent: it certifies itself.
            self.assertNotEqual(evil_approved, approved)
            self.assertEqual(
                self._validate(evil_cache, evil_identity, evil_approved),
                evil_identity,
            )
            # ...but it cannot satisfy the independent approved anchor.
            with self.assertRaisesRegex(ValueError, "approved expected"):
                self._validate(evil_cache, evil_identity, approved)

    def test_coordinated_extraction_identity_and_ready_rewrite_is_rejected(self):
        """Case 16: witness C rewritten together with identity and READY."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, _, approved = self._honest_cache(root, name="honest")
            _, evil_cache, evil_runtime, evil_identity, evil_approved = (
                self._honest_cache(
                    root, name="evil", mutate=lambda b: b"z" * len(b)
                )
            )
            local = evil_runtime / "data"
            with self.assertRaisesRegex(ValueError, "approved expected"):
                panderm_run.reuse_validation_archive_cache(
                    evil_cache, evil_runtime, local,
                    expected_file_content_identity_sha256=approved,
                    expected_fixed_split_identity="fixed-split",
                    expected_manifest_sha256={
                        "train": evil_identity["train_manifest_sha256"],
                        "val": evil_identity["val_manifest_sha256"],
                    },
                    expected_class_mapping_sha256=
                        evil_identity["class_mapping_sha256"],
                    write_guard=AllowDurableWriteGuard().require,
                )
            # Rejected before anything was published to the runtime data root.
            self.assertFalse(local.exists())

    def test_same_path_and_size_but_different_bytes_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, _, approved = self._honest_cache(root, name="honest")
            evil_shared = self._fixture(
                root / "evil", mutate=lambda b: b"Z" * len(b)
            )
            parent = root / "evil_parent"
            parent.mkdir()
            runtime = root / "evil_runtime"
            runtime.mkdir()
            # Identical member paths and byte sizes, different content.
            with self.assertRaisesRegex(ValueError, "approved expected"):
                self._build(evil_shared, parent / "cache", runtime, approved)

    def test_two_equal_size_images_swapped_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plain = self._fixture(root / "plain")
            approved = self._approved(plain)
            swapped = self._fixture(root / "swapped", swap=True)
            self.assertNotEqual(self._approved(swapped), approved)
            parent = root / "swapped_parent"
            parent.mkdir()
            runtime = root / "swapped_runtime"
            runtime.mkdir()
            with self.assertRaisesRegex(ValueError, "approved expected"):
                self._build(swapped, parent / "cache", runtime, approved)

    def test_persisted_rows_consistent_with_themselves_but_not_approved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, _, approved = self._honest_cache(root, name="honest")
            _, evil_cache, _, evil_identity, evil_approved = self._honest_cache(
                root, name="evil", mutate=lambda b: b[:-1] + bytes([b[-1] ^ 2])
            )
            persisted = json.loads(
                (evil_cache / panderm_run.VALIDATION_ARCHIVE_IDENTITY_FILENAME)
                .read_text(encoding="utf-8")
            )
            # Rows hash to their own stored aggregate...
            self.assertEqual(
                panderm_run._canonical_file_content_rows_sha256(
                    persisted["file_content_rows"]
                ),
                persisted["file_content_identity_sha256"],
            )
            # ...which is still not the approved digest.
            self.assertNotEqual(
                persisted["file_content_identity_sha256"], approved
            )
            with self.assertRaisesRegex(ValueError, "approved expected"):
                self._validate(evil_cache, evil_identity, approved)

    def test_placeholder_blocks_publish_reuse_and_extraction_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared, cache, runtime, identity, _ = self._honest_cache(root)
            placeholder = panderm_run.VALIDATION_CONTENT_IDENTITY_PLACEHOLDER
            with self.assertRaisesRegex(ValueError, "placeholder"):
                self._build(shared, root / "np", runtime, placeholder)
            with self.assertRaisesRegex(ValueError, "placeholder"):
                self._validate(cache, identity, placeholder)
            with self.assertRaisesRegex(ValueError, "placeholder"):
                panderm_run.reuse_validation_archive_cache(
                    cache, runtime, runtime / "data",
                    expected_file_content_identity_sha256=placeholder,
                    expected_fixed_split_identity="fixed-split",
                    expected_manifest_sha256={
                        "train": identity["train_manifest_sha256"],
                        "val": identity["val_manifest_sha256"],
                    },
                    expected_class_mapping_sha256=identity["class_mapping_sha256"],
                    write_guard=AllowDurableWriteGuard().require,
                )
            self.assertFalse((runtime / "data").exists())


class ValidationLockProviderTopologyTests(unittest.TestCase):
    def _probe(self, **overrides):
        value = {
            "id": "private-probe-id",
            "name": "private-probe.json",
            "mimeType": panderm_run.DRIVE_JSON_MIME_TYPE,
            "parents": [self.MY_DRIVE_ROOT_ID],
            "ownedByMe": True,
            "trashed": False,
        }
        value.update(overrides)
        return value

    def test_private_my_drive_probe_proves_api_and_fuse_account_alignment(self):
        probe = self._probe()
        accepted = panderm_run.require_drive_api_fuse_account_alignment(
            probe,
            expected_probe_name="private-probe.json",
            expected_root_id=self.MY_DRIVE_ROOT_ID,
        )
        self.assertEqual(accepted["status"], "passed")
        for mutation in (
            self._probe(ownedByMe=False),
            self._probe(driveId="shared-drive-id"),
            self._probe(parents=["different-root"]),
        ):
            with self.assertRaises(ValueError):
                panderm_run.require_drive_api_fuse_account_alignment(
                    mutation,
                    expected_probe_name="private-probe.json",
                    expected_root_id=self.MY_DRIVE_ROOT_ID,
                )

    def test_api_fuse_probe_is_bound_to_the_resolved_my_drive_root_folder_id(self):
        """The alias "root" is a query word, never an identity to compare against.

        Drive accepts "root" only inside a files.list parent clause and always
        answers with the account's real My Drive root folder id, so comparing a
        returned `parents` entry against the alias can never succeed for any
        account. Rejecting the alias outright is what stops that unpassable
        gate from being reintroduced, and the probe must still be bound to a
        concrete root so a probe created somewhere else is caught.
        """
        probe = self._probe()
        before = copy.deepcopy(probe)
        accepted = panderm_run.require_drive_api_fuse_account_alignment(
            probe,
            expected_probe_name="private-probe.json",
            expected_root_id=self.MY_DRIVE_ROOT_ID,
        )
        self.assertEqual(accepted, {"file_id": "private-probe-id", "status": "passed"})
        with self.assertRaisesRegex(ValueError, "query alias"):
            panderm_run.require_drive_api_fuse_account_alignment(
                probe,
                expected_probe_name="private-probe.json",
                expected_root_id="root",
            )
        # The literal alias never appears in a real provider record, so a probe
        # claiming it must not be accepted as living in the real root either.
        with self.assertRaisesRegex(ValueError, "parent identity drift"):
            panderm_run.require_drive_api_fuse_account_alignment(
                self._probe(parents=["root"]),
                expected_probe_name="private-probe.json",
                expected_root_id=self.MY_DRIVE_ROOT_ID,
            )
        for expected_root_id in ("", None, 0, b"root", ["root"]):
            with self.subTest(expected_root_id=expected_root_id):
                with self.assertRaisesRegex(ValueError, "must be non-empty"):
                    panderm_run.require_drive_api_fuse_account_alignment(
                        probe,
                        expected_probe_name="private-probe.json",
                        expected_root_id=expected_root_id,
                    )
        self.assertEqual(probe, before, "provider metadata must not be mutated")

    ROOT_ID = "root-folder-id"
    DRIVE_ID = "shared-drive-id"
    RUN_UUID = "765b971f-d148-4960-a77d-b73f28fc013c"
    # A real My Drive root folder id observed from a live Colab account; the
    # API reports this shape, never the "root" alias that queries accept.
    MY_DRIVE_ROOT_ID = "0AN3kCPQfWpI4Uk9PVA"
    SHARED_ROOT_ALIAS = "ddpm-derm-panderm-runs"

    def _root_metadata(self, **overrides):
        value = {
            "id": self.ROOT_ID,
            "name": "ddpm-derm-panderm-runs",
            "mimeType": panderm_run.DRIVE_FOLDER_MIME_TYPE,
            "parents": ["my-drive-root"],
            "driveId": None,
            "ownedByMe": True,
            "trashed": False,
        }
        value.update(overrides)
        return value

    def _child_metadata(self, name, mime_type, **overrides):
        value = {
            "id": f"{name}-id",
            "name": name,
            "mimeType": mime_type,
            "parents": [self.ROOT_ID],
            "driveId": None,
            "ownedByMe": True,
            "trashed": False,
        }
        value.update(overrides)
        return value

    def _owned_topology(self):
        return panderm_run.require_validation_lock_storage_topology(
            self._root_metadata(),
            self._child_metadata("probe.json", panderm_run.DRIVE_JSON_MIME_TYPE),
            expected_root_id=self.ROOT_ID,
            expected_probe_name="probe.json",
        )

    def _shared_topology(self):
        return panderm_run.require_validation_lock_storage_topology(
            self._root_metadata(driveId=self.DRIVE_ID, ownedByMe=False),
            self._child_metadata(
                "probe.json",
                panderm_run.DRIVE_JSON_MIME_TYPE,
                driveId=self.DRIVE_ID,
                ownedByMe=False,
            ),
            expected_root_id=self.ROOT_ID,
            expected_probe_name="probe.json",
        )

    def _active_metadata(self, parent_id="version-id", **overrides):
        value = {
            "id": "active-session-id",
            "name": panderm_run.ACTIVE_SESSION_FILENAME,
            "mimeType": panderm_run.DRIVE_JSON_MIME_TYPE,
            "parents": [parent_id],
            "driveId": None,
            "ownedByMe": True,
            "trashed": False,
        }
        value.update(overrides)
        return value

    def _marker(self, session_id, **overrides):
        value = {
            "schema_version": panderm_run.VALIDATION_ARCHIVE_SCHEMA_VERSION,
            "session_id": session_id,
            "run_version": panderm_run.RUN_VERSION,
            "git_commit": "c" * 40,
            "shared_root_uuid": self.RUN_UUID,
            "evaluation_scope": panderm_run.VALIDATION_ONLY,
            "account_label": "A",
            "hostname": "runtime",
            "acquired_utc": "2026-07-29T07:35:20+00:00",
        }
        value.update(overrides)
        return value

    def test_provider_shortcut_target_id_is_required_and_exact(self):
        shortcut = self._child_metadata(
            "ddpm-derm-panderm-runs",
            panderm_run.DRIVE_SHORTCUT_MIME_TYPE,
            shortcutDetails={
                "targetId": self.ROOT_ID,
                "targetMimeType": panderm_run.DRIVE_FOLDER_MIME_TYPE,
                "targetResourceKey": "target-resource-key",
            },
        )
        target = panderm_run.require_drive_shortcut_target(
            [shortcut], expected_alias="ddpm-derm-panderm-runs"
        )
        self.assertEqual(
            target,
            {
                "target_id": self.ROOT_ID,
                "target_resource_key": "target-resource-key",
            },
        )
        self.assertEqual(
            panderm_run.require_drive_shortcut_target_id(
                [shortcut], expected_alias="ddpm-derm-panderm-runs"
            ),
            self.ROOT_ID,
        )
        for records in (
            [],
            [shortcut, dict(shortcut, id="duplicate-shortcut")],
            [dict(shortcut, name="wrong-alias")],
            [dict(shortcut, shortcutDetails={})],
            [
                dict(
                    shortcut,
                    shortcutDetails={
                        "targetId": self.ROOT_ID,
                        "targetMimeType": panderm_run.DRIVE_JSON_MIME_TYPE,
                        "targetResourceKey": "target-resource-key",
                    },
                )
            ],
        ):
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    panderm_run.require_drive_shortcut_target_id(
                        records, expected_alias="ddpm-derm-panderm-runs"
                    )

    def test_shortcut_resource_key_is_optional_but_drift_is_still_rejected(self):
        """A keyless shortcut target must resolve; a malformed one must not.

        Drive omits targetResourceKey when the target has no resource key, which
        is the normal case for a folder shared directly with accounts A, B and C
        rather than by a pre-2021 link. What Drive reports also depends on how
        the calling account obtained access, so requiring a key would lock out
        exactly the account rotation this run version is built around. A present
        but malformed value is still evidence of a broken provider record.
        """
        shortcut = self._child_metadata(
            "ddpm-derm-panderm-runs",
            panderm_run.DRIVE_SHORTCUT_MIME_TYPE,
            shortcutDetails={
                "targetId": self.ROOT_ID,
                "targetMimeType": panderm_run.DRIVE_FOLDER_MIME_TYPE,
                "targetResourceKey": "first-resource-key",
            },
        )
        keyless = copy.deepcopy(shortcut)
        keyless["shortcutDetails"].pop("targetResourceKey")
        self.assertEqual(
            panderm_run.require_drive_shortcut_target(
                [keyless], expected_alias="ddpm-derm-panderm-runs"
            ),
            {"target_id": self.ROOT_ID, "target_resource_key": ""},
        )
        for invalid in ("", 1, True, [], {}):
            broken = copy.deepcopy(shortcut)
            broken["shortcutDetails"]["targetResourceKey"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "resource key is invalid"
            ):
                panderm_run.require_drive_shortcut_target(
                    [broken], expected_alias="ddpm-derm-panderm-runs"
                )
        self.assertNotEqual(
            panderm_run.require_drive_shortcut_target(
                [shortcut], expected_alias="ddpm-derm-panderm-runs"
            )["target_resource_key"],
            panderm_run.require_drive_shortcut_target(
                [
                    {
                        **shortcut,
                        "shortcutDetails": {
                            **shortcut["shortcutDetails"],
                            "targetResourceKey": "second-resource-key",
                        },
                    }
                ],
                expected_alias="ddpm-derm-panderm-runs",
            )["target_resource_key"],
        )
        with self.assertRaisesRegex(ValueError, "resource key drift"):
            panderm_run.drive_resource_key_header(
                [
                    (self.ROOT_ID, "first-resource-key"),
                    (self.ROOT_ID, "second-resource-key"),
                ]
            )

    def _granted_account_shortcut(self, **overrides):
        """One granted account's My Drive entry: a shortcut, no resource key."""
        value = self._child_metadata(
            self.SHARED_ROOT_ALIAS,
            panderm_run.DRIVE_SHORTCUT_MIME_TYPE,
            id="granted-account-shortcut-id",
            parents=[self.MY_DRIVE_ROOT_ID],
            shortcutDetails={
                "targetId": self.ROOT_ID,
                "targetMimeType": panderm_run.DRIVE_FOLDER_MIME_TYPE,
            },
        )
        value.update(overrides)
        return value

    def _owner_account_folder(self, **overrides):
        """The owner account's My Drive entry: the folder itself, no shortcut."""
        value = self._child_metadata(
            self.SHARED_ROOT_ALIAS,
            panderm_run.DRIVE_FOLDER_MIME_TYPE,
            id=self.ROOT_ID,
            parents=[self.MY_DRIVE_ROOT_ID],
            ownedByMe=True,
        )
        value.update(overrides)
        return value

    def test_shared_run_root_resolves_from_either_my_drive_shape_to_one_id(self):
        """The owner has no shortcut to its own folder, so demanding one locks it out.

        This run version is designed to be handed between accounts A, B and C
        against one physical folder. A granted account holds a shortcut, while
        the account that owns the folder holds the folder itself and Drive will
        never give it a shortcut to its own item. Requiring a shortcut makes the
        workflow run only for the accounts that do not own the data, which is
        the opposite of account-neutral. Both shapes must therefore resolve to
        the same folder id, because that id is what every later provider check,
        the sentinel binding and the durable identity all consume.
        """
        shortcut = self._granted_account_shortcut()
        folder = self._owner_account_folder()
        before = copy.deepcopy([shortcut, folder])
        granted = panderm_run.require_drive_shared_root_target(
            [shortcut], [], expected_alias=self.SHARED_ROOT_ALIAS
        )
        owner = panderm_run.require_drive_shared_root_target(
            [], [folder], expected_alias=self.SHARED_ROOT_ALIAS
        )
        self.assertEqual(
            granted,
            {
                "target_id": self.ROOT_ID,
                "target_resource_key": "",
                "source": panderm_run.DRIVE_SHARED_ROOT_SOURCE_SHORTCUT,
            },
        )
        self.assertEqual(
            owner,
            {
                "target_id": self.ROOT_ID,
                "target_resource_key": "",
                "source": panderm_run.DRIVE_SHARED_ROOT_SOURCE_OWNED_FOLDER,
            },
        )
        self.assertEqual(granted["target_id"], owner["target_id"])
        # The resolved pair must stay consumable by the durable identity, so a
        # keyless owner-resolved root still binds without inventing a key.
        identity = panderm_run.build_durable_root_provider_identity(
            self._root_metadata(),
            expected_root_id=owner["target_id"],
            shortcut_target_resource_key=owner["target_resource_key"],
            shared_root_uuid=self.RUN_UUID,
            topology=self._owned_topology(),
        )
        self.assertEqual(identity["root_file_id"], self.ROOT_ID)
        self.assertEqual(identity["root_resource_key"], "")
        self.assertEqual([shortcut, folder], before)

    def test_shared_run_root_rejects_ambiguous_missing_and_unowned_shapes(self):
        """Relaxing the shape must not relax which folder is allowed to win.

        Accepting a same-named My Drive folder is only safe while exactly one
        candidate exists and the account owns it. A shortcut and a same-named
        folder together is the private replacement folder the shared-root rules
        forbid, and silently preferring either one would point a whole run at
        the wrong physical root.
        """
        shortcut = self._granted_account_shortcut()
        folder = self._owner_account_folder()
        cases = {
            "shortcut_and_folder": ([shortcut], [folder], "private replacement"),
            "nothing_visible": ([], [], "exactly one"),
            "duplicate_folders": (
                [],
                [folder, self._owner_account_folder(id="duplicate-folder-id")],
                "exactly one",
            ),
            "folder_not_owned": (
                [],
                [self._owner_account_folder(ownedByMe=False)],
                "account owns it",
            ),
            "folder_ownership_unknown": (
                [],
                [self._owner_account_folder(ownedByMe=None)],
                "account owns it",
            ),
            "folder_alias_drift": (
                [],
                [self._owner_account_folder(name="ddpm-derm-panderm-runs-copy")],
                "name drift",
            ),
            "folder_trashed": (
                [],
                [self._owner_account_folder(trashed=True)],
                "trashed=false",
            ),
            "folder_resource_key_invalid": (
                [],
                [self._owner_account_folder(resourceKey="")],
                "resource key is invalid",
            ),
            "duplicate_shortcuts": (
                [shortcut, self._granted_account_shortcut(id="duplicate-shortcut")],
                [],
                "exactly one",
            ),
        }
        before = copy.deepcopy([shortcut, folder])
        for name, (shortcuts, folders, message) in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, message):
                    panderm_run.require_drive_shared_root_target(
                        shortcuts, folders, expected_alias=self.SHARED_ROOT_ALIAS
                    )
        for shortcuts, folders in ((shortcut, []), ([], "folder"), (None, [])):
            with self.subTest(shortcuts=shortcuts, folders=folders):
                with self.assertRaisesRegex(ValueError, "must be a sequence"):
                    panderm_run.require_drive_shared_root_target(
                        shortcuts, folders, expected_alias=self.SHARED_ROOT_ALIAS
                    )
        self.assertEqual([shortcut, folder], before)

    def test_durable_root_fingerprint_is_identical_across_accounts(self):
        """One physical folder must fingerprint identically for every account.

        Drive only reports the parents the calling account can itself see: the
        owner sees its own My Drive root folder id, while a granted account is
        given nothing because it cannot see the owner's root. Projecting that
        field would make the same durable root fingerprint differently on every
        account switch, which is unreadable evidence in a workflow whose whole
        point is handing one run between accounts A, B and C.
        """
        topology = self._owned_topology()
        owner_view = self._root_metadata(parents=[self.MY_DRIVE_ROOT_ID])
        granted_view = self._root_metadata(ownedByMe=False)
        granted_view.pop("parents")
        fingerprints = set()
        for label, root in (("owner", owner_view), ("granted", granted_view)):
            with self.subTest(account=label):
                identity = panderm_run.build_durable_root_provider_identity(
                    root,
                    expected_root_id=self.ROOT_ID,
                    shortcut_target_resource_key="",
                    shared_root_uuid=self.RUN_UUID,
                    topology=topology,
                )
                self.assertEqual(identity["root_file_id"], self.ROOT_ID)
                fingerprints.add(identity["provider_fingerprint"])
        self.assertEqual(len(fingerprints), 1, "fingerprint drifted between accounts")

    def test_true_shared_drive_and_account_neutral_shared_my_drive_are_supported(self):
        owned = self._owned_topology()
        shared = self._shared_topology()
        self.assertEqual(
            owned["mode"],
            panderm_run.VALIDATION_LOCK_TOPOLOGY_SHARED_MY_DRIVE,
        )
        self.assertEqual(
            shared["mode"],
            panderm_run.VALIDATION_LOCK_TOPOLOGY_SHARED_DRIVE,
        )
        self.assertIsNone(owned["drive_id"])
        self.assertEqual(shared["drive_id"], self.DRIVE_ID)

    def test_shared_my_drive_root_owner_is_neutral_but_api_fuse_probe_must_align(self):
        account_neutral = panderm_run.require_validation_lock_storage_topology(
            self._root_metadata(ownedByMe=False),
            self._child_metadata(
                "probe.json",
                panderm_run.DRIVE_JSON_MIME_TYPE,
                ownedByMe=True,
            ),
            expected_root_id=self.ROOT_ID,
            expected_probe_name="probe.json",
        )
        self.assertEqual(
            account_neutral["mode"],
            panderm_run.VALIDATION_LOCK_TOPOLOGY_SHARED_MY_DRIVE,
        )
        cases = {
            "probe_not_owned": (
                self._root_metadata(),
                self._child_metadata(
                    "probe.json",
                    panderm_run.DRIVE_JSON_MIME_TYPE,
                    ownedByMe=False,
                ),
            ),
            "probe_drive_mismatch": (
                self._root_metadata(),
                self._child_metadata(
                    "probe.json",
                    panderm_run.DRIVE_JSON_MIME_TYPE,
                    driveId=self.DRIVE_ID,
                ),
            ),
        }
        for name, (root, probe) in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(
                    ValueError, "API/FUSE account mismatch|disagree"
                ):
                    panderm_run.require_validation_lock_storage_topology(
                        root,
                        probe,
                        expected_root_id=self.ROOT_ID,
                        expected_probe_name="probe.json",
                    )

    def test_root_probe_and_shared_drive_identity_drift_are_rejected(self):
        cases = {
            "root_id": (
                self._root_metadata(id="other-root"),
                self._child_metadata(
                    "probe.json",
                    panderm_run.DRIVE_JSON_MIME_TYPE,
                ),
                "root provider id drift",
            ),
            "probe_parent": (
                self._root_metadata(),
                self._child_metadata(
                    "probe.json",
                    panderm_run.DRIVE_JSON_MIME_TYPE,
                    parents=["other-root"],
                ),
                "parent identity drift",
            ),
            "shared_drive": (
                self._root_metadata(driveId=self.DRIVE_ID, ownedByMe=False),
                self._child_metadata(
                    "probe.json",
                    panderm_run.DRIVE_JSON_MIME_TYPE,
                    driveId="other-drive",
                    ownedByMe=False,
                ),
                "escaped",
            ),
        }
        for name, (root, probe, message) in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, message):
                    panderm_run.require_validation_lock_storage_topology(
                        root,
                        probe,
                        expected_root_id=self.ROOT_ID,
                        expected_probe_name="probe.json",
                    )

    def test_drive_list_request_shape_is_dynamic_and_shared_drive_scoped(self):
        common = {
            "query": "'root' in parents",
            "fields": "nextPageToken,incompleteSearch,files(id)",
            "page_token": None,
        }
        my_drive = panderm_run.drive_provider_list_request_kwargs(
            **common, drive_id=None
        )
        shared_drive = panderm_run.drive_provider_list_request_kwargs(
            **common, drive_id=self.DRIVE_ID
        )
        self.assertEqual(my_drive["corpora"], "user")
        self.assertNotIn("driveId", my_drive)
        self.assertEqual(shared_drive["corpora"], "drive")
        self.assertEqual(shared_drive["driveId"], self.DRIVE_ID)
        for request in (my_drive, shared_drive):
            self.assertIs(request["supportsAllDrives"], True)
            self.assertIs(request["includeItemsFromAllDrives"], True)

    def test_drive_list_paginates_and_incomplete_namespace_fails_loud(self):
        pages = {
            None: {
                "files": [{"id": "first"}],
                "incompleteSearch": False,
                "nextPageToken": "page-2",
            },
            "page-2": {
                "files": [{"id": "second"}],
                "incompleteSearch": False,
            },
        }
        calls = []
        records = panderm_run.collect_drive_provider_pages(
            lambda token: calls.append(token) or pages[token]
        )
        self.assertEqual([record["id"] for record in records], ["first", "second"])
        self.assertEqual(calls, [None, "page-2"])
        for response in (
            {"files": [], "incompleteSearch": True},
            {"files": []},
        ):
            with self.subTest(response=response):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    panderm_run.collect_drive_provider_pages(
                        lambda token, response=response: response
                    )

    def test_root_provider_identity_binds_shortcut_resource_key(self):
        topology = self._owned_topology()
        root = self._root_metadata(
            ownedByMe=False, resourceKey="target-resource-key"
        )
        identity = panderm_run.build_durable_root_provider_identity(
            root,
            expected_root_id=self.ROOT_ID,
            shortcut_target_resource_key="target-resource-key",
            shared_root_uuid=self.RUN_UUID,
            topology=topology,
        )
        self.assertEqual(identity["root_file_id"], self.ROOT_ID)
        self.assertEqual(identity["root_resource_key"], "target-resource-key")
        self.assertEqual(
            identity["topology_mode"],
            panderm_run.VALIDATION_LOCK_TOPOLOGY_SHARED_MY_DRIVE,
        )
        with self.assertRaisesRegex(ValueError, "resource key drift"):
            panderm_run.build_durable_root_provider_identity(
                root,
                expected_root_id=self.ROOT_ID,
                shortcut_target_resource_key="different-resource-key",
                shared_root_uuid=self.RUN_UUID,
                topology=topology,
            )

    def test_root_provider_identity_binds_a_keyless_root_without_collapsing_it(self):
        """A keyless root must bind, and must not fingerprint as a keyed one.

        This check exists to prove the shortcut and the provider record describe
        the same folder, not to prove a resource key exists. Requiring one locks
        out directly shared roots; ignoring the field entirely would let a keyed
        and a keyless record be treated as the same durable root.
        """
        topology = self._owned_topology()
        keyless_root = self._root_metadata(ownedByMe=False)
        keyed_root = self._root_metadata(
            ownedByMe=False, resourceKey="target-resource-key"
        )
        keyless = panderm_run.build_durable_root_provider_identity(
            keyless_root,
            expected_root_id=self.ROOT_ID,
            shortcut_target_resource_key="",
            shared_root_uuid=self.RUN_UUID,
            topology=topology,
        )
        self.assertEqual(keyless["root_resource_key"], "")
        self.assertEqual(keyless["root_file_id"], self.ROOT_ID)
        keyed = panderm_run.build_durable_root_provider_identity(
            keyed_root,
            expected_root_id=self.ROOT_ID,
            shortcut_target_resource_key="target-resource-key",
            shared_root_uuid=self.RUN_UUID,
            topology=topology,
        )
        self.assertNotEqual(
            keyless["provider_fingerprint"],
            keyed["provider_fingerprint"],
            "a keyless root must not fingerprint as a keyed root",
        )
        for root, shortcut_key in (
            (keyless_root, "target-resource-key"),
            (keyed_root, ""),
        ):
            with self.subTest(shortcut_key=shortcut_key), self.assertRaisesRegex(
                ValueError, "resource key drift"
            ):
                panderm_run.build_durable_root_provider_identity(
                    root,
                    expected_root_id=self.ROOT_ID,
                    shortcut_target_resource_key=shortcut_key,
                    shared_root_uuid=self.RUN_UUID,
                    topology=topology,
                )

    def test_provider_version_visibility_and_identity_are_reconciled(self):
        topology = self._owned_topology()
        version = self._child_metadata(
            panderm_run.RUN_VERSION,
            panderm_run.DRIVE_FOLDER_MIME_TYPE,
        )
        observed = panderm_run.require_validation_version_provider_state(
            [version],
            expected_parent_id=self.ROOT_ID,
            run_version=panderm_run.RUN_VERSION,
            topology=topology,
            local_version_exists=True,
        )
        self.assertEqual(observed["id"], version["id"])
        self.assertIsNone(
            panderm_run.require_validation_version_provider_state(
                [],
                expected_parent_id=self.ROOT_ID,
                run_version=panderm_run.RUN_VERSION,
                topology=topology,
                local_version_exists=False,
            )
        )
        with self.assertRaisesRegex(ValueError, "invisible through FUSE"):
            panderm_run.require_validation_version_provider_state(
                [version],
                expected_parent_id=self.ROOT_ID,
                run_version=panderm_run.RUN_VERSION,
                topology=topology,
                local_version_exists=False,
            )
        with self.assertRaisesRegex(ValueError, "ambiguous duplicate"):
            panderm_run.require_validation_version_provider_state(
                [version, dict(version, id="duplicate-version")],
                expected_parent_id=self.ROOT_ID,
                run_version=panderm_run.RUN_VERSION,
                topology=topology,
                local_version_exists=True,
            )

    def test_provider_visible_but_fuse_invisible_active_marker_blocks_new_session(self):
        topology = self._owned_topology()
        active = self._active_metadata()
        with self.assertRaisesRegex(
            FileExistsError,
            "even if the FUSE alias cannot see it",
        ):
            panderm_run.require_active_session_provider_state(
                [active],
                expected_parent_id="version-id",
                topology=topology,
                expected_present=False,
            )
        self.assertIsNone(
            panderm_run.require_active_session_provider_state(
                [],
                expected_parent_id="version-id",
                topology=topology,
                expected_present=False,
            )
        )

    def test_provider_active_marker_missing_duplicate_parent_and_id_drift_reject(self):
        topology = self._owned_topology()
        active = self._active_metadata()
        with self.assertRaises(FileNotFoundError):
            panderm_run.require_active_session_provider_state(
                [],
                expected_parent_id="version-id",
                topology=topology,
                expected_present=True,
            )
        with self.assertRaisesRegex(ValueError, "ambiguous duplicate"):
            panderm_run.require_active_session_provider_state(
                [active, dict(active, id="duplicate-active")],
                expected_parent_id="version-id",
                topology=topology,
                expected_present=True,
            )
        with self.assertRaisesRegex(ValueError, "parent identity drift"):
            panderm_run.require_active_session_provider_state(
                [dict(active, parents=["other-version"])],
                expected_parent_id="version-id",
                topology=topology,
                expected_present=True,
            )
        with self.assertRaisesRegex(ValueError, "file identity drift"):
            panderm_run.require_active_session_provider_state(
                [active],
                expected_parent_id="version-id",
                topology=topology,
                expected_present=True,
                expected_file_id="other-active-id",
            )
        cross_owner = panderm_run.require_active_session_provider_state(
            [dict(active, ownedByMe=False)],
            expected_parent_id="version-id",
            topology=topology,
            expected_present=True,
        )
        self.assertEqual(cross_owner["id"], active["id"])


class SequentialHandoffTests(unittest.TestCase):
    ROOT_UUID = "765b971f-d148-4960-a77d-b73f28fc013c"
    COMMIT = "c" * 40

    def test_concurrent_candidate_files_are_absent(self):
        root = Path(__file__).resolve().parents[1]
        removed = (
            "apps_script/panderm_coordinator/Code.gs",
            "apps_script/panderm_coordinator/appsscript.json",
            "docs/panderm_coordinator_deployment.md",
            "scripts/panderm_coordinator_release.py",
            "src/ddpm_derm/panderm_coordinator.py",
            "tests/js/test_panderm_coordinator.js",
            "tests/test_panderm_coordinator.py",
        )
        self.assertEqual(
            [relative for relative in removed if (root / relative).exists()],
            [],
        )

    def _paths(self, base):
        version = Path(base) / panderm_run.RUN_VERSION
        version.mkdir()
        history = version / panderm_run.SESSION_HISTORY_DIRECTORY
        history.mkdir()
        return (
            panderm_run.active_session_path(base),
            history,
            panderm_run.canonical_identity_sha256(identity()),
        )

    def _start(self, marker, history, run_hash, account, session, *, takeover=False):
        return panderm_run.start_sequential_session(
            marker,
            session_id=session,
            run_version=panderm_run.RUN_VERSION,
            git_commit=self.COMMIT,
            shared_root_uuid=self.ROOT_UUID,
            account_label=account,
            run_identity_sha256=run_hash,
            manual_takeover_confirmed=takeover,
            history_directory=history,
        )

    def test_graceful_a_to_b_preserves_audit_and_same_run_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker, history, run_hash = self._paths(temporary)
            session_a = str(uuid.uuid4())
            session_b = str(uuid.uuid4())
            active_a = self._start(marker, history, run_hash, "A", session_a)
            completion = panderm_run.complete_sequential_session(
                marker,
                session_id=session_a,
                history_directory=history,
                checkpoint_integrity={"epoch": 2, "global_step": 16, "sha256": "a" * 64},
                result_identity={"epoch": 2, "global_step": 16, "run_identity_sha256": run_hash},
            )
            self.assertEqual(completion["active_session"], active_a)
            self.assertFalse(marker.exists())
            self.assertTrue((history / f"{session_a}.active.json").is_file())
            self.assertTrue((history / f"{session_a}.completed.json").is_file())
            active_b = self._start(marker, history, run_hash, "B", session_b)
            self.assertEqual(active_b["run_identity_sha256"], run_hash)
            self.assertEqual(active_b["shared_root_uuid"], self.ROOT_UUID)
            self.assertEqual(active_b["checkpoint_cadence"], "every_epoch")
            self.assertEqual(active_b["maximum_quota_loss"], "one_incomplete_epoch")

    def test_abrupt_stop_requires_explicit_manual_takeover_and_preserves_old_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker, history, run_hash = self._paths(temporary)
            session_a = str(uuid.uuid4())
            session_b = str(uuid.uuid4())
            self._start(marker, history, run_hash, "A", session_a)
            before = marker.read_bytes()
            abandoned_temp = marker.parent / ".last.pt.interrupted.tmp"
            abandoned_temp.write_bytes(b"partial")
            with self.assertRaisesRegex(FileExistsError, "confirm"):
                self._start(marker, history, run_hash, "B", session_b)
            self.assertEqual(marker.read_bytes(), before)
            self.assertEqual(list(history.iterdir()), [])
            active_b = self._start(
                marker, history, run_hash, "B", session_b, takeover=True
            )
            audit = history / f"{session_a}.takeover.json"
            self.assertTrue(audit.is_file())
            recorded = json.loads(audit.read_text(encoding="utf-8"))
            self.assertEqual(
                recorded["previous_active_session"]["session_id"], session_a
            )
            self.assertEqual(
                recorded["event_id"],
                panderm_run.audit_event_id(
                    panderm_run.MANUAL_TAKEOVER_EVENT,
                    run_version=panderm_run.RUN_VERSION,
                    subject_session_id=session_a,
                ),
            )
            self.assertEqual(active_b["session_id"], session_b)
            self.assertTrue(abandoned_temp.is_file())

    def test_conflicting_audit_blocks_takeover_without_replacing_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker, history, run_hash = self._paths(temporary)
            session_a = str(uuid.uuid4())
            session_b = str(uuid.uuid4())
            self._start(marker, history, run_hash, "A", session_a)
            before = marker.read_bytes()
            audit = history / f"{session_a}.takeover.json"
            # A well-formed audit whose stable fields disagree must never be
            # silently replaced, even though its replacement id is reusable.
            audit.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "event": panderm_run.MANUAL_TAKEOVER_EVENT,
                        "event_id": panderm_run.audit_event_id(
                            panderm_run.MANUAL_TAKEOVER_EVENT,
                            run_version=panderm_run.RUN_VERSION,
                            subject_session_id=session_a,
                        ),
                        "confirmation": "SOMETHING ELSE",
                        "confirmed_utc": "2026-01-01T00:00:00Z",
                        # The retired snapshot must be the real marker so this
                        # stays a stable-field conflict rather than a record the
                        # schema guard rejects before the conflict is reached.
                        "previous_active_session": json.loads(
                            before.decode("utf-8")
                        ),
                        "replacement_session_id": session_b,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(FileExistsError):
                self._start(
                    marker, history, run_hash, "B", session_b, takeover=True
                )
            self.assertEqual(marker.read_bytes(), before)

    def test_active_session_guard_rejects_identity_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker, history, run_hash = self._paths(temporary)
            session = str(uuid.uuid4())
            active = self._start(marker, history, run_hash, "A", session)
            accepted = panderm_run.require_active_session_identity(
                active,
                session_id=session,
                run_version=panderm_run.RUN_VERSION,
                git_commit=self.COMMIT,
                shared_root_uuid=self.ROOT_UUID,
                run_identity_sha256=run_hash,
            )
            self.assertEqual(accepted, active)
            for field, value in (
                ("session_id", str(uuid.uuid4())),
                ("run_version", "other-version"),
                ("git_commit", "d" * 40),
                ("shared_root_uuid", str(uuid.uuid4())),
                ("run_identity_sha256", "e" * 64),
            ):
                kwargs = {
                    "session_id": session,
                    "run_version": panderm_run.RUN_VERSION,
                    "git_commit": self.COMMIT,
                    "shared_root_uuid": self.ROOT_UUID,
                    "run_identity_sha256": run_hash,
                }
                kwargs[field] = value
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, "identity drift"
                ):
                    panderm_run.require_active_session_identity(active, **kwargs)


class SequentialHandoffBlockerRegressionTests(unittest.TestCase):
    """Session-transfer regressions for provider visibility, ownership,
    active markers, and retry recovery.
    """

    ROOT_UUID = "765b971f-d148-4960-a77d-b73f28fc013c"
    COMMIT = "c" * 40
    PREVIOUS_COMMIT = "a" * 40

    # --- shared fixtures ---------------------------------------------------
    def _session_paths(self, base):
        version = Path(base) / panderm_run.RUN_VERSION
        version.mkdir()
        history = version / panderm_run.SESSION_HISTORY_DIRECTORY
        history.mkdir()
        return (
            panderm_run.active_session_path(base),
            history,
            panderm_run.canonical_identity_sha256(identity()),
        )

    def _start(
        self,
        marker,
        history,
        run_hash,
        account,
        session,
        *,
        takeover=False,
        git_commit=None,
    ):
        return panderm_run.start_sequential_session(
            marker,
            session_id=session,
            run_version=panderm_run.RUN_VERSION,
            git_commit=self.COMMIT if git_commit is None else git_commit,
            shared_root_uuid=self.ROOT_UUID,
            account_label=account,
            run_identity_sha256=run_hash,
            manual_takeover_confirmed=takeover,
            history_directory=history,
        )

    def _components(self):
        model = build_mock_model()
        optimizer = panderm.build_optimizer(model, num_layers=4)
        schedule = panderm.WarmupCosineSchedule(
            optimizer, warmup_epochs=1, epochs=2, steps_per_epoch=2
        )
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        return model, optimizer, schedule, scaler

    def _write_checkpoint_pair(self, directory, run_identity, *, epoch, perturb=0.0):
        """Save one production checkpoint named ``last.pt`` with a real sidecar."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model, optimizer, schedule, scaler = self._components()
        if perturb:
            with torch.no_grad():
                next(model.parameters()).add_(perturb)
        path = directory / "last.pt"
        train_panderm.save_checkpoint(
            path,
            model,
            optimizer,
            schedule,
            scaler,
            epoch,
            0.5,
            [
                {"epoch": completed, "optimizer_steps": schedule.step_count}
                for completed in range(1, epoch + 1)
            ],
            type("A", (), {"seed": 0, "epochs": 5})(),
            run_identity,
            write_guard=AllowDurableWriteGuard(),
        )
        return path, train_panderm.checkpoint_integrity_path(path)

    def _install_backup(self, final_path, source_pair, previous_id, *, sidecar_edit=None):
        """Copy a saved pair into ``final_path``'s directory as a predecessor."""
        source_checkpoint, source_sidecar = source_pair
        prefix = f".{final_path.name}.previous.{previous_id}"
        backup = final_path.parent / f"{prefix}.pt"
        backup_sidecar = final_path.parent / f"{prefix}.integrity.json"
        shutil.copy2(source_checkpoint, backup)
        record = json.loads(source_sidecar.read_text(encoding="utf-8"))
        if sidecar_edit is not None:
            record = sidecar_edit(record)
        backup_sidecar.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return backup, backup_sidecar

    def _corrupt_final(self, final_path):
        final_path.write_bytes(b"interrupted-publish")
        train_panderm.checkpoint_integrity_path(final_path).write_text(
            '{"partial":true}\n', encoding="utf-8"
        )

    class _MutationRecorder:
        """Fail the test if any resumable component state is touched."""

        def __init__(self, model, optimizer, schedule, scaler):
            self.mutations = []
            self._patches = [
                mock.patch.object(
                    component,
                    "load_state_dict",
                    side_effect=lambda *a, **k: self.mutations.append(name),
                )
                for name, component in (
                    ("model", model),
                    ("optimizer", optimizer),
                    ("scheduler", schedule),
                    ("scaler", scaler),
                )
            ]
            self._patches.append(
                mock.patch.object(
                    train_panderm.random,
                    "setstate",
                    side_effect=lambda *a, **k: self.mutations.append("rng"),
                )
            )

        def __enter__(self):
            for patch in self._patches:
                patch.start()
            return self

        def __exit__(self, *exc_info):
            for patch in self._patches:
                patch.stop()
            return False

    # --- blocker 3 ---------------------------------------------------------
    def test_three_way_same_step_backup_divergence_is_never_ignored(self):
        """probe three_way_same_step_ambiguity_accepted must be False.

        Comparing only the first two candidates let a third divergent backup at
        the same (epoch, global_step) be silently restored, so a resume could
        continue from bytes that were never the published state.
        """
        run_identity = identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final_path, final_sidecar = self._write_checkpoint_pair(
                root / "run", run_identity, epoch=1
            )
            divergent = self._write_checkpoint_pair(
                root / "divergent", run_identity, epoch=1, perturb=1.5
            )
            self.assertNotEqual(
                train_panderm.sha256_file(final_path),
                train_panderm.sha256_file(divergent[0]),
            )
            # The two candidates a same-step comparison reaches first agree, so
            # only a comparison across every candidate can see the third.
            self._install_backup(final_path, (final_path, final_sidecar), "a" * 32)
            self._install_backup(final_path, (final_path, final_sidecar), "b" * 32)
            self._install_backup(final_path, divergent, "c" * 32)
            self._corrupt_final(final_path)

            model, optimizer, schedule, scaler = self._components()
            ambiguity_accepted = True
            with self._MutationRecorder(
                model, optimizer, schedule, scaler
            ) as recorder:
                with self.assertRaisesRegex(
                    ValueError, "ambiguous preserved checkpoints"
                ):
                    train_panderm.load_checkpoint_for_resume(
                        final_path,
                        map_location="cpu",
                        model=model,
                        expected_identity=run_identity,
                        write_guard=AllowDurableWriteGuard(),
                    )
                ambiguity_accepted = False
            self.assertFalse(
                ambiguity_accepted,
                "three_way_same_step_ambiguity_accepted must be False",
            )
            self.assertEqual(recorder.mutations, [])

    def test_same_step_backups_that_all_agree_still_recover_deterministically(self):
        """Full agreement must still resume; the gate rejects only real drift."""
        run_identity = identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final_path, final_sidecar = self._write_checkpoint_pair(
                root / "run", run_identity, epoch=1
            )
            expected_bytes = final_path.read_bytes()
            for previous_id in ("a" * 32, "b" * 32, "c" * 32):
                self._install_backup(
                    final_path, (final_path, final_sidecar), previous_id
                )
            self._corrupt_final(final_path)
            model, _, _, _ = self._components()
            recovered = train_panderm.load_checkpoint_for_resume(
                final_path,
                map_location="cpu",
                model=model,
                expected_identity=run_identity,
                write_guard=AllowDurableWriteGuard(),
            )
            self.assertEqual(recovered["epoch"], 1)
            self.assertEqual(final_path.read_bytes(), expected_bytes)

    # --- blocker 4 ---------------------------------------------------------
    def test_invalid_highest_recovery_candidate_falls_back_to_valid_lower(self):
        """probe invalid_highest_falls_back_to_valid_lower must be True.

        The old inline candidate check never verified schema_version, so a
        sidecar the production validator rejects could still win the ranking and
        then blow up after the final path had already been overwritten.
        """
        run_identity = identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final_path, final_sidecar = self._write_checkpoint_pair(
                root / "run", run_identity, epoch=1
            )
            valid_lower_bytes = final_path.read_bytes()
            newer = self._write_checkpoint_pair(
                root / "newer", run_identity, epoch=2
            )
            self._install_backup(
                final_path, (final_path, final_sidecar), "a" * 32
            )
            self._install_backup(
                final_path,
                newer,
                "b" * 32,
                sidecar_edit=lambda record: {**record, "schema_version": 2},
            )
            self._corrupt_final(final_path)

            model, _, _, _ = self._components()
            recovered = train_panderm.load_checkpoint_for_resume(
                final_path,
                map_location="cpu",
                model=model,
                expected_identity=run_identity,
                write_guard=AllowDurableWriteGuard(),
            )
            self.assertEqual(
                recovered["epoch"],
                1,
                "invalid_highest_falls_back_to_valid_lower must be True",
            )
            self.assertEqual(final_path.read_bytes(), valid_lower_bytes)

    def test_recovery_candidates_are_validated_by_the_production_validator(self):
        """Every authoritative sidecar field must disqualify a candidate."""
        run_identity = identity()
        edits = {
            "schema_version": lambda record: {**record, "schema_version": 2},
            "checkpoint_format": lambda record: {
                **record,
                "checkpoint_format": "other_format_v1",
            },
            "checkpoint_filename": lambda record: {
                **record,
                "checkpoint_filename": "best.pt",
            },
            "sha256": lambda record: {**record, "sha256": "0" * 64},
            "byte_size": lambda record: {**record, "byte_size": record["byte_size"] + 1},
            "run_identity_sha256": lambda record: {
                **record,
                "run_identity_sha256": "1" * 64,
            },
            "epoch": lambda record: {**record, "epoch": record["epoch"] + 1},
            "global_step": lambda record: {
                **record,
                "global_step": record["global_step"] + 1,
            },
            "extra_key": lambda record: {**record, "unexpected": True},
            "missing_key": lambda record: {
                key: value for key, value in record.items() if key != "epoch"
            },
        }
        for field, edit in edits.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                final_path, final_sidecar = self._write_checkpoint_pair(
                    root / "run", run_identity, epoch=1
                )
                self._install_backup(
                    final_path, (final_path, final_sidecar), "a" * 32,
                    sidecar_edit=edit,
                )
                self._corrupt_final(final_path)
                interrupted = final_path.read_bytes()
                model, optimizer, schedule, scaler = self._components()
                with self._MutationRecorder(
                    model, optimizer, schedule, scaler
                ) as recorder:
                    with self.assertRaises((ValueError, RuntimeError)):
                        train_panderm.load_checkpoint_for_resume(
                            final_path,
                            map_location="cpu",
                            model=model,
                            expected_identity=run_identity,
                            write_guard=AllowDurableWriteGuard(),
                        )
                self.assertEqual(recorder.mutations, [])
                # A candidate the authoritative validator rejects must never be
                # promoted onto the final path before the failure is raised.
                self.assertEqual(final_path.read_bytes(), interrupted)

    # --- blocker 5 ---------------------------------------------------------
    def test_graceful_completion_is_idempotent_after_a_crash(self):
        """probe graceful_completion_retry_after_crash must be True.

        A fresh ``completed_utc`` on every retry made the already published
        audit payload drift, so a crashed handoff could never be finished.
        """
        run_identity_sha256_integrity = {"epoch": 5, "global_step": 40, "sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            marker, history, run_hash = self._session_paths(temporary)
            session_a = str(uuid.uuid4())
            self._start(marker, history, run_hash, "A", session_a)
            snapshot_path = history / f"{session_a}.active.json"
            completion_path = history / f"{session_a}.completed.json"
            result_identity = {
                "epoch": 5,
                "global_step": 40,
                "run_identity_sha256": run_hash,
            }
            real_replace = panderm_run.os.replace

            def crash_on_marker_transition(source, destination):
                if Path(destination) == snapshot_path:
                    raise OSError("simulated crash before the marker transition")
                return real_replace(source, destination)

            with mock.patch.object(
                panderm_run.os, "replace", side_effect=crash_on_marker_transition
            ):
                with self.assertRaisesRegex(OSError, "simulated crash"):
                    panderm_run.complete_sequential_session(
                        marker,
                        session_id=session_a,
                        history_directory=history,
                        checkpoint_integrity=run_identity_sha256_integrity,
                        result_identity=result_identity,
                    )
            # Audit written, marker not yet transitioned.
            self.assertTrue(completion_path.is_file())
            self.assertTrue(marker.is_file())
            self.assertFalse(snapshot_path.exists())
            published_utc = json.loads(
                completion_path.read_text(encoding="utf-8")
            )["completed_utc"]

            retried = panderm_run.complete_sequential_session(
                marker,
                session_id=session_a,
                history_directory=history,
                checkpoint_integrity=run_identity_sha256_integrity,
                result_identity=result_identity,
            )
            self.assertEqual(retried["completed_utc"], published_utc)
            self.assertFalse(marker.exists())
            self.assertTrue(snapshot_path.is_file())

            # Marker already transitioned and the response was lost.
            again = panderm_run.complete_sequential_session(
                marker,
                session_id=session_a,
                history_directory=history,
                checkpoint_integrity=run_identity_sha256_integrity,
                result_identity=result_identity,
            )
            self.assertEqual(again["completed_utc"], published_utc)
            self.assertEqual(
                again["event_id"],
                panderm_run.audit_event_id(
                    panderm_run.GRACEFUL_HANDOFF_EVENT,
                    run_version=panderm_run.RUN_VERSION,
                    subject_session_id=session_a,
                ),
            )
            self.assertEqual(
                sorted(path.name for path in history.iterdir()),
                sorted([snapshot_path.name, completion_path.name]),
            )

    def test_graceful_completion_retry_rejects_drifted_stable_fields(self):
        """A retry with different durable content must never overwrite the audit."""
        with tempfile.TemporaryDirectory() as temporary:
            marker, history, run_hash = self._session_paths(temporary)
            session_a = str(uuid.uuid4())
            self._start(marker, history, run_hash, "A", session_a)
            integrity = {"epoch": 5, "global_step": 40, "sha256": "a" * 64}
            result_identity = {
                "epoch": 5,
                "global_step": 40,
                "run_identity_sha256": run_hash,
            }
            panderm_run.complete_sequential_session(
                marker,
                session_id=session_a,
                history_directory=history,
                checkpoint_integrity=integrity,
                result_identity=result_identity,
            )
            completion_path = history / f"{session_a}.completed.json"
            before = completion_path.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "audit record differs"):
                panderm_run.complete_sequential_session(
                    marker,
                    session_id=session_a,
                    history_directory=history,
                    checkpoint_integrity={**integrity, "sha256": "b" * 64},
                    result_identity=result_identity,
                )
            self.assertEqual(completion_path.read_bytes(), before)

    def test_manual_takeover_is_idempotent_after_a_crash(self):
        """probe manual_takeover_retry_after_crash must be True.

        The audit path used to embed the replacement session id, so every retry
        minted a new id and a new audit file instead of finishing one event.
        """
        with tempfile.TemporaryDirectory() as temporary:
            marker, history, run_hash = self._session_paths(temporary)
            session_a = str(uuid.uuid4())
            session_b = str(uuid.uuid4())
            session_c = str(uuid.uuid4())
            active_a = self._start(marker, history, run_hash, "A", session_a)
            audit_path = history / f"{session_a}.takeover.json"

            with mock.patch.object(
                panderm_run,
                "_replace_json_atomic",
                side_effect=OSError("simulated crash before the marker replacement"),
            ):
                with self.assertRaisesRegex(OSError, "simulated crash"):
                    self._start(
                        marker, history, run_hash, "B", session_b, takeover=True
                    )
            # Audit written, replacement marker not yet created.
            self.assertTrue(audit_path.is_file())
            published = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(published["replacement_session_id"], session_b)
            self.assertEqual(
                json.loads(marker.read_text(encoding="utf-8"))["session_id"],
                session_a,
            )

            # A retry from a new runtime must reuse the published replacement id.
            retried = self._start(
                marker, history, run_hash, "C", session_c, takeover=True
            )
            self.assertEqual(
                retried["session_id"],
                session_b,
                "manual_takeover_retry_after_crash must reuse the published id",
            )
            self.assertEqual(
                json.loads(audit_path.read_text(encoding="utf-8")), published
            )
            self.assertEqual(
                published["previous_active_session"]["session_id"],
                active_a["session_id"],
            )

            # Replacement marker created and the response was lost.
            again = self._start(
                marker, history, run_hash, "C", session_b, takeover=True
            )
            self.assertEqual(again, retried)
            self.assertEqual(
                json.loads(audit_path.read_text(encoding="utf-8")), published
            )
            self.assertEqual(
                [path.name for path in history.iterdir()], [audit_path.name]
            )

    def _completed_a_to_b_takeover(self, marker, history, run_hash):
        """Return (session_a, session_b) after one finished A->B takeover."""
        session_a = str(uuid.uuid4())
        session_b = str(uuid.uuid4())
        self._start(marker, history, run_hash, "A", session_a)
        active_b = self._start(
            marker, history, run_hash, "B", session_b, takeover=True
        )
        self.assertEqual(active_b["session_id"], session_b)
        self.assertEqual(
            [path.name for path in history.iterdir()],
            [f"{session_a}.takeover.json"],
        )
        return session_a, session_b

    def test_cross_runtime_lost_response_retry_adopts_the_published_marker(self):
        """probe cross_runtime_lost_response_retry must be True.

        B's runtime restart mints a brand new candidate UUID, so UUID equality
        alone read the finished A->B transition as a fresh B takeover subject
        and published a second transition for a response that was merely lost.
        """
        with tempfile.TemporaryDirectory() as temporary:
            marker, history, run_hash = self._session_paths(temporary)
            session_a, session_b = self._completed_a_to_b_takeover(
                marker, history, run_hash
            )
            audit_path = history / f"{session_a}.takeover.json"
            marker_before = marker.read_bytes()
            audit_before = audit_path.read_bytes()

            # Same operator B, new runtime, therefore a new candidate id.
            restarted = self._start(
                marker, history, run_hash, "B", str(uuid.uuid4()), takeover=True
            )
            self.assertEqual(
                restarted["session_id"],
                session_b,
                "cross_runtime_lost_response_retry must return the B marker",
            )
            self.assertEqual(
                [path.name for path in history.iterdir()],
                [audit_path.name],
                "second_transition_created_by_retry must be False",
            )
            self.assertEqual(marker.read_bytes(), marker_before)
            self.assertEqual(audit_path.read_bytes(), audit_before)

            # A second restarted candidate must stay equally idempotent.
            again = self._start(
                marker, history, run_hash, "B", str(uuid.uuid4()), takeover=True
            )
            self.assertEqual(again, restarted)
            self.assertEqual(marker.read_bytes(), marker_before)
            self.assertEqual(audit_path.read_bytes(), audit_before)
            self.assertEqual(
                [path.name for path in history.iterdir()], [audit_path.name]
            )

    def test_real_b_to_c_takeover_still_publishes_the_next_transition(self):
        """Adoption must not disarm the next genuine confirmed takeover."""
        with tempfile.TemporaryDirectory() as temporary:
            marker, history, run_hash = self._session_paths(temporary)
            session_a, session_b = self._completed_a_to_b_takeover(
                marker, history, run_hash
            )
            session_c = str(uuid.uuid4())
            active_c = self._start(
                marker, history, run_hash, "C", session_c, takeover=True
            )
            self.assertEqual(active_c["session_id"], session_c)
            self.assertEqual(active_c["account_label"], "C")
            self.assertEqual(
                sorted(path.name for path in history.iterdir()),
                sorted(
                    [f"{session_a}.takeover.json", f"{session_b}.takeover.json"]
                ),
            )
            published = json.loads(
                (history / f"{session_b}.takeover.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                published["previous_active_session"]["session_id"], session_b
            )
            self.assertEqual(published["replacement_session_id"], session_c)
            self.assertEqual(
                published["event_id"],
                panderm_run.audit_event_id(
                    panderm_run.MANUAL_TAKEOVER_EVENT,
                    run_version=panderm_run.RUN_VERSION,
                    subject_session_id=session_b,
                ),
            )

    def _completed_cross_commit_takeover(self, marker, history, run_hash):
        """Return (session_a, session_b) after a confirmed A@old -> B@new takeover.

        This is the shape a real cross-commit validation takeover leaves behind:
        the retired snapshot names the commit and identity that session really
        ran, and the replacement marker names the caller's current ones.
        """
        previous_hash = panderm_run.canonical_identity_sha256(
            identity(git_commit=self.PREVIOUS_COMMIT)
        )
        self.assertNotEqual(previous_hash, run_hash)
        session_a = str(uuid.uuid4())
        session_b = str(uuid.uuid4())
        self._start(
            marker,
            history,
            previous_hash,
            "A",
            session_a,
            git_commit=self.PREVIOUS_COMMIT,
        )
        active_b = self._start(
            marker, history, run_hash, "B", session_b, takeover=True
        )
        self.assertEqual(active_b["session_id"], session_b)
        self.assertEqual(active_b["git_commit"], self.COMMIT)
        self.assertEqual(active_b["run_identity_sha256"], run_hash)
        audit = json.loads(
            (history / f"{session_a}.takeover.json").read_text(encoding="utf-8")
        )
        retired = audit["previous_active_session"]
        self.assertEqual(retired["git_commit"], self.PREVIOUS_COMMIT)
        self.assertEqual(retired["run_identity_sha256"], previous_hash)
        self.assertEqual(audit["replacement_session_id"], session_b)
        self.assertEqual(
            [path.name for path in history.iterdir()],
            [f"{session_a}.takeover.json"],
        )
        return session_a, session_b

    def test_cross_commit_takeover_requires_one_explicit_followup_transition(self):
        """A cross-identity audit is not proof of who published the marker.

        A confirmed manual takeover is allowed to retire a session pinned to an
        older commit, so the audit records the commit and identity that session
        really ran. Those historical fields are then not independent authority:
        A@old -> B@new and the next published code revision arriving at B's
        marker leave the same evidence behind. Adopting on that evidence would
        hand the marker to a caller that never published it, so the already
        confirmed takeover records exactly one explicit follow-up transition and
        only then becomes idempotent.
        """
        with tempfile.TemporaryDirectory() as temporary:
            marker, history, run_hash = self._session_paths(temporary)
            session_a, session_b = self._completed_cross_commit_takeover(
                marker, history, run_hash
            )
            audit_path = history / f"{session_a}.takeover.json"
            audit_before = audit_path.read_bytes()

            # Fresh runtime, same operator B, therefore a brand new candidate id.
            session_c = str(uuid.uuid4())
            active_c = self._start(
                marker, history, run_hash, "B", session_c, takeover=True
            )
            self.assertEqual(
                active_c["session_id"],
                session_c,
                "a cross-identity audit must not be read as B's own lost response",
            )
            self.assertEqual(active_c["account_label"], "B")
            self.assertEqual(active_c["git_commit"], self.COMMIT)
            self.assertEqual(
                audit_path.read_bytes(),
                audit_before,
                "the A->B audit is history and must never be rewritten",
            )
            followup_path = history / f"{session_b}.takeover.json"
            self.assertEqual(
                sorted(path.name for path in history.iterdir()),
                sorted([audit_path.name, followup_path.name]),
                "exactly one follow-up transition may be recorded",
            )
            followup = json.loads(followup_path.read_text(encoding="utf-8"))
            self.assertEqual(
                followup["previous_active_session"]["session_id"], session_b
            )
            self.assertEqual(followup["replacement_session_id"], session_c)

            # With both the retired snapshot and the marker now on the caller's
            # own identity, the next fresh candidate is a plain lost response.
            marker_after = marker.read_bytes()
            followup_before = followup_path.read_bytes()
            adopted = self._start(
                marker, history, run_hash, "B", str(uuid.uuid4()), takeover=True
            )
            self.assertEqual(adopted, active_c)
            self.assertEqual(marker.read_bytes(), marker_after)
            self.assertEqual(audit_path.read_bytes(), audit_before)
            self.assertEqual(followup_path.read_bytes(), followup_before)
            self.assertEqual(
                sorted(path.name for path in history.iterdir()),
                sorted([audit_path.name, followup_path.name]),
                "third_transition_created_by_retry must be False",
            )

    def test_cross_commit_adoption_still_arms_the_next_real_takeover(self):
        """Accepting a cross-commit audit must not disarm the next operator."""
        with tempfile.TemporaryDirectory() as temporary:
            marker, history, run_hash = self._session_paths(temporary)
            session_a, session_b = self._completed_cross_commit_takeover(
                marker, history, run_hash
            )
            session_c = str(uuid.uuid4())
            active_c = self._start(
                marker, history, run_hash, "C", session_c, takeover=True
            )
            self.assertEqual(active_c["session_id"], session_c)
            self.assertEqual(active_c["account_label"], "C")
            self.assertEqual(
                sorted(path.name for path in history.iterdir()),
                sorted([f"{session_a}.takeover.json", f"{session_b}.takeover.json"]),
            )
            published = json.loads(
                (history / f"{session_b}.takeover.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                published["previous_active_session"]["session_id"], session_b
            )
            self.assertEqual(
                published["previous_active_session"]["git_commit"], self.COMMIT
            )
            self.assertEqual(published["replacement_session_id"], session_c)
            self.assertEqual(
                published["event_id"],
                panderm_run.audit_event_id(
                    panderm_run.MANUAL_TAKEOVER_EVENT,
                    run_version=panderm_run.RUN_VERSION,
                    subject_session_id=session_b,
                ),
            )

    def test_replacement_id_disagreeing_with_the_marker_blocks_every_mutation(self):
        """A dangling replacement id must never be resolved by guessing."""
        with tempfile.TemporaryDirectory() as temporary:
            marker, history, run_hash = self._session_paths(temporary)
            session_a, _ = self._completed_a_to_b_takeover(
                marker, history, run_hash
            )
            audit_path = history / f"{session_a}.takeover.json"
            tampered = json.loads(audit_path.read_text(encoding="utf-8"))
            tampered["replacement_session_id"] = str(uuid.uuid4())
            audit_path.write_text(
                json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8"
            )
            marker_before = marker.read_bytes()
            audit_before = audit_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "contradicts"):
                self._start(
                    marker, history, run_hash, "B", str(uuid.uuid4()), takeover=True
                )
            self.assertEqual(marker.read_bytes(), marker_before)
            self.assertEqual(audit_path.read_bytes(), audit_before)
            self.assertEqual(
                [path.name for path in history.iterdir()], [audit_path.name]
            )

    def test_audit_identity_drift_blocks_adoption_without_any_mutation(self):
        """Adoption requires the retired session to belong to this same run."""
        for field, value in (
            ("shared_root_uuid", "8b0d4b25-1b6b-4a52-9f0f-6b9a3a5e2c11"),
            ("run_version", "other-version"),
            # git_commit is hashed into run_identity_sha256, so a retired
            # snapshot cannot name one of them differently from the caller and
            # the other identically. A cross-commit takeover moves both; each of
            # these moves one, which no real session could have written.
            ("run_identity_sha256", "e" * 64),
            ("git_commit", "d" * 40),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                marker, history, run_hash = self._session_paths(temporary)
                session_a, _ = self._completed_a_to_b_takeover(
                    marker, history, run_hash
                )
                audit_path = history / f"{session_a}.takeover.json"
                tampered = json.loads(audit_path.read_text(encoding="utf-8"))
                tampered["previous_active_session"][field] = value
                audit_path.write_text(
                    json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8"
                )
                marker_before = marker.read_bytes()
                audit_before = audit_path.read_bytes()
                with self.assertRaisesRegex(ValueError, "identity drift"):
                    self._start(
                        marker,
                        history,
                        run_hash,
                        "B",
                        str(uuid.uuid4()),
                        takeover=True,
                    )
                self.assertEqual(marker.read_bytes(), marker_before)
                self.assertEqual(audit_path.read_bytes(), audit_before)
                self.assertEqual(
                    [path.name for path in history.iterdir()], [audit_path.name]
                )

    def test_audit_record_schema_drift_blocks_adoption_without_any_mutation(self):
        """Adoption evidence must match the published audit schema exactly.

        Adoption resolves a lost response by trusting one stored record to
        prove the active marker is this operator's own replacement. A record
        carrying an unknown key or a schema_version this build cannot interpret
        is unreadable evidence, so it must stop the takeover rather than be
        read as a weaker yes.
        """
        for field, value, expected in (
            ("schema_version", 2, "schema_version mismatch"),
            ("unexpected", True, "schema mismatch"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                marker, history, run_hash = self._session_paths(temporary)
                session_a, _ = self._completed_a_to_b_takeover(
                    marker, history, run_hash
                )
                audit_path = history / f"{session_a}.takeover.json"
                tampered = json.loads(audit_path.read_text(encoding="utf-8"))
                tampered[field] = value
                audit_path.write_text(
                    json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8"
                )
                marker_before = marker.read_bytes()
                audit_before = audit_path.read_bytes()
                with self.assertRaisesRegex(ValueError, expected):
                    self._start(
                        marker,
                        history,
                        run_hash,
                        "B",
                        str(uuid.uuid4()),
                        takeover=True,
                    )
                self.assertEqual(marker.read_bytes(), marker_before)
                self.assertEqual(audit_path.read_bytes(), audit_before)
                self.assertEqual(
                    [path.name for path in history.iterdir()], [audit_path.name]
                )

    def test_audit_retired_session_corruption_blocks_adoption_without_any_mutation(self):
        """The retired snapshot inside an audit is held to the marker schema.

        The graceful path revalidates its retired marker through
        _read_active_session, so a takeover audit that stores an unusable
        retired session must fail the same way. Otherwise a corrupted snapshot
        still authorises adopting the active marker, which is the operator
        deciding a real handoff on evidence nothing checked.
        """
        deleted = object()
        for field, value, expected in (
            ("account_label", "Z", "account label must be A, B, or C"),
            ("hostname", "", "hostname must be a non-empty string"),
            ("hostname", deleted, "schema mismatch"),
            ("checkpoint_cadence", "monthly", "cadence must be every_epoch"),
            (
                "maximum_quota_loss",
                "unbounded",
                "maximum quota loss must be one_incomplete_epoch",
            ),
            ("evaluation_scope", "full", "prohibited"),
        ):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as temporary:
                marker, history, run_hash = self._session_paths(temporary)
                session_a, _ = self._completed_a_to_b_takeover(
                    marker, history, run_hash
                )
                audit_path = history / f"{session_a}.takeover.json"
                tampered = json.loads(audit_path.read_text(encoding="utf-8"))
                if value is deleted:
                    del tampered["previous_active_session"][field]
                else:
                    tampered["previous_active_session"][field] = value
                audit_path.write_text(
                    json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8"
                )
                marker_before = marker.read_bytes()
                audit_before = audit_path.read_bytes()
                with self.assertRaisesRegex(ValueError, expected):
                    self._start(
                        marker,
                        history,
                        run_hash,
                        "B",
                        str(uuid.uuid4()),
                        takeover=True,
                    )
                self.assertEqual(marker.read_bytes(), marker_before)
                self.assertEqual(audit_path.read_bytes(), audit_before)
                self.assertEqual(
                    [path.name for path in history.iterdir()], [audit_path.name]
                )

    def test_account_change_cannot_bypass_active_marker_validation(self):
        """A different account label must not skip validating the marker.

        The different-account answer is a decision about the marker in front of
        the caller: it retires that marker and publishes the next transition
        from it. Answering it before the marker has been held to this run means
        a marker naming another run version, another shared root, or a
        self-contradicting commit/identity pair still authorises a real
        transition, and the operator only has to arrive under a different
        account for the validation to be skipped entirely.
        """
        for field, value in (
            ("run_version", "other-version"),
            ("shared_root_uuid", "8b0d4b25-1b6b-4a52-9f0f-6b9a3a5e2c11"),
            # git_commit is hashed into run_identity_sha256, so a real marker
            # cannot name one of them differently from the caller and the other
            # identically. Each of these moves exactly one.
            ("git_commit", "d" * 40),
            ("run_identity_sha256", "e" * 64),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                marker, history, run_hash = self._session_paths(temporary)
                session_a, _ = self._completed_a_to_b_takeover(
                    marker, history, run_hash
                )
                audit_path = history / f"{session_a}.takeover.json"
                tampered = json.loads(marker.read_text(encoding="utf-8"))
                tampered[field] = value
                marker.write_text(
                    json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8"
                )
                marker_before = marker.read_bytes()
                audit_before = audit_path.read_bytes()
                with self.assertRaisesRegex(
                    ValueError, "active session identity drift"
                ):
                    self._start(
                        marker,
                        history,
                        run_hash,
                        "C",
                        str(uuid.uuid4()),
                        takeover=True,
                    )
                self.assertEqual(marker.read_bytes(), marker_before)
                self.assertEqual(audit_path.read_bytes(), audit_before)
                self.assertEqual(
                    [path.name for path in history.iterdir()],
                    [audit_path.name],
                    "no transition may be published from an unvalidated marker",
                )

    def test_initial_marker_validation_is_not_skipped_by_empty_history(self):
        """An empty audit history must not skip validating the marker.

        The very first marker has no takeover audit naming it, so the adoption
        search finds no replacement and answers immediately. Deciding that
        before the marker has been held to this run means the most common real
        state on Drive -- one active session, no history yet -- is exactly the
        state in which a marker naming another run version, another shared
        root, or a self-contradicting commit/identity pair still authorises a
        real transition.
        """
        for field, value in (
            ("run_version", "other-version"),
            ("shared_root_uuid", "8b0d4b25-1b6b-4a52-9f0f-6b9a3a5e2c11"),
            # git_commit is hashed into run_identity_sha256, so each of these
            # moves exactly one half of a pair that cannot really disagree.
            ("git_commit", "d" * 40),
            ("run_identity_sha256", "e" * 64),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                marker, history, run_hash = self._session_paths(temporary)
                self._start(marker, history, run_hash, "A", str(uuid.uuid4()))
                self.assertEqual(list(history.iterdir()), [])
                tampered = json.loads(marker.read_text(encoding="utf-8"))
                tampered[field] = value
                marker.write_text(
                    json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8"
                )
                marker_before = marker.read_bytes()
                with self.assertRaisesRegex(
                    ValueError, "active session identity drift"
                ):
                    self._start(
                        marker,
                        history,
                        run_hash,
                        "B",
                        str(uuid.uuid4()),
                        takeover=True,
                    )
                self.assertEqual(marker.read_bytes(), marker_before)
                self.assertEqual(
                    list(history.iterdir()),
                    [],
                    "no transition may be published from an unvalidated marker",
                )

    # --- durable marker publish under Google Drive FUSE ---------------------
    MARKER_VALUE = {
        "schema_version": 1,
        "session_id": "ec2112e9-382c-47c7-b487-b3fe01d00d37",
    }

    @contextlib.contextmanager
    def _drive_readback(
        self, target, responses, *, timeout=5.0, poll=0.0, heartbeat=0.0
    ):
        """Answer reads of ``target`` from ``responses`` and count attempts.

        A response is either an exception to raise or the exact text to return;
        the last entry repeats. Every other path reads normally, so the
        temporary file written before ``os.replace`` is untouched.
        """
        attempts = []
        real_read_text = Path.read_text

        def read_text(path_self, *args, **kwargs):
            if path_self != target:
                return real_read_text(path_self, *args, **kwargs)
            attempts.append(path_self)
            answer = responses[min(len(attempts) - 1, len(responses) - 1)]
            if isinstance(answer, BaseException):
                raise answer
            return answer

        with mock.patch.object(Path, "read_text", read_text), mock.patch.object(
            panderm_run, "DRIVE_VISIBILITY_TIMEOUT_SECONDS", timeout
        ), mock.patch.object(
            panderm_run, "DRIVE_VISIBILITY_POLL_SECONDS", poll
        ), mock.patch.object(
            panderm_run, "DRIVE_VISIBILITY_HEARTBEAT_SECONDS", heartbeat
        ), mock.patch("builtins.print") as printed:
            yield attempts, printed

    @staticmethod
    def _heartbeats(printed):
        return [
            call
            for call in printed.call_args_list
            if call.args and str(call.args[0]).startswith("[drive-visibility]")
        ]

    def test_delayed_marker_visibility_after_replace_converges_on_exact_json(self):
        """probe delayed_marker_visibility must be tolerated.

        Google Drive FUSE answered ENOENT for active_session.json immediately
        after os.replace had published it, so a takeover that had already
        written its audit and its replacement marker failed on a readback of
        bytes the cloud later proved were correct. A single attempt is not
        evidence the publish failed.
        """
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / panderm_run.ACTIVE_SESSION_FILENAME
            target.write_text('{"retired": true}\n', encoding="ascii")
            hidden = FileNotFoundError("simulated Drive FUSE delay")
            published = json.dumps(
                self.MARKER_VALUE, allow_nan=False, ensure_ascii=True, sort_keys=True
            ) + "\n"
            with self._drive_readback(
                target, [hidden, hidden, hidden, published]
            ) as (attempts, printed):
                panderm_run._replace_json_atomic(target, self.MARKER_VALUE)
            self.assertEqual(len(attempts), 4)
            self.assertEqual(
                json.loads(target.read_text(encoding="ascii")), self.MARKER_VALUE
            )
            self.assertEqual(
                [path.name for path in Path(temporary).iterdir()], [target.name]
            )
            heartbeats = self._heartbeats(printed)
            self.assertEqual(len(heartbeats), 3)
            for call in heartbeats:
                self.assertIs(call.kwargs.get("flush"), True)
                self.assertIn(target.name, call.args[0])
            self.assertIn("attempts=3", heartbeats[-1].args[0])

    def test_marker_visibility_wait_stays_silent_until_the_heartbeat_interval(self):
        """A truthful heartbeat reports real silence, so it must be timed."""
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / panderm_run.ACTIVE_SESSION_FILENAME
            target.write_text('{"retired": true}\n', encoding="ascii")
            published = json.dumps(
                self.MARKER_VALUE, allow_nan=False, ensure_ascii=True, sort_keys=True
            ) + "\n"
            with self._drive_readback(
                target,
                [FileNotFoundError("simulated Drive FUSE delay"), published],
                heartbeat=3600.0,
            ) as (attempts, printed):
                panderm_run._replace_json_atomic(target, self.MARKER_VALUE)
            self.assertEqual(len(attempts), 2)
            self.assertEqual(self._heartbeats(printed), [])

            # An immediately visible marker prints nothing at all.
            second = Path(temporary) / "second.json"
            with self._drive_readback(second, [published]) as (again, quiet):
                panderm_run._replace_json_atomic(second, self.MARKER_VALUE)
            self.assertEqual(len(again), 1)
            self.assertEqual(self._heartbeats(quiet), [])

    def test_permanently_invisible_marker_fails_bounded_with_no_false_success(self):
        """The wait is bounded: a marker that never appears must fail loud."""
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / panderm_run.ACTIVE_SESSION_FILENAME
            target.write_text('{"retired": true}\n', encoding="ascii")
            with self._drive_readback(
                target,
                [FileNotFoundError("simulated Drive FUSE delay")],
                timeout=0.05,
                poll=0.02,
                heartbeat=3600.0,
            ) as (attempts, printed):
                with self.assertRaises(TimeoutError) as raised:
                    panderm_run._replace_json_atomic(target, self.MARKER_VALUE)
            self.assertIn("never became visible", str(raised.exception))
            self.assertIn("FileNotFoundError", str(raised.exception))
            self.assertGreaterEqual(len(attempts), 1)
            # The temporary file was consumed by os.replace, so no residue is
            # left behind by the failure.
            self.assertEqual(
                [path.name for path in Path(temporary).iterdir()], [target.name]
            )

    def test_wrong_bytes_after_delayed_visibility_are_never_accepted(self):
        """Late is retried; wrong is not. Mismatched bytes must fail at once."""
        wrong = json.dumps({"schema_version": 1, "session_id": "wrong"}) + "\n"
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / panderm_run.ACTIVE_SESSION_FILENAME
            target.write_text('{"retired": true}\n', encoding="ascii")
            with self._drive_readback(
                target, [FileNotFoundError("simulated Drive FUSE delay"), wrong]
            ) as (attempts, printed):
                with self.assertRaisesRegex(
                    ValueError, "published JSON reopen mismatch"
                ):
                    panderm_run._replace_json_atomic(target, self.MARKER_VALUE)
            self.assertEqual(
                len(attempts), 2, "wrong bytes must not be retried into a timeout"
            )
            self.assertEqual(
                [path.name for path in Path(temporary).iterdir()], [target.name]
            )

    def test_refused_read_of_the_published_marker_fails_immediately(self):
        """A refused read is a permission answer, not delayed visibility.

        Polling it would spend the whole bounded wait and then report a
        timeout, which names the wrong fault and hides the real one from the
        operator who has to fix it.
        """
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / panderm_run.ACTIVE_SESSION_FILENAME
            target.write_text('{"retired": true}\n', encoding="ascii")
            with self._drive_readback(
                target, [PermissionError("simulated shared-root permission loss")]
            ) as (attempts, printed):
                with self.assertRaises(PermissionError):
                    panderm_run._replace_json_atomic(target, self.MARKER_VALUE)
            self.assertEqual(
                len(attempts), 1, "a refused read must not be retried"
            )
            self.assertEqual(self._heartbeats(printed), [])
            self.assertEqual(
                [path.name for path in Path(temporary).iterdir()], [target.name]
            )

    def test_malformed_visible_marker_bytes_fail_immediately(self):
        """Bytes that are visible are an answer, even when they do not parse."""
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / panderm_run.ACTIVE_SESSION_FILENAME
            target.write_text('{"retired": true}\n', encoding="ascii")
            with self._drive_readback(target, ['{"session_id": ']) as (
                attempts,
                printed,
            ):
                with self.assertRaises(json.JSONDecodeError):
                    panderm_run._replace_json_atomic(target, self.MARKER_VALUE)
            self.assertEqual(
                len(attempts), 1, "malformed bytes must not be retried"
            )
            self.assertEqual(self._heartbeats(printed), [])
            self.assertEqual(
                [path.name for path in Path(temporary).iterdir()], [target.name]
            )

    def test_first_active_session_publish_waits_for_drive_visibility(self):
        """The initial marker is published onto the same mount as a takeover.

        Only the replacement path was given the bounded readback, so the very
        first session of a run still failed on the one ENOENT Google Drive FUSE
        answers right after os.replace. That refuses to start a run whose marker
        the cloud proves moments later was written correctly, and it does so at
        the point where the operator has no published state to inspect.
        """
        with tempfile.TemporaryDirectory() as temporary:
            marker, history, run_hash = self._session_paths(temporary)
            self.assertFalse(marker.exists())
            session_a = str(uuid.uuid4())
            expected = {
                "schema_version": 1,
                "session_id": session_a,
                "run_version": panderm_run.RUN_VERSION,
                "git_commit": self.COMMIT,
                "shared_root_uuid": self.ROOT_UUID,
                "evaluation_scope": panderm_run.VALIDATION_ONLY,
                "account_label": "A",
                "hostname": "fixed-host",
                "started_utc": "2026-08-03T00:00:00Z",
                "run_identity_sha256": run_hash,
                "checkpoint_cadence": "every_epoch",
                "maximum_quota_loss": "one_incomplete_epoch",
            }
            published = json.dumps(
                expected, allow_nan=False, ensure_ascii=True, sort_keys=True
            ) + "\n"
            hidden = FileNotFoundError("simulated Drive FUSE delay")
            with mock.patch.object(
                panderm_run, "utc_now", return_value=expected["started_utc"]
            ), mock.patch.object(
                panderm_run.socket, "gethostname", return_value=expected["hostname"]
            ), self._drive_readback(marker, [hidden, hidden, published]) as (
                attempts,
                printed,
            ):
                active = self._start(marker, history, run_hash, "A", session_a)
            self.assertEqual(active, expected)
            self.assertEqual(json.loads(marker.read_text(encoding="utf-8")), expected)
            self.assertEqual(
                len(attempts),
                4,
                "two delayed readbacks, the visible one, then the active-session "
                "read that answers the caller",
            )
            self.assertEqual(
                sorted(path.name for path in marker.parent.iterdir()),
                sorted([marker.name, history.name]),
                "the delayed publish must leave no temporary behind",
            )
            self.assertEqual(list(history.iterdir()), [])
            heartbeats = self._heartbeats(printed)
            self.assertEqual(len(heartbeats), 2)
            for call in heartbeats:
                self.assertIs(call.kwargs.get("flush"), True)
                self.assertIn(marker.name, call.args[0])
            self.assertIn("attempts=2", heartbeats[-1].args[0])

    def test_manual_takeover_is_still_refused_without_confirmation(self):
        """probe manual_takeover_forced_false must not become an auto-takeover."""
        with tempfile.TemporaryDirectory() as temporary:
            marker, history, run_hash = self._session_paths(temporary)
            session_a = str(uuid.uuid4())
            session_b = str(uuid.uuid4())
            self._start(marker, history, run_hash, "A", session_a)
            before = marker.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "confirm"):
                self._start(marker, history, run_hash, "B", session_b)
            self.assertEqual(marker.read_bytes(), before)
            self.assertEqual(list(history.iterdir()), [])
            taken_over = self._start(
                marker, history, run_hash, "B", session_b, takeover=True
            )
            self.assertEqual(taken_over["session_id"], session_b)
            self.assertEqual(taken_over["account_label"], "B")

    # --- blocker 1 (runner side) -------------------------------------------
    def test_resume_survives_session_account_and_hostname_change(self):
        """A different session/account/host must never block the same run resume."""
        run_identity = identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final_path, _ = self._write_checkpoint_pair(
                root / "run", run_identity, epoch=1
            )
            model, optimizer, schedule, scaler = self._components()
            checkpoint = train_panderm.load_checkpoint_for_resume(
                final_path,
                map_location="cpu",
                model=model,
                expected_identity=run_identity,
                write_guard=AllowDurableWriteGuard(),
            )
            start_epoch, best, history = train_panderm.restore_checkpoint_state(
                checkpoint,
                model,
                optimizer,
                schedule,
                scaler,
                write_guard=AllowDurableWriteGuard(),
            )
            self.assertEqual(start_epoch, 2)
            self.assertEqual(best, 0.5)
            self.assertEqual(len(history), 1)
            self.assertNotIn("session_id", checkpoint["run_identity"])
            self.assertNotIn("account_label", checkpoint["run_identity"])
            self.assertNotIn("hostname", checkpoint["run_identity"])


class IdentityAdversarialMatrixTests(unittest.TestCase):
    def test_exact_nested_tuples_are_accepted(self):
        value = {
            "outer": (
                "value",
                [1, {"nested": (True, None, 2.5)}],
                {"inner": ("x", [3])},
            )
        }
        panderm_run.require_primitive_identity(value)

    def test_tuple_rejects_nonprimitive_values_with_exact_field_paths(self):
        cases = {
            "tensor": (torch.tensor(1), "run_identity.outer[0]"),
            "numpy": (np.int64(1), "run_identity.outer[0]"),
            "nan": (float("nan"), "run_identity.outer[0]"),
            "positive_inf": (float("inf"), "run_identity.outer[0]"),
            "negative_inf": (float("-inf"), "run_identity.outer[0]"),
            "non_string_key": ({1: "value"}, "run_identity.outer[0]"),
        }
        for name, (item, path) in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, re.escape(path)):
                    panderm_run.require_primitive_identity({"outer": (item,)})

    def test_tuple_subclass_is_rejected(self):
        class TupleSubclass(tuple):
            pass

        with self.assertRaisesRegex(
            ValueError, r"run_identity\.outer.*TupleSubclass"
        ):
            panderm_run.require_primitive_identity(
                {"outer": TupleSubclass(("value",))}
            )

    def test_tuple_rejects_exact_primitive_type_subclasses(self):
        class StringSubclass(str):
            pass

        class IntegerSubclass(int):
            pass

        class FloatSubclass(float):
            pass

        class ListSubclass(list):
            pass

        class DictSubclass(dict):
            pass

        for value in (
            StringSubclass("x"),
            IntegerSubclass(1),
            FloatSubclass(1.0),
            ListSubclass([1]),
            DictSubclass({"key": "value"}),
        ):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(
                    ValueError, r"run_identity\.outer\[0\]"
                ):
                    panderm_run.require_primitive_identity({"outer": (value,)})

    def test_authoritative_schema_is_exactly_the_25_built_fields(self):
        built = identity()
        self.assertEqual(len(built), 25)
        self.assertEqual(tuple(built), panderm_run.IMMUTABLE_IDENTITY_KEYS)

    def test_deletion_and_drift_matrix_rejects_175_of_175(self):
        expected = identity()
        keys = panderm_run.IMMUTABLE_IDENTITY_KEYS
        counts = {
            "expected_deleted": 0,
            "saved_deleted": 0,
            "nested_deleted": 0,
            "top_level_deleted": 0,
            "duplicate_drift": 0,
            "result_best_last_common_wrong": 0,
            "notebook_completed_validator_common_wrong": 0,
        }
        accepted = 0
        for key in keys:
            truncated = copy.deepcopy(expected)
            del truncated[key]
            with self.assertRaises(ValueError):
                panderm_run.require_expected_identity_complete(truncated)
            counts["expected_deleted"] += 1

            with self.assertRaises(ValueError):
                panderm_run.require_matching_identity(truncated, expected)
            counts["saved_deleted"] += 1

            result, _, _ = completed_artifacts(expected)
            nested = copy.deepcopy(expected)
            del nested[key]
            with self.assertRaises(ValueError):
                panderm_run.require_identity_duplicates(
                    expected=expected,
                    record=result,
                    nested=nested,
                    top_level={
                        name: result[name]
                        for name in panderm_run.IMMUTABLE_IDENTITY_KEYS
                    },
                )
            counts["nested_deleted"] += 1

            top_level = copy.deepcopy(expected)
            del top_level[key]
            with self.assertRaises(ValueError):
                panderm_run.require_identity_duplicates(
                    expected=expected,
                    record=result,
                    nested=expected,
                    top_level=top_level,
                )
            counts["top_level_deleted"] += 1

            top_level = copy.deepcopy(expected)
            top_level[key] = {"drifted": True}
            with self.assertRaises(ValueError):
                panderm_run.require_identity_duplicates(
                    expected=expected,
                    record=result,
                    nested=expected,
                    top_level=top_level,
                )
            counts["duplicate_drift"] += 1

            wrong = copy.deepcopy(expected)
            wrong[key] = {"uniformly_wrong": True}
            wrong_result, wrong_best, wrong_last = completed_artifacts(wrong)
            with self.assertRaises(ValueError):
                panderm_run.require_completed_artifact_identities(
                    expected=expected,
                    result=wrong_result,
                    best_checkpoint=wrong_best,
                    last_checkpoint=wrong_last,
                )
            counts["result_best_last_common_wrong"] += 1

            with self.assertRaises(ValueError):
                panderm_run.require_completed_artifact_identities(
                    expected=expected,
                    result=wrong_result,
                    best_checkpoint=wrong_best,
                    last_checkpoint=wrong_last,
                )
            counts["notebook_completed_validator_common_wrong"] += 1

        self.assertEqual(set(counts.values()), {25})
        self.assertEqual(sum(counts.values()), 175)
        self.assertEqual(accepted, 0)

    def test_complete_matching_artifacts_are_accepted(self):
        expected = identity()
        result, best, last = completed_artifacts(expected)
        panderm_run.require_completed_artifact_identities(
            expected=expected,
            result=result,
            best_checkpoint=best,
            last_checkpoint=last,
        )

    def test_incomplete_current_identity_is_rejected(self):
        expected = identity()
        current = copy.deepcopy(expected)
        del current["dependency_versions"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            panderm_run.require_matching_identity(expected, current)


class PersistenceTamperMatrixTests(unittest.TestCase):
    def test_tensor_shape_dtype_and_value_tamper_rejected(self):
        expected = {"tensor": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
        tampered = (
            {"tensor": expected["tensor"].reshape(3, 2)},
            {"tensor": expected["tensor"].to(torch.float64)},
            {"tensor": expected["tensor"].clone()},
        )
        tampered[2]["tensor"][0, 0] += 1
        for payload in tampered:
            with self.assertRaises(ValueError):
                train_panderm.require_recursive_exact(expected, payload)

    def test_optimizer_scheduler_scaler_nested_tamper_rejected(self):
        expected = {
            "optimizer_state_dict": {"state": {0: {"step": 3}}},
            "scheduler_state_dict": {"step_count": 4},
            "scaler_state_dict": {"scale": 65536.0},
        }
        for branch in expected:
            tampered = copy.deepcopy(expected)
            leaf = tampered[branch]
            key = next(iter(leaf))
            leaf[key] = {"drifted": True}
            with self.subTest(branch=branch):
                with self.assertRaises(ValueError):
                    train_panderm.require_recursive_exact(expected, tampered)

    def test_result_json_tamper_matrix_and_existing_output(self):
        real_replace = panderm_run.os.replace
        tampered_payloads = ('{"a":', "{}", '{"a": 2}')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, tampered in enumerate(tampered_payloads):
                path = root / f"result_{index}.json"

                def replace_then_tamper(source, destination, value=tampered):
                    real_replace(source, destination)
                    Path(destination).write_text(value, encoding="utf-8")

                with mock.patch.object(
                    panderm_run.os, "replace", side_effect=replace_then_tamper
                ):
                    with self.assertRaises((ValueError, json.JSONDecodeError)):
                        panderm_run.write_json_atomic(path, {"a": 1})

            existing = root / "existing.json"
            existing.write_text('{"a": 1}', encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                panderm_run.write_json_atomic(existing, {"a": 2})


class CheckpointIntegritySidecarTests(unittest.TestCase):
    class _VisibilitySimulator:
        def __init__(
            self,
            checkpoint,
            *,
            checkpoint_visible_after=0,
            sidecar_visible_after=0,
            checkpoint_mutation=None,
            sidecar_mutation=None,
        ):
            self.checkpoint = Path(checkpoint)
            self.sidecar = train_panderm.checkpoint_integrity_path(checkpoint)
            self.checkpoint_visible_after = checkpoint_visible_after
            self.sidecar_visible_after = sidecar_visible_after
            self.checkpoint_mutation = checkpoint_mutation
            self.sidecar_mutation = sidecar_mutation
            self.pending = {}
            self.now = 0.0
            self.sleep_calls = 0
            self.replace_calls = []
            self.real_replace = train_panderm.os.replace

        def replace(self, source, destination):
            source = Path(source)
            destination = Path(destination)
            self.replace_calls.append((source.name, destination.name))
            if any(
                marker in source.name
                for marker in (
                    ".previous.",
                    ".restore.",
                    ".restore-sidecar.",
                )
            ):
                self.pending.pop(destination, None)
                return self.real_replace(source, destination)
            visible_after = None
            mutation = None
            if destination == self.checkpoint:
                visible_after = self.checkpoint_visible_after
                mutation = self.checkpoint_mutation
            elif destination == self.sidecar:
                visible_after = self.sidecar_visible_after
                mutation = self.sidecar_mutation
            if destination.exists() and visible_after != 0:
                stale = destination.read_bytes()
                self.real_replace(source, destination)
                candidate = destination.read_bytes()
                if mutation is not None:
                    candidate = mutation(candidate)
                self.pending[destination] = candidate
                destination.write_bytes(stale)
                return None
            return self.real_replace(source, destination)

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds
            self.sleep_calls += 1
            for path, visible_after in (
                (self.checkpoint, self.checkpoint_visible_after),
                (self.sidecar, self.sidecar_visible_after),
            ):
                if (
                    visible_after is not None
                    and self.sleep_calls >= visible_after
                    and path in self.pending
                ):
                    path.write_bytes(self.pending.pop(path))

    @staticmethod
    def _same_size_byte_tamper(value):
        tampered = bytearray(value)
        tampered[len(tampered) // 2] ^= 1
        return bytes(tampered)

    @staticmethod
    def _same_size_sidecar_tamper(value):
        marker = b'"sha256": "'
        index = value.index(marker) + len(marker)
        replacement = b"0" if value[index:index + 1] != b"0" else b"1"
        return value[:index] + replacement + value[index + 1:]

    @contextlib.contextmanager
    def _simulate_visibility(self, simulator):
        with (
            mock.patch.object(
                train_panderm.os, "replace", side_effect=simulator.replace
            ),
            mock.patch.object(
                train_panderm,
                "_checkpoint_visibility_monotonic",
                side_effect=simulator.monotonic,
                create=True,
            ),
            mock.patch.object(
                train_panderm,
                "_checkpoint_visibility_sleep",
                side_effect=simulator.sleep,
                create=True,
            ),
            mock.patch.object(
                train_panderm,
                "CHECKPOINT_PUBLICATION_VISIBILITY_TIMEOUT_SECONDS",
                6.0,
                create=True,
            ),
            mock.patch.object(
                train_panderm,
                "CHECKPOINT_PUBLICATION_VISIBILITY_POLL_SECONDS",
                1.0,
                create=True,
            ),
            mock.patch.object(
                train_panderm,
                "CHECKPOINT_PUBLICATION_HASH_RETRY_SECONDS",
                2.0,
                create=True,
            ),
            mock.patch.object(
                train_panderm,
                "CHECKPOINT_PUBLICATION_HEARTBEAT_SECONDS",
                2.0,
                create=True,
            ),
        ):
            yield

    def _components(self):
        model = build_mock_model()
        optimizer = panderm.build_optimizer(model, num_layers=4)
        schedule = panderm.WarmupCosineSchedule(
            optimizer, warmup_epochs=1, epochs=2, steps_per_epoch=2
        )
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        return model, optimizer, schedule, scaler

    def _visibility_components(self):
        model = torch.nn.Linear(2, len(train_panderm.config.CLASS_NAMES))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        schedule = mock.Mock(step_count=0)
        schedule.state_dict.return_value = {"step_count": 0}
        scaler = mock.Mock()
        scaler.state_dict.return_value = {}
        return model, optimizer, schedule, scaler

    def _save(self, path, run_identity, epoch=1, write_guard=None, components=None):
        model, optimizer, schedule, scaler = (
            self._components() if components is None else components
        )
        args = type("A", (), {"seed": 0, "epochs": 5})()
        train_panderm.save_checkpoint(
            path,
            model,
            optimizer,
            schedule,
            scaler,
            epoch,
            0.5,
            [
                {"epoch": completed_epoch, "optimizer_steps": schedule.step_count}
                for completed_epoch in range(1, epoch + 1)
            ],
            args,
            run_identity,
            write_guard=write_guard or AllowDurableWriteGuard(),
        )
        return model, optimizer, schedule, scaler

    def _save_visibility(self, path, run_identity, epoch=1, write_guard=None):
        return self._save(
            path,
            run_identity,
            epoch=epoch,
            write_guard=write_guard,
            components=self._visibility_components(),
        )

    def _sidecar_path(self, checkpoint):
        return checkpoint.with_name(checkpoint.name + ".integrity.json")

    def test_overwrite_waits_for_stale_checkpoint_visibility(self):
        run_identity = identity()
        for filename in ("best.pt", "last.pt"):
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / filename
                    self._save_visibility(path, run_identity, epoch=1)
                    sidecar_path = self._sidecar_path(path)
                    before = (path.read_bytes(), sidecar_path.read_bytes())
                    simulator = self._VisibilitySimulator(
                        path, checkpoint_visible_after=3
                    )
                    with self._simulate_visibility(simulator):
                        self._save_visibility(path, run_identity, epoch=2)
                    self.assertEqual(
                        train_panderm.load_checkpoint_safe(
                            path, expected_identity=run_identity
                        )["epoch"],
                        2,
                    )
                    self.assertNotEqual(
                        (path.read_bytes(), sidecar_path.read_bytes()), before
                    )
                    self.assertFalse(simulator.pending)

    def test_overwrite_waits_for_independently_stale_sidecar_visibility(self):
        run_identity = identity()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best.pt"
            self._save_visibility(path, run_identity, epoch=1)
            simulator = self._VisibilitySimulator(
                path, sidecar_visible_after=3
            )
            with self._simulate_visibility(simulator):
                self._save_visibility(path, run_identity, epoch=2)
            self.assertEqual(
                train_panderm.load_checkpoint_safe(
                    path, expected_identity=run_identity
                )["epoch"],
                2,
            )
            self.assertFalse(simulator.pending)

    def test_visibility_timeout_restores_exact_pair_without_residue(self):
        run_identity = identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "last.pt"
            self._save_visibility(path, run_identity, epoch=1)
            sidecar_path = self._sidecar_path(path)
            before = (path.read_bytes(), sidecar_path.read_bytes())
            simulator = self._VisibilitySimulator(
                path, checkpoint_visible_after=None
            )
            with self._simulate_visibility(simulator):
                with self.assertRaisesRegex(TimeoutError, "visibility timeout"):
                    self._save_visibility(path, run_identity, epoch=2)
            self.assertEqual((path.read_bytes(), sidecar_path.read_bytes()), before)
            self.assertEqual(
                sorted(item.name for item in root.iterdir()),
                ["last.pt", "last.pt.integrity.json"],
            )

    def test_same_size_tampered_candidate_never_bypasses_sha(self):
        run_identity = identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "last.pt"
            self._save_visibility(path, run_identity, epoch=1)
            sidecar_path = self._sidecar_path(path)
            before = (path.read_bytes(), sidecar_path.read_bytes())
            simulator = self._VisibilitySimulator(
                path,
                checkpoint_visible_after=2,
                checkpoint_mutation=self._same_size_byte_tamper,
            )
            with self._simulate_visibility(simulator):
                with self.assertRaisesRegex(TimeoutError, "visibility timeout"):
                    self._save_visibility(path, run_identity, epoch=2)
            self.assertEqual((path.read_bytes(), sidecar_path.read_bytes()), before)
            self.assertEqual(
                sorted(item.name for item in root.iterdir()),
                ["last.pt", "last.pt.integrity.json"],
            )

    def test_wrong_sidecar_never_converges_or_survives_rollback(self):
        run_identity = identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "best.pt"
            self._save_visibility(path, run_identity, epoch=1)
            sidecar_path = self._sidecar_path(path)
            before = (path.read_bytes(), sidecar_path.read_bytes())
            simulator = self._VisibilitySimulator(
                path,
                sidecar_visible_after=2,
                sidecar_mutation=self._same_size_sidecar_tamper,
            )
            with self._simulate_visibility(simulator):
                with self.assertRaisesRegex(TimeoutError, "visibility timeout"):
                    self._save_visibility(path, run_identity, epoch=2)
            self.assertEqual((path.read_bytes(), sidecar_path.read_bytes()), before)
            self.assertEqual(
                sorted(item.name for item in root.iterdir()),
                ["best.pt", "best.pt.integrity.json"],
            )

    def test_visibility_guard_loss_stops_before_rollback_mutation(self):
        run_identity = identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "last.pt"
            self._save_visibility(path, run_identity, epoch=1)
            simulator = self._VisibilitySimulator(
                path, checkpoint_visible_after=None
            )

            class ExpiringGuard:
                def __init__(self):
                    self.lost = False
                    self.replace_count_at_loss = None

                def require(inner_self, phase):
                    if "checkpoint publication visibility" in phase:
                        inner_self.lost = True
                        inner_self.replace_count_at_loss = len(
                            simulator.replace_calls
                        )
                        raise RuntimeError("stale publication fence")
                    if inner_self.lost:
                        raise RuntimeError("stale publication fence")

            guard = ExpiringGuard()
            with self._simulate_visibility(simulator):
                with self.assertRaisesRegex(
                    RuntimeError, "stale publication fence|restoration"
                ):
                    self._save_visibility(
                        path, run_identity, epoch=2, write_guard=guard
                    )
            self.assertTrue(guard.lost)
            self.assertEqual(
                len(simulator.replace_calls), guard.replace_count_at_loss
            )
            predecessor = sorted(root.glob(".last.pt.previous.*"))
            self.assertEqual(len(predecessor), 2)

    def test_torch_version_subclass_is_normalized_to_exact_str(self):
        class TorchVersionLike(str):
            pass

        with mock.patch.object(
            panderm.torch, "__version__", TorchVersionLike("2.6.0+cu124")
        ):
            versions = panderm.dependency_versions()
        self.assertIs(type(versions["torch"]), str)
        self.assertEqual(versions["torch"], "2.6.0+cu124")

    def test_run_identity_rejects_non_primitive_version_object(self):
        class TorchVersionLike(str):
            pass

        with self.assertRaisesRegex(ValueError, "non-primitive"):
            panderm_run.build_run_identity(
                git_commit="c" * 40,
                seed=0,
                epochs=5,
                evaluation_scope="validation_only",
                checkpoint_sha256="a" * 64,
                model_identity={
                    "arch": panderm_run.ARCH,
                    "drop_path": panderm_run.DROP_PATH,
                },
                manifest_sha256={"train": "t", "val": "v"},
                fixed_split_identity="t",
                shared_root_uuid="uuid",
                formal_output_identity="validation-output",
                dependency_versions={"torch": TorchVersionLike("2.6.0")},
                warmup_epochs=5,
                drop_path=0.2,
                amp_requested=True,
                amp_effective=False,
                device_type="cpu",
            )

    def test_dataset_free_production_payload_round_trip_cleans_temporary(self):
        model, optimizer, schedule, scaler = self._components()
        args = type("A", (), {"seed": 0, "epochs": 5})()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = set(root.iterdir())
            report = train_panderm.checkpoint_serialization_preflight(
                temporary_directory=root,
                model=model,
                optimizer=optimizer,
                schedule=schedule,
                scaler=scaler,
                args=args,
                run_identity=identity(),
            )
            self.assertTrue(report["weights_only_round_trip"])
            self.assertEqual(
                report["checkpoint_format"], panderm_run.CHECKPOINT_FORMAT
            )
            self.assertEqual(set(root.iterdir()), before)

    def test_post_step_optimizer_state_uses_same_production_round_trip(self):
        model, optimizer, schedule, scaler = self._components()
        inputs = torch.randn(2, 3, 224, 224)
        targets = torch.tensor([0, 1])
        loss = torch.nn.CrossEntropyLoss()(model(inputs), targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        schedule.step()
        self.assertTrue(optimizer.state)
        args = type("A", (), {"seed": 0, "epochs": 5})()
        with tempfile.TemporaryDirectory() as temporary:
            report = train_panderm.checkpoint_serialization_preflight(
                temporary_directory=temporary,
                model=model,
                optimizer=optimizer,
                schedule=schedule,
                scaler=scaler,
                args=args,
                run_identity=identity(),
            )
        self.assertTrue(report["weights_only_round_trip"])

    def test_normal_save_load_and_resume_have_verified_sidecar(self):
        run_identity = identity()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.pt"
            model, _, _, _ = self._save(path, run_identity)
            sidecar_path = self._sidecar_path(path)
            self.assertTrue(path.is_file())
            self.assertTrue(sidecar_path.is_file())
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(sidecar),
                {
                    "schema_version",
                    "checkpoint_filename",
                    "byte_size",
                    "sha256",
                    "epoch",
                    "global_step",
                    "checkpoint_format",
                    "run_identity_sha256",
                },
            )
            reopened = train_panderm.load_checkpoint_safe(
                path,
                map_location="cpu",
                model=model,
                expected_identity=run_identity,
            )
            fresh_model, fresh_opt, fresh_schedule, fresh_scaler = self._components()
            resumed = train_panderm.restore_checkpoint_state(
                reopened,
                fresh_model,
                fresh_opt,
                fresh_schedule,
                fresh_scaler,
                write_guard=AllowDurableWriteGuard(),
            )
            self.assertEqual(
                resumed,
                (2, 0.5, [{"epoch": 1, "optimizer_steps": 0}]),
            )

    def test_checkpoint_publish_rejects_rollback_and_same_step_drift(self):
        run_identity = identity()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.pt"
            model, optimizer, schedule, scaler = self._save(
                path, run_identity, epoch=2
            )
            sidecar_path = self._sidecar_path(path)
            before = (path.read_bytes(), sidecar_path.read_bytes())
            args = type("A", (), {"seed": 0, "epochs": 5})()
            history = [
                {"epoch": completed, "optimizer_steps": schedule.step_count}
                for completed in (1, 2)
            ]
            train_panderm.save_checkpoint(
                path,
                model,
                optimizer,
                schedule,
                scaler,
                2,
                0.5,
                history,
                args,
                run_identity,
                write_guard=AllowDurableWriteGuard(),
            )
            self.assertEqual(
                (path.read_bytes(), sidecar_path.read_bytes()),
                before,
            )
            with self.assertRaisesRegex(ValueError, "rollback"):
                self._save(path, run_identity, epoch=1)
            self.assertEqual(
                (path.read_bytes(), sidecar_path.read_bytes()),
                before,
            )

            model, optimizer, schedule, scaler = self._components()
            with torch.no_grad():
                next(model.parameters()).add_(1.0)
            history = [
                {"epoch": completed, "optimizer_steps": schedule.step_count}
                for completed in (1, 2)
            ]
            with self.assertRaisesRegex(ValueError, "same-step checkpoint differs"):
                train_panderm.save_checkpoint(
                    path,
                    model,
                    optimizer,
                    schedule,
                    scaler,
                    2,
                    0.5,
                    history,
                    args,
                    run_identity,
                    write_guard=AllowDurableWriteGuard(),
                )
            self.assertEqual(
                (path.read_bytes(), sidecar_path.read_bytes()),
                before,
            )

    def test_checkpoint_reopen_failure_restores_previous_complete_pair(self):
        run_identity = identity()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.pt"
            self._save(path, run_identity, epoch=1)
            sidecar_path = self._sidecar_path(path)
            before = (path.read_bytes(), sidecar_path.read_bytes())
            model, optimizer, schedule, scaler = self._components()
            args = type("A", (), {"seed": 0, "epochs": 5})()
            history = [
                {"epoch": completed, "optimizer_steps": schedule.step_count}
                for completed in (1, 2)
            ]
            real_wait = train_panderm._wait_for_checkpoint_pair_visibility
            publication_waits = []

            def fail_candidate_reopen(*args, phase, **kwargs):
                reopened = real_wait(*args, phase=phase, **kwargs)
                if phase == "checkpoint publication":
                    publication_waits.append(
                        train_panderm.sha256_file(args[0])
                    )
                    raise RuntimeError("simulated final reopen failure")
                return reopened

            with mock.patch.object(
                train_panderm,
                "_wait_for_checkpoint_pair_visibility",
                side_effect=fail_candidate_reopen,
            ):
                with self.assertRaisesRegex(RuntimeError, "reopen failure"):
                    train_panderm.save_checkpoint(
                        path,
                        model,
                        optimizer,
                        schedule,
                        scaler,
                        2,
                        0.5,
                        history,
                        args,
                        run_identity,
                        write_guard=AllowDurableWriteGuard(),
                    )
            self.assertEqual(
                (path.read_bytes(), sidecar_path.read_bytes()),
                before,
            )
            self.assertEqual(
                train_panderm.load_checkpoint_safe(
                    path, expected_identity=run_identity
                )["epoch"],
                1,
            )
            self.assertEqual(len(publication_waits), 1)

    def test_abrupt_publish_recovers_only_the_last_complete_predecessor(self):
        run_identity = identity()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.pt"
            model, _, _, _ = self._save(path, run_identity, epoch=1)
            sidecar_path = self._sidecar_path(path)
            previous_id = "a" * 32
            backup = path.parent / (
                f".{path.name}.previous.{previous_id}.pt"
            )
            backup_sidecar = path.parent / (
                f".{path.name}.previous.{previous_id}.integrity.json"
            )
            shutil.copy2(path, backup)
            shutil.copy2(sidecar_path, backup_sidecar)
            path.write_bytes(b"interrupted-new-checkpoint")
            sidecar_path.write_text('{"partial":true}\n', encoding="utf-8")

            recovered = train_panderm.load_checkpoint_for_resume(
                path,
                map_location="cpu",
                model=model,
                expected_identity=run_identity,
                write_guard=AllowDurableWriteGuard(),
            )
            self.assertEqual(recovered["epoch"], 1)
            self.assertEqual(path.read_bytes(), backup.read_bytes())
            self.assertEqual(
                sidecar_path.read_bytes(),
                backup_sidecar.read_bytes(),
            )
            self.assertTrue(backup.is_file())
            self.assertTrue(backup_sidecar.is_file())

    def test_resume_recovery_waits_for_stale_checkpoint_and_sidecar_visibility(self):
        run_identity = identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "last.pt"
            model, _, _, _ = self._save_visibility(
                path, run_identity, epoch=1
            )
            sidecar_path = self._sidecar_path(path)
            previous_id = "a" * 32
            backup = root / f".last.pt.previous.{previous_id}.pt"
            backup_sidecar = root / (
                f".last.pt.previous.{previous_id}.integrity.json"
            )
            shutil.copy2(path, backup)
            shutil.copy2(sidecar_path, backup_sidecar)
            expected_pair = (
                backup.read_bytes(),
                backup_sidecar.read_bytes(),
            )
            path.write_bytes(b"interrupted-new-checkpoint")
            sidecar_path.write_text('{"partial":true}\n', encoding="utf-8")
            simulator = self._VisibilitySimulator(
                path,
                checkpoint_visible_after=3,
                sidecar_visible_after=2,
            )
            with self._simulate_visibility(simulator):
                recovered = train_panderm.load_checkpoint_for_resume(
                    path,
                    map_location="cpu",
                    model=model,
                    expected_identity=run_identity,
                    write_guard=AllowDurableWriteGuard(),
                )
            self.assertEqual(recovered["epoch"], 1)
            self.assertEqual(
                (path.read_bytes(), sidecar_path.read_bytes()),
                expected_pair,
            )
            self.assertFalse(simulator.pending)
            self.assertTrue(backup.is_file())
            self.assertTrue(backup_sidecar.is_file())

    def test_resume_recovery_guard_loss_blocks_unauthorized_replace(self):
        run_identity = identity()
        cases = (
            ("checkpoint resume recovery checkpoint staging", 0),
            ("checkpoint resume recovery sidecar publish", 1),
            ("checkpoint resume recovery publication visibility wait", 2),
        )
        for target_phase, expected_replace_count in cases:
            with (
                self.subTest(target_phase=target_phase),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                path = root / "last.pt"
                model, _, _, _ = self._save_visibility(
                    path, run_identity, epoch=1
                )
                sidecar_path = self._sidecar_path(path)
                previous_id = "a" * 32
                backup = root / f".last.pt.previous.{previous_id}.pt"
                backup_sidecar = root / (
                    f".last.pt.previous.{previous_id}.integrity.json"
                )
                shutil.copy2(path, backup)
                shutil.copy2(sidecar_path, backup_sidecar)
                predecessor_pair = (
                    backup.read_bytes(),
                    backup_sidecar.read_bytes(),
                )
                path.write_bytes(b"interrupted-new-checkpoint")
                sidecar_path.write_text(
                    '{"partial":true}\n', encoding="utf-8"
                )
                replace_calls = []
                real_replace = train_panderm.os.replace

                def observed_replace(source, destination):
                    replace_calls.append(
                        (Path(source).name, Path(destination).name)
                    )
                    return real_replace(source, destination)

                class ExpiringGuard:
                    def __init__(inner_self):
                        inner_self.lost = False
                        inner_self.replace_count_at_loss = None

                    def require(inner_self, phase):
                        if target_phase in phase:
                            inner_self.lost = True
                            inner_self.replace_count_at_loss = len(
                                replace_calls
                            )
                            raise RuntimeError("stale recovery fence")
                        if inner_self.lost:
                            raise RuntimeError("stale recovery fence")

                guard = ExpiringGuard()
                with mock.patch.object(
                    train_panderm.os,
                    "replace",
                    side_effect=observed_replace,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "stale recovery fence"
                    ):
                        train_panderm.load_checkpoint_for_resume(
                            path,
                            map_location="cpu",
                            model=model,
                            expected_identity=run_identity,
                            write_guard=guard,
                        )
                self.assertTrue(guard.lost)
                self.assertEqual(
                    len(replace_calls), guard.replace_count_at_loss
                )
                self.assertEqual(
                    len(replace_calls), expected_replace_count
                )
                self.assertEqual(
                    (backup.read_bytes(), backup_sidecar.read_bytes()),
                    predecessor_pair,
                )
                self.assertTrue(backup.is_file())
                self.assertTrue(backup_sidecar.is_file())
                self.assertFalse(list(root.glob(".*.recovery.*")))
                self.assertFalse(list(root.glob(".*.recovery-sidecar.*")))

    def test_monotonic_result_publish_allows_exact_retry_and_rejects_rollback(self):
        run_identity = identity()
        first = {
            "epoch": 1,
            "global_step": 2,
            "history": [{"epoch": 1, "optimizer_steps": 2}],
            "run_identity": run_identity,
            "status": "complete",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            guard = AllowDurableWriteGuard().require
            panderm_run.write_monotonic_run_record_atomic(
                path, first, write_guard=guard
            )
            before = path.read_bytes()
            panderm_run.write_monotonic_run_record_atomic(
                path, copy.deepcopy(first), write_guard=guard
            )
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "rollback"):
                panderm_run.write_monotonic_run_record_atomic(
                    path,
                    {**first, "epoch": 0, "global_step": 0, "history": []},
                    write_guard=guard,
                )
            with self.assertRaisesRegex(ValueError, "same-step"):
                panderm_run.write_monotonic_run_record_atomic(
                    path,
                    {**first, "status": "different"},
                    write_guard=guard,
                )
            self.assertEqual(path.read_bytes(), before)

    def test_result_reopen_failure_restores_previous_history(self):
        run_identity = identity()
        first = {
            "epoch": 1,
            "global_step": 2,
            "history": [{"epoch": 1, "optimizer_steps": 2}],
            "run_identity": run_identity,
        }
        second = {
            "epoch": 2,
            "global_step": 4,
            "history": first["history"]
            + [{"epoch": 2, "optimizer_steps": 4}],
            "run_identity": run_identity,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            guard = AllowDurableWriteGuard().require
            panderm_run.write_monotonic_run_record_atomic(
                path, first, write_guard=guard
            )
            before = path.read_bytes()
            real_replace = panderm_run.os.replace
            publishes = []

            def replace_then_corrupt(source, destination):
                real_replace(source, destination)
                if Path(destination) == path and not publishes:
                    publishes.append(True)
                    path.write_text('{"corrupt":true}\n', encoding="utf-8")

            with mock.patch.object(
                panderm_run.os, "replace", side_effect=replace_then_corrupt
            ):
                with self.assertRaisesRegex(ValueError, "final reopen"):
                    panderm_run.write_monotonic_run_record_atomic(
                        path, second, write_guard=guard
                    )
            self.assertEqual(path.read_bytes(), before)

    def test_resume_loader_active_session_loss_rejects_before_any_restore(self):
        class ExpiringGuard:
            def __init__(self):
                self.active = True
                self.calls = []

            def require(self, phase):
                self.calls.append(phase)
                if not self.active:
                    raise RuntimeError("stale resume fence")

        guard = ExpiringGuard()
        restore = mock.Mock()

        def expire_during_load(*args, **kwargs):
            guard.active = False
            return {"checkpoint_format": panderm_run.CHECKPOINT_FORMAT}

        with mock.patch.object(
            train_panderm,
            "load_checkpoint_safe",
            side_effect=expire_during_load,
        ):
            with self.assertRaisesRegex(RuntimeError, "stale resume fence"):
                checkpoint = train_panderm.load_checkpoint_for_resume(
                    "last.pt",
                    map_location="cpu",
                    model=mock.Mock(),
                    expected_identity=identity(),
                    write_guard=guard,
                )
                restore(checkpoint)
        restore.assert_not_called()
        self.assertEqual(
            guard.calls,
            [
                "checkpoint resume before state load",
                "checkpoint resume after state load",
            ],
        )

    def test_resume_component_boundaries_block_all_stale_followup_mutations(self):
        run_identity = identity()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.pt"
            model, _, _, _ = self._save(path, run_identity)
            checkpoint = train_panderm.load_checkpoint_safe(
                path,
                map_location="cpu",
                model=model,
                expected_identity=run_identity,
            )

        phases = (
            ("checkpoint resume model state", []),
            ("checkpoint resume optimizer state", ["model"]),
            (
                "checkpoint resume scheduler state",
                ["model", "optimizer"],
            ),
            (
                "checkpoint resume scaler state",
                ["model", "optimizer", "scheduler"],
            ),
        )
        for failure_phase, expected_mutations in phases:
            with self.subTest(failure_phase=failure_phase):
                model, optimizer, schedule, scaler = self._components()
                mutations = []
                original_model = model.load_state_dict
                original_optimizer = optimizer.load_state_dict
                original_schedule = schedule.load_state_dict
                original_scaler = scaler.load_state_dict

                def guarded_call(name, function):
                    def invoke(*args, **kwargs):
                        mutations.append(name)
                        return function(*args, **kwargs)

                    return invoke

                class RejectingGuard:
                    def require(self, phase):
                        if phase == failure_phase:
                            raise RuntimeError("stale resume fence")

                with (
                    mock.patch.object(
                        model,
                        "load_state_dict",
                        side_effect=guarded_call("model", original_model),
                    ),
                    mock.patch.object(
                        optimizer,
                        "load_state_dict",
                        side_effect=guarded_call(
                            "optimizer", original_optimizer
                        ),
                    ),
                    mock.patch.object(
                        schedule,
                        "load_state_dict",
                        side_effect=guarded_call(
                            "scheduler", original_schedule
                        ),
                    ),
                    mock.patch.object(
                        scaler,
                        "load_state_dict",
                        side_effect=guarded_call("scaler", original_scaler),
                    ),
                    self.assertRaisesRegex(
                        RuntimeError, "stale resume fence"
                    ),
                ):
                    train_panderm.restore_checkpoint_state(
                        checkpoint,
                        model,
                        optimizer,
                        schedule,
                        scaler,
                        write_guard=RejectingGuard(),
                    )
                self.assertEqual(mutations, expected_mutations)

    def test_resume_rng_boundaries_block_stale_followup_mutations(self):
        state = train_panderm._get_rng_state()

        class RejectingGuard:
            def require(self, phase):
                if phase == "checkpoint resume NumPy RNG state":
                    raise RuntimeError("stale resume fence")

        with (
            mock.patch.object(train_panderm.random, "setstate") as python_rng,
            mock.patch.object(train_panderm.np.random, "set_state") as numpy_rng,
            mock.patch.object(train_panderm.torch, "set_rng_state") as torch_rng,
            self.assertRaisesRegex(RuntimeError, "stale resume fence"),
        ):
            train_panderm._set_rng_state(
                state,
                write_guard=RejectingGuard(),
            )
        python_rng.assert_called_once()
        numpy_rng.assert_not_called()
        torch_rng.assert_not_called()

    def test_checkpoint_shape_dtype_value_and_byte_tamper_reject_before_load(self):
        run_identity = identity()
        mutations = {
            "shape": lambda payload: payload["model_state_dict"].__setitem__(
                "head.weight",
                payload["model_state_dict"]["head.weight"].reshape(-1),
            ),
            "dtype": lambda payload: payload["model_state_dict"].__setitem__(
                "head.weight",
                payload["model_state_dict"]["head.weight"].to(torch.float64),
            ),
            "value": lambda payload: payload["model_state_dict"].__setitem__(
                "head.weight",
                payload["model_state_dict"]["head.weight"] + 1,
            ),
        }
        for name in (*mutations, "arbitrary_byte"):
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "last.pt"
                    model, _, _, _ = self._save(path, run_identity)
                    if name == "arbitrary_byte":
                        path.write_bytes(path.read_bytes() + b"tamper")
                    else:
                        payload = train_panderm.load_checkpoint_safe(path)
                        mutations[name](payload)
                        torch.save(payload, path)
                    before = panderm.snapshot_parameters(model)
                    with mock.patch.object(
                        train_panderm.torch,
                        "load",
                        side_effect=AssertionError("deserialize reached"),
                    ):
                        with self.assertRaisesRegex(ValueError, "integrity"):
                            train_panderm.load_checkpoint_safe(
                                path,
                                model=model,
                                expected_identity=run_identity,
                            )
                    self.assertEqual(
                        panderm.changed_parameter_count(before, model), 0
                    )

    def test_sidecar_tamper_matrix_rejects_before_deserialize(self):
        run_identity = identity()
        mutations = {
            "wrong_sha": lambda value: value.__setitem__("sha256", "0" * 64),
            "wrong_size": lambda value: value.__setitem__(
                "byte_size", value["byte_size"] + 1
            ),
            "wrong_filename": lambda value: value.__setitem__(
                "checkpoint_filename", "other.pt"
            ),
            "wrong_identity": lambda value: value.__setitem__(
                "run_identity_sha256", "0" * 64
            ),
            "wrong_schema": lambda value: value.__setitem__(
                "schema_version", 99
            ),
        }
        for name in (*mutations, "missing"):
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "last.pt"
                    model, _, _, _ = self._save(path, run_identity)
                    sidecar_path = self._sidecar_path(path)
                    if name == "missing":
                        sidecar_path.unlink()
                    else:
                        sidecar = json.loads(
                            sidecar_path.read_text(encoding="utf-8")
                        )
                        mutations[name](sidecar)
                        sidecar_path.write_text(
                            json.dumps(sidecar), encoding="utf-8"
                        )
                    with mock.patch.object(
                        train_panderm.torch,
                        "load",
                        side_effect=AssertionError("deserialize reached"),
                    ):
                        with self.assertRaises((ValueError, FileNotFoundError)):
                            train_panderm.load_checkpoint_safe(
                                path,
                                model=model,
                                expected_identity=run_identity,
                            )

    def test_sidecar_atomic_replace_fails_bounded_when_never_exact(self):
        value = {
            "schema_version": 1,
            "checkpoint_filename": "last.pt",
            "byte_size": 1,
            "sha256": "a" * 64,
            "epoch": 1,
            "checkpoint_format": panderm_run.CHECKPOINT_FORMAT,
            "run_identity_sha256": "b" * 64,
        }
        real_replace = train_panderm.os.replace
        now = [0.0]

        def monotonic():
            return now[0]

        def sleep(seconds):
            now[0] += seconds

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.pt.integrity.json"

            def replace_then_tamper(source, destination):
                real_replace(source, destination)
                Path(destination).write_text("{}", encoding="utf-8")

            with mock.patch.object(
                train_panderm.os,
                "replace",
                side_effect=replace_then_tamper,
            ), mock.patch.object(
                train_panderm,
                "_checkpoint_visibility_monotonic",
                side_effect=monotonic,
            ), mock.patch.object(
                train_panderm,
                "_checkpoint_visibility_sleep",
                side_effect=sleep,
            ), mock.patch.object(
                train_panderm,
                "CHECKPOINT_PUBLICATION_VISIBILITY_TIMEOUT_SECONDS",
                2.0,
            ), mock.patch.object(
                train_panderm,
                "CHECKPOINT_PUBLICATION_VISIBILITY_POLL_SECONDS",
                1.0,
            ), mock.patch.object(
                train_panderm,
                "CHECKPOINT_PUBLICATION_HEARTBEAT_SECONDS",
                1.0,
            ):
                with self.assertRaisesRegex(
                    TimeoutError,
                    "checkpoint integrity publication visibility timeout",
                ):
                    train_panderm._write_integrity_sidecar_atomic(
                        path,
                        value,
                        write_guard=AllowDurableWriteGuard(),
                    )

    def test_result_checkpoint_relationship_rejects_stale_or_wrong_record(self):
        run_identity = identity()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.pt"
            model, _, _, _ = self._save(path, run_identity)
            record = train_panderm.checkpoint_integrity_record(
                path, expected_identity=run_identity
            )
            wrong = copy.deepcopy(record)
            wrong["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "result"):
                train_panderm.load_checkpoint_safe(
                    path,
                    model=model,
                    expected_identity=run_identity,
                    expected_result_checkpoint=wrong,
                )

            with torch.no_grad():
                model.head.weight.add_(1)
            self._save(path, run_identity, epoch=2)
            with self.assertRaisesRegex(ValueError, "result"):
                train_panderm.load_checkpoint_safe(
                    path,
                    model=model,
                    expected_identity=run_identity,
                    expected_result_checkpoint=record,
                )

    def test_completed_pair_value_tamper_rejects_before_state_mutation(self):
        run_identity = identity()
        for tampered_name in ("best.pt", "last.pt"):
            with self.subTest(tampered=tampered_name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    best_path = root / "best.pt"
                    last_path = root / "last.pt"
                    model, _, _, _ = self._save(best_path, run_identity)
                    self._save(last_path, run_identity)
                    result = {
                        "checkpoint_integrity": {
                            "best.pt": train_panderm.checkpoint_integrity_record(
                                best_path, expected_identity=run_identity
                            ),
                            "last.pt": train_panderm.checkpoint_integrity_record(
                                last_path, expected_identity=run_identity
                            ),
                        }
                    }
                    path = root / tampered_name
                    payload = train_panderm.load_checkpoint_safe(path)
                    payload["model_state_dict"]["head.weight"] += 1
                    torch.save(payload, path)
                    before = panderm.snapshot_parameters(model)
                    with self.assertRaisesRegex(ValueError, "integrity"):
                        train_panderm.load_completed_checkpoint_pair_safe(
                            best_path=best_path,
                            last_path=last_path,
                            result=result,
                            model=model,
                            expected_identity=run_identity,
                            map_location="cpu",
                        )
                    self.assertEqual(
                        panderm.changed_parameter_count(before, model), 0
                    )


class PanDermRunnerMockSmokeTests(unittest.TestCase):
    def test_runner_writes_verified_pairs_and_completed_resume_skips(self):
        import pandas as pd

        class Loader:
            def __len__(self):
                return 8

        train_frame = pd.DataFrame(
            [{"dx": "df", "image_path": "train.jpg", "label_idx": 3}]
        )
        val_frame = pd.DataFrame(
            [{"dx": "df", "image_path": "val.jpg", "label_idx": 3}]
        )
        validation_metrics = {
            "target_f1": 0.5,
            "macro_f1": 0.4,
            "confusion_matrix": torch.eye(7, dtype=torch.int64).tolist(),
        }
        models = [build_mock_model(), build_mock_model()]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            argv = [
                "--checkpoint", str(root / "weights.pth"),
                "--checkpoint-sha256", "a" * 64,
                "--upstream-dir", str(root / "upstream"),
                "--output-dir", str(output),
                "--shared-root-uuid", "uuid",
                "--formal-output-identity", "validation-output",
                "--fixed-split-identity", "train-hash",
                "--device", "cpu",
            ]
            with contextlib.ExitStack() as stack:
                runner_guard = AllowDurableWriteGuard()
                stack.enter_context(
                    mock.patch.object(
                        train_panderm.panderm_run.SequentialSessionWriteGuard,
                        "from_environment",
                        return_value=runner_guard,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        train_panderm.panderm_run,
                        "require_checkpoint_sha256",
                        return_value="a" * 64,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        train_panderm.panderm_run,
                        "require_no_deployment_contamination",
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        train_panderm.panderm_run,
                        "require_provenance_clearance",
                        return_value={"cleared_for": "validation_only"},
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        train_panderm, "build_c1_frame", return_value=train_frame
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        train_panderm.manifests,
                        "load_split",
                        return_value=val_frame,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        train_panderm, "build_loader", return_value=Loader()
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        train_panderm.panderm,
                        "build_train_transform",
                        return_value=object(),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        train_panderm.panderm,
                        "build_eval_transform",
                        return_value=object(),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        train_panderm.panderm,
                        "build_panderm_classifier",
                        side_effect=models,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        train_panderm.panderm,
                        "model_identity",
                        return_value={
                            "arch": panderm_run.ARCH,
                            "drop_path": panderm_run.DROP_PATH,
                            "total_parameter_count": 1,
                            "trainable_parameter_count": 1,
                        },
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        train_panderm.panderm,
                        "dependency_versions",
                        return_value={"torch": "mock"},
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        train_panderm,
                        "manifest_identity",
                        return_value={"train": "train-hash", "val": "val-hash"},
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        train_panderm, "git_commit", return_value="c" * 40
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        train_panderm,
                        "train_one_epoch",
                        return_value=(1.0, 1),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        train_panderm,
                        "evaluate",
                        return_value=validation_metrics,
                    )
                )
                train_panderm.main(argv)
                train_panderm.main(argv + ["--resume"])

            checkpoint_dir = (
                output / "checkpoints" / panderm_run.ARCH / "C1_seed0"
            )
            result_path = (
                output / "results" / panderm_run.ARCH / "results_C1_seed0.json"
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(result["checkpoint_integrity"]), {"best.pt", "last.pt"}
            )
            for name in ("best.pt", "last.pt"):
                self.assertTrue((checkpoint_dir / name).is_file())
                self.assertTrue(
                    (checkpoint_dir / f"{name}.integrity.json").is_file()
                )


class CliRuntimeIdentityTests(unittest.TestCase):
    def test_runtime_drop_path_and_amp_are_recorded_exactly(self):
        cpu = identity()
        self.assertEqual(cpu["model_identity"]["drop_path"], 0.2)
        self.assertEqual(cpu["optimization"]["drop_path"], 0.2)
        self.assertTrue(cpu["optimization"]["amp_requested"])
        self.assertFalse(cpu["optimization"]["amp_effective"])
        self.assertEqual(cpu["optimization"]["device_type"], "cpu")

        cuda = panderm_run.build_run_identity(
            git_commit="c" * 40,
            seed=0,
            epochs=5,
            evaluation_scope="validation_only",
            checkpoint_sha256="a" * 64,
            model_identity={"arch": panderm_run.ARCH, "drop_path": 0.2},
            manifest_sha256={"train": "t", "val": "v"},
            fixed_split_identity="t",
            shared_root_uuid="uuid",
            formal_output_identity="validation-output",
            dependency_versions={},
            warmup_epochs=5,
            drop_path=0.2,
            amp_requested=True,
            amp_effective=True,
            device_type="cuda",
        )
        self.assertTrue(cuda["optimization"]["amp_effective"])

    def test_unsupported_runtime_semantics_rejected(self):
        base = [
            "--checkpoint", "weights.pth",
            "--upstream-dir", "upstream",
            "--output-dir", "out",
        ]
        for extra in (
            ["--drop-path", "0.3"],
            ["--no-amp"],
            ["--evaluation-scope", "full"],
        ):
            with self.subTest(extra=extra):
                with self.assertRaises(SystemExit):
                    train_panderm.parse_args(base + extra)

        factory = mock.Mock()
        with self.assertRaisesRegex(ValueError, "drop_path must be exactly"):
            panderm.build_panderm_classifier(
                drop_path=0.3,
                model_factory=factory,
                state_dict={},
            )
        factory.assert_not_called()

    def test_resume_amp_and_drop_path_drift_rejected_before_mutation(self):
        saved = identity()
        model = build_mock_model()
        before = panderm.snapshot_parameters(model)
        for mutate in (
            lambda current: current["optimization"].update(drop_path=0.3),
            lambda current: current["optimization"].update(amp_effective=True),
        ):
            current = copy.deepcopy(saved)
            mutate(current)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                panderm_run.require_matching_identity(saved, current)
            self.assertEqual(panderm.changed_parameter_count(before, model), 0)


if __name__ == "__main__":
    unittest.main()
