"""Regression matrices for the five PanDerm reviewer blockers."""

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

from tests.test_panderm_base_c1_finetune import build_mock_model

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
                )

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
                    )
            self.assertTrue(cache.is_dir())
            self.assertFalse(
                (cache / panderm_run.VALIDATION_ARCHIVE_READY_FILENAME).exists()
            )

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

    # --- the reviewer's coordinated-rewrite attack ---------------------------
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
                )
            self.assertFalse((runtime / "data").exists())


RACE_WORKER = r'''
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from ddpm_derm import panderm_run

lock_path = Path(sys.argv[2])
session_id = sys.argv[3]
start_at = float(sys.argv[4])
calls = {"staging": 0, "attempt": 0, "runner": 0}
# Spin to the shared wall-clock instant so both processes contend at once.
while time.time() < start_at:
    pass
try:
    marker = panderm_run.acquire_validation_run_lock(
        lock_path,
        session_id=session_id,
        run_version=panderm_run.RUN_VERSION,
        git_commit="c" * 40,
        shared_root_uuid="765b971f-d148-4960-a77d-b73f28fc013c",
        account_label="A",
    )
    # Only a winner may do any of these.
    calls["staging"] += 1
    calls["attempt"] += 1
    calls["runner"] += 1
    print(json.dumps({"acquired": True, "session_id": session_id,
                      "owner": marker["session_id"], "calls": calls}))
except FileExistsError as error:
    print(json.dumps({"acquired": False, "session_id": session_id,
                      "calls": calls, "error": str(error)[:200]}))
'''


class AtomicValidationRunLockTests(unittest.TestCase):
    """One validation per run version, enforced atomically across accounts.

    The previous guard only listed existing attempt directories, which is a
    TOCTOU: account A and account B can both observe "no attempt yet" and both
    proceed. These tests use two genuinely concurrent OS processes rather than
    sequential mocks, because a sequential test cannot distinguish an atomic
    exclusive create from a check-then-write.
    """

    def _root(self, base):
        root = Path(base) / panderm_run.RUN_VERSION
        root.mkdir(parents=True)
        return Path(base)

    def _acquire(self, lock_path, session_id, **overrides):
        kwargs = dict(
            session_id=session_id,
            run_version=panderm_run.RUN_VERSION,
            git_commit="c" * 40,
            shared_root_uuid="765b971f-d148-4960-a77d-b73f28fc013c",
            account_label="A",
        )
        kwargs.update(overrides)
        return panderm_run.acquire_validation_run_lock(lock_path, **kwargs)

    def test_lock_path_is_fixed_per_run_version_not_per_attempt(self):
        path = panderm_run.validation_run_lock_path(Path("/shared"))
        self.assertEqual(path.name, panderm_run.VALIDATION_RUN_LOCK_FILENAME)
        self.assertEqual(path.parent.name, panderm_run.RUN_VERSION)
        # Never nested under a timestamped attempt directory.
        self.assertNotIn("validation_runs", path.parts)
        other = panderm_run.validation_run_lock_path(Path("/shared"))
        self.assertEqual(path, other)

    def test_two_concurrent_processes_yield_exactly_one_winner(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = self._root(temporary)
            lock_path = panderm_run.validation_run_lock_path(base)
            worker = Path(temporary) / "worker.py"
            worker.write_text(RACE_WORKER, encoding="utf-8")
            src = str(Path(panderm_run.__file__).resolve().parents[1])
            start_at = time.time() + 1.5
            sessions = [str(uuid.uuid4()), str(uuid.uuid4())]
            processes = [
                subprocess.Popen(
                    [sys.executable, "-B", "-u", str(worker), src,
                     str(lock_path), session, str(start_at)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
                for session in sessions
            ]
            outputs = [process.communicate()[0] for process in processes]
            results = []
            for output in outputs:
                line = [l for l in output.strip().splitlines() if l.startswith("{")]
                self.assertTrue(line, output)
                results.append(json.loads(line[-1]))

            winners = [r for r in results if r["acquired"]]
            losers = [r for r in results if not r["acquired"]]
            self.assertEqual(len(winners), 1, results)
            self.assertEqual(len(losers), 1, results)
            # The loser did no staging, created no attempt, started no runner.
            self.assertEqual(
                losers[0]["calls"], {"staging": 0, "attempt": 0, "runner": 0}
            )
            self.assertIn("already holds the run lock", losers[0]["error"])
            # The surviving marker belongs to the winner.
            holder = panderm_run._read_validation_run_lock(lock_path)
            self.assertEqual(holder["session_id"], winners[0]["session_id"])

    def test_second_sequential_account_is_refused_and_lock_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = self._root(temporary)
            lock_path = panderm_run.validation_run_lock_path(base)
            first = str(uuid.uuid4())
            marker = self._acquire(lock_path, first)
            before = lock_path.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "already holds the run lock"):
                self._acquire(lock_path, str(uuid.uuid4()), account_label="B")
            self.assertEqual(lock_path.read_bytes(), before)
            self.assertEqual(
                panderm_run._read_validation_run_lock(lock_path)["session_id"],
                marker["session_id"],
            )

    def test_lock_identity_carries_the_required_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = self._root(temporary)
            lock_path = panderm_run.validation_run_lock_path(base)
            marker = self._acquire(lock_path, str(uuid.uuid4()))
            self.assertEqual(
                set(marker), set(panderm_run.VALIDATION_RUN_LOCK_FIELDS)
            )
            self.assertEqual(marker["run_version"], panderm_run.RUN_VERSION)
            self.assertEqual(marker["evaluation_scope"], panderm_run.VALIDATION_ONLY)
            self.assertEqual(len(marker["git_commit"]), 40)
            uuid.UUID(marker["session_id"])
            uuid.UUID(marker["shared_root_uuid"])
            self.assertTrue(marker["acquired_utc"])
            self.assertTrue(marker["hostname"])

    def test_formal_or_test_scope_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = self._root(temporary)
            lock_path = panderm_run.validation_run_lock_path(base)
            for scope in (panderm_run.FORMAL_TRAINING, panderm_run.TEST_ACCESS):
                with self.subTest(scope=scope):
                    with self.assertRaises(ValueError):
                        self._acquire(
                            lock_path, str(uuid.uuid4()), evaluation_scope=scope
                        )
            self.assertFalse(lock_path.exists())

    def test_owner_can_release_and_next_session_can_acquire(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = self._root(temporary)
            lock_path = panderm_run.validation_run_lock_path(base)
            first = str(uuid.uuid4())
            self._acquire(lock_path, first)
            panderm_run.release_validation_run_lock(lock_path, session_id=first)
            self.assertFalse(lock_path.exists())
            second = str(uuid.uuid4())
            self._acquire(lock_path, second, account_label="B")
            self.assertEqual(
                panderm_run._read_validation_run_lock(lock_path)["session_id"], second
            )

    def test_wrong_owner_cannot_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = self._root(temporary)
            lock_path = panderm_run.validation_run_lock_path(base)
            owner = str(uuid.uuid4())
            self._acquire(lock_path, owner)
            with self.assertRaisesRegex(PermissionError, "owned by another session"):
                panderm_run.release_validation_run_lock(
                    lock_path, session_id=str(uuid.uuid4())
                )
            self.assertTrue(lock_path.exists())
            self.assertEqual(
                panderm_run._read_validation_run_lock(lock_path)["session_id"], owner
            )

    def test_caught_failure_releases_only_its_own_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = self._root(temporary)
            lock_path = panderm_run.validation_run_lock_path(base)
            owner = str(uuid.uuid4())
            self._acquire(lock_path, owner)
            released = False
            try:
                raise RuntimeError("validation failed")
            except RuntimeError:
                panderm_run.release_validation_run_lock(lock_path, session_id=owner)
                released = True
            self.assertTrue(released)
            self.assertFalse(lock_path.exists())

    def test_abrupt_termination_leaves_the_marker_for_the_next_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = self._root(temporary)
            lock_path = panderm_run.validation_run_lock_path(base)
            worker = Path(temporary) / "worker.py"
            worker.write_text(RACE_WORKER, encoding="utf-8")
            src = str(Path(panderm_run.__file__).resolve().parents[1])
            dead = str(uuid.uuid4())
            process = subprocess.Popen(
                [sys.executable, "-B", "-u", str(worker), src, str(lock_path),
                 dead, str(time.time())],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            process.communicate()
            # The process is gone; nothing time-based may reclaim its lock.
            self.assertTrue(lock_path.exists())
            with self.assertRaisesRegex(FileExistsError, "already holds the run lock"):
                self._acquire(lock_path, str(uuid.uuid4()), account_label="B")
            self.assertEqual(
                panderm_run._read_validation_run_lock(lock_path)["session_id"], dead
            )

    def test_stale_lock_is_never_cleared_automatically(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = self._root(temporary)
            lock_path = panderm_run.validation_run_lock_path(base)
            dead = str(uuid.uuid4())
            self._acquire(lock_path, dead)
            for _ in range(3):
                with self.assertRaises(FileExistsError):
                    self._acquire(lock_path, str(uuid.uuid4()))
            self.assertTrue(lock_path.exists())

    def test_manual_stale_clear_requires_confirmation_and_exact_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = self._root(temporary)
            lock_path = panderm_run.validation_run_lock_path(base)
            dead = str(uuid.uuid4())
            self._acquire(lock_path, dead)
            with self.assertRaisesRegex(ValueError, "confirmation text"):
                panderm_run.clear_stale_validation_run_lock(
                    lock_path, stale_session_id=dead, confirmation="yes"
                )
            self.assertTrue(lock_path.exists())
            with self.assertRaisesRegex(PermissionError, "owner mismatch"):
                panderm_run.clear_stale_validation_run_lock(
                    lock_path,
                    stale_session_id=str(uuid.uuid4()),
                    confirmation=panderm_run.VALIDATION_RUN_LOCK_CLEAR_CONFIRMATION,
                )
            self.assertTrue(lock_path.exists())
            panderm_run.clear_stale_validation_run_lock(
                lock_path,
                stale_session_id=dead,
                confirmation=panderm_run.VALIDATION_RUN_LOCK_CLEAR_CONFIRMATION,
            )
            self.assertFalse(lock_path.exists())

    def test_account_label_is_operational_only_and_does_not_split_the_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = self._root(temporary)
            lock_path = panderm_run.validation_run_lock_path(base)
            self._acquire(lock_path, str(uuid.uuid4()), account_label="A")
            for label in ("B", "C"):
                with self.subTest(account_label=label):
                    with self.assertRaises(FileExistsError):
                        self._acquire(
                            lock_path, str(uuid.uuid4()), account_label=label
                        )

    def test_different_attempt_timestamps_share_one_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = self._root(temporary)
            runs = base / panderm_run.RUN_VERSION / "validation_runs"
            (runs / "20260101T000000Z").mkdir(parents=True)
            (runs / "20260102T000000Z").mkdir(parents=True)
            lock_path = panderm_run.validation_run_lock_path(base)
            self._acquire(lock_path, str(uuid.uuid4()))
            # A second attempt timestamp must not get its own lock.
            with self.assertRaises(FileExistsError):
                self._acquire(lock_path, str(uuid.uuid4()))
            self.assertEqual(
                len(list(panderm_run.validation_run_lock_path(base).parent.glob(
                    "*.lock.json"))), 1
            )

    def test_a_different_run_version_uses_a_separate_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / panderm_run.RUN_VERSION).mkdir(parents=True)
            (base / "v2_other_run").mkdir(parents=True)
            first = panderm_run.validation_run_lock_path(base)
            second = panderm_run.validation_run_lock_path(
                base, run_version="v2_other_run"
            )
            self.assertNotEqual(first, second)
            self._acquire(first, str(uuid.uuid4()))
            # Holding v1 must not deadlock an unrelated version.
            self._acquire(
                second, str(uuid.uuid4()), run_version="v2_other_run"
            )
            self.assertTrue(first.exists() and second.exists())

    def test_marker_tamper_and_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = self._root(temporary)
            lock_path = panderm_run.validation_run_lock_path(base)
            owner = str(uuid.uuid4())
            self._acquire(lock_path, owner)
            good = json.loads(lock_path.read_text(encoding="utf-8"))
            cases = {
                "missing_field": {k: v for k, v in good.items() if k != "git_commit"},
                "extra_field": {**good, "sneaky": "x"},
                "empty_owner": {**good, "session_id": ""},
                "wrong_type": {**good, "schema_version": "1"},
            }
            for name, tampered in cases.items():
                with self.subTest(case=name):
                    lock_path.write_text(json.dumps(tampered), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        panderm_run._read_validation_run_lock(lock_path)
                    with self.assertRaises(ValueError):
                        panderm_run.release_validation_run_lock(
                            lock_path, session_id=owner
                        )
            self.assertTrue(lock_path.exists())

    def test_acquire_requires_an_existing_run_version_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = panderm_run.validation_run_lock_path(Path(temporary))
            with self.assertRaisesRegex(FileNotFoundError, "directory is missing"):
                self._acquire(lock_path, str(uuid.uuid4()))
            self.assertFalse(lock_path.exists())


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
    def _components(self):
        model = build_mock_model()
        optimizer = panderm.build_optimizer(model, num_layers=4)
        schedule = panderm.WarmupCosineSchedule(
            optimizer, warmup_epochs=1, epochs=2, steps_per_epoch=2
        )
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        return model, optimizer, schedule, scaler

    def _save(self, path, run_identity, epoch=1):
        model, optimizer, schedule, scaler = self._components()
        args = type("A", (), {"seed": 0, "epochs": 5})()
        train_panderm.save_checkpoint(
            path,
            model,
            optimizer,
            schedule,
            scaler,
            epoch,
            0.5,
            [{"epoch": epoch}],
            args,
            run_identity,
        )
        return model, optimizer, schedule, scaler

    def _sidecar_path(self, checkpoint):
        return checkpoint.with_name(checkpoint.name + ".integrity.json")

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
            )
            self.assertEqual(resumed, (2, 0.5, [{"epoch": 1}]))

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

    def test_sidecar_atomic_replace_reopens_exactly(self):
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
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.pt.integrity.json"

            def replace_then_tamper(source, destination):
                real_replace(source, destination)
                Path(destination).write_text("{}", encoding="utf-8")

            with mock.patch.object(
                train_panderm.os,
                "replace",
                side_effect=replace_then_tamper,
            ):
                with self.assertRaisesRegex(ValueError, "sidecar reopen"):
                    train_panderm._write_integrity_sidecar_atomic(path, value)

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
