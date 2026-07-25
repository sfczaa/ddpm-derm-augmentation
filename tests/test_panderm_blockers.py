"""Regression coverage for PanDerm archive integrity and session recovery."""

from __future__ import annotations

import copy
import ast
import contextlib
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def _fixture(self, root):
        shared = root / "shared_data"
        mixed = shared / "raw" / "mixed"
        mixed.mkdir(parents=True)
        for name in ("train_a.jpg", "val_a.jpg", "test_only.jpg"):
            (mixed / name).write_bytes(name.encode("ascii"))
        manifests = shared / "manifests"
        self._write_manifest(manifests / "train.csv", ["raw/mixed/train_a.jpg"])
        self._write_manifest(manifests / "val.csv", ["raw/mixed/val_a.jpg"])
        self._write_manifest(manifests / "test.csv", ["raw/mixed/test_only.jpg"])
        (manifests / "class_to_idx.json").write_text(
            '{"akiec": 0}\n', encoding="utf-8"
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


class IdentityAdversarialMatrixTests(unittest.TestCase):
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
