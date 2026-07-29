"""Static safety contracts for the two PanDerm Colab notebooks."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from contextlib import redirect_stderr
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm import panderm_run, train_panderm  # noqa: E402


VALIDATION = "colab_panderm_base_c1_finetune_validation.ipynb"
FORMAL = "colab_panderm_base_c1_finetune_classifier.ipynb"
NAMES = (VALIDATION, FORMAL)

PROTECTED_NOTEBOOK = "colab_balanced_ddpm.ipynb"
PROTECTED_SHA256 = "ef8bb8be8fa0865a3297e361f1984631141eadca073cc1451ad5223ce27882b8"
PHASE2_TRAIN_SHA256 = "eea3fdf281120687b45dfcb5139d927888c6dc84867c67f053c7a743eb02b6fa"
PHASE2_VAL_SHA256 = "22a87a1ab4009c9e87462381f9ef35ad7a5eae7217057049fc24e5531df819f4"
PHASE2_MAPPING_SHA256 = "5a034b7dc0c6f44543f558aa589b8e1cba12a05b71a18ff0e2d2029a2ad2e66c"

PIN_PLACEHOLDER = "REPLACE_AFTER_PUSH"
PINNED_IMPLEMENTATION_COMMIT = "bd4b532bcba34f4942fe99a759d4165b3ee45c06"

FROZEN_NOTEBOOKS = (
    "colab_balanced_ddpm_classifier_train.ipynb",
    "colab_balanced_ddpm_classifier_validate.ipynb",
    "colab_classifier_baseline.ipynb",
    "colab_coca_classifier.ipynb",
    "colab_coca_v2_weighted_classifier.ipynb",
    "colab_coca_v2_weighted_validation.ipynb",
    "colab_coca_v3_inverse_frequency_classifier.ipynb",
    "colab_coca_v3_inverse_frequency_validation.ipynb",
    "colab_coca_v4_all_class_separability_diagnostic.ipynb",
    "colab_coca_v4_focal_inverse_frequency_classifier.ipynb",
    "colab_coca_v4_focal_inverse_frequency_validation.ipynb",
    "colab_coca_v4_post_failure_embedding_diagnostic.ipynb",
    "colab_coca_v4_synthetic_mixture_diagnostic.ipynb",
    "colab_coca_validation.ipynb",
    "colab_ddpm.ipynb",
)


def load(name):
    notebook = json.loads(
        (ROOT / "notebooks" / name).read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    return notebook, code


def load_phase2_manifest_helpers():
    notebook, _ = load(VALIDATION)
    phase2 = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "prepare_test_manifest_data_root" in "".join(cell.get("source", []))
    )
    tree = ast.parse(phase2)
    helper_names = {
        "assert_phase2_real_path",
        "prepare_test_manifest_data_root",
        "verify_test_manifest_data_root",
    }
    constant_names = {
        "PHASE2_EXPECTED_MANIFEST_SHA256",
        "PHASE2_EXPECTED_CLASS_MAPPING_SHA256",
        "PHASE2_EXPECTED_CLASS_TO_IDX",
    }
    selected_nodes = []
    selected_names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in helper_names:
                selected_nodes.append(node)
                selected_names.add(node.name)
        elif isinstance(node, ast.Assign):
            names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if names & constant_names:
                selected_nodes.append(node)
                selected_names.update(names & constant_names)
    required_names = helper_names | constant_names
    if selected_names != required_names:
        raise AssertionError(
            f"Phase 2 manifest helper definitions are incomplete: "
            f"{sorted(required_names - selected_names)}"
        )
    namespace = {
        "Path": Path,
        "csv": csv,
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "stat": stat,
    }
    exec(
        compile(
            ast.Module(body=selected_nodes, type_ignores=[]),
            "phase2-manifest-helpers",
            "exec",
        ),
        namespace,
    )
    return (
        namespace["prepare_test_manifest_data_root"],
        namespace["verify_test_manifest_data_root"],
    )


VALIDATION_ORDER_TOKENS = (
    (
        "checkpoint SHA",
        "checkpoint_sha256 = panderm_run.require_checkpoint_sha256",
    ),
    (
        "CPU checkpoint layout",
        "pretrained_layout = panderm.detect_checkpoint_layout",
    ),
    (
        "CPU model",
        "preflight_model = panderm.build_panderm_classifier",
    ),
    (
        "checkpoint serialization",
        "checkpoint_preflight = train_panderm.checkpoint_serialization_preflight",
    ),
    (
        "manifest checks",
        "frames = {split: manifests.load_split(split)",
    ),
    ("GPU smoke start", 'print("[Phase 4] START'),
    ("GPU smoke assertions complete", "GPU_SMOKE_COMPLETE = True"),
    (
        "GPU model and tensor cleanup",
        "del model, optimizer, schedule, scaler, batch, logits",
    ),
    ("GPU cache cleanup", "torch.cuda.empty_cache()"),
    ("all preflights complete", "PHASE3_COMPLETE = True"),
    (
        "version parent setup",
        "verified_v1_root = panderm_run.ensure_tree",
    ),
    (
        "version parent verification",
        "resolved_v1_root = verified_v1_root.resolve(strict=True)",
    ),
    (
        "validation lock path",
        "VALIDATION_RUN_LOCK = panderm_run.validation_run_lock_path",
    ),
    (
        "validation lock parent verification",
        "assert VALIDATION_RUN_LOCK.parent.samefile(verified_v1_root)",
    ),
    ("validation lock acquisition", "panderm_run.acquire_validation_run_lock"),
    ("archive build", "panderm_run.build_validation_archive_cache"),
    ("archive reuse", "panderm_run.reuse_validation_archive_cache"),
    ("validation id", "validation_id = datetime"),
    (
        "attempt directory",
        "VALIDATION_DIR = panderm_run.ensure_tree",
    ),
    (
        "fresh five-epoch runner",
        "gate_seconds, gate_output = run_stream(gate_command",
    ),
)


def validate_validation_notebook_order(source):
    """Reject durable validation work before every cheap/GPU preflight passes."""
    positions = {}
    for label, token in VALIDATION_ORDER_TOKENS:
        try:
            positions[label] = source.index(token)
        except ValueError as error:
            raise AssertionError(f"missing validation ordering token: {label}") from error
    for (earlier, _), (later, _) in zip(
        VALIDATION_ORDER_TOKENS,
        VALIDATION_ORDER_TOKENS[1:],
    ):
        if positions[earlier] >= positions[later]:
            raise AssertionError(
                f"validation ordering violation: {earlier} must precede {later}"
            )
    return positions


def prepare_fresh_validation_lock(shared_run_root):
    """Mirror the notebook's bounded, fail-loud version-parent setup."""
    resolved_root = panderm_run.require_existing_shared_root(shared_run_root)
    version_root = Path(shared_run_root) / panderm_run.RUN_VERSION
    if version_root.is_symlink():
        raise ValueError(f"validation version parent must not be a symlink: {version_root}")
    try:
        verified_v1_root = panderm_run.ensure_tree(
            shared_run_root,
            version_root.relative_to(shared_run_root),
        )
    except FileExistsError:
        verified_v1_root = panderm_run.ensure_tree(
            shared_run_root,
            version_root.relative_to(shared_run_root),
        )
    if not verified_v1_root.is_dir() or verified_v1_root.is_symlink():
        raise ValueError(
            f"validation version parent is not a real directory: {verified_v1_root}"
        )
    resolved_v1_root = verified_v1_root.resolve(strict=True)
    if (
        not resolved_v1_root.is_relative_to(resolved_root)
        or not resolved_v1_root.parent.samefile(resolved_root)
    ):
        raise ValueError(
            f"validation version parent escaped the shared root: {resolved_v1_root}"
        )
    lock_path = panderm_run.validation_run_lock_path(shared_run_root)
    if not lock_path.parent.samefile(verified_v1_root):
        raise ValueError("validation lock parent does not match the version parent")
    return lock_path


FRESH_ROOT_RACE_WORKER = r"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from ddpm_derm import panderm_run

shared_run_root = Path(sys.argv[2])
session_id = sys.argv[3]
start_at = float(sys.argv[4])
calls = {"archive": 0, "attempt": 0, "runner": 0}
while time.time() < start_at:
    pass
resolved_root = panderm_run.require_existing_shared_root(shared_run_root)
version_root = shared_run_root / panderm_run.RUN_VERSION
if version_root.is_symlink():
    raise ValueError("validation version parent must not be a symlink")
try:
    verified_v1_root = panderm_run.ensure_tree(
        shared_run_root, version_root.relative_to(shared_run_root)
    )
except FileExistsError:
    verified_v1_root = panderm_run.ensure_tree(
        shared_run_root, version_root.relative_to(shared_run_root)
    )
resolved_v1_root = verified_v1_root.resolve(strict=True)
if (
    verified_v1_root.is_symlink()
    or not resolved_v1_root.is_relative_to(resolved_root)
    or not resolved_v1_root.parent.samefile(resolved_root)
):
    raise ValueError("validation version parent escaped the shared root")
lock_path = panderm_run.validation_run_lock_path(shared_run_root)
if not lock_path.parent.samefile(verified_v1_root):
    raise ValueError("validation lock parent mismatch")
try:
    marker = panderm_run.acquire_validation_run_lock(
        lock_path,
        session_id=session_id,
        run_version=panderm_run.RUN_VERSION,
        git_commit="c" * 40,
        shared_root_uuid="765b971f-d148-4960-a77d-b73f28fc013c",
        account_label="A",
    )
    calls = {"archive": 1, "attempt": 1, "runner": 1}
    print(json.dumps({
        "acquired": True,
        "session_id": session_id,
        "owner": marker["session_id"],
        "parent_ready": True,
        "calls": calls,
    }))
except FileExistsError as error:
    print(json.dumps({
        "acquired": False,
        "session_id": session_id,
        "parent_ready": verified_v1_root.is_dir(),
        "calls": calls,
        "error": str(error),
    }))
"""


class NotebookHygieneTests(unittest.TestCase):
    def test_notebooks_are_unexecuted_and_every_code_cell_compiles(self):
        for name in NAMES:
            notebook, _ = load(name)
            with self.subTest(name=name):
                self.assertGreater(len(notebook["cells"]), 0)
                for index, cell in enumerate(notebook["cells"]):
                    self.assertFalse(cell.get("outputs"), f"{name} cell {index}")
                    if cell["cell_type"] == "code":
                        self.assertIsNone(
                            cell.get("execution_count"), f"{name} cell {index}"
                        )
                        compile(
                            "".join(cell["source"]),
                            f"{name}:cell-{index}",
                            "exec",
                        )

    def test_no_secrets_emails_or_tool_attribution(self):
        for name in NAMES:
            notebook, code = load(name)
            raw = json.dumps(notebook)
            with self.subTest(name=name):
                self.assertNotRegex(
                    code,
                    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                )
                self.assertNotRegex(raw, r"gh[pousr]_[A-Za-z0-9]")
                for forbidden in (
                    "ghp_",
                    "github_pat_",
                    "@github.com",
                    "DRIVE_FOLDER_ID =",
                ):
                    self.assertNotIn(forbidden, code)
                self.assertNotRegex(
                    raw,
                    r"(?i)(co-authored-by|generated (with|by)\s"
                    r"|assistant-(generated|assisted)"
                    r"|ai[- ](generated|assisted|written))",
                )


class Phase2ManifestBindingTests(unittest.TestCase):
    @staticmethod
    def _sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def _approved_manifest_root():
        data_root = Path(os.environ.get("DDPM_DERM_DATA_DIR", ROOT / "data"))
        return data_root / "manifests"

    def _copy_approved_source_manifests(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True)
        expected = {
            "train.csv": PHASE2_TRAIN_SHA256,
            "val.csv": PHASE2_VAL_SHA256,
            "class_to_idx.json": PHASE2_MAPPING_SHA256,
        }
        source_root = self._approved_manifest_root()
        for name, expected_sha256 in expected.items():
            source = source_root / name
            self.assertTrue(source.is_file(), source)
            self.assertEqual(self._sha256(source), expected_sha256)
            shutil.copyfile(source, destination / name)
        return destination

    def _prepare(self, source_root, target_root):
        prepare, verify = load_phase2_manifest_helpers()
        source_manifest_root = self._copy_approved_source_manifests(
            Path(source_root) / "manifests"
        )
        prepare(source_manifest_root, target_root)
        return verify

    def _make_directory_link(self, link, target):
        link, target = Path(link), Path(target)
        if os.name == "nt":
            result = subprocess.run(
                ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
        else:
            link.symlink_to(target, target_is_directory=True)

    @staticmethod
    def _remove_directory_link(link):
        link = Path(link)
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            link.rmdir()

    def test_manifest_only_root_has_exact_verified_files_and_no_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "manifest-only"
            verify = self._prepare(root / "shared", data_root)
            report = verify(data_root)
            self.assertEqual(
                report["files"],
                [
                    "manifests/class_to_idx.json",
                    "manifests/train.csv",
                    "manifests/val.csv",
                ],
            )
            self.assertEqual(report["train_rows"], 6995)
            self.assertEqual(report["val_rows"], 1510)
            self.assertFalse((data_root / "manifests" / "test.csv").exists())
            self.assertFalse(any(path.is_symlink() for path in data_root.rglob("*")))
            self.assertFalse(
                any(path.suffix.lower() == ".jpg" for path in data_root.rglob("*"))
            )

    def test_common_wrong_source_and_recomputed_expected_is_rejected(self):
        prepare, _ = load_phase2_manifest_helpers()
        self.assertEqual(prepare.__code__.co_argcount, 2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = self._copy_approved_source_manifests(
                root / "shared" / "manifests"
            )
            train_path = source_root / "train.csv"
            payload = train_path.read_bytes()
            first_row = payload.find(b"\n") + 1
            first_comma = payload.find(b",", first_row)
            self.assertGreater(first_row, 0)
            self.assertGreater(first_comma, first_row)
            train_path.write_bytes(
                payload[:first_comma] + b"-wrong" + payload[first_comma:]
            )
            recomputed_wrong_sha256 = self._sha256(train_path)
            self.assertNotEqual(recomputed_wrong_sha256, PHASE2_TRAIN_SHA256)
            self.assertEqual(
                train_path.read_bytes().count(b"\n"), payload.count(b"\n")
            )
            target_root = root / "manifest-only"
            with self.assertRaisesRegex(
                AssertionError, "authoritative train.csv SHA-256 drift"
            ):
                prepare(source_root, target_root)
            self.assertFalse(target_root.exists())

    def test_source_directory_junction_or_symlink_is_rejected(self):
        prepare, _ = load_phase2_manifest_helpers()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_source = self._copy_approved_source_manifests(
                root / "real-manifests"
            )
            source_link = root / "source-link"
            self._make_directory_link(source_link, real_source)
            try:
                with self.assertRaisesRegex(AssertionError, "reparse point"):
                    prepare(source_link, root / "manifest-only")
            finally:
                self._remove_directory_link(source_link)

    def test_destination_parent_junction_or_symlink_is_rejected(self):
        prepare, _ = load_phase2_manifest_helpers()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = self._copy_approved_source_manifests(
                root / "shared" / "manifests"
            )
            real_parent = root / "real-parent"
            real_parent.mkdir()
            destination_parent_link = root / "destination-parent-link"
            self._make_directory_link(destination_parent_link, real_parent)
            try:
                with self.assertRaisesRegex(AssertionError, "reparse point"):
                    prepare(
                        source_root,
                        destination_parent_link / "manifest-only",
                    )
            finally:
                self._remove_directory_link(destination_parent_link)
            self.assertFalse((real_parent / "manifest-only").exists())

    def test_manifest_gate_rejects_test_extra_and_hash_drift(self):
        mutations = (
            ("test manifest", "manifests/test.csv", b"forbidden"),
            ("extra file", "extra.txt", b"forbidden"),
            ("train hash", "manifests/train.csv", b"\ndrift"),
            ("val hash", "manifests/val.csv", b"\ndrift"),
            ("mapping hash", "manifests/class_to_idx.json", b"\n"),
        )
        for label, relative_path, payload in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data_root = root / "manifest-only"
                verify = self._prepare(root / "shared", data_root)
                path = data_root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("ab") as handle:
                    handle.write(payload)
                with self.assertRaises(AssertionError):
                    verify(data_root)

    def test_missing_class_mapping_fails_loud_on_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "checkout"
            shutil.copytree(ROOT / "src", checkout / "src")
            shutil.copytree(ROOT / "tests", checkout / "tests")
            data_root = root / "manifest-only"
            manifests_root = data_root / "manifests"
            manifests_root.mkdir(parents=True)
            for split in ("train", "val"):
                (manifests_root / f"{split}.csv").write_text(
                    "image_path,label_idx,dx,lesion_id,image_id\n",
                    encoding="utf-8",
                )
            env = os.environ.copy()
            env["DDPM_DERM_DATA_DIR"] = str(data_root)
            env["DDPM_DERM_OUTPUTS_DIR"] = str(root / "outputs")
            env["PYTHONPATH"] = str(checkout / "src")
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    "-c",
                    "import tests.test_panderm_blockers",
                ],
                cwd=checkout,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("Could not locate the HAM10000 data directory", result.stdout)

    def test_fresh_checkout_imports_and_runs_c1_data_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "checkout"
            shutil.copytree(ROOT / "src", checkout / "src")
            shutil.copytree(ROOT / "tests", checkout / "tests")
            shutil.copytree(ROOT / "notebooks", checkout / "notebooks")
            data_root = root / "manifest-only"
            self._prepare(root / "shared", data_root)
            env = os.environ.copy()
            env["DDPM_DERM_DATA_DIR"] = str(data_root)
            env["DDPM_DERM_OUTPUTS_DIR"] = str(root / "outputs")
            env["PYTHONPATH"] = str(checkout / "src")
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            imports = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    "-c",
                    (
                        "import tests.test_panderm_blockers; "
                        "import tests.test_panderm_base_c1_finetune; "
                        "import tests.test_panderm_notebooks; "
                        "print('MANIFEST_ONLY_IMPORTS_OK')"
                    ),
                ],
                cwd=checkout,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(imports.returncode, 0, imports.stdout)
            self.assertIn("MANIFEST_ONLY_IMPORTS_OK", imports.stdout)
            c1 = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    "-m",
                    "unittest",
                    "-v",
                    "tests.test_panderm_base_c1_finetune.C1DataTests",
                ],
                cwd=checkout,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(c1.returncode, 0, c1.stdout)
            self.assertIn("Ran 4 tests", c1.stdout)
            self.assertFalse((checkout / "data").exists())
            self.assertFalse((checkout.parent / "data").exists())

    def test_illegal_cli_matrix_fails_for_each_expected_reason(self):
        base = [
            "--checkpoint",
            "weights.pth",
            "--upstream-dir",
            "upstream",
            "--output-dir",
            "outputs",
        ]
        cases = (
            (["--variant", "C4"], "invalid choice"),
            (
                ["--generated-manifest", "/tmp/nope.csv"],
                "synthetic manifests are rejected",
            ),
            (
                ["--run-version", "v2_other"],
                "--run-version must be v1_panderm_base_c1_finetune",
            ),
            (["--df-target-count", "586"], "--df-target-count must be 585"),
            (
                ["--accumulation-steps", "0"],
                "--accumulation-steps must be exactly 8",
            ),
            (["--batch-size", "0"], "--batch-size must be exactly 16"),
            (["--warmup-epochs", "999"], "--warmup-epochs must be exactly 5"),
            (["--evaluation-scope", "full"], "invalid choice"),
            (["--drop-path", "0.3"], "unrecognized arguments"),
            (["--no-amp"], "unrecognized arguments"),
            (["--seed", "1"], "--seed must be exactly 0"),
            (["--epochs", "50"], "--epochs must be exactly 5"),
        )
        for extra, expected_error in cases:
            with self.subTest(extra=extra):
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit):
                    train_panderm.parse_args(base + extra)
                self.assertIn(expected_error, stderr.getvalue())

    def test_manifest_only_root_is_removed_after_test_failure(self):
        temporary_path = None
        with self.assertRaisesRegex(RuntimeError, "forced Phase 2 failure"):
            with tempfile.TemporaryDirectory(
                prefix="panderm-phase2-failure-"
            ) as temporary:
                temporary_path = Path(temporary)
                self._prepare(
                    temporary_path / "shared",
                    temporary_path / "manifest-only",
                )
                raise RuntimeError("forced Phase 2 failure")
        self.assertIsNotNone(temporary_path)
        self.assertFalse(temporary_path.exists())


class ValidationNotebookTests(unittest.TestCase):
    def test_first_cell_is_pinned_to_the_implementation_commit(self):
        notebook, _ = load(VALIDATION)
        first = "".join(notebook["cells"][0]["source"])
        self.assertEqual(notebook["cells"][0]["cell_type"], "code")
        self.assertIn(
            f'EXPECTED_GIT_COMMIT = "{PINNED_IMPLEMENTATION_COMMIT}"',
            first,
        )
        self.assertNotIn(f'EXPECTED_GIT_COMMIT = "{PIN_PLACEHOLDER}"', first)
        self.assertIn(f'EXPECTED_GIT_COMMIT != "{PIN_PLACEHOLDER}"', first)
        self.assertIn("len(EXPECTED_GIT_COMMIT) == 40", first)
        self.assertIn("Pin the reviewed pushed commit", first)

    def test_pinned_first_cell_passes_its_own_guard(self):
        notebook, _ = load(VALIDATION)
        first = "".join(notebook["cells"][0]["source"])
        namespace = {}
        exec(compile(first, "cell-0", "exec"), namespace)
        self.assertEqual(
            namespace["EXPECTED_GIT_COMMIT"],
            PINNED_IMPLEMENTATION_COMMIT,
        )

    def test_archive_expected_identity_comes_from_the_reviewed_constant(self):
        """The notebook must not carry its own copy of the approved digest."""
        _, code = load(VALIDATION)
        self.assertIn(
            "panderm_run.require_approved_content_identity(", code
        )
        self.assertIn(
            "panderm_run.EXPECTED_VALIDATION_CONTENT_IDENTITY_SHA256", code
        )
        for call in (
            "panderm_run.build_validation_archive_cache",
            "panderm_run.reuse_validation_archive_cache",
            "panderm_run.validate_validation_archive_cache",
        ):
            with self.subTest(call=call):
                self.assertIn(call, code)
        self.assertEqual(
            code.count("expected_file_content_identity_sha256=APPROVED_CONTENT_IDENTITY"),
            3,
        )
        # Archive content identity remains centralized in production code, while
        # Phase 2 intentionally binds its three approved manifest byte identities.
        digest_literals = {
            node.value
            for node in ast.walk(ast.parse(code))
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value) == 64
            and all(character in "0123456789abcdef" for character in node.value)
        }
        self.assertEqual(
            digest_literals,
            {
                PHASE2_TRAIN_SHA256,
                PHASE2_VAL_SHA256,
                PHASE2_MAPPING_SHA256,
            },
        )

    def test_validation_run_lock_follows_all_preflights_and_precedes_durable_work(self):
        """Cheap checks and GPU cleanup must finish before the durable lock."""
        notebook, code = load(VALIDATION)
        positions = validate_validation_notebook_order(code)
        self.assertLess(
            positions["GPU smoke assertions complete"],
            positions["GPU model and tensor cleanup"],
        )
        self.assertLess(
            positions["GPU cache cleanup"],
            positions["all preflights complete"],
        )
        self.assertLess(
            positions["all preflights complete"],
            positions["version parent setup"],
        )
        self.assertLess(
            positions["version parent verification"],
            positions["validation lock path"],
        )
        self.assertLess(
            positions["validation lock parent verification"],
            positions["validation lock acquisition"],
        )
        self.assertIn("panderm_run.validation_run_lock_path(SHARED_RUN_ROOT)", code)
        self.assertIn("VALIDATION_LOCK_HELD", code)
        self.assertNotIn("V1_ROOT.mkdir", code)
        sources = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        phase_zero = next(text for text in sources if 'drive.mount("/content/drive")' in text)
        gpu_smoke = next(text for text in sources if "GPU_SMOKE_COMPLETE = True" in text)
        for forbidden in (
            "validation_run_lock_path",
            "acquire_validation_run_lock",
            "VALIDATION_SESSION",
        ):
            with self.subTest(phase_zero_forbidden=forbidden):
                self.assertNotIn(forbidden, phase_zero)
        self.assertNotIn(
            "V1_ROOT.relative_to(SHARED_RUN_ROOT)",
            phase_zero,
        )
        self.assertLess(
            gpu_smoke.index("PHASE3_COMPLETE = True"),
            gpu_smoke.index("verified_v1_root = panderm_run.ensure_tree"),
        )
        self.assertIn("acquire_validation_run_lock", gpu_smoke)

    def test_validation_order_validator_rejects_the_old_early_lock_order(self):
        ordered_tokens = [token for _, token in VALIDATION_ORDER_TOKENS]
        correct = "\n".join(ordered_tokens)
        validate_validation_notebook_order(correct)

        early_lock = ordered_tokens.copy()
        lock_path = early_lock.pop(
            [label for label, _ in VALIDATION_ORDER_TOKENS].index(
                "validation lock path"
            )
        )
        lock_acquire = early_lock.pop(
            [token for _, token in VALIDATION_ORDER_TOKENS].index(
                "panderm_run.acquire_validation_run_lock"
            )
        )
        early_lock[0:0] = [lock_path, lock_acquire]
        with self.assertRaisesRegex(
            AssertionError,
            "validation ordering violation",
        ):
            validate_validation_notebook_order("\n".join(early_lock))

    def test_validator_rejects_version_parent_after_lock_or_before_gpu(self):
        labels = [label for label, _ in VALIDATION_ORDER_TOKENS]
        ordered_tokens = [token for _, token in VALIDATION_ORDER_TOKENS]
        setup_tokens = [
            ordered_tokens[labels.index("version parent setup")],
            ordered_tokens[labels.index("version parent verification")],
        ]

        parent_after_lock = [
            token for token in ordered_tokens if token not in setup_tokens
        ]
        acquire_index = parent_after_lock.index(
            "panderm_run.acquire_validation_run_lock"
        )
        parent_after_lock[acquire_index + 1:acquire_index + 1] = setup_tokens
        with self.assertRaisesRegex(AssertionError, "validation ordering violation"):
            validate_validation_notebook_order("\n".join(parent_after_lock))

        parent_before_gpu = [
            token for token in ordered_tokens if token not in setup_tokens
        ]
        gpu_index = parent_before_gpu.index('print("[Phase 4] START')
        parent_before_gpu[gpu_index:gpu_index] = setup_tokens
        with self.assertRaisesRegex(AssertionError, "validation ordering violation"):
            validate_validation_notebook_order("\n".join(parent_before_gpu))

    def test_fresh_version_parent_setup_then_lock_acquire_and_owner_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            shared_run_root = Path(temporary)
            version_root = shared_run_root / panderm_run.RUN_VERSION
            self.assertFalse(version_root.exists())

            lock_path = prepare_fresh_validation_lock(shared_run_root)
            self.assertTrue(version_root.is_dir())
            owner = str(uuid.uuid4())
            marker = panderm_run.acquire_validation_run_lock(
                lock_path,
                session_id=owner,
                run_version=panderm_run.RUN_VERSION,
                git_commit="c" * 40,
                shared_root_uuid="765b971f-d148-4960-a77d-b73f28fc013c",
                account_label="A",
            )
            self.assertEqual(marker["session_id"], owner)
            released = panderm_run.release_validation_run_lock(
                lock_path, session_id=owner
            )
            self.assertEqual(released["session_id"], owner)
            self.assertFalse(lock_path.exists())

    def test_old_fresh_root_order_fails_before_any_durable_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            shared_run_root = Path(temporary)
            lock_path = panderm_run.validation_run_lock_path(shared_run_root)
            calls = {"archive": 0, "attempt": 0, "runner": 0}
            with self.assertRaisesRegex(FileNotFoundError, "directory is missing"):
                panderm_run.acquire_validation_run_lock(
                    lock_path,
                    session_id=str(uuid.uuid4()),
                    run_version=panderm_run.RUN_VERSION,
                    git_commit="c" * 40,
                    shared_root_uuid="765b971f-d148-4960-a77d-b73f28fc013c",
                    account_label="A",
                )
            self.assertEqual(calls, {"archive": 0, "attempt": 0, "runner": 0})
            self.assertFalse(lock_path.parent.exists())

    def test_version_parent_setup_failures_do_not_acquire_the_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            missing_root = base / "missing"
            with self.assertRaisesRegex(FileNotFoundError, "shared run root is missing"):
                prepare_fresh_validation_lock(missing_root)
            self.assertFalse(missing_root.exists())

            shared_run_root = base / "shared"
            shared_run_root.mkdir()
            version_root = shared_run_root / panderm_run.RUN_VERSION
            version_root.write_text("not a directory\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_fresh_validation_lock(shared_run_root)
            self.assertFalse(
                panderm_run.validation_run_lock_path(shared_run_root).exists()
            )

    def test_version_parent_rejects_path_escape_and_preserves_legal_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            shared_run_root = base / "shared"
            outside = base / "outside"
            shared_run_root.mkdir()
            outside.mkdir()
            version_root = shared_run_root / panderm_run.RUN_VERSION
            try:
                version_root.symlink_to(outside, target_is_directory=True)
            except OSError:
                result = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(version_root), str(outside)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with self.assertRaisesRegex(ValueError, "symlink|escaped"):
                prepare_fresh_validation_lock(shared_run_root)
            self.assertFalse(
                panderm_run.validation_run_lock_path(shared_run_root).exists()
            )

        with tempfile.TemporaryDirectory() as temporary:
            shared_run_root = Path(temporary)
            version_root = shared_run_root / panderm_run.RUN_VERSION
            version_root.mkdir()
            preserved = version_root / "reviewer.txt"
            preserved.write_text("keep\n", encoding="utf-8")
            prepare_fresh_validation_lock(shared_run_root)
            self.assertEqual(preserved.read_text(encoding="utf-8"), "keep\n")

    def test_two_fresh_root_processes_create_parent_then_one_wins_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            shared_run_root = base / "shared"
            shared_run_root.mkdir()
            worker = base / "worker.py"
            worker.write_text(FRESH_ROOT_RACE_WORKER, encoding="utf-8")
            src = str((ROOT / "src").resolve())
            start_at = time.time() + 1.5
            sessions = [str(uuid.uuid4()), str(uuid.uuid4())]
            processes = [
                subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        "-u",
                        str(worker),
                        src,
                        str(shared_run_root),
                        session,
                        str(start_at),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for session in sessions
            ]
            outputs = [process.communicate()[0] for process in processes]
            results = []
            for output in outputs:
                rows = [
                    json.loads(line)
                    for line in output.splitlines()
                    if line.startswith("{")
                ]
                self.assertEqual(len(rows), 1, output)
                results.append(rows[0])
            self.assertTrue(all(result["parent_ready"] for result in results))
            winners = [result for result in results if result["acquired"]]
            losers = [result for result in results if not result["acquired"]]
            self.assertEqual(len(winners), 1, results)
            self.assertEqual(len(losers), 1, results)
            self.assertEqual(
                losers[0]["calls"], {"archive": 0, "attempt": 0, "runner": 0}
            )
            self.assertEqual(
                winners[0]["calls"], {"archive": 1, "attempt": 1, "runner": 1}
            )
            lock_path = panderm_run.validation_run_lock_path(shared_run_root)
            self.assertEqual(
                panderm_run._read_validation_run_lock(lock_path)["session_id"],
                winners[0]["session_id"],
            )
            panderm_run.release_validation_run_lock(
                lock_path, session_id=winners[0]["session_id"]
            )

    def test_run_all_never_clears_a_stale_lock_and_releases_only_its_own(self):
        _, code = load(VALIDATION)
        self.assertNotIn("clear_stale_validation_run_lock", code)
        self.assertNotIn("clear_stale_marker", code)
        self.assertIn(
            "panderm_run.release_validation_run_lock(VALIDATION_RUN_LOCK, "
            "session_id=VALIDATION_SESSION_ID)",
            code,
        )
        # Released on the success path and on the caught-failure path only.
        self.assertEqual(code.count("release_validation_run_lock"), 2)
        self.assertNotIn("VALIDATION_RUN_LOCK.unlink", code)

    def test_account_label_is_operational_metadata_only(self):
        notebook, code = load(VALIDATION)
        self.assertIn('ACCOUNT_LABEL = "A"', code)
        self.assertIn('ACCOUNT_LABEL in {"A", "B", "C"}', code)
        identity_block = code[code.index("run_identity") if "run_identity" in code else 0:]
        self.assertNotIn('"account_label": ACCOUNT_LABEL', identity_block)
        self.assertNotIn("ACCOUNT_LABEL", json.dumps(
            panderm_run.IMMUTABLE_IDENTITY_KEYS
        ))

    def test_validation_is_seed_zero_five_epoch_validation_only(self):
        _, code = load(VALIDATION)
        for required in (
            "VALIDATION_EPOCHS = 5",
            '"--seed", "0"',
            '"--epochs", str(VALIDATION_EPOCHS)',
            '"--evaluation-scope", "validation_only"',
            "BATCH_SIZE = 16",
            "ACCUMULATION_STEPS = 8",
            "DROP_PATH = 0.2",
            "purpose=panderm_run.VALIDATION_ONLY",
        ):
            self.assertIn(required, code)
        self.assertNotIn('"--drop-path", str(DROP_PATH)', code)

    def test_validation_never_opens_or_hashes_test_manifest(self):
        _, code = load(VALIDATION)
        self.assertNotIn('manifests.load_split("test")', code)
        self.assertNotIn('for split in ("train", "val", "test")', code)
        self.assertIn('for split in ("train", "val")', code)
        self.assertNotIn("test_manifest_rows_read", code)
        self.assertNotIn("shutil.copytree", code)
        self.assertIn("panderm_run.build_validation_archive_cache", code)
        self.assertIn("panderm_run.reuse_validation_archive_cache", code)
        tree = ast.parse(code)
        legacy_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "stage_validation_data"
        ]
        self.assertEqual(legacy_calls, [])
        panderm_run.validate_validation_notebook_source(code)
        self.assertIn('result["test_metrics"] is None', code)
        self.assertIn('"test": None', code)

    def test_test_access_regressions_are_program_path_checks(self):
        _, code = load(VALIDATION)
        for required in (
            "trainer_source",
            "formal_code",
            "source_guards",
            "test_access_probes",
            '"before_validation_match"',
            '"before_three_seed_completion"',
            '"forged_validation_pass"',
            "purpose=panderm_run.TEST_ACCESS",
            "all(source_guards.values())",
            "all(test_access_probes.values())",
        ):
            self.assertIn(required, code)

    def test_validation_reopens_artifacts_safely_and_checks_current_identity(self):
        _, code = load(VALIDATION)
        for required in (
            "train_panderm.load_checkpoint_safe",
            "load_completed_checkpoint_pair_safe",
            'result["checkpoint_integrity"]',
            "require_completed_artifact_identities",
            "require_matching_identity",
            "require_expected_identity_complete",
            "a rejected resume must not mutate the checkpoint",
        ):
            self.assertIn(required, code)
        self.assertNotIn("weights_only=False", code)

    def test_validation_runs_targeted_full_and_runner_smoke_checks(self):
        _, code = load(VALIDATION)
        for required in (
            "tests.test_panderm_blockers",
            "tests.test_panderm_base_c1_finetune",
            "tests.test_panderm_notebooks",
            "tests.test_panderm_fresh_runtime",
            '"unittest", "discover", "-s", "tests"',
            "tests.test_panderm_blockers.PanDermRunnerMockSmokeTests",
            '"runner_smoke_ok": "OK" in runner_smoke_output',
            '["--evaluation-scope", "full"]',
            '["--drop-path", "0.3"]',
            '["--no-amp"]',
        ):
            self.assertIn(required, code)
        self.assertNotIn("scripts/smoke_test.py", code)

    def test_phase2_uses_verified_temporary_manifest_and_output_roots(self):
        notebook, code = load(VALIDATION)
        phase2 = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if "prepare_test_manifest_data_root" in "".join(cell.get("source", []))
        )
        for required in (
            'tempfile.TemporaryDirectory(dir="/content", prefix="panderm-test-manifests-")',
            'SHARED_PROJECT_DIR / "data" / "manifests"',
            'test_env["DDPM_DERM_DATA_DIR"] = str(TEST_MANIFEST_DATA_ROOT)',
            'test_env["DDPM_DERM_OUTPUTS_DIR"] = str(TEST_OUTPUT_ROOT)',
            'test_env["PYTHONPATH"] = str(CODE_DIR / "src")',
            'test_env["PYTHONUNBUFFERED"] = "1"',
            'test_env["PYTHONDONTWRITEBYTECODE"] = "1"',
            'TEST_OUTPUT_ROOT / "illegal"',
            "for extra, expected_error in illegal:",
            "assert expected_error in illegal_output",
            "assert not LOCAL_DATA_DIR.exists()",
            "assert not list(Path(\"/content\").glob(\"panderm-test-manifests-*\"))",
        ):
            with self.subTest(required=required):
                self.assertIn(required, phase2)
        self.assertNotIn('test_env.pop("DDPM_DERM_DATA_DIR", None)', phase2)
        self.assertNotIn('"--output-dir", "/content/panderm-illegal"', phase2)
        self.assertNotIn("scripts/smoke_test.py", code)

    def test_phase2_manifest_gate_is_exact_train_val_only(self):
        _, code = load(VALIDATION)
        phase2 = next(
            "".join(cell["source"])
            for cell in load(VALIDATION)[0]["cells"]
            if "prepare_test_manifest_data_root" in "".join(cell.get("source", []))
        )
        for required in (
            PHASE2_TRAIN_SHA256,
            PHASE2_VAL_SHA256,
            PHASE2_MAPPING_SHA256,
            'source_manifest_root / "train.csv"',
            'source_manifest_root / "val.csv"',
            'source_manifest_root / "class_to_idx.json"',
            '"manifests/train.csv"',
            '"manifests/val.csv"',
            '"manifests/class_to_idx.json"',
            "actual_entries == expected_entries",
            'len(train_rows) == 6995',
            'len(val_rows) == 1510',
            "class_mapping == PHASE2_EXPECTED_CLASS_TO_IDX",
            'not (manifests_root / "test.csv").exists()',
            '"source manifest root"',
            '"temporary data root parent"',
            '"temporary test data root"',
            '"temporary manifests directory"',
            '"st_file_attributes"',
            '"FILE_ATTRIBUTE_REPARSE_POINT"',
        ):
            with self.subTest(required=required):
                self.assertIn(required, phase2)
        prepare, verify = load_phase2_manifest_helpers()
        self.assertEqual(prepare.__code__.co_argcount, 2)
        self.assertEqual(verify.__code__.co_argcount, 1)
        self.assertNotIn("expected_manifest_sha256", phase2)
        self.assertNotIn("expected_class_mapping_sha256", phase2)
        self.assertNotIn("expected_class_to_idx", phase2)
        self.assertNotIn("copytree(source_manifest_root", code)
        self.assertNotIn("copytree(SHARED_PROJECT_DIR", code)

    def test_fresh_runtime_probe_still_removes_parent_data_binding(self):
        source = (ROOT / "tests" / "test_panderm_fresh_runtime.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('env.pop("DDPM_DERM_DATA_DIR", None)', source)
        self.assertIn('assert "DDPM_DERM_DATA_DIR" not in os.environ', source)

    def test_preflight_and_archive_staging_enforce_runtime_binding(self):
        notebook, code = load(VALIDATION)
        sources = ["".join(cell["source"]) for cell in notebook["cells"]]
        serialization = next(
            text for text in sources
            if "checkpoint_preflight = train_panderm.checkpoint_serialization_preflight" in text
        )
        staging = next(
            text for text in sources
            if "panderm_run.reuse_validation_archive_cache" in text
        )
        self.assertIn("assert not LOCAL_DATA_DIR.exists()", serialization)
        self.assertIn('temporary_directory=Path("/content")', serialization)
        self.assertIn("stage_after_validation_preflights", staging)
        self.assertIn("training_env[\"DDPM_DERM_DATA_DIR\"] = str(LOCAL_DATA_DIR)", staging)
        for required in (
            "build_validation_archive_cache",
            "reuse_validation_archive_cache",
            'expected_manifest_sha256=manifest_sha256',
            'expected_class_mapping_sha256=class_mapping_sha256',
            '"train.csv", "val.csv", "class_to_idx.json"',
            '"test.csv").exists()',
        ):
            self.assertIn(required, staging)
        self.assertNotIn("importlib.reload", code)
        self.assertIn(
            'os.environ["DDPM_DERM_DATA_DIR"] = str(SHARED_PROJECT_DIR / "data")',
            code,
        )

    def test_python_subprocesses_are_unbuffered_cache_free_and_heartbeat_visible(self):
        _, code = load(VALIDATION)
        self.assertIn('env["PYTHONDONTWRITEBYTECODE"] = "1"', code)
        self.assertIn('output_queue.get(timeout=60)', code)
        self.assertIn('"[subprocess] heartbeat elapsed=', code)
        self.assertIn('alive={process.poll() is None}', code)
        for required in (
            '[sys.executable, "-B", "-u", "-m", "unittest"',
            '[sys.executable, "-B", "-u", "-m", "ddpm_derm.train_panderm"',
            '[sys.executable, "-B", "-u", "-c", child_code]',
        ):
            self.assertIn(required, code)

    def test_local_tests_package_prevents_colab_package_shadowing(self):
        self.assertTrue((ROOT / "tests" / "__init__.py").is_file())

    def test_checkpoint_and_gpu_preflights_run_before_full_staging(self):
        notebook, _ = load(VALIDATION)
        sources = ["".join(cell["source"]) for cell in notebook["cells"]]
        sha_gate = next(
            index
            for index, text in enumerate(sources)
            if "panderm_run.require_checkpoint_sha256(" in text
        )
        serialization = next(
            index
            for index, text in enumerate(sources)
            if "checkpoint_preflight = train_panderm.checkpoint_serialization_preflight" in text
        )
        gpu_smoke = next(
            index for index, text in enumerate(sources)
            if "stage_train_smoke_sample" in text
        )
        staging = next(
            index
            for index, text in enumerate(sources)
            if "panderm_run.build_validation_archive_cache" in text
        )
        self.assertLess(sha_gate, serialization)
        self.assertLess(serialization, gpu_smoke)
        self.assertLess(gpu_smoke, staging)

    def test_checkpoint_preflight_reopens_remaps_loads_and_round_trips(self):
        _, code = load(VALIDATION)
        for required in (
            "panderm.load_pretrained_state(CHECKPOINT_PATH)",
            "panderm.detect_checkpoint_layout(pretrained_state)",
            "panderm.remap_pretrained_state_dict(pretrained_state, layout=pretrained_layout)",
            "panderm.build_panderm_classifier(checkpoint_path=CHECKPOINT_PATH",
            "pretrained_layout == panderm.LAYOUT_DIRECT_BACKBONE",
            'assert "fc_norm.weight" in remapped_state',
            "checkpoint_serialization_preflight",
            'type(dependency_versions["torch"]) is str',
            "panderm_run.require_primitive_identity(production_run_identity)",
            "preflight_num_batches == 469 and preflight_steps_per_epoch == 58",
            'checkpoint_preflight["weights_only_round_trip"] is True',
            "gc.collect()",
        ):
            with self.subTest(required=required):
                self.assertIn(required, code)
    def test_archive_build_and_reuse_keep_live_progress(self):
        _, code = load(VALIDATION)
        self.assertIn("panderm_run.build_validation_archive_cache", code)
        self.assertIn("panderm_run.reuse_validation_archive_cache", code)
        self.assertIn('staging_report["images_staged"] == 8505', code)
        staging = (
            ROOT / "src" / "ddpm_derm" / "panderm_run.py"
        ).read_text(encoding="utf-8")
        for marker in (
            "[archive-build] START",
            "files={index}/{total}",
            "current_file={member_name}",
            'phase="archive-runtime-copy"',
            "source_bytes={source_bytes_completed}",
            "tar_stream_bytes={archive.fileobj.tell()}",
            "bytes={copied}/{total}",
            "eta={eta:.1f}s",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, staging)

    def test_phase_four_gpu_smoke_has_full_update_contract(self):
        _, code = load(VALIDATION)
        for marker in (
            "stage_train_smoke_sample",
            "train_panderm.set_seed(0)",
            "sample_tensors = [train_transform(",
            "[Phase 4] accumulation micro-step=",
            'gradient_report["blocks_with_gradient"] == list(range(12))',
            'gradient_report["head_parameters_have_gradient"]',
            "model.pos_embed.grad is None",
            "verify_optimizer_covers_parameters_once",
            "scaler.step(optimizer)",
            "changed_backbone > 0",
            "post_step_checkpoint_preflight",
            '"smoke_model_discarded": True',
            "torch.cuda.empty_cache()",
            "[Phase 4] COMPLETE elapsed=",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, code)
        phase_four = next(
            text
            for text in (
                "".join(cell["source"]) for cell in load(VALIDATION)[0]["cells"]
            )
            if "stage_train_smoke_sample" in text
        )
        for line in phase_four.splitlines():
            stripped = line.strip()
            if stripped.startswith("print(") and "[Phase 4]" in stripped:
                with self.subTest(line=stripped[:60]):
                    self.assertIn("flush=True", stripped)
        self.assertIn(
            'post_step_checkpoint_preflight["weights_only_round_trip"] is True',
            code,
        )
        self.assertIn(
            'require_matching_identity(result["run_identity"], '
            "production_run_identity)",
            code,
        )

    def test_smoke_state_is_discarded_before_fresh_non_resume_subprocess(self):
        notebook, _ = load(VALIDATION)
        sources = ["".join(cell["source"]) for cell in notebook["cells"]]
        smoke_index = next(
            index for index, text in enumerate(sources)
            if "stage_train_smoke_sample" in text
        )
        run_index = next(
            index for index, text in enumerate(sources)
            if "gate_seconds, gate_output = run_stream(gate_command" in text
        )
        smoke = sources[smoke_index]
        run = sources[run_index]
        self.assertLess(smoke_index, run_index)
        self.assertIn("del model, optimizer, schedule, scaler", smoke)
        self.assertIn('"smoke_model_discarded": True', smoke)
        self.assertIn("[start] fresh run (no --resume) from epoch 1", run)
        self.assertNotIn(
            'if (checkpoint_dir / "last.pt").is_file(): '
            'gate_command.append("--resume")',
            run,
        )
        initial_call = run.split(
            "gate_seconds, gate_output = run_stream(gate_command", 1
        )[1].splitlines()[0]
        self.assertNotIn("--resume", initial_call)

    def test_notebook_is_account_neutral_with_shared_root_prerequisites(self):
        notebook, code = load(VALIDATION)
        markdown = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        for required in (
            "/content/drive/MyDrive/ddpm-derm-augmentation",
            "/content/drive/MyDrive/ddpm-derm-panderm-runs",
            "Editor permission",
            "GH_TOKEN",
        ):
            with self.subTest(required=required):
                self.assertIn(required, markdown)
        self.assertIn("Never create a private folder of the", markdown)
        # Account-neutral: no hard-coded identity anywhere in the notebook.
        self.assertNotRegex(
            json.dumps(notebook), r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        )
        self.assertNotIn("PANDERM_CHECKPOINT_DRIVE_ID = None", code)
        for forbidden in ("MyDrive/ddpm-derm-panderm-runs-", "user_id", "account_email"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, code)
        # Both shared roots are asserted, never created.
        self.assertIn("assert SHARED_PROJECT_DIR.is_dir()", code)
        self.assertIn("assert SHARED_RUN_ROOT.is_dir()", code)
        self.assertIn("do not create a private replacement", code)
        self.assertIn("panderm_run.require_existing_shared_root", code)
        self.assertIn("assert SHARED_ROOT_SENTINEL.is_file()", code)
        self.assertIn("do not recreate it", code)
        self.assertIn("panderm_run.create_or_validate_sentinel", code)
        self.assertIn("panderm_run.require_shared_root_sentinel_identity", code)
        self.assertIn("panderm_run.probe_shared_drive", code)
        self.assertNotIn(
            'sentinel["resolved_path"] == str(resolved_root)', code
        )
        # Pre-existing markers/records stop the run rather than being removed.
        self.assertIn("a validation_record already exists", code)
        self.assertIn("a prior gate already failed this version", code)
        self.assertIn("shutil.rmtree(SMOKE_SAMPLE_DIR)", code)
        self.assertNotIn("shutil.rmtree(SHARED", code)
        self.assertNotIn("SHARED_RUN_ROOT.mkdir", code)
        self.assertNotIn("SHARED_PROJECT_DIR.mkdir", code)

    def test_ddpm_train_only_test_runs_without_real_test_manifest(self):
        fieldnames = [
            "image_path",
            "label_idx",
            "dx",
            "lesion_id",
            "image_id",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            manifests_root = data_root / "manifests"
            manifests_root.mkdir()
            (manifests_root / "class_to_idx.json").write_text(
                '{"akiec": 0}\n', encoding="utf-8"
            )
            for split in ("train", "val"):
                with (manifests_root / f"{split}.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    if split == "train":
                        for index in range(3):
                            writer.writerow(
                                {
                                    "image_path": f"train_{index}.jpg",
                                    "label_idx": 3,
                                    "dx": "df",
                                    "lesion_id": f"train_lesion_{index}",
                                    "image_id": f"train_image_{index}",
                                }
                            )

            env = os.environ.copy()
            env["DDPM_DERM_DATA_DIR"] = str(data_root)
            result = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "unittest",
                    "-v",
                    "tests.test_ddpm_sampler.DDPMSamplerTests."
                    "test_ddpm_frame_is_train_only_and_seeded_limit_repeats",
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertFalse((manifests_root / "test.csv").exists())
            self.assertEqual(
                result.returncode,
                0,
                result.stdout + result.stderr,
            )

    def test_failure_and_success_records_keep_prohibition_flags(self):
        _, code = load(VALIDATION)
        failure_write = code.index(
            "panderm_run.write_json_atomic(LATEST_FAILURE_RECORD, failure_record)"
        )
        failure_raise = code.index("raise RuntimeError(gate_failures)")
        success_write = code.index(
            "panderm_run.write_json_atomic(VALIDATION_RECORD, record)"
        )
        self.assertLess(failure_write, failure_raise)
        self.assertLess(failure_raise, success_write)
        for required in (
            '"formal_training_allowed": False',
            '"test_access_allowed": False',
            '"test_metrics": None',
            'assert not FORMAL_ROOT.exists()',
            "formal_training_allowed=false",
            "test_access_allowed=false",
            "claim_boundary=suggestive_exploratory_only",
        ):
            self.assertIn(required, code)

    def test_validation_records_unauditable_overlap_boundary(self):
        _, code = load(VALIDATION)
        for required in (
            '"not_independently_excludable"',
            '"independent_audit_possible"] is False',
            '"patient_level_overlap"] == "not_excludable"',
            '"exact_fixed_validation_test_overlap"] == "unproven"',
            '"suggestive_exploratory_only"',
        ):
            self.assertIn(required, code)


class StopDecisionNotebookTests(unittest.TestCase):
    def test_first_substantive_cell_is_the_exact_fail_loud_stop(self):
        notebook, code = load(FORMAL)
        first = "".join(notebook["cells"][0]["source"])
        self.assertEqual(notebook["cells"][0]["cell_type"], "code")
        self.assertIn(
            panderm_run.PROHIBITED_FORMAL_TEST_REASON,
            first,
        )
        self.assertIn("raise RuntimeError(STOP_REASON)", first)
        self.assertLess(first.index("STOP_REASON"), first.index("raise RuntimeError"))

    def test_stop_notebook_has_no_bypass_helper(self):
        notebook, code = load(FORMAL)
        raw = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        for forbidden in (
            "ddpm_derm.train_panderm",
            "--evaluation-scope",
            "load_split",
            "test.csv",
            "test_metrics",
            "run_queue",
            "aggregate_results",
            "validate_completed",
            "create_running_marker",
            "torch.load",
            "subprocess",
            "gdown",
        ):
            self.assertNotIn(forbidden, code)
        for required in (
            "STOP / DECISION",
            "formal_training_allowed=False",
            "test_access_allowed=False",
            "no training, resume, aggregation",
            "CC BY-NC-ND 4.0",
            "must not be shared or deployed",
        ):
            self.assertIn(required, raw)


class ProtectedArtifactTests(unittest.TestCase):
    def test_existing_notebooks_are_unchanged(self):
        for name in FROZEN_NOTEBOOKS:
            with self.subTest(name=name):
                result = subprocess.run(
                    ["git", "diff", "--quiet", "HEAD", "--", f"notebooks/{name}"],
                    cwd=ROOT,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, f"{name} was modified")

    def test_protected_untracked_user_notebook_is_unchanged(self):
        path = ROOT / "notebooks" / PROTECTED_NOTEBOOK
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, PROTECTED_SHA256)
            return
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", f"notebooks/{PROTECTED_NOTEBOOK}"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        self.assertNotEqual(tracked.returncode, 0)

    def test_protected_documents_are_unchanged(self):
        for name in (
            "README.md",
            "HANDOFF.md",
            "EXPERIMENT_LOG.md",
            "render.yaml",
            "Dockerfile",
            "Dockerfile.render",
            "requirements-deploy.txt",
        ):
            with self.subTest(name=name):
                result = subprocess.run(
                    ["git", "diff", "--quiet", "HEAD", "--", name],
                    cwd=ROOT,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, f"{name} was modified")

    def test_deployment_directories_are_unchanged(self):
        for path in ("app", "deploy", "outputs", "scripts"):
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "diff", "--quiet", "HEAD", "--", path],
                    cwd=ROOT,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, f"{path} was modified")

    def test_plan_records_the_binding_validation_only_decision(self):
        plan = (ROOT / "PANDERM_BASE_C1_FINETUNE_PLAN.md").read_text(
            encoding="utf-8"
        )
        for required in (
            panderm_run.UPSTREAM_COMMIT,
            panderm_run.CHECKPOINT_FILENAME,
            panderm_run.CHECKPOINT_DRIVE_FILE_ID,
            "CC-BY-NC-ND 4.0",
            "REPLACE_AFTER_FIRST_DOWNLOAD",
            "REPLACE_AFTER_PUSH",
            "independent_audit_possible=False",
            "patient_overlap=not_excludable",
            "exact_fixed_validation_test_overlap=unproven",
            "formal_training_allowed=False",
            "test_access_allowed=False",
            "validation-only",
            "must not produce test metrics",
            panderm_run.PROHIBITED_FORMAL_TEST_REASON,
        ):
            self.assertIn(required, plan)


if __name__ == "__main__":
    unittest.main()
