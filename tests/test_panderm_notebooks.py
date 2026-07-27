"""Static safety contracts for the two PanDerm Colab notebooks."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm import panderm_run  # noqa: E402


VALIDATION = "colab_panderm_base_c1_finetune_validation.ipynb"
FORMAL = "colab_panderm_base_c1_finetune_classifier.ipynb"
NAMES = (VALIDATION, FORMAL)

PROTECTED_NOTEBOOK = "colab_balanced_ddpm.ipynb"
PROTECTED_SHA256 = "ef8bb8be8fa0865a3297e361f1984631141eadca073cc1451ad5223ce27882b8"

PIN_PLACEHOLDER = "REPLACE_AFTER_PUSH"
# The reviewed, pushed implementation commit the canonical notebook checks out.
# This is the *implementation* commit, never the pinning commit that carries this
# line: Colab clones the pinned SHA, so pinning the later commit would be
# unresolvable at the moment it is written.
IMPLEMENTATION_COMMIT = "d8d562ce26f2e8b82a109dead7cf1390617d77b5"

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


class ValidationNotebookTests(unittest.TestCase):
    def test_first_cell_is_pinned_to_the_implementation_commit(self):
        notebook, _ = load(VALIDATION)
        first = "".join(notebook["cells"][0]["source"])
        self.assertEqual(notebook["cells"][0]["cell_type"], "code")
        self.assertRegex(IMPLEMENTATION_COMMIT, r"^[0-9a-f]{40}$")
        self.assertIn(f'EXPECTED_GIT_COMMIT = "{IMPLEMENTATION_COMMIT}"', first)
        self.assertNotIn(f'EXPECTED_GIT_COMMIT = "{PIN_PLACEHOLDER}"', first)
        # Both halves of the fail-loud guard must survive the pinning.
        self.assertIn(f'EXPECTED_GIT_COMMIT != "{PIN_PLACEHOLDER}"', first)
        self.assertIn("len(EXPECTED_GIT_COMMIT) == 40", first)
        self.assertIn("Pin the reviewed pushed commit", first)

    def test_pinned_first_cell_passes_its_own_guard(self):
        """Executing cell 0 as-is must now succeed, not merely look right."""
        notebook, _ = load(VALIDATION)
        first = "".join(notebook["cells"][0]["source"])
        namespace = {}
        exec(compile(first, "cell-0", "exec"), namespace)
        self.assertEqual(namespace["EXPECTED_GIT_COMMIT"], IMPLEMENTATION_COMMIT)

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
        self.assertIn("panderm_run.stage_validation_data", code)
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
            '"unittest", "discover", "-s", "tests"',
            "tests.test_panderm_blockers.PanDermRunnerMockSmokeTests",
            '"runner_smoke_ok": "OK" in runner_smoke_output',
            '["--evaluation-scope", "full"]',
            '["--drop-path", "0.3"]',
            '["--no-amp"]',
        ):
            self.assertIn(required, code)
        self.assertNotIn("scripts/smoke_test.py", code)

    def test_local_tests_package_prevents_colab_package_shadowing(self):
        self.assertTrue((ROOT / "tests" / "__init__.py").is_file())

    def test_checkpoint_preflight_runs_before_any_image_is_staged(self):
        """Ordering is the whole point: fail on a bad checkpoint, not after 67 min.

        The preflight must sit after the SHA-256 gate and before the cell that
        copies the 8,505 train/val images, so an incompatible checkpoint costs
        seconds rather than an hour of staging.
        """
        notebook, _ = load(VALIDATION)
        sources = ["".join(cell["source"]) for cell in notebook["cells"]]
        sha_gate = next(
            index
            for index, text in enumerate(sources)
            if "panderm_run.require_checkpoint_sha256(" in text
        )
        preflight = next(
            index
            for index, text in enumerate(sources)
            if "CHECKPOINT_PREFLIGHT_COMPLETE = True" in text
        )
        staging = next(
            index
            for index, text in enumerate(sources)
            if "panderm_run.stage_validation_data" in text
        )
        self.assertLess(sha_gate, preflight)
        self.assertLess(preflight, staging)

    def test_checkpoint_preflight_really_reopens_remaps_and_loads(self):
        _, code = load(VALIDATION)
        for required in (
            "panderm.load_pretrained_state(CHECKPOINT_PATH)",
            "panderm.detect_checkpoint_layout(preflight_state)",
            "panderm.remap_pretrained_state_dict(preflight_state, layout=preflight_layout)",
            "panderm.build_panderm_classifier(checkpoint_path=CHECKPOINT_PATH",
            "preflight_layout == panderm.LAYOUT_DIRECT_BACKBONE",
            'assert "fc_norm.weight" in preflight_remapped',
            'assert not any(key.startswith("head.") for key in preflight_remapped)',
            "assert preflight_unexpected == []",
            "assert preflight_missing == []",
            "gc.collect()",
        ):
            with self.subTest(required=required):
                self.assertIn(required, code)
        for printed in (
            '"[preflight] checkpoint bytes:"',
            '"[preflight] checkpoint sha256:"',
            '"[preflight] detected layout:"',
            '"[preflight] tensor entries:"',
            '"[preflight] remapped keys:"',
            '"[preflight] unexpected keys:"',
            '"[preflight] missing non-head keys:"',
            '"CHECKPOINT_PREFLIGHT_COMPLETE=True"',
        ):
            with self.subTest(printed=printed):
                self.assertIn(printed, code)

    def test_phase_one_keeps_its_live_copy_progress(self):
        _, code = load(VALIDATION)
        self.assertIn("panderm_run.stage_validation_data", code)
        self.assertIn('staging_report["images_copied"] == 6995 + 1510', code)
        staging = (
            ROOT / "src" / "ddpm_derm" / "panderm_run.py"
        ).read_text(encoding="utf-8")
        for marker in (
            "[Phase 1] START validation staging",
            "[Phase 1] copy images:",
            "[Phase 1] verify images:",
            "images/s eta=",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, staging)

    def test_phase_three_streams_real_progress_markers(self):
        """Phase 3 must never go silent for minutes again."""
        _, code = load(VALIDATION)
        for marker in (
            '"[Phase 3] START"',
            "[Phase 3] loading upstream module",
            "[Phase 3] building model and loading checkpoint on CPU",
            "[Phase 3] checkpoint load/remap/load_state complete",
            "[Phase 3] moving model to CUDA",
            "[Phase 3] batch preparation ",
            "[Phase 3] initial forward complete",
            "[Phase 3] optimizer created",
            "[Phase 3] accumulation micro-step ",
            "[Phase 3] gradient validation complete",
            "[Phase 3] optimizer step complete",
            "[Phase 3] changed backbone tensors:",
            "[Phase 3] COMPLETE elapsed=",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, code)
        phase_three = next(
            text
            for text in (
                "".join(cell["source"]) for cell in load(VALIDATION)[0]["cells"]
            )
            if "panderm.backbone_gradient_report" in text
        )
        # Every progress print must flush, or Colab buffers it into silence.
        for line in phase_three.splitlines():
            stripped = line.strip()
            if stripped.startswith("print(") and "[Phase 3]" in stripped:
                with self.subTest(line=stripped[:60]):
                    self.assertIn("flush=True", stripped)
        self.assertIn("assert CHECKPOINT_PREFLIGHT_COMPLETE is True", phase_three)

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
        self.assertIn("panderm_run.create_or_validate_sentinel", code)
        self.assertIn("panderm_run.require_shared_root_sentinel_identity", code)
        self.assertIn("panderm_run.probe_shared_drive", code)
        self.assertNotIn(
            'sentinel["resolved_path"] == str(resolved_root)', code
        )
        # Pre-existing markers/records stop the run rather than being removed.
        self.assertIn("a validation_record already exists", code)
        self.assertIn("a prior gate already failed this version", code)
        self.assertNotIn("shutil.rmtree", code)
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
