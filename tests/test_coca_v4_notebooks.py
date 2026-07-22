"""Static safety checks for the isolated CoCa v4 focal notebooks."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm import coca_run  # noqa: E402


NAMES = (
    "colab_coca_v4_focal_inverse_frequency_validation.ipynb",
    "colab_coca_v4_focal_inverse_frequency_classifier.ipynb",
)
PROTECTED_NOTEBOOK = "colab_balanced_ddpm.ipynb"
PROTECTED_SHA256 = (
    "ef8bb8be8fa0865a3297e361f1984631141eadca073cc1451ad5223ce27882b8"
)
PINNED_GIT_COMMIT = "61cd4dc1b2113218621613ebd23ab08a12ad53fb"
HEAD_MATCH_NOTEBOOKS = (
    "colab_coca_validation.ipynb",
    "colab_coca_classifier.ipynb",
    "colab_coca_v2_weighted_validation.ipynb",
    "colab_coca_v2_weighted_classifier.ipynb",
    "colab_coca_v3_inverse_frequency_validation.ipynb",
    "colab_coca_v3_inverse_frequency_classifier.ipynb",
)


def load(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    return notebook, code


class CoCaV4NotebookTests(unittest.TestCase):
    def test_json_code_cells_are_unexecuted_and_pinned(self):
        for name in NAMES:
            notebook, code = load(name)
            first = "".join(notebook["cells"][0]["source"])
            self.assertEqual(notebook["cells"][0]["cell_type"], "code")
            self.assertIn(
                f'EXPECTED_GIT_COMMIT = "{PINNED_GIT_COMMIT}"', first
            )
            self.assertNotIn('EXPECTED_GIT_COMMIT = "REPLACE_AFTER_PUSH"', first)
            self.assertIn('EXPECTED_GIT_COMMIT != "REPLACE_AFTER_PUSH"', first)
            self.assertIn("Pin the reviewed pushed commit", first)
            for index, cell in enumerate(notebook["cells"]):
                self.assertFalse(cell.get("outputs"))
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell.get("execution_count"))
                    compile("".join(cell["source"]), f"{name}:cell-{index}", "exec")
            self.assertNotRegex(code, r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")
            self.assertNotIn("DRIVE_FOLDER_ID =", code)

    def test_private_clone_keeps_token_out_of_url_output_and_metadata(self):
        for name in NAMES:
            notebook, code = load(name)
            self.assertIn('userdata.get("GH_TOKEN")', code)
            self.assertIn(
                'REPO_URL = "https://github.com/sfczaa/ddpm-derm-augmentation.git"',
                code,
            )
            self.assertIn(
                'subprocess.run(["git", "clone", REPO_URL, str(CODE_DIR)], '
                "check=True, env=clone_env)",
                code,
            )
            self.assertIn('clone_env["GIT_CONFIG_VALUE_0"] = ""', code)
            self.assertIn("del token, basic_credential, clone_env", code)
            self.assertNotIn("ghp_", code)
            self.assertNotIn("github_pat_", code)
            self.assertNotIn("@github.com", code)
            self.assertNotRegex(json.dumps(notebook), r"gh[pousr]_[A-Za-z0-9]")
            self.assertIn('CODE_DIR = Path("/content/ddpm-coca-v4-code")', code)
            self.assertIn('os.environ["HF_HOME"] = "/content/hf-cache"', code)

    def test_v4_paths_and_objective_are_fixed_and_isolated(self):
        for name in NAMES:
            _, code = load(name)
            for required in (
                'RUN_VERSION = "v4_focal_inverse_frequency"',
                'LOSS_NAME = "focal_cross_entropy"',
                "FOCAL_GAMMA = 2.0",
                '"--loss-name", "focal_cross_entropy"',
                '"--class-weighting", "inverse_frequency"',
                '"--focal-gamma", "2.0"',
                '/ "v1"',
                '/ "v2_weighted_ce"',
                '/ "v3_inverse_frequency_ce"',
            ):
                self.assertIn(required, code)
            self.assertNotIn("ensure_tree(SHARED_RUN_ROOT, V1_ROOT", code)
            self.assertNotIn("ensure_tree(SHARED_RUN_ROOT, V2_ROOT", code)
            self.assertNotIn("ensure_tree(SHARED_RUN_ROOT, V3_ROOT", code)
            self.assertNotRegex(code, r"write_json_atomic\([^\n]*V[123]_ROOT")

    def test_validation_has_full_checks_focal_math_and_no_formal_queue(self):
        notebook, code = load(NAMES[0])
        all_source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertNotIn("run_queue =", code)
        self.assertNotIn("FORMAL_ROOT =", code)
        for phase in (
            "## Phase 0 CHECK",
            "## Phase 1 CHECK",
            "## Phases 2-3 CHECK",
            "## Phase 4 RUN",
            "## Phase 5 RUN",
            "## Phase 6 REVIEW",
        ):
            self.assertIn(phase, all_source)
        for required in (
            '"--evaluation-scope", "validation_only"',
            '"--epochs", "5"',
            '"--batch-size", "32"',
            "best_validation_df_f1_positive",
            "predicted_df_positive",
            "not_all_nv",
            "not_all_df",
            "at_least_two_predicted_classes",
            "manual_alpha_t.sum()",
            "focal_gamma=0.0",
            "torch.nn.CrossEntropyLoss(weight=gamma_zero.weight)",
            'result["test_metrics"] is None',
            '"[test]" not in output',
            "VALIDATION FAILED",
            "VALIDATION PASSED",
            "formal_training_started=false",
            "loss_name=focal_cross_entropy",
            "class_weighting=inverse_train_frequency",
            "focal_gamma=2.0",
            "child_process_drive_only_restore",
        ):
            self.assertIn(required, code)
        failure_write = code.index(
            "coca_run.write_json_atomic(LATEST_FAILURE_RECORD, failure_record)"
        )
        failure_raise = code.index("raise RuntimeError(gate_failures)")
        success_write = code.index(
            "coca_run.write_json_atomic(VALIDATION_RECORD, record)"
        )
        self.assertLess(failure_write, failure_raise)
        self.assertLess(failure_raise, success_write)

    def test_validation_verifies_known_v3_failure_and_read_only_guards(self):
        _, code = load(NAMES[0])
        for required in (
            "V3_LATEST_FAILURE_RECORD.is_file()",
            "not V3_VALIDATION_RECORD.exists()",
            'v3_failure["validation_status"] == "VALIDATION FAILED"',
            'v3_failure["formal_training_started"] is False',
            'v3_failure["non_collapse_gate"]["C1"]["best_validation_df_f1"] == 0.08',
            'v3_failure["non_collapse_gate"]["C1"]["prediction_counts"]["df"] == 86',
            'v3_failure["non_collapse_gate"]["C4"]["best_validation_df_f1"] == 0.0',
            'v3_failure["non_collapse_gate"]["C4"]["prediction_counts"]["df"] == 0',
            "v3_failure_evidence",
            "guard_after == before_guard",
        ):
            self.assertIn(required, code)

    def test_formal_notebook_blocks_any_failure_and_has_fixed_six_runs(self):
        _, code = load(NAMES[1])
        failure_guard = code.index("if LATEST_FAILURE_RECORD.exists():")
        validation_read = code.index("validation = json.loads")
        self.assertLess(failure_guard, validation_read)
        for required in (
            'ACCOUNT_LABEL = ""',
            'RUN_MODE = "fresh"',
            'RUN_MODE == "resume"',
            "CLEAR STALE MARKER",
            'run_queue = [(variant, seed) for variant in ("C1", "C4") for seed in (0, 1, 2)]',
            '"--epochs", "20"',
            '"--evaluation-scope", "full"',
            "checkpoint_saved=last.pt",
            "coca_run.create_running_marker",
            "coca_run.require_validation_record",
            "coca_run.require_resume_identity",
            "validate_completed",
            '{"train": 7495, "val": 1510, "test": 1510}',
            "FORMAL COCA V4 FOCAL RUNS COMPLETED AND VERIFIED",
            "colab_coca_v4_focal_inverse_frequency_classifier_executed.ipynb",
        ):
            self.assertIn(required, code)

    def test_formal_rejects_six_uniformly_wrong_immutable_identities(self):
        _, code = load(NAMES[1])
        result_check = (
            'coca_run.require_resume_identity(result["run_identity"], '
            "formal_identity)"
        )
        checkpoint_check = (
            'coca_run.require_resume_identity(checkpoint["run_identity"], '
            "formal_identity)"
        )
        self.assertIn(result_check, code)
        self.assertIn(checkpoint_check, code)
        self.assertLess(code.index(result_check), code.index("completed identity verified"))
        self.assertLess(
            code.index("runs = [validate_completed"),
            code.index("coca_run.aggregate_results(runs)"),
        )

        formal_identity = {
            key: f"expected-{key}" for key in coca_run.IMMUTABLE_IDENTITY_KEYS
        }
        wrong_keys = (
            "git_commit",
            "fixed_split_identity",
            "shared_root_uuid",
            "formal_output_identity",
        )
        uniformly_wrong = dict(formal_identity)
        uniformly_wrong.update({key: f"wrong-{key}" for key in wrong_keys})
        for variant in ("C1", "C4"):
            for seed in (0, 1, 2):
                with self.subTest(variant=variant, seed=seed):
                    with self.assertRaisesRegex(
                        ValueError,
                        ".*".join(wrong_keys),
                    ):
                        coca_run.require_resume_identity(
                            uniformly_wrong, formal_identity
                        )

    def test_existing_notebooks_and_protected_user_file_are_unchanged(self):
        for name in HEAD_MATCH_NOTEBOOKS:
            result = subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--", f"notebooks/{name}"],
                cwd=ROOT,
                check=False,
            )
            self.assertEqual(result.returncode, 0)
        digest = hashlib.sha256(
            (ROOT / "notebooks" / PROTECTED_NOTEBOOK).read_bytes()
        ).hexdigest()
        self.assertEqual(digest, PROTECTED_SHA256)


if __name__ == "__main__":
    unittest.main()
