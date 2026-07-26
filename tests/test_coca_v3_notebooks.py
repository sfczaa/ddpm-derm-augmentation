"""Static safety checks for the two isolated CoCa v3 notebooks.

Verifies the notebooks stay Run-all-safe placeholders (fail loud until pinned),
clone the private repo via a Colab Secret without leaking the token, keep v1/v2
artifacts read-only, isolate v3 outputs, and encode the failure/formal guards.
No real weights are downloaded and nothing here executes a cell.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "colab_coca_v3_inverse_frequency_validation.ipynb",
    "colab_coca_v3_inverse_frequency_classifier.ipynb",
)
# The untracked user notebook must never be touched by this work.
PROTECTED_NOTEBOOK = "colab_balanced_ddpm.ipynb"
PROTECTED_SHA256 = (
    "ef8bb8be8fa0865a3297e361f1984631141eadca073cc1451ad5223ce27882b8"
)
# Every earlier CoCa notebook must remain byte-for-byte identical to HEAD.
HEAD_MATCH_NOTEBOOKS = (
    "colab_coca_validation.ipynb",
    "colab_coca_classifier.ipynb",
    "colab_coca_v2_weighted_validation.ipynb",
    "colab_coca_v2_weighted_classifier.ipynb",
)


def load(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    return notebook, code


class CoCaV3NotebookTests(unittest.TestCase):
    def test_json_cells_pin_and_token_safety(self):
        for name in NAMES:
            notebook, code = load(name)
            self.assertEqual(notebook["cells"][0]["cell_type"], "code")
            first = "".join(notebook["cells"][0]["source"])
            # The release notebook pins one reviewed implementation commit.
            self.assertRegex(first, r'EXPECTED_GIT_COMMIT = "[0-9a-f]{40}"')
            self.assertIn('EXPECTED_GIT_COMMIT != "REPLACE_AFTER_PUSH"', first)
            self.assertIn("len(EXPECTED_GIT_COMMIT) == 40", first)
            self.assertIn("Pin the reviewed pushed commit", first)
            for index, cell in enumerate(notebook["cells"]):
                self.assertFalse(cell.get("outputs"))
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell.get("execution_count"))
                    compile("".join(cell["source"]), f"{name}:cell-{index}", "exec")
            # no email / private Drive folder id
            self.assertNotRegex(code, r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")
            self.assertNotIn("DRIVE_FOLDER_ID =", code)
            # private GitHub clone via Colab Secret GH_TOKEN
            self.assertIn('userdata.get("GH_TOKEN")', code)
            self.assertIn(
                'REPO_URL = "https://github.com/sfczaa/ddpm-derm-augmentation.git"',
                code,
            )
            # no token literal; URL embeds no credential
            self.assertNotIn("ghp_", code)
            self.assertNotIn("github_pat_", code)
            self.assertNotIn("@github.com", code)
            # clone command carries no token, so a CalledProcessError cannot leak it
            self.assertIn(
                'subprocess.run(["git", "clone", REPO_URL, str(CODE_DIR)], '
                "check=True, env=clone_env)",
                code,
            )
            # credential lives only in the child clone env and is scrubbed after
            self.assertIn('clone_env["GIT_CONFIG_VALUE_0"] = ""', code)
            self.assertIn("del token, basic_credential, clone_env", code)
            # runtime-local code + model cache; never run from Drive
            self.assertIn('CODE_DIR = Path("/content/ddpm-coca-v3-code")', code)
            self.assertIn('os.environ["HF_HOME"] = "/content/hf-cache"', code)

    def test_v1_v2_read_only_and_v3_paths_isolated(self):
        for name in NAMES:
            _, code = load(name)
            self.assertIn('RUN_VERSION = "v3_inverse_frequency_ce"', code)
            self.assertIn('/ "v1"', code)
            self.assertIn('/ "v2_weighted_ce"', code)
            self.assertIn("v1 must remain present and read-only", code)
            self.assertIn("v2 must remain present and read-only", code)
            self.assertIn("before_guard", code)
            self.assertIn("guard_after == before_guard", code)
            # v3 must never create/write inside a v1 or v2 root
            self.assertNotIn("ensure_tree(SHARED_RUN_ROOT, V1_ROOT", code)
            self.assertNotIn("ensure_tree(SHARED_RUN_ROOT, V2_ROOT", code)
            self.assertNotRegex(code, r"write_json_atomic\([^\n]*V1_ROOT")
            self.assertNotRegex(code, r"write_json_atomic\([^\n]*V2_ROOT")

    def test_validation_notebook_gate_and_no_formal_queue(self):
        _, code = load("colab_coca_v3_inverse_frequency_validation.ipynb")
        self.assertNotIn("run_queue =", code)
        self.assertNotIn("FORMAL_ROOT =", code)
        for required in (
            'RUN_VERSION = "v3_inverse_frequency_ce"',
            '"--class-weighting", "inverse_frequency"',
            '"--evaluation-scope", "validation_only"',
            '"--epochs", "5"',
            '"--batch-size", "32"',
            "best_validation_df_f1_positive",
            "predicted_df_positive",
            "not_all_nv",
            "at_least_two_predicted_classes",
            'result["test_metrics"] is None',
            '"[test]" not in output',
            "VALIDATION FAILED",
            "VALIDATION PASSED",
            "formal_training_started=false",
            "class_weighting=inverse_train_frequency",
            "child_process_drive_only_restore",
            "(1/n_c)/mean_j(1/n_j)",
            "EXPECTED_EQUAL_CONTRIBUTION",
            "3591",
            "253563656",
            'for wrong_mode in ("none", "inverse_sqrt")',
            "do not re-draw validation for the same version",
        ):
            self.assertIn(required, code)
        self.assertIn('identity_columns = ("image_id", "lesion_id")', code)
        for split, field in (
            ("val", "image_id"),
            ("test", "image_id"),
            ("val", "lesion_id"),
            ("test", "lesion_id"),
        ):
            self.assertIn(f'assert_candidate_disjoint("{split}", "{field}")', code)
        # failure record is written before the stop; success record never on failure
        fail_idx = code.index(
            "coca_run.write_json_atomic(LATEST_FAILURE_RECORD, failure_record)"
        )
        raise_idx = code.index("raise RuntimeError(gate_failures)")
        success_idx = code.index(
            "coca_run.write_json_atomic(VALIDATION_RECORD, record)"
        )
        self.assertLess(fail_idx, raise_idx)
        self.assertLess(raise_idx, success_idx)
        self.assertIn(
            "a failed gate must never leave a success validation_record", code
        )

    def test_validation_verifies_v2_failure_evidence(self):
        _, code = load("colab_coca_v3_inverse_frequency_validation.ipynb")
        for required in (
            "V2_FAILURE_RUN",
            "20260720T085738Z",
            '"v2_weighted_ce"',
            "inverse_sqrt_train_frequency",
            "f14b041d691c9a6fa9dc7e407e574b69c753ada1",
            "v2_failure_evidence",
        ):
            self.assertIn(required, code)

    def test_formal_notebook_fixed_queue_and_validation_guards(self):
        _, code = load("colab_coca_v3_inverse_frequency_classifier.ipynb")
        for required in (
            'ACCOUNT_LABEL = ""',
            'RUN_MODE = "fresh"',
            'RUN_MODE == "resume"',
            "CLEAR STALE MARKER",
            'run_queue = [(variant, seed) for variant in ("C1", "C4") '
            "for seed in (0, 1, 2)]",
            '"--epochs", "20"',
            '"--class-weighting", "inverse_frequency"',
            '"--evaluation-scope", "full"',
            "coca_run.create_running_marker",
            "coca_run.require_validation_record",
            "coca_run.require_resume_identity",
            "validate_completed",
            '{"train": 7495, "val": 1510, "test": 1510}',
            "FORMAL COCA V3 INVERSE-FREQUENCY RUNS COMPLETED AND VERIFIED",
            "colab_coca_v3_inverse_frequency_classifier_executed.ipynb",
        ):
            self.assertIn(required, code)
        # a failed/missing validation blocks formal training
        self.assertIn("formal training is blocked", code)
        self.assertIn(
            "formal training requires a PASSED v3 validation record", code
        )

    def test_shared_root_checked_before_tree_creation(self):
        for name in NAMES:
            _, code = load(name)
            missing_guard = code.index("SHARED_RUN_ROOT.is_dir()")
            first_tree_create = code.index("coca_run.ensure_tree")
            self.assertLess(missing_guard, first_tree_create)

    def test_existing_v1_v2_notebooks_match_head(self):
        for name in HEAD_MATCH_NOTEBOOKS:
            expected = subprocess.check_output(
                ["git", "show", f"HEAD:notebooks/{name}"], cwd=ROOT
            )
            actual = (ROOT / "notebooks" / name).read_bytes()
            self.assertEqual(actual, expected)

    def test_protected_user_notebook_is_untouched(self):
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


if __name__ == "__main__":
    unittest.main()
