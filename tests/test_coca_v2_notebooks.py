"""Static safety checks for the two isolated CoCa v2 notebooks."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "colab_coca_v2_weighted_validation.ipynb",
    "colab_coca_v2_weighted_classifier.ipynb",
)
IMPLEMENTATION_COMMIT = "6e2c60c52b367d864a3f96d75cfd401076b72186"


def load(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    return notebook, code


class CoCaV2NotebookTests(unittest.TestCase):
    def test_json_code_cells_outputs_and_commit_pin(self):
        for name in NAMES:
            notebook, code = load(name)
            self.assertEqual(notebook["cells"][0]["cell_type"], "code")
            first = "".join(notebook["cells"][0]["source"])
            self.assertIn(
                f'EXPECTED_GIT_COMMIT = "{IMPLEMENTATION_COMMIT}"', first
            )
            self.assertNotIn('EXPECTED_GIT_COMMIT = "REPLACE_AFTER_PUSH"', first)
            self.assertIn('EXPECTED_GIT_COMMIT != "REPLACE_AFTER_PUSH"', first)
            self.assertIn("Pin the reviewed pushed commit", first)
            for index, cell in enumerate(notebook["cells"]):
                self.assertFalse(cell.get("outputs"))
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), f"{name}:cell-{index}", "exec")
            self.assertNotRegex(code, r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")
            self.assertNotIn("DRIVE_FOLDER_ID =", code)

    def test_validation_notebook_is_validation_only_and_has_fixed_gate(self):
        _, code = load("colab_coca_v2_weighted_validation.ipynb")
        self.assertNotIn("run_queue =", code)
        self.assertNotIn("FORMAL_ROOT =", code)
        for required in (
            'RUN_VERSION = "v2_weighted_ce"',
            '"--class-weighting", "inverse_sqrt"',
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
            "class_weighting=inverse_sqrt_train_frequency",
            "child_process_drive_only_restore",
        ):
            self.assertIn(required, code)
        self.assertIn('identity_columns = ("image_id", "lesion_id")', code)
        for split, field in (
            ("val", "image_id"),
            ("test", "image_id"),
            ("val", "lesion_id"),
            ("test", "lesion_id"),
        ):
            self.assertIn(
                f'assert_candidate_disjoint("{split}", "{field}")', code
            )
        self.assertIn(
            "candidate leakage: split={split} field={field} "
            "conflicts={conflicts[:10]}",
            code,
        )

    def test_formal_notebook_has_fixed_six_run_queue_and_guards(self):
        _, code = load("colab_coca_v2_weighted_classifier.ipynb")
        for required in (
            'ACCOUNT_LABEL = ""',
            'RUN_MODE = "fresh"',
            "CLEAR STALE MARKER",
            'run_queue = [(variant, seed) for variant in ("C1", "C4") for seed in (0, 1, 2)]',
            '"--epochs", "20"',
            '"--class-weighting", "inverse_sqrt"',
            '"--evaluation-scope", "full"',
            "coca_run.create_running_marker",
            "coca_run.require_validation_record",
            "coca_run.require_resume_identity",
            "FORMAL COCA V2 WEIGHTED RUNS COMPLETED AND VERIFIED",
        ):
            self.assertIn(required, code)

    def test_v1_is_read_only_and_v2_paths_are_isolated(self):
        for name in NAMES:
            _, code = load(name)
            self.assertIn('/ "v1"', code)
            self.assertIn('RUN_VERSION = "v2_weighted_ce"', code)
            self.assertIn("v1_before", code)
            self.assertIn("v1_after == v1_before", code)
            self.assertNotIn("ensure_tree(SHARED_RUN_ROOT, V1_ROOT", code)
            self.assertNotRegex(code, r"write_json_atomic\([^\n]*V1_ROOT")

    def test_existing_v1_notebooks_match_head(self):
        for name in ("colab_coca_validation.ipynb", "colab_coca_classifier.ipynb"):
            expected = subprocess.check_output(
                ["git", "show", f"HEAD:notebooks/{name}"], cwd=ROOT
            )
            actual = (ROOT / "notebooks" / name).read_bytes()
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
