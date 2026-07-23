"""Static safety checks for the synthetic mixture Colab diagnostic."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_coca_v4_synthetic_mixture_diagnostic.ipynb"


class SyntheticMixtureNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell["source"])
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_unexecuted_and_unpinned_fail_loud(self):
        self.assertIn('EXPECTED_GIT_COMMIT = "REPLACE_AFTER_PUSH"', self.code)
        self.assertIn('EXPECTED_GIT_COMMIT != "REPLACE_AFTER_PUSH"', self.code)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_fixed_validation_only_queue_and_objective(self):
        self.assertIn("SYNTHETIC_COUNTS = (0, 125, 250, 375, 500)", self.code)
        self.assertIn('"--epochs", "5"', self.code)
        self.assertIn('"--seed", "0"', self.code)
        self.assertIn('"--evaluation-scope", "validation_only"', self.code)
        self.assertIn('"--mixture-synthetic-count", str(synthetic_count)', self.code)
        self.assertIn('"--loss-name", "focal_cross_entropy"', self.code)
        self.assertIn('"--class-weighting", "inverse_frequency"', self.code)
        self.assertIn('"--focal-gamma", "2.0"', self.code)

    def test_no_test_or_formal_run_path(self):
        self.assertNotIn('load_split("test")', self.code)
        self.assertNotIn("FORMAL_ROOT", self.code)
        self.assertNotIn('"--evaluation-scope", "full"', self.code)
        self.assertIn('"class_to_idx.json"', self.code)
        self.assertIn('"test_data_accessed": False', self.code)
        self.assertIn('"formal_training_started": False', self.code)

    def test_requires_prior_failure_and_embedding_evidence(self):
        self.assertIn('v4_failure["validation_status"] == "VALIDATION FAILED"', self.code)
        self.assertIn('embedding["diagnostic_status"] == "COMPLETED"', self.code)
        self.assertIn("guard_after == guard_before", self.code)
        self.assertIn('"interpretation_scope": "descriptive_not_candidate_selection"', self.code)

    def test_candidate_guard_allows_shared_synthetic_lesion_marker(self):
        self.assertIn(
            'assert not candidate["image_id"].duplicated().any()', self.code
        )
        self.assertNotIn("not candidate[field].duplicated().any()", self.code)

    def test_phase1_local_copy_reports_visible_progress_and_fails_loud(self):
        self.assertIn("def copy_group(", self.code)
        self.assertIn('f"START {group} copy: total={total}"', self.code)
        self.assertIn("flush=True", self.code)
        self.assertIn("copy failed at ", self.code)
        self.assertIn(") from exc", self.code)


if __name__ == "__main__":
    unittest.main()
