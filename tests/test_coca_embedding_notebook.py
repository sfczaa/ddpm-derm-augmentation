"""Static safety checks for the post-v4 frozen-CoCa embedding diagnostic."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_coca_v4_post_failure_embedding_diagnostic.ipynb"
PINNED_GIT_COMMIT = "b584bd321dd11258469f8c564bcc8a82a3ae11ac"


class CoCaEmbeddingNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )
        cls.all_source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_notebook_is_unexecuted_valid_and_pinned(self):
        first = "".join(self.notebook["cells"][0]["source"])
        self.assertIn(f'EXPECTED_GIT_COMMIT = "{PINNED_GIT_COMMIT}"', first)
        self.assertNotIn('EXPECTED_GIT_COMMIT = "REPLACE_AFTER_PUSH"', first)
        self.assertIn('EXPECTED_GIT_COMMIT != "REPLACE_AFTER_PUSH"', first)
        for index, cell in enumerate(self.notebook["cells"]):
            self.assertFalse(cell.get("outputs"))
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                compile("".join(cell["source"]), f"diagnostic:cell-{index}", "exec")

    def test_scope_is_descriptive_embedding_only_without_test_or_formal(self):
        for required in (
            'DIAGNOSTIC_VERSION = "v1_frozen_coca_df_embeddings_safe_v2"',
            '"real_train_df": 85',
            '"synthetic_df": 500',
            '"validation_df": 14',
            "centroid cosine distance",
            'record["test_data_accessed"] is False',
            'record["formal_training_started"] is False',
            "EMBEDDING DIAGNOSTIC COMPLETED",
        ):
            self.assertIn(required, self.all_source)
        self.assertNotIn('load_split("test")', self.code)
        self.assertNotIn("run_queue", self.code)
        self.assertNotIn("train_classifier", self.code)
        self.assertNotIn("FORMAL_ROOT", self.code)
        self.assertNotIn("evaluation_scope", self.code)

    def test_v4_failure_is_required_and_guarded_before_diagnostic_write(self):
        for required in (
            'V4_FAILURE_RECORD = V4_ROOT / "latest_validation_failure.json"',
            'v4_failure["validation_status"] == "VALIDATION FAILED"',
            'v4_failure["formal_training_started"] is False',
            'v4_failure["non_collapse_gate"]["C4"]["prediction_counts"]["df"] == 0',
            "guard_after == guard_before",
        ):
            self.assertIn(required, self.code)
        self.assertLess(
            self.code.index("guard_before"),
            self.code.index("coca_run.write_json_atomic(LATEST_RECORD, record)"),
        )

    def test_secure_clone_and_fixed_candidate_identity(self):
        for required in (
            'userdata.get("GH_TOKEN")',
            'subprocess.run(["git", "clone", REPO_URL, str(CODE_DIR)], check=True, env=clone_env)',
            'clone_env["GIT_CONFIG_VALUE_0"] = ""',
            "del token, basic_credential, clone_env",
            'EXPECTED_CANDIDATE_SHA256 = "9ef9b44e404f74aab8211f4e7d123da3258ba8ba4e3004a4147d1761ed343b34"',
            '"--device", "cuda"',
            '"--batch-size", "32"',
        ):
            self.assertIn(required, self.code)
        self.assertNotIn("@github.com", self.code)
        self.assertNotRegex(json.dumps(self.notebook), r"gh[pousr]_[A-Za-z0-9]")
        self.assertNotRegex(json.dumps(self.notebook), r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")


if __name__ == "__main__":
    unittest.main()
