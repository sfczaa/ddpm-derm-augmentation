"""Static safety checks for the two CoCa Colab notebooks."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_notebook(name):
    path = ROOT / "notebooks" / name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    return notebook, code


class CoCaNotebookTests(unittest.TestCase):
    def test_all_code_cells_compile(self):
        for name in ("colab_coca_validation.ipynb", "colab_coca_classifier.ipynb"):
            notebook, _ = load_notebook(name)
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), f"{name}:cell-{index}", "exec")

    def test_validation_notebook_cannot_start_formal_queue(self):
        _, code = load_notebook("colab_coca_validation.ipynb")
        self.assertNotIn("run_queue =", code)
        self.assertNotIn("FORMAL_ROOT =", code)
        self.assertIn("VALIDATION PASSED", code)
        self.assertIn("formal_training_started=false", code)
        self.assertIn("CANDIDATE_MANIFEST.is_file()", code)
        self.assertIn("EXPECTED_CANDIDATE_SHA256", code)
        for required in (
            "frozen_backbone_head_only_v1", "MAX_COCA_CHECKPOINT_BYTES",
            "head_state_dict", "model_state_dict\" not in checkpoint",
            "encoder_state_dict", "validate_checkpoint_payload",
            "child_process_drive_restore", "checkpoint_sizes",
            "encoder_weights_stored",
        ):
            self.assertIn(required, code)

    def test_training_notebook_requires_validation_and_concurrency_guard(self):
        _, code = load_notebook("colab_coca_classifier.ipynb")
        self.assertIn("coca_run.require_validation_record", code)
        self.assertIn("formal_training_started", code)
        self.assertIn("coca_run.create_running_marker", code)
        self.assertIn("CLEAR STALE MARKER", code)
        self.assertIn("run_queue =", code)
        self.assertIn("RUN_MODE == \"fresh\"", code)
        self.assertIn("RUN_MODE == \"resume\"", code)
        for required in (
            "frozen_backbone_head_only_v1", "MAX_COCA_CHECKPOINT_BYTES",
            "head_state_dict", "model_state_dict\" not in checkpoint",
            "coca_run.checkpoint_size", "checkpoint_sizes",
            "encoder_weights_stored", "validate_completed",
        ):
            self.assertIn(required, code)

    def test_missing_shared_root_is_checked_before_tree_creation(self):
        for name in ("colab_coca_validation.ipynb", "colab_coca_classifier.ipynb"):
            _, code = load_notebook(name)
            missing_guard = code.index("assert SHARED_RUN_ROOT.is_dir()")
            first_tree_create = code.index("coca_run.ensure_tree")
            self.assertLess(missing_guard, first_tree_create)


if __name__ == "__main__":
    unittest.main()
