"""Static safety checks for the C4-filtered Colab notebook.

Two real failures in this repository's history are what these guard against.

The first: a notebook cloned the private repo anonymously and exit-128'd before
anything ran. The clone here must use the reviewed GH_TOKEN pattern -- a Colab
Secret turned into an extraheader Basic credential on a *child* environment,
never on REPO_URL, the remote, or process-wide state.

The second, which this project hit three times: pinning a notebook to a commit
that predates the code it means to run. The sampling sweep was pinned to
`d2683eb`, which was earlier than the fix to the scripts' default paths, so
re-running it would have measured the wrong batch again. So the pin is not
merely checked for being a 40-character string: the commit must exist in this
repository and must actually contain the selection script.

Nothing here executes a cell, and no credential is involved.
"""

from __future__ import annotations

import json
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_c4_filtered_classifier.ipynb"
SELECTION_SCRIPT = "scripts/c4_filtered_select.py"

# The formal matched-585 configuration, from
# outputs/classifier_df585/results/results_C4_seed0.json. C4-filtered must
# differ from C1 and C4 in the source of the df rows and nothing else, so these
# are locked here: changing one should require changing this test on purpose.
FORMAL_CONFIG = {
    "EPOCHS": "20",
    "DF_TARGET_COUNT": "585",
    "BATCH_SIZE": "32",
    "IMG_SIZE": "128",
    "LEARNING_RATE": "3e-4",
    "WEIGHT_DECAY": "1e-4",
    "NUM_WORKERS": "2",
}


class C4FilteredNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = [
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        cls.code = "\n".join(cls.cells)

    # --- publication hygiene -------------------------------------------------

    def test_notebook_is_committed_unexecuted(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])

    def test_every_code_cell_compiles(self):
        for index, source in enumerate(self.cells):
            with self.subTest(cell=index):
                compile(source, f"cell{index}", "exec")

    # --- the pin -------------------------------------------------------------

    def test_pin_is_a_real_commit_that_contains_the_selection_script(self):
        match = re.search(r'EXPECTED_GIT_COMMIT = "([^"]+)"', self.code)
        self.assertIsNotNone(match, "the notebook must declare EXPECTED_GIT_COMMIT")
        pin = match.group(1)
        self.assertNotEqual(pin, "REPLACE_AFTER_PUSH", "pin the reviewed pushed commit")
        self.assertRegex(pin, r"^[0-9a-f]{40}$")

        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{pin}^{{commit}}"],
            cwd=ROOT, check=False,
        )
        self.assertEqual(exists.returncode, 0, f"{pin} is not a commit in this repository")

        # The notebook checks this commit out and runs the script from it.
        contains = subprocess.run(
            ["git", "cat-file", "-e", f"{pin}:{SELECTION_SCRIPT}"],
            cwd=ROOT, check=False,
        )
        self.assertEqual(
            contains.returncode, 0,
            f"{pin} does not contain {SELECTION_SCRIPT}; a run against this pin "
            "would fail on Colab after the clone",
        )

    def test_pin_guard_fails_loud_on_the_placeholder(self):
        self.assertIn(
            'assert len(EXPECTED_GIT_COMMIT) == 40 and EXPECTED_GIT_COMMIT != "REPLACE_AFTER_PUSH"',
            self.code,
        )

    # --- clone credential handling -------------------------------------------

    def test_clone_uses_an_extraheader_credential_on_a_child_environment(self):
        self.assertIn('userdata.get("GH_TOKEN")', self.code)
        self.assertIn('clone_env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"', self.code)
        self.assertIn('clone_env["GIT_TERMINAL_PROMPT"] = "0"', self.code)
        self.assertIn("env=clone_env", self.code)

    def test_the_credential_never_reaches_the_url_the_remote_or_the_process(self):
        self.assertIn('REPO_URL = "https://github.com/sfczaa/ddpm-derm-augmentation.git"', self.code)
        self.assertNotIn("x-access-token@", self.code)
        self.assertNotIn("os.environ[\"GIT_CONFIG_VALUE_0\"]", self.code)
        # Cleared and dropped in a finally, so a failed clone does not leave it.
        self.assertIn("finally:", self.code)
        self.assertIn("del token, basic_credential, clone_env", self.code)
        self.assertIn('assert "@" not in remote and "x-access-token" not in remote', self.code)

    def test_clone_must_be_clean_and_detached_at_the_pin(self):
        self.assertIn(
            'assert commit == EXPECTED_GIT_COMMIT and not status',
            self.code,
        )

    # --- the pre-registered rules --------------------------------------------

    def test_training_configuration_matches_the_formal_matched_585_run(self):
        for name, value in FORMAL_CONFIG.items():
            with self.subTest(setting=name):
                self.assertRegex(self.code, rf"(?m)^{name} = {re.escape(value)}$")

    def test_seeds_are_the_three_formal_seeds(self):
        self.assertIn("SEEDS = (0, 1, 2)", self.code)

    def test_the_gate_is_asserted_before_any_training_starts(self):
        gate = self.code.index('assert selection["condition_runnable"]')
        train = self.code.index('"--variant", "C4_FILTERED"')
        self.assertLess(
            gate, train,
            "the shortfall gate must be checked before the training loop, or a "
            "run could train on a handful of accepted images",
        )

    def test_a_completed_run_is_not_silently_overwritten(self):
        # The design evaluates the test split once.
        self.assertIn("results_C4_FILTERED_seed", self.code)
        self.assertIn("evaluates the test split once", self.code)

    def test_the_published_pool_directory_is_not_written_to(self):
        # The accepted manifest goes into the run root; the images are still
        # resolved out of the published pool through --generated-root.
        self.assertIn("ACCEPTED_MANIFEST = RUN_ROOT /", self.code)
        self.assertIn('"--generated-root", str(SYNTHETIC_POOL_DIR)', self.code)
        self.assertIn('"--accepted-out", str(ACCEPTED_MANIFEST)', self.code)

    def test_the_published_pool_is_the_one_the_formal_c4_trained_on(self):
        self.assertIn('"outputs" / "synthetic_df" / "epoch0100_seed0"', self.code)
        self.assertIn('_READY.json', self.code)

    def test_the_judge_is_the_real_data_only_c1(self):
        self.assertIn('"classifier_df585" / "checkpoints" / "C1_seed2" / "best.pt"', self.code)


if __name__ == "__main__":
    unittest.main()
