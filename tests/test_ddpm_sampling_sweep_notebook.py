"""Static checks for sampling-sweep syntax and optional GH_TOKEN authentication.

Credentials use a child-process environment, never the repository URL,
saved remote or process-wide environment. Tests use no real credentials.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_NAME = "colab_ddpm_sampling_sweep_diagnostic.ipynb"
NOTEBOOK = ROOT / "notebooks" / NOTEBOOK_NAME


class DdpmSamplingSweepGhTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_notebook_cells_are_clean_and_compile(self):
        for index, cell in enumerate(self.notebook["cells"]):
            self.assertFalse(cell.get("outputs"), f"cell {index} has stored outputs")
            if cell["cell_type"] == "code":
                self.assertIsNone(
                    cell.get("execution_count"), f"cell {index} has an execution count"
                )
                compile("".join(cell["source"]), f"{NOTEBOOK_NAME}:cell-{index}", "exec")

    def test_clone_uses_gh_token_secret_and_fails_loud_when_absent(self):
        self.assertIn("from google.colab import drive, userdata", self.code)
        self.assertIn('token = userdata.get("GH_TOKEN")', self.code)
        self.assertIn(
            'assert token and len(token) > 20, '
            '"Colab Secret GH_TOKEN with read access to this repo is required"',
            self.code,
        )
        # the anonymous clone this fix replaces must be gone
        self.assertNotIn(
            'subprocess.run(["git", "clone", REPO_URL, str(CODE_DIR)], check=True)\n',
            self.code,
        )

    def test_clone_is_noninteractive_scoped_to_child_env_not_repo_url(self):
        self.assertIn(
            'REPO_URL = "https://github.com/sfczaa/ddpm-derm-augmentation.git"',
            self.code,
        )
        self.assertIn('basic_credential = base64.b64encode', self.code)
        self.assertIn('clone_env = os.environ.copy()', self.code)
        self.assertIn('clone_env["GIT_TERMINAL_PROMPT"] = "0"', self.code)
        self.assertIn('clone_env["GIT_CONFIG_COUNT"] = "1"', self.code)
        self.assertIn(
            'clone_env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"',
            self.code,
        )
        self.assertIn(
            'clone_env["GIT_CONFIG_VALUE_0"] = "Authorization: Basic " + basic_credential',
            self.code,
        )
        # clone command carries no token, so a CalledProcessError cannot leak it
        self.assertIn(
            'subprocess.run(["git", "clone", REPO_URL, str(CODE_DIR)], '
            "check=True, env=clone_env)",
            self.code,
        )
        # override lives only on the child clone_env, never process-wide or persisted git config
        self.assertNotIn('os.environ["GIT_CONFIG', self.code)
        self.assertNotIn("git config --global", self.code)
        self.assertNotIn("git config --system", self.code)

    def test_credential_cleared_after_clone_and_remote_is_credential_free(self):
        self.assertIn('clone_env["GIT_CONFIG_VALUE_0"] = ""', self.code)
        self.assertIn("token = basic_credential = None", self.code)
        self.assertIn("del token, basic_credential, clone_env", self.code)
        self.assertIn(
            'remote = subprocess.check_output(["git", "-C", str(CODE_DIR), '
            '"remote", "get-url", "origin"], text=True).strip()',
            self.code,
        )
        self.assertIn(
            'assert "@" not in remote and "x-access-token" not in remote, '
            '"clone URL must not embed a credential"',
            self.code,
        )

    def test_no_token_literal_or_secret_leakage(self):
        self.assertNotIn("ghp_", self.code)
        self.assertNotIn("github_pat_", self.code)
        self.assertNotIn("@github.com", self.code)
        self.assertNotRegex(self.code, r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")
        self.assertNotIn("print(token", self.code)
        self.assertNotIn("print(basic_credential", self.code)
        self.assertNotIn("token}", self.code)

    def test_pinned_checkout_and_safety_boundaries_preserved(self):
        self.assertRegex(self.code, r'EXPECTED_GIT_COMMIT = "[0-9a-f]{40}"')
        self.assertIn('EXPECTED_GIT_COMMIT != "REPLACE_AFTER_PUSH"', self.code)
        self.assertIn(
            'subprocess.run(["git", "-C", str(CODE_DIR), "checkout", "--detach", '
            'EXPECTED_GIT_COMMIT], check=True)',
            self.code,
        )
        self.assertIn(
            'assert commit == EXPECTED_GIT_COMMIT and not status, '
            '"clone must be a clean detached checkout of the pinned commit"',
            self.code,
        )
        self.assertIn(
            'SHARED_PROJECT_DIR = Path("/content/drive/MyDrive/ddpm-derm-augmentation")',
            self.code,
        )
        self.assertIn(
            'DDPM_CHECKPOINT_DIR = SHARED_PROJECT_DIR / "outputs" / "ddpm" / "checkpoints"',
            self.code,
        )
        self.assertIn(
            'JUDGE_CHECKPOINT = SHARED_PROJECT_DIR / "outputs" / "classifier_df585" '
            '/ "checkpoints" / "C1_seed2" / "best.pt"',
            self.code,
        )
        self.assertIn('def run_stream(command, cwd=CODE_DIR, heartbeat=60):', self.code)
        self.assertIn(
            'assert not record_path.exists(), f"a record already exists; move it aside deliberately: {record_path}"',
            self.code,
        )
        self.assertIn('assert torch.cuda.is_available()', self.code)
        self.assertIn("Read-only against the fixed split", "".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "markdown"
        ))


if __name__ == "__main__":
    unittest.main()
