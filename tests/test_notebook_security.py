"""Notebook entry points use restricted checkpoints and isolated run versions."""

import ast
import json
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def notebooks():
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        yield path.name, notebook, code


class NotebookSecurityTests(unittest.TestCase):
    def test_every_notebook_pins_code_with_the_restricted_loader(self):
        pins = set()
        for name, _, code in notebooks():
            with self.subTest(notebook=name):
                match = re.search(r'^EXPECTED_(?:GIT_)?COMMIT = "([0-9a-f]{40})"', code, re.M)
                self.assertIsNotNone(match)
                pins.add(match[1])
                self.assertNotIn('BRANCH = "balanced-ddpm-exploration"', code)
        for pin in pins:
            with self.subTest(pin=pin):
                source = subprocess.check_output(
                    ["git", "show", f"{pin}:src/ddpm_derm/checkpoint.py"],
                    cwd=ROOT, text=True,
                )
                self.assertIn("weights_only=True", source)
                self.assertNotIn("weights_only=False", source)

    def test_notebooks_do_not_request_unrestricted_torch_loading(self):
        for name, notebook, _ in notebooks():
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                code = "".join(cell["source"])
                if any(line.lstrip().startswith(("!", "%")) for line in code.splitlines()):
                    continue
                with self.subTest(notebook=name, cell=index):
                    tree = ast.parse(code)
                    for node in ast.walk(tree):
                        if not isinstance(node, ast.Call):
                            continue
                        if ast.unparse(node.func) == "torch.load":
                            values = {arg.arg: arg.value for arg in node.keywords}
                            self.assertIn("weights_only", values)
                            self.assertIsInstance(values["weights_only"], ast.Constant)
                            self.assertIs(values["weights_only"].value, True)

    def test_run_and_diagnostic_versions_have_separate_output_namespaces(self):
        for name, _, code in notebooks():
            with self.subTest(notebook=name):
                versions = re.findall(r'^(?:RUN_VERSION|DIAGNOSTIC_VERSION) = [\'"]([^\'"\n]+)[\'"]', code, re.M)
                self.assertTrue(versions)
                self.assertTrue(all(value.endswith("_safe_v2") for value in versions))
        sources = {name: code for name, _, code in notebooks()}
        for family in ("coca", "coca_v2_weighted", "coca_v3_inverse_frequency", "coca_v4_focal_inverse_frequency", "panderm_base_c1_finetune"):
            validation = sources[f"colab_{family}_validation.ipynb"]
            formal = sources[f"colab_{family}_classifier.ipynb"]
            version = lambda code: re.search(r'^RUN_VERSION = "([^"]+)"', code, re.M)[1]
            self.assertEqual(version(validation), version(formal))

    def test_historical_shared_root_identity_is_preserved(self):
        sources = {name: code for name, _, code in notebooks()}
        self.assertIn('run_version="v1")', sources["colab_coca_validation.ipynb"])
        for suffix in ("validation", "classifier"):
            code = sources[f"colab_panderm_base_c1_finetune_{suffix}.ipynb"]
            sentinel_calls = [line for line in code.splitlines() if "create_or_validate_sentinel(" in line or "require_shared_root_sentinel_identity(" in line]
            self.assertEqual(len(sentinel_calls), 2)
            self.assertTrue(all('run_version="v1_panderm_base_c1_finetune"' in line for line in sentinel_calls))


if __name__ == "__main__":
    unittest.main()
