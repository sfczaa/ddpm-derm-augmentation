"""Notebook entry points use restricted checkpoints and isolated run versions."""

import ast
import json
import re
import subprocess
import tempfile
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
                self.assertIn("require_training_runtime()", code)
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

    def test_balanced_validation_does_not_create_the_formal_run(self):
        source = dict((name, code) for name, _, code in notebooks())["colab_balanced_ddpm_classifier_validate.ipynb"]
        tree = ast.parse(source)
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id in {"VALIDATION_ROOT", "FORMAL_RUN_DIR"}]
        self.assertEqual(len(assignments), 2)
        with tempfile.TemporaryDirectory() as temporary:
            def create(path):
                path.mkdir(parents=True)
                return path
            namespace = {"RUNNER_DOWNSTREAM_ROOT": Path(temporary),
                         "RUN_VERSION": "c4_sqrt_balanced_v1_safe_v2",
                         "ensure_runner_tree": create}
            exec(compile(ast.Module(body=assignments, type_ignores=[]), "paths", "exec"), namespace)
            self.assertTrue(namespace["VALIDATION_ROOT"].is_dir())
            self.assertFalse(namespace["FORMAL_RUN_DIR"].exists())
            self.assertNotIn(namespace["FORMAL_RUN_DIR"], namespace["VALIDATION_ROOT"].parents)

    def test_balanced_formal_gate_rejects_validation_from_the_old_identity(self):
        code = dict((name, code) for name, _, code in notebooks())["colab_balanced_ddpm_classifier_train.ipynb"]
        gate = code[code.index("SOURCE_MANIFEST_SHA256 ="):code.index("protected =")]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            version = "c4_sqrt_balanced_v1_safe_v2"
            record_path = root / "validation_runs" / version / "validation_record.json"
            record_path.parent.mkdir(parents=True)
            record = {"status": "passed", "formal_training_started": False,
                      "run_version": version, "git_commit": "a" * 40,
                      "candidate_manifest_sha256": "candidate", "source_manifest_sha256": "source",
                      "runner_output_root": str(root)}
            namespace = {"LOCAL_DATA_DIR": root, "sha256": lambda path: "source", "json": json,
                         "RUNNER_DOWNSTREAM_ROOT": root, "RUNNER_OUTPUTS_DIR": root,
                         "RUN_VERSION": version, "EXPECTED_COMMIT": "a" * 40,
                         "CANDIDATE_SHA256": "candidate"}
            record_path.write_text(json.dumps(record), encoding="utf-8")
            exec(compile(gate, "validation-gate", "exec"), namespace.copy())
            for key, wrong in (("run_version", "c4_sqrt_balanced_v1"),
                               ("git_commit", "b" * 40), ("source_manifest_sha256", "changed")):
                with self.subTest(key=key):
                    record_path.write_text(json.dumps({**record, key: wrong}), encoding="utf-8")
                    with self.assertRaises(AssertionError):
                        exec(compile(gate, "validation-gate", "exec"), namespace.copy())


if __name__ == "__main__":
    unittest.main()
