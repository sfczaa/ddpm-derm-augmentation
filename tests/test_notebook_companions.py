"""Keep percent-format companions aligned with their canonical notebooks."""

import ast
import json
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
COMPANIONS = (
    "colab_balanced_ddpm_classifier_train",
    "colab_balanced_ddpm_classifier_validate",
    "colab_classifier_baseline",
)


def notebook_code_cells(name):
    notebook = json.loads((NOTEBOOKS / f"{name}.ipynb").read_text(encoding="utf-8"))
    cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        transformed = []
        for line in "".join(cell.get("source", [])).splitlines(keepends=True):
            stripped = line.lstrip()
            if stripped.startswith("!"):
                indent = line[: len(line) - len(stripped)]
                command = stripped[1:].rstrip("\r\n")
                transformed.append(f"{indent}get_ipython().system({command!r})\n")
            else:
                if stripped.startswith("%"):
                    raise AssertionError(f"unsupported notebook magic: {stripped.rstrip()}")
                transformed.append(line)
        cells.append("".join(transformed))
    return cells


def companion_code_cells(name):
    cells = []
    active = None
    for line in (NOTEBOOKS / f"{name}.py").read_text(encoding="utf-8").splitlines(keepends=True):
        marker = re.match(r"^# %% \[(markdown|\d+)\](?: .*)?$", line.rstrip("\r\n"))
        if marker:
            if active is not None:
                cells.append("".join(active))
            active = [] if marker[1].isdigit() else None
        elif active is not None:
            active.append(line)
    if active is not None:
        cells.append("".join(active))
    return cells


def ast_identity(source):
    return ast.dump(ast.parse(source), include_attributes=False)


class NotebookCompanionTests(unittest.TestCase):
    def test_code_cells_match_canonical_notebooks(self):
        for name in COMPANIONS:
            with self.subTest(companion=name):
                canonical = notebook_code_cells(name)
                companion = companion_code_cells(name)
                self.assertEqual(len(companion), len(canonical))
                self.assertEqual(
                    [ast_identity(cell) for cell in companion],
                    [ast_identity(cell) for cell in canonical],
                )

    def test_companions_keep_safe_pins_and_namespaces(self):
        sources = {
            name: (NOTEBOOKS / f"{name}.py").read_text(encoding="utf-8")
            for name in COMPANIONS
        }
        for name, source in sources.items():
            with self.subTest(companion=name):
                match = re.search(r'^EXPECTED_(?:GIT_)?COMMIT = "([0-9a-f]{40})"', source, re.M)
                self.assertIsNotNone(match)
                subprocess.run(
                    ["git", "cat-file", "-e", f"{match[1]}^{{commit}}"],
                    cwd=ROOT,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.assertIn("_safe_v2", source)
                self.assertIn("require_training_runtime()", source)
                self.assertNotIn("a6fc90c8f946c0d11e3e5e22d65131a092be361a", source)
                self.assertNotIn('BRANCH = "balanced-ddpm-exploration"', source)
                self.assertNotIn("weights_only=False", source)

        for name in COMPANIONS[:2]:
            self.assertIn("from ddpm_derm.checkpoint import load_checkpoint", sources[name])

        baseline = sources["colab_classifier_baseline"]
        self.assertIn('RUN_MODE = "fresh"', baseline)
        self.assertIn("run_root = SHARED_PROJECT_DIR / 'outputs' / 'notebook_runs' / RUN_VERSION", baseline)
        self.assertIn("INPUT_OUTPUTS_DIR = str(SHARED_PROJECT_DIR / 'outputs')", baseline)
        self.assertIn("OUTPUTS_DIR = str(run_root)", baseline)


if __name__ == "__main__":
    unittest.main()
