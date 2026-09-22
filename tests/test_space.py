"""Static checks for the Hugging Face Space entry point.

The Space entry point targets the Gradio SDK on ZeroGPU hardware. Everything
guarded here is a mistake that would only surface after a Space build, or worse,
would build fine while quietly serving something other than the published artifacts:

  - the pinned asset revisions drifting away from the ones the Render image
    builds from, which would break the claim that both demos serve the same
    checkpoint;
  - the SDK reverting to `docker`, which this entry point is not built for;
  - the asset paths not matching the layout `download_render_assets.py`
    actually writes;
  - the requirements pinning the CPU-only torch index, which is correct for
    Render and wrong for a ZeroGPU Space;
  - the CC BY-NC 4.0 attribution going missing from a public page.

The inference path itself is exercised against the real pinned assets rather
than mocked, so it is not restated here.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "deploy" / "space"
APP = SPACE / "app.py"
REQUIREMENTS = SPACE / "requirements.txt"
SPACE_README = SPACE / "README.md"
DOCKERFILE_RENDER = ROOT / "Dockerfile.render"


class SpaceAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = APP.read_text(encoding="utf-8")
        cls.requirements = REQUIREMENTS.read_text(encoding="utf-8")
        cls.readme = SPACE_README.read_text(encoding="utf-8")
        cls.dockerfile = DOCKERFILE_RENDER.read_text(encoding="utf-8")

    def test_app_compiles(self):
        compile(self.app, str(APP), "exec")

    def test_pinned_assets_match_the_render_image(self):
        # Both demos must serve the same published checkpoint and gallery. If
        # these drift, the Space's provenance table becomes a false claim.
        for name, arg in (
            ("MODEL_REPO", "HF_MODEL_REPO"),
            ("MODEL_REVISION", "HF_MODEL_REVISION"),
            ("DATASET_REPO", "HF_DATASET_REPO"),
            ("DATASET_REVISION", "HF_DATASET_REVISION"),
            ("GALLERY_ARCHIVE_SHA256", "HF_GALLERY_ARCHIVE_SHA256"),
        ):
            with self.subTest(setting=name):
                in_app = re.search(rf'(?m)^{name} = "([^"]+)"', self.app)
                in_docker = re.search(rf"(?m)^ARG {arg}=(\S+)", self.dockerfile)
                self.assertIsNotNone(in_app, f"{name} missing from app.py")
                self.assertIsNotNone(in_docker, f"{arg} missing from Dockerfile.render")
                self.assertEqual(in_app.group(1), in_docker.group(1))

    def test_asset_paths_match_what_the_download_script_writes(self):
        # download_render_assets.py writes model/deploy_weights.pt and
        # gallery/<version>/; an app expecting best.pt would fail at startup.
        self.assertIn('ASSET_ROOT / "model" / "deploy_weights.pt"', self.app)
        self.assertIn('ASSET_ROOT / "model" / "model_manifest.json"', self.app)
        self.assertIn('ASSET_ROOT / "gallery" / "epoch0100_seed0"', self.app)
        self.assertNotIn('"best.pt"', self.app)

    def test_assets_are_fetched_before_the_service_is_built(self):
        self.assertLess(
            self.app.index("fetch_assets()"),
            self.app.index("from ddpm_derm.deploy import"),
        )
        self.assertLess(
            self.app.index("from ddpm_derm.deploy import"),
            self.app.index("SERVICE = ClassifierService("),
        )

    def test_data_sentinel_is_staged_before_the_deploy_import(self):
        # ddpm_derm.config runs DATA_DIR = resolve_data_dir() at *import* time
        # and raises unless it finds manifests/class_to_idx.json. The Render
        # image satisfies this with a Dockerfile COPY; the Space has no build
        # step, so app.py must stage it itself. Verified in an isolated copy of
        # the Space layout: without this the import raises FileNotFoundError,
        # with it the import succeeds. Ordering is the whole point, so it is
        # asserted rather than assumed.
        self.assertIn("def stage_data_sentinel()", self.app)
        self.assertIn('os.environ["DDPM_DERM_DATA_DIR"]', self.app)
        self.assertIn('manifests / "class_to_idx.json"', self.app)
        call = self.app.index("stage_data_sentinel()", self.app.index("fetch_assets()"))
        self.assertLess(call, self.app.index("from ddpm_derm.deploy import"))
        self.assertLess(
            self.app.index("def stage_data_sentinel()"),
            self.app.index("from ddpm_derm.deploy import"),
        )

    def test_readme_declares_a_gradio_space(self):
        # The entry point is a Gradio app, not a Docker Space.
        self.assertRegex(self.readme, r"(?m)^sdk: gradio$")
        self.assertRegex(self.readme, r"(?m)^app_file: app\.py$")
        self.assertNotRegex(self.readme, r"(?m)^sdk: docker$")

    def test_readme_links_the_hub_repositories_it_serves(self):
        self.assertIn("sfczaa/ddpm-derm-c1-seed2", self.readme)
        self.assertIn("sfczaa/ddpm-derm-synthetic-gallery", self.readme)

    def test_requirements_suit_zerogpu_not_the_render_cpu_build(self):
        self.assertRegex(self.requirements, r"(?m)^spaces>=")
        self.assertRegex(self.requirements, r"(?m)^gradio>=")
        self.assertRegex(self.requirements, r"(?m)^torch>=")
        # Render installs from the CPU wheel index; the ZeroGPU Space needs
        # the CUDA build.
        self.assertNotIn("download.pytorch.org/whl/cpu", self.requirements)

    def test_zerogpu_decorator_degrades_when_the_package_is_absent(self):
        # The Space runtime provides `spaces`; local runs and tests do not, and
        # an import error there would make the module unimportable anywhere else.
        block = re.search(
            r"try:.*?except ImportError:.*?\n(?=\n\ndef fetch_assets)",
            self.app,
            re.S,
        )
        self.assertIsNotNone(block, "the guarded spaces import is missing")
        namespace: dict = {}
        exec(block.group(0), namespace)
        gpu_slot = namespace["gpu_slot"]

        @gpu_slot(duration=20)
        def with_kwargs(value):
            return value + 1

        @gpu_slot
        def bare(value):
            return value + 2

        self.assertEqual(with_kwargs(1), 2)
        self.assertEqual(bare(1), 3)

    def test_public_page_carries_the_non_commercial_attribution(self):
        for name, text in (("README", self.readme), ("app", self.app)):
            with self.subTest(page=name):
                self.assertIn("CC BY-NC 4.0", text)
                self.assertIn("ViDIR Group", text)
                self.assertIn("10.1038/sdata.2018.161", text)
                self.assertIn("Non-commercial use only", text)

    def test_ui_states_what_is_served_and_how_it_was_selected(self):
        # The deploy candidate was chosen on validation df F1, never on test.
        # A page that omits that invites the opposite reading.
        self.assertIn("selection_metric", self.app)
        self.assertIn("never on test", self.app)
        self.assertIn("Not for diagnosis or treatment", self.readme)

    def test_readme_does_not_overstate_the_synthetic_result(self):
        prose = " ".join(self.readme.split())
        self.assertIn("No significance test was performed", prose)
        self.assertIn("do not establish a reliable benefit", prose)


if __name__ == "__main__":
    unittest.main()
