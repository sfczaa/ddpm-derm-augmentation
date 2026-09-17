"""Gradio entry point for the Hugging Face Space.

Wraps the same `ClassifierService` the FastAPI deployment uses, so the Space and
the Render demo serve identical predictions from identical assets: the pinned
public checkpoint and the pinned synthetic gallery, both fetched by revision at
startup and hash-checked by the shared download script.

This entry point targets a Gradio Space on ZeroGPU hardware, which is why it
differs from the Render deployment. `@spaces.GPU` is there for that hardware,
not because the model needs a GPU: this 42.7 MB ResNet-18 serves single-image
inference on CPU, and stays on CPU deliberately so a between-call GPU
deallocation cannot strand the weights on a dead device.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Same pinned revisions the Render image builds from. Never float these: the
# demo's provenance claim is that it serves one specific published checkpoint.
MODEL_REPO = "sfczaa/ddpm-derm-c1-seed2"
MODEL_REVISION = "30b41486b5353f2a99aceecef2fc41b178c2697b"
DATASET_REPO = "sfczaa/ddpm-derm-synthetic-gallery"
DATASET_REVISION = "60b046e4c2ae77f1505e8c2b426c24763741c3a5"
GALLERY_ARCHIVE_SHA256 = "da3d582082323728e2b0558c27e26af124c683dacf336915d1212acd8abd0bc5"
ASSET_ROOT = Path(os.environ.get("DDPM_DERM_ASSET_ROOT", "/tmp/ddpm-derm-assets"))

try:  # only present in the Space runtime
    import spaces

    gpu_slot = spaces.GPU
except ImportError:  # local runs and tests
    def gpu_slot(func=None, **_kwargs):
        return func if func is not None else (lambda inner: inner)


def fetch_assets() -> None:
    """Download the pinned model and gallery, or fail before the UI is served."""
    if (ASSET_ROOT / "model" / "deploy_weights.pt").is_file():
        return
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-B", str(ROOT / "deploy" / "download_render_assets.py"),
            "--model-repo", MODEL_REPO,
            "--model-revision", MODEL_REVISION,
            "--dataset-repo", DATASET_REPO,
            "--dataset-revision", DATASET_REVISION,
            "--gallery-archive-sha256", GALLERY_ARCHIVE_SHA256,
            "--output-root", str(ASSET_ROOT),
        ],
        check=True,
    )


def stage_data_sentinel() -> None:
    """Give `ddpm_derm.config` a data directory before anything imports it.

    `config.py` runs `DATA_DIR = resolve_data_dir()` at *import* time and raises
    unless it finds a directory containing `manifests/class_to_idx.json`. The
    Render image satisfies this by copying the class map into `data/manifests/`
    in its Dockerfile; the Space has no build step, so it is staged here instead.
    Without this the Space crashes on startup with a FileNotFoundError that says
    nothing about the real cause.
    """
    manifests = ASSET_ROOT / "data" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "deploy" / "class_to_idx.json", manifests / "class_to_idx.json")
    os.environ["DDPM_DERM_DATA_DIR"] = str(ASSET_ROOT / "data")


fetch_assets()
stage_data_sentinel()

from ddpm_derm.deploy import (  # noqa: E402  (import after assets and data dir exist)
    DISCLAIMER,
    MAX_IMAGE_PIXELS,
    ClassifierService,
    DeploymentSettings,
)

SERVICE = ClassifierService(
    DeploymentSettings(
        model_path=ASSET_ROOT / "model" / "deploy_weights.pt",
        model_manifest_path=ASSET_ROOT / "model" / "model_manifest.json",
        class_map_path=ROOT / "deploy" / "class_to_idx.json",
        gallery_dir=ASSET_ROOT / "gallery" / "epoch0100_seed0",
    )
)
HEALTH = SERVICE.health()


@gpu_slot(duration=20)
def classify(image: Image.Image):
    if image is None:
        raise gr.Error("Upload a dermatoscopic image first.")
    width, height = image.size
    if width * height > MAX_IMAGE_PIXELS:
        raise gr.Error(
            f"Image is too large ({width}x{height}). The limit is "
            f"{MAX_IMAGE_PIXELS:,} pixels."
        )
    result = SERVICE.predict(image.convert("RGB"))
    return {
        item["class_name"]: item["probability"] for item in result["probabilities"]
    }


def gallery_images():
    return [(str(item.path), item.image_id) for item in SERVICE.gallery(limit=24)]


ATTRIBUTION = """
**Educational portfolio demonstration only. Not for diagnosis or treatment.**
The model can be wrong, and its training data and population coverage are limited.

Non-commercial use only. HAM10000 Dataset &copy; ViDIR Group, Department of
Dermatology, Medical University of Vienna, distributed with the ISIC 2018 data
under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Cite
Tschandl, Rosendahl &amp; Kittler, *Scientific Data* 5, 180161 (2018),
<https://doi.org/10.1038/sdata.2018.161>. Project changes include a fixed
lesion-level split, resizing, classifier training, and generation of the derived
synthetic gallery. No endorsement by the dataset creators is implied.
"""

with gr.Blocks(title="HAM10000 classifier portfolio demo") as demo:
    gr.Markdown("# HAM10000 skin-lesion classifier &mdash; portfolio demo")
    gr.Markdown(
        f"Serving **{HEALTH['variant']} seed {HEALTH['seed']}**, selected on "
        f"{HEALTH['selection_metric']} among the C1 seeds &mdash; never on test "
        f"performance. Gallery version `{HEALTH['gallery_version']}`."
    )
    gr.Markdown(f"> {DISCLAIMER}")

    with gr.Tab("Classify"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Dermatoscopic image")
                submit = gr.Button("Classify", variant="primary")
            output = gr.Label(num_top_classes=7, label="Class probabilities")
        submit.click(classify, inputs=image_input, outputs=output)

    with gr.Tab("Synthetic gallery"):
        gr.Markdown(
            "24 of the 500 DDPM-generated `df` images used by the C4 condition. "
            "These are generated, not real patient images. The published "
            "experiment found they do not beat duplicating the 85 real `df` "
            "images; see the project README."
        )
        gr.Gallery(value=gallery_images(), columns=6, height="auto")

    gr.Markdown(ATTRIBUTION)

if __name__ == "__main__":
    demo.launch()
