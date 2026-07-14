from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.main import create_app  # noqa: E402
from ddpm_derm.deploy import (  # noqa: E402
    DISCLAIMER,
    MAX_UPLOAD_BYTES,
    DeploymentSettings,
    GalleryItem,
    UploadValidationError,
    decode_uploaded_image,
    load_deployment_assets,
)

CLASS_MAP = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3,
             "mel": 4, "nv": 5, "vasc": 6}


def png_bytes() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (12, 8), (90, 60, 30)).save(out, format="PNG")
    return out.getvalue()


class FakeService:
    def __init__(self, gallery_path: Path):
        self.item = GalleryItem("synthetic_df_0000", gallery_path)

    def health(self):
        return {"status": "ok", "device": "cpu", "variant": "C1", "seed": 2,
                "selection_metric": "validation df F1",
                "gallery_version": "epoch0100_seed0"}

    def gallery(self, limit=24):
        return [self.item][:limit]

    def gallery_path(self, image_id):
        if image_id != self.item.image_id:
            raise KeyError(image_id)
        return self.item.path

    def predict(self, image):
        self.last_mode = image.mode
        probabilities = [0.04, 0.06, 0.10, 0.12, 0.14, 0.46, 0.08]
        return {
            "predicted_class": "nv",
            "probabilities": [
                {"class_name": name, "probability": probabilities[index]}
                for name, index in CLASS_MAP.items()
            ],
        }


class UploadTests(unittest.TestCase):
    def test_valid_png_is_loaded_as_rgb(self):
        image = decode_uploaded_image("lesion.png", "image/png", png_bytes())
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (12, 8))

    def test_mime_mismatch_fails_with_415(self):
        with self.assertRaises(UploadValidationError) as caught:
            decode_uploaded_image("lesion.jpg", "image/png", png_bytes())
        self.assertEqual(caught.exception.status_code, 415)

    def test_damaged_image_fails_with_400(self):
        with self.assertRaises(UploadValidationError) as caught:
            decode_uploaded_image("lesion.png", "image/png", b"not an image")
        self.assertEqual(caught.exception.status_code, 400)

    def test_oversized_upload_fails_with_413(self):
        with self.assertRaises(UploadValidationError) as caught:
            decode_uploaded_image(
                "lesion.png", "image/png", b"x" * (MAX_UPLOAD_BYTES + 1)
            )
        self.assertEqual(caught.exception.status_code, 413)


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "best.pt"
        self.model.write_bytes(b"test checkpoint fixture")
        (self.root / "class_to_idx.json").write_text(
            json.dumps(CLASS_MAP), encoding="utf-8"
        )
        gallery = self.root / "gallery" / "epoch0100_seed0"
        images = gallery / "images"
        images.mkdir(parents=True)
        rows = []
        for index in range(2):
            image_id = f"synthetic_df_{index:04d}"
            Image.new("RGB", (8, 8), (index * 20, 40, 60)).save(images / f"{image_id}.png")
            rows.append(
                f"images/{image_id}.png,3,df,synthetic,{image_id},synthetic\n"
            )
        (gallery / "synthetic_df.csv").write_text(
            "image_path,label_idx,dx,lesion_id,image_id,source\n" + "".join(rows),
            encoding="utf-8",
        )
        metadata = {"epoch": 100, "seed": 0, "num_steps": 50, "eta": 0.0,
                    "weights": "ema_state_dict", "class_name": "df", "class_idx": 3}
        (gallery / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (gallery / "_READY.json").write_text(
            json.dumps({"rows": 2, "metadata": metadata}), encoding="utf-8"
        )
        manifest = {
            "checkpoint_sha256": hashlib.sha256(self.model.read_bytes()).hexdigest(),
            "class_to_idx": CLASS_MAP,
            "gallery": {"version": "epoch0100_seed0", "rows": 2, **metadata},
        }
        (self.root / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.settings = DeploymentSettings(
            model_path=self.model,
            model_manifest_path=self.root / "model_manifest.json",
            class_map_path=self.root / "class_to_idx.json",
            gallery_dir=gallery,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_assets_validate_with_published_gallery(self):
        assets = load_deployment_assets(self.settings)
        self.assertEqual(len(assets.gallery_items), 2)
        self.assertEqual(assets.class_to_idx, CLASS_MAP)

    def test_missing_checkpoint_fails_loudly(self):
        self.model.unlink()
        with self.assertRaises(FileNotFoundError):
            load_deployment_assets(self.settings)

    def test_unversioned_gallery_directory_fails_loudly(self):
        wrong_dir = self.settings.gallery_dir.parent / "unversioned"
        shutil.copytree(self.settings.gallery_dir, wrong_dir)
        wrong = DeploymentSettings(
            model_path=self.settings.model_path,
            model_manifest_path=self.settings.model_manifest_path,
            class_map_path=self.settings.class_map_path,
            gallery_dir=wrong_dir,
        )
        with self.assertRaises(ValueError):
            load_deployment_assets(wrong)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temp.name) / "gallery.png"
        self.image_path.write_bytes(png_bytes())
        self.service = FakeService(self.image_path)

    def tearDown(self):
        self.temp.cleanup()

    def test_predict_returns_seven_probabilities_and_disclaimer(self):
        with TestClient(create_app(service=self.service)) as client:
            response = client.post(
                "/api/predict",
                files={"image": ("lesion.png", png_bytes(), "image/png")},
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["predicted_class"], "nv")
        self.assertEqual(len(payload["probabilities"]), 7)
        self.assertEqual(self.service.last_mode, "RGB")
        self.assertEqual(payload["disclaimer"], DISCLAIMER)

    def test_invalid_upload_returns_clear_4xx(self):
        with TestClient(create_app(service=self.service)) as client:
            response = client.post(
                "/api/predict",
                files={"image": ("lesion.txt", b"bad", "text/plain")},
            )
        self.assertEqual(response.status_code, 415)
        self.assertIn("Only", response.json()["detail"])

    def test_openapi_exposes_prediction_schema(self):
        app = create_app(service=self.service)
        schema = app.openapi()
        self.assertIn("/api/predict", schema["paths"])
        self.assertIn("PredictionResponse", schema["components"]["schemas"])

    def test_ui_contains_dataset_attribution_and_license(self):
        with TestClient(create_app(service=self.service)) as client:
            response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("HAM10000 Dataset", response.text)
        self.assertIn("CC BY-NC 4.0", response.text)
        self.assertIn("10.1038/sdata.2018.161", response.text)
        self.assertIn("non-commercial portfolio demo", response.text)


if __name__ == "__main__":
    unittest.main()
