"""Deployment asset validation, upload decoding, and classifier inference.

The validation and upload helpers are torch-free. Torch is imported only when
``ClassifierService`` is constructed, so local tests can exercise the safety
boundary without loading the model.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DISCLAIMER = (
    "Educational and portfolio demonstration only. This output is not medical "
    "advice and must not be used for diagnosis or treatment. The model can be "
    "wrong, and HAM10000 and the training population have limited representation."
)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_FORMAT_BY_MIME = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}


class UploadValidationError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class DeploymentSettings:
    model_path: Path
    model_manifest_path: Path
    class_map_path: Path
    gallery_dir: Path

    @classmethod
    def from_env(cls) -> "DeploymentSettings":
        return cls(
            model_path=_project_path(os.environ.get(
                "DDPM_DERM_MODEL_PATH",
                "outputs/classifier_df585/checkpoints/C1_seed2/best.pt",
            )),
            model_manifest_path=_project_path(os.environ.get(
                "DDPM_DERM_MODEL_MANIFEST", "deploy/model_manifest.json"
            )),
            class_map_path=_project_path(os.environ.get(
                "DDPM_DERM_CLASS_MAP_PATH", "deploy/class_to_idx.json"
            )),
            gallery_dir=_project_path(os.environ.get(
                "DDPM_DERM_GALLERY_DIR",
                "outputs/synthetic_df/epoch0100_seed0",
            )),
        )


@dataclass(frozen=True)
class GalleryItem:
    image_id: str
    path: Path


@dataclass(frozen=True)
class DeploymentAssets:
    settings: DeploymentSettings
    model_manifest: dict[str, Any]
    class_to_idx: dict[str, int]
    gallery_metadata: dict[str, Any]
    gallery_items: tuple[GalleryItem, ...]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_deployment_assets(
    settings: DeploymentSettings | None = None,
) -> DeploymentAssets:
    """Validate every required deployment asset or raise immediately."""
    settings = settings or DeploymentSettings.from_env()
    if not settings.model_path.is_file():
        raise FileNotFoundError(f"deploy checkpoint not found: {settings.model_path}")

    model_manifest = _read_json(settings.model_manifest_path, "model manifest")
    class_to_idx = _read_json(settings.class_map_path, "class mapping")
    if class_to_idx != model_manifest.get("class_to_idx"):
        raise ValueError("class mapping does not match deploy model manifest")
    indices = sorted(class_to_idx.values())
    if indices != list(range(len(class_to_idx))):
        raise ValueError(f"class mapping indices must be contiguous from zero: {indices}")

    expected_hash = str(model_manifest.get("checkpoint_sha256", "")).lower()
    if len(expected_hash) != 64:
        raise ValueError("model manifest has no valid checkpoint_sha256")
    actual_hash = _sha256(settings.model_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )

    gallery_root = settings.gallery_dir.resolve()
    ready = _read_json(gallery_root / "_READY.json", "gallery ready marker")
    metadata = _read_json(gallery_root / "metadata.json", "gallery metadata")
    if ready.get("metadata") != metadata:
        raise ValueError("gallery _READY metadata does not match metadata.json")

    expected_gallery = model_manifest.get("gallery")
    if not isinstance(expected_gallery, dict):
        raise ValueError("model manifest has no gallery specification")
    if gallery_root.name != expected_gallery.get("version"):
        raise ValueError(
            f"gallery directory version mismatch: expected "
            f"{expected_gallery.get('version')!r}, got {gallery_root.name!r}"
        )
    for key in ("epoch", "seed", "num_steps", "eta", "weights", "class_name", "class_idx"):
        if metadata.get(key) != expected_gallery.get(key):
            raise ValueError(
                f"gallery metadata mismatch for {key}: "
                f"expected {expected_gallery.get(key)!r}, got {metadata.get(key)!r}"
            )

    manifest_path = gallery_root / "synthetic_df.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"gallery manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_rows = int(expected_gallery.get("rows", 0))
    if len(rows) != expected_rows or ready.get("rows") != expected_rows:
        raise ValueError(
            f"gallery row count mismatch: CSV={len(rows)}, "
            f"_READY={ready.get('rows')}, expected={expected_rows}"
        )

    items: list[GalleryItem] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for row_number, row in enumerate(rows, start=2):
        image_id = row.get("image_id", "")
        rel = Path(row.get("image_path", ""))
        if not image_id or image_id in seen_ids:
            raise ValueError(f"invalid or duplicate gallery image_id at CSV row {row_number}")
        if rel.is_absolute():
            raise ValueError(f"gallery image path must be relative at CSV row {row_number}")
        path = (gallery_root / rel).resolve()
        if not path.is_relative_to(gallery_root) or path in seen_paths:
            raise ValueError(f"unsafe or duplicate gallery path at CSV row {row_number}")
        if row.get("source") != "synthetic" or row.get("dx") != "df":
            raise ValueError(f"unverified gallery provenance at CSV row {row_number}")
        if int(row.get("label_idx", -1)) != class_to_idx["df"]:
            raise ValueError(f"wrong gallery label at CSV row {row_number}")
        if not path.is_file():
            raise FileNotFoundError(f"gallery image missing at CSV row {row_number}: {path}")
        seen_ids.add(image_id)
        seen_paths.add(path)
        items.append(GalleryItem(image_id=image_id, path=path))

    return DeploymentAssets(
        settings=settings,
        model_manifest=model_manifest,
        class_to_idx={str(k): int(v) for k, v in class_to_idx.items()},
        gallery_metadata=metadata,
        gallery_items=tuple(items),
    )


def decode_uploaded_image(
    filename: str | None,
    content_type: str | None,
    data: bytes,
) -> Image.Image:
    """Validate an in-memory upload and return a loaded RGB image."""
    suffix = Path(filename or "").suffix.lower()
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    expected_mime = _MIME_BY_SUFFIX.get(suffix)
    if expected_mime is None or mime != expected_mime:
        raise UploadValidationError(
            "Only .jpg/.jpeg, .png, or .webp files with a matching image MIME type are allowed.",
            status_code=415,
        )
    if not data:
        raise UploadValidationError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadValidationError(
            f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
            status_code=413,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                actual_format = probe.format
                if probe.width * probe.height > MAX_IMAGE_PIXELS:
                    raise UploadValidationError("Image dimensions are too large.", 413)
                probe.verify()
            if actual_format != _FORMAT_BY_MIME[mime]:
                raise UploadValidationError(
                    "Image contents do not match the filename and MIME type.", 415
                )
            with Image.open(io.BytesIO(data)) as image:
                rgb = image.convert("RGB")
                rgb.load()
                return rgb
    except UploadValidationError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise UploadValidationError("The uploaded image is invalid or damaged.") from exc


class ClassifierService:
    """Load the trusted checkpoint once and provide CPU/GPU inference."""

    def __init__(self, settings: DeploymentSettings | None = None):
        self.assets = load_deployment_assets(settings)

        import torch
        from torchvision import transforms

        from .model import build_model

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from ddpm_derm.checkpoint import load_checkpoint
        checkpoint = load_checkpoint(
            self.assets.settings.model_path,
            map_location=self._device
        )
        for key in ("model_state_dict", "config", "class_to_idx"):
            if key not in checkpoint:
                raise ValueError(f"deploy checkpoint is missing {key!r}")
        if checkpoint["class_to_idx"] != self.assets.class_to_idx:
            raise ValueError("checkpoint class_to_idx does not match deploy mapping")

        manifest = self.assets.model_manifest
        checkpoint_config = checkpoint["config"]
        expected = {
            "variant": manifest["variant"],
            "seed": manifest["seed"],
            "img_size": manifest["image_size"],
            "df_target_count": manifest["df_target_count"],
        }
        mismatches = {
            key: (checkpoint_config.get(key), value)
            for key, value in expected.items()
            if checkpoint_config.get(key) != value
        }
        if mismatches:
            raise ValueError(f"checkpoint config does not match deploy manifest: {mismatches}")

        self._model = build_model(
            num_classes=len(self.assets.class_to_idx),
            arch=manifest["architecture"],
            pretrained=False,
        ).to(self._device)
        self._model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self._model.eval()
        preprocessing = manifest["preprocessing"]
        self._transform = transforms.Compose(
            [
                transforms.Resize((manifest["image_size"], manifest["image_size"])),
                transforms.ToTensor(),
                transforms.Normalize(
                    tuple(preprocessing["mean"]), tuple(preprocessing["std"])
                ),
            ]
        )
        self._class_names = [
            name for name, _ in sorted(self.assets.class_to_idx.items(), key=lambda x: x[1])
        ]

    def health(self) -> dict[str, Any]:
        manifest = self.assets.model_manifest
        return {
            "status": "ok",
            "device": str(self._device),
            "variant": manifest["variant"],
            "seed": manifest["seed"],
            "selection_metric": "validation df F1",
            "gallery_version": manifest["gallery"]["version"],
        }

    def gallery(self, limit: int = 24) -> list[GalleryItem]:
        return list(self.assets.gallery_items[:limit])

    def gallery_path(self, image_id: str) -> Path:
        for item in self.assets.gallery_items:
            if item.image_id == image_id:
                return item.path
        raise KeyError(image_id)

    def predict(self, image: Image.Image) -> dict[str, Any]:
        tensor = self._transform(image).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            probabilities = self._torch.softmax(self._model(tensor), dim=1)[0].cpu().tolist()
        values = [
            {"class_name": name, "probability": float(probabilities[index])}
            for index, name in enumerate(self._class_names)
        ]
        predicted = max(values, key=lambda item: item["probability"])["class_name"]
        return {"predicted_class": predicted, "probabilities": values}
