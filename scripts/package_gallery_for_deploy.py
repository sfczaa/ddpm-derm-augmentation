"""Package the validated gallery as one deployment-only ZIP archive."""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm.deploy import DeploymentSettings, load_deployment_assets  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "outputs/deploy/C1_seed2/deploy_weights.pt",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/deploy/C1_seed2/model_manifest.json",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=ROOT / "deploy/class_to_idx.json",
    )
    parser.add_argument(
        "--gallery",
        type=Path,
        default=ROOT / "outputs/synthetic_df/epoch0100_seed0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/deploy/gallery_epoch0100_seed0.zip",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = DeploymentSettings(
        model_path=args.model.resolve(),
        model_manifest_path=args.manifest.resolve(),
        class_map_path=args.class_map.resolve(),
        gallery_dir=args.gallery.resolve(),
    )
    assets = load_deployment_assets(settings)
    if len(assets.gallery_items) != 500:
        raise ValueError(f"expected 500 validated gallery images, found {len(assets.gallery_items)}")

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"gallery deployment archive already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")

    gallery_root = settings.gallery_dir.resolve()
    files = [
        gallery_root / "_READY.json",
        gallery_root / "metadata.json",
        gallery_root / "synthetic_df.csv",
        *(item.path for item in assets.gallery_items),
    ]
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in files:
            arcname = Path(gallery_root.name) / path.relative_to(gallery_root)
            archive.write(path, arcname=arcname.as_posix())

    temporary.replace(output)
    with zipfile.ZipFile(output) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"gallery archive CRC failed for {bad_member}")
        if len(archive.namelist()) != 503:
            raise ValueError(f"expected 503 archive files, found {len(archive.namelist())}")

    print("GALLERY_ARCHIVE_OK")
    print(f"archive={output}")
    print(f"files=503")
    print(f"size_bytes={output.stat().st_size}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
