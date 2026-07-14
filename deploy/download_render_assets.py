"""Download pinned public deployment assets and fail if their layout is wrong."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import stat
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


GALLERY_ARCHIVE = "gallery_epoch0100_seed0.zip"
GALLERY_VERSION = "epoch0100_seed0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--gallery-archive-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_gallery(archive_path: Path, gallery_root: Path) -> None:
    required = {
        f"{GALLERY_VERSION}/_READY.json",
        f"{GALLERY_VERSION}/metadata.json",
        f"{GALLERY_VERSION}/synthetic_df.csv",
    }
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("gallery deployment archive failed CRC validation")
        members = archive.infolist()
        names = [member.filename for member in members]
        pngs = [
            name
            for name in names
            if name.startswith(f"{GALLERY_VERSION}/images/") and name.endswith(".png")
        ]
        if len(names) != 503 or len(set(names)) != 503 or len(pngs) != 500:
            raise ValueError(
                f"gallery archive layout mismatch: files={len(names)}, "
                f"unique={len(set(names))}, pngs={len(pngs)}"
            )
        if not required.issubset(names):
            raise ValueError(f"gallery archive is missing: {sorted(required - set(names))}")

        root = gallery_root.resolve()
        for member in members:
            relative = Path(member.filename)
            mode = member.external_attr >> 16
            if (
                member.is_dir()
                or relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or relative.parts[0] != GALLERY_VERSION
                or stat.S_ISLNK(mode)
            ):
                raise ValueError(f"unsafe gallery archive member: {member.filename!r}")
            destination = (root / relative).resolve()
            if not destination.is_relative_to(root):
                raise ValueError(f"unsafe gallery extraction path: {member.filename!r}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def main() -> None:
    args = parse_args()
    model_dir = args.output_root / "model"
    gallery_root = args.output_root / "gallery"

    snapshot_download(
        repo_id=args.model_repo,
        repo_type="model",
        revision=args.model_revision,
        allow_patterns=["deploy_weights.pt", "model_manifest.json"],
        local_dir=model_dir,
    )
    archive_path = Path(
        hf_hub_download(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            revision=args.dataset_revision,
            filename=GALLERY_ARCHIVE,
        )
    )
    actual_archive_hash = sha256(archive_path)
    if actual_archive_hash != args.gallery_archive_sha256.lower():
        raise ValueError(
            f"gallery archive SHA-256 mismatch: expected "
            f"{args.gallery_archive_sha256.lower()}, got {actual_archive_hash}"
        )
    extract_gallery(archive_path, gallery_root)

    required = [
        model_dir / "deploy_weights.pt",
        model_dir / "model_manifest.json",
        gallery_root / GALLERY_VERSION / "_READY.json",
        gallery_root / GALLERY_VERSION / "metadata.json",
        gallery_root / GALLERY_VERSION / "synthetic_df.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"pinned deployment assets are incomplete: {missing}")

    images = list((gallery_root / GALLERY_VERSION / "images").glob("*.png"))
    if len(images) != 500:
        raise ValueError(f"expected 500 gallery PNGs, found {len(images)}")

    print(
        f"RENDER_ASSETS_OK model_revision={args.model_revision} "
        f"dataset_revision={args.dataset_revision} archive_sha256={actual_archive_hash} "
        f"gallery_pngs={len(images)}"
    )


if __name__ == "__main__":
    main()
