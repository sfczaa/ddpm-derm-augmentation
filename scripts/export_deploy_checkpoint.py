"""Export a minimal inference checkpoint without changing the formal checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "outputs/classifier_df585/checkpoints/C1_seed2/best.pt",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "deploy/model_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/deploy/C1_seed2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_checkpoint = output_dir / "deploy_weights.pt"
    output_manifest = output_dir / "model_manifest.json"

    if not source.is_file():
        raise FileNotFoundError(f"formal checkpoint not found: {source}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"base model manifest not found: {manifest_path}")
    if output_checkpoint.exists() or output_manifest.exists():
        raise FileExistsError(
            f"deployment export already exists; inspect before replacing: {output_dir}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_hash = sha256(source)
    expected_hash = str(manifest.get("checkpoint_sha256", "")).lower()
    if source_hash != expected_hash:
        raise ValueError(
            f"formal checkpoint SHA-256 mismatch: expected {expected_hash}, got {source_hash}"
        )

    import torch

    from ddpm_derm.checkpoint import load_checkpoint
    checkpoint = load_checkpoint(source, map_location="cpu")
    required = ("model_state_dict", "config", "class_to_idx")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"formal checkpoint is missing required keys: {missing}")

    payload = {key: checkpoint[key] for key in required}
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_checkpoint.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_checkpoint)

    deployment_hash = sha256(output_checkpoint)
    derived_manifest = dict(manifest)
    derived_manifest.update(
        {
            "checkpoint_sha256": deployment_hash,
            "checkpoint_file": output_checkpoint.name,
            "source_checkpoint_sha256": source_hash,
            "artifact_role": "deployment-only checkpoint derived from the formal checkpoint",
        }
    )
    output_manifest.write_text(
        json.dumps(derived_manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("DEPLOY_EXPORT_OK")
    print(f"checkpoint={output_checkpoint}")
    print(f"size_bytes={output_checkpoint.stat().st_size}")
    print(f"sha256={deployment_hash}")
    print(f"manifest={output_manifest}")


if __name__ == "__main__":
    main()
