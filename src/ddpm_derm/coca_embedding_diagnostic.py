"""Frozen-CoCa embedding diagnostic for real, synthetic, and validation df images."""

from __future__ import annotations

import argparse
import hashlib
import os
from itertools import combinations
from pathlib import Path
from typing import Mapping

import numpy as np

from . import config, manifests


DIAGNOSTIC_VERSION = "v1_frozen_coca_df_embeddings"
GROUP_ORDER = ("real_train_df", "synthetic_df", "validation_df")
EXPECTED_GROUP_COUNTS = {
    "real_train_df": 85,
    "synthetic_df": 500,
    "validation_df": 14,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_df_groups(
    generated_manifest: str | Path,
    *,
    generated_root: str | Path | None = None,
) -> dict[str, object]:
    """Load only train/validation df images and the fixed synthetic candidate."""
    train = manifests.load_split("train")
    validation = manifests.load_split("val")
    synthetic = manifests.load_generated_manifest(
        generated_manifest, root=generated_root
    )
    groups = {
        "real_train_df": train[train["dx"] == config.TARGET_CLASS].copy(),
        "synthetic_df": synthetic.copy(),
        "validation_df": validation[validation["dx"] == config.TARGET_CLASS].copy(),
    }
    counts = {name: len(frame) for name, frame in groups.items()}
    if counts != EXPECTED_GROUP_COUNTS:
        raise ValueError(
            f"unexpected df diagnostic group counts: {counts}; "
            f"expected {EXPECTED_GROUP_COUNTS}"
        )
    for name, frame in groups.items():
        if frame["image_id"].isna().any() or frame["image_id"].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} contains missing image_id values")
        if frame["image_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} contains duplicated image_id values")
        if not frame["dx"].eq(config.TARGET_CLASS).all():
            raise ValueError(f"{name} contains non-df rows")
    return groups


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("embeddings must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embeddings contain zero-norm rows")
    return values / norms


def _distance_summary(distances: np.ndarray) -> dict[str, float]:
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 1 or not len(distances) or not np.isfinite(distances).all():
        raise ValueError("distance summary requires finite one-dimensional values")
    return {
        "count": int(len(distances)),
        "min": float(np.min(distances)),
        "p10": float(np.quantile(distances, 0.10)),
        "median": float(np.median(distances)),
        "mean": float(np.mean(distances)),
        "p90": float(np.quantile(distances, 0.90)),
        "max": float(np.max(distances)),
    }


def analyze_embedding_groups(
    embeddings: Mapping[str, np.ndarray],
) -> dict[str, object]:
    if tuple(embeddings) != GROUP_ORDER:
        raise ValueError(f"embedding groups must be ordered as {GROUP_ORDER}")
    normalized = {name: l2_normalize(embeddings[name]) for name in GROUP_ORDER}
    dimensions = {values.shape[1] for values in normalized.values()}
    if len(dimensions) != 1:
        raise ValueError("embedding groups have different feature dimensions")

    centroids = {
        name: l2_normalize(values.mean(axis=0, keepdims=True))[0]
        for name, values in normalized.items()
    }
    centroid_distances = {}
    for left, right in combinations(GROUP_ORDER, 2):
        similarity = float(np.clip(centroids[left] @ centroids[right], -1.0, 1.0))
        centroid_distances[f"{left}__{right}"] = {
            "cosine_similarity": similarity,
            "cosine_distance": 1.0 - similarity,
        }

    within_group = {}
    for name, values in normalized.items():
        similarities = np.clip(values @ centroids[name], -1.0, 1.0)
        within_group[name] = _distance_summary(1.0 - similarities)

    nearest_neighbor = {}
    for source in GROUP_ORDER:
        for target in GROUP_ORDER:
            if source == target:
                continue
            similarities = np.clip(normalized[source] @ normalized[target].T, -1.0, 1.0)
            nearest_neighbor[f"{source}_to_{target}"] = _distance_summary(
                1.0 - similarities.max(axis=1)
            )

    return {
        "group_counts": {
            name: int(normalized[name].shape[0]) for name in GROUP_ORDER
        },
        "feature_dimension": int(next(iter(dimensions))),
        "centroid_cosine_distances": centroid_distances,
        "distance_to_own_centroid": within_group,
        "cross_group_nearest_neighbor_cosine_distances": nearest_neighbor,
    }


def encode_group(model, frame, *, device, batch_size: int, num_workers: int):
    import torch

    from .dataset import build_dataloader

    loader = build_dataloader(
        frame,
        batch_size=batch_size,
        train=False,
        num_workers=num_workers,
        transform=model.eval_preprocess,
    )
    model.encoder.eval()
    batches = []
    with torch.no_grad():
        for images, _ in loader:
            features = model.encoder.encode_image(images.to(device, non_blocking=True))
            batches.append(features.detach().to(dtype=torch.float32).cpu().numpy())
    if not batches:
        raise ValueError("cannot encode an empty diagnostic group")
    return l2_normalize(np.concatenate(batches, axis=0)).astype(np.float32)


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise ValueError("embedding NPZ keys changed after save")
        for name, values in arrays.items():
            if saved[name].shape != values.shape or not np.array_equal(saved[name], values):
                raise ValueError(f"embedding NPZ verification failed for {name}")


def prepare_output_dir(path: str | Path) -> Path:
    """Accept a probed empty Drive directory, but never overwrite artifacts."""
    path = Path(path)
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"diagnostic output is not empty: {path}")
    else:
        path.mkdir(parents=True)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-manifest", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--shared-root-uuid", required=True)
    parser.add_argument("--v4-failure-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    import torch
    from importlib.metadata import version

    from . import coca_run
    from .model import build_model, model_identity

    args = parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    candidate_hash = sha256_file(args.generated_manifest)
    if candidate_hash != args.expected_candidate_sha256:
        raise ValueError(
            f"candidate SHA-256 mismatch: {candidate_hash} != "
            f"{args.expected_candidate_sha256}"
        )
    prepare_output_dir(args.output_dir)

    groups = build_df_groups(
        args.generated_manifest, generated_root=args.generated_root
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real CoCa diagnostic")
    model = build_model(
        arch="coca_vit_b32",
        freeze_backbone=True,
        coca_pretrained="laion2b_s13b_b90k",
    ).to(device)
    model.eval()
    if model.encoder.training or any(
        parameter.requires_grad for parameter in model.encoder.parameters()
    ):
        raise ValueError("CoCa encoder is not frozen in eval mode")

    embeddings = {
        name: encode_group(
            model,
            groups[name],
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        for name in GROUP_ORDER
    }
    analysis = analyze_embedding_groups(embeddings)
    embedding_path = args.output_dir / "df_embeddings.npz"
    _write_npz_atomic(embedding_path, embeddings)
    record = {
        "diagnostic_status": "COMPLETED",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "formal_training_started": False,
        "test_data_accessed": False,
        "git_commit": args.git_commit,
        "candidate_manifest_sha256": candidate_hash,
        "fixed_split_identity": sha256_file(config.MANIFESTS_DIR / "train.csv"),
        "validation_manifest_sha256": sha256_file(config.MANIFESTS_DIR / "val.csv"),
        "shared_root_uuid": args.shared_root_uuid,
        "v4_failure_sha256": args.v4_failure_sha256,
        "model_identity": model_identity(model, "coca_vit_b32", 224),
        "embedding_extractor": {
            "feature_source": "frozen_image_encoder_output_before_linear_head",
            "linear_head_used": False,
            "native_eval_preprocessing": True,
            "l2_normalized": True,
        },
        "dependency_versions": {
            "open_clip_torch": version("open_clip_torch"),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "embedding_artifact": {
            "path": str(embedding_path),
            "sha256": sha256_file(embedding_path),
        },
        "analysis": analysis,
        "interpretation_boundary": (
            "descriptive frozen-embedding diagnostic only; no test evaluation, "
            "model selection, classifier training, or formal-run authorization"
        ),
    }
    record_path = args.output_dir / "embedding_diagnostic.json"
    coca_run.write_json_atomic(record_path, record)
    print(record_path)
    print("EMBEDDING DIAGNOSTIC COMPLETED")
    print("formal_training_started=false")
    print("test_data_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
