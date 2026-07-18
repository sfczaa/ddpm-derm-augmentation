"""Torch-free identity and resume guards for classifier runs."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Mapping


IDENTITY_SCHEMA_VERSION = 2
FROZEN_COCA_CHECKPOINT_FORMAT = "frozen_backbone_head_only_v1"
FULL_MODEL_CHECKPOINT_FORMAT = "full_model_v1"


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of file contents without depending on its path."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project_root: str | Path) -> str | None:
    """Return the current Git commit, or ``None`` outside a Git checkout."""
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_run_identity(
    *,
    run_label: str | None,
    variant: str,
    seed: int,
    epochs: int,
    img_size: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    df_target_count: int,
    pretrained: bool,
    limit: int | None,
    candidate_manifest: str | Path | None,
    source_split: str,
    source_manifest: str | Path,
    source_git_commit: str | None,
    model_identity: Mapping[str, Any] | None = None,
    fixed_split_identity: str | None = None,
    shared_root_uuid: str | None = None,
    formal_output_identity: str | None = None,
    run_version: str | None = None,
    model_selection_metric: str = "validation_df_f1_strict_improvement",
    class_mapping: Mapping[str, int] | None = None,
    experiment_candidate_sha256: str | None = None,
    checkpoint_format: str | None = None,
) -> dict[str, Any]:
    """Build a portable identity for checkpoint compatibility checks."""
    if source_split != "train":
        raise ValueError(
            f"classifier training source_split must be 'train', got {source_split!r}"
        )
    if variant == "C4" and candidate_manifest is None:
        raise ValueError("C4 run identity requires a candidate manifest")
    if variant != "C4" and candidate_manifest is not None:
        raise ValueError(f"candidate manifest is not valid for {variant}")
    candidate_sha256 = (
        sha256_file(candidate_manifest) if candidate_manifest is not None else None
    )
    if experiment_candidate_sha256 is not None:
        if (
            len(experiment_candidate_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in experiment_candidate_sha256)
        ):
            raise ValueError("experiment candidate SHA-256 must be 64 lowercase hex chars")
        if candidate_sha256 is not None and candidate_sha256 != experiment_candidate_sha256:
            raise ValueError(
                "candidate manifest SHA-256 does not match the fixed experiment hash"
            )
        candidate_sha256 = experiment_candidate_sha256

    source_manifest_sha256 = sha256_file(source_manifest)
    if model_identity is None:
        model_identity = {
            "arch": "resnet18",
            "model_name": "torchvision_resnet18",
            "pretrained_tag": "IMAGENET1K_V1" if pretrained else None,
            "freeze_mode": "trainable",
            "preprocessing_identity": None,
            "input_resolution": [img_size, img_size],
            "total_parameter_count": None,
            "trainable_parameter_count": None,
            "open_clip_torch_version": None,
            "torch_version": None,
        }
    if checkpoint_format is None:
        checkpoint_format = (
            FROZEN_COCA_CHECKPOINT_FORMAT
            if model_identity.get("arch") == "coca_vit_b32"
            else FULL_MODEL_CHECKPOINT_FORMAT
        )
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "run_label": run_label,
        "variant": variant,
        "seed": int(seed),
        "fixed_config": {
            "epochs": int(epochs),
            "img_size": int(img_size),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "df_target_count": int(df_target_count),
            "pretrained": bool(pretrained),
            "limit": None if limit is None else int(limit),
        },
        "candidate_manifest_sha256": candidate_sha256,
        "source_split": source_split,
        "source_manifest_sha256": source_manifest_sha256,
        "git_commit": source_git_commit,
        "run_version": run_version if run_version is not None else run_label,
        "model_identity": dict(model_identity),
        "fixed_split_identity": fixed_split_identity or source_manifest_sha256,
        "shared_root_uuid": shared_root_uuid,
        "formal_output_identity": formal_output_identity,
        "model_selection_metric": model_selection_metric,
        "class_mapping": None if class_mapping is None else dict(class_mapping),
        "checkpoint_format": checkpoint_format,
    }


def require_matching_resume_identity(
    saved: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    """Reject resume before state loading when any identity field differs."""
    legacy_required = (
        "schema_version",
        "run_label",
        "variant",
        "seed",
        "fixed_config",
        "candidate_manifest_sha256",
        "source_split",
        "source_manifest_sha256",
        "git_commit",
    )
    if saved.get("schema_version") == 1:
        if current.get("model_identity", {}).get("arch") != "resnet18":
            raise ValueError("legacy checkpoint cannot resume a non-ResNet run")
        legacy_current = dict(current)
        legacy_current["schema_version"] = 1
        mismatches = [
            f"{key}: saved={saved.get(key)!r} current={legacy_current.get(key)!r}"
            for key in legacy_required
            if key not in saved or saved.get(key) != legacy_current.get(key)
        ]
        if mismatches:
            raise ValueError(
                "classifier resume identity mismatch: " + "; ".join(mismatches)
            )
        return
    required = legacy_required + (
        "run_version",
        "model_identity",
        "fixed_split_identity",
        "shared_root_uuid",
        "formal_output_identity",
        "model_selection_metric",
        "class_mapping",
        "checkpoint_format",
    )
    missing = [key for key in required if key not in saved]
    if missing:
        raise ValueError(
            "checkpoint lacks strict classifier run identity fields: "
            f"{missing}; refusing an unverifiable resume"
        )
    mismatches = [
        f"{key}: saved={saved[key]!r} current={current.get(key)!r}"
        for key in required
        if saved[key] != current.get(key)
    ]
    if mismatches:
        raise ValueError("classifier resume identity mismatch: " + "; ".join(mismatches))


def require_matching_legacy_config(
    saved: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    """Keep legacy C0/C1 resume usable while still rejecting config drift.

    Old C4 checkpoints have no candidate hash and therefore cannot use this
    compatibility path safely.
    """
    keys = (
        "variant",
        "seed",
        "epochs",
        "batch_size",
        "img_size",
        "lr",
        "weight_decay",
        "df_target_count",
        "no_pretrained",
        "limit",
    )
    missing = [key for key in keys if key not in saved]
    if missing:
        raise ValueError(
            f"legacy checkpoint config is missing fields {missing}; refusing resume"
        )
    mismatches = [
        f"{key}: saved={saved[key]!r} current={current.get(key)!r}"
        for key in keys
        if saved[key] != current.get(key)
    ]
    if mismatches:
        raise ValueError("legacy classifier resume config mismatch: " + "; ".join(mismatches))


def require_isolated_output_dir(
    output_dir: str | Path, exploratory_root: str | Path
) -> Path:
    """Require an exploratory output to stay under its dedicated root."""
    output = Path(output_dir).expanduser().resolve()
    root = Path(exploratory_root).expanduser().resolve()
    if output == root or root not in output.parents:
        raise ValueError(
            f"exploratory classifier output must be a child of {root}, got {output}"
        )
    return output
