"""Manifest reading and C0/C1/C4 training-frame construction.

Pure pandas / numpy: no torch, so this module (and the metrics module) can be
exercised by the local smoke test without a deep-learning stack.

Split integrity is NOT re-derived here. The lesion_id group split under
``data/manifests`` is fixed; we only read it. C4 keeps the train split
untouched and appends DDPM-generated df rows from an explicitly supplied
manifest (``load_generated_manifest``); val/test are never augmented.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd

from . import config

REQUIRED_COLUMNS = ["image_path", "label_idx", "dx", "lesion_id", "image_id"]
VALID_SPLITS = ("train", "val", "test")

# Schema sample_ddpm.py writes for generated rows; ``source`` marks provenance.
GENERATED_REQUIRED_COLUMNS = REQUIRED_COLUMNS + ["source"]
GENERATED_SOURCE = "synthetic"


def load_split(split: str) -> pd.DataFrame:
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    path = config.MANIFESTS_DIR / f"{split}.csv"
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Row count per class name, ordered by the canonical class index."""
    counts = frame["label_idx"].value_counts().to_dict()
    return {config.IDX_TO_CLASS[i]: int(counts.get(i, 0)) for i in range(config.NUM_CLASSES)}


def oversample_class(
    frame: pd.DataFrame, class_idx: int, target_count: int, seed: int = 42
) -> pd.DataFrame:
    """Duplicate rows of ``class_idx`` (deterministically) up to ``target_count``.

    Rows are only ever duplicated from within ``frame`` (the train split), so no
    val/test image is ever introduced. If the class already has >= target_count
    rows the frame is returned unchanged.
    """
    cls_rows = frame[frame["label_idx"] == class_idx]
    n = len(cls_rows)
    if n == 0:
        raise ValueError(f"class index {class_idx} has no rows to oversample")
    if target_count <= n:
        return frame.copy()

    extra_needed = target_count - n
    rng = np.random.default_rng(seed)
    # full repeats + a deterministic remainder sample
    full_repeats = extra_needed // n
    remainder = extra_needed % n
    parts = [cls_rows] * (full_repeats + 1)  # +1 keeps the originals
    if remainder:
        idx = rng.choice(cls_rows.index.to_numpy(), size=remainder, replace=False)
        parts.append(cls_rows.loc[idx])
    duplicated = pd.concat(parts, ignore_index=True)

    others = frame[frame["label_idx"] != class_idx]
    out = pd.concat([others, duplicated], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _is_nonportable(path_str: str) -> bool:
    """True for absolute or root-anchored paths in either OS convention.

    Generated manifests must stay portable when the folder moves between
    machines (Colab ``/content``, Drive, local disk), so anchored paths are
    rejected rather than resolved.
    """
    s = str(path_str)
    return (
        PurePosixPath(s).is_absolute()
        or PureWindowsPath(s).is_absolute()
        or s.replace("\\", "/").startswith("/")
    )


def load_generated_manifest(
    manifest_path: str | Path, root: str | Path | None = None
) -> pd.DataFrame:
    """Load and validate a manifest of DDPM-generated df rows for C4.

    ``image_path`` entries must be *relative*; they are resolved against
    ``root`` (default: the manifest's own directory) and rewritten to absolute
    paths in the returned frame, so a manifest+images folder keeps working
    after being moved as a unit. Every row must be labelled df, carry
    ``source == "synthetic"`` and point at an existing file; anything else
    raises instead of being silently dropped.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"generated manifest not found: {manifest_path}")
    gen = pd.read_csv(manifest_path)

    missing_cols = [c for c in GENERATED_REQUIRED_COLUMNS if c not in gen.columns]
    if missing_cols:
        raise ValueError(f"{manifest_path} is missing expected columns: {missing_cols}")
    if len(gen) == 0:
        raise ValueError(f"{manifest_path} has no rows")

    bad_label = gen[
        (gen["dx"] != config.TARGET_CLASS)
        | (gen["label_idx"] != config.TARGET_CLASS_IDX)
    ]
    if len(bad_label):
        raise ValueError(
            f"{manifest_path}: {len(bad_label)} row(s) not labelled "
            f"'{config.TARGET_CLASS}'/idx {config.TARGET_CLASS_IDX}, e.g. "
            f"{bad_label[['image_path', 'dx', 'label_idx']].head(3).to_dict('records')}"
        )
    bad_source = gen[gen["source"] != GENERATED_SOURCE]
    if len(bad_source):
        raise ValueError(
            f"{manifest_path}: {len(bad_source)} row(s) with source != "
            f"{GENERATED_SOURCE!r}; refusing rows of unknown origin"
        )
    dupes = gen["image_path"].duplicated()
    if dupes.any():
        raise ValueError(
            f"{manifest_path}: {int(dupes.sum())} duplicated image_path row(s)"
        )

    nonportable = [p for p in gen["image_path"] if _is_nonportable(p)]
    if nonportable:
        raise ValueError(
            f"{manifest_path}: image_path must be relative to the manifest's "
            f"directory (or an explicit root) so the folder stays portable; "
            f"got anchored path(s) e.g. {nonportable[:3]}"
        )
    base = (Path(root) if root is not None else manifest_path.parent).resolve()
    resolved = [base / p for p in gen["image_path"]]
    missing = [str(p) for p in resolved if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{manifest_path}: {len(missing)} referenced image(s) missing under "
            f"{base}, e.g. {missing[:3]}"
        )
    out = gen.copy()
    out["image_path"] = [str(p) for p in resolved]
    return out


def build_classifier_frame(
    variant: str,
    split: str = "train",
    df_target_count: int = 585,
    seed: int = 42,
    limit: int | None = None,
    generated_manifest: str | Path | None = None,
    generated_root: str | Path | None = None,
) -> pd.DataFrame:
    """Return the training frame for a classifier variant.

    C0: original imbalanced split (baseline).
    C1: real df oversampled (duplicated) to ``df_target_count`` total rows.
    C4: untouched train split plus generated df rows from
        ``generated_manifest`` (required; no default output directory is
        consulted). The manifest must provide exactly ``df_target_count``
        minus the real train df count rows.

    C1 and C4 must use the same ``df_target_count`` so the two differ only in
    where the extra df rows come from. The default matches the agreed C4
    composition: 585 = 85 real train df + 500 generated.
    """
    variant = variant.upper()
    if variant != "C4" and (generated_manifest is not None or generated_root is not None):
        raise ValueError(
            f"generated_manifest/generated_root only apply to C4, not {variant!r}"
        )
    frame = load_split(split)

    if variant == "C0":
        out = frame
    elif variant == "C1":
        out = oversample_class(frame, config.TARGET_CLASS_IDX, df_target_count, seed)
    elif variant == "C4":
        if split != "train":
            raise ValueError(f"C4 only augments the train split, got split={split!r}")
        if generated_manifest is None:
            raise ValueError(
                "C4 requires generated_manifest= pointing at the synthetic-df "
                "manifest CSV; no default output directory is consulted."
            )
        gen = load_generated_manifest(generated_manifest, root=generated_root)
        n_real = int((frame["label_idx"] == config.TARGET_CLASS_IDX).sum())
        needed = df_target_count - n_real
        if needed <= 0:
            raise ValueError(
                f"C4 df_target_count={df_target_count} must exceed the "
                f"{n_real} real train df rows"
            )
        if len(gen) != needed:
            raise ValueError(
                f"C4 needs exactly {needed} generated df rows "
                f"(df_target_count {df_target_count} - {n_real} real) but "
                f"{generated_manifest} provides {len(gen)}"
            )
        real = frame.copy()
        real["source"] = "real"
        out = pd.concat([real, gen], ignore_index=True)
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        raise ValueError(f"unknown variant {variant!r} (expected C0, C1 or C4)")

    if limit is not None:
        out = out.sample(n=min(limit, len(out)), random_state=seed).reset_index(drop=True)
    return out.reset_index(drop=True)


def inverse_freq_class_weights(frame: pd.DataFrame) -> np.ndarray:
    """Inverse-frequency weights (for an optional C3 weighted loss)."""
    counts = np.array([class_counts(frame)[config.IDX_TO_CLASS[i]] for i in range(config.NUM_CLASSES)], dtype=float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (config.NUM_CLASSES * counts)
    return weights
