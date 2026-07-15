"""Project paths and environment overrides.

Data-directory resolution order:
1. ``DDPM_DERM_DATA_DIR``
2. ``<project_root>/data``
3. ``<project_root>/../data``

A candidate must contain ``manifests/class_to_idx.json``.
``DDPM_DERM_OUTPUTS_DIR`` overrides the output directory.
"""

from __future__ import annotations

import os
from pathlib import Path

# <root>/src/ddpm_derm/config.py -> parents[2] == <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_MANIFEST_SENTINEL = Path("manifests") / "class_to_idx.json"


def _looks_like_data_dir(path: Path) -> bool:
    return (path / _MANIFEST_SENTINEL).is_file()


def resolve_data_dir() -> Path:
    """Return the HAM10000 data directory, raising a clear error if not found."""
    env = os.environ.get("DDPM_DERM_DATA_DIR")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(PROJECT_ROOT / "data")
    candidates.append(PROJECT_ROOT.parent / "data")

    for cand in candidates:
        if _looks_like_data_dir(cand):
            return cand.resolve()

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate the HAM10000 data directory (expected to contain "
        f"{_MANIFEST_SENTINEL.as_posix()}).\nTried:\n  {tried}\n"
        "Set the DDPM_DERM_DATA_DIR environment variable to point at it."
    )


DATA_DIR = resolve_data_dir()
MANIFESTS_DIR = DATA_DIR / "manifests"

OUTPUTS_DIR = Path(
    os.environ.get("DDPM_DERM_OUTPUTS_DIR", PROJECT_ROOT / "outputs")
).expanduser()

# --- output layout (all under OUTPUTS_DIR; point that at Drive on Colab) ------
# Nothing here is created on import; train scripts mkdir what they need. A
# committed skeleton with .gitkeep files documents the intended homes.
CLASSIFIER_DIR = OUTPUTS_DIR / "classifier"
CLASSIFIER_CKPT_DIR = CLASSIFIER_DIR / "checkpoints"       # per-run best.pt / last.pt
CLASSIFIER_RESULTS_DIR = CLASSIFIER_DIR / "results"        # results_*.json
DDPM_CKPT_DIR = OUTPUTS_DIR / "ddpm" / "checkpoints"       # stage 2
DDPM_SAMPLES_DIR = OUTPUTS_DIR / "ddpm" / "samples"        # stage 2
SYNTHETIC_DF_DIR = OUTPUTS_DIR / "synthetic_df"            # stage 2: images that feed C4
EXPLORATORY_BALANCED_DDPM_DIR = (
    OUTPUTS_DIR / "exploratory_balanced_ddpm"
)                                                           # post-stage-4 experiment
FIGURES_DIR = OUTPUTS_DIR / "figures"                      # plots for the report


def classifier_run_ckpt_dir(variant: str, seed: int) -> Path:
    """Per-run checkpoint dir, e.g. outputs/classifier/checkpoints/C0_seed0/."""
    return CLASSIFIER_CKPT_DIR / f"{variant}_seed{seed}"


# --- dataset constants -------------------------------------------------------
CLASS_TO_IDX = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
CLASS_NAMES = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]
NUM_CLASSES = len(CLASS_TO_IDX)

TARGET_CLASS = "df"
TARGET_CLASS_IDX = CLASS_TO_IDX[TARGET_CLASS]


def resolve_image_path(rel_path: str) -> Path:
    """Manifest ``image_path`` values are relative to the data dir.

    C4 frames carry already-resolved absolute paths for generated rows (see
    ``manifests.load_generated_manifest``); those pass through unchanged.
    """
    p = Path(rel_path)
    if p.is_absolute():
        return p
    return DATA_DIR / p
