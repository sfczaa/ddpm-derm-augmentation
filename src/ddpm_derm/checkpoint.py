"""Restricted checkpoint loading, including legacy NumPy RNG state."""

import numpy as np
import torch


def load_checkpoint(path, map_location="cpu"):
    # Historical trainer checkpoints contain an MT19937 uint32 array and may
    # contain NumPy float64 metric scalars. Permit only these NumPy types and
    # constructors; never fall back to unrestricted pickle loading.
    core = np._core if hasattr(np, "_core") else np.core
    allowed = [np.ndarray, np.dtype, type(np.dtype("uint32")), type(np.dtype("float64"))]
    for module in ("numpy.core.multiarray", "numpy._core.multiarray"):
        allowed.extend([
            (core.multiarray._reconstruct, module + "._reconstruct"),
            (core.multiarray.scalar, module + ".scalar"),
        ])
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location=map_location, weights_only=True)
