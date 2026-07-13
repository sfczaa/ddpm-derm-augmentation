"""Validate and publish a sampled synthetic-df folder. Torch-free.

A "synthetic dir" is what ``sample_ddpm`` writes for one run:

    <dir>/images/synthetic_df_0000.png ...
    <dir>/synthetic_df.csv     # image_path RELATIVE to <dir> (portable)
    <dir>/metadata.json        # checkpoint / epoch / weights / n / seed / steps
    <dir>/nn_check.png         # df only

Formal flow (Colab): sample onto fast local ``/content`` staging, validate
there, copy the whole folder to a NEW versioned Drive directory, re-validate
on the destination, and only then write a ``_READY.json`` completion marker.
Consumers (the C4 classifier run) must treat a folder without ``_READY.json``
as incomplete. Nothing is overwritten or deleted here: an existing destination
is refused, and a failed copy/validation leaves the partial folder WITHOUT the
marker — remove it manually before retrying.

Usage
-----
    # validate only (e.g. the local staging dir)
    python -m ddpm_derm.publish_synthetic --src /content/synthetic_df_epoch0100_seed0
    # validate + publish + re-validate + _READY.json
    python -m ddpm_derm.publish_synthetic \
        --src /content/synthetic_df_epoch0100_seed0 \
        --dest /path/to/outputs/synthetic_df/epoch0100_seed0
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import config, manifests

MANIFEST_NAME = "synthetic_df.csv"
METADATA_NAME = "metadata.json"
READY_MARKER = "_READY.json"


def validate_synthetic_dir(
    dir_path: str | Path,
    expect_n: int = 500,
    expect_epoch: int = 100,
    expect_seed: int = 0,
    expect_steps: int = 50,
    expect_weights: str = "ema_state_dict",
) -> dict:
    """Validate one synthetic dir; raise on any problem, else return a summary.

    Reuses ``manifests.load_generated_manifest`` (relative-only portable paths,
    df-only labels, source == synthetic, no duplicate image_path, every file
    present) and adds: exact row count, unique image_id, and a metadata.json
    whose epoch / n / seed / num_steps / weights match the expected formal run.
    """
    d = Path(dir_path)
    if not d.is_dir():
        raise FileNotFoundError(f"synthetic dir not found: {d}")

    gen = manifests.load_generated_manifest(d / MANIFEST_NAME)
    if len(gen) != expect_n:
        raise ValueError(
            f"{d / MANIFEST_NAME}: expected exactly {expect_n} rows, got {len(gen)}"
        )
    dup_ids = gen["image_id"].duplicated()
    if dup_ids.any():
        raise ValueError(
            f"{d / MANIFEST_NAME}: {int(dup_ids.sum())} duplicated image_id row(s)"
        )

    meta_path = d / METADATA_NAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing {METADATA_NAME} in {d}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = {
        "epoch": expect_epoch,
        "n": expect_n,
        "seed": expect_seed,
        "num_steps": expect_steps,
        "weights": expect_weights,
        "class_name": config.TARGET_CLASS,
    }
    mismatches = {k: {"got": meta.get(k), "expected": v}
                  for k, v in expected.items() if meta.get(k) != v}
    if mismatches:
        raise ValueError(f"{meta_path}: metadata mismatch: {mismatches}")

    return {"dir": str(d), "rows": int(len(gen)), "missing_files": 0,
            "metadata": meta, "ready": (d / READY_MARKER).is_file()}


def publish_synthetic_dir(src: str | Path, dest: str | Path, **expect) -> dict:
    """Validate ``src``, copy it whole to the new ``dest``, re-validate there,
    then (and only then) write the ``_READY.json`` marker. Never overwrites."""
    src, dest = Path(src), Path(dest)
    if (src / READY_MARKER).exists():
        raise ValueError(
            f"staging {src} already contains {READY_MARKER}; copying it could mark "
            f"an unvalidated destination as complete — remove the marker first")
    local = validate_synthetic_dir(src, **expect)
    print(f"[publish] staging validated: {local['rows']} rows @ {src}")

    if dest.exists():
        raise FileExistsError(
            f"destination {dest} already exists; refusing to overwrite a published "
            f"set — use a new versioned directory")
    shutil.copytree(src, dest)
    print(f"[publish] copied whole folder -> {dest}")

    try:
        final = validate_synthetic_dir(dest, **expect)
    except Exception:
        print(f"[publish] REVALIDATION FAILED: {dest} is partial or inconsistent; "
              f"no {READY_MARKER} was written. Remove the folder manually and retry.")
        raise

    ready = {
        "published_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(src),
        "rows": final["rows"],
        "metadata": final["metadata"],
    }
    (dest / READY_MARKER).write_text(json.dumps(ready, indent=2), encoding="utf-8")
    print(f"[publish] re-validated on destination; wrote {dest / READY_MARKER}")
    return final


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate (and optionally publish) a synthetic-df folder.")
    p.add_argument("--src", required=True,
                   help="Local staging dir written by sample_ddpm; validated first.")
    p.add_argument("--dest", default=None,
                   help="New versioned destination dir (e.g. on Drive). "
                        "Omit to only validate --src.")
    p.add_argument("--expect-n", type=int, default=500)
    p.add_argument("--expect-epoch", type=int, default=100)
    p.add_argument("--expect-seed", type=int, default=0)
    p.add_argument("--expect-steps", type=int, default=50)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    expect = dict(expect_n=args.expect_n, expect_epoch=args.expect_epoch,
                  expect_seed=args.expect_seed, expect_steps=args.expect_steps)
    if args.dest is None:
        summary = validate_synthetic_dir(args.src, **expect)
        print(f"[validate] OK: {summary['rows']} rows, missing_files="
              f"{summary['missing_files']}, metadata epoch "
              f"{summary['metadata']['epoch']} @ {args.src}")
    else:
        summary = publish_synthetic_dir(args.src, args.dest, **expect)
        print(f"[publish] DONE: {summary['rows']} rows published to {args.dest}")


if __name__ == "__main__":
    main()
