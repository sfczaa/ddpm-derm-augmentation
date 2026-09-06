"""Select the C4-filtered synthetic subset under the pre-registered rule.

Every part of the acceptance rule was fixed in C4_FILTERED_EXPERIMENT_DESIGN.md
section 4 before any outcome was seen, and this script is its only
implementation:

  judge      the C1 seed 2 ResNet-18, penultimate 512-d features, L2-normalised.
             C1 is trained on real images only, so it never saw the pool it is
             filtering. A C4 model would be circular and PanDerm is disqualified
             because its pretraining corpus cannot be shown to exclude HAM10000.
  distance   nearest neighbour into the 85 real train df *and their horizontal
             mirrors*, because DDPM training used RandomHorizontalFlip.
  threshold  max(val df -> train), the farthest of the 14 genuinely-new real df.
             Recomputed here from the val set every run, never hardcoded, so it
             tracks whichever pool and judge are actually passed in.
  selection  every image at or below the threshold. No top-N: a fixed count
             would change meaning with pool size and invites tuning.
  gate       below FILTERED_MINIMUM_ACCEPTED accepted images the condition is
             not run. The shortfall is written to the record and the accepted
             manifest is deliberately *not* written, so the training step that
             consumes it fails loud instead of training on a handful of images.

The judge and the distance come from the memorisation diagnostic rather than
being restated here, so the filter cannot drift away from the measurement that
motivated it.

The test split is never opened.

    python scripts/c4_filtered_select.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddpm_derm import manifests  # noqa: E402
from ddpm_memorization_diagnostic import (  # noqa: E402
    describe,
    embedding_features,
    load_real_data_judge,
    load_real_df,
    nn_distance,
    synthetic_provenance,
)


def load_candidate(manifest_path: Path, root: Path | None = None):
    """Candidate rows and their images, in manifest order.

    Reading the manifest rather than globbing the directory is what keeps the
    distance for row *i* attached to the image_id for row *i*, and guarantees
    the accepted manifest is a true subset of the candidate.
    """
    frame = manifests.load_generated_manifest(manifest_path, root=root)
    images = [Image.open(p).convert("RGB") for p in frame["image_path"]]
    return frame, images


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--candidate-manifest",
        default="outputs/synthetic_df/epoch0100_seed0/synthetic_df.csv",
        help="The published epoch-100 pool the formal C4 condition trains on.",
    )
    p.add_argument(
        "--judge-checkpoint",
        default="outputs/classifier_df585/checkpoints/C1_seed2/best.pt",
    )
    p.add_argument(
        "--accepted-out",
        default="outputs/synthetic_df/epoch0100_seed0/c4_filtered_accepted.csv",
        help="Written next to the candidate so its relative image_path stays valid.",
    )
    p.add_argument("--record-out", default="outputs/figures/c4_filtered_selection.json")
    args = p.parse_args()

    candidate_path = Path(args.candidate_manifest)
    print("[load] real train df, real val df, candidate pool  (test split untouched)")
    train_imgs = load_real_df("train")
    val_imgs = load_real_df("val")
    candidate, cand_imgs = load_candidate(candidate_path)
    images_dir = Path(candidate["image_path"].iloc[0]).parent
    print(
        f"  train df={len(train_imgs)}  val df={len(val_imgs)}  "
        f"candidate={len(cand_imgs)}  from {images_dir}"
    )

    judge_path = Path(args.judge_checkpoint)
    model, judge_cfg = load_real_data_judge(judge_path)
    img_size = int(judge_cfg.get("img_size", 128))
    print(
        f"[judge] {judge_path}  variant={judge_cfg.get('variant')} "
        f"seed={judge_cfg.get('seed')} img_size={img_size}  "
        "(real-data-only; never trained on synthetic)"
    )

    flipped = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in train_imgs]
    train_f = embedding_features(train_imgs, model, img_size)
    gallery = np.concatenate([train_f, embedding_features(flipped, model, img_size)])
    val_f = embedding_features(val_imgs, model, img_size)
    cand_f = embedding_features(cand_imgs, model, img_size)

    val_d = nn_distance(val_f, gallery)
    cand_d = nn_distance(cand_f, gallery)
    threshold = float(val_d.max())

    print("\n nearest-neighbour distance into the 85 real train df (+flips):")
    distances = {
        "val_to_train": describe("val df (14) -> train  [threshold source]", val_d),
        "candidate_to_train": describe("candidate -> train", cand_d),
    }
    print(
        f"\n[threshold] max(val df -> train) = {threshold!r}\n"
        "  'no farther from real df than the most unusual genuinely new real df'"
    )

    accepted_mask = cand_d <= threshold
    accepted_count = int(accepted_mask.sum())
    fraction = float(accepted_mask.mean())
    print(
        f"[accept] {accepted_count} of {len(cand_d)} images clear it "
        f"({fraction:.1%} of the pool)"
    )

    runnable = accepted_count >= manifests.FILTERED_MINIMUM_ACCEPTED
    record: dict = {
        "design": "C4_FILTERED_EXPERIMENT_DESIGN.md section 4",
        "selection_algorithm": manifests.FILTERED_SELECTION_ALGORITHM,
        "judge": {
            "checkpoint": str(judge_path),
            "variant": judge_cfg.get("variant"),
            "seed": judge_cfg.get("seed"),
            "img_size": img_size,
        },
        "flip_aware": True,
        "test_split_accessed": False,
        "candidate_manifest": str(candidate_path),
        "synthetic_provenance": synthetic_provenance(images_dir),
        "counts": {
            "train_df": len(train_imgs),
            "val_df": len(val_imgs),
            "candidate": len(cand_imgs),
        },
        "threshold": threshold,
        "distances": distances,
        "accepted_count": accepted_count,
        "accepted_fraction": fraction,
        "minimum_accepted": manifests.FILTERED_MINIMUM_ACCEPTED,
        "condition_runnable": runnable,
    }

    if accepted_count:
        record["distances"]["accepted_to_train"] = describe(
            "accepted -> train", cand_d[accepted_mask]
        )

    # Section 5: filtering by proximity to real data can collapse variety, and a
    # subset that is closer but far less varied is a different animal from one
    # that is closer and equally varied. Make it visible rather than inferred.
    if accepted_count > 1:
        accepted_f = cand_f[accepted_mask]
        within_accepted = nn_distance(
            accepted_f, accepted_f, exclude=[[i] for i in range(len(accepted_f))]
        )
        within_real = nn_distance(
            train_f, train_f, exclude=[[i] for i in range(len(train_f))]
        )
        print("\n diversity (nearest neighbour within each set):")
        record["diversity"] = {
            "within_accepted": describe("accepted <-> accepted", within_accepted),
            "within_train_df": describe("real train df <-> real train df", within_real),
            "within_set_median_ratio": float(
                np.median(within_accepted) / np.median(within_real)
            ),
        }
        print(
            "  median within-set spacing, accepted / real = "
            f"{record['diversity']['within_set_median_ratio']:.3f}   "
            "(<1 means the accepted subset is more tightly clustered than the real one)"
        )

    record_path = Path(args.record_out)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\n[write] {record_path}")

    if not runnable:
        print(
            f"\n[gate] {accepted_count} accepted is below the pre-registered "
            f"minimum of {manifests.FILTERED_MINIMUM_ACCEPTED}. The condition is "
            "NOT run and the shortfall above is the result. No accepted manifest "
            "is written, so the training step cannot proceed by accident."
        )
        return

    # Write from the raw CSV, not the loaded frame: the loader rewrites
    # image_path to absolute paths, and the manifest has to stay relative to
    # keep the manifest+images folder portable.
    raw = pd.read_csv(candidate_path)
    accepted_ids = set(candidate.loc[accepted_mask, "image_id"].astype(str))
    subset = raw[raw["image_id"].astype(str).isin(accepted_ids)].reset_index(drop=True)
    if len(subset) != accepted_count:
        raise RuntimeError(
            f"accepted rows {len(subset)} do not match accepted images "
            f"{accepted_count}; refusing to write a manifest that is not a "
            "subset of the candidate"
        )
    accepted_path = Path(args.accepted_out)
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(accepted_path, index=False)
    print(f"[write] {accepted_path}  ({accepted_count} rows)")
    print(
        "\nNext: train C4_FILTERED with --accepted-manifest pointing at that file. "
        "df stays at 585; the remaining slots are filled by real duplication, "
        "which is what C1 does, so the comparison isolates the accepted images."
    )


if __name__ == "__main__":
    main()
