"""Describe class proximity, resolution sensitivity, and synthetic-df appearance.

Test A compares nearest real classes using balanced galleries of 85 images
per class. Test B recomputes distances at multiple resolutions to measure
the persistence of coarse differences. Test C writes a visual montage.

These diagnostics do not isolate a unique causal explanation. Inputs use
train and validation manifests only; no test data or pass/fail threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddpm_derm import config, manifests  # noqa: E402
from ddpm_memorization_diagnostic import (  # noqa: E402
    embedding_features, load_real_data_judge, load_synthetic, nn_distance,
    pixel_features,
)

BALANCE_SEED = 0


def load_train_by_class(per_class: int) -> dict[str, list[Image.Image]]:
    """Equal-sized real galleries per class, so gallery size cannot decide the winner."""
    frame = manifests.load_split("train")
    rng = np.random.default_rng(BALANCE_SEED)
    out = {}
    for idx, name in enumerate(config.CLASS_NAMES):
        rows = frame[frame["label_idx"] == idx]
        paths = list(rows["image_path"])
        if len(paths) > per_class:
            paths = [paths[i] for i in rng.choice(len(paths), per_class, replace=False)]
        out[name] = [
            Image.open(config.resolve_image_path(p)).convert("RGB") for p in paths
        ]
    return out


def load_val_df() -> list[Image.Image]:
    frame = manifests.load_split("val")
    rows = frame[frame["label_idx"] == config.TARGET_CLASS_IDX]
    return [
        Image.open(config.resolve_image_path(p)).convert("RGB")
        for p in rows["image_path"]
    ]


def nearest_class(query_feats, gallery_feats, gallery_labels) -> np.ndarray:
    q2 = (query_feats ** 2).sum(1, keepdims=True)
    g2 = (gallery_feats ** 2).sum(1, keepdims=True).T
    d2 = np.clip(q2 + g2 - 2.0 * query_feats @ gallery_feats.T, 0.0, None)
    return np.asarray(gallery_labels)[d2.argmin(1)]


def test_a(synth, val_df, by_class, model, img_size, results) -> None:
    print("\n=== Test A: which real class do the synthetic images land nearest to? ===")
    print(f"    galleries balanced to {len(by_class['df'])} images per class "
          f"(seed {BALANCE_SEED})")
    images, labels = [], []
    for name, imgs in by_class.items():
        images.extend(imgs)
        labels.extend([name] * len(imgs))

    for space, embed in (
        ("pixel @ 64px", lambda ims: pixel_features(ims, 64)),
        ("C1 embedding", lambda ims: embedding_features(ims, model, img_size)),
    ):
        gallery = embed(images)
        block = {}
        for who, feats in (
            ("synthetic df (500)", embed(synth)),
            ("real val df (14) [reference]", embed(val_df)),
        ):
            hits = nearest_class(feats, gallery, labels)
            counts = {c: int((hits == c).sum()) for c in config.CLASS_NAMES}
            share_df = counts["df"] / len(hits)
            top = max(counts, key=counts.get)
            print(f"  [{space}] {who:30} nearest class = {top:5} "
                  f"({counts[top]}/{len(hits)});  df share = {share_df:.1%}")
            print(f"      {counts}")
            block[who] = {"counts": counts, "df_share": share_df, "argmax": top}
        results.setdefault("test_a_nearest_class", {})[space] = block


def test_b(synth, val_df, train_df, results) -> None:
    print("\n=== Test B: does the gap survive when high-frequency detail is removed? ===")
    print("    lower resolution keeps only coarse colour and shape")
    flipped = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in train_df]
    block = {}
    for px in (64, 32, 16, 8):
        gallery = np.concatenate([
            pixel_features(train_df, px), pixel_features(flipped, px)
        ])
        syn = np.median(nn_distance(pixel_features(synth, px), gallery))
        val = np.median(nn_distance(pixel_features(val_df, px), gallery))
        # Scale-free: how many times farther the synthetic sit than genuinely
        # new real df do. Raw distances shrink with resolution; this does not.
        ratio = float(syn / val)
        print(f"  {px:3}px  synthetic median={syn:7.4f}   "
              f"val df median={val:7.4f}   ratio={ratio:.3f}")
        block[f"{px}px"] = {"synthetic_median": float(syn),
                            "val_median": float(val), "ratio": ratio}
    print("  ratio stays high as resolution drops -> structural;"
          " ratio falls toward 1.0 -> high-frequency texture")
    results["test_b_resolution_ladder"] = block


def test_c(synth, train_df, model, img_size, out_path, results) -> None:
    print("\n=== Test C: montage ===")
    flipped = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in train_df]
    gallery = np.concatenate([
        embedding_features(train_df, model, img_size),
        embedding_features(flipped, model, img_size),
    ])
    dist = nn_distance(embedding_features(synth, model, img_size), gallery)
    order = np.argsort(dist)
    n = 6
    picks = {
        "real train df (reference)": [train_df[i] for i in range(n)],
        "synthetic: closest to real": [synth[i] for i in order[:n]],
        "synthetic: median": [synth[i] for i in order[len(order) // 2 - n // 2:][:n]],
        "synthetic: farthest": [synth[i] for i in order[-n:]],
    }
    cell = 128
    canvas = Image.new("RGB", (cell * n, cell * len(picks)), "white")
    for row, imgs in enumerate(picks.values()):
        for col, im in enumerate(imgs):
            canvas.paste(im.resize((cell, cell), Image.NEAREST), (col * cell, row * cell))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print("  rows, top to bottom: " + " | ".join(picks))
    print(f"  -> {out_path}")
    results["test_c_montage"] = {
        "path": str(out_path),
        "rows": list(picks),
        "closest_distance": float(dist[order[0]]),
        "farthest_distance": float(dist[order[-1]]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic-dir", default="outputs/synthetic_df/images")
    p.add_argument("--judge-checkpoint",
                   default="outputs/classifier_df585/checkpoints/C1_seed2/best.pt")
    p.add_argument("--out", default="outputs/figures/ddpm_failure_localization.json")
    p.add_argument("--montage", default="outputs/figures/ddpm_failure_montage.png")
    args = p.parse_args()

    print("[load] real train (all 7 classes, balanced), real val df, synthetic df")
    synth = load_synthetic(Path(args.synthetic_dir))
    val_df = load_val_df()
    by_class = load_train_by_class(per_class=85)
    train_df = by_class["df"]
    print("  " + "  ".join(f"{k}={len(v)}" for k, v in by_class.items()))
    print(f"  val df={len(val_df)}  synthetic={len(synth)}")

    model, cfg = load_real_data_judge(Path(args.judge_checkpoint))
    img_size = int(cfg.get("img_size", 128))

    results: dict = {
        "balanced_per_class": len(train_df),
        "balance_seed": BALANCE_SEED,
        "judge": {"variant": cfg.get("variant"), "seed": cfg.get("seed"),
                  "img_size": img_size},
        "test_split_accessed": False,
    }

    test_a(synth, val_df, by_class, model, img_size, results)
    test_b(synth, val_df, train_df, results)
    test_c(synth, train_df, model, img_size, Path(args.montage), results)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"\n[done] record -> {out}")


if __name__ == "__main__":
    main()
