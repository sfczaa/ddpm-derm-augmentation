"""Measure synthetic-df sensitivity to DDIM settings and checkpoint epoch.

Varies step count and eta on the primary checkpoint, then compares saved
epochs at the published baseline setting (50 steps, eta 0.0). Measurements
include saturation, contrast, channel means, and nearest-neighbour distances
with real train/validation df as references. These comparisons describe the
tested grid without establishing a unique cause for the observed differences.

Reads train and validation manifests; the test split is not accessed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ddpm_derm import config, manifests  # noqa: E402
from ddpm_derm.sample_ddpm import load_model, sample_images  # noqa: E402
from ddpm_memorization_diagnostic import (  # noqa: E402
    embedding_features, load_real_data_judge, nn_distance, pixel_features,
)

PUBLISHED_BASELINE = {"num_steps": 50, "eta": 0.0}


def to_pil(batch) -> list[Image.Image]:
    return [Image.fromarray(arr) for arr in batch.numpy()]


def colour_stats(images) -> dict:
    """Saturation, contrast and channel means -- where the failure showed up."""
    arr = np.stack([
        np.asarray(im.resize((64, 64)), dtype=np.float32) / 255.0 for im in images
    ])
    return {
        "saturation": float((arr.max(-1) - arr.min(-1)).mean()),
        "contrast_std": float(arr.std()),
        "mean_rgb": [float(arr[..., c].mean()) for c in range(3)],
    }


def load_real(split: str) -> list[Image.Image]:
    if split == "test":
        raise ValueError("the test split is prohibited in this diagnostic")
    frame = manifests.load_split(split)
    rows = frame[frame["label_idx"] == config.TARGET_CLASS_IDX]
    return [
        Image.open(config.resolve_image_path(p)).convert("RGB")
        for p in rows["image_path"]
    ]


def measure(images, refs, model, img_size) -> dict:
    """Colour stats plus nearest-neighbour distance into the real train df."""
    out = colour_stats(images)
    out["pixel_nn_median"] = float(np.median(
        nn_distance(pixel_features(images, 64), refs["pixel_gallery"])
    ))
    out["embedding_nn_median"] = float(np.median(
        nn_distance(embedding_features(images, model, img_size),
                    refs["embed_gallery"])
    ))
    out["n"] = len(images)
    return out


def build_reference(train_df, val_df, model, img_size) -> dict:
    flipped = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in train_df]
    refs = {
        "pixel_gallery": np.concatenate([
            pixel_features(train_df, 64), pixel_features(flipped, 64)
        ]),
        "embed_gallery": np.concatenate([
            embedding_features(train_df, model, img_size),
            embedding_features(flipped, model, img_size),
        ]),
    }
    refs["real_train_df"] = colour_stats(train_df)
    refs["real_val_df"] = measure(val_df, refs, model, img_size)
    return refs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", required=True,
                   help="Directory holding run_seed0_epoch*.pt")
    p.add_argument("--judge-checkpoint",
                   default="outputs/classifier_df585/checkpoints/C1_seed2/best.pt")
    p.add_argument("--steps", type=int, nargs="+", default=[50, 250, 1000])
    p.add_argument("--etas", type=float, nargs="+", default=[0.0, 1.0])
    p.add_argument("--epochs", type=int, nargs="+", default=[60, 80, 100],
                   help="Epoch sweep, run at the published baseline setting.")
    p.add_argument("--primary-epoch", type=int, default=100)
    p.add_argument("--n", type=int, default=128,
                   help="Images per configuration; 128 is ample for these statistics.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="outputs/figures/ddpm_sampling_sweep.json")
    p.add_argument("--montage", default="outputs/figures/ddpm_sampling_sweep.png")
    args = p.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt_dir = Path(args.checkpoint_dir)

    print("[load] real train df, real val df  (test split never opened)")
    train_df, val_df = load_real("train"), load_real("val")
    judge, judge_cfg = load_real_data_judge(Path(args.judge_checkpoint))
    img_size = int(judge_cfg.get("img_size", 128))
    refs = build_reference(train_df, val_df, judge, img_size)
    print(f"  train df={len(train_df)}  val df={len(val_df)}  device={device}")
    print(f"  real train df   : saturation={refs['real_train_df']['saturation']:.4f} "
          f"contrast={refs['real_train_df']['contrast_std']:.4f}")
    print(f"  real val df ref : embedding_nn_median="
          f"{refs['real_val_df']['embedding_nn_median']:.4f}")

    results = {
        "reference": {k: refs[k] for k in ("real_train_df", "real_val_df")},
        "published_baseline": PUBLISHED_BASELINE,
        "n_per_config": args.n,
        "seed": args.seed,
        "judge": {"variant": judge_cfg.get("variant"), "seed": judge_cfg.get("seed"),
                  "img_size": img_size},
        "test_split_accessed": False,
        "sampler_sweep": {},
        "epoch_sweep": {},
    }
    montage_rows: list[tuple[str, list[Image.Image]]] = [
        ("real train df", train_df[:6])
    ]

    def run(ckpt_path, num_steps, eta, tag, bucket):
        torch.manual_seed(args.seed)
        model, diffusion, gen_size, epoch, state_key, _ = load_model(ckpt_path, device)
        started = time.time()
        batch = sample_images(model, diffusion, args.n, config.TARGET_CLASS_IDX,
                              gen_size, num_steps, eta, args.batch_size, device)
        images = to_pil(batch)
        stats = measure(images, refs, judge, img_size)
        stats.update({"num_steps": num_steps, "eta": eta, "epoch": epoch,
                      "weights": state_key, "seconds": round(time.time() - started, 1)})
        results[bucket][tag] = stats
        print(f"  [{tag}] saturation={stats['saturation']:.4f} "
              f"contrast={stats['contrast_std']:.4f} "
              f"embed_nn={stats['embedding_nn_median']:.4f} "
              f"({stats['seconds']}s)")
        montage_rows.append((tag, images[:6]))
        del model
        return stats

    primary = ckpt_dir / f"run_seed0_epoch{args.primary_epoch:04d}.pt"
    print(f"\n=== sampler sweep on epoch {args.primary_epoch} ===")
    print("  real train df saturation is the target to recover: "
          f"{refs['real_train_df']['saturation']:.4f}")
    for num_steps in args.steps:
        for eta in args.etas:
            run(primary, num_steps, eta, f"steps{num_steps}_eta{eta}", "sampler_sweep")

    print("\n=== epoch sweep at the published baseline setting ===")
    for epoch in args.epochs:
        path = ckpt_dir / f"run_seed0_epoch{epoch:04d}.pt"
        if not path.is_file():
            print(f"  [epoch{epoch}] missing: {path}")
            continue
        run(path, PUBLISHED_BASELINE["num_steps"], PUBLISHED_BASELINE["eta"],
            f"epoch{epoch}", "epoch_sweep")

    cell = 128
    canvas = Image.new("RGB", (cell * 6, cell * len(montage_rows)), "white")
    for row, (_, imgs) in enumerate(montage_rows):
        for col, im in enumerate(imgs[:6]):
            canvas.paste(im.resize((cell, cell), Image.NEAREST), (col * cell, row * cell))
    montage = Path(args.montage)
    montage.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(montage)
    results["montage"] = {"path": str(montage),
                          "rows": [tag for tag, _ in montage_rows]}
    print(f"\n[montage] rows: {', '.join(tag for tag, _ in montage_rows)}")
    print(f"[montage] -> {montage}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"[done] record -> {out}")


if __name__ == "__main__":
    main()
