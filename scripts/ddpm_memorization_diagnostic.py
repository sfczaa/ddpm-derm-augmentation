"""Does the df DDPM generate new lesions, or reproduce its 85 training images?

This is the one measurement that separates the two readings of the C4-vs-C1
result. The generator saw only 85 real train ``df`` images. If it memorised
them, then C4 (synthetic df) is functionally C1 (duplicated real df), and the
observed small C4-C1 gap is exactly what that would produce.

A nearest-neighbour distance is meaningless on its own -- "min = 3.2" answers
nothing without a scale. So every distance here is reported against two
reference distributions computed from the same fixed split:

* ``val->train``  -- the 14 validation df are real lesions the DDPM never saw.
  Their distance to the training set is what a genuinely new real df looks
  like. This is the sharpest available yardstick.
* ``train->train`` (leave-one-out) -- how far apart two different real df
  lesions are.

Synthetic distances sitting well below the val->train distribution is
memorisation-shaped. Comparable distances are the strongest reassurance this
data can support.

Two spaces are measured because they fail differently: pixel space catches
literal copying, embedding space catches near-duplicates that survive small
transforms. The gallery includes horizontally flipped real images because DDPM
training used ``RandomHorizontalFlip`` -- memorisation-up-to-flip is an
expected failure mode that an unflipped comparison cannot see.

Read-only. Touches train and val manifests only; the test split is never
opened. Descriptive output only -- no thresholds, no pass/fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddpm_derm import config, manifests  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PERCENTILES = (0, 1, 5, 25, 50, 75, 100)


def load_real_df(split: str) -> list[Image.Image]:
    """Real df images for one split. Never called with 'test'."""
    if split == "test":
        raise ValueError("the test split is prohibited in this diagnostic")
    frame = manifests.load_split(split)
    rows = frame[frame["label_idx"] == config.TARGET_CLASS_IDX]
    return [
        Image.open(config.resolve_image_path(rel)).convert("RGB")
        for rel in rows["image_path"]
    ]


def load_synthetic(directory: Path) -> list[Image.Image]:
    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no synthetic images under {directory}")
    return [Image.open(p).convert("RGB") for p in paths]


def synthetic_provenance(directory: Path) -> dict:
    """Identify the batch that was actually measured.

    Two different 500-image batches exist under identical filenames, and an
    earlier diagnostic measured one while quoting the other's metadata. The
    content hash is computed over the image bytes, so a record can always be
    tied back to the exact set it describes regardless of what any neighbouring
    metadata file claims.
    """
    directory = Path(directory)
    paths = sorted(directory.glob("*.png"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    record = {
        "directory": str(directory),
        "image_count": len(paths),
        "content_sha256": digest.hexdigest(),
        "metadata": None,
    }
    # The generator writes metadata.json beside the images it produced; a batch
    # that has none cannot state its own checkpoint or sampler settings.
    for candidate in (directory.parent / "metadata.json",):
        if candidate.is_file():
            record["metadata"] = json.loads(candidate.read_text(encoding="utf-8"))
            record["metadata_path"] = str(candidate)
            break
    return record


def colour_stats(images) -> dict:
    """Saturation, contrast and channel means.

    Kept alongside the distance measurements because a distance says only that
    two sets differ, not how. These three say whether the difference is in
    colour and dynamic range, which a nearest-neighbour number cannot.
    """
    arr = np.stack([
        np.asarray(im.resize((64, 64)), dtype=np.float32) / 255.0 for im in images
    ])
    return {
        "saturation": float((arr.max(-1) - arr.min(-1)).mean()),
        "contrast_std": float(arr.std()),
        "mean_rgb": [float(arr[..., channel].mean()) for channel in range(3)],
        "n": len(images),
    }


def pixel_features(images, size: int) -> np.ndarray:
    """Flat [0,1] RGB vectors at the generator's native resolution."""
    out = []
    for img in images:
        small = img.resize((size, size), Image.BILINEAR)
        out.append(np.asarray(small, dtype=np.float32).reshape(-1) / 255.0)
    return np.stack(out)


def load_real_data_judge(checkpoint: Path) -> nn.Module:
    """The C1 ResNet-18 as feature extractor.

    C1 is trained on real images only, so it never saw the synthetic set it is
    being asked to measure. A C4 model would be circular, and PanDerm is
    disqualified because its pretraining corpus cannot be shown to exclude
    HAM10000.
    """
    from ddpm_derm.checkpoint import load_checkpoint
    payload = load_checkpoint(checkpoint, map_location="cpu")
    state = payload.get("model_state_dict", payload)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, state["fc.weight"].shape[0])
    model.load_state_dict(state)
    model.fc = nn.Identity()  # penultimate 512-d features
    model.eval()
    return model, payload.get("config", {})


@torch.no_grad()
def embedding_features(images, model, img_size: int, batch: int = 64) -> np.ndarray:
    """L2-normalised penultimate features, so distance is scale-free."""
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    out = []
    for start in range(0, len(images), batch):
        chunk = torch.stack([tf(im) for im in images[start:start + batch]])
        feats = model(chunk)
        out.append(torch.nn.functional.normalize(feats, dim=1).numpy())
    return np.concatenate(out)


def nn_distance(query: np.ndarray, gallery: np.ndarray, exclude=None) -> np.ndarray:
    """Nearest-neighbour L2 distance from each query row into the gallery.

    ``exclude`` is a list of gallery column indices to mask per query row, used
    for leave-one-out so an image is not its own nearest neighbour.
    """
    q2 = (query ** 2).sum(1, keepdims=True)
    g2 = (gallery ** 2).sum(1, keepdims=True).T
    d2 = np.clip(q2 + g2 - 2.0 * query @ gallery.T, 0.0, None)
    if exclude is not None:
        for row, cols in enumerate(exclude):
            d2[row, cols] = np.inf
    return np.sqrt(d2.min(1))


def describe(name: str, values: np.ndarray) -> dict:
    stats = {f"p{p}": float(np.percentile(values, p)) for p in PERCENTILES}
    stats["mean"] = float(values.mean())
    stats["n"] = int(len(values))
    print(
        f"  {name:34} n={stats['n']:4}  min={stats['p0']:.4f}  "
        f"p5={stats['p5']:.4f}  median={stats['p50']:.4f}  "
        f"mean={stats['mean']:.4f}  max={stats['p100']:.4f}"
    )
    return stats


def compare(synth: np.ndarray, val: np.ndarray, label: str) -> dict:
    """How the synthetic distances sit inside the val->train reference."""
    below_min = float((synth < val.min()).mean())
    fraction = float((synth[:, None] < val[None, :]).mean())
    print(
        f"  [{label}] synthetic below the closest genuinely-new real df: "
        f"{below_min:.1%} of 500"
    )
    print(
        f"  [{label}] P(a synthetic is closer to train than a random new "
        f"real df) = {fraction:.3f}   (0.5 = indistinguishable)"
    )
    return {"fraction_below_val_min": below_min, "prob_closer_than_val": fraction}


def analyse(space: str, synth_f, train_f, val_f, results: dict) -> None:
    print(f"\n=== {space} ===")
    # Flip-aware gallery: DDPM training used RandomHorizontalFlip, so a
    # memorised image may only match its mirror.
    n_train = len(train_f["orig"])
    gallery = np.concatenate([train_f["orig"], train_f["flip"]])

    syn = nn_distance(synth_f, gallery)
    val = nn_distance(val_f, gallery)
    loo = nn_distance(
        train_f["orig"],
        gallery,
        exclude=[[i, i + n_train] for i in range(n_train)],
    )

    print(" nearest-neighbour distance into the 85 real train df (+flips):")
    block = {
        "synthetic_to_train": describe("synthetic (500) -> train", syn),
        "val_to_train": describe("val df (14) -> train  [reference]", val),
        "train_to_train_loo": describe("train df -> train, leave-one-out", loo),
    }
    print()
    block.update(compare(syn, val, space))

    print("\n diversity (nearest neighbour within each set):")
    within_syn = nn_distance(
        synth_f, synth_f, exclude=[[i] for i in range(len(synth_f))]
    )
    within_real = nn_distance(
        train_f["orig"], train_f["orig"], exclude=[[i] for i in range(n_train)]
    )
    block["within_synthetic"] = describe("synthetic <-> synthetic", within_syn)
    block["within_train_df"] = describe("real train df <-> real train df", within_real)
    ratio = float(np.median(within_syn) / np.median(within_real))
    print(
        f"  median within-set spacing, synthetic / real = {ratio:.3f}   "
        "(<1 means the synthetic set is more tightly clustered than the real one)"
    )
    block["within_set_median_ratio"] = ratio
    results[space] = block


def resolution_control(synth_imgs, train_imgs, val_imgs, model, img_size,
                       pixel_size, results) -> None:
    """Rule out resolution as the explanation for any synthetic-real gap.

    The synthetic images are born at 64px and upsampled to the judge's input;
    the real images are downsampled to it from full resolution. That asymmetry
    alone could push synthetic features away from real ones. Pushing the real
    val df through the same 64px bottleneck isolates how much of the gap is
    resolution rather than distribution.
    """
    print("\n=== resolution control ===")
    bottlenecked = [im.resize((pixel_size, pixel_size), Image.BILINEAR)
                    for im in val_imgs]
    flipped = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in train_imgs]
    gallery = np.concatenate([
        embedding_features(train_imgs, model, img_size),
        embedding_features(flipped, model, img_size),
    ])
    block = {}
    for label, imgs in (
        ("synthetic (born at 64px)", synth_imgs),
        ("val df, full resolution", val_imgs),
        (f"val df through the {pixel_size}px bottleneck", bottlenecked),
    ):
        d = nn_distance(embedding_features(imgs, model, img_size), gallery)
        median = float(np.median(d))
        # L2 on unit vectors: cos = 1 - d^2/2.
        cosine = 1.0 - median ** 2 / 2.0
        print(f"  {label:44} n={len(d):4}  median={median:.4f}  "
              f"cosine={cosine:.3f}")
        block[label] = {"n": len(d), "median": median, "cosine": cosine}
    results["resolution_control"] = block


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # The published epoch-100 set, which is what the formal C4 condition trains
    # on (`colab_classifier_baseline.py` pins the manifest to this directory and
    # gates on its `_READY.json`). `outputs/synthetic_df/images` holds an older
    # epoch-60 batch under identical filenames and must not be the default:
    # measuring it while quoting epoch-100 provenance is exactly the mistake
    # that produced a wrong entry once already.
    p.add_argument("--synthetic-dir",
                   default="outputs/synthetic_df/epoch0100_seed0/images")
    p.add_argument(
        "--judge-checkpoint",
        default="outputs/classifier_df585/checkpoints/C1_seed2/best.pt",
    )
    p.add_argument("--pixel-size", type=int, default=64,
                   help="Native generator resolution; do not upscale.")
    p.add_argument("--out", default="outputs/figures/ddpm_memorization_diagnostic.json")
    args = p.parse_args()

    print("[load] real train df, real val df, synthetic df  (test split untouched)")
    train_imgs = load_real_df("train")
    val_imgs = load_real_df("val")
    synth_imgs = load_synthetic(Path(args.synthetic_dir))
    print(f"  train df={len(train_imgs)}  val df={len(val_imgs)}  "
          f"synthetic={len(synth_imgs)}  synthetic size={synth_imgs[0].size}")

    results: dict = {
        "synthetic_provenance": synthetic_provenance(Path(args.synthetic_dir)),
        "counts": {
            "train_df": len(train_imgs),
            "val_df": len(val_imgs),
            "synthetic": len(synth_imgs),
        },
        "pixel_size": args.pixel_size,
        "flip_aware": True,
        "test_split_accessed": False,
    }

    flipped = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in train_imgs]

    analyse(
        f"pixel space @ {args.pixel_size}px",
        pixel_features(synth_imgs, args.pixel_size),
        {
            "orig": pixel_features(train_imgs, args.pixel_size),
            "flip": pixel_features(flipped, args.pixel_size),
        },
        pixel_features(val_imgs, args.pixel_size),
        results,
    )

    judge_path = Path(args.judge_checkpoint)
    model, judge_cfg = load_real_data_judge(judge_path)
    img_size = int(judge_cfg.get("img_size", 128))
    print(f"\n[judge] {judge_path}  variant={judge_cfg.get('variant')} "
          f"seed={judge_cfg.get('seed')} img_size={img_size}  "
          "(real-data-only; never trained on synthetic)")
    results["judge"] = {
        "checkpoint": str(judge_path),
        "variant": judge_cfg.get("variant"),
        "seed": judge_cfg.get("seed"),
        "img_size": img_size,
    }

    analyse(
        "C1 ResNet-18 embedding",
        embedding_features(synth_imgs, model, img_size),
        {
            "orig": embedding_features(train_imgs, model, img_size),
            "flip": embedding_features(flipped, model, img_size),
        },
        embedding_features(val_imgs, model, img_size),
        results,
    )

    resolution_control(synth_imgs, train_imgs, val_imgs, model, img_size,
                       args.pixel_size, results)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"\n[done] record -> {out}")


if __name__ == "__main__":
    main()
