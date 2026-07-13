"""Sample synthetic ``df`` images from a trained DDPM and sanity-check them.

Requires torch + diffusers (Colab). Loads a ``train_ddpm`` checkpoint, generates
``df`` images with DDIM, writes them plus a manifest that ``C4`` will consume,
and runs a nearest-neighbour check against the real train-split ``df`` images so
we can see the samples are not near-copies of the 85 real ones.

The output folder is self-contained and portable: ``images/``, a manifest with
paths *relative* to the folder, ``metadata.json`` and (for df) ``nn_check.png``.
A non-empty ``--out-dir`` is refused so an earlier run (e.g. the epoch-60 set)
can never be overwritten or mixed into.

Example
-------
    # from the src/ directory; always pass a FRESH --out-dir
    python -m ddpm_derm.sample_ddpm --ckpt /path/to/last.pt --n 64 --out-dir /content/df_preview
    # formal epoch-100 staging run (validated/published by publish_synthetic)
    python -m ddpm_derm.sample_ddpm --ckpt /path/to/run_seed0_epoch0100.pt \
        --require-epoch 100 --require-ema --n 500 --num-steps 50 --seed 0 \
        --out-dir /content/synthetic_df_epoch0100_seed0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import config, manifests
from .ddpm.diffusion import GaussianDiffusion, to_uint8_images
from .ddpm.unet import build_unet
from .train_classifier import set_seed


def _default_ckpt() -> Path:
    local = Path(
        os.environ.get("DDPM_DERM_LOCAL_CKPT_DIR", "/content/ddpm_ckpt")
    ) / "run_seed0_last.pt"
    if local.is_file():
        return local
    snapshots = sorted(
        config.DDPM_CKPT_DIR.glob("run_seed0_epoch[0-9][0-9][0-9][0-9].pt")
    )
    return snapshots[-1] if snapshots else local


def load_model(ckpt_path: Path, device):
    # Trusted checkpoint produced by train_ddpm; it contains full training
    # state, so PyTorch 2.6's weights_only=True default cannot load it.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    img_size = int(ckpt.get("img_size", 64))
    tiny = bool(ckpt.get("tiny", False))
    model = build_unet(img_size=img_size, tiny=tiny).to(device)
    state_key = "ema_state_dict" if "ema_state_dict" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[state_key])
    model.eval()
    d = ckpt.get("diffusion", {})
    diffusion = GaussianDiffusion(
        timesteps=int(d.get("timesteps", 1000)),
        beta_start=float(d.get("beta_start", 1e-4)),
        beta_end=float(d.get("beta_end", 2e-2)),
    )
    return model, diffusion, img_size, int(ckpt.get("epoch", -1)), state_key


@torch.no_grad()
def sample_images(model, diffusion, n, class_idx, img_size, num_steps, eta, batch_size, device):
    """Return an [N,H,W,3] uint8 tensor of generated images."""
    out = []
    remaining = n
    while remaining > 0:
        b = min(batch_size, remaining)
        labels = torch.full((b,), class_idx, device=device, dtype=torch.long)
        x = diffusion.ddim_sample(
            model, b, labels, img_size=img_size, num_steps=num_steps, eta=eta, device=device,
        )
        out.append(to_uint8_images(x))
        remaining -= b
        print(f"[sample] {n - remaining}/{n}")
    return torch.cat(out, dim=0)


def _to_feats(uint8_imgs, nn_size: int) -> np.ndarray:
    """Downscale a [N,H,W,3] uint8 batch to flat [0,1] vectors for L2 distance."""
    feats = []
    for arr in uint8_imgs.numpy():
        small = Image.fromarray(arr).resize((nn_size, nn_size), Image.BILINEAR)
        feats.append(np.asarray(small, dtype=np.float32).reshape(-1) / 255.0)
    return np.stack(feats, axis=0)


def load_real_df(nn_size: int, disp_size: int):
    """Load real train-split df images: (feats [M,D], display [M,disp,disp,3] uint8)."""
    frame = manifests.load_split("train")
    df_rows = frame[frame["label_idx"] == config.TARGET_CLASS_IDX]
    feats, disp = [], []
    for rel in df_rows["image_path"]:
        img = Image.open(config.resolve_image_path(rel)).convert("RGB")
        small = img.resize((nn_size, nn_size), Image.BILINEAR)
        feats.append(np.asarray(small, dtype=np.float32).reshape(-1) / 255.0)
        disp.append(np.asarray(img.resize((disp_size, disp_size), Image.BILINEAR), dtype=np.uint8))
    return np.stack(feats, axis=0), np.stack(disp, axis=0)


def nearest_neighbors(feats_gen: np.ndarray, feats_real: np.ndarray):
    """For each generated feat, index + L2 distance of the nearest real feat."""
    # ||g - r||^2 = ||g||^2 + ||r||^2 - 2 g.r
    g2 = (feats_gen ** 2).sum(1, keepdims=True)          # [N,1]
    r2 = (feats_real ** 2).sum(1, keepdims=True).T        # [1,M]
    cross = feats_gen @ feats_real.T                      # [N,M]
    d2 = np.clip(g2 + r2 - 2.0 * cross, 0.0, None)
    idx = d2.argmin(1)
    dist = np.sqrt(d2[np.arange(len(idx)), idx])
    return idx, dist


def _save_pairs_montage(gen_disp, real_disp, nn_idx, path: Path, k: int) -> None:
    """Save k rows of [generated | nearest-real] side by side."""
    k = min(k, len(gen_disp))
    if k == 0:
        return
    h, w, _ = gen_disp[0].shape
    canvas = Image.new("RGB", (2 * w + 4, k * h + (k - 1) * 4), (255, 255, 255))
    for row in range(k):
        canvas.paste(Image.fromarray(gen_disp[row]), (0, row * (h + 4)))
        canvas.paste(Image.fromarray(real_disp[nn_idx[row]]), (w + 4, row * (h + 4)))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample synthetic df images from a trained DDPM.")
    p.add_argument("--ckpt", default=None,
                   help="Checkpoint path (default: local last.pt, else latest Drive snapshot).")
    p.add_argument("--class-name", default=config.TARGET_CLASS,
                   choices=list(config.CLASS_TO_IDX), help="Class to sample.")
    p.add_argument("--n", type=int, default=500, help="Number of images to generate.")
    p.add_argument("--num-steps", type=int, default=50, help="DDIM steps.")
    p.add_argument("--eta", type=float, default=0.0, help="0 = deterministic DDIM.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", default=None,
                   help="Default: config.SYNTHETIC_DF_DIR. Refused if it exists "
                        "and is not empty (never overwrites an earlier set).")
    p.add_argument("--require-epoch", type=int, default=None,
                   help="Formal runs: fail unless the checkpoint's epoch equals this.")
    p.add_argument("--require-ema", action="store_true",
                   help="Formal runs: fail unless EMA weights are present and used.")
    p.add_argument("--nn-size", type=int, default=32, help="Resolution for the NN distance.")
    p.add_argument("--nn-montage", type=int, default=8, help="Rows in the NN montage.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_path = Path(args.ckpt) if args.ckpt else _default_ckpt()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}; train with train_ddpm first")
    model, diffusion, img_size, epoch, state_key = load_model(ckpt_path, device)
    if args.require_epoch is not None and epoch != args.require_epoch:
        raise RuntimeError(
            f"checkpoint epoch {epoch} != required {args.require_epoch} "
            f"({ckpt_path}); refusing a formal run on the wrong snapshot")
    if args.require_ema and state_key != "ema_state_dict":
        raise RuntimeError(
            f"EMA weights required but checkpoint provides {state_key} ({ckpt_path})")
    class_idx = config.CLASS_TO_IDX[args.class_name]

    # refuse a used output dir BEFORE the (slow) sampling: an earlier synthetic
    # set (e.g. epoch 60 under the default outputs/synthetic_df) must never be
    # overwritten or mixed with a new run.
    out_dir = Path(args.out_dir) if args.out_dir else config.SYNTHETIC_DF_DIR
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(
            f"--out-dir {out_dir} already exists and is not empty; refusing to "
            f"overwrite an existing synthetic set. Pass a fresh directory.")
    print(f"[load] {ckpt_path} (epoch {epoch}, img={img_size}, weights={state_key}) "
          f"-> sampling "
          f"{args.n}x '{args.class_name}' with {args.num_steps} DDIM steps on {device}")

    gen = sample_images(model, diffusion, args.n, class_idx, img_size,
                        args.num_steps, args.eta, args.batch_size, device)

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"synthetic_{args.class_name}.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "label_idx", "dx", "lesion_id", "image_id", "source"])
        for i, arr in enumerate(gen.numpy()):
            stem = f"synthetic_{args.class_name}_{i:04d}"
            Image.fromarray(arr).save(img_dir / f"{stem}.png")
            # path relative to the manifest's directory: the folder must stay
            # portable when copied from /content staging to a versioned Drive
            # dir; C4 resolves it via manifests.load_generated_manifest.
            writer.writerow([f"images/{stem}.png", class_idx, args.class_name,
                             "synthetic", stem, "synthetic"])
    print(f"[done] {args.n} images -> {img_dir}")
    print(f"[done] manifest -> {manifest_path}")

    # --- nearest-neighbour check vs real train df ----------------------------
    nn_stats = None
    if class_idx == config.TARGET_CLASS_IDX:
        feats_gen = _to_feats(gen, args.nn_size)
        feats_real, real_disp = load_real_df(args.nn_size, img_size)
        nn_idx, nn_dist = nearest_neighbors(feats_gen, feats_real)
        print(f"[nn-check] nearest-real L2 over {len(feats_real)} real df "
              f"(resized {args.nn_size}px): "
              f"min={nn_dist.min():.3f} mean={nn_dist.mean():.3f} max={nn_dist.max():.3f}")
        print("[nn-check] very small min distance => possible memorisation; "
              "inspect the montage before trusting C4.")
        # versioned artifact inside this run's folder, never a shared filename
        montage = out_dir / "nn_check.png"
        # show the most-suspicious (closest) samples first
        order = np.argsort(nn_dist)
        _save_pairs_montage(gen.numpy()[order], real_disp, nn_idx[order],
                            montage, k=args.nn_montage)
        print(f"[nn-check] montage (left=generated, right=nearest real) -> {montage}")
        nn_stats = {"min": float(nn_dist.min()), "mean": float(nn_dist.mean()),
                    "max": float(nn_dist.max()), "n_real": int(len(feats_real)),
                    "resize_px": args.nn_size}

    metadata = {
        "checkpoint": ckpt_path.name,
        "epoch": epoch,
        "weights": state_key,
        "class_name": args.class_name,
        "class_idx": class_idx,
        "n": args.n,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "eta": args.eta,
        "image_size": img_size,
        "nn_check": nn_stats,
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[done] metadata -> {meta_path}")


if __name__ == "__main__":
    main()
