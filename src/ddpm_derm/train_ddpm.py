"""Train the class-conditional DDPM on the HAM10000 train split (all 7 classes).

Requires torch + torchvision + diffusers, so this runs on Colab (T4). The
generator sees the *train split only*; ``df`` images are produced later by
``sample_ddpm`` through class conditioning.

There is no validation-based model selection here: val/test images are off
limits to the generator, so we cannot compute a held-out generative score.
The per-seed checkpoint (refreshed every epoch) is both the resume point and
the model used for sampling/deployment. Sample quality is judged by eye from
the periodic preview grids and by the nearest-neighbour check in ``sample_ddpm``.

Example
-------
    # from the src/ directory
    python -m ddpm_derm.train_ddpm --epochs 150 --img-size 64 --batch-size 64
    python -m ddpm_derm.train_ddpm --resume            # continue after a disconnect

Tiny smoke run (a few images, 1 epoch, small net):
    python -m ddpm_derm.train_ddpm --epochs 1 --img-size 32 --batch-size 4 \
        --limit 32 --tiny --num-workers 0
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch

from . import config, manifests
from .dataset import build_ddpm_dataloader
from .ddpm.diffusion import GaussianDiffusion, to_uint8_images
from .ddpm.unet import build_unet
# RNG-state helpers are shared with the classifier trainer (same resume story).
from .train_classifier import _get_rng_state, _set_rng_state, set_seed


def save_checkpoint(path: Path, model, ema_model, optimizer, epoch, history,
                    args, diffusion) -> None:
    """Overwrite one checkpoint on local runtime storage with full train state."""
    payload = {
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "history": history,
        "config": vars(args),
        "class_to_idx": config.CLASS_TO_IDX,
        "img_size": args.img_size,
        "tiny": args.tiny,
        "ema_decay": args.ema_decay,
        "diffusion": {
            "timesteps": diffusion.timesteps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
        },
        "rng_state": _get_rng_state(),
    }
    with path.open("wb") as fh:
        torch.save(payload, fh)


def latest_snapshot(snapshot_dir: Path, seed: int) -> Path | None:
    """Return the latest immutable per-epoch snapshot for this seed."""
    snapshots = sorted(snapshot_dir.glob(f"run_seed{seed}_epoch[0-9][0-9][0-9][0-9].pt"))
    return snapshots[-1] if snapshots else None


def copy_checkpoint(source: Path, destination: Path) -> None:
    """Copy one checkpoint and verify the visible destination size."""
    shutil.copyfile(source, destination)
    if hasattr(os, "sync"):
        os.sync()
    if not destination.is_file():
        raise RuntimeError(f"checkpoint copy is not visible after write: {destination}")
    if destination.stat().st_size != source.stat().st_size:
        raise RuntimeError(f"checkpoint copy size mismatch: {destination}")


def save_snapshot(local_path: Path, snapshot_dir: Path, seed: int, epoch: int) -> Path:
    """Create one immutable Drive snapshot; never overwrite an existing name."""
    snapshot = snapshot_dir / f"run_seed{seed}_epoch{epoch:04d}.pt"
    if snapshot.exists():
        raise FileExistsError(f"refusing to overwrite existing snapshot: {snapshot}")
    print(f"[snapshot] writing unique epoch {epoch} snapshot -> {snapshot}")
    copy_checkpoint(local_path, snapshot)
    print(f"[snapshot] complete ({snapshot.stat().st_size / 1024**2:.1f} MB)")
    return snapshot


def _save_grid(images_uint8, path: Path, ncol: int = 8) -> None:
    """Save a [N,H,W,C] uint8 batch as a single row-major PNG montage."""
    from PIL import Image

    n, h, w, _ = images_uint8.shape
    ncol = min(ncol, n)
    nrow = (n + ncol - 1) // ncol
    canvas = Image.new("RGB", (ncol * w, nrow * h), (0, 0, 0))
    arr = images_uint8.numpy()
    for i in range(n):
        tile = Image.fromarray(arr[i])
        canvas.paste(tile, ((i % ncol) * w, (i // ncol) * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def update_ema(ema_model, model, decay: float) -> None:
    """Move EMA parameters toward the current model and copy model buffers."""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(param, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def train_one_epoch(model, ema_model, loader, diffusion, optimizer,
                    device, ema_decay: float) -> float:
    model.train()
    running, n = 0.0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = diffusion.p_losses(model, images, class_labels=labels)
        loss.backward()
        optimizer.step()
        update_ema(ema_model, model, ema_decay)
        running += loss.item() * images.size(0)
        n += images.size(0)
    return running / max(n, 1)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the class-conditional DDPM.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay updated after every optimizer step.")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=2e-2)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the train frame size for a quick smoke run.")
    p.add_argument("--tiny", action="store_true",
                   help="Use the small smoke-test UNet instead of the full one.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--preview-every", type=int, default=0,
                   help="If >0, sample a df preview grid every N epochs.")
    p.add_argument("--preview-steps", type=int, default=50,
                   help="DDIM steps for the training preview grid.")
    p.add_argument("--output-dir", default=None,
                   help="Local checkpoint directory; defaults to "
                        "$DDPM_DERM_LOCAL_CKPT_DIR or /content/ddpm_ckpt.")
    p.add_argument("--snapshot-dir", default=None,
                   help="Existing Drive directory for immutable per-epoch snapshots.")
    p.add_argument("--snapshot-every", type=int, default=10,
                   help="Write a unique Drive snapshot every N epochs and at the target epoch.")
    p.add_argument("--resume", action="store_true",
                   help="Require and resume from this seed's checkpoint.")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be between 0 and 1")
    if args.snapshot_every < 1:
        raise ValueError("--snapshot-every must be >= 1")
    # Line-buffer stdout so per-epoch lines stream live through a Colab pipe
    # instead of being held back until the process exits.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_dir = Path(
        args.output_dir
        or os.environ.get("DDPM_DERM_LOCAL_CKPT_DIR", "/content/ddpm_ckpt")
    )
    if ckpt_dir.as_posix().startswith("/content/drive/"):
        raise ValueError(
            "refusing to write the per-epoch checkpoint directly to Drive; "
            "use --output-dir /content/ddpm_ckpt and --snapshot-dir on Drive"
        )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"local checkpoint directory does not exist: {ckpt_dir}"
        )
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else None
    if snapshot_dir is not None and not snapshot_dir.is_dir():
        raise FileNotFoundError(
            f"snapshot directory does not exist: {snapshot_dir}; create it in Drive first"
        )
    last_path = ckpt_dir / f"run_seed{args.seed}_last.pt"
    if args.resume and not last_path.is_file():
        snapshot = latest_snapshot(snapshot_dir, args.seed) if snapshot_dir else None
        if snapshot is None:
            raise FileNotFoundError(
                f"--resume requested but neither local checkpoint nor Drive snapshot exists "
                f"for seed {args.seed}"
            )
        print(f"[restore] latest Drive snapshot -> local checkpoint: {snapshot}")
        copy_checkpoint(snapshot, last_path)
        print(f"[restore] local checkpoint ready -> {last_path}")
    elif not args.resume and last_path.exists():
        raise FileExistsError(
            f"fresh run would overwrite existing local checkpoint: {last_path}; "
            "use --resume or remove it intentionally"
        )
    print(f"[run] ddpm seed={args.seed} epochs={args.epochs} img={args.img_size} "
          f"bs={args.batch_size} lr={args.lr} T={args.timesteps} "
          f"tiny={args.tiny} device={device}")

    train_frame = manifests.load_split("train")
    if args.limit is not None:
        train_frame = train_frame.sample(
            n=min(args.limit, len(train_frame)), random_state=args.seed
        ).reset_index(drop=True)
    print(f"[data] train={len(train_frame)} (all 7 classes, train split only)")
    print(f"[data] train class counts: {manifests.class_counts(train_frame)}")

    loader = build_ddpm_dataloader(
        train_frame, img_size=args.img_size, batch_size=args.batch_size,
        train=True, num_workers=args.num_workers,
    )
    diffusion = GaussianDiffusion(
        timesteps=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end,
    )
    model = build_unet(img_size=args.img_size, tiny=args.tiny).to(device)
    ema_model = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] class-conditional UNet2DModel params={n_params/1e6:.1f}M")

    start_epoch = 1
    history = []
    if args.resume:
        # This is our own full training checkpoint (optimizer + RNG state), not
        # an untrusted weights-only artifact. PyTorch 2.6 defaults to
        # weights_only=True, which cannot deserialize the saved NumPy RNG state.
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "ema_state_dict" in ckpt:
            saved_decay = float(ckpt.get("ema_decay", args.ema_decay))
            if not math.isclose(saved_decay, args.ema_decay, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"EMA decay mismatch: checkpoint={saved_decay} "
                    f"current={args.ema_decay}"
                )
            ema_model.load_state_dict(ckpt["ema_state_dict"])
            print(f"[ema] restored EMA weights (decay={args.ema_decay})")
        else:
            ema_model.load_state_dict(model.state_dict())
            print(f"[ema] legacy checkpoint has no EMA -> initialized from raw model "
                  f"(decay={args.ema_decay})")
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        rng = ckpt.get("rng_state")
        if rng is not None:
            _set_rng_state(rng)
        print(f"[resume] found {last_path.name} (epoch {ckpt['epoch']}) -> continuing from "
              f"epoch {start_epoch}{'' if rng is None else ' (RNG restored)'}")
        del ckpt
    else:
        print("[start] fresh run (no --resume) from epoch 1")
    print(f"[ckpt] local overwrite every epoch -> {last_path} (resume + sampling model)")
    if start_epoch > args.epochs:
        print(f"[skip] already trained all {args.epochs} epochs -> nothing to do")

    df_idx = config.TARGET_CLASS_IDX
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(
            model, ema_model, loader, diffusion, optimizer, device, args.ema_decay
        )
        history.append({"epoch": epoch, "train_loss": loss})
        save_checkpoint(
            last_path, model, ema_model, optimizer, epoch, history, args, diffusion
        )
        if snapshot_dir is not None and (
            epoch % args.snapshot_every == 0 or epoch == args.epochs
        ):
            save_snapshot(last_path, snapshot_dir, args.seed, epoch)
        msg = f"[epoch {epoch:03d}/{args.epochs}] loss={loss:.4f} ({time.time()-t0:.0f}s)"

        if args.preview_every and epoch % args.preview_every == 0:
            n_prev = min(8, args.batch_size)
            labels = torch.full((n_prev,), df_idx, device=device, dtype=torch.long)
            samples = diffusion.ddim_sample(
                ema_model, n_prev, labels, img_size=args.img_size,
                num_steps=args.preview_steps, device=device,
            )
            _save_grid(to_uint8_images(samples),
                       config.DDPM_SAMPLES_DIR / f"preview_seed{args.seed}_epoch{epoch:03d}.png")
            msg += "  <- saved df preview"
        print(msg)

    print(f"[done] checkpoint (sampling/deploy model) -> {last_path}")


if __name__ == "__main__":
    main()
