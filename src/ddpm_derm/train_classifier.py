"""Train a downstream classifier for one variant/seed and write a results JSON.

Requires torch + torchvision, so this is meant to run on Colab (T4). Locally it
is only import-checked indirectly; the data/metric logic it relies on is covered
by scripts/smoke_test.py.

Model selection uses the validation df F1 (the project's primary metric); the
reported numbers are computed on the held-out test split. Run this once per
(variant, seed) and aggregate with scripts/aggregate_results.py.

Example
-------
    python -m ddpm_derm.train_classifier --variant C0 --seed 0 --epochs 20
    python -m ddpm_derm.train_classifier --variant C1 --seed 0 --epochs 20 --df-target-count 585
    python -m ddpm_derm.train_classifier --variant C4 --seed 0 --epochs 20 \
        --generated-manifest /path/to/synthetic_df/synthetic_df.csv

Quick smoke run (tiny subset, 1 epoch):
    python -m ddpm_derm.train_classifier --variant C0 --seed 0 --epochs 1 --limit 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config, manifests, metrics
from .dataset import build_dataloader
from .model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_rng_state() -> dict:
    """Snapshot the global RNG streams so a resumed run continues them instead
    of restarting from the base seed. This does not make a resumed run
    bit-identical to an uninterrupted one (GPU conv is nondeterministic anyway),
    but it stops the shuffle/augmentation stream from replaying from epoch 1."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    # torch.load(..., map_location=device) may have moved the byte tensors onto
    # the GPU; set_rng_state wants them back on the CPU.
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path, model, optimizer, epoch, best_val_f1, history, args,
                    val_metrics=None) -> None:
    """Write a checkpoint atomically (tmp + replace) so a Colab disconnect
    mid-write cannot leave a corrupt file."""
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_df_f1": best_val_f1,
        "history": history,
        "config": vars(args),
        "class_to_idx": config.CLASS_TO_IDX,
        "rng_state": _get_rng_state(),
    }
    if val_metrics is not None:
        payload["val_metrics"] = val_metrics
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())
    return metrics.classification_summary(y_true, y_pred)


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running = 0.0
    n = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
        n += images.size(0)
    return running / max(n, 1)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a downstream HAM10000 classifier.")
    p.add_argument("--variant", default="C0", choices=["C0", "C1", "C4"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--df-target-count", type=int, default=585,
                   help="C1/C4: total df rows in the train frame (C1 duplicates "
                        "real df up to it; C4 requires real+generated to equal "
                        "it exactly). Keep identical across C1 and C4 runs.")
    p.add_argument("--generated-manifest", default=None,
                   help="C4 only (required): path to the synthetic-df manifest "
                        "CSV; no default output directory is read.")
    p.add_argument("--generated-root", default=None,
                   help="C4 only: root for the manifest's relative image paths "
                        "(default: the manifest's own directory).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the train frame size for a quick smoke run.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--output-dir", default=None,
                   help="Base classifier dir; defaults to config.CLASSIFIER_DIR.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last.pt in this run's checkpoint dir if present.")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    if args.variant == "C4" and not args.generated_manifest:
        p.error("--variant C4 requires --generated-manifest "
                "(no default synthetic dir is read)")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    # Stream per-epoch prints live: a subprocess/pipe stdout is block-buffered by
    # default, which would hold the [epoch NN] lines back until the run ends and
    # make training look stuck. Line buffering flushes on every newline.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    set_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    base_dir = Path(args.output_dir) if args.output_dir else config.CLASSIFIER_DIR
    ckpt_dir = base_dir / "checkpoints" / f"{args.variant}_seed{args.seed}"
    results_dir = base_dir / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] variant={args.variant} seed={args.seed} epochs={args.epochs} "
          f"img={args.img_size} bs={args.batch_size} lr={args.lr} device={device}")

    train_frame = manifests.build_classifier_frame(
        args.variant, split="train", df_target_count=args.df_target_count,
        seed=args.seed, limit=args.limit,
        generated_manifest=args.generated_manifest,
        generated_root=args.generated_root,
    )
    val_frame = manifests.load_split("val")
    test_frame = manifests.load_split("test")
    print(f"[data] variant={args.variant} train={len(train_frame)} "
          f"val={len(val_frame)} test={len(test_frame)}")
    print(f"[data] train class counts: {manifests.class_counts(train_frame)}")

    train_loader = build_dataloader(train_frame, args.img_size, args.batch_size,
                                    train=True, num_workers=args.num_workers)
    val_loader = build_dataloader(val_frame, args.img_size, args.batch_size,
                                  train=False, num_workers=args.num_workers)
    test_loader = build_dataloader(test_frame, args.img_size, args.batch_size,
                                   train=False, num_workers=args.num_workers)

    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    start_epoch = 1
    best_val_f1 = -1.0
    best_state = None
    history = []

    last_path = ckpt_dir / "last.pt"
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_df_f1", -1.0)
        history = ckpt.get("history", [])
        rng = ckpt.get("rng_state")
        if rng is not None:
            _set_rng_state(rng)
        print(f"[resume] found last.pt (epoch {ckpt['epoch']}) -> continuing from "
              f"epoch {start_epoch} (best val df_f1 so far={best_val_f1:.4f}"
              f"{'' if rng is None else ', RNG restored'})")
    elif args.resume:
        print("[start] --resume set but no checkpoint yet -> fresh run from epoch 1")
    else:
        print("[start] fresh run (no --resume) from epoch 1")
    print(f"[ckpt] saving to {ckpt_dir}  (last.pt refreshed every epoch, "
          f"best.pt on val df_f1 improvement)")
    if start_epoch > args.epochs:
        print(f"[skip] already trained all {args.epochs} epochs -> straight to test")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_df_f1": val_metrics["target_f1"],
                        "val_macro_f1": val_metrics["macro_f1"]})
        improved = val_metrics["target_f1"] > best_val_f1
        marker = ""
        if improved:
            best_val_f1 = val_metrics["target_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch,
                            best_val_f1, history, args, val_metrics=val_metrics)
            marker = "  <- new best, saved best.pt"
        # refresh last.pt every epoch so a Colab disconnect can --resume
        save_checkpoint(last_path, model, optimizer, epoch,
                        best_val_f1, history, args)
        print(f"[epoch {epoch:02d}/{args.epochs}] loss={train_loss:.4f} "
              f"val_df_f1={val_metrics['target_f1']:.4f} "
              f"val_macro_f1={val_metrics['macro_f1']:.4f} "
              f"({time.time()-t0:.0f}s){marker}")

    if best_state is not None:
        model.load_state_dict(best_state)
    elif (ckpt_dir / "best.pt").exists():
        # e.g. resumed a run that had already finished all epochs
        model.load_state_dict(
            torch.load(ckpt_dir / "best.pt", map_location=device)["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print(f"[test] variant={args.variant} seed={args.seed} "
          f"df_f1={test_metrics['target_f1']:.4f} "
          f"macro_f1={test_metrics['macro_f1']:.4f} "
          f"acc={test_metrics['accuracy']:.4f}")
    recalls = {k: round(v, 3) for k, v in test_metrics["per_class_recall"].items()}
    print(f"[test] per-class recall: {recalls}")

    result = {
        "variant": args.variant,
        "seed": args.seed,
        "config": vars(args),
        "best_val_df_f1": best_val_f1,
        "history": history,
        "test_metrics": test_metrics,
    }
    out_path = results_dir / f"results_{args.variant}_seed{args.seed}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[done] results -> {out_path}")
    print(f"[done] best.pt (val df_f1={best_val_f1:.4f}, for eval/deploy) + "
          f"last.pt (for --resume) -> {ckpt_dir}")


if __name__ == "__main__":
    main()
