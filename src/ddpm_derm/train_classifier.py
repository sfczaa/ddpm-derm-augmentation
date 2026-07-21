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
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import classifier_objective, classifier_run, coca_run, config, manifests, metrics
from .dataset import build_dataloader
from .model import (
    COCA_ARCH,
    COCA_PRETRAINED,
    build_model,
    model_identity as build_model_identity,
    trainable_parameters,
)


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


def _load_trusted_checkpoint(path, device) -> dict:
    """Load a checkpoint produced by this trainer, including its RNG state."""
    return torch.load(path, map_location=device, weights_only=False)


def _checkpoint_format(run_identity: dict) -> str:
    value = run_identity.get("checkpoint_format")
    if not isinstance(value, str):
        raise ValueError("run identity is missing checkpoint_format")
    return value


def validate_checkpoint_payload(checkpoint, model, run_identity, arch: str) -> None:
    """Validate the architecture-specific payload before loading any state."""
    expected_format = _checkpoint_format(run_identity)
    if checkpoint.get("checkpoint_format") != expected_format:
        raise ValueError(
            "unsupported checkpoint format: "
            f"saved={checkpoint.get('checkpoint_format')!r} "
            f"expected={expected_format!r}"
        )
    if arch != COCA_ARCH:
        if "model_state_dict" not in checkpoint:
            raise ValueError("full-model checkpoint is missing model_state_dict")
        return
    if expected_format != classifier_run.FROZEN_COCA_CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported frozen CoCa checkpoint format: {expected_format!r}")
    allowed = {
        "checkpoint_schema_version", "checkpoint_format", "head_state_dict",
        "optimizer_state_dict", "epoch", "best_val_df_f1", "history",
        "config", "class_to_idx", "rng_state", "run_identity", "val_metrics",
    }
    unexpected_fields = sorted(set(checkpoint) - allowed)
    if unexpected_fields:
        raise ValueError(f"unexpected frozen CoCa checkpoint fields: {unexpected_fields}")
    required = allowed - {"val_metrics"}
    missing_fields = sorted(required - set(checkpoint))
    if missing_fields:
        raise ValueError(f"frozen CoCa checkpoint fields missing: {missing_fields}")
    saved_head = checkpoint["head_state_dict"]
    expected_keys = set(model.head.state_dict())
    saved_keys = set(saved_head)
    missing_keys = sorted(expected_keys - saved_keys)
    unexpected_keys = sorted(saved_keys - expected_keys)
    if missing_keys or unexpected_keys:
        raise ValueError(
            f"head state keys mismatch: missing={missing_keys}, "
            f"unexpected={unexpected_keys}"
        )
    optimizer_params = [
        item
        for group in checkpoint["optimizer_state_dict"].get("param_groups", [])
        for item in group.get("params", [])
    ]
    if len(optimizer_params) != len(list(model.head.parameters())):
        raise ValueError("checkpoint optimizer does not contain exactly the linear head")


def restore_checkpoint_state(checkpoint, model, optimizer, run_identity, arch: str):
    """Restore model/optimizer state after the caller has checked run identity."""
    if arch == COCA_ARCH:
        validate_checkpoint_payload(checkpoint, model, run_identity, arch)
        model.head.load_state_dict(checkpoint["head_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        model.encoder.eval()
    else:
        if checkpoint.get("checkpoint_format") is not None:
            validate_checkpoint_payload(checkpoint, model, run_identity, arch)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        checkpoint["epoch"] + 1,
        checkpoint.get("best_val_df_f1", -1.0),
        checkpoint.get("history", []),
    )


def _ensure_durable_directory(path) -> Path:
    """Create one output directory and keep it non-empty on Drive FUSE."""
    path = Path(path)
    if not path.is_dir():
        if not path.parent.is_dir():
            raise FileNotFoundError(
                f"output parent directory is missing; refusing recursive mkdir: "
                f"{path.parent}"
            )
        path.mkdir()
    marker = path / ".directory_ready"
    if not marker.is_file():
        marker.write_text("ready\n", encoding="utf-8")
    return path


def save_checkpoint(path, model, optimizer, epoch, best_val_f1, history, args,
                    run_identity, val_metrics=None) -> None:
    """Write a checkpoint atomically (tmp + replace) so a Colab disconnect
    mid-write cannot leave a corrupt file."""
    checkpoint_format = _checkpoint_format(run_identity)
    payload = {
        "checkpoint_schema_version": 1,
        "checkpoint_format": checkpoint_format,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_df_f1": best_val_f1,
        "history": history,
        "config": vars(args),
        "class_to_idx": config.CLASS_TO_IDX,
        "rng_state": _get_rng_state(),
        "run_identity": run_identity,
    }
    if checkpoint_format == classifier_run.FROZEN_COCA_CHECKPOINT_FORMAT:
        payload["head_state_dict"] = model.head.state_dict()
    else:
        payload["model_state_dict"] = model.state_dict()
    if val_metrics is not None:
        payload["val_metrics"] = val_metrics
    path = Path(path)
    drive_tmp = path.with_suffix(path.suffix + ".tmp")
    with tempfile.NamedTemporaryFile(
        prefix=f"{path.stem}_", suffix=path.suffix, delete=False
    ) as handle:
        local_tmp = Path(handle.name)
    try:
        torch.save(payload, local_tmp)
        if not path.parent.is_dir():
            if args.run_label:
                raise FileNotFoundError(
                    "prepared checkpoint directory disappeared; refusing "
                    f"recursive mkdir: {path.parent}"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_tmp, drive_tmp)
        drive_tmp.replace(path)
        saved = _load_trusted_checkpoint(path, torch.device("cpu"))
        if saved.get("epoch") != epoch:
            raise ValueError(f"checkpoint epoch verification failed: {path}")
        if len(saved.get("history", [])) != len(history):
            raise ValueError(f"checkpoint history verification failed: {path}")
        if saved.get("run_identity") != run_identity:
            raise ValueError(f"checkpoint run identity verification failed: {path}")
        arch = run_identity.get("model_identity", {}).get("arch", "resnet18")
        validate_checkpoint_payload(saved, model, run_identity, arch)
        coca_run.checkpoint_size(path, arch=arch)
    finally:
        local_tmp.unlink(missing_ok=True)


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


def evaluate_test_scope(evaluation_scope, model, loader, device):
    if evaluation_scope == "validation_only":
        return None
    if evaluation_scope != "full":
        raise ValueError(f"unsupported evaluation scope: {evaluation_scope!r}")
    if loader is None:
        raise ValueError("full evaluation requires a test loader")
    return evaluate(model, loader, device)


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


def build_optimizer(model, learning_rate: float, weight_decay: float):
    parameters = trainable_parameters(model)
    if not parameters:
        raise ValueError("classifier has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def build_criterion(class_weighting: str, class_weights, device):
    if class_weighting == classifier_objective.CLASS_WEIGHTING_NONE:
        if class_weights is not None:
            raise ValueError("unweighted cross entropy cannot receive class weights")
        return nn.CrossEntropyLoss()
    if class_weighting not in (
        classifier_objective.CLASS_WEIGHTING_INVERSE_SQRT,
        classifier_objective.CLASS_WEIGHTING_INVERSE_FREQUENCY,
    ):
        raise ValueError(f"unsupported class weighting mode: {class_weighting!r}")
    if class_weights is None:
        raise ValueError(f"{class_weighting} requires an ordered class-weight vector")
    weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    if tuple(weights.shape) != (config.NUM_CLASSES,):
        raise ValueError(
            f"class weight tensor must have shape ({config.NUM_CLASSES},)"
        )
    return nn.CrossEntropyLoss(weight=weights)


def classifier_output_paths(base_dir, arch: str, variant: str, seed: int):
    base_dir = Path(base_dir)
    checkpoint_root = base_dir / "checkpoints"
    results_dir = base_dir / "results"
    if arch != "resnet18":
        checkpoint_root = checkpoint_root / arch
        results_dir = results_dir / arch
    return (
        checkpoint_root / f"{variant}_seed{seed}",
        results_dir,
    )


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
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--coca-pretrained", default=COCA_PRETRAINED)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--output-dir", default=None,
                   help="Base classifier dir; defaults to config.CLASSIFIER_DIR.")
    p.add_argument("--run-label", default=None,
                   help="Portable run identity for an isolated exploratory run.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last.pt in this run's checkpoint dir if present.")
    p.add_argument("--device", default=None)
    p.add_argument("--run-version", default=None)
    p.add_argument("--shared-root-uuid", default=None)
    p.add_argument("--formal-output-identity", default=None)
    p.add_argument("--fixed-split-identity", default=None)
    p.add_argument("--candidate-sha256", default=None)
    p.add_argument(
        "--class-weighting", default="none",
        choices=["none", "inverse_sqrt", "inverse_frequency"],
    )
    p.add_argument(
        "--evaluation-scope", default="full",
        choices=["full", "validation_only"],
    )
    args = p.parse_args(argv)
    if args.variant == "C4" and not args.generated_manifest:
        p.error("--variant C4 requires --generated-manifest "
                "(no default synthetic dir is read)")
    if args.run_label and not args.output_dir:
        p.error("--run-label requires an explicit isolated --output-dir")
    if args.arch not in {"resnet18", COCA_ARCH}:
        p.error(f"unsupported classifier architecture: {args.arch}")
    if args.arch == COCA_ARCH and not args.freeze_backbone:
        p.error("--arch coca_vit_b32 requires --freeze-backbone")
    if args.arch != COCA_ARCH and args.freeze_backbone:
        p.error("--freeze-backbone is only supported for coca_vit_b32")
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
    if args.run_label and args.arch == "resnet18":
        base_dir = classifier_run.require_isolated_output_dir(
            base_dir, config.EXPLORATORY_BALANCED_DDPM_DIR
        )
    source_split = "train"
    ckpt_dir, results_dir = classifier_output_paths(
        base_dir, args.arch, args.variant, args.seed
    )
    if args.run_label:
        base_dir = _ensure_durable_directory(base_dir)
        checkpoint_root = _ensure_durable_directory(base_dir / "checkpoints")
        results_root = _ensure_durable_directory(base_dir / "results")
        if args.arch != "resnet18":
            checkpoint_root = _ensure_durable_directory(checkpoint_root / args.arch)
            results_root = _ensure_durable_directory(results_root / args.arch)
        ckpt_dir = _ensure_durable_directory(
            checkpoint_root / f"{args.variant}_seed{args.seed}"
        )
        results_dir = results_root
    else:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    full_train_frame = manifests.build_classifier_frame(
        args.variant, split="train", df_target_count=args.df_target_count,
        seed=args.seed, limit=None,
        generated_manifest=args.generated_manifest,
        generated_root=args.generated_root,
    )
    training_objective, class_weights = (
        classifier_objective.build_training_objective(
            args.class_weighting, full_train_frame
        )
    )
    train_frame = full_train_frame
    if args.limit is not None:
        train_frame = train_frame.sample(
            n=min(args.limit, len(train_frame)), random_state=args.seed
        ).reset_index(drop=True)
    val_frame = manifests.load_split("val")
    test_frame = (
        manifests.load_split("test") if args.evaluation_scope == "full" else None
    )
    print(f"[data] variant={args.variant} train={len(train_frame)} "
          f"val={len(val_frame)} "
          f"test={len(test_frame) if test_frame is not None else 'not_loaded'}")
    print(f"[data] train class counts: {manifests.class_counts(train_frame)}")
    if training_objective is None:
        print("[objective] class_weighting=none")
    else:
        print("[objective] " + json.dumps(training_objective, sort_keys=True))

    model = build_model(
        arch=args.arch,
        pretrained=not args.no_pretrained,
        coca_pretrained=args.coca_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    if args.arch == COCA_ARCH:
        train_transform = model.train_preprocess
        eval_transform = model.eval_preprocess
    else:
        train_transform = eval_transform = None

    train_loader = build_dataloader(train_frame, args.img_size, args.batch_size,
                                    train=True, num_workers=args.num_workers,
                                    transform=train_transform)
    val_loader = build_dataloader(val_frame, args.img_size, args.batch_size,
                                  train=False, num_workers=args.num_workers,
                                  transform=eval_transform)
    test_loader = None
    if test_frame is not None:
        test_loader = build_dataloader(
            test_frame, args.img_size, args.batch_size, train=False,
            num_workers=args.num_workers, transform=eval_transform,
        )

    model_details = build_model_identity(
        model, args.arch, args.img_size, pretrained=not args.no_pretrained
    )
    run_identity = classifier_run.build_run_identity(
        run_label=args.run_label,
        variant=args.variant,
        seed=args.seed,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        df_target_count=args.df_target_count,
        pretrained=not args.no_pretrained,
        limit=args.limit,
        candidate_manifest=args.generated_manifest,
        source_split=source_split,
        source_manifest=config.MANIFESTS_DIR / f"{source_split}.csv",
        source_git_commit=classifier_run.git_commit(config.PROJECT_ROOT),
        model_identity=model_details,
        fixed_split_identity=args.fixed_split_identity,
        shared_root_uuid=args.shared_root_uuid,
        formal_output_identity=args.formal_output_identity,
        run_version=args.run_version,
        class_mapping=config.CLASS_TO_IDX,
        experiment_candidate_sha256=args.candidate_sha256,
        training_objective=training_objective,
        evaluation_scope=args.evaluation_scope,
    )
    print(f"[run] arch={args.arch} variant={args.variant} seed={args.seed} "
          f"epochs={args.epochs} bs={args.batch_size} lr={args.lr} device={device}")
    print(f"[identity] run_label={args.run_label!r} "
          f"candidate_sha256={run_identity['candidate_manifest_sha256']} "
          f"source_split={source_split} git_commit={run_identity['git_commit']}")
    print(f"[model] {json.dumps(model_details, sort_keys=True)}")
    criterion = build_criterion(args.class_weighting, class_weights, device)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    start_epoch = 1
    best_val_f1 = -1.0
    best_state = None
    history = []

    last_path = ckpt_dir / "last.pt"
    if args.resume and last_path.exists():
        ckpt = _load_trusted_checkpoint(last_path, device)
        saved_identity = ckpt.get("run_identity")
        if saved_identity is None:
            if args.run_label or args.variant == "C4":
                raise ValueError(
                    "checkpoint predates strict classifier run identity; "
                    "refusing unverifiable exploratory/C4 resume"
                )
            classifier_run.require_matching_legacy_config(
                ckpt.get("config", {}), vars(args)
            )
        else:
            classifier_run.require_matching_resume_identity(
                saved_identity, run_identity
            )
        start_epoch, best_val_f1, history = restore_checkpoint_state(
            ckpt, model, optimizer, run_identity, args.arch
        )
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
        print(f"[skip] already trained all {args.epochs} epochs -> evaluation")

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
            state = model.head.state_dict() if args.arch == COCA_ARCH else model.state_dict()
            best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch,
                            best_val_f1, history, args, run_identity,
                            val_metrics=val_metrics)
            marker = "  <- new best, saved best.pt"
        # refresh last.pt every epoch so a Colab disconnect can --resume
        save_checkpoint(last_path, model, optimizer, epoch,
                        best_val_f1, history, args, run_identity)
        print(f"[epoch {epoch:02d}/{args.epochs}] loss={train_loss:.4f} "
              f"val_df_f1={val_metrics['target_f1']:.4f} "
              f"val_macro_f1={val_metrics['macro_f1']:.4f} "
              f"elapsed={time.time()-t0:.0f}s checkpoint_saved=last.pt{marker}")

    if best_state is not None:
        target = model.head if args.arch == COCA_ARCH else model
        target.load_state_dict(best_state, strict=True)
    elif (ckpt_dir / "best.pt").exists():
        # e.g. resumed a run that had already finished all epochs
        best_ckpt = _load_trusted_checkpoint(ckpt_dir / "best.pt", device)
        best_identity = best_ckpt.get("run_identity")
        if best_identity is None:
            if args.run_label or args.variant == "C4":
                raise ValueError(
                    "best.pt predates strict classifier run identity; "
                    "refusing unverifiable exploratory/C4 evaluation"
                )
            classifier_run.require_matching_legacy_config(
                best_ckpt.get("config", {}), vars(args)
            )
        else:
            classifier_run.require_matching_resume_identity(
                best_identity, run_identity
            )
        restore_checkpoint_state(
            best_ckpt, model, optimizer, run_identity, args.arch
        )
    test_metrics = evaluate_test_scope(
        args.evaluation_scope, model, test_loader, device
    )
    validation_metrics = None
    if args.evaluation_scope == "full":
        print(f"[test] variant={args.variant} seed={args.seed} "
              f"df_f1={test_metrics['target_f1']:.4f} "
              f"macro_f1={test_metrics['macro_f1']:.4f} "
              f"acc={test_metrics['accuracy']:.4f}")
        recalls = {
            k: round(v, 3)
            for k, v in test_metrics["per_class_recall"].items()
        }
        print(f"[test] per-class recall: {recalls}")
    else:
        best_checkpoint = _load_trusted_checkpoint(ckpt_dir / "best.pt", device)
        classifier_run.require_matching_resume_identity(
            best_checkpoint["run_identity"], run_identity
        )
        validation_metrics = best_checkpoint.get("val_metrics")
        if validation_metrics is None:
            raise ValueError("best checkpoint is missing validation metrics")
        predicted = np.asarray(validation_metrics["confusion_matrix"]).sum(axis=0)
        predicted_counts = {
            config.CLASS_NAMES[index]: int(value)
            for index, value in enumerate(predicted)
        }
        print(
            "[validation-only] "
            f"best_df_f1={best_val_f1:.4f} "
            f"predicted_counts={predicted_counts}"
        )

    result = {
        "variant": args.variant,
        "seed": args.seed,
        "config": vars(args),
        "best_val_df_f1": best_val_f1,
        "history": history,
        "test_metrics": test_metrics,
        "validation_metrics": validation_metrics,
        "evaluation_scope": args.evaluation_scope,
        "training_objective": training_objective,
        "run_identity": run_identity,
        "checkpoint_format": run_identity["checkpoint_format"],
        "checkpoint_sizes": {
            "best_pt_bytes": coca_run.checkpoint_size(
                ckpt_dir / "best.pt", arch=args.arch
            ),
            "last_pt_bytes": coca_run.checkpoint_size(last_path, arch=args.arch),
        },
        "encoder_weights_stored": False if args.arch == COCA_ARCH else None,
        "data_counts": {
            "train": len(train_frame),
            "val": len(val_frame),
            "test": len(test_frame) if test_frame is not None else None,
        },
    }
    out_path = results_dir / f"results_{args.variant}_seed{args.seed}.json"
    if not out_path.parent.is_dir():
        if args.run_label:
            raise FileNotFoundError(
                "prepared results directory disappeared; refusing recursive "
                f"mkdir: {out_path.parent}"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[done] results -> {out_path}")
    print(f"[done] best.pt (val df_f1={best_val_f1:.4f}, for eval/deploy) + "
          f"last.pt (for --resume) -> {ckpt_dir}")


if __name__ == "__main__":
    main()
