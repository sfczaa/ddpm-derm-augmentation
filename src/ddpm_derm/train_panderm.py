"""Fine-tune PanDerm-Base on the fixed C1 train frame. Requires torch + timm.

Separate from ``train_classifier`` on purpose: this loop adds gradient
accumulation, an AMP GradScaler, a warm-up+cosine schedule stepped per optimizer
step and layer-wise LR decay, and it stores full-model checkpoints. Folding that
into the shared trainer would change the code path of the frozen ResNet-18 and
CoCa runs, so those modules are left byte-identical.

C1 real data only: no synthetic images, no sampler, no class weighting.

Example
-------
    python -m ddpm_derm.train_panderm --seed 0 --epochs 50 \
        --checkpoint /content/panderm-weights/panderm_bb_data6_checkpoint-499.pth \
        --upstream-dir /content/panderm-upstream \
        --output-dir /content/drive/MyDrive/ddpm-derm-panderm-runs/... \
        --evaluation-scope validation_only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config, manifests, metrics, panderm, panderm_run
from .classifier_run import git_commit, sha256_file
from .dataset import HAMDataset


CHECKPOINT_INTEGRITY_SCHEMA_VERSION = 1
CHECKPOINT_INTEGRITY_KEYS = {
    "schema_version",
    "checkpoint_filename",
    "byte_size",
    "sha256",
    "epoch",
    "checkpoint_format",
    "run_identity_sha256",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_rng_state() -> dict:
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def manifest_identity() -> dict[str, str]:
    """Hash only the train/validation manifests allowed in PanDerm v1."""
    return {
        split: sha256_file(config.MANIFESTS_DIR / f"{split}.csv")
        for split in ("train", "val")
    }


def build_c1_frame(seed: int, df_target_count: int, limit: int | None):
    frame = manifests.build_classifier_frame(
        "C1", split="train", df_target_count=df_target_count, seed=seed
    )
    if "source" in frame.columns and (frame["source"] != "real").any():
        raise ValueError("C1 frame must contain only real train-split rows")
    if len(frame) != panderm_run.EXPECTED_C1_TRAIN_ROWS:
        raise ValueError(
            f"C1 frame must have {panderm_run.EXPECTED_C1_TRAIN_ROWS} rows, "
            f"got {len(frame)}"
        )
    if limit is not None:
        frame = frame.sample(n=min(limit, len(frame)), random_state=seed).reset_index(
            drop=True
        )
    return frame


def build_loader(frame, transform, batch_size, train, num_workers):
    return torch.utils.data.DataLoader(
        HAMDataset(frame, transform=transform),
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _ensure_durable_directory(path) -> Path:
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


def checkpoint_integrity_path(path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".integrity.json")


def _canonical_identity_sha256(run_identity) -> str:
    panderm_run.require_expected_identity_complete(run_identity)
    encoded = json.dumps(
        dict(run_identity),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_integrity_sidecar_atomic(path, value) -> None:
    path = Path(path)
    canonical = json.loads(json.dumps(dict(value), sort_keys=True))
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(canonical, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        staged = json.loads(temporary.read_text(encoding="utf-8"))
        if staged != canonical:
            raise ValueError(f"checkpoint integrity sidecar staging failed: {path}")
        os.replace(temporary, path)
        temporary = None
        reopened = json.loads(path.read_text(encoding="utf-8"))
        if reopened != canonical:
            raise ValueError(f"checkpoint integrity sidecar reopen failed: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _build_checkpoint_integrity(path, *, epoch, run_identity) -> dict:
    path = Path(path)
    return {
        "schema_version": CHECKPOINT_INTEGRITY_SCHEMA_VERSION,
        "checkpoint_filename": path.name,
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
        "epoch": int(epoch),
        "checkpoint_format": panderm_run.CHECKPOINT_FORMAT,
        "run_identity_sha256": _canonical_identity_sha256(run_identity),
    }


def checkpoint_integrity_record(path, *, expected_identity=None) -> dict:
    """Verify sidecar and final checkpoint bytes without deserializing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {path}")
    sidecar_path = checkpoint_integrity_path(path)
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"checkpoint integrity sidecar is missing: {sidecar_path}"
        )
    try:
        record = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(
            f"checkpoint integrity sidecar is invalid: {sidecar_path}"
        ) from error
    if not isinstance(record, dict) or set(record) != CHECKPOINT_INTEGRITY_KEYS:
        raise ValueError(
            f"checkpoint integrity sidecar schema mismatch: {sidecar_path}"
        )
    if record["schema_version"] != CHECKPOINT_INTEGRITY_SCHEMA_VERSION:
        raise ValueError("checkpoint integrity schema_version mismatch")
    if record["checkpoint_filename"] != path.name:
        raise ValueError("checkpoint integrity filename mismatch")
    if (
        not isinstance(record["byte_size"], int)
        or isinstance(record["byte_size"], bool)
        or record["byte_size"] < 0
    ):
        raise ValueError("checkpoint integrity byte_size is invalid")
    if record["byte_size"] != path.stat().st_size:
        raise ValueError("checkpoint integrity byte size mismatch")
    if not panderm_run.is_pinned_sha256(record["sha256"]):
        raise ValueError("checkpoint integrity sha256 is invalid")
    if record["sha256"] != sha256_file(path):
        raise ValueError("checkpoint integrity SHA-256 mismatch")
    if (
        not isinstance(record["epoch"], int)
        or isinstance(record["epoch"], bool)
        or record["epoch"] < 0
    ):
        raise ValueError("checkpoint integrity epoch is invalid")
    if record["checkpoint_format"] != panderm_run.CHECKPOINT_FORMAT:
        raise ValueError("checkpoint integrity format mismatch")
    if not panderm_run.is_pinned_sha256(record["run_identity_sha256"]):
        raise ValueError("checkpoint integrity run identity hash is invalid")
    if (
        expected_identity is not None
        and record["run_identity_sha256"]
        != _canonical_identity_sha256(expected_identity)
    ):
        raise ValueError("checkpoint integrity run identity mismatch")
    return record


def _load_staged_checkpoint_for_save(path, *, map_location="cpu"):
    """Deserialize only a just-written temporary file during atomic save."""
    return torch.load(path, map_location=map_location, weights_only=True)


def build_checkpoint_payload(
    model,
    optimizer,
    schedule,
    scaler,
    epoch,
    best_val_f1,
    history,
    args,
    run_identity,
    val_metrics=None,
) -> dict:
    """Build the production checkpoint payload shared by save and preflight."""
    panderm_run.require_primitive_identity(run_identity)
    payload = {
        "checkpoint_schema_version": 1,
        "checkpoint_format": panderm_run.CHECKPOINT_FORMAT,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": schedule.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_val_df_f1": best_val_f1,
        "history": history,
        "config": vars(args),
        "class_to_idx": config.CLASS_TO_IDX,
        "rng_state": _get_rng_state(),
        "run_identity": run_identity,
    }
    if val_metrics is not None:
        payload["val_metrics"] = val_metrics
    return payload


def checkpoint_serialization_preflight(
    *,
    temporary_directory,
    model,
    optimizer,
    schedule,
    scaler,
    args,
    run_identity,
) -> dict:
    """Round-trip a production payload locally without creating durable artifacts."""
    temporary_directory = Path(temporary_directory)
    if not temporary_directory.is_dir():
        raise FileNotFoundError(
            f"checkpoint preflight temporary directory is missing: "
            f"{temporary_directory}"
        )
    payload = build_checkpoint_payload(
        model,
        optimizer,
        schedule,
        scaler,
        epoch=0,
        best_val_f1=-1.0,
        history=[],
        args=args,
        run_identity=run_identity,
    )
    temporary: Path | None = None
    print(
        f"[checkpoint-preflight] START directory={temporary_directory}",
        flush=True,
    )
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".panderm-checkpoint-preflight.",
            suffix=".pt",
            dir=temporary_directory,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        reopened = _load_staged_checkpoint_for_save(
            temporary, map_location="cpu"
        )
        validate_checkpoint_payload(reopened, model, expected_payload=payload)
        print("[checkpoint-preflight] COMPLETE weights_only_round_trip=true", flush=True)
        return {
            "checkpoint_format": reopened["checkpoint_format"],
            "payload_keys": sorted(reopened),
            "weights_only_round_trip": True,
            "temporary_cleanup_required": True,
        }
    except Exception as error:
        raise RuntimeError(
            f"checkpoint serialization preflight failed during save/reopen: {error}"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_checkpoint(
    path, model, optimizer, schedule, scaler, epoch, best_val_f1, history,
    args, run_identity, val_metrics=None,
) -> None:
    """Same-directory atomic write with recursive exact reopen validation."""
    payload = build_checkpoint_payload(
        model,
        optimizer,
        schedule,
        scaler,
        epoch,
        best_val_f1,
        history,
        args,
        run_identity,
        val_metrics,
    )
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"prepared checkpoint directory disappeared: {path.parent}"
        )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        staged = _load_staged_checkpoint_for_save(temporary, map_location="cpu")
        validate_checkpoint_payload(staged, model, expected_payload=payload)
        os.replace(temporary, path)
        temporary = None
        sidecar = _build_checkpoint_integrity(
            path, epoch=epoch, run_identity=run_identity
        )
        _write_integrity_sidecar_atomic(checkpoint_integrity_path(path), sidecar)
        checkpoint_integrity_record(path, expected_identity=run_identity)
        reopened = load_checkpoint_safe(
            path,
            map_location="cpu",
            model=model,
            expected_identity=run_identity,
        )
        validate_checkpoint_payload(reopened, model, expected_payload=payload)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint_safe(
    path,
    *,
    map_location="cpu",
    model=None,
    expected_identity=None,
    expected_result_checkpoint=None,
):
    """Verify final bytes and sidecar before restricted deserialization."""
    record = checkpoint_integrity_record(
        path, expected_identity=expected_identity
    )
    if (
        expected_result_checkpoint is not None
        and dict(expected_result_checkpoint) != record
    ):
        raise ValueError("result checkpoint integrity record mismatch")
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint integrity payload is not a mapping")
    if checkpoint.get("checkpoint_format") != record["checkpoint_format"]:
        raise ValueError("checkpoint integrity payload format mismatch")
    if checkpoint.get("epoch") != record["epoch"]:
        raise ValueError("checkpoint integrity payload epoch mismatch")
    run_identity = checkpoint.get("run_identity")
    if not isinstance(run_identity, dict):
        raise ValueError("checkpoint integrity payload has no run identity")
    if _canonical_identity_sha256(run_identity) != record["run_identity_sha256"]:
        raise ValueError("checkpoint integrity payload identity mismatch")
    if expected_identity is not None:
        panderm_run.require_matching_identity(run_identity, expected_identity)
    if model is not None:
        validate_checkpoint_payload(checkpoint, model)
    return checkpoint


def load_completed_checkpoint_pair_safe(
    *,
    best_path,
    last_path,
    result,
    model,
    expected_identity,
    map_location="cpu",
):
    """Verify both completed checkpoints before any caller mutates state."""
    integrity = result.get("checkpoint_integrity")
    if not isinstance(integrity, dict) or set(integrity) != {"best.pt", "last.pt"}:
        raise ValueError("completed result checkpoint integrity records are missing")
    best = load_checkpoint_safe(
        best_path,
        map_location=map_location,
        model=model,
        expected_identity=expected_identity,
        expected_result_checkpoint=integrity["best.pt"],
    )
    last = load_checkpoint_safe(
        last_path,
        map_location=map_location,
        model=model,
        expected_identity=expected_identity,
        expected_result_checkpoint=integrity["last.pt"],
    )
    return best, last


def require_recursive_exact(expected, actual, path="root") -> None:
    """Compare nested checkpoint payloads with exact type/value semantics."""
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual):
            raise ValueError(f"{path} type mismatch")
        if expected.shape != actual.shape:
            raise ValueError(f"{path} tensor shape mismatch")
        if expected.dtype != actual.dtype:
            raise ValueError(f"{path} tensor dtype mismatch")
        if not torch.equal(expected.detach().cpu(), actual.detach().cpu()):
            raise ValueError(f"{path} tensor value mismatch")
        return
    if isinstance(expected, np.ndarray):
        if type(actual) is not type(expected):
            raise ValueError(f"{path} NumPy type mismatch")
        if expected.shape != actual.shape:
            raise ValueError(f"{path} NumPy shape mismatch")
        if expected.dtype != actual.dtype:
            raise ValueError(f"{path} NumPy dtype mismatch")
        if not np.array_equal(expected, actual, equal_nan=True):
            raise ValueError(f"{path} NumPy value mismatch")
        return
    if isinstance(expected, np.generic):
        if type(actual) is not type(expected) or expected != actual:
            raise ValueError(f"{path} NumPy scalar mismatch")
        return
    if isinstance(expected, dict):
        if type(actual) is not type(expected):
            raise ValueError(f"{path} dict type mismatch")
        if set(expected) != set(actual):
            raise ValueError(f"{path} dict keys mismatch")
        for key in expected:
            require_recursive_exact(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected):
            raise ValueError(f"{path} sequence type mismatch")
        if len(expected) != len(actual):
            raise ValueError(f"{path} sequence length mismatch")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            require_recursive_exact(
                expected_item, actual_item, f"{path}[{index}]"
            )
        return
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{path} scalar type/value mismatch")


def validate_checkpoint_payload(checkpoint, model, expected_payload=None) -> None:
    """Require a complete full-model checkpoint, never a head-only one."""
    required = {
        "checkpoint_schema_version", "checkpoint_format", "model_state_dict",
        "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
        "epoch", "best_val_df_f1", "history", "config", "class_to_idx",
        "rng_state", "run_identity",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"PanDerm checkpoint fields missing: {missing}")
    if "head_state_dict" in checkpoint:
        raise ValueError("PanDerm checkpoints must store the full model, not a head")
    saved_keys = set(checkpoint["model_state_dict"])
    expected_keys = set(model.state_dict())
    missing_keys = sorted(expected_keys - saved_keys)
    unexpected_keys = sorted(saved_keys - expected_keys)
    if missing_keys or unexpected_keys:
        raise ValueError(
            f"model state keys mismatch: missing={missing_keys[:10]}, "
            f"unexpected={unexpected_keys[:10]}"
        )
    for key, expected_tensor in model.state_dict().items():
        actual_tensor = checkpoint["model_state_dict"][key]
        if not torch.is_tensor(actual_tensor):
            raise ValueError(f"model_state_dict.{key} is not a tensor")
        if actual_tensor.shape != expected_tensor.shape:
            raise ValueError(f"model_state_dict.{key} tensor shape mismatch")
        if actual_tensor.dtype != expected_tensor.dtype:
            raise ValueError(f"model_state_dict.{key} tensor dtype mismatch")
    if expected_payload is not None:
        require_recursive_exact(expected_payload, checkpoint)


def restore_checkpoint_state(checkpoint, model, optimizer, schedule, scaler):
    """Load state only after the caller has verified immutable identity."""
    validate_checkpoint_payload(checkpoint, model)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    schedule.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    panderm.assert_full_trainability(model)
    return (
        checkpoint["epoch"] + 1,
        checkpoint.get("best_val_df_f1", -1.0),
        checkpoint.get("history", []),
    )


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        y_true.extend(labels.numpy().tolist())
    summary = metrics.classification_summary(y_true, y_pred)
    panderm.require_finite(summary["target_f1"], "validation df F1")
    panderm.require_finite(summary["macro_f1"], "validation macro F1")
    return summary


def train_one_epoch(
    model, loader, optimizer, schedule, scaler, criterion, device,
    accumulation_steps, amp_enabled,
) -> tuple[float, int]:
    """One epoch of accumulated AMP steps. Returns (mean loss, optimizer steps)."""
    model.train()
    batches = len(loader)
    running, counted, steps = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    for index, (images, labels) in enumerate(loader):
        scale = panderm.accumulation_loss_scale(index, accumulation_steps, batches)
        if scale == 0.0:  # trailing micro-batches cannot complete a window
            continue
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            loss = criterion(model(images), labels)
        panderm.require_finite(loss.item(), "training loss")
        scaler.scale(loss * scale).backward()
        running += loss.item() * images.size(0)
        counted += images.size(0)
        if (index + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            panderm.require_finite_gradients(model.parameters())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            schedule.step()
            steps += 1
    return running / max(counted, 1), steps


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the PanDerm-Base C1 exploratory validation."
    )
    p.add_argument("--variant", default="C1", choices=["C1"],
                   help="C1 real duplicated df only; synthetic variants are rejected.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=panderm_run.VALIDATION_EPOCHS)
    p.add_argument("--batch-size", type=int, default=panderm_run.BATCH_SIZE)
    p.add_argument("--accumulation-steps", type=int,
                   default=panderm_run.ACCUMULATION_STEPS)
    p.add_argument("--lr", type=float, default=panderm_run.LEARNING_RATE)
    p.add_argument("--weight-decay", type=float, default=panderm_run.WEIGHT_DECAY)
    p.add_argument(
        "--warmup-epochs",
        type=int,
        default=panderm_run.VALIDATION_WARMUP_EPOCHS,
    )
    p.add_argument("--layer-decay", type=float, default=panderm_run.LAYER_DECAY)
    p.add_argument("--df-target-count", type=int,
                   default=panderm_run.DF_TARGET_COUNT)
    p.add_argument("--checkpoint", required=True,
                   help="Path to the verified PanDerm_Base pretrained checkpoint.")
    p.add_argument("--checkpoint-sha256", default=None,
                   help="Pinned SHA-256; required, verified before any training.")
    p.add_argument("--upstream-dir", required=True,
                   help="Pinned PanDerm upstream detached checkout.")
    p.add_argument("--upstream-commit", default=None,
                   help="Commit the upstream checkout is pinned to.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-version", default=panderm_run.RUN_VERSION)
    p.add_argument("--shared-root-uuid", default=None)
    p.add_argument("--formal-output-identity", default=None)
    p.add_argument("--fixed-split-identity", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--evaluation-scope", default="validation_only",
                   choices=["validation_only"])
    p.add_argument("--generated-manifest", default=None,
                   help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.generated_manifest is not None:
        p.error("PanDerm C1 is real-data only; synthetic manifests are rejected")
    frozen = {
        "seed": (0, "--seed"),
        "epochs": (panderm_run.VALIDATION_EPOCHS, "--epochs"),
        "batch_size": (panderm_run.BATCH_SIZE, "--batch-size"),
        "accumulation_steps": (
            panderm_run.ACCUMULATION_STEPS,
            "--accumulation-steps",
        ),
        "lr": (panderm_run.LEARNING_RATE, "--lr"),
        "weight_decay": (panderm_run.WEIGHT_DECAY, "--weight-decay"),
        "warmup_epochs": (
            panderm_run.VALIDATION_WARMUP_EPOCHS,
            "--warmup-epochs",
        ),
        "layer_decay": (panderm_run.LAYER_DECAY, "--layer-decay"),
    }
    for attribute, (expected, option) in frozen.items():
        if getattr(args, attribute) != expected:
            p.error(f"{option} must be exactly {expected} for PanDerm v1 validation")
    if args.run_version != panderm_run.RUN_VERSION:
        p.error(f"--run-version must be {panderm_run.RUN_VERSION}")
    if args.df_target_count != panderm_run.DF_TARGET_COUNT:
        p.error(f"--df-target-count must be {panderm_run.DF_TARGET_COUNT}")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    set_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    amp_requested = True
    amp_effective = device.type == "cuda"

    upstream_commit = args.upstream_commit or panderm_run.UPSTREAM_COMMIT
    if upstream_commit != panderm_run.UPSTREAM_COMMIT:
        raise ValueError(
            f"upstream commit mismatch: {upstream_commit!r} != "
            f"{panderm_run.UPSTREAM_COMMIT!r}"
        )
    checkpoint_sha256 = panderm_run.require_checkpoint_sha256(
        args.checkpoint,
        args.checkpoint_sha256 or panderm_run.EXPECTED_CHECKPOINT_SHA256,
    )
    panderm_run.require_no_deployment_contamination(config.PROJECT_ROOT)
    panderm_run.require_provenance_clearance(
        upstream_commit=upstream_commit,
        checkpoint_sha256=checkpoint_sha256,
        expected_checkpoint_sha256=(
            args.checkpoint_sha256 or panderm_run.EXPECTED_CHECKPOINT_SHA256
        ),
        purpose=panderm_run.VALIDATION_ONLY,
    )

    base_dir = _ensure_durable_directory(args.output_dir)
    checkpoint_root = _ensure_durable_directory(base_dir / "checkpoints")
    results_root = _ensure_durable_directory(base_dir / "results")
    checkpoint_root = _ensure_durable_directory(checkpoint_root / panderm.ARCH)
    results_dir = _ensure_durable_directory(results_root / panderm.ARCH)
    ckpt_dir = _ensure_durable_directory(
        checkpoint_root / f"{args.variant}_seed{args.seed}"
    )

    train_frame = build_c1_frame(args.seed, args.df_target_count, None)
    val_frame = manifests.load_split("val")
    print(f"[data] variant={args.variant} train={len(train_frame)} "
          f"val={len(val_frame)} test=prohibited")
    print(f"[data] train class counts: {manifests.class_counts(train_frame)}")

    train_transform = panderm.build_train_transform()
    eval_transform = panderm.build_eval_transform()
    model = panderm.build_panderm_classifier(
        checkpoint_path=args.checkpoint,
        upstream_dir=args.upstream_dir,
        drop_path=panderm_run.DROP_PATH,
    ).to(device)
    backbone_count = panderm.assert_full_trainability(model)

    train_loader = build_loader(
        train_frame, train_transform, args.batch_size, True, 2
    )
    val_loader = build_loader(
        val_frame, eval_transform, args.batch_size, False, 2
    )

    steps_per_epoch = panderm.optimizer_steps_per_epoch(
        len(train_loader), args.accumulation_steps
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = panderm.build_optimizer(
        model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        layer_decay=args.layer_decay,
    )
    covered = panderm.verify_optimizer_covers_parameters_once(optimizer, model)
    schedule = panderm.WarmupCosineSchedule(
        optimizer,
        base_lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_effective)

    model_details = panderm.model_identity(
        model,
        train_transform=train_transform,
        eval_transform=eval_transform,
        checkpoint_sha256=checkpoint_sha256,
    )
    manifest_sha256 = manifest_identity()
    run_identity = panderm_run.build_run_identity(
        git_commit=git_commit(config.PROJECT_ROOT),
        seed=args.seed,
        epochs=args.epochs,
        evaluation_scope=args.evaluation_scope,
        checkpoint_sha256=checkpoint_sha256,
        model_identity=model_details,
        manifest_sha256=manifest_sha256,
        fixed_split_identity=args.fixed_split_identity or manifest_sha256["train"],
        shared_root_uuid=args.shared_root_uuid,
        formal_output_identity=args.formal_output_identity,
        dependency_versions=panderm.dependency_versions(),
        df_target_count=args.df_target_count,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        layer_decay=args.layer_decay,
        drop_path=model_details["drop_path"],
        amp_requested=amp_requested,
        amp_effective=amp_effective,
        device_type=device.type,
        run_version=args.run_version,
    )
    print(f"[run] arch={panderm.ARCH} variant={args.variant} seed={args.seed} "
          f"epochs={args.epochs} bs={args.batch_size} "
          f"accum={args.accumulation_steps} "
          f"effective_bs={args.batch_size * args.accumulation_steps} "
          f"amp_requested={amp_requested} amp_effective={amp_effective} "
          f"device={device}")
    print(f"[identity] git_commit={run_identity['git_commit']} "
          f"upstream_commit={upstream_commit} "
          f"checkpoint_sha256={checkpoint_sha256}")
    print(f"[model] backbone_params={backbone_count} optimizer_covers={covered} "
          f"steps_per_epoch={steps_per_epoch} "
          f"total_params={model_details['total_parameter_count']} "
          f"trainable={model_details['trainable_parameter_count']}")
    print(f"[scheduler] {json.dumps(panderm.scheduler_identity(schedule))}")

    start_epoch, best_val_f1, history = 1, -1.0, []
    best_state = None
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"
    out_path = results_dir / f"results_{args.variant}_seed{args.seed}.json"
    if args.resume and last_path.exists():
        checkpoint = load_checkpoint_safe(
            last_path,
            map_location=device,
            model=model,
            expected_identity=run_identity,
        )
        if checkpoint["epoch"] > args.epochs:
            raise ValueError("last.pt epoch exceeds the fixed validation budget")
        if checkpoint["epoch"] == args.epochs:
            if not out_path.is_file() or not best_path.is_file():
                raise ValueError(
                    "completed last.pt requires matching best.pt and result JSON"
                )
            completed_result = json.loads(out_path.read_text(encoding="utf-8"))
            best_checkpoint, last_checkpoint = (
                load_completed_checkpoint_pair_safe(
                    best_path=best_path,
                    last_path=last_path,
                    result=completed_result,
                    model=model,
                    expected_identity=run_identity,
                    map_location="cpu",
                )
            )
            panderm_run.require_completed_artifact_identities(
                expected=run_identity,
                result=completed_result,
                best_checkpoint=best_checkpoint,
                last_checkpoint=last_checkpoint,
            )
            print(
                f"[skip] already completed all {args.epochs} validation epochs; "
                "result/best.pt/last.pt identities and bytes verified"
            )
            return
        # Identity, bytes and sidecar are verified before any model/optimizer/
        # scheduler/scaler state is touched.
        start_epoch, best_val_f1, history = restore_checkpoint_state(
            checkpoint, model, optimizer, schedule, scaler
        )
        rng = checkpoint.get("rng_state")
        if rng is not None:
            _set_rng_state(rng)
        print(f"[resume] found last.pt (epoch {checkpoint['epoch']}) -> continuing "
              f"from epoch {start_epoch} (best val df_f1 so far={best_val_f1:.4f}"
              f"{'' if rng is None else ', RNG restored'})")
    elif args.resume:
        print("[start] --resume set but no checkpoint yet -> fresh run from epoch 1")
    else:
        print("[start] fresh run (no --resume) from epoch 1")
    print(f"[ckpt] saving to {ckpt_dir}  (last.pt refreshed every epoch, "
          f"best.pt on val df_f1 improvement; worst-case loss: one epoch)")

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        train_loss, steps = train_one_epoch(
            model, train_loader, optimizer, schedule, scaler, criterion, device,
            args.accumulation_steps, amp_effective,
        )
        if steps != steps_per_epoch:
            raise ValueError(
                f"epoch {epoch} took {steps} optimizer steps, expected "
                f"{steps_per_epoch}"
            )
        val_metrics = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": panderm.require_finite(train_loss, "epoch train loss"),
            "val_df_f1": val_metrics["target_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": schedule.lr_at(min(schedule.step_count, schedule.total_steps - 1)),
            "optimizer_steps": schedule.step_count,
        })
        marker = ""
        # Strict improvement only, so ties keep the earliest epoch and test is
        # never consulted for selection.
        if val_metrics["target_f1"] > best_val_f1:
            best_val_f1 = val_metrics["target_f1"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            save_checkpoint(
                best_path, model, optimizer, schedule, scaler, epoch,
                best_val_f1, history, args, run_identity, val_metrics=val_metrics,
            )
            marker = "  <- new best, saved best.pt"
        save_checkpoint(
            last_path, model, optimizer, schedule, scaler, epoch, best_val_f1,
            history, args, run_identity,
        )
        print(f"[epoch {epoch:02d}/{args.epochs}] loss={train_loss:.4f} "
              f"val_df_f1={val_metrics['target_f1']:.4f} "
              f"val_macro_f1={val_metrics['macro_f1']:.4f} "
              f"steps={steps} lr={history[-1]['lr']:.3e} "
              f"elapsed={time.time()-started:.0f}s "
              f"checkpoint_saved=last.pt{marker}")

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    elif best_path.exists():
        best_checkpoint = load_checkpoint_safe(
            best_path,
            map_location=device,
            model=model,
            expected_identity=run_identity,
        )
        restore_checkpoint_state(
            best_checkpoint, model, optimizer, schedule, scaler
        )

    best_checkpoint = load_checkpoint_safe(
        best_path,
        map_location=device,
        model=model,
        expected_identity=run_identity,
    )
    validation_metrics = best_checkpoint.get("val_metrics")
    if validation_metrics is None:
        raise ValueError("best checkpoint is missing validation metrics")
    predicted = np.asarray(validation_metrics["confusion_matrix"]).sum(axis=0)
    predicted_counts = {
        config.CLASS_NAMES[index]: int(value)
        for index, value in enumerate(predicted)
    }
    print(f"[validation-only] best_df_f1={best_val_f1:.4f} "
          f"predicted_counts={predicted_counts}")

    checkpoint_integrity = {
        "best.pt": checkpoint_integrity_record(
            best_path, expected_identity=run_identity
        ),
        "last.pt": checkpoint_integrity_record(
            last_path, expected_identity=run_identity
        ),
    }
    result = {
        "variant": args.variant,
        "seed": args.seed,
        "config": vars(args),
        "best_val_df_f1": best_val_f1,
        "history": history,
        "test_metrics": None,
        "validation_metrics": validation_metrics,
        "evaluation_scope": args.evaluation_scope,
        "run_identity": run_identity,
        "checkpoint_format": panderm_run.CHECKPOINT_FORMAT,
        "checkpoint_sizes": {
            "best_pt_bytes": best_path.stat().st_size,
            "last_pt_bytes": last_path.stat().st_size,
        },
        "checkpoint_integrity": checkpoint_integrity,
        "backbone_parameter_count": backbone_count,
        "optimizer_parameter_coverage": covered,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "scheduler": panderm.scheduler_identity(schedule),
        "amp_requested": amp_requested,
        "amp_effective": amp_effective,
        "device_type": device.type,
        "formal_training_allowed": False,
        "test_access_allowed": False,
        "claim_boundary": panderm_run.CLAIM_BOUNDARY,
        "data_counts": {
            "train": len(train_frame),
            "val": len(val_frame),
            "test": None,
        },
        # Top-level duplicates of the immutable identity; drift against the
        # nested run_identity is rejected by require_identity_duplicates.
        **{key: run_identity[key] for key in panderm_run.IMMUTABLE_IDENTITY_KEYS},
    }
    panderm_run.write_json_atomic(out_path, result)
    verified_best, verified_last = load_completed_checkpoint_pair_safe(
        best_path=best_path,
        last_path=last_path,
        result=result,
        model=model,
        expected_identity=run_identity,
        map_location="cpu",
    )
    panderm_run.require_completed_artifact_identities(
        expected=run_identity,
        result=result,
        best_checkpoint=verified_best,
        last_checkpoint=verified_last,
    )
    print(f"[done] results -> {out_path}")
    print(f"[done] best.pt (val df_f1={best_val_f1:.4f}) + last.pt -> {ckpt_dir}")


if __name__ == "__main__":
    main()
