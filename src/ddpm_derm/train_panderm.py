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
import re
import shutil
import sys
import tempfile
import time
import uuid
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
    "global_step",
    "checkpoint_format",
    "run_identity_sha256",
}
CHECKPOINT_POINTER_FILENAME = "checkpoint_pointer.json"
CHECKPOINT_POINTER_SCHEMA_VERSION = 1
CHECKPOINT_POINTER_KEYS = frozenset({"schema_version", "best", "last"})
CHECKPOINT_PUBLICATION_VISIBILITY_TIMEOUT_SECONDS = 300.0
CHECKPOINT_PUBLICATION_VISIBILITY_POLL_SECONDS = 2.0
CHECKPOINT_PUBLICATION_HASH_RETRY_SECONDS = 30.0
CHECKPOINT_PUBLICATION_HEARTBEAT_SECONDS = 30.0
CHECKPOINT_PUBLICATION_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def _checkpoint_visibility_monotonic() -> float:
    return time.monotonic()


def _checkpoint_visibility_sleep(seconds: float) -> None:
    time.sleep(seconds)




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


def _set_rng_state(state: dict, *, write_guard=None) -> None:
    if write_guard is not None:
        _require_durable_write_guard(
            write_guard, "checkpoint resume Python RNG state"
        )
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if write_guard is not None:
        _require_durable_write_guard(
            write_guard, "checkpoint resume NumPy RNG state"
        )
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    if write_guard is not None:
        _require_durable_write_guard(
            write_guard, "checkpoint resume Torch RNG state"
        )
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        if write_guard is not None:
            _require_durable_write_guard(
                write_guard, "checkpoint resume CUDA RNG state"
            )
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


def checkpoint_pointer_path(ckpt_dir) -> Path:
    return Path(ckpt_dir) / CHECKPOINT_POINTER_FILENAME


def _epoch_checkpoint_filename(epoch, global_step) -> str:
    return f"epoch{int(epoch):03d}_step{int(global_step):06d}.pt"


def read_checkpoint_pointer(ckpt_dir):
    """Return the validated pointer record, or None if no pointer exists yet."""
    path = checkpoint_pointer_path(ckpt_dir)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"checkpoint pointer is invalid: {path}") from error
    if not isinstance(record, dict) or set(record) != CHECKPOINT_POINTER_KEYS:
        raise ValueError(f"checkpoint pointer schema mismatch: {path}")
    if record["schema_version"] != CHECKPOINT_POINTER_SCHEMA_VERSION:
        raise ValueError("checkpoint pointer schema_version mismatch")
    ckpt_dir = Path(ckpt_dir)
    for role in ("best", "last"):
        name = record[role]
        if (
            not isinstance(name, str)
            or not name
            or name in (".", "..")
            or "/" in name
            or "\\" in name
        ):
            raise ValueError(f"checkpoint pointer {role} filename is invalid: {name!r}")
        if not (ckpt_dir / name).is_file():
            raise FileNotFoundError(
                f"checkpoint pointer {role} target is missing: {name}"
            )
    return record


def write_checkpoint_pointer_atomic(
    ckpt_dir, *, best_filename, last_filename, write_guard
) -> None:
    record = {
        "schema_version": CHECKPOINT_POINTER_SCHEMA_VERSION,
        "best": best_filename,
        "last": last_filename,
    }
    _write_integrity_sidecar_atomic(
        checkpoint_pointer_path(ckpt_dir),
        record,
        write_guard=write_guard,
        phase="checkpoint pointer",
    )


def _canonical_identity_sha256(run_identity) -> str:
    panderm_run.require_expected_identity_complete(run_identity)
    return panderm_run.canonical_identity_sha256(run_identity)


def _require_durable_write_guard(write_guard, phase):
    if write_guard is None or not hasattr(write_guard, "require"):
        raise ValueError("durable PanDerm writes require an active-session guard")
    return write_guard.require(phase)


def _checkpoint_visibility_timeout(
    *, phase, path, started, last_observation
) -> TimeoutError:
    elapsed = _checkpoint_visibility_monotonic() - started
    return TimeoutError(
        f"{phase} visibility timeout path={Path(path).name} "
        f"elapsed={elapsed:.1f}s last_observation={last_observation}"
    )


def _sidecar_visibility_observation(record) -> str:
    if not isinstance(record, dict):
        return f"sidecar_type={type(record).__name__}"
    return (
        f"sidecar_keys={sorted(record)} "
        f"epoch={record.get('epoch')!r} "
        f"global_step={record.get('global_step')!r} "
        f"byte_size={record.get('byte_size')!r} "
        f"sha256={str(record.get('sha256', ''))[:12]}"
    )


def _read_sidecar_visibility(path):
    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None, "sidecar_missing"
    except OSError as error:
        return None, f"sidecar_read_error={type(error).__name__}:{error}"
    try:
        record = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return None, f"sidecar_invalid={type(error).__name__}:{error}"
    return record, _sidecar_visibility_observation(record)


def _wait_for_integrity_sidecar_visibility(
    path, expected_record, *, write_guard, phase
) -> None:
    path = Path(path)
    expected_record = dict(expected_record)
    started = _checkpoint_visibility_monotonic()
    deadline = started + CHECKPOINT_PUBLICATION_VISIBILITY_TIMEOUT_SECONDS
    next_heartbeat = started
    waiting = False
    last_observation = "not_observed"
    while True:
        _require_durable_write_guard(
            write_guard, f"{phase} visibility wait {path.name}"
        )
        record, last_observation = _read_sidecar_visibility(path)
        if record == expected_record:
            if waiting:
                elapsed = _checkpoint_visibility_monotonic() - started
                print(
                    f"[checkpoint-publication] {phase} COMPLETE "
                    f"path={path.name} elapsed={elapsed:.1f}s",
                    flush=True,
                )
            return
        now = _checkpoint_visibility_monotonic()
        if now >= deadline:
            raise _checkpoint_visibility_timeout(
                phase=phase,
                path=path,
                started=started,
                last_observation=last_observation,
            )
        if not waiting or now >= next_heartbeat:
            print(
                f"[checkpoint-publication] {phase} WAIT path={path.name} "
                f"elapsed={now-started:.1f}s "
                f"last_observation={last_observation} still_waiting=true",
                flush=True,
            )
            waiting = True
            next_heartbeat = now + CHECKPOINT_PUBLICATION_HEARTBEAT_SECONDS
        _checkpoint_visibility_sleep(
            min(
                CHECKPOINT_PUBLICATION_VISIBILITY_POLL_SECONDS,
                max(deadline - now, 0.0),
            )
        )


def _sha256_checkpoint_visibility(
    path,
    *,
    expected_bytes,
    write_guard,
    phase,
    started,
    deadline,
) -> str:
    path = Path(path)
    _require_durable_write_guard(
        write_guard, f"{phase} visibility SHA-256 start {path.name}"
    )
    now = _checkpoint_visibility_monotonic()
    print(
        f"[checkpoint-publication] {phase} SHA256_START path={path.name} "
        f"bytes_total={expected_bytes} elapsed={now-started:.1f}s "
        f"still_waiting=true",
        flush=True,
    )
    digest = hashlib.sha256()
    bytes_read = 0
    next_heartbeat = now + CHECKPOINT_PUBLICATION_HEARTBEAT_SECONDS
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHECKPOINT_PUBLICATION_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            bytes_read += len(chunk)
            now = _checkpoint_visibility_monotonic()
            if now >= deadline:
                raise _checkpoint_visibility_timeout(
                    phase=phase,
                    path=path,
                    started=started,
                    last_observation=(
                        f"sha256_in_progress bytes={bytes_read}/{expected_bytes}"
                    ),
                )
            if now >= next_heartbeat:
                _require_durable_write_guard(
                    write_guard,
                    f"{phase} visibility SHA-256 wait {path.name}",
                )
                print(
                    f"[checkpoint-publication] {phase} SHA256_WAIT "
                    f"path={path.name} bytes={bytes_read}/{expected_bytes} "
                    f"elapsed={now-started:.1f}s still_waiting=true",
                    flush=True,
                )
                next_heartbeat = (
                    now + CHECKPOINT_PUBLICATION_HEARTBEAT_SECONDS
                )
    _require_durable_write_guard(
        write_guard, f"{phase} visibility SHA-256 complete {path.name}"
    )
    return digest.hexdigest()


def _wait_for_checkpoint_pair_visibility(
    path,
    sidecar_path,
    expected_record,
    *,
    expected_payload,
    expected_identity,
    model,
    write_guard,
    phase,
    map_location="cpu",
):
    path = Path(path)
    sidecar_path = Path(sidecar_path)
    expected_record = dict(expected_record)
    started = _checkpoint_visibility_monotonic()
    deadline = started + CHECKPOINT_PUBLICATION_VISIBILITY_TIMEOUT_SECONDS
    next_heartbeat = started
    next_hash_at = started
    last_hashed_metadata = None
    waiting = False
    last_observation = "not_observed"
    while True:
        _require_durable_write_guard(
            write_guard, f"{phase} visibility wait {path.name}"
        )
        sidecar, sidecar_observation = _read_sidecar_visibility(sidecar_path)
        try:
            stat = path.stat()
        except FileNotFoundError:
            stat = None
            last_observation = (
                f"checkpoint_missing {sidecar_observation}"
            )
        except OSError as error:
            stat = None
            last_observation = (
                f"checkpoint_stat_error={type(error).__name__}:{error} "
                f"{sidecar_observation}"
            )
        if stat is not None:
            metadata = (
                stat.st_size,
                stat.st_mtime_ns,
                getattr(stat, "st_ino", None),
            )
            last_observation = (
                f"checkpoint_size={stat.st_size} "
                f"checkpoint_mtime_ns={stat.st_mtime_ns} "
                f"{sidecar_observation}"
            )
            now = _checkpoint_visibility_monotonic()
            if (
                sidecar == expected_record
                and stat.st_size == expected_record["byte_size"]
                and (
                    metadata != last_hashed_metadata
                    or now >= next_hash_at
                )
            ):
                try:
                    observed_sha256 = _sha256_checkpoint_visibility(
                        path,
                        expected_bytes=expected_record["byte_size"],
                        write_guard=write_guard,
                        phase=phase,
                        started=started,
                        deadline=deadline,
                    )
                except OSError as error:
                    last_observation = (
                        f"checkpoint_hash_error={type(error).__name__}:{error}"
                    )
                else:
                    last_hashed_metadata = metadata
                    next_hash_at = (
                        _checkpoint_visibility_monotonic()
                        + CHECKPOINT_PUBLICATION_HASH_RETRY_SECONDS
                    )
                    if observed_sha256 == expected_record["sha256"]:
                        _require_durable_write_guard(
                            write_guard,
                            f"{phase} visibility payload reopen {path.name}",
                        )
                        reopened = require_verified_checkpoint_payload(
                            path,
                            expected_record,
                            map_location=map_location,
                            model=model,
                            expected_identity=expected_identity,
                        )
                        validate_checkpoint_payload(
                            reopened,
                            model,
                            expected_payload=expected_payload,
                        )
                        visible_sidecar, _ = _read_sidecar_visibility(
                            sidecar_path
                        )
                        final_stat = path.stat()
                        if visible_sidecar != expected_record:
                            last_observation = (
                                "sidecar_drifted_after_payload_validation"
                            )
                        elif (
                            final_stat.st_size,
                            final_stat.st_mtime_ns,
                            getattr(final_stat, "st_ino", None),
                        ) != metadata:
                            last_observation = (
                                "checkpoint_metadata_drifted_after_payload_validation"
                            )
                        elif _checkpoint_visibility_monotonic() >= deadline:
                            last_observation = (
                                "payload_validation_exceeded_deadline"
                            )
                        else:
                            _require_durable_write_guard(
                                write_guard,
                                f"{phase} visibility complete {path.name}",
                            )
                            elapsed = (
                                _checkpoint_visibility_monotonic() - started
                            )
                            print(
                                f"[checkpoint-publication] {phase} COMPLETE "
                                f"path={path.name} elapsed={elapsed:.1f}s "
                                f"sha256={observed_sha256}",
                                flush=True,
                            )
                            return reopened
                    else:
                        last_observation = (
                            f"checkpoint_sha256_mismatch "
                            f"observed={observed_sha256[:12]} "
                            f"expected={expected_record['sha256'][:12]}"
                        )
        now = _checkpoint_visibility_monotonic()
        if now >= deadline:
            raise _checkpoint_visibility_timeout(
                phase=phase,
                path=path,
                started=started,
                last_observation=last_observation,
            )
        if not waiting or now >= next_heartbeat:
            print(
                f"[checkpoint-publication] {phase} WAIT path={path.name} "
                f"elapsed={now-started:.1f}s "
                f"last_observation={last_observation} still_waiting=true",
                flush=True,
            )
            waiting = True
            next_heartbeat = now + CHECKPOINT_PUBLICATION_HEARTBEAT_SECONDS
        _checkpoint_visibility_sleep(
            min(
                CHECKPOINT_PUBLICATION_VISIBILITY_POLL_SECONDS,
                max(deadline - now, 0.0),
            )
        )


def _write_integrity_sidecar_atomic(
    path, value, *, write_guard, phase: str = "checkpoint integrity"
) -> None:
    path = Path(path)
    canonical = json.loads(json.dumps(dict(value), sort_keys=True))
    temporary: Path | None = None
    try:
        _require_durable_write_guard(
            write_guard, f"{phase} temporary write {path.name}"
        )
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
            raise ValueError(f"{phase} sidecar staging failed: {path}")
        _require_durable_write_guard(
            write_guard, f"{phase} publish {path.name}"
        )
        os.replace(temporary, path)
        temporary = None
        _wait_for_integrity_sidecar_visibility(
            path,
            canonical,
            write_guard=write_guard,
            phase=f"{phase} publication",
        )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _build_checkpoint_integrity(
    path, *, checkpoint_filename=None, byte_size=None, sha256=None,
    epoch, global_step, run_identity,
) -> dict:
    path = Path(path)
    return {
        "schema_version": CHECKPOINT_INTEGRITY_SCHEMA_VERSION,
        "checkpoint_filename": checkpoint_filename or path.name,
        "byte_size": path.stat().st_size if byte_size is None else int(byte_size),
        "sha256": sha256_file(path) if sha256 is None else str(sha256),
        "epoch": int(epoch),
        "global_step": int(global_step),
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
    return require_checkpoint_integrity_sidecar(
        path,
        sidecar_path,
        expected_filename=path.name,
        expected_identity=expected_identity,
    )


def require_checkpoint_integrity_sidecar(
    path, sidecar_path, *, expected_filename, expected_identity=None
) -> dict:
    """The one authoritative sidecar validator for a checkpoint's final bytes."""
    path = Path(path)
    sidecar_path = Path(sidecar_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {path}")
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
    if record["checkpoint_filename"] != expected_filename:
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
    if (
        not isinstance(record["global_step"], int)
        or isinstance(record["global_step"], bool)
        or record["global_step"] < 0
    ):
        raise ValueError("checkpoint integrity global_step is invalid")
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
        "global_step": int(schedule.step_count),
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
    args, run_identity, val_metrics=None, *, write_guard,
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
    if path.exists():
        raise FileExistsError(
            f"checkpoint target already exists (write-once violation): {path}"
        )
    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"prepared checkpoint directory disappeared: {path.parent}"
        )
    local_temporary: Path | None = None
    shared_temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.runtime.",
            suffix=".pt",
            delete=False,
        ) as handle:
            local_temporary = Path(handle.name)
        torch.save(payload, local_temporary)
        with local_temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        local_staged = _load_staged_checkpoint_for_save(
            local_temporary, map_location="cpu"
        )
        validate_checkpoint_payload(local_staged, model, expected_payload=payload)

        _require_durable_write_guard(
            write_guard, f"checkpoint temporary write {path.name}"
        )
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            shared_temporary = Path(handle.name)
        with local_temporary.open("rb") as source, shared_temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        local_sha256 = sha256_file(local_temporary)
        shared_sha256 = sha256_file(shared_temporary)
        if local_sha256 != shared_sha256:
            raise ValueError("shared checkpoint staging copy hash mismatch")
        with shared_temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        candidate_record = _build_checkpoint_integrity(
            shared_temporary,
            checkpoint_filename=path.name,
            byte_size=shared_temporary.stat().st_size,
            sha256=shared_sha256,
            epoch=epoch,
            global_step=payload["global_step"],
            run_identity=run_identity,
        )
        staged = _load_staged_checkpoint_for_save(shared_temporary, map_location="cpu")
        validate_checkpoint_payload(staged, model, expected_payload=payload)
        _require_durable_write_guard(
            write_guard, f"checkpoint publish {path.name}"
        )
        os.replace(shared_temporary, path)
        shared_temporary = None
        _write_integrity_sidecar_atomic(
            checkpoint_integrity_path(path),
            candidate_record,
            write_guard=write_guard,
        )
        _wait_for_checkpoint_pair_visibility(
            path,
            checkpoint_integrity_path(path),
            candidate_record,
            expected_payload=payload,
            expected_identity=run_identity,
            model=model,
            write_guard=write_guard,
            phase="checkpoint publication",
        )
    finally:
        if local_temporary is not None:
            local_temporary.unlink(missing_ok=True)
        if shared_temporary is not None:
            shared_temporary.unlink(missing_ok=True)


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
    return require_verified_checkpoint_payload(
        path,
        record,
        map_location=map_location,
        model=model,
        expected_identity=expected_identity,
    )


def require_verified_checkpoint_payload(
    path, record, *, map_location="cpu", model=None, expected_identity=None
):
    """Deserialize restricted bytes and require exact sidecar/payload agreement."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint integrity payload is not a mapping")
    if checkpoint.get("checkpoint_format") != record["checkpoint_format"]:
        raise ValueError("checkpoint integrity payload format mismatch")
    if checkpoint.get("epoch") != record["epoch"]:
        raise ValueError("checkpoint integrity payload epoch mismatch")
    if checkpoint.get("global_step") != record["global_step"]:
        raise ValueError("checkpoint integrity payload global_step mismatch")
    history = checkpoint.get("history")
    if (
        not isinstance(history, list)
        or len(history) != checkpoint["epoch"]
        or (
            history
            and (
                history[-1].get("epoch") != checkpoint["epoch"]
                or history[-1].get("optimizer_steps") != checkpoint["global_step"]
            )
        )
    ):
        raise ValueError("checkpoint epoch/global_step/history mismatch")
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
        "epoch", "global_step", "best_val_df_f1", "history", "config", "class_to_idx",
        "rng_state", "run_identity",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"PanDerm checkpoint fields missing: {missing}")
    if "head_state_dict" in checkpoint:
        raise ValueError("PanDerm checkpoints must store the full model, not a head")
    if (
        type(checkpoint["epoch"]) is not int
        or checkpoint["epoch"] < 0
        or type(checkpoint["global_step"]) is not int
        or checkpoint["global_step"] < 0
        or not isinstance(checkpoint["history"], list)
        or len(checkpoint["history"]) != checkpoint["epoch"]
        or (
            checkpoint["history"]
            and (
                checkpoint["history"][-1].get("epoch") != checkpoint["epoch"]
                or checkpoint["history"][-1].get("optimizer_steps")
                != checkpoint["global_step"]
            )
        )
    ):
        raise ValueError("checkpoint epoch/global_step/history mismatch")
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


def restore_checkpoint_state(
    checkpoint, model, optimizer, schedule, scaler, *, write_guard
):
    """Load state only after the caller has verified immutable identity."""
    validate_checkpoint_payload(checkpoint, model)
    _require_durable_write_guard(
        write_guard, "checkpoint resume model state"
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    _require_durable_write_guard(
        write_guard, "checkpoint resume optimizer state"
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _require_durable_write_guard(
        write_guard, "checkpoint resume scheduler state"
    )
    schedule.load_state_dict(checkpoint["scheduler_state_dict"])
    _require_durable_write_guard(
        write_guard, "checkpoint resume scaler state"
    )
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

    # Constructed only after every read-only validation above, so an illegal CLI
    # invocation fails for its own reason instead of the generic missing-session
    # error, and still before the first durable write below.
    write_guard = panderm_run.SequentialSessionWriteGuard.from_environment()
    write_guard.require("durable output directory preparation")
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
    write_guard.bind_run_identity(run_identity)
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
    current_best_filename = None
    out_path = results_dir / f"results_{args.variant}_seed{args.seed}.json"
    progress_path = results_dir / f"progress_{args.variant}_seed{args.seed}.json"
    pointer = read_checkpoint_pointer(ckpt_dir)
    if args.resume and pointer is not None:
        current_best_filename = pointer["best"]
        last_path = ckpt_dir / pointer["last"]
        checkpoint = load_checkpoint_safe(
            last_path,
            map_location=device,
            model=model,
            expected_identity=run_identity,
        )
        if checkpoint["epoch"] > args.epochs:
            raise ValueError("last.pt epoch exceeds the fixed validation budget")
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            panderm_run.require_matching_identity(
                progress.get("run_identity"), run_identity
            )
            progress_position = (progress.get("epoch"), progress.get("global_step"))
            checkpoint_position = (checkpoint["epoch"], checkpoint["global_step"])
            if (
                not all(type(value) is int for value in progress_position)
                or progress_position[0] > checkpoint_position[0]
                or progress_position[1] > checkpoint_position[1]
                or (
                    progress_position == checkpoint_position
                    and progress.get("history") != checkpoint["history"]
                )
            ):
                raise ValueError(
                    "progress/checkpoint epoch, global_step, or history mismatch"
                )
        if checkpoint["epoch"] == args.epochs:
            best_path = ckpt_dir / current_best_filename
            if not out_path.is_file() or not best_path.is_file():
                raise ValueError(
                    "completed last.pt requires matching best.pt and result JSON"
                )
            completed_result = json.loads(out_path.read_text(encoding="utf-8"))
            best_checkpoint, last_checkpoint = load_completed_checkpoint_pair_safe(
                best_path=best_path,
                last_path=last_path,
                result=completed_result,
                model=model,
                expected_identity=run_identity,
                map_location="cpu",
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
        start_epoch, best_val_f1, history = restore_checkpoint_state(
            checkpoint, model, optimizer, schedule, scaler, write_guard=write_guard,
        )
        rng = checkpoint.get("rng_state")
        if rng is not None:
            _set_rng_state(rng, write_guard=write_guard)
        print(f"[resume] found {pointer['last']} (epoch {checkpoint['epoch']}) -> "
              f"continuing from epoch {start_epoch} (best val df_f1 so far="
              f"{best_val_f1:.4f}{'' if rng is None else ', RNG restored'})")
    elif args.resume:
        print("[start] --resume set but no checkpoint yet -> fresh run from epoch 1")
    else:
        print("[start] fresh run (no --resume) from epoch 1")
    print(f"[ckpt] saving to {ckpt_dir}  (one immutable file per epoch, "
          f"best/last tracked via {CHECKPOINT_POINTER_FILENAME}; "
          f"worst-case loss: one epoch)")

    for epoch in range(start_epoch, args.epochs + 1):
        write_guard.require(f"epoch {epoch} start")
        started = time.time()
        train_loss, steps = train_one_epoch(
            model, train_loader, optimizer, schedule, scaler, criterion, device,
            args.accumulation_steps, amp_effective,
        )
        if steps != steps_per_epoch:
            raise ValueError(
                f"epoch {epoch} took {steps} optimizer steps, expected {steps_per_epoch}"
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
        is_new_best = val_metrics["target_f1"] > best_val_f1
        marker = ""
        if is_new_best:
            best_val_f1 = val_metrics["target_f1"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        epoch_filename = _epoch_checkpoint_filename(epoch, schedule.step_count)
        epoch_path = ckpt_dir / epoch_filename
        save_checkpoint(
            epoch_path, model, optimizer, schedule, scaler, epoch, best_val_f1,
            history, args, run_identity, val_metrics=val_metrics,
            write_guard=write_guard,
        )
        if is_new_best:
            current_best_filename = epoch_filename
            marker = f"  <- new best, saved {epoch_filename}"
        write_checkpoint_pointer_atomic(
            ckpt_dir,
            best_filename=current_best_filename,
            last_filename=epoch_filename,
            write_guard=write_guard,
        )
        progress_record = {
            "schema_version": 1,
            "epoch": epoch,
            "global_step": int(schedule.step_count),
            "history": list(history),
            "run_identity": run_identity,
            "last_checkpoint_integrity": checkpoint_integrity_record(
                epoch_path, expected_identity=run_identity
            ),
            "checkpoint_cadence": "every_epoch",
            "maximum_quota_loss": "one_incomplete_epoch",
            "formal_training_allowed": False,
            "test_access_allowed": False,
        }
        panderm_run.write_monotonic_run_record_atomic(
            progress_path, progress_record, write_guard=write_guard.require,
        )
        print(f"[epoch {epoch:02d}/{args.epochs}] loss={train_loss:.4f} "
              f"val_df_f1={val_metrics['target_f1']:.4f} "
              f"val_macro_f1={val_metrics['macro_f1']:.4f} "
              f"steps={steps} lr={history[-1]['lr']:.3e} "
              f"elapsed={time.time()-started:.0f}s "
              f"checkpoint_saved={epoch_filename}{marker}")

    final_best_path = ckpt_dir / current_best_filename
    final_last_path = epoch_path  # last iteration's epoch_path

    if best_state is not None:
        write_guard.require("best validation state restore")
        model.load_state_dict(best_state, strict=True)
    elif final_best_path.exists():
        write_guard.require("best checkpoint before state load")
        best_checkpoint = load_checkpoint_safe(
            final_best_path, map_location=device, model=model,
            expected_identity=run_identity,
        )
        write_guard.require("best checkpoint after state load")
        restore_checkpoint_state(
            best_checkpoint, model, optimizer, schedule, scaler, write_guard=write_guard,
        )

    best_checkpoint = load_checkpoint_safe(
        final_best_path, map_location=device, model=model, expected_identity=run_identity,
    )
    validation_metrics = best_checkpoint.get("val_metrics")
    if validation_metrics is None:
        raise ValueError("best checkpoint is missing validation metrics")
    predicted = np.asarray(validation_metrics["confusion_matrix"]).sum(axis=0)
    predicted_counts = {
        config.CLASS_NAMES[index]: int(value) for index, value in enumerate(predicted)
    }
    print(f"[validation-only] best_df_f1={best_val_f1:.4f} predicted_counts={predicted_counts}")

    checkpoint_integrity = {
        "best.pt": checkpoint_integrity_record(final_best_path, expected_identity=run_identity),
        "last.pt": checkpoint_integrity_record(final_last_path, expected_identity=run_identity),
    }
    result = {
        "epoch": history[-1]["epoch"],
        "global_step": history[-1]["optimizer_steps"],
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
            "best_pt_bytes": final_best_path.stat().st_size,
            "last_pt_bytes": final_last_path.stat().st_size,
        },
        "checkpoint_integrity": checkpoint_integrity,
        "checkpoint_pointer": {"best": current_best_filename, "last": final_last_path.name},
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
        "data_counts": {"train": len(train_frame), "val": len(val_frame), "test": None},
        **{key: run_identity[key] for key in panderm_run.IMMUTABLE_IDENTITY_KEYS},
    }
    panderm_run.write_monotonic_run_record_atomic(out_path, result, write_guard=write_guard.require)
    verified_best, verified_last = load_completed_checkpoint_pair_safe(
        best_path=final_best_path, last_path=final_last_path, result=result,
        model=model, expected_identity=run_identity, map_location="cpu",
    )
    panderm_run.require_completed_artifact_identities(
        expected=run_identity, result=result,
        best_checkpoint=verified_best, last_checkpoint=verified_last,
    )
    print(f"[done] results -> {out_path}")
    print(f"[done] best={current_best_filename} (val df_f1={best_val_f1:.4f}) "
          f"+ last={final_last_path.name} -> {ckpt_dir}")


if __name__ == "__main__":
    main()
