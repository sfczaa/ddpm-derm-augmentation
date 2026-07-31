"""Torch-free identity, provenance gates and aggregation for the PanDerm run.

Kept importable without torch (like ``classifier_run`` / ``coca_run``) so the
provenance, license and contamination review can be unit-tested locally without
a deep-learning stack or a 400 MB checkpoint download.

Arch-neutral Drive helpers are re-used from ``coca_run`` rather than copied;
only the PanDerm-specific probe/aggregation live here. ``coca_run`` itself is
never modified, so the frozen CoCa artifacts keep their exact behaviour.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .classifier_run import sha256_file
from .coca_run import (  # noqa: F401  (re-exported arch-neutral helpers)
    clear_stale_marker,
    create_or_validate_sentinel,
    create_running_marker,
    ensure_tree,
    require_existing_shared_root,
    require_validation_record,
    session_marker,
    utc_now,
)

# --- pinned upstream identity ------------------------------------------------
ARCH = "panderm_base_vit_b16"
RUN_VERSION = "v1_panderm_base_c1_finetune"
CHECKPOINT_FORMAT = "panderm_full_model_v1"

UPSTREAM_REPO = "https://github.com/SiyuanYan1/PanDerm"
UPSTREAM_COMMIT = "fd7a80748ba7fc3e203fed88f909f4689d0d6f24"
UPSTREAM_MODEL_SELECTOR = "PanDerm_Base_FT"
UPSTREAM_MODEL_FACTORY = "panderm_base_patch16_224_finetune"
UPSTREAM_MODEL_MODULE = "classification/models/modeling_finetune.py"

CHECKPOINT_FILENAME = "panderm_bb_data6_checkpoint-499.pth"
CHECKPOINT_DRIVE_FILE_ID = "removed-from-public-history"
CHECKPOINT_SOURCE_URL = (
    "https://github.com/SiyuanYan1/PanDerm"
)

# Upstream publishes no digest for any checkpoint, so this is the reviewed
# trust-on-first-use digest from the Drive file id in the pinned README. Keep
# the placeholder for fail-loud regression tests and future unreviewed weights.
CHECKPOINT_SHA256_PLACEHOLDER = "REPLACE_AFTER_FIRST_DOWNLOAD"
EXPECTED_CHECKPOINT_SHA256 = (
    "be1e0fb108b3bc58721cb5195f136c948160799438f222acf1fd142230ac1ff1"
)
CHECKPOINT_SHA256_PROVENANCE = "trust_on_first_use_no_upstream_published_hash"

PAPER_DOI = "https://doi.org/10.1038/s41591-025-03747-y"
ATTRIBUTION = (
    'Yan et al., "A multimodal vision foundation model for clinical dermatology", '
    "Nature Medicine (2025). " + PAPER_DOI + " PanDerm weights used under "
    "CC BY-NC-ND 4.0, non-commercial academic research only."
)

# --- reviewed provenance verdicts (see PANDERM_BASE_C1_FINETUNE_PLAN.md) ------
LICENSE_REVIEW: dict[str, Any] = {
    "status": "reviewed_permits_noncommercial_finetuning",
    "license": "CC-BY-NC-ND-4.0",
    "license_source": "upstream README.md '## License' section at UPSTREAM_COMMIT",
    "license_full_text_url": (
        "https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode"
    ),
    "license_file_in_repo": False,
    "finetuning_allowed": True,
    "finetuning_basis": (
        "CC BY-NC-ND 4.0 section 2(a)(1): produce and reproduce, but not Share, "
        "Adapted Material for NonCommercial purposes only"
    ),
    "sharing_adapted_weights_allowed": False,
    "deployment_allowed": False,
    "attribution_required": ATTRIBUTION,
}

CONTAMINATION_REVIEW: dict[str, Any] = {
    "ham10000_in_pretraining": "private_corpus_exact_membership_unknown",
    "ham10000_in_upstream_finetuning_or_evaluation": "yes_evaluation_benchmark",
    "loaded_checkpoint_is_pretraining_only": True,
    "image_level_ham10000_overlap": "not_independently_excludable",
    "patient_level_overlap": "not_excludable",
    "independent_audit_possible": False,
    "exact_fixed_validation_test_overlap": "unproven",
    "claim_boundary": "suggestive_exploratory_only",
    "results_grade": "exploratory",
    "evidence": [
        "Pretraining corpus enumerated in Nature Medicine paper (PMC12353815) as "
        "MYM/HOP TBP, MYM+HOP dermoscopic, MMT, ACEMID, NSSI, Edu1, Edu2, "
        "ISIC2024 tiles, TCGA-SKCM, UAH89k; HAM10000 is not among them",
        "Authors state benchmark data was deliberately excluded from pretraining "
        "to avoid the leakage seen in web-sourced efforts such as SwAVDerm",
        "Upstream README lists HAM10000 as an evaluation dataset only",
        "HAM10000 draws partly on a Queensland practice and several pretraining "
        "cohorts are Australian, so patient-level overlap is not excludable",
    ],
    "unresolved_risks": [
        "pretraining corpus is private and the non-overlap claim is an "
        "unverifiable authors' assertion",
        "patient-level overlap between Queensland cohorts and HAM10000 cannot be "
        "ruled out from any published artifact",
        "upstream publishes no checkpoint hash",
    ],
}

DEPLOYMENT_ALLOWED = False
FORMAL_TRAINING_ALLOWED = False
TEST_ACCESS_ALLOWED = False
CLAIM_BOUNDARY = CONTAMINATION_REVIEW["claim_boundary"]
VALIDATION_ONLY = "validation_only"
FORMAL_TRAINING = "formal_training"
TEST_ACCESS = "test_access"
PROHIBITED_FORMAL_TEST_REASON = (
    "PanDerm v1 formal training and test access are prohibited because exact "
    "pretraining overlap is independently unauditable."
)

# --- frozen experiment configuration ----------------------------------------
VARIANT = "C1"
SEEDS = (0, 1, 2)
FORMAL_EPOCHS = 50
VALIDATION_EPOCHS = 5
VALIDATION_WARMUP_EPOCHS = 5
BATCH_SIZE = 16
ACCUMULATION_STEPS = 8
EFFECTIVE_BATCH_SIZE = 128
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 10
LAYER_DECAY = 0.65
DROP_PATH = 0.2
DF_TARGET_COUNT = 585
INPUT_RESOLUTION = 224
MIN_LR = 1e-6

EXPECTED_SPLIT_COUNTS = {"train": 6995, "val": 1510, "test": 1510}
EXPECTED_SPLIT_DF_COUNTS = {"train": 85, "val": 14, "test": 16}
EXPECTED_C1_CLASS_COUNTS = {
    "akiec": 226, "bcc": 348, "bkl": 778, "df": 585,
    "mel": 782, "nv": 4684, "vasc": 92,
}
EXPECTED_C1_TRAIN_ROWS = 7495
EXPECTED_VALIDATION_IMAGE_COUNT = 8505
VALIDATION_ARCHIVE_FILENAME = "ham10000_train_val_only_v1.tar"
VALIDATION_ARCHIVE_IDENTITY_FILENAME = "archive_identity.json"
VALIDATION_ARCHIVE_READY_FILENAME = "_READY.json"
VALIDATION_ARCHIVE_CACHE_FORMAT = "ham10000_train_val_only_tar_v1"
VALIDATION_ARCHIVE_SCHEMA_VERSION = 1

# Approved content identity of the train/val archive payload: the canonical
# aggregate over every member's (POSIX relative path, exact byte size, SHA-256).
#
# This is the independent trust anchor. Every other content witness -- the tar
# bytes, the persisted per-file rows, the aggregate inside archive_identity.json,
# the archive SHA-256 and _READY.json -- lives inside the cache directory and can
# all be rewritten together, so validating them only against each other proves
# nothing. This constant is Git-reviewed and must never be read back from the
# artifact under validation.
#
# Derived from the fixed train/val split at
#   train.csv     eea3fdf281120687b45dfcb5139d927888c6dc84867c67f053c7a743eb02b6fa
#   val.csv       22a87a1ab4009c9e87462381f9ef35ad7a5eae7217057049fc24e5531df819f4
#   class_to_idx  5a034b7dc0c6f44543f558aa589b8e1cba12a05b71a18ff0e2d2029a2ad2e66c
# over 8508 members (8505 images + 3 manifest members), 2352169696 bytes.
VALIDATION_CONTENT_IDENTITY_PLACEHOLDER = "REPLACE_AFTER_CONTENT_IDENTITY_PIN"
EXPECTED_VALIDATION_CONTENT_IDENTITY_SHA256 = (
    "189c4c1fc630cb1717cd3b87390b9dbff2e412fd2c6eefab525ff057760fe74e"
)
EXPECTED_VALIDATION_CONTENT_MEMBER_COUNT = 8508
EXPECTED_CLASS_TO_IDX = {
    name: index for index, name in enumerate(EXPECTED_C1_CLASS_COUNTS)
}

# Frozen historical ResNet-18 matched-585 benchmark. Descriptive comparison
# only: different architecture, resolution, optimizer, schedule and budget.
FROZEN_HISTORICAL_C1 = {
    "arch": "resnet18",
    "variant": "C1",
    "test_df_f1_mean": 0.660,
    "test_df_f1_population_std": 0.042,
    "df_target_count": 585,
    "comparison_kind": "descriptive_cross_experiment_only",
    "not_comparable_because": (
        "different architecture, input resolution, optimizer, schedule and "
        "epoch budget; no significance testing is performed"
    ),
}

IMMUTABLE_IDENTITY_KEYS = (
    "schema_version",
    "git_commit",
    "run_version",
    "upstream_repo",
    "upstream_commit",
    "checkpoint_filename",
    "checkpoint_source_url",
    "checkpoint_sha256",
    "checkpoint_sha256_provenance",
    "checkpoint_format",
    "model_identity",
    "variant",
    "seed",
    "fixed_split_identity",
    "manifest_sha256",
    "c1_construction",
    "objective",
    "optimization",
    "dependency_versions",
    "shared_root_uuid",
    "formal_output_identity",
    "evaluation_scope",
    "license_review",
    "contamination_review",
    "claim_boundary",
)

NON_COLLAPSE_CHECK_KEYS = (
    "finite_losses",
    "finite_metrics",
    "best_validation_df_f1_positive",
    "predicted_df_positive",
    "at_least_two_predicted_classes",
    "not_all_nv",
    "not_all_df",
    "backbone_gradient_verified",
    "backbone_parameters_updated",
    "identity_complete",
    "test_metrics_null",
    "no_test_access",
    "provenance_allows_next_stage",
)


def is_pinned_sha256(value: Any) -> bool:
    """True only for a real 64-char lowercase hex digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def write_json_atomic(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    write_guard: Callable[[str], Any] | None = None,
) -> None:
    """Write one new JSON record atomically, then re-open it exactly."""
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"JSON parent directory is missing: {path.parent}")
    if path.exists() and path.stat().st_size:
        raise FileExistsError(f"refusing to overwrite existing JSON record: {path}")

    canonical = json.loads(json.dumps(dict(value), sort_keys=True))
    temporary: Path | None = None
    try:
        if write_guard is not None:
            write_guard(f"JSON temporary write {path.name}")
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
            raise ValueError(f"temporary JSON verification failed: {temporary}")
        if write_guard is not None:
            write_guard(f"JSON publish {path.name}")
        os.replace(temporary, path)
        temporary = None
        reopened = json.loads(path.read_text(encoding="utf-8"))
        if reopened != canonical:
            raise ValueError(f"JSON replace/read verification failed: {path}")
    finally:
        if temporary is not None:
            if write_guard is None:
                temporary.unlink(missing_ok=True)


def write_monotonic_run_record_atomic(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    write_guard: Callable[[str], Any],
) -> None:
    """Publish one run history without rollback, truncation, or cross-run drift."""
    path = Path(path)
    record = json.loads(json.dumps(dict(value), allow_nan=False, sort_keys=True))
    required = {"epoch", "global_step", "history", "run_identity"}
    if not required.issubset(record):
        raise ValueError("monotonic run record is missing identity/progress fields")
    epoch = record["epoch"]
    global_step = record["global_step"]
    history = record["history"]
    if (
        type(epoch) is not int
        or epoch < 0
        or type(global_step) is not int
        or global_step < 0
        or not isinstance(history, list)
        or len(history) != epoch
        or (
            history
            and (
                history[-1].get("epoch") != epoch
                or history[-1].get("optimizer_steps") != global_step
            )
        )
    ):
        raise ValueError("run record epoch/global_step/history mismatch")
    require_expected_identity_complete(record["run_identity"])
    previous_bytes: bytes | None = None
    if path.exists():
        previous_bytes = path.read_bytes()
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or not required.issubset(existing):
            raise ValueError("existing run record schema is invalid")
        require_matching_identity(
            existing["run_identity"], record["run_identity"]
        )
        existing_position = (existing["epoch"], existing["global_step"])
        new_position = (epoch, global_step)
        if (
            new_position[0] < existing_position[0]
            or new_position[1] < existing_position[1]
            or history[: len(existing["history"])] != existing["history"]
        ):
            raise ValueError(
                "run record rollback or history truncation rejected: "
                f"existing={existing_position} new={new_position}"
            )
        if new_position == existing_position:
            if existing != record:
                raise ValueError(
                    "same-step run record differs; refusing overwrite"
                )
            return
    temporary: Path | None = None
    published = False
    try:
        write_guard(f"run record temporary write {path.name}")
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
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if json.loads(temporary.read_text(encoding="utf-8")) != record:
            raise ValueError("run record temporary reopen mismatch")
        write_guard(f"run record publish {path.name}")
        os.replace(temporary, path)
        temporary = None
        published = True
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError("run record final reopen mismatch")
    except Exception:
        if published:
            write_guard(f"run record restore previous {path.name}")
            if previous_bytes is None:
                path.unlink(missing_ok=True)
            else:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{path.name}.restore.",
                    suffix=".tmp",
                    dir=path.parent,
                    delete=False,
                ) as handle:
                    restore = Path(handle.name)
                    handle.write(previous_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.replace(restore, path)
                    if path.read_bytes() != previous_bytes:
                        raise ValueError("run record previous-version restore failed")
                finally:
                    restore.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _canonical_manifest_image_path(value: Any) -> str:
    raw = str(value).strip()
    normalized = raw.replace("\\", "/")
    if (
        not raw
        or PurePosixPath(normalized).is_absolute()
        or PureWindowsPath(raw).is_absolute()
    ):
        raise ValueError(f"manifest image_path must be relative: {raw!r}")
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"manifest image_path is not canonical: {raw!r}")
    return "/".join(parts)


def _sorted_string_sha256(values: Sequence[str]) -> str:
    encoded = "".join(f"{value}\n" for value in sorted(values)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_file_content_rows_sha256(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    canonical_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {"path", "size_bytes", "sha256"}:
            raise ValueError(f"file content row {index} schema mismatch")
        path = _canonical_manifest_image_path(row["path"])
        if path != row["path"] or path.casefold() in seen:
            raise ValueError(f"file content row {index} path mismatch")
        seen.add(path.casefold())
        size_bytes = row["size_bytes"]
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError(f"file content row {index} size mismatch")
        digest = row["sha256"]
        if not is_pinned_sha256(digest):
            raise ValueError(f"file content row {index} SHA-256 mismatch")
        canonical_rows.append(
            {"path": path, "size_bytes": size_bytes, "sha256": digest}
        )
    if [row["path"] for row in canonical_rows] != sorted(
        row["path"] for row in canonical_rows
    ):
        raise ValueError("file content rows are not in canonical sorted order")
    encoded = json.dumps(
        canonical_rows,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_approved_content_identity(value: Any) -> str:
    """Accept only a real, reviewed, non-placeholder approved content digest.

    Callers must pass this in explicitly. It deliberately has no default and is
    never read back from the tar, ``archive_identity.json``, ``_READY.json``, the
    cache directory or the extracted tree: an expected value supplied by the
    artifact under validation would be rewritten together with everything else it
    is supposed to police.
    """
    if value is None:
        raise ValueError(
            "approved expected file content identity is required; it must come "
            "from the reviewed constant, never from the artifact being validated"
        )
    if value == VALIDATION_CONTENT_IDENTITY_PLACEHOLDER:
        raise ValueError(
            "approved expected file content identity is still the placeholder "
            f"{VALIDATION_CONTENT_IDENTITY_PLACEHOLDER!r}; run the one-off "
            "bootstrap source-hash step and pin the reviewed digest before any "
            "archive publish, reuse, attempt or runner start"
        )
    if not is_pinned_sha256(value):
        raise ValueError(
            "approved expected file content identity must be 64 lowercase hex "
            f"characters, got {value!r}"
        )
    return value


def _require_content_identity_matches_approved(
    observed: Mapping[str, Any],
    approved_sha256: str,
    *,
    witness: str,
) -> None:
    """Every witness (source, tar, extracted tree, persisted rows) must agree."""
    recomputed = _canonical_file_content_rows_sha256(observed["rows"])
    if recomputed != observed["sha256"]:
        raise ValueError(
            f"{witness} content rows do not hash to their own aggregate: "
            f"{recomputed} != {observed['sha256']}"
        )
    if recomputed != approved_sha256:
        raise ValueError(
            f"{witness} content identity does not match the approved expected "
            f"identity: {recomputed} != {approved_sha256}"
        )


def _file_content_identity_from_sources(
    member_sources: Mapping[str, str | Path],
    *,
    phase: str,
) -> dict[str, Any]:
    names = sorted(member_sources)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_report = started
    completed_bytes = 0
    print(f"[{phase}] START files_total={len(names)}", flush=True)
    for index, name in enumerate(names, start=1):
        canonical = _canonical_manifest_image_path(name)
        if canonical != name:
            raise ValueError(f"non-canonical file content path: {name!r}")
        source = Path(member_sources[name])
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"file content source must be a regular file: {source}")
        digest = hashlib.sha256()
        size_bytes = 0
        with source.open("rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size_bytes += len(chunk)
                now = time.perf_counter()
                if now - last_report >= 60:
                    elapsed = now - started
                    print(
                        f"[{phase}] files={index - 1}/{len(names)} "
                        f"bytes={completed_bytes + size_bytes} "
                        f"elapsed={elapsed:.1f}s current_file={name}",
                        flush=True,
                    )
                    last_report = now
        rows.append(
            {
                "path": name,
                "size_bytes": size_bytes,
                "sha256": digest.hexdigest(),
            }
        )
        completed_bytes += size_bytes
        now = time.perf_counter()
        if index % 250 == 0 or index == len(names) or now - last_report >= 60:
            elapsed = now - started
            rate = index / elapsed if elapsed > 0 else float("inf")
            eta = (len(names) - index) / rate if rate > 0 else float("inf")
            print(
                f"[{phase}] files={index}/{len(names)} bytes={completed_bytes} "
                f"elapsed={elapsed:.1f}s rate={rate:.1f}files/s "
                f"eta={eta:.1f}s current_file={name}",
                flush=True,
            )
            last_report = now
    return {
        "rows": rows,
        "sha256": _canonical_file_content_rows_sha256(rows),
    }


def _file_content_identity_from_tar(
    archive_path: str | Path,
    *,
    expected_members: Sequence[str],
    phase: str,
) -> dict[str, Any]:
    expected = sorted(expected_members)
    _validated_tar_members(archive_path, expected_members=expected)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_report = started
    completed_bytes = 0
    print(f"[{phase}] START files_total={len(expected)}", flush=True)
    with tarfile.open(archive_path, mode="r:") as archive:
        members = {member.name: member for member in archive.getmembers()}
        for index, name in enumerate(expected, start=1):
            member = members[name]
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"regular tar member has no payload: {name}")
            digest = hashlib.sha256()
            size_bytes = 0
            with source:
                while True:
                    chunk = source.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size_bytes += len(chunk)
                    now = time.perf_counter()
                    if now - last_report >= 60:
                        elapsed = now - started
                        print(
                            f"[{phase}] files={index - 1}/{len(expected)} "
                            f"bytes={completed_bytes + size_bytes} "
                            f"elapsed={elapsed:.1f}s current_file={name}",
                            flush=True,
                        )
                        last_report = now
            if size_bytes != member.size:
                raise ValueError(
                    f"tar member byte count mismatch for {name}: "
                    f"{size_bytes} != {member.size}"
                )
            rows.append(
                {
                    "path": name,
                    "size_bytes": size_bytes,
                    "sha256": digest.hexdigest(),
                }
            )
            completed_bytes += size_bytes
            now = time.perf_counter()
            if (
                index % 250 == 0
                or index == len(expected)
                or now - last_report >= 60
            ):
                elapsed = now - started
                rate = index / elapsed if elapsed > 0 else float("inf")
                eta = (
                    (len(expected) - index) / rate
                    if rate > 0 else float("inf")
                )
                print(
                    f"[{phase}] files={index}/{len(expected)} "
                    f"bytes={completed_bytes} elapsed={elapsed:.1f}s "
                    f"rate={rate:.1f}files/s eta={eta:.1f}s "
                    f"current_file={name}",
                    flush=True,
                )
                last_report = now
    return {
        "rows": rows,
        "sha256": _canonical_file_content_rows_sha256(rows),
    }


def validation_source_inventory(
    shared_data_root: str | Path,
    *,
    expected_train_rows: int = EXPECTED_SPLIT_COUNTS["train"],
    expected_val_rows: int = EXPECTED_SPLIT_COUNTS["val"],
    expected_unique_images: int = EXPECTED_VALIDATION_IMAGE_COUNT,
) -> dict[str, Any]:
    """Read only train/val allowlists and return their exact source inventory."""
    shared_data_root = Path(shared_data_root).resolve(strict=True)
    inventory_started = time.perf_counter()
    print(
        f"[validation-inventory] START source={shared_data_root}",
        flush=True,
    )
    manifests_root = shared_data_root / "manifests"
    manifest_paths = {
        split: manifests_root / f"{split}.csv" for split in ("train", "val")
    }
    class_mapping_path = manifests_root / "class_to_idx.json"
    for required in (*manifest_paths.values(), class_mapping_path):
        if required.is_symlink() or not required.is_file():
            raise FileNotFoundError(
                f"required regular validation source file missing: {required}"
            )

    class_mapping = json.loads(class_mapping_path.read_text(encoding="utf-8"))
    if class_mapping != EXPECTED_CLASS_TO_IDX:
        raise ValueError(
            f"class mapping mismatch: {class_mapping!r} != "
            f"{EXPECTED_CLASS_TO_IDX!r}"
        )

    split_paths: dict[str, list[str]] = {}
    image_sources: dict[str, Path] = {}
    canonical_destinations: set[str] = set()
    expected_rows = {
        "train": int(expected_train_rows),
        "val": int(expected_val_rows),
    }
    for split, manifest_path in manifest_paths.items():
        rows: list[str] = []
        split_started = time.perf_counter()
        last_report = split_started
        print(
            f"[validation-inventory] scan={split} START manifest={manifest_path}",
            flush=True,
        )
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "image_path" not in reader.fieldnames:
                raise ValueError(f"{manifest_path} is missing image_path")
            for row_number, row in enumerate(reader, start=2):
                relative = _canonical_manifest_image_path(row.get("image_path"))
                destination_key = relative.casefold()
                if destination_key in canonical_destinations:
                    raise ValueError(
                        f"duplicate validation archive destination at "
                        f"{manifest_path}:{row_number}: {relative}"
                    )
                canonical_destinations.add(destination_key)
                unresolved = shared_data_root / Path(*relative.split("/"))
                if unresolved.is_symlink():
                    raise ValueError(
                        f"validation archive source may not be a symlink: {relative}"
                    )
                source = unresolved.resolve(strict=True)
                try:
                    source.relative_to(shared_data_root)
                except ValueError as error:
                    raise ValueError(
                        f"validation archive source escapes shared root: {relative}"
                    ) from error
                if not source.is_file():
                    raise FileNotFoundError(
                        f"validation archive source is not a file: {source}"
                    )
                rows.append(relative)
                image_sources[relative] = source
                now = time.perf_counter()
                if len(rows) % 250 == 0 or now - last_report >= 60:
                    elapsed = now - split_started
                    rate = len(rows) / elapsed if elapsed > 0 else float("inf")
                    eta = (
                        (expected_rows[split] - len(rows)) / rate
                        if rate > 0 else float("inf")
                    )
                    print(
                        f"[validation-inventory] scan={split} "
                        f"rows={len(rows)}/{expected_rows[split]} "
                        f"elapsed={elapsed:.1f}s rate={rate:.1f}files/s "
                        f"eta={eta:.1f}s "
                        f"current_file={relative}",
                        flush=True,
                    )
                    last_report = now
        if len(rows) != expected_rows[split]:
            raise ValueError(
                f"{split} manifest row count mismatch: "
                f"{len(rows)} != {expected_rows[split]}"
            )
        split_paths[split] = rows
        print(
            f"[validation-inventory] scan={split} COMPLETE "
            f"rows={len(rows)} elapsed={time.perf_counter() - split_started:.1f}s",
            flush=True,
        )

    if len(image_sources) != int(expected_unique_images):
        raise ValueError(
            f"unique train/val image count mismatch: "
            f"{len(image_sources)} != {expected_unique_images}"
        )
    member_sources = {
        "manifests/train.csv": manifest_paths["train"],
        "manifests/val.csv": manifest_paths["val"],
        "manifests/class_to_idx.json": class_mapping_path,
        **image_sources,
    }
    member_names = sorted(member_sources)
    file_content_identity = _file_content_identity_from_sources(
        member_sources,
        phase="validation-inventory-content",
    )
    inventory = {
        "shared_data_root": shared_data_root,
        "manifest_paths": manifest_paths,
        "class_mapping_path": class_mapping_path,
        "class_mapping": class_mapping,
        "split_relative_paths": split_paths,
        "image_sources": image_sources,
        "member_sources": member_sources,
        "member_names": member_names,
        "file_content_identity": file_content_identity,
        "train_rows": len(split_paths["train"]),
        "val_rows": len(split_paths["val"]),
        "unique_images": len(image_sources),
        "manifest_sha256": {
            split: sha256_file(path)
            for split, path in manifest_paths.items()
        },
        "class_mapping_sha256": sha256_file(class_mapping_path),
        "canonical_sorted_member_list_sha256": _sorted_string_sha256(
            member_names
        ),
        "exact_relative_file_set_identity": _sorted_string_sha256(
            list(image_sources)
        ),
        "test_manifest_included": False,
        "whole_data_copy_used": False,
    }
    print(
        f"[validation-inventory] COMPLETE images={len(image_sources)} "
        f"elapsed={time.perf_counter() - inventory_started:.1f}s",
        flush=True,
    )
    return inventory


def stage_train_smoke_sample(
    shared_data_root: str | Path,
    sample_directory: str | Path,
    *,
    count: int = 4,
) -> dict[str, Any]:
    """Deterministically copy only fixed train-manifest images for GPU smoke."""
    shared_data_root = Path(shared_data_root).resolve(strict=True)
    sample_directory = Path(sample_directory)
    if sample_directory.exists():
        raise FileExistsError(f"smoke sample directory already exists: {sample_directory}")
    train_manifest = shared_data_root / "manifests" / "train.csv"
    if train_manifest.is_symlink() or not train_manifest.is_file():
        raise FileNotFoundError(f"train manifest missing: {train_manifest}")
    relative_paths: list[str] = []
    with train_manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "image_path" not in reader.fieldnames:
            raise ValueError(f"{train_manifest} is missing image_path")
        for row in reader:
            relative_paths.append(
                _canonical_manifest_image_path(row.get("image_path"))
            )
    selected = sorted(set(relative_paths))[: int(count)]
    if len(selected) != int(count):
        raise ValueError(
            f"train manifest has only {len(selected)} unique images; "
            f"{count} required"
        )
    sample_directory.mkdir()
    try:
        for index, relative in enumerate(selected, start=1):
            unresolved = shared_data_root / Path(*relative.split("/"))
            if unresolved.is_symlink():
                raise ValueError(f"smoke sample source may not be a symlink: {relative}")
            source = unresolved.resolve(strict=True)
            source.relative_to(shared_data_root)
            if not source.is_file():
                raise FileNotFoundError(f"smoke sample source is not a file: {source}")
            destination = sample_directory / Path(*relative.split("/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            print(
                f"[gpu-smoke-sample] copied={index}/{count} "
                f"source_manifest=train.csv relative_path={relative}",
                flush=True,
            )
    except Exception:
        shutil.rmtree(sample_directory, ignore_errors=True)
        raise
    return {
        "source_manifest": "manifests/train.csv",
        "relative_paths": selected,
        "sample_count": len(selected),
        "test_manifest_read": False,
        "whole_data_copy_used": False,
    }


def _validated_tar_members(
    archive_path: str | Path,
    *,
    expected_members: Sequence[str] | None = None,
) -> list[tarfile.TarInfo]:
    archive_path = Path(archive_path)
    seen: set[str] = set()
    validated: list[tarfile.TarInfo] = []
    with tarfile.open(archive_path, mode="r:") as archive:
        for member in archive.getmembers():
            canonical = _canonical_manifest_image_path(member.name)
            key = canonical.casefold()
            if key in seen:
                raise ValueError(
                    f"duplicate canonical tar member path: {canonical}"
                )
            seen.add(key)
            if member.name != canonical:
                raise ValueError(f"non-canonical tar member path: {member.name!r}")
            if not member.isreg():
                raise ValueError(
                    f"tar member must be a regular file: {member.name} "
                    f"type={member.type!r}"
                )
            validated.append(member)
    if expected_members is not None:
        expected = sorted(expected_members)
        actual = sorted(member.name for member in validated)
        if actual != expected:
            raise ValueError(
                "validation archive member allowlist mismatch: "
                f"missing={sorted(set(expected) - set(actual))[:10]} "
                f"extra={sorted(set(actual) - set(expected))[:10]}"
            )
    return validated


def _safe_extract_validation_archive(
    archive_path: str | Path,
    destination: str | Path,
    *,
    expected_members: Sequence[str],
    phase: str = "archive-extract",
) -> None:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"archive extraction destination exists: {destination}")
    members = _validated_tar_members(
        archive_path, expected_members=expected_members
    )
    destination.mkdir()
    started = time.perf_counter()
    last_report = started
    extracted_bytes = 0
    print(
        f"[{phase}] START members_total={len(members)}",
        flush=True,
    )
    with tarfile.open(archive_path, mode="r:") as archive:
        for index, member in enumerate(members, start=1):
            target = destination / Path(*member.name.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"regular tar member has no payload: {member.name}")
            with source, target.open("xb") as handle:
                shutil.copyfileobj(source, handle, length=8 * 1024 * 1024)
            extracted_bytes += member.size
            now = time.perf_counter()
            if index % 250 == 0 or index == len(members) or now - last_report >= 60:
                elapsed = now - started
                rate = index / elapsed if elapsed > 0 else float("inf")
                eta = (
                    (len(members) - index) / rate
                    if rate > 0 else float("inf")
                )
                print(
                    f"[{phase}] members={index}/{len(members)} "
                    f"bytes={extracted_bytes} elapsed={elapsed:.1f}s "
                    f"rate={rate:.1f}files/s eta={eta:.1f}s "
                    f"current_file={member.name}",
                    flush=True,
                )
                last_report = now


def _copy_file_with_progress(
    source: str | Path,
    destination: str | Path,
    *,
    phase: str,
    write_guard: Callable[[str], Any] | None = None,
) -> None:
    source = Path(source)
    destination = Path(destination)
    total = source.stat().st_size
    copied = 0
    started = time.perf_counter()
    last_report = started
    print(
        f"[{phase}] START file={source.name} bytes_total={total}",
        flush=True,
    )
    if write_guard is not None:
        write_guard(f"{phase} destination create")
    with source.open("rb") as reader, destination.open("xb") as writer:
        while True:
            chunk = reader.read(8 * 1024 * 1024)
            if not chunk:
                break
            if write_guard is not None:
                write_guard(f"{phase} chunk {copied}")
            writer.write(chunk)
            copied += len(chunk)
            now = time.perf_counter()
            if copied == total or now - last_report >= 60:
                elapsed = now - started
                rate = copied / elapsed if elapsed > 0 else float("inf")
                eta = (total - copied) / rate if rate > 0 else float("inf")
                print(
                    f"[{phase}] bytes={copied}/{total} elapsed={elapsed:.1f}s "
                    f"rate={rate:.1f}B/s eta={eta:.1f}s file={source.name}",
                    flush=True,
                )
                last_report = now
        writer.flush()
        os.fsync(writer.fileno())
    if copied != total:
        raise ValueError(f"{phase} byte count mismatch: {copied} != {total}")


def _validate_extracted_validation_data(
    extracted_root: str | Path,
    identity: Mapping[str, Any],
    *,
    approved_file_content_identity_sha256: str,
) -> dict[str, Any]:
    approved = require_approved_content_identity(
        approved_file_content_identity_sha256
    )
    inventory = validation_source_inventory(
        extracted_root,
        expected_train_rows=int(identity["train_rows"]),
        expected_val_rows=int(identity["val_rows"]),
        expected_unique_images=int(identity["unique_images"]),
    )
    actual_files = sorted(
        path.relative_to(extracted_root).as_posix()
        for path in Path(extracted_root).rglob("*")
        if path.is_file()
    )
    if actual_files != inventory["member_names"]:
        raise ValueError("extracted validation archive contains an unexpected file set")
    comparisons = {
        "canonical_sorted_member_list_sha256":
            inventory["canonical_sorted_member_list_sha256"],
        "train_manifest_sha256": inventory["manifest_sha256"]["train"],
        "val_manifest_sha256": inventory["manifest_sha256"]["val"],
        "class_mapping_sha256": inventory["class_mapping_sha256"],
        "exact_relative_file_set_identity":
            inventory["exact_relative_file_set_identity"],
        "file_content_identity_sha256":
            inventory["file_content_identity"]["sha256"],
        "file_content_rows": inventory["file_content_identity"]["rows"],
    }
    for key, actual in comparisons.items():
        if actual != identity[key]:
            raise ValueError(f"extracted validation archive {key} mismatch")
    # Witness C: the extracted tree, judged against the approved constant as
    # well, so a tar/identity/READY rewritten together cannot certify itself.
    _require_content_identity_matches_approved(
        inventory["file_content_identity"],
        approved,
        witness="extracted validation archive tree",
    )
    return inventory


def build_validation_archive_cache(
    shared_data_root: str | Path,
    cache_directory: str | Path,
    runtime_temporary_root: str | Path,
    *,
    expected_file_content_identity_sha256: str,
    source_fixed_split_identity: str,
    expected_train_rows: int = EXPECTED_SPLIT_COUNTS["train"],
    expected_val_rows: int = EXPECTED_SPLIT_COUNTS["val"],
    expected_unique_images: int = EXPECTED_VALIDATION_IMAGE_COUNT,
    write_guard: Callable[[str], Any],
) -> dict[str, Any]:
    """Build and verify one immutable train/val-only tar, then write READY last."""
    approved_content_sha256 = require_approved_content_identity(
        expected_file_content_identity_sha256
    )
    cache_directory = Path(cache_directory)
    runtime_temporary_root = Path(runtime_temporary_root).resolve(strict=True)
    if not callable(write_guard):
        raise ValueError("archive build requires an active-session write_guard")
    if cache_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite existing validation archive cache: "
            f"{cache_directory}"
        )
    if not cache_directory.parent.is_dir():
        raise FileNotFoundError(
            f"validation archive cache parent is missing: {cache_directory.parent}"
        )
    stale_staging = sorted(
        cache_directory.parent.glob(f".{cache_directory.name}.*.staging")
    )
    if stale_staging:
        raise FileExistsError(
            "incomplete validation archive staging requires manual review: "
            f"{stale_staging}"
        )
    inventory = validation_source_inventory(
        shared_data_root,
        expected_train_rows=expected_train_rows,
        expected_val_rows=expected_val_rows,
        expected_unique_images=expected_unique_images,
    )
    # Witness A: the source tree itself, before a tar exists to be tampered with.
    _require_content_identity_matches_approved(
        inventory["file_content_identity"],
        approved_content_sha256,
        witness="validation source",
    )
    with tempfile.TemporaryDirectory(
        prefix=".panderm-validation-archive.",
        dir=runtime_temporary_root,
    ) as temporary:
        temporary_root = Path(temporary)
        temporary_archive = temporary_root / VALIDATION_ARCHIVE_FILENAME
        started = time.perf_counter()
        last_report = started
        total = len(inventory["member_names"])
        source_bytes_completed = 0
        print(
            f"[archive-build] START files_total={total} "
            f"images_total={inventory['unique_images']}",
            flush=True,
        )
        with tarfile.open(temporary_archive, mode="w") as archive:
            for index, member_name in enumerate(
                inventory["member_names"], start=1
            ):
                if index == 1 or index % 250 == 0 or index == total:
                    write_guard(
                        f"archive runtime build progress {index}/{total}"
                    )
                source = inventory["member_sources"][member_name]
                info = tarfile.TarInfo(member_name)
                info.size = source.stat().st_size
                info.mode = 0o644
                info.mtime = 0
                with source.open("rb") as handle:
                    archive.addfile(info, handle)
                source_bytes_completed += info.size
                now = time.perf_counter()
                if index % 250 == 0 or index == total or now - last_report >= 60:
                    elapsed = now - started
                    rate = index / elapsed if elapsed > 0 else float("inf")
                    eta = (total - index) / rate if rate > 0 else float("inf")
                    print(
                        f"[archive-build] files={index}/{total} "
                        f"source_bytes={source_bytes_completed} "
                        f"tar_stream_bytes={archive.fileobj.tell()} "
                        f"elapsed={elapsed:.1f}s rate={rate:.1f}files/s "
                        f"eta={eta:.1f}s current_file={member_name}",
                        flush=True,
                    )
                    last_report = now
        tar_file_content_identity = _file_content_identity_from_tar(
            temporary_archive,
            expected_members=inventory["member_names"],
            phase="archive-build-content",
        )
        if tar_file_content_identity != inventory["file_content_identity"]:
            raise ValueError(
                "validation archive source and reopened tar content mismatch"
            )
        # Witness B: the reopened tar, judged against the approved constant too,
        # not merely against the inventory it was built from.
        _require_content_identity_matches_approved(
            tar_file_content_identity,
            approved_content_sha256,
            witness="reopened validation archive tar",
        )
        extracted = temporary_root / "extracted"
        _safe_extract_validation_archive(
            temporary_archive,
            extracted,
            expected_members=inventory["member_names"],
            phase="archive-build-extract",
        )
        archive_sha256 = sha256_file(temporary_archive)
        identity = {
            "schema_version": VALIDATION_ARCHIVE_SCHEMA_VERSION,
            "cache_format_identity": VALIDATION_ARCHIVE_CACHE_FORMAT,
            "archive_filename": VALIDATION_ARCHIVE_FILENAME,
            "archive_sha256": archive_sha256,
            "byte_size": temporary_archive.stat().st_size,
            "tar_member_count": len(inventory["member_names"]),
            "canonical_sorted_member_list_sha256":
                inventory["canonical_sorted_member_list_sha256"],
            "train_manifest_sha256": inventory["manifest_sha256"]["train"],
            "val_manifest_sha256": inventory["manifest_sha256"]["val"],
            "class_mapping_sha256": inventory["class_mapping_sha256"],
            "train_rows": inventory["train_rows"],
            "val_rows": inventory["val_rows"],
            "unique_images": inventory["unique_images"],
            "exact_relative_file_set_identity":
                inventory["exact_relative_file_set_identity"],
            "file_content_identity_sha256":
                inventory["file_content_identity"]["sha256"],
            "file_content_rows": inventory["file_content_identity"]["rows"],
            "test_manifest_included": False,
            "whole_data_copy_used": False,
            "source_fixed_split_identity": source_fixed_split_identity,
            "created_utc": utc_now(),
        }
        _validate_extracted_validation_data(
            extracted,
            identity,
            approved_file_content_identity_sha256=approved_content_sha256,
        )
        write_guard("archive durable staging directory create")
        staging_directory = cache_directory.parent / (
            f".{cache_directory.name}.{uuid.uuid4().hex}.staging"
        )
        staging_directory.mkdir()
        cached_archive = staging_directory / VALIDATION_ARCHIVE_FILENAME
        _copy_file_with_progress(
            temporary_archive,
            cached_archive,
            phase="archive durable staging",
            write_guard=write_guard,
        )
        if sha256_file(cached_archive) != archive_sha256:
            raise ValueError("published validation archive SHA-256 mismatch")
        identity_path = staging_directory / VALIDATION_ARCHIVE_IDENTITY_FILENAME
        write_json_atomic(identity_path, identity, write_guard=write_guard)
        ready = {
            "schema_version": VALIDATION_ARCHIVE_SCHEMA_VERSION,
            "cache_format_identity": VALIDATION_ARCHIVE_CACHE_FORMAT,
            "archive_filename": VALIDATION_ARCHIVE_FILENAME,
            "archive_sha256": archive_sha256,
            "file_content_identity_sha256":
                identity["file_content_identity_sha256"],
            "archive_identity_sha256": _canonical_mapping_sha256(identity),
        }
        write_json_atomic(
            staging_directory / VALIDATION_ARCHIVE_READY_FILENAME,
            ready,
            write_guard=write_guard,
        )
        validate_validation_archive_cache(
            staging_directory,
            expected_file_content_identity_sha256=approved_content_sha256,
            expected_fixed_split_identity=source_fixed_split_identity,
            expected_manifest_sha256=inventory["manifest_sha256"],
            expected_class_mapping_sha256=inventory["class_mapping_sha256"],
            write_guard=write_guard,
        )
        write_guard("archive cache directory publish")
        os.replace(staging_directory, cache_directory)
    return identity


def validate_validation_archive_cache(
    cache_directory: str | Path,
    *,
    expected_file_content_identity_sha256: str,
    expected_fixed_split_identity: str,
    expected_manifest_sha256: Mapping[str, str],
    expected_class_mapping_sha256: str,
    write_guard: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Fail loud on any incomplete, drifted, or tampered durable cache."""
    approved_content_sha256 = require_approved_content_identity(
        expected_file_content_identity_sha256
    )
    if write_guard is not None:
        write_guard("archive cache validation start")
    cache_directory = Path(cache_directory)
    ready_path = cache_directory / VALIDATION_ARCHIVE_READY_FILENAME
    identity_path = cache_directory / VALIDATION_ARCHIVE_IDENTITY_FILENAME
    archive_path = cache_directory / VALIDATION_ARCHIVE_FILENAME
    for required in (ready_path, identity_path, archive_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"validation archive cache is incomplete; inspect manually: {required}"
            )
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    required_identity = {
        "schema_version", "cache_format_identity", "archive_filename",
        "archive_sha256", "byte_size", "tar_member_count",
        "canonical_sorted_member_list_sha256", "train_manifest_sha256",
        "val_manifest_sha256", "class_mapping_sha256", "train_rows",
        "val_rows", "unique_images", "exact_relative_file_set_identity",
        "file_content_identity_sha256", "file_content_rows",
        "test_manifest_included", "whole_data_copy_used",
        "source_fixed_split_identity", "created_utc",
    }
    if not isinstance(identity, dict) or set(identity) != required_identity:
        raise ValueError("validation archive identity schema mismatch")
    expected_ready = {
        "schema_version": VALIDATION_ARCHIVE_SCHEMA_VERSION,
        "cache_format_identity": VALIDATION_ARCHIVE_CACHE_FORMAT,
        "archive_filename": VALIDATION_ARCHIVE_FILENAME,
        "archive_sha256": identity["archive_sha256"],
        "file_content_identity_sha256":
            identity["file_content_identity_sha256"],
        "archive_identity_sha256": _canonical_mapping_sha256(identity),
    }
    if ready != expected_ready:
        raise ValueError("validation archive READY identity mismatch")
    if (
        identity["schema_version"] != VALIDATION_ARCHIVE_SCHEMA_VERSION
        or identity["cache_format_identity"] != VALIDATION_ARCHIVE_CACHE_FORMAT
        or identity["archive_filename"] != VALIDATION_ARCHIVE_FILENAME
        or identity["test_manifest_included"] is not False
        or identity["whole_data_copy_used"] is not False
    ):
        raise ValueError("validation archive immutable identity mismatch")
    if identity["source_fixed_split_identity"] != expected_fixed_split_identity:
        raise ValueError("validation archive fixed split identity mismatch")
    if set(expected_manifest_sha256) != {"train", "val"}:
        raise ValueError("expected archive manifests must be exactly train/val")
    for split in ("train", "val"):
        if identity[f"{split}_manifest_sha256"] != expected_manifest_sha256[split]:
            raise ValueError(f"validation archive {split} manifest hash mismatch")
    if identity["class_mapping_sha256"] != expected_class_mapping_sha256:
        raise ValueError("validation archive class mapping hash mismatch")
    if (
        _canonical_file_content_rows_sha256(identity["file_content_rows"])
        != identity["file_content_identity_sha256"]
    ):
        raise ValueError("validation archive file content identity mismatch")
    # Persisted per-file rows must re-canonicalize to the *approved* aggregate,
    # not merely to the aggregate stored beside them.
    _require_content_identity_matches_approved(
        {
            "rows": identity["file_content_rows"],
            "sha256": identity["file_content_identity_sha256"],
        },
        approved_content_sha256,
        witness="persisted validation archive identity",
    )
    if archive_path.stat().st_size != identity["byte_size"]:
        raise ValueError("validation archive byte size mismatch")
    if sha256_file(archive_path) != identity["archive_sha256"]:
        raise ValueError("validation archive SHA-256 mismatch")
    members = _validated_tar_members(archive_path)
    member_names = sorted(member.name for member in members)
    if len(member_names) != identity["tar_member_count"]:
        raise ValueError("validation archive tar member count mismatch")
    if (
        _sorted_string_sha256(member_names)
        != identity["canonical_sorted_member_list_sha256"]
    ):
        raise ValueError("validation archive member-list hash mismatch")
    tar_file_content_identity = _file_content_identity_from_tar(
        archive_path,
        expected_members=member_names,
        phase="archive-cache-content",
    )
    if tar_file_content_identity != {
        "rows": identity["file_content_rows"],
        "sha256": identity["file_content_identity_sha256"],
    }:
        raise ValueError("validation archive file content mismatch")
    # Witness B against the independent anchor: a tar, its rows, its aggregate,
    # its archive SHA-256 and READY rewritten consistently still fail here.
    _require_content_identity_matches_approved(
        tar_file_content_identity,
        approved_content_sha256,
        witness="reopened validation archive tar",
    )
    if write_guard is not None:
        write_guard("archive cache validation completion")
    return identity


def reuse_validation_archive_cache(
    cache_directory: str | Path,
    runtime_temporary_root: str | Path,
    local_data_root: str | Path,
    *,
    expected_file_content_identity_sha256: str,
    expected_fixed_split_identity: str,
    expected_manifest_sha256: Mapping[str, str],
    expected_class_mapping_sha256: str,
    write_guard: Callable[[str], Any],
) -> dict[str, Any]:
    """Copy one verified tar to runtime, safely extract, then publish atomically."""
    approved_content_sha256 = require_approved_content_identity(
        expected_file_content_identity_sha256
    )
    if not callable(write_guard):
        raise ValueError("archive reuse requires an active-session write_guard")
    write_guard("archive reuse start")
    runtime_temporary_root = Path(runtime_temporary_root).resolve(strict=True)
    local_data_root = Path(local_data_root)
    if local_data_root.exists():
        raise FileExistsError(f"local validation data already exists: {local_data_root}")
    if local_data_root.parent.resolve(strict=True) != runtime_temporary_root:
        raise ValueError(
            "local validation data must be a direct child of runtime temporary root"
        )
    identity = validate_validation_archive_cache(
        cache_directory,
        expected_file_content_identity_sha256=approved_content_sha256,
        expected_fixed_split_identity=expected_fixed_split_identity,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_class_mapping_sha256=expected_class_mapping_sha256,
        write_guard=write_guard,
    )
    write_guard("archive reuse cache validation complete")
    source_archive = Path(cache_directory) / VALIDATION_ARCHIVE_FILENAME
    with tempfile.TemporaryDirectory(
        prefix=".panderm-validation-reuse.",
        dir=runtime_temporary_root,
    ) as temporary:
        temporary_root = Path(temporary)
        runtime_archive = temporary_root / VALIDATION_ARCHIVE_FILENAME
        _copy_file_with_progress(
            source_archive, runtime_archive, phase="archive-runtime-copy"
        )
        write_guard("archive reuse runtime copy complete")
        if sha256_file(runtime_archive) != identity["archive_sha256"]:
            raise ValueError("runtime validation archive SHA-256 mismatch")
        members = _validated_tar_members(runtime_archive)
        member_names = sorted(member.name for member in members)
        extracted = temporary_root / "extracted"
        _safe_extract_validation_archive(
            runtime_archive,
            extracted,
            expected_members=member_names,
            phase="archive-reuse-extract",
        )
        write_guard("archive reuse extraction complete")
        inventory = _validate_extracted_validation_data(
            extracted,
            identity,
            approved_file_content_identity_sha256=approved_content_sha256,
        )
        write_guard("archive reuse local publish")
        os.replace(extracted, local_data_root)
    return {
        "archive_files_copied": 1,
        "approved_file_content_identity_sha256": approved_content_sha256,
        "images_staged": inventory["unique_images"],
        "manifest_files_staged": 2,
        "class_mapping_files_staged": 1,
        "archive_sha256": identity["archive_sha256"],
        "image_set_exact": True,
        "test_manifest_present": False,
        "whole_data_copy_used": False,
    }


def stage_after_validation_preflights(
    *,
    checkpoint_preflight: Mapping[str, Any],
    gpu_smoke: Mapping[str, Any],
    staging: Callable[[], Any],
) -> Any:
    """Make full staging unreachable until both fast production gates pass."""
    if checkpoint_preflight.get("weights_only_round_trip") is not True:
        raise RuntimeError(
            "validation preflight failed before full staging: "
            "checkpoint serialization round-trip"
        )
    required_smoke = {
        "official_preprocessing",
        "official_checkpoint_loaded",
        "cuda_forward",
        "backward",
        "all_12_blocks_have_gradients",
        "head_has_gradients",
        "fixed_pos_embed_has_no_gradient",
        "optimizer_coverage",
        "optimizer_step",
        "backbone_updated",
        "post_step_checkpoint_round_trip",
        "smoke_model_discarded",
    }
    failed = sorted(key for key in required_smoke if gpu_smoke.get(key) is not True)
    if failed:
        raise RuntimeError(
            f"validation preflight failed before full staging: gpu smoke {failed}"
        )
    return staging()


def stage_validation_data(
    shared_data_root: str | Path, local_data_root: str | Path
) -> dict[str, Any]:
    """Copy only train/val manifests, class mapping, and referenced images."""
    shared_data_root = Path(shared_data_root).resolve(strict=True)
    local_data_root = Path(local_data_root)
    if local_data_root.exists():
        raise FileExistsError(
            f"validation staging destination already exists: {local_data_root}"
        )

    manifests_root = shared_data_root / "manifests"
    manifest_paths = {
        split: manifests_root / f"{split}.csv" for split in ("train", "val")
    }
    class_mapping = manifests_root / "class_to_idx.json"
    for required in (*manifest_paths.values(), class_mapping):
        if not required.is_file():
            raise FileNotFoundError(f"required validation data file missing: {required}")

    staging_started = time.perf_counter()
    print(
        "[Phase 1] START validation staging: "
        f"source={shared_data_root} destination={local_data_root}",
        flush=True,
    )
    relative_paths: list[str] = []
    destination_keys: set[str] = set()
    sources: dict[str, Path] = {}
    for split, manifest_path in manifest_paths.items():
        scan_started = time.perf_counter()
        last_scan_progress = scan_started
        split_rows = 0
        print(
            f"[Phase 1] scan {split}: START manifest={manifest_path}",
            flush=True,
        )
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "image_path" not in reader.fieldnames:
                raise ValueError(f"{manifest_path} is missing image_path")
            for row_number, row in enumerate(reader, start=2):
                split_rows += 1
                relative = _canonical_manifest_image_path(row.get("image_path"))
                destination_key = relative.casefold()
                if destination_key in destination_keys:
                    raise ValueError(
                        f"duplicate validation staging destination at "
                        f"{manifest_path}:{row_number}: {relative}"
                    )
                destination_keys.add(destination_key)
                source = (shared_data_root / Path(*relative.split("/"))).resolve(
                    strict=True
                )
                try:
                    source.relative_to(shared_data_root)
                except ValueError as error:
                    raise ValueError(
                        f"manifest image source escapes shared data root: {relative}"
                    ) from error
                if not source.is_file():
                    raise FileNotFoundError(
                        f"manifest image source is not a file: {source}"
                    )
                relative_paths.append(relative)
                sources[relative] = source
                now = time.perf_counter()
                if split_rows % 1000 == 0 or now - last_scan_progress >= 30:
                    print(
                        f"[Phase 1] scan {split}: rows={split_rows} "
                        f"last={relative} elapsed={now - scan_started:.1f}s",
                        flush=True,
                    )
                    last_scan_progress = now
        print(
            f"[Phase 1] scan {split}: DONE rows={split_rows} "
            f"elapsed={time.perf_counter() - scan_started:.1f}s",
            flush=True,
        )

    local_manifests = local_data_root / "manifests"
    local_manifests.mkdir(parents=True)
    for split, manifest_path in manifest_paths.items():
        shutil.copy2(manifest_path, local_manifests / f"{split}.csv")
    shutil.copy2(class_mapping, local_manifests / class_mapping.name)
    total_images = len(relative_paths)
    copy_started = time.perf_counter()
    last_progress = copy_started
    print(f"[Phase 1] copy images: START total={total_images}", flush=True)
    for index, relative in enumerate(relative_paths, start=1):
        destination = local_data_root / Path(*relative.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(sources[relative], destination)
        except Exception as error:
            raise RuntimeError(
                f"validation image copy failed at {index}/{total_images} "
                f"for relative path {relative!r}: {error}"
            ) from error
        now = time.perf_counter()
        if index % 250 == 0 or index == total_images or now - last_progress >= 30:
            elapsed = now - copy_started
            rate = index / elapsed if elapsed > 0 else float("inf")
            eta = (total_images - index) / rate if rate > 0 else float("inf")
            print(
                f"[Phase 1] copy images: {index}/{total_images} "
                f"last={relative} elapsed={elapsed:.1f}s "
                f"rate={rate:.1f} images/s eta={eta:.1f}s",
                flush=True,
            )
            last_progress = now
    copy_elapsed = time.perf_counter() - copy_started
    average_rate = (
        total_images / copy_elapsed if copy_elapsed > 0 else float("inf")
    )
    print(
        f"[Phase 1] copy images: DONE total={total_images} "
        f"elapsed={copy_elapsed:.1f}s avg={average_rate:.1f} images/s",
        flush=True,
    )

    allowed_manifests = {"train.csv", "val.csv", "class_to_idx.json"}
    copied_manifests = {
        path.name for path in local_manifests.iterdir() if path.is_file()
    }
    if copied_manifests != allowed_manifests:
        raise ValueError(
            f"validation staging manifest allowlist mismatch: {copied_manifests}"
        )
    if (local_manifests / "test.csv").exists():
        raise ValueError("validation staging unexpectedly contains test.csv")
    verify_started = time.perf_counter()
    print(
        f"[Phase 1] verify images: START expected={total_images}",
        flush=True,
    )
    copied_images: set[str] = set()
    last_verify_progress = verify_started
    for path in local_data_root.rglob("*"):
        if not path.is_file() or local_manifests in path.parents:
            continue
        copied_images.add(path.relative_to(local_data_root).as_posix())
        now = time.perf_counter()
        copied_count = len(copied_images)
        if (
            copied_count % 1000 == 0
            or copied_count == total_images
            or now - last_verify_progress >= 30
        ):
            print(
                f"[Phase 1] verify images: {copied_count}/{total_images} "
                f"last={path.relative_to(local_data_root).as_posix()} "
                f"elapsed={now - verify_started:.1f}s",
                flush=True,
            )
            last_verify_progress = now
    expected_images = set(relative_paths)
    if copied_images != expected_images:
        raise ValueError(
            "validation staging image set mismatch: "
            f"missing={sorted(expected_images - copied_images)[:10]} "
            f"extra={sorted(copied_images - expected_images)[:10]}"
        )
    print(
        f"[Phase 1] verify images: DONE actual={len(copied_images)} "
        f"elapsed={time.perf_counter() - verify_started:.1f}s",
        flush=True,
    )
    print(
        f"[Phase 1] DONE validation staging: copied={len(copied_images)} "
        f"elapsed={time.perf_counter() - staging_started:.1f}s",
        flush=True,
    )
    return {
        "manifest_files_copied": 2,
        "class_mapping_files_copied": 1,
        "images_copied": len(copied_images),
        "image_relative_paths": sorted(copied_images),
        "test_manifest_present": False,
        "image_set_exact": True,
    }


def validate_validation_notebook_source(source: str) -> None:
    """Reject executable copy logic; staging must stay in the tested helper."""
    tree = ast.parse(source)
    shutil_aliases = {"shutil"}
    imported_copy_names: set[str] = set()
    copy_functions = {"copy", "copy2", "copyfile", "copytree"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "shutil":
                    shutil_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "shutil":
            for alias in node.names:
                if alias.name in copy_functions:
                    imported_copy_names.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called_copy = None
        if isinstance(node.func, ast.Name) and node.func.id in imported_copy_names:
            called_copy = node.func.id
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in shutil_aliases
            and node.func.attr in copy_functions
        ):
            called_copy = node.func.attr
        if called_copy is not None:
            raise ValueError(
                f"validation notebook contains executable {called_copy} copy; "
                "use stage_validation_data instead"
            )


def require_checkpoint_sha256(
    path: str | Path, expected: str = EXPECTED_CHECKPOINT_SHA256
) -> str:
    """Hash the downloaded checkpoint and refuse anything but the pinned digest.

    While ``expected`` is still the placeholder this always raises, printing the
    observed digest so a human can review and pin it. A download is never
    accepted on trust after the first review.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PanDerm checkpoint not found: {path}")
    observed = sha256_file(path)
    size = path.stat().st_size
    if expected == CHECKPOINT_SHA256_PLACEHOLDER:
        raise ValueError(
            "PanDerm checkpoint SHA-256 is not pinned yet. Upstream publishes no "
            "digest, so the first download is trust-on-first-use and must be "
            "reviewed by a human before any training.\n"
            f"  file:     {path}\n"
            f"  bytes:    {size}\n"
            f"  observed: {observed}\n"
            "Review this digest, set panderm_run.EXPECTED_CHECKPOINT_SHA256 to it, "
            "push, and re-pin the notebook commit."
        )
    if not is_pinned_sha256(expected):
        raise ValueError(
            f"expected PanDerm checkpoint SHA-256 must be 64 lowercase hex chars, "
            f"got {expected!r}"
        )
    if observed != expected:
        raise ValueError(
            "PanDerm checkpoint SHA-256 mismatch; refusing unverified weights: "
            f"observed={observed} expected={expected} file={path}"
        )
    return observed


def require_provenance_clearance(
    *,
    upstream_commit: str,
    checkpoint_sha256: str | None,
    expected_checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256,
    license_review: Mapping[str, Any] | None = None,
    contamination_review: Mapping[str, Any] | None = None,
    purpose: str = VALIDATION_ONLY,
) -> dict[str, Any]:
    """Permit only exploratory validation after provenance checks.

    Formal training and test access are policy-prohibited, independent of an
    author assertion, pinned commit, or pinned checkpoint hash.
    """
    if purpose in {FORMAL_TRAINING, TEST_ACCESS}:
        raise ValueError(PROHIBITED_FORMAL_TEST_REASON)
    if purpose != VALIDATION_ONLY:
        raise ValueError(f"unsupported PanDerm provenance purpose: {purpose!r}")

    license_review = dict(license_review or LICENSE_REVIEW)
    contamination_review = dict(contamination_review or CONTAMINATION_REVIEW)
    failures: list[str] = []

    if license_review.get("status") != "reviewed_permits_noncommercial_finetuning":
        failures.append(f"license status not reviewed: {license_review.get('status')!r}")
    if license_review.get("finetuning_allowed") is not True:
        failures.append("license review does not permit fine-tuning")
    if license_review.get("deployment_allowed") is not False:
        failures.append("license review must forbid deployment of adapted weights")
    if contamination_review.get("independent_audit_possible") is not False:
        failures.append(
            "validation records must acknowledge that independent overlap audit "
            "is not possible"
        )
    if contamination_review.get("patient_level_overlap") != "not_excludable":
        failures.append(
            "validation records must acknowledge patient overlap is not excludable"
        )
    if contamination_review.get("exact_fixed_validation_test_overlap") != "unproven":
        failures.append(
            "validation records must acknowledge exact fixed validation/test "
            "overlap is unproven"
        )
    if contamination_review.get("claim_boundary") != "suggestive_exploratory_only":
        failures.append(
            "claim boundary must stay suggestive/exploratory: "
            f"{contamination_review.get('claim_boundary')!r}"
        )
    if upstream_commit != UPSTREAM_COMMIT:
        failures.append(
            f"upstream commit mismatch: {upstream_commit!r} != {UPSTREAM_COMMIT!r}"
        )
    if not is_pinned_sha256(expected_checkpoint_sha256):
        failures.append(
            "PanDerm checkpoint SHA-256 is not pinned "
            f"({expected_checkpoint_sha256!r}); review the first download and pin it"
        )
    elif checkpoint_sha256 != expected_checkpoint_sha256:
        failures.append(
            f"checkpoint SHA-256 mismatch: {checkpoint_sha256!r} != "
            f"{expected_checkpoint_sha256!r}"
        )

    if failures:
        raise ValueError(
            "PanDerm provenance/license/contamination gate failed before "
            "exploratory validation: "
            + "; ".join(failures)
        )
    return {
        "cleared_for": VALIDATION_ONLY,
        "upstream_commit": upstream_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_sha256_provenance": CHECKPOINT_SHA256_PROVENANCE,
        "license_review": license_review,
        "contamination_review": contamination_review,
        "claim_boundary": contamination_review["claim_boundary"],
        "deployment_allowed": False,
        "formal_training_allowed": False,
        "test_access_allowed": False,
        "attribution": ATTRIBUTION,
        "reviewed_utc": utc_now(),
    }


def require_primitive_identity(value: Any, path: str = "run_identity") -> None:
    """Require exact JSON primitives so weights-only checkpoint loads stay safe."""
    value_type = type(value)
    if value is None or value_type in {str, int, bool}:
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if value_type is list:
        for index, item in enumerate(value):
            require_primitive_identity(item, f"{path}[{index}]")
        return
    if value_type is tuple:
        for index, item in enumerate(value):
            require_primitive_identity(item, f"{path}[{index}]")
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{path} contains a non-string mapping key")
            require_primitive_identity(item, f"{path}.{key}")
        return
    raise ValueError(
        f"{path} contains non-primitive {value_type.__module__}."
        f"{value_type.__qualname__}"
    )


def build_run_identity(
    *,
    git_commit: str | None,
    seed: int,
    epochs: int,
    evaluation_scope: str,
    checkpoint_sha256: str | None,
    model_identity: Mapping[str, Any],
    manifest_sha256: Mapping[str, str],
    fixed_split_identity: str,
    shared_root_uuid: str | None,
    formal_output_identity: str | None,
    dependency_versions: Mapping[str, Any],
    variant: str = VARIANT,
    df_target_count: int = DF_TARGET_COUNT,
    batch_size: int = BATCH_SIZE,
    accumulation_steps: int = ACCUMULATION_STEPS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    warmup_epochs: int = VALIDATION_WARMUP_EPOCHS,
    layer_decay: float = LAYER_DECAY,
    drop_path: float = DROP_PATH,
    amp_requested: bool = True,
    amp_effective: bool = False,
    device_type: str = "cpu",
    run_version: str = RUN_VERSION,
) -> dict[str, Any]:
    """Assemble the full immutable identity carried by every artifact."""
    if variant != VARIANT:
        raise ValueError(
            f"PanDerm run is C1 real-data only, got variant {variant!r}"
        )
    if evaluation_scope != VALIDATION_ONLY:
        raise ValueError(PROHIBITED_FORMAL_TEST_REASON)
    if int(seed) != 0 or int(epochs) != VALIDATION_EPOCHS:
        raise ValueError(PROHIBITED_FORMAL_TEST_REASON)
    if int(warmup_epochs) != VALIDATION_WARMUP_EPOCHS:
        raise ValueError(
            f"PanDerm v1 validation warmup must be "
            f"{VALIDATION_WARMUP_EPOCHS} epochs"
        )
    missing_manifests = [
        split for split in ("train", "val") if split not in manifest_sha256
    ]
    if missing_manifests:
        raise ValueError(f"manifest identity is missing splits: {missing_manifests}")
    if set(manifest_sha256) != {"train", "val"}:
        raise ValueError(
            "PanDerm v1 identity may contain only train/val manifest hashes; "
            "test manifest access is prohibited"
        )
    if float(drop_path) != DROP_PATH:
        raise ValueError(f"PanDerm drop_path must be exactly {DROP_PATH}")
    if amp_requested is not True:
        raise ValueError("PanDerm validation requires AMP to be requested")
    if device_type == "cuda" and amp_effective is not True:
        raise ValueError("PanDerm CUDA validation requires effective AMP")
    if device_type != "cuda" and amp_effective is not False:
        raise ValueError("PanDerm non-CUDA validation cannot claim effective AMP")
    identity = {
        "schema_version": 1,
        "git_commit": git_commit,
        "run_version": run_version,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint_filename": CHECKPOINT_FILENAME,
        "checkpoint_source_url": CHECKPOINT_SOURCE_URL,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_sha256_provenance": CHECKPOINT_SHA256_PROVENANCE,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "model_identity": dict(model_identity),
        "variant": variant,
        "seed": int(seed),
        "fixed_split_identity": fixed_split_identity,
        "manifest_sha256": dict(manifest_sha256),
        "c1_construction": {
            "strategy": "duplicate_real_train_df",
            "df_target_count": int(df_target_count),
            "synthetic_images_used": False,
            "expected_train_rows": EXPECTED_C1_TRAIN_ROWS,
            "expected_class_counts": dict(EXPECTED_C1_CLASS_COUNTS),
        },
        "objective": {
            "loss_name": "cross_entropy",
            "class_weighting": "none",
            "label_smoothing": 0.0,
            "sampler": "none",
            "mixup": False,
            "cutmix": False,
            "tta": False,
            "primary_metric": "df_f1",
            "model_selection_metric": "validation_df_f1_strict_improvement",
        },
        "optimization": {
            "optimizer": "adamw",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "scheduler": "warmup_cosine",
            "warmup_epochs": int(warmup_epochs),
            "min_lr": MIN_LR,
            "layer_decay": float(layer_decay),
            "drop_path": float(drop_path),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "accumulation_steps": int(accumulation_steps),
            "effective_batch_size": int(batch_size) * int(accumulation_steps),
            "amp_requested": bool(amp_requested),
            "amp_effective": bool(amp_effective),
            "device_type": str(device_type),
            "scheduler_step_unit": "optimizer_step",
        },
        "dependency_versions": dict(dependency_versions),
        "shared_root_uuid": shared_root_uuid,
        "formal_output_identity": formal_output_identity,
        "evaluation_scope": evaluation_scope,
        "license_review": dict(LICENSE_REVIEW),
        "contamination_review": dict(CONTAMINATION_REVIEW),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    require_primitive_identity(identity)
    return identity


def require_matching_identity(
    saved: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    """Reject a resume/verification before any state is loaded or mutated."""
    require_expected_identity_complete(current)
    missing = [key for key in IMMUTABLE_IDENTITY_KEYS if key not in saved]
    if missing:
        raise ValueError(
            f"artifact lacks immutable PanDerm identity fields: {missing}; "
            "refusing an unverifiable resume"
        )
    mismatches = [
        f"{key}: saved={saved[key]!r} current={current.get(key)!r}"
        for key in IMMUTABLE_IDENTITY_KEYS
        if saved[key] != current.get(key)
    ]
    if mismatches:
        raise ValueError("PanDerm identity mismatch: " + "; ".join(mismatches))


def require_expected_identity_complete(expected: Mapping[str, Any]) -> None:
    """Refuse an expected-identity block that has had fields deleted."""
    missing = [key for key in IMMUTABLE_IDENTITY_KEYS if key not in expected]
    if missing:
        raise ValueError(
            f"expected PanDerm identity is incomplete: {missing}; a truncated "
            "expectation would silently accept a drifted artifact"
        )


def require_identity_duplicates(
    *,
    expected: Mapping[str, Any],
    record: Mapping[str, Any],
    nested: Mapping[str, Any],
    top_level: Mapping[str, Any],
) -> None:
    """Require the record, its nested identity and top-level duplicates to agree.

    Catches all four failure modes: an incomplete expectation, a missing
    duplicate, duplicate drift, and duplicates that agree with each other but
    disagree with the reviewed expectation.
    """
    require_expected_identity_complete(expected)
    require_expected_identity_complete(nested)
    missing = [key for key in IMMUTABLE_IDENTITY_KEYS if key not in top_level]
    if missing:
        raise ValueError(f"record is missing duplicated identity fields: {missing}")
    drift = [
        f"{key}: nested={nested.get(key)!r} top_level={top_level.get(key)!r}"
        for key in IMMUTABLE_IDENTITY_KEYS
        if nested.get(key) != top_level.get(key)
    ]
    if drift:
        raise ValueError("duplicated identity drift: " + "; ".join(drift))
    require_matching_identity(nested, expected)
    if record.get("run_version") != expected.get("run_version"):
        raise ValueError(
            f"record run_version {record.get('run_version')!r} does not match "
            f"expected {expected.get('run_version')!r}"
        )


def require_completed_artifact_identities(
    *,
    expected: Mapping[str, Any],
    result: Mapping[str, Any],
    best_checkpoint: Mapping[str, Any],
    last_checkpoint: Mapping[str, Any],
) -> None:
    """Validate result/best/last against the current complete identity."""
    require_expected_identity_complete(expected)
    nested = result.get("run_identity")
    if not isinstance(nested, Mapping):
        raise ValueError("completed result is missing run_identity")
    top_level = {
        key: result[key]
        for key in IMMUTABLE_IDENTITY_KEYS
        if key in result
    }
    require_identity_duplicates(
        expected=expected,
        record=result,
        nested=nested,
        top_level=top_level,
    )
    for label, checkpoint in (
        ("best.pt", best_checkpoint),
        ("last.pt", last_checkpoint),
    ):
        identity = checkpoint.get("run_identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"{label} is missing run_identity")
        require_matching_identity(identity, expected)
        if dict(identity) != dict(nested):
            raise ValueError(f"{label} identity does not equal result identity")


def require_shared_root_sentinel_identity(
    sentinel: Mapping[str, Any],
    *,
    shortcut_alias: str,
    resolved_root: str | Path,
    run_version: str = RUN_VERSION,
) -> str:
    """Require one shared root even when Drive shortcut strings differ."""
    if not isinstance(sentinel, Mapping):
        raise ValueError("shared-root sentinel must be a mapping")
    required = ("shared_root_uuid", "shortcut_alias", "resolved_path", "run_version")
    missing = [key for key in required if key not in sentinel]
    if missing:
        raise ValueError(f"shared-root sentinel is missing fields: {missing}")
    if sentinel["shortcut_alias"] != shortcut_alias:
        raise ValueError(
            "shared-root sentinel shortcut alias mismatch: "
            f"saved={sentinel['shortcut_alias']!r} expected={shortcut_alias!r}"
        )
    if sentinel["run_version"] != run_version:
        raise ValueError(
            "shared-root sentinel run version mismatch: "
            f"saved={sentinel['run_version']!r} expected={run_version!r}"
        )
    shared_root_uuid = sentinel["shared_root_uuid"]
    if not isinstance(shared_root_uuid, str):
        raise ValueError("shared-root sentinel UUID must be a string")
    try:
        parsed_uuid = uuid.UUID(shared_root_uuid)
    except ValueError as exc:
        raise ValueError("shared-root sentinel UUID is invalid") from exc
    if str(parsed_uuid) != shared_root_uuid:
        raise ValueError("shared-root sentinel UUID is not canonical")
    stored_path = sentinel["resolved_path"]
    if not isinstance(stored_path, str) or not stored_path:
        raise ValueError("shared-root sentinel resolved path must be a string")
    current_root = require_existing_shared_root(resolved_root)
    try:
        same_root = Path(stored_path).samefile(current_root)
    except OSError as exc:
        raise ValueError(
            "shared-root sentinel physical identity could not be verified: "
            f"saved={stored_path!r} current={str(current_root)!r}"
        ) from exc
    if not same_root:
        raise ValueError(
            "shared-root sentinel points to a different physical directory: "
            f"saved={stored_path!r} current={str(current_root)!r}"
        )
    return shared_root_uuid


ACTIVE_SESSION_FILENAME = "active_session.json"
SESSION_HISTORY_DIRECTORY = "session_history"
TAKEOVER_AUDIT_SUFFIX = ".takeover.json"
MANUAL_TAKEOVER_CONFIRMATION = "I CONFIRM THE PREVIOUS RUNTIME IS STOPPED"
DRIVE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
DRIVE_JSON_MIME_TYPE = "application/json"
DRIVE_SHORTCUT_MIME_TYPE = "application/vnd.google-apps.shortcut"
VALIDATION_LOCK_TOPOLOGY_SHARED_DRIVE = "shared_drive"
VALIDATION_LOCK_TOPOLOGY_SHARED_MY_DRIVE = "shared_my_drive"
# Compatibility alias for older local callers; the value is deliberately no
# longer root-owner-only.
VALIDATION_LOCK_TOPOLOGY_OWNED_MY_DRIVE = VALIDATION_LOCK_TOPOLOGY_SHARED_MY_DRIVE
ACTIVE_SESSION_FIELDS = (
    "schema_version",
    "session_id",
    "run_version",
    "git_commit",
    "shared_root_uuid",
    "evaluation_scope",
    "account_label",
    "hostname",
    "started_utc",
    "run_identity_sha256",
    "checkpoint_cadence",
    "maximum_quota_loss",
)
GRACEFUL_HANDOFF_EVENT = "graceful_handoff_complete"
MANUAL_TAKEOVER_EVENT = "manual_takeover"
GRACEFUL_HANDOFF_STABLE_FIELDS = (
    "schema_version",
    "event",
    "event_id",
    "completed_utc",
    "active_session",
    "checkpoint_integrity",
    "result_identity",
)
MANUAL_TAKEOVER_STABLE_FIELDS = (
    "schema_version",
    "event",
    "event_id",
    "confirmation",
    "confirmed_utc",
    "previous_active_session",
    "replacement_session_id",
)


def require_drive_shortcut_target(
    records: Sequence[Mapping[str, Any]], *, expected_alias: str
) -> dict[str, str]:
    """Resolve one My Drive shortcut using provider metadata, never its hidden path."""
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("shared run shortcut provider records must be a sequence")
    normalized = [
        _require_drive_provider_record(
            record,
            label="shared run shortcut",
            expected_name=expected_alias,
            expected_mime_type=DRIVE_SHORTCUT_MIME_TYPE,
        )
        for record in records
    ]
    if len(normalized) != 1:
        raise ValueError(
            "exactly one provider-visible shared run shortcut is required in "
            f"My Drive: alias={expected_alias!r} count={len(normalized)}"
        )
    details = normalized[0].get("shortcutDetails")
    if not isinstance(details, Mapping):
        raise ValueError("shared run shortcut target metadata is missing")
    target_id = details.get("targetId")
    if type(target_id) is not str or not target_id:
        raise ValueError("shared run shortcut target id is invalid")
    if details.get("targetMimeType") != DRIVE_FOLDER_MIME_TYPE:
        raise ValueError("shared run shortcut target must be a Drive folder")
    target_resource_key = details.get("targetResourceKey")
    if type(target_resource_key) is not str or not target_resource_key:
        raise ValueError("shared run shortcut target resource key is missing or invalid")
    return {
        "target_id": target_id,
        "target_resource_key": target_resource_key,
    }


def require_drive_shortcut_target_id(
    records: Sequence[Mapping[str, Any]], *, expected_alias: str
) -> str:
    """Compatibility wrapper that still validates the target resource key."""
    return require_drive_shortcut_target(
        records, expected_alias=expected_alias
    )["target_id"]


def drive_resource_key_header(
    resource_keys: Sequence[tuple[str, str]],
) -> str:
    """Build Google's X-Goog-Drive-Resource-Keys header without ambiguity."""
    if not isinstance(resource_keys, Sequence) or isinstance(
        resource_keys, (str, bytes)
    ):
        raise ValueError("Drive resource key pairs must be a sequence")
    observed: dict[str, str] = {}
    for pair in resource_keys:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or type(pair[0]) is not str
            or not pair[0]
            or type(pair[1]) is not str
            or not pair[1]
            or any(character in pair[0] + pair[1] for character in "/,\r\n")
        ):
            raise ValueError("Drive resource key pair is invalid")
        file_id, resource_key = pair
        previous = observed.get(file_id)
        if previous is not None and previous != resource_key:
            raise ValueError(
                f"Drive resource key drift for file id {file_id!r}"
            )
        observed[file_id] = resource_key
    return ",".join(
        f"{file_id}/{observed[file_id]}" for file_id in sorted(observed)
    )


def drive_provider_list_request_kwargs(
    *,
    query: str,
    fields: str,
    drive_id: str | None,
    page_token: str | None,
    page_size: int = 100,
) -> dict[str, Any]:
    """Return distinct My Drive and true Shared Drive request shapes."""
    if type(query) is not str or not query:
        raise ValueError("Drive provider query must be non-empty")
    if type(fields) is not str or not fields:
        raise ValueError("Drive provider fields must be non-empty")
    if type(page_size) is not int or not 1 <= page_size <= 1000:
        raise ValueError("Drive provider page size is invalid")
    if page_token is not None and (type(page_token) is not str or not page_token):
        raise ValueError("Drive provider page token is invalid")
    kwargs: dict[str, Any] = {
        "q": query,
        "spaces": "drive",
        "pageSize": page_size,
        "fields": fields,
        "includeItemsFromAllDrives": True,
        "supportsAllDrives": True,
        "corpora": "user",
    }
    if page_token is not None:
        kwargs["pageToken"] = page_token
    if drive_id is not None:
        if type(drive_id) is not str or not drive_id:
            raise ValueError("Shared Drive drive_id is invalid")
        kwargs["corpora"] = "drive"
        kwargs["driveId"] = drive_id
    return kwargs


def collect_drive_provider_pages(
    fetch_page: Callable[[str | None], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collect a complete provider namespace or fail loud."""
    if not callable(fetch_page):
        raise ValueError("Drive provider page fetcher must be callable")
    records: list[dict[str, Any]] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()
    while True:
        response = fetch_page(page_token)
        if not isinstance(response, Mapping):
            raise ValueError("Drive provider list response must be a mapping")
        if response.get("incompleteSearch") is not False:
            raise ValueError(
                "Drive provider namespace is incomplete; never infer unlocked"
            )
        files = response.get("files")
        if not isinstance(files, list) or any(
            not isinstance(record, Mapping) for record in files
        ):
            raise ValueError("Drive provider list files are invalid")
        records.extend(dict(record) for record in files)
        next_token = response.get("nextPageToken")
        if next_token in (None, ""):
            return records
        if type(next_token) is not str or next_token in seen_tokens:
            raise ValueError("Drive provider pagination token is invalid or repeated")
        seen_tokens.add(next_token)
        page_token = next_token


def _require_drive_provider_record(
    record: Mapping[str, Any],
    *,
    label: str,
    expected_name: str | None = None,
    expected_mime_type: str | None = None,
    expected_parent_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} provider metadata must be a mapping")
    normalized = dict(record)
    for field in ("id", "name", "mimeType"):
        if type(normalized.get(field)) is not str or not normalized[field]:
            raise ValueError(f"{label} provider metadata {field} must be non-empty")
    if normalized.get("trashed") is not False:
        raise ValueError(f"{label} provider metadata must prove trashed=false")
    if expected_name is not None and normalized["name"] != expected_name:
        raise ValueError(
            f"{label} provider name drift: saved={normalized['name']!r} "
            f"expected={expected_name!r}"
        )
    if (
        expected_mime_type is not None
        and normalized["mimeType"] != expected_mime_type
    ):
        raise ValueError(
            f"{label} provider MIME drift: saved={normalized['mimeType']!r} "
            f"expected={expected_mime_type!r}"
        )
    if expected_parent_id is not None:
        parents = normalized.get("parents")
        if (
            not isinstance(parents, list)
            or len(parents) != 1
            or parents[0] != expected_parent_id
        ):
            raise ValueError(
                f"{label} provider parent identity drift: "
                f"saved={parents!r} expected={[expected_parent_id]!r}"
            )
    return normalized


def require_validation_lock_storage_topology(
    root_metadata: Mapping[str, Any],
    probe_metadata: Mapping[str, Any],
    *,
    expected_root_id: str,
    expected_probe_name: str,
) -> dict[str, Any]:
    """Require owner-stable lock descendants before any expensive phase.

    A true Shared Drive has one provider-owned namespace. A shared My Drive
    folder has per-creator descendants, so the API account only needs to own
    the FUSE-created probe (proving API/mount account alignment). Cross-account
    descendant visibility must be established by the sequential handoff's
    shared-root sentinel and reopen checks; root ownership is account-neutral.
    """
    root = _require_drive_provider_record(
        root_metadata,
        label="shared run root",
        expected_mime_type=DRIVE_FOLDER_MIME_TYPE,
    )
    if root["id"] != expected_root_id:
        raise ValueError(
            "shared run root provider id drift: "
            f"saved={root['id']!r} expected={expected_root_id!r}"
        )
    probe = _require_drive_provider_record(
        probe_metadata,
        label="lock topology probe",
        expected_name=expected_probe_name,
        expected_parent_id=expected_root_id,
    )
    root_drive_id = root.get("driveId")
    probe_drive_id = probe.get("driveId")
    if type(root_drive_id) is str and root_drive_id:
        if probe_drive_id != root_drive_id:
            raise ValueError(
                "lock topology probe escaped the root Shared Drive: "
                f"root_drive_id={root_drive_id!r} "
                f"probe_drive_id={probe_drive_id!r}"
            )
        mode = VALIDATION_LOCK_TOPOLOGY_SHARED_DRIVE
    else:
        if probe_drive_id not in (None, ""):
            raise ValueError(
                "shared My Drive root and lock topology probe disagree on driveId"
            )
        if probe.get("ownedByMe") is not True:
            raise ValueError(
                "Drive API/FUSE account mismatch: the authenticated API account "
                "must own the uniquely named child created through the public "
                "MyDrive alias"
            )
        mode = VALIDATION_LOCK_TOPOLOGY_SHARED_MY_DRIVE
    return {
        "status": "passed",
        "mode": mode,
        "folder_id": expected_root_id,
        "drive_id": root_drive_id,
    }


def require_drive_api_fuse_account_alignment(
    probe_metadata: Mapping[str, Any], *, expected_probe_name: str
) -> dict[str, str]:
    """Prove the Drive API account owns a FUSE-created private My Drive probe."""
    probe = _require_drive_provider_record(
        probe_metadata,
        label="Drive API/FUSE account probe",
        expected_name=expected_probe_name,
        expected_parent_id="root",
    )
    if probe.get("ownedByMe") is not True:
        raise ValueError(
            "Drive API/FUSE account mismatch: the API account does not own "
            "the private My Drive probe created through FUSE"
        )
    if probe.get("driveId") not in (None, ""):
        raise ValueError(
            "Drive API/FUSE account probe must be in private My Drive, not a "
            "Shared Drive"
        )
    return {"file_id": probe["id"], "status": "passed"}


def build_durable_root_provider_identity(
    root_metadata: Mapping[str, Any],
    *,
    expected_root_id: str,
    shortcut_target_resource_key: str,
    shared_root_uuid: str,
    topology: Mapping[str, Any],
) -> dict[str, str]:
    """Bind the qualified artifact root to provider and shortcut identity."""
    root = _require_drive_provider_record(
        root_metadata,
        label="shared run root",
        expected_mime_type=DRIVE_FOLDER_MIME_TYPE,
    )
    normalized_topology = _require_provider_topology_mapping(topology)
    if root["id"] != expected_root_id:
        raise ValueError("shared run root provider id drift")
    root_resource_key = root.get("resourceKey")
    if type(root_resource_key) is not str or not root_resource_key:
        raise ValueError("shared run root provider resource key is missing")
    if (
        type(shortcut_target_resource_key) is not str
        or not shortcut_target_resource_key
        or root_resource_key != shortcut_target_resource_key
    ):
        raise ValueError(
            "shared run shortcut target and provider resource key drift"
        )
    try:
        parsed_uuid = uuid.UUID(shared_root_uuid)
    except (TypeError, ValueError) as error:
        raise ValueError("shared_root_uuid is invalid") from error
    if str(parsed_uuid) != shared_root_uuid or parsed_uuid.version != 4:
        raise ValueError("shared_root_uuid must be a canonical UUIDv4")
    drive_id = normalized_topology.get("drive_id") or ""
    provider_projection = {
        "drive_id": drive_id,
        "mime_type": root["mimeType"],
        "name": root["name"],
        "parents": root.get("parents"),
        "root_file_id": root["id"],
        "root_resource_key": root_resource_key,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            provider_projection,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "drive_id": drive_id,
        "provider_fingerprint": fingerprint,
        "root_file_id": root["id"],
        "root_resource_key": root_resource_key,
        "run_version": RUN_VERSION,
        "shared_root_uuid": shared_root_uuid,
        "topology_mode": normalized_topology["mode"],
    }


def _require_provider_topology_mapping(
    topology: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(topology, Mapping):
        raise ValueError("validation lock storage topology must be a mapping")
    normalized = dict(topology)
    if normalized.get("status") != "passed":
        raise ValueError("validation lock storage topology did not pass")
    mode = normalized.get("mode")
    if mode not in {
        VALIDATION_LOCK_TOPOLOGY_SHARED_DRIVE,
        VALIDATION_LOCK_TOPOLOGY_SHARED_MY_DRIVE,
    }:
        raise ValueError("validation lock storage topology mode is invalid")
    if type(normalized.get("folder_id")) is not str or not normalized["folder_id"]:
        raise ValueError("validation lock storage topology folder_id is invalid")
    if mode == VALIDATION_LOCK_TOPOLOGY_SHARED_DRIVE:
        if type(normalized.get("drive_id")) is not str or not normalized["drive_id"]:
            raise ValueError("Shared Drive topology must carry a drive_id")
    elif normalized.get("drive_id") not in (None, ""):
        raise ValueError("shared My Drive topology must not carry a drive_id")
    return normalized


def _require_provider_child_topology(
    record: Mapping[str, Any],
    *,
    label: str,
    expected_name: str,
    expected_mime_type: str,
    expected_parent_id: str,
    topology: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_topology = _require_provider_topology_mapping(topology)
    child = _require_drive_provider_record(
        record,
        label=label,
        expected_name=expected_name,
        expected_mime_type=expected_mime_type,
        expected_parent_id=expected_parent_id,
    )
    if normalized_topology["mode"] == VALIDATION_LOCK_TOPOLOGY_SHARED_DRIVE:
        if child.get("driveId") != normalized_topology["drive_id"]:
            raise ValueError(f"{label} provider Shared Drive identity drift")
    else:
        if child.get("driveId") not in (None, ""):
            raise ValueError(f"{label} unexpectedly belongs to a Shared Drive")
    return child


def require_validation_version_provider_state(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_parent_id: str,
    run_version: str,
    topology: Mapping[str, Any],
    local_version_exists: bool,
) -> dict[str, Any] | None:
    """Reconcile the FUSE version directory with the provider namespace."""
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("validation version provider records must be a sequence")
    normalized = [
        _require_provider_child_topology(
            record,
            label="validation version directory",
            expected_name=run_version,
            expected_mime_type=DRIVE_FOLDER_MIME_TYPE,
            expected_parent_id=expected_parent_id,
            topology=topology,
        )
        for record in records
    ]
    if len(normalized) > 1:
        raise ValueError(
            "ambiguous duplicate validation version directories exist in the "
            "provider namespace; inspect manually"
        )
    if local_version_exists:
        if not normalized:
            raise FileNotFoundError(
                "validation version directory is visible through FUSE but absent "
                "from the Drive provider namespace"
            )
        return normalized[0]
    if normalized:
        raise ValueError(
            "validation version directory exists in the Drive provider namespace "
            "but is invisible through FUSE; refusing to create a duplicate"
        )
    return None


def require_active_session_provider_state(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_parent_id: str,
    topology: Mapping[str, Any],
    expected_present: bool,
    expected_file_id: str | None = None,
) -> dict[str, Any] | None:
    """Require one provider-visible active marker or prove that none exists."""
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("active session provider records must be a sequence")
    normalized = [
        _require_provider_child_topology(
            record,
            label="active session marker",
            expected_name=ACTIVE_SESSION_FILENAME,
            expected_mime_type=DRIVE_JSON_MIME_TYPE,
            expected_parent_id=expected_parent_id,
            topology=topology,
        )
        for record in records
    ]
    if len(normalized) > 1:
        raise ValueError(
            "ambiguous duplicate active session markers exist in the provider "
            "namespace; no staging, attempt, runner, or automatic cleanup is allowed"
        )
    if expected_present:
        if not normalized:
            raise FileNotFoundError(
                "active session marker is absent from the Drive provider namespace"
            )
        observed = normalized[0]
        if expected_file_id is not None and observed["id"] != expected_file_id:
            raise ValueError(
                "active session provider file identity drift: "
                f"saved={observed['id']!r} expected={expected_file_id!r}"
            )
        return observed
    if normalized:
        raise FileExistsError(
            "the Drive provider namespace already contains an active session "
            "marker, even if the FUSE alias cannot see it; do not create a "
            "replacement without explicit manual takeover"
        )
    return None


def active_session_path(
    shared_run_root: str | Path, *, run_version: str = RUN_VERSION
) -> Path:
    """Return the fixed operational marker for one sequential run version."""
    return Path(shared_run_root) / run_version / ACTIVE_SESSION_FILENAME


def _validate_active_session_marker(
    marker: Mapping[str, Any], *, marker_path: str | Path
) -> dict[str, Any]:
    if not isinstance(marker, dict) or set(marker) != set(ACTIVE_SESSION_FIELDS):
        raise ValueError(
            f"active session schema mismatch at {marker_path}; inspect manually"
        )
    for field in ACTIVE_SESSION_FIELDS:
        if field == "schema_version":
            if marker[field] != 1:
                raise ValueError(
                    "active session schema_version mismatch: "
                    f"saved={marker[field]!r} expected=1"
                )
            continue
        if type(marker[field]) is not str or not marker[field]:
            raise ValueError(f"active session {field} must be a non-empty string")
    for field in ("session_id", "shared_root_uuid"):
        try:
            parsed = uuid.UUID(marker[field])
        except ValueError as error:
            raise ValueError(
                f"active session {field} must be a canonical UUID"
            ) from error
        if str(parsed) != marker[field]:
            raise ValueError(
                f"active session {field} must be a canonical UUID"
            )
    commit = marker["git_commit"]
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError(
            "active session git_commit must be a full lowercase commit SHA"
        )
    if not is_pinned_sha256(marker["run_identity_sha256"]):
        raise ValueError("active session run identity SHA-256 is invalid")
    if marker["account_label"] not in {"A", "B", "C"}:
        raise ValueError("active session account label must be A, B, or C")
    if marker["evaluation_scope"] != VALIDATION_ONLY:
        raise ValueError(PROHIBITED_FORMAL_TEST_REASON)
    if marker["checkpoint_cadence"] != "every_epoch":
        raise ValueError("active session checkpoint cadence must be every_epoch")
    if marker["maximum_quota_loss"] != "one_incomplete_epoch":
        raise ValueError(
            "active session maximum quota loss must be one_incomplete_epoch"
        )
    return dict(marker)


def _read_active_session(marker_path: str | Path) -> dict[str, Any]:
    marker = json.loads(Path(marker_path).read_text(encoding="utf-8"))
    return _validate_active_session_marker(marker, marker_path=marker_path)


def require_active_session_identity(
    marker: Mapping[str, Any],
    *,
    session_id: str,
    run_version: str,
    git_commit: str,
    shared_root_uuid: str,
    run_identity_sha256: str,
    evaluation_scope: str = VALIDATION_ONLY,
    marker_path: str | Path = "<provider>",
) -> dict[str, Any]:
    """Require the exact active session before a durable publish boundary."""
    observed = _validate_active_session_marker(marker, marker_path=marker_path)
    expected = {
        "session_id": str(session_id),
        "run_version": str(run_version),
        "git_commit": str(git_commit),
        "shared_root_uuid": str(shared_root_uuid),
        "run_identity_sha256": str(run_identity_sha256),
        "evaluation_scope": str(evaluation_scope),
    }
    mismatches = [
        f"{field}: saved={observed[field]!r} expected={value!r}"
        for field, value in expected.items()
        if observed[field] != value
    ]
    if mismatches:
        raise ValueError(
            "active session identity drift; refusing durable work: "
            + "; ".join(mismatches)
        )
    return observed


def canonical_identity_sha256(value: Mapping[str, Any]) -> str:
    require_primitive_identity(dict(value))
    encoded = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _replace_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    canonical = json.dumps(
        dict(value), allow_nan=False, ensure_ascii=True, sort_keys=True
    ) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(canonical)
            handle.flush()
            os.fsync(handle.fileno())
        reopened = json.loads(temporary.read_text(encoding="ascii"))
        if reopened != dict(value):
            raise ValueError(f"temporary JSON reopen mismatch: {path.name}")
        os.replace(temporary, path)
        temporary = None
        if json.loads(path.read_text(encoding="ascii")) != dict(value):
            raise ValueError(f"published JSON reopen mismatch: {path.name}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json_once_or_identical(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != dict(value):
            raise FileExistsError(
                f"existing audit record differs; refusing overwrite: {path}"
            )
        return
    write_json_atomic(path, value)


def audit_event_id(event: str, *, run_version: str, subject_session_id: str) -> str:
    """Deterministic identity for one handoff/takeover event.

    The identity never contains a timestamp, hostname or account label, so a
    crashed session that retries the same event produces the same identity
    instead of a second, drifted audit payload.
    """
    if type(event) is not str or not event:
        raise ValueError("audit event name must be a non-empty string")
    return f"{event}:{run_version}:{subject_session_id}"


def _read_existing_audit_event(path: Path) -> dict[str, Any] | None:
    """Return an already published audit payload, or None when absent."""
    if not path.exists():
        return None
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"existing audit record is unreadable: {path}") from error
    if not isinstance(observed, dict):
        raise ValueError(f"existing audit record is not a mapping: {path}")
    return observed


def _reused_audit_timestamp(
    published: Mapping[str, Any] | None, field: str, path: Path
) -> str:
    """Reuse the first published timestamp so retries stay byte-identical."""
    if published is None:
        return utc_now()
    saved = published.get(field)
    if type(saved) is not str or not saved:
        raise ValueError(
            f"existing audit record has no usable {field}; inspect manually: {path}"
        )
    return saved


def _publish_audit_event_once(
    path: Path, value: Mapping[str, Any], *, stable_fields: Sequence[str]
) -> dict[str, Any]:
    """Publish one deterministic audit event or accept the identical retry."""
    published = _read_existing_audit_event(path)
    if published is None:
        write_json_atomic(path, value)
        return json.loads(path.read_text(encoding="utf-8"))
    mismatches = [
        f"{field}: saved={published.get(field)!r} retry={value[field]!r}"
        for field in stable_fields
        if published.get(field) != value[field]
    ]
    if mismatches or published != dict(value):
        raise FileExistsError(
            "existing audit record differs; refusing overwrite: "
            f"{path}" + ("" if not mismatches else ": " + "; ".join(mismatches))
        )
    return published


def _read_takeover_audit_history(
    history_directory: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    """Read every published takeover audit or fail loud on an unusable one."""
    history: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(history_directory.glob(f"*{TAKEOVER_AUDIT_SUFFIX}")):
        record = _read_existing_audit_event(path)
        if record is None:
            continue
        if set(record) != set(MANUAL_TAKEOVER_STABLE_FIELDS):
            raise ValueError(
                f"takeover audit schema mismatch at {path}; inspect manually"
            )
        if record["schema_version"] != 1:
            raise ValueError(
                "takeover audit schema_version mismatch: "
                f"saved={record['schema_version']!r} expected=1: {path}"
            )
        subject = record.get("previous_active_session")
        subject_id = (
            subject.get("session_id") if isinstance(subject, Mapping) else None
        )
        if type(subject_id) is not str or not subject_id:
            raise ValueError(
                f"takeover audit has no usable previous session id: {path}"
            )
        if path.name != f"{subject_id}{TAKEOVER_AUDIT_SUFFIX}":
            raise ValueError(
                f"takeover audit name disagrees with its retired session: {path}"
            )
        # The graceful path revalidates its retired marker through
        # _read_active_session, so the takeover path must hold the retired
        # snapshot to the same schema instead of trusting the stored JSON.
        try:
            _validate_active_session_marker(subject, marker_path=path)
        except ValueError as error:
            raise ValueError(
                f"takeover audit retired session is unusable: {path}: {error}"
            ) from error
        history.append((path, record))
    return history


def _adopt_published_takeover_replacement(
    history_directory: Path,
    *,
    current: Mapping[str, Any],
    run_version: str,
    git_commit: str,
    shared_root_uuid: str,
    account_label: str,
    run_identity_sha256: str,
    evaluation_scope: str,
    marker_path: Path,
) -> dict[str, Any] | None:
    """Recognise a lost response after an already completed takeover.

    A restarted runtime mints a new candidate UUID, so a lost response cannot be
    detected by comparing UUIDs. The published audit answers it instead: the
    active marker is this operator's own replacement when exactly one audit
    names it and no stable field drifted. Returns ``None`` when the caller is a
    genuinely different operator, which is the next real takeover.
    """
    current_session_id = current["session_id"]
    history = _read_takeover_audit_history(history_directory)
    retired = {
        record["previous_active_session"]["session_id"] for _, record in history
    }
    replacements: list[tuple[Path, dict[str, Any]]] = []
    for path, record in history:
        if record["previous_active_session"]["session_id"] == current_session_id:
            # This audit retires the active marker, so it belongs to the crashed
            # takeover the caller completes instead of to an adoption.
            continue
        replacement = record.get("replacement_session_id")
        if type(replacement) is not str or not replacement:
            raise ValueError(
                f"takeover audit has no usable replacement_session_id: {path}"
            )
        if replacement == current_session_id:
            replacements.append((path, record))
        elif replacement not in retired:
            raise ValueError(
                "takeover audit history contradicts the active session marker: "
                f"{path} names replacement {replacement!r}, which is neither the "
                f"active session {current_session_id!r} nor retired by a later "
                "audit; inspect manually"
            )
    if not replacements:
        return None
    if len(replacements) > 1:
        raise ValueError(
            "ambiguous takeover audits name the same replacement session; "
            "inspect manually: "
            + ", ".join(str(path) for path, _ in replacements)
        )
    audit_path, audit = replacements[0]
    missing = [
        field for field in MANUAL_TAKEOVER_STABLE_FIELDS if field not in audit
    ]
    if missing:
        raise ValueError(
            f"takeover audit is missing stable fields {missing}: {audit_path}"
        )
    retired_session = audit["previous_active_session"]
    expected_event_id = audit_event_id(
        MANUAL_TAKEOVER_EVENT,
        run_version=str(run_version),
        subject_session_id=retired_session["session_id"],
    )
    if (
        audit["event"] != MANUAL_TAKEOVER_EVENT
        or audit["event_id"] != expected_event_id
        or audit["confirmation"] != MANUAL_TAKEOVER_CONFIRMATION
    ):
        raise ValueError(
            "takeover audit event identity drift; refusing to adopt the active "
            f"session: {audit_path}"
        )
    drift = [
        f"{field}: audit={retired_session.get(field)!r} expected={value!r}"
        for field, value in (
            ("run_version", str(run_version)),
            ("git_commit", str(git_commit)),
            ("shared_root_uuid", str(shared_root_uuid)),
            ("run_identity_sha256", str(run_identity_sha256)),
            ("evaluation_scope", str(evaluation_scope)),
        )
        if retired_session.get(field) != value
    ]
    if drift:
        raise ValueError(
            "takeover audit identity drift; refusing to adopt the active "
            f"session: {audit_path}: " + "; ".join(drift)
        )
    if current["account_label"] != str(account_label):
        # A different confirmed operator is the next real takeover, so the
        # caller must retire this marker instead of adopting it.
        return None
    return require_active_session_identity(
        current,
        session_id=current_session_id,
        run_version=run_version,
        git_commit=git_commit,
        shared_root_uuid=shared_root_uuid,
        run_identity_sha256=run_identity_sha256,
        evaluation_scope=evaluation_scope,
        marker_path=marker_path,
    )


def start_sequential_session(
    marker_path: str | Path,
    *,
    session_id: str,
    run_version: str,
    git_commit: str,
    shared_root_uuid: str,
    account_label: str,
    run_identity_sha256: str,
    manual_takeover_confirmed: bool = False,
    history_directory: str | Path | None = None,
    evaluation_scope: str = VALIDATION_ONLY,
) -> dict[str, Any]:
    """Start one manually serialized session; this is deliberately not a CAS."""
    marker_path = Path(marker_path)
    if not marker_path.parent.is_dir():
        raise FileNotFoundError(
            f"run-version directory is missing: {marker_path.parent}"
        )
    def build_marker(effective_session_id: str) -> dict[str, Any]:
        return _validate_active_session_marker(
            {
                "schema_version": 1,
                "session_id": str(effective_session_id),
                "run_version": str(run_version),
                "git_commit": str(git_commit),
                "shared_root_uuid": str(shared_root_uuid),
                "evaluation_scope": str(evaluation_scope),
                "account_label": str(account_label),
                "hostname": socket.gethostname(),
                "started_utc": utc_now(),
                "run_identity_sha256": str(run_identity_sha256),
                "checkpoint_cadence": "every_epoch",
                "maximum_quota_loss": "one_incomplete_epoch",
            },
            marker_path=marker_path,
        )

    if not marker_path.exists():
        write_json_atomic(marker_path, build_marker(session_id))
        return require_active_session_identity(
            _read_active_session(marker_path),
            session_id=session_id,
            run_version=run_version,
            git_commit=git_commit,
            shared_root_uuid=shared_root_uuid,
            run_identity_sha256=run_identity_sha256,
            marker_path=marker_path,
        )
    previous = _read_active_session(marker_path)
    if manual_takeover_confirmed is not True:
        raise FileExistsError(
            "an active session already exists; confirm the previous runtime is "
            "stopped before an explicit manual takeover"
        )
    if history_directory is None:
        raise ValueError("manual takeover requires an existing history directory")
    history_directory = Path(history_directory)
    if not history_directory.is_dir():
        raise FileNotFoundError(
            f"session history directory is missing: {history_directory}"
        )
    if previous["session_id"] == str(session_id):
        # The replacement marker was already published and only the response was
        # lost; re-publishing it would drift hostname/started_utc for no reason.
        return require_active_session_identity(
            previous,
            session_id=session_id,
            run_version=run_version,
            git_commit=git_commit,
            shared_root_uuid=shared_root_uuid,
            run_identity_sha256=run_identity_sha256,
            marker_path=marker_path,
        )
    adopted = _adopt_published_takeover_replacement(
        history_directory,
        current=previous,
        run_version=run_version,
        git_commit=git_commit,
        shared_root_uuid=shared_root_uuid,
        account_label=account_label,
        run_identity_sha256=run_identity_sha256,
        evaluation_scope=evaluation_scope,
        marker_path=marker_path,
    )
    if adopted is not None:
        # The previous takeover already published this marker and only the
        # response was lost; a restarted runtime must not mint a second
        # transition just because it generated a new candidate id.
        return adopted
    # The audit is keyed on the retired session only, so a crashed takeover
    # retries the same event instead of minting a second replacement id.
    audit_path = history_directory / f"{previous['session_id']}{TAKEOVER_AUDIT_SUFFIX}"
    published = _read_existing_audit_event(audit_path)
    replacement_session_id = str(session_id)
    if published is not None:
        saved_replacement = published.get("replacement_session_id")
        if type(saved_replacement) is not str or not saved_replacement:
            raise ValueError(
                "existing takeover audit has no usable replacement_session_id; "
                f"inspect manually: {audit_path}"
            )
        replacement_session_id = saved_replacement
    marker = build_marker(replacement_session_id)
    audit = {
        "schema_version": 1,
        "event": MANUAL_TAKEOVER_EVENT,
        "event_id": audit_event_id(
            MANUAL_TAKEOVER_EVENT,
            run_version=str(run_version),
            subject_session_id=previous["session_id"],
        ),
        "confirmation": MANUAL_TAKEOVER_CONFIRMATION,
        "confirmed_utc": _reused_audit_timestamp(
            published, "confirmed_utc", audit_path
        ),
        "previous_active_session": previous,
        "replacement_session_id": replacement_session_id,
    }
    _publish_audit_event_once(
        audit_path, audit, stable_fields=MANUAL_TAKEOVER_STABLE_FIELDS
    )
    _replace_json_atomic(marker_path, marker)
    return _read_active_session(marker_path)


def complete_sequential_session(
    marker_path: str | Path,
    *,
    session_id: str,
    history_directory: str | Path,
    checkpoint_integrity: Mapping[str, Any],
    result_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Record a verified graceful handoff, then atomically retire the marker."""
    marker_path = Path(marker_path)
    history_directory = Path(history_directory)
    if not history_directory.is_dir():
        raise FileNotFoundError(
            f"session history directory is missing: {history_directory}"
        )
    completion_path = history_directory / f"{session_id}.completed.json"
    snapshot_path = history_directory / f"{session_id}.active.json"
    published = _read_existing_audit_event(completion_path)

    def build_completion(marker: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "event": GRACEFUL_HANDOFF_EVENT,
            "event_id": audit_event_id(
                GRACEFUL_HANDOFF_EVENT,
                run_version=marker["run_version"],
                subject_session_id=marker["session_id"],
            ),
            "completed_utc": _reused_audit_timestamp(
                published, "completed_utc", completion_path
            ),
            "active_session": dict(marker),
            "checkpoint_integrity": dict(checkpoint_integrity),
            "result_identity": dict(result_identity),
        }

    if not marker_path.exists():
        # The marker already transitioned; only a lost response is retryable and
        # it must reproduce the published payload exactly.
        if published is None or not snapshot_path.is_file():
            raise FileNotFoundError(
                "active session marker is missing without a completed graceful "
                f"handoff: {marker_path}"
            )
        retired = _read_active_session(snapshot_path)
        if retired["session_id"] != session_id:
            raise PermissionError("retired active session belongs to another session")
        return _publish_audit_event_once(
            completion_path,
            build_completion(retired),
            stable_fields=GRACEFUL_HANDOFF_STABLE_FIELDS,
        )
    marker = _read_active_session(marker_path)
    if marker["session_id"] != session_id:
        raise PermissionError("active session belongs to another session")
    if snapshot_path.exists():
        raise FileExistsError(
            f"active-session audit snapshot already exists: {snapshot_path}"
        )
    completion = build_completion(marker)
    _publish_audit_event_once(
        completion_path, completion, stable_fields=GRACEFUL_HANDOFF_STABLE_FIELDS
    )
    os.replace(marker_path, snapshot_path)
    if _read_active_session(snapshot_path) != marker or marker_path.exists():
        raise ValueError("graceful handoff marker transition failed reopen validation")
    return completion


class SequentialSessionWriteGuard:
    """Re-open the active marker at durable publish boundaries.

    This enforces the declared manual-serialization contract in this process; it
    is not a cross-client compare-and-swap primitive.
    """

    ENV_KEYS = {
        "marker_path": "PANDERM_ACTIVE_SESSION_PATH",
        "session_id": "PANDERM_ACTIVE_SESSION_ID",
        "run_version": "PANDERM_ACTIVE_RUN_VERSION",
        "git_commit": "PANDERM_ACTIVE_GIT_COMMIT",
        "shared_root_uuid": "PANDERM_ACTIVE_SHARED_ROOT_UUID",
        "run_identity_sha256": "PANDERM_ACTIVE_RUN_IDENTITY_SHA256",
    }

    def __init__(self, **values):
        self._values = values

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None):
        source = os.environ if environment is None else environment
        values = {}
        for field, key in cls.ENV_KEYS.items():
            value = source.get(key)
            if type(value) is not str or not value:
                raise RuntimeError(f"missing sequential session environment: {key}")
            values[field] = value
        return cls(**values)

    def require(self, phase: str) -> dict[str, Any]:
        if type(phase) is not str or not phase:
            raise ValueError("sequential session phase must be non-empty")
        marker = _read_active_session(self._values["marker_path"])
        return require_active_session_identity(
            marker,
            session_id=self._values["session_id"],
            run_version=self._values["run_version"],
            git_commit=self._values["git_commit"],
            shared_root_uuid=self._values["shared_root_uuid"],
            run_identity_sha256=self._values["run_identity_sha256"],
            marker_path=self._values["marker_path"],
        )

    def bind_run_identity(self, run_identity: Mapping[str, Any]) -> None:
        observed = canonical_identity_sha256(run_identity)
        if observed != self._values["run_identity_sha256"]:
            raise ValueError("active session run identity does not match this process")
        self.require("bind immutable run identity")


def probe_shared_drive(root: str | Path) -> dict[str, str]:
    """Verify read/write/delete, child-process and atomic replace on the root."""
    root = require_existing_shared_root(root)
    probe_id = uuid.uuid4().hex
    direct = root / f".panderm_probe_{probe_id}.txt"
    child = root / f".panderm_child_probe_{probe_id}.txt"
    replaced = root / f".panderm_replace_probe_{probe_id}.txt"
    nested = root / f".panderm_nested_probe_{probe_id}"
    try:
        direct.write_text("direct-ok\n", encoding="utf-8")
        if direct.read_text(encoding="utf-8") != "direct-ok\n":
            raise OSError("direct shared Drive read/write probe failed")
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; "
                "Path(sys.argv[1]).write_text('child-ok\\n', encoding='utf-8')",
                str(child),
            ],
            check=True,
        )
        if child.read_text(encoding="utf-8") != "child-ok\n":
            raise OSError("child-process shared Drive probe failed")
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(b"replace-ok\n")
        os.replace(temp_path, replaced)
        if replaced.read_text(encoding="utf-8") != "replace-ok\n":
            raise OSError("same-directory replace probe failed")
        nested.mkdir()
        checkpoint_dir = ensure_tree(nested, Path("checkpoints") / ARCH / "C1_seed0")
        checkpoint_probe = checkpoint_dir / "last.pt.probe"
        checkpoint_probe.write_bytes(b"checkpoint-ok")
        if checkpoint_probe.read_bytes() != b"checkpoint-ok":
            raise OSError("nested checkpoint read/write probe failed")
        checkpoint_probe.unlink()
        if checkpoint_probe.exists():
            raise OSError("shared Drive delete probe failed")
        return {"status": "passed", "resolved_root": str(root)}
    finally:
        for path in (direct, child, replaced):
            path.unlink(missing_ok=True)
        if nested.exists():
            for directory in sorted(
                (item for item in nested.rglob("*") if item.is_dir()), reverse=True
            ):
                directory.rmdir()
            nested.rmdir()


def require_no_deployment_contamination(project_root: str | Path) -> list[str]:
    """Refuse any PanDerm reference inside the public deployment surface."""
    project_root = Path(project_root)
    guarded = [
        Path("app"),
        Path("deploy"),
        Path("Dockerfile"),
        Path("Dockerfile.render"),
        Path("render.yaml"),
        Path("requirements-deploy.txt"),
    ]
    hits: list[str] = []
    for relative in guarded:
        target = project_root / relative
        if not target.exists():
            continue
        files = [target] if target.is_file() else [
            item for item in target.rglob("*") if item.is_file()
        ]
        for item in files:
            try:
                text = item.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if "panderm" in text.lower():
                hits.append(str(item.relative_to(project_root)).replace(os.sep, "/"))
    if hits:
        raise ValueError(
            "CC BY-NC-ND 4.0 forbids sharing adapted PanDerm weights; PanDerm must "
            f"never reach the public deployment surface, found in: {sorted(hits)}"
        )
    return hits


def evaluate_non_collapse_gate(
    *,
    result: Mapping[str, Any],
    prediction_counts: Mapping[str, int],
    backbone_gradient_verified: bool,
    backbone_parameters_updated: bool,
    identity_complete: bool,
    no_test_access: bool,
    provenance_allows_next_stage: bool,
) -> dict[str, bool]:
    """Compute the fixed non-collapse checks for a validation-only run."""
    history = result.get("history") or []
    losses = [float(item["train_loss"]) for item in history]
    metric_values = [float(item["val_df_f1"]) for item in history] + [
        float(item["val_macro_f1"]) for item in history
    ]
    total_predicted = int(sum(prediction_counts.values()))
    best = float(result.get("best_val_df_f1", -1.0))
    checks = {
        "finite_losses": bool(losses) and all(np.isfinite(losses)),
        "finite_metrics": bool(metric_values) and all(np.isfinite(metric_values)),
        "best_validation_df_f1_positive": best > 0,
        "predicted_df_positive": int(prediction_counts.get("df", 0)) > 0,
        "at_least_two_predicted_classes": sum(
            1 for value in prediction_counts.values() if int(value) > 0
        ) >= 2,
        "not_all_nv": int(prediction_counts.get("nv", 0)) < total_predicted,
        "not_all_df": int(prediction_counts.get("df", 0)) < total_predicted,
        "backbone_gradient_verified": bool(backbone_gradient_verified),
        "backbone_parameters_updated": bool(backbone_parameters_updated),
        "identity_complete": bool(identity_complete),
        "test_metrics_null": result.get("test_metrics") is None,
        "no_test_access": bool(no_test_access),
        "provenance_allows_next_stage": bool(provenance_allows_next_stage),
    }
    missing = [key for key in NON_COLLAPSE_CHECK_KEYS if key not in checks]
    if missing:
        raise ValueError(f"non-collapse gate is missing checks: {missing}")
    return checks


def aggregate_results(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """PanDerm v1 has no formal/test aggregation entry point."""
    del runs
    raise ValueError(PROHIBITED_FORMAL_TEST_REASON)


def summarize_provenance() -> dict[str, Any]:
    """Portable provenance block embedded in every record."""
    return {
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_model_selector": UPSTREAM_MODEL_SELECTOR,
        "upstream_model_factory": UPSTREAM_MODEL_FACTORY,
        "checkpoint_filename": CHECKPOINT_FILENAME,
        "checkpoint_source_url": CHECKPOINT_SOURCE_URL,
        "checkpoint_sha256_provenance": CHECKPOINT_SHA256_PROVENANCE,
        "paper_doi": PAPER_DOI,
        "license_review": dict(LICENSE_REVIEW),
        "contamination_review": dict(CONTAMINATION_REVIEW),
        "claim_boundary": CLAIM_BOUNDARY,
        "deployment_allowed": DEPLOYMENT_ALLOWED,
        "formal_training_allowed": FORMAL_TRAINING_ALLOWED,
        "test_access_allowed": TEST_ACCESS_ALLOWED,
        "attribution": ATTRIBUTION,
    }


def dump_provenance() -> str:
    return json.dumps(summarize_provenance(), indent=2, sort_keys=True)
