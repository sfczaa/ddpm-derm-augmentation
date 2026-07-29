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


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    """Write one new JSON record atomically, then re-open it exactly."""
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"JSON parent directory is missing: {path.parent}")
    if path.exists() and path.stat().st_size:
        raise FileExistsError(f"refusing to overwrite existing JSON record: {path}")

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
            raise ValueError(f"temporary JSON verification failed: {temporary}")
        os.replace(temporary, path)
        temporary = None
        reopened = json.loads(path.read_text(encoding="utf-8"))
        if reopened != canonical:
            raise ValueError(f"JSON replace/read verification failed: {path}")
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
    with source.open("rb") as reader, destination.open("xb") as writer:
        while True:
            chunk = reader.read(8 * 1024 * 1024)
            if not chunk:
                break
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
) -> dict[str, Any]:
    """Build and verify one immutable train/val-only tar, then write READY last."""
    approved_content_sha256 = require_approved_content_identity(
        expected_file_content_identity_sha256
    )
    cache_directory = Path(cache_directory)
    runtime_temporary_root = Path(runtime_temporary_root).resolve(strict=True)
    if cache_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite existing validation archive cache: "
            f"{cache_directory}"
        )
    if not cache_directory.parent.is_dir():
        raise FileNotFoundError(
            f"validation archive cache parent is missing: {cache_directory.parent}"
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
        cache_directory.mkdir()
        cached_archive = cache_directory / VALIDATION_ARCHIVE_FILENAME
        _copy_file_with_progress(
            temporary_archive, cached_archive, phase="archive-publish"
        )
        if sha256_file(cached_archive) != archive_sha256:
            raise ValueError("published validation archive SHA-256 mismatch")
        identity_path = cache_directory / VALIDATION_ARCHIVE_IDENTITY_FILENAME
        write_json_atomic(identity_path, identity)
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
            cache_directory / VALIDATION_ARCHIVE_READY_FILENAME, ready
        )
    return identity


def validate_validation_archive_cache(
    cache_directory: str | Path,
    *,
    expected_file_content_identity_sha256: str,
    expected_fixed_split_identity: str,
    expected_manifest_sha256: Mapping[str, str],
    expected_class_mapping_sha256: str,
) -> dict[str, Any]:
    """Fail loud on any incomplete, drifted, or tampered durable cache."""
    approved_content_sha256 = require_approved_content_identity(
        expected_file_content_identity_sha256
    )
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
) -> dict[str, Any]:
    """Copy one verified tar to runtime, safely extract, then publish atomically."""
    approved_content_sha256 = require_approved_content_identity(
        expected_file_content_identity_sha256
    )
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
    )
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
        inventory = _validate_extracted_validation_data(
            extracted,
            identity,
            approved_file_content_identity_sha256=approved_content_sha256,
        )
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


VALIDATION_RUN_LOCK_FILENAME = "validation_run.lock.json"
VALIDATION_RUN_LOCK_CLEAR_CONFIRMATION = "CLEAR STALE PANDERM VALIDATION LOCK"
VALIDATION_RUN_LOCK_FIELDS = (
    "schema_version",
    "session_id",
    "run_version",
    "git_commit",
    "shared_root_uuid",
    "evaluation_scope",
    "account_label",
    "hostname",
    "acquired_utc",
)


def validation_run_lock_path(
    shared_run_root: str | Path, *, run_version: str = RUN_VERSION
) -> Path:
    """The one durable lock for a run version.

    Deliberately fixed per run version and *outside* any timestamped attempt
    directory: a lock that lived inside ``validation_runs/<timestamp>/`` would be
    a different path for every attempt, so two accounts would each create their
    own and never collide.
    """
    return Path(shared_run_root) / run_version / VALIDATION_RUN_LOCK_FILENAME


def _read_validation_run_lock(lock_path: str | Path) -> dict[str, Any]:
    marker = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if not isinstance(marker, dict) or set(marker) != set(VALIDATION_RUN_LOCK_FIELDS):
        raise ValueError(
            f"validation run lock schema mismatch at {lock_path}; inspect manually"
        )
    for field in VALIDATION_RUN_LOCK_FIELDS:
        if field == "schema_version":
            if type(marker[field]) is not int:
                raise ValueError("validation run lock schema_version must be int")
            continue
        if type(marker[field]) is not str or not marker[field]:
            raise ValueError(f"validation run lock {field} must be a non-empty string")
    return marker


def acquire_validation_run_lock(
    lock_path: str | Path,
    *,
    session_id: str,
    run_version: str,
    git_commit: str,
    shared_root_uuid: str,
    account_label: str,
    evaluation_scope: str = VALIDATION_ONLY,
) -> dict[str, Any]:
    """Atomically take the single validation run lock, or fail loud.

    Acquisition is a single exclusive create (``open("x")`` inside
    ``create_running_marker``), never ``exists()`` then write: two accounts that
    both see "no attempt yet" must not both proceed. The marker is re-read after
    creation so a caller only continues if it still owns the lock it just took.
    """
    if evaluation_scope != VALIDATION_ONLY:
        raise ValueError(PROHIBITED_FORMAL_TEST_REASON)
    try:
        uuid.UUID(str(session_id))
    except ValueError as error:
        raise ValueError("validation run lock session_id must be a UUID") from error
    lock_path = Path(lock_path)
    if not lock_path.parent.is_dir():
        raise FileNotFoundError(
            f"validation run lock directory is missing: {lock_path.parent}"
        )
    marker = {
        "schema_version": VALIDATION_ARCHIVE_SCHEMA_VERSION,
        "session_id": str(session_id),
        "run_version": str(run_version),
        "git_commit": str(git_commit),
        "shared_root_uuid": str(shared_root_uuid),
        "evaluation_scope": evaluation_scope,
        "account_label": str(account_label),
        "hostname": socket.gethostname(),
        "acquired_utc": utc_now(),
    }
    try:
        create_running_marker(lock_path, marker)
    except FileExistsError as error:
        try:
            holder = _read_validation_run_lock(lock_path)
            detail = (
                f"session_id={holder['session_id']} "
                f"account_label={holder['account_label']} "
                f"hostname={holder['hostname']} "
                f"acquired_utc={holder['acquired_utc']} "
                f"run_version={holder['run_version']}"
            )
        except (OSError, ValueError) as read_error:
            detail = f"existing lock could not be parsed: {read_error}"
        raise FileExistsError(
            "another PanDerm validation session already holds the run lock; "
            "no staging, attempt or runner may start. Existing owner: "
            f"{detail}. Lock: {lock_path}. If that runtime is definitely stopped, "
            "clear it manually with clear_stale_validation_run_lock; Run all must "
            "never clear it automatically."
        ) from error
    # Re-read: only continue while we still own what we just created.
    observed = _read_validation_run_lock(lock_path)
    if observed != marker:
        raise ValueError(
            "validation run lock changed between create and re-read; refusing to "
            f"proceed. Lock: {lock_path}"
        )
    return observed


def release_validation_run_lock(
    lock_path: str | Path, *, session_id: str
) -> dict[str, Any]:
    """Release only a lock this session owns; a wrong owner never deletes it."""
    lock_path = Path(lock_path)
    if not lock_path.is_file():
        raise FileNotFoundError(
            f"validation run lock is not present to release: {lock_path}"
        )
    holder = _read_validation_run_lock(lock_path)
    if holder["session_id"] != str(session_id):
        raise PermissionError(
            "refusing to release a validation run lock owned by another session: "
            f"owner={holder['session_id']} caller={session_id} lock={lock_path}"
        )
    lock_path.unlink()
    return holder


def clear_stale_validation_run_lock(
    lock_path: str | Path,
    *,
    stale_session_id: str,
    confirmation: str,
) -> dict[str, Any]:
    """Manual, human-confirmed recovery only.

    Never called by Run all and never time-based: an abruptly killed runtime must
    leave the lock behind so the next run fails loud, and the operator must first
    confirm the old Colab runtime really is stopped. Switching account A/B/C is
    not by itself a reason to clear.
    """
    lock_path = Path(lock_path)
    if confirmation != VALIDATION_RUN_LOCK_CLEAR_CONFIRMATION:
        raise ValueError(
            "stale validation run lock confirmation text did not match "
            f"{VALIDATION_RUN_LOCK_CLEAR_CONFIRMATION!r}"
        )
    holder = _read_validation_run_lock(lock_path)
    if holder["session_id"] != str(stale_session_id):
        raise PermissionError(
            "stale validation run lock owner mismatch; refusing to clear: "
            f"owner={holder['session_id']} supplied={stale_session_id}"
        )
    clear_stale_marker(lock_path, "CLEAR STALE MARKER")
    return holder


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
