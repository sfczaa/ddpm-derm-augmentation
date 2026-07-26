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
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

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
    "be1e0fb108b3bc58721cb5195f136c948160799438f222ac1ffd142230ac1ff1"
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

    relative_paths: list[str] = []
    destination_keys: set[str] = set()
    sources: dict[str, Path] = {}
    for split, manifest_path in manifest_paths.items():
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "image_path" not in reader.fieldnames:
                raise ValueError(f"{manifest_path} is missing image_path")
            for row_number, row in enumerate(reader, start=2):
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

    local_manifests = local_data_root / "manifests"
    local_manifests.mkdir(parents=True)
    for split, manifest_path in manifest_paths.items():
        shutil.copy2(manifest_path, local_manifests / f"{split}.csv")
    shutil.copy2(class_mapping, local_manifests / class_mapping.name)
    for relative in relative_paths:
        destination = local_data_root / Path(*relative.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sources[relative], destination)

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
    copied_images = {
        path.relative_to(local_data_root).as_posix()
        for path in local_data_root.rglob("*")
        if path.is_file() and local_manifests not in path.parents
    }
    expected_images = set(relative_paths)
    if copied_images != expected_images:
        raise ValueError(
            "validation staging image set mismatch: "
            f"missing={sorted(expected_images - copied_images)[:10]} "
            f"extra={sorted(copied_images - expected_images)[:10]}"
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
    return {
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
