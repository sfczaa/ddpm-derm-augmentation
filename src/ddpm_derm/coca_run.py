"""Shared-Drive guards and result aggregation for the CoCa experiment."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ARCH = "coca_vit_b32"
MODEL_NAME = "coca_ViT-B-32"
PRETRAINED_TAG = "laion2b_s13b_b90k"
RUN_VERSION = "v1"
CHECKPOINT_FORMAT = "frozen_backbone_head_only_v1"
MAX_COCA_CHECKPOINT_BYTES = 100 * 1024 * 1024
EXPECTED_CANDIDATE_SHA256 = (
    "9ef9b44e404f74aab8211f4e7d123da3258ba8ba4e3004a4147d1761ed343b34"
)
IMMUTABLE_IDENTITY_KEYS = (
    "git_commit",
    "run_version",
    "candidate_manifest_sha256",
    "fixed_split_identity",
    "shared_root_uuid",
    "formal_output_identity",
    "model_identity",
    "checkpoint_format",
)


def checkpoint_size(path: str | Path, *, arch: str) -> int:
    """Return size and enforce the CoCa-only checkpoint limit."""
    size = Path(path).stat().st_size
    if arch == ARCH and size > MAX_COCA_CHECKPOINT_BYTES:
        raise ValueError(
            f"CoCa checkpoint exceeds {MAX_COCA_CHECKPOINT_BYTES} bytes: "
            f"{size} bytes ({size / 1024 / 1024:.2f} MiB): {path}"
        )
    return size


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"JSON parent directory is missing: {path.parent}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved != dict(value):
        raise ValueError(f"JSON replace/read verification failed: {path}")


def require_existing_shared_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(
            f"shared run root is missing; create/share its MyDrive shortcut first: {root}"
        )
    return root.resolve()


def ensure_tree(root: str | Path, relative: str | Path) -> Path:
    root = require_existing_shared_root(root)
    current = root
    for part in Path(relative).parts:
        current = current / part
        if not current.is_dir():
            current.mkdir()
        if not current.is_dir():
            raise OSError(f"shared Drive directory is not visible: {current}")
        probe = current / f".dir_probe_{uuid.uuid4().hex}"
        probe.write_text("ok\n", encoding="utf-8")
        if probe.read_text(encoding="utf-8") != "ok\n":
            raise OSError(f"shared Drive directory is not writable: {current}")
        probe.unlink()
    return current


def probe_shared_drive(root: str | Path) -> dict[str, str]:
    root = require_existing_shared_root(root)
    probe_id = uuid.uuid4().hex
    direct = root / f".coca_probe_{probe_id}.txt"
    child = root / f".coca_child_probe_{probe_id}.txt"
    replaced = root / f".coca_replace_probe_{probe_id}.txt"
    nested = root / f".coca_nested_probe_{probe_id}"
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
        return {"status": "passed", "resolved_root": str(root)}
    finally:
        for path in (direct, child, replaced):
            path.unlink(missing_ok=True)
        if nested.exists():
            for directory in sorted(
                (p for p in nested.rglob("*") if p.is_dir()), reverse=True
            ):
                directory.rmdir()
            nested.rmdir()


def create_or_validate_sentinel(
    path: str | Path,
    *,
    shortcut_alias: str,
    resolved_path: str,
    drive_folder_id: str | None,
    run_version: str = RUN_VERSION,
) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    sentinel = {
        "shared_root_uuid": str(uuid.uuid4()),
        "shortcut_alias": shortcut_alias,
        "resolved_path": resolved_path,
        "drive_folder_id": drive_folder_id,
        "created_utc": utc_now(),
        "run_version": run_version,
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(sentinel, handle, indent=2)
    return sentinel


def require_validation_record(
    record: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if record.get("validation_status") != "VALIDATION PASSED":
        raise ValueError("validation record is not VALIDATION PASSED")
    if record.get("formal_training_started") is not False:
        raise ValueError("validation record does not prove formal_training_started=false")
    mismatches = [
        f"{key}: record={record.get(key)!r} expected={value!r}"
        for key, value in expected.items()
        if record.get(key) != value
    ]
    if mismatches:
        raise ValueError("validation record identity mismatch: " + "; ".join(mismatches))


def create_running_marker(path: str | Path, marker: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        raise FileExistsError(f"concurrent run marker exists; retained:\n{existing}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(marker), handle, indent=2)


def clear_stale_marker(path: str | Path, confirmation: str) -> None:
    if confirmation != "CLEAR STALE MARKER":
        raise ValueError("stale marker confirmation text did not match")
    Path(path).unlink()


def require_resume_identity(
    saved: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    mismatches = [
        key for key in IMMUTABLE_IDENTITY_KEYS if saved.get(key) != current.get(key)
    ]
    if mismatches:
        raise ValueError(f"formal resume identity mismatch: {mismatches}")


def session_marker(account_label: str, run_mode: str, identity: Mapping[str, Any]):
    if account_label not in {"A", "B", "C"}:
        raise ValueError("ACCOUNT_LABEL must be A, B, or C")
    if run_mode not in {"fresh", "resume"}:
        raise ValueError("RUN_MODE must be fresh or resume")
    return {
        "session_id": str(uuid.uuid4()),
        "account_label": account_label,
        "hostname": socket.gethostname(),
        "started_utc": utc_now(),
        "last_updated_utc": utc_now(),
        "run_mode": run_mode,
        **dict(identity),
    }


def aggregate_results(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {(variant, seed) for variant in ("C1", "C4") for seed in (0, 1, 2)}
    found = {(run.get("variant"), run.get("seed")) for run in runs}
    if found != expected or len(runs) != 6:
        raise ValueError(f"expected exactly six C1/C4 seed runs, found {sorted(found)}")
    model_identities = [run["run_identity"]["model_identity"] for run in runs]
    first = model_identities[0]
    if first.get("arch") != ARCH or any(identity != first for identity in model_identities[1:]):
        raise ValueError("refusing to aggregate mixed architecture/model identities")
    formats = {run["run_identity"].get("checkpoint_format") for run in runs}
    if formats != {CHECKPOINT_FORMAT}:
        raise ValueError(f"refusing to aggregate mixed checkpoint formats: {formats}")
    output: dict[str, Any] = {"ddof": 0, "variants": {}, "paired_c4_minus_c1": {}}
    metric_keys = {
        "df_f1": "target_f1",
        "macro_f1": "macro_f1",
        "df_recall": "target_recall",
    }
    by_key = {(run["variant"], run["seed"]): run for run in runs}
    for variant in ("C1", "C4"):
        block = {}
        variant_runs = [by_key[(variant, seed)] for seed in (0, 1, 2)]
        for label, key in metric_keys.items():
            values = [float(run["test_metrics"][key]) for run in variant_runs]
            block[label] = {
                "values": values,
                "mean": float(np.mean(values)),
                "population_std": float(np.std(values, ddof=0)),
            }
        classes = sorted(variant_runs[0]["test_metrics"]["per_class_recall"])
        block["per_class_recall"] = {
            name: {
                "mean": float(np.mean([run["test_metrics"]["per_class_recall"][name] for run in variant_runs])),
                "population_std": float(np.std([run["test_metrics"]["per_class_recall"][name] for run in variant_runs], ddof=0)),
            }
            for name in classes
        }
        output["variants"][variant] = block
    differences = [
        float(by_key[("C4", seed)]["test_metrics"]["target_f1"])
        - float(by_key[("C1", seed)]["test_metrics"]["target_f1"])
        for seed in (0, 1, 2)
    ]
    output["paired_c4_minus_c1"] = {
        "seed_differences": differences,
        "mean": float(np.mean(differences)),
        "population_std": float(np.std(differences, ddof=0)),
        "c4_mean_minus_c1_mean": (
            output["variants"]["C4"]["df_f1"]["mean"]
            - output["variants"]["C1"]["df_f1"]["mean"]
        ),
    }
    return output
