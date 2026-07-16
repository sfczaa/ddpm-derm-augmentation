# %% [markdown]
# # C4-sqrt-balanced@585 — formal classifier training
# Run only after the executed validation notebook has been reviewed. This
# notebook trains seeds 0/1/2 sequentially, resumes from Drive, validates every
# completed seed, aggregates results, and writes descriptive comparisons only.
# Shared project files are read-only inputs; checkpoints go to the signed-in
# account's own MyDrive. Resume with that same account unless the run folder has
# first been shared or copied to another account.

# %%
RUN_MODE = "fresh"  # "fresh" or "resume"
RUN_VERSION = "c4_sqrt_balanced_v1"

# %%
EXPECTED_COMMIT = "2ea7a63834f55473ebaf6477bad3b33a538eb083"
REPO_URL = "https://github.com/sfczaa/ddpm-derm-augmentation.git"
BRANCH = "balanced-ddpm-exploration"
CANDIDATE_SHA256 = "9ef9b44e404f74aab8211f4e7d123da3258ba8ba4e3004a4147d1761ed343b34"
RUN_STORAGE_DIRNAME = "ddpm-derm-classifier-runs"
assert RUN_MODE in {"fresh", "resume"}
assert RUN_VERSION == "c4_sqrt_balanced_v1"
assert len(EXPECTED_COMMIT) == 40 and EXPECTED_COMMIT != "REPLACE_AFTER_PUSH", (
    "Notebook is not pinned yet. Stop: Phase 2 commit/push must happen first."
)

# %% [markdown]
# ## 1. GPU, Drive, exact code, fixed data, and candidate

# %%
import os
import subprocess
from pathlib import Path

from google.colab import drive, userdata

drive.mount("/content/drive")
subprocess.run(["nvidia-smi"], check=True)
PROJECT_DIR = Path("/content/ddpm-classifier-train-code")
assert not PROJECT_DIR.exists(), f"fresh runtime setup required; already exists: {PROJECT_DIR}"
token = userdata.get("GH_TOKEN")
assert token and len(token) > 20, "Colab Secret GH_TOKEN is required"
askpass = Path("/content/git-askpass-classifier-train.sh")
askpass.write_text(
    '#!/bin/sh\ncase "$1" in\n*Username*) printf \'%s\\n\' "sfczaa" ;;\n'
    '*Password*) printf \'%s\\n\' "$GH_TOKEN" ;;\nesac\n',
    encoding="utf-8",
)
askpass.chmod(0o700)
clone_env = os.environ.copy()
clone_env.update(GIT_ASKPASS=str(askpass), GIT_TERMINAL_PROMPT="0", GH_TOKEN=token)
try:
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(PROJECT_DIR)],
        check=True, env=clone_env,
    )
finally:
    askpass.unlink(missing_ok=True)
    clone_env.pop("GH_TOKEN", None)
subprocess.run(["git", "-C", str(PROJECT_DIR), "checkout", "--detach", EXPECTED_COMMIT], check=True)
commit = subprocess.run(
    ["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"],
    check=True, capture_output=True, text=True,
).stdout.strip()
remote = subprocess.run(
    ["git", "-C", str(PROJECT_DIR), "remote", "get-url", "origin"],
    check=True, capture_output=True, text=True,
).stdout.strip()
status = subprocess.run(
    ["git", "-C", str(PROJECT_DIR), "status", "--short"],
    check=True, capture_output=True, text=True,
).stdout.strip()
assert commit == EXPECTED_COMMIT and not status
assert token not in remote and "@" not in remote
print("exact clean commit:", commit, "remote contains token: False")

# %%
import hashlib
import json
import shutil
import socket
import sys
import uuid
from datetime import datetime, timezone

import pandas as pd
from PIL import Image
import torch

assert torch.cuda.is_available(), "T4/CUDA required"
DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/ddpm-derm-augmentation").resolve()
DRIVE_DATA_DIR = DRIVE_PROJECT_DIR / "data"
LOCAL_DATA_DIR = Path("/content/data")
OUTPUTS_DIR = DRIVE_PROJECT_DIR / "outputs"
SOURCE_EXPLORATORY_ROOT = OUTPUTS_DIR / "exploratory_balanced_ddpm" / "sqrt_balanced_seed0_v1"
CANDIDATE_DIR = SOURCE_EXPLORATORY_ROOT / "candidate_synthetic_df" / "epoch0100_seed0"
CANDIDATE_MANIFEST = CANDIDATE_DIR / "synthetic_df.csv"
RUNNER_OUTPUTS_DIR = Path("/content/drive/MyDrive") / RUN_STORAGE_DIRNAME
resolved_runner_outputs = RUNNER_OUTPUTS_DIR.resolve()
assert str(resolved_runner_outputs).startswith("/content/drive/MyDrive/"), (
    "runner output root must belong to the signed-in account, not a shared "
    f"shortcut: {resolved_runner_outputs}"
)

def ensure_runner_directory(path):
    path = Path(path)
    assert path.parent.is_dir(), f"runner output parent is missing: {path.parent}"
    if not path.is_dir():
        path.mkdir()
    marker = path / ".directory_ready"
    marker.write_text("ready\n", encoding="utf-8")
    assert marker.read_text(encoding="utf-8") == "ready\n"
    return path

def ensure_runner_tree(path):
    path = Path(path)
    current = ensure_runner_directory(RUNNER_OUTPUTS_DIR)
    for part in path.relative_to(RUNNER_OUTPUTS_DIR).parts:
        current = ensure_runner_directory(current / part)
    return current

RUNNER_DOWNSTREAM_ROOT = ensure_runner_tree(
    RUNNER_OUTPUTS_DIR / "exploratory_balanced_ddpm" / "sqrt_balanced_seed0_v1"
    / "downstream_classifier"
)
RUN_DIR = RUNNER_DOWNSTREAM_ROOT / RUN_VERSION
RUN_METADATA = RUN_DIR / "run_metadata.json"
RUNNING_MARKER = RUN_DIR / "_RUNNING.json"
COMPLETED_MARKER = RUN_DIR / "_COMPLETED.json"
FROZEN_RESULTS = OUTPUTS_DIR / "classifier_df585" / "results"
probe = RUNNER_OUTPUTS_DIR / f".write_probe_{uuid.uuid4().hex}.json"
child_probe = RUNNER_OUTPUTS_DIR / f".child_write_probe_{uuid.uuid4().hex}.txt"
try:
    probe.write_text(json.dumps({"write": "ok"}), encoding="utf-8")
    assert json.loads(probe.read_text(encoding="utf-8")) == {"write": "ok"}
    subprocess.run(
        [
            sys.executable, "-c",
            "from pathlib import Path; import sys; "
            "assert Path(sys.argv[1]).read_text(encoding='utf-8'); "
            "Path(sys.argv[2]).write_text('child-ok\\n', encoding='utf-8')",
            str(probe), str(child_probe),
        ],
        check=True,
    )
    assert child_probe.read_text(encoding="utf-8") == "child-ok\n"
finally:
    probe.unlink(missing_ok=True)
    child_probe.unlink(missing_ok=True)
print("shared project source:", DRIVE_PROJECT_DIR)
print("runner-owned output root:", RUNNER_OUTPUTS_DIR)

if LOCAL_DATA_DIR.exists():
    shutil.rmtree(LOCAL_DATA_DIR)
shutil.copytree(DRIVE_DATA_DIR, LOCAL_DATA_DIR)
os.environ["DDPM_DERM_DATA_DIR"] = str(LOCAL_DATA_DIR)
os.environ["DDPM_DERM_OUTPUTS_DIR"] = str(RUNNER_OUTPUTS_DIR)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

frames = {
    split: pd.read_csv(LOCAL_DATA_DIR / "manifests" / f"{split}.csv")
    for split in ("train", "val", "test")
}
assert {k: len(v) for k, v in frames.items()} == {"train": 6995, "val": 1510, "test": 1510}
assert int((frames["train"]["dx"] == "df").sum()) == 85
for other in ("val", "test"):
    assert not set(frames["train"]["lesion_id"]) & set(frames[other]["lesion_id"])
    assert not set(frames["train"]["image_id"]) & set(frames[other]["image_id"])
rows = pd.concat(frames.values(), ignore_index=True)
assert not [p for p in rows["image_path"] if not (LOCAL_DATA_DIR / p).is_file()]

ready = json.loads((CANDIDATE_DIR / "_READY.json").read_text(encoding="utf-8"))
candidate_meta = json.loads((CANDIDATE_DIR / "metadata.json").read_text(encoding="utf-8"))
candidate = pd.read_csv(CANDIDATE_MANIFEST)
candidate_images = sorted((CANDIDATE_DIR / "images").glob("*.png"))
assert len(candidate) == len(candidate_images) == ready["rows"] == 500
assert sha256(CANDIDATE_MANIFEST) == CANDIDATE_SHA256
assert candidate["dx"].eq("df").all() and candidate["source"].eq("synthetic").all()
for key, value in {
    "checkpoint": "run_seed0_epoch0100.pt", "epoch": 100,
    "weights": "ema_state_dict", "sampler_strategy": "sqrt_balanced",
    "n": 500, "seed": 0, "num_steps": 50, "eta": 0, "image_size": 64,
}.items():
    assert candidate_meta[key] == value, (key, candidate_meta[key], value)
for path in candidate_images:
    with Image.open(path) as image:
        assert image.mode == "RGB" and image.size == (64, 64)
        image.verify()
print("data/candidate preflight passed; candidate SHA256", CANDIDATE_SHA256)

# %% [markdown]
# ## 2. Fresh/resume gate and cross-account concurrency marker
# `last.pt` is atomically replaced on Drive after every completed epoch. The
# worst-case lost work is one unfinished epoch. Never run this `RUN_VERSION`
# from two accounts at the same time. A leftover marker requires an explicit
# human confirmation; the notebook never silently overwrites it.

# %%
FIXED_CONFIG = {
    "run_label": RUN_VERSION,
    "variant": "C4",
    "seeds": [0, 1, 2],
    "epochs": 20,
    "img_size": 128,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "df_target_count": 585,
    "pretrained": True,
    "source_split": "train",
    "train_count": 7495,
    "val_count": 1510,
    "test_count": 1510,
}
SOURCE_MANIFEST_SHA256 = sha256(LOCAL_DATA_DIR / "manifests" / "train.csv")
protected = [
    OUTPUTS_DIR / "classifier",
    OUTPUTS_DIR / "classifier_df585",
    OUTPUTS_DIR / "ddpm",
    OUTPUTS_DIR / "synthetic_df",
    OUTPUTS_DIR / "deploy",
]

def inventory(paths):
    result = {}
    for root in paths:
        if root.exists():
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                stat = path.stat()
                result[str(path.relative_to(OUTPUTS_DIR))] = [stat.st_size, stat.st_mtime_ns]
    return result

def write_json_atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)

expected_shared = {
    "run_version": RUN_VERSION,
    "git_commit": EXPECTED_COMMIT,
    "runner_output_root": str(RUNNER_OUTPUTS_DIR),
    "candidate_manifest_sha256": CANDIDATE_SHA256,
    "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
    "fixed_config": FIXED_CONFIG,
}
if RUN_MODE == "fresh":
    assert not RUN_DIR.exists(), f"fresh mode refuses existing run root: {RUN_DIR}"
    ensure_runner_tree(RUN_DIR)
    protected_before = inventory(protected)
    metadata = {
        **expected_shared,
        "status": "running",
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "end_utc": None,
        "output_path": str(RUN_DIR),
        "runtime": {
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "checkpoint_cadence": "last.pt atomically replaced after every completed epoch",
        "worst_case_lost_work": "one unfinished epoch",
        "protected_inventory_before": protected_before,
        "completed_seeds": [],
    }
    write_json_atomic(RUN_METADATA, metadata)
else:
    assert RUN_DIR.is_dir() and RUN_METADATA.is_file(), f"resume run missing: {RUN_DIR}"
    metadata = json.loads(RUN_METADATA.read_text(encoding="utf-8"))
    for key, value in expected_shared.items():
        assert metadata[key] == value, f"resume metadata mismatch for {key}"
    protected_before = metadata["protected_inventory_before"]

if RUNNING_MARKER.exists():
    old_marker = json.loads(RUNNING_MARKER.read_text(encoding="utf-8"))
    print("Existing concurrency marker:", json.dumps(old_marker, indent=2))
    confirmation = input(
        "Only if every previous session/account is stopped, type "
        "CLEAR STALE MARKER exactly: "
    )
    if confirmation != "CLEAR STALE MARKER":
        raise RuntimeError("concurrency marker retained; stop and resolve the other session")
    RUNNING_MARKER.unlink()

session_marker = {
    "session_id": str(uuid.uuid4()),
    "run_version": RUN_VERSION,
    "mode": RUN_MODE,
    "host": socket.gethostname(),
    "started_utc": datetime.now(timezone.utc).isoformat(),
}
with RUNNING_MARKER.open("x", encoding="utf-8") as handle:
    json.dump(session_marker, handle, indent=2)
print("concurrency marker created:", RUNNING_MARKER)
print("checkpoint cadence: every epoch; worst-case loss: one unfinished epoch")

# %% [markdown]
# ## 3. Sequential training with live output and heartbeat

# %%
import queue
import threading
import time

env = os.environ.copy()
env.update(PYTHONPATH=str(PROJECT_DIR / "src"), PYTHONUNBUFFERED="1")

def wait_for_files(paths, timeout_seconds=120):
    paths = tuple(paths)
    deadline = time.monotonic() + timeout_seconds
    missing = [path for path in paths if not path.is_file()]
    while missing and time.monotonic() < deadline:
        time.sleep(1)
        missing = [path for path in paths if not path.is_file()]
    assert not missing, (
        f"Drive artifacts not visible after {timeout_seconds}s: "
        + ", ".join(map(str, missing))
    )

def result_path(seed):
    return RUN_DIR / "results" / f"results_C4_seed{seed}.json"

def checkpoint_dir(seed):
    return RUN_DIR / "checkpoints" / f"C4_seed{seed}"

def validate_seed(seed):
    result_file = result_path(seed)
    best = checkpoint_dir(seed) / "best.pt"
    last = checkpoint_dir(seed) / "last.pt"
    wait_for_files((result_file, best, last))
    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert result["variant"] == "C4" and result["seed"] == seed
    assert len(result["history"]) == 20
    assert [h["epoch"] for h in result["history"]] == list(range(1, 21))
    assert result["data_counts"] == {"train": 7495, "val": 1510, "test": 1510}
    identity = result["run_identity"]
    assert identity["run_label"] == RUN_VERSION
    assert identity["variant"] == "C4" and identity["seed"] == seed
    assert identity["candidate_manifest_sha256"] == CANDIDATE_SHA256
    assert identity["source_split"] == "train"
    assert identity["source_manifest_sha256"] == SOURCE_MANIFEST_SHA256
    assert identity["git_commit"] == EXPECTED_COMMIT
    assert identity["fixed_config"] == {
        "epochs": 20, "img_size": 128, "batch_size": 32,
        "learning_rate": 3e-4, "weight_decay": 1e-4,
        "df_target_count": 585, "pretrained": True, "limit": None,
    }
    for path, expected_epoch in ((best, None), (last, 20)):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["run_identity"] == identity
        checkpoint_epoch = checkpoint["epoch"]
        checkpoint_history = checkpoint["history"]
        assert 1 <= checkpoint_epoch <= 20
        assert len(checkpoint_history) == checkpoint_epoch
        assert [h["epoch"] for h in checkpoint_history] == list(
            range(1, checkpoint_epoch + 1)
        )
        if expected_epoch is not None:
            assert checkpoint_epoch == expected_epoch
    return result

def stream_with_heartbeat(command, heartbeat_seconds=120):
    process = subprocess.Popen(
        command, cwd=PROJECT_DIR / "src", env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    output = queue.Queue()
    def reader():
        for line in process.stdout:
            output.put(line)
        output.put(None)
    threading.Thread(target=reader, daemon=True).start()
    started = time.monotonic()
    while True:
        try:
            line = output.get(timeout=heartbeat_seconds)
        except queue.Empty:
            print(f"[heartbeat] training still active; elapsed={(time.monotonic()-started)/60:.1f} min", flush=True)
            continue
        if line is None:
            break
        print(line, end="", flush=True)
    code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)

for seed in FIXED_CONFIG["seeds"]:
    if result_path(seed).exists():
        validate_seed(seed)
        print(f"[seed {seed}] completed result/checkpoints verified; skip")
        continue
    command = [
        sys.executable, "-u", "-m", "ddpm_derm.train_classifier",
        "--variant", "C4", "--seed", str(seed), "--epochs", "20",
        "--img-size", "128", "--batch-size", "32", "--lr", "3e-4",
        "--weight-decay", "1e-4", "--df-target-count", "585",
        "--generated-manifest", str(CANDIDATE_MANIFEST),
        "--num-workers", "2", "--run-label", RUN_VERSION,
        "--output-dir", str(RUN_DIR),
    ]
    if RUN_MODE == "resume":
        command.append("--resume")
    print(f"===== formal C4-sqrt-balanced seed {seed} =====", flush=True)
    stream_with_heartbeat(command)
    validate_seed(seed)
    metadata = json.loads(RUN_METADATA.read_text(encoding="utf-8"))
    metadata["completed_seeds"] = sorted(set(metadata["completed_seeds"] + [seed]))
    write_json_atomic(RUN_METADATA, metadata)
    print(f"[seed {seed}] result + best.pt + last.pt verified")

# %% [markdown]
# ## 4. Post-run verification, aggregation, and frozen matched-585 comparison

# %%
import numpy as np

new_runs = [validate_seed(seed) for seed in FIXED_CONFIG["seeds"]]

def metric_block(runs):
    keys = {"df_f1": "target_f1", "macro_f1": "macro_f1", "df_recall": "target_recall"}
    block = {"seeds": [r["seed"] for r in runs], "raw": {}, "summary": {}}
    for label, key in keys.items():
        values = [float(r["test_metrics"][key]) for r in runs]
        block["raw"][label] = values
        block["summary"][label] = {
            "mean": float(np.mean(values)),
            "population_std": float(np.std(values, ddof=0)),
        }
    return block

frozen = {}
for variant in ("C1", "C4"):
    runs = []
    for seed in (0, 1, 2):
        path = FROZEN_RESULTS / f"results_{variant}_seed{seed}.json"
        assert path.is_file(), f"missing frozen result: {path}"
        run = json.loads(path.read_text(encoding="utf-8"))
        cfg = run["config"]
        assert run["variant"] == variant and run["seed"] == seed
        assert len(run["history"]) == 20
        assert cfg["epochs"] == 20 and cfg["img_size"] == 128
        assert cfg["batch_size"] == 32 and cfg["lr"] == 3e-4
        assert cfg["weight_decay"] == 1e-4 and cfg["df_target_count"] == 585
        assert cfg["no_pretrained"] is False and cfg["limit"] is None
        if variant == "C1":
            assert cfg["generated_manifest"] is None
        else:
            normalized = cfg["generated_manifest"].replace("\\", "/")
            assert normalized.endswith("/outputs/synthetic_df/epoch0100_seed0/synthetic_df.csv")
        runs.append(run)
    frozen[variant] = runs

blocks = {
    "c4_sqrt_balanced_585": metric_block(new_runs),
    "natural_c4_585": metric_block(frozen["C4"]),
    "c1_585": metric_block(frozen["C1"]),
}
new_mean = blocks["c4_sqrt_balanced_585"]["summary"]["df_f1"]["mean"]
natural_mean = blocks["natural_c4_585"]["summary"]["df_f1"]["mean"]
c1_mean = blocks["c1_585"]["summary"]["df_f1"]["mean"]
comparison = {
    "comparison_type": "descriptive_only",
    "fixed_split": True,
    "statistical_significance_test": None,
    "conditions": blocks,
    "mean_difference_test_df_f1": {
        "c4_sqrt_balanced_585_minus_natural_c4_585": new_mean - natural_mean,
        "c4_sqrt_balanced_585_minus_c1_585": new_mean - c1_mean,
    },
    "limitations": [
        "fixed split and three seeds only",
        "df test support is 16",
        "candidate nearest-neighbour QC does not exclude memorisation",
        "no statistically significant, validated, clinical, or medical-effectiveness claim",
    ],
}
write_json_atomic(RUN_DIR / "comparison_summary.json", comparison)

aggregate = subprocess.run(
    [sys.executable, "scripts/aggregate_results.py", "--results-dir", str(RUN_DIR / "results")],
    cwd=PROJECT_DIR, env=env, check=True, capture_output=True, text=True,
).stdout
(RUN_DIR / "aggregate.txt").write_text(aggregate, encoding="utf-8")
print(aggregate)
print(json.dumps(comparison["mean_difference_test_df_f1"], indent=2))

# %% [markdown]
# ## 5. Final artifact inventory and completion marker

# %%
checkpoint_inventory = []
for seed in FIXED_CONFIG["seeds"]:
    for name in ("best.pt", "last.pt"):
        path = checkpoint_dir(seed) / name
        checkpoint_inventory.append({
            "seed": seed,
            "name": name,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

assert inventory(protected) == protected_before, "a frozen/formal output changed"
metadata = json.loads(RUN_METADATA.read_text(encoding="utf-8"))
metadata.update(
    status="completed",
    end_utc=datetime.now(timezone.utc).isoformat(),
    completed_seeds=[0, 1, 2],
    checkpoint_inventory=checkpoint_inventory,
    comparison_summary=str(RUN_DIR / "comparison_summary.json"),
    aggregate_output=str(RUN_DIR / "aggregate.txt"),
)
write_json_atomic(RUN_METADATA, metadata)
training_record = {
    "status": "completed",
    "run_version": RUN_VERSION,
    "git_commit": EXPECTED_COMMIT,
    "candidate_manifest_sha256": CANDIDATE_SHA256,
    "seeds": [0, 1, 2],
    "formal_outputs_untouched": True,
    "completed_utc": metadata["end_utc"],
}
write_json_atomic(RUN_DIR / "training_record.json", training_record)
write_json_atomic(COMPLETED_MARKER, {**session_marker, **training_record})
RUNNING_MARKER.unlink()
print(json.dumps(training_record, indent=2))
print("FORMAL TRAINING + POST-RUN VERIFICATION COMPLETED")
