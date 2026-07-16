# %% [markdown]
# # C4-sqrt-balanced@585 — validation only
# Safe to **Run all** in a fresh Colab T4 runtime. This notebook validates the
# exact code, fixed data, candidate, checkpoint guards, and a tiny one-epoch
# classifier run. It cannot start the formal 3-seed × 20-epoch experiment.

# %%
EXPECTED_COMMIT = "f575380efb042937c9f124c217cee9b1da981fee"
REPO_URL = "https://github.com/sfczaa/ddpm-derm-augmentation.git"
BRANCH = "balanced-ddpm-exploration"
CANDIDATE_SHA256 = "9ef9b44e404f74aab8211f4e7d123da3258ba8ba4e3004a4147d1761ed343b34"

# %% [markdown]
# ## 1. GPU, Drive, and an exact private-Git checkout

# %%
import os
import subprocess
from pathlib import Path

from google.colab import drive, userdata

assert len(EXPECTED_COMMIT) == 40 and EXPECTED_COMMIT != "REPLACE_AFTER_PUSH", (
    "Notebook is not pinned yet. Stop: Phase 2 commit/push must happen first."
)
drive.mount("/content/drive")
subprocess.run(["nvidia-smi"], check=True)

PROJECT_DIR = Path("/content/ddpm-classifier-validate-code")
assert not PROJECT_DIR.exists(), f"fresh runtime required; already exists: {PROJECT_DIR}"
token = userdata.get("GH_TOKEN")
assert token and len(token) > 20, "Colab Secret GH_TOKEN is required"
askpass = Path("/content/git-askpass-classifier-validate.sh")
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
        check=True,
        env=clone_env,
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
print("exact clean commit:", commit)
print("remote contains token: False")

# %% [markdown]
# ## 2. Fixed data, candidate, and protected-output inventory

# %%
import hashlib
import json
import shutil
from datetime import datetime, timezone

import pandas as pd
from PIL import Image

DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/ddpm-derm-augmentation").resolve()
DRIVE_DATA_DIR = DRIVE_PROJECT_DIR / "data"
LOCAL_DATA_DIR = Path("/content/data")
OUTPUTS_DIR = DRIVE_PROJECT_DIR / "outputs"
CANDIDATE_DIR = (
    OUTPUTS_DIR / "exploratory_balanced_ddpm" / "sqrt_balanced_seed0_v1"
    / "candidate_synthetic_df" / "epoch0100_seed0"
)
CANDIDATE_MANIFEST = CANDIDATE_DIR / "synthetic_df.csv"
FORMAL_RUN_DIR = (
    OUTPUTS_DIR / "exploratory_balanced_ddpm" / "sqrt_balanced_seed0_v1"
    / "downstream_classifier" / "c4_sqrt_balanced_v1"
)
VALIDATION_ROOT = (
    OUTPUTS_DIR / "exploratory_balanced_ddpm" / "sqrt_balanced_seed0_v1"
    / "downstream_classifier" / "validation_runs"
)
VALIDATION_DIR = VALIDATION_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
assert not FORMAL_RUN_DIR.exists(), f"formal run already exists; stop: {FORMAL_RUN_DIR}"
assert not VALIDATION_DIR.exists()

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

protected_before = inventory(protected)

if LOCAL_DATA_DIR.exists():
    shutil.rmtree(LOCAL_DATA_DIR)
shutil.copytree(DRIVE_DATA_DIR, LOCAL_DATA_DIR)
os.environ["DDPM_DERM_DATA_DIR"] = str(LOCAL_DATA_DIR)
os.environ["DDPM_DERM_OUTPUTS_DIR"] = str(OUTPUTS_DIR)

frames = {
    split: pd.read_csv(LOCAL_DATA_DIR / "manifests" / f"{split}.csv")
    for split in ("train", "val", "test")
}
assert {k: len(v) for k, v in frames.items()} == {"train": 6995, "val": 1510, "test": 1510}
assert int((frames["train"]["dx"] == "df").sum()) == 85
for other in ("val", "test"):
    assert not set(frames["train"]["lesion_id"]) & set(frames[other]["lesion_id"])
    assert not set(frames["train"]["image_id"]) & set(frames[other]["image_id"])
all_rows = pd.concat(frames.values(), ignore_index=True)
missing = [p for p in all_rows["image_path"] if not (LOCAL_DATA_DIR / p).is_file()]
assert not missing
print("fixed split/data OK: 6995/1510/1510, train df=85, leakage=0, missing=0")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

ready = json.loads((CANDIDATE_DIR / "_READY.json").read_text(encoding="utf-8"))
metadata = json.loads((CANDIDATE_DIR / "metadata.json").read_text(encoding="utf-8"))
candidate = pd.read_csv(CANDIDATE_MANIFEST)
images = sorted((CANDIDATE_DIR / "images").glob("*.png"))
assert len(candidate) == len(images) == ready["rows"] == 500
assert candidate["image_path"].nunique() == candidate["image_id"].nunique() == 500
assert candidate["dx"].eq("df").all() and candidate["source"].eq("synthetic").all()
assert sha256(CANDIDATE_MANIFEST) == CANDIDATE_SHA256
expected_meta = {
    "checkpoint": "run_seed0_epoch0100.pt", "epoch": 100,
    "weights": "ema_state_dict", "sampler_strategy": "sqrt_balanced",
    "class_name": "df", "class_idx": 3, "n": 500, "seed": 0,
    "num_steps": 50, "eta": 0, "image_size": 64,
}
for key, value in expected_meta.items():
    assert metadata[key] == value, (key, metadata[key], value)
for path in images:
    with Image.open(path) as image:
        assert image.mode == "RGB" and image.size == (64, 64)
        image.verify()
print("candidate OK: 500 RGB 64x64 PNG; SHA256", CANDIDATE_SHA256)

# %% [markdown]
# ## 3. Dependencies and repository checks

# %%
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas", "pillow"], check=True)
import torch
assert torch.cuda.is_available(), "T4/CUDA required"
env = os.environ.copy()
env["PYTHONPATH"] = str(PROJECT_DIR / "src")
subprocess.run([sys.executable, "scripts/smoke_test.py"], cwd=PROJECT_DIR, env=env, check=True)
subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_classifier_run.py", "-v"],
    cwd=PROJECT_DIR, env=env, check=True,
)
print("repository smoke + torch-free classifier guards passed; GPU:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## 4. Tiny C4-sqrt run, strict resume, mismatch rejection, Drive-only restore

# %%
import time

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

def run_stream(command, expect_success=True):
    process = subprocess.Popen(
        command, cwd=PROJECT_DIR / "src", env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lines = []
    for line in process.stdout:
        lines.append(line)
        print(line, end="", flush=True)
    code = process.wait()
    if expect_success and code != 0:
        raise subprocess.CalledProcessError(code, command)
    if not expect_success and code == 0:
        raise AssertionError("command unexpectedly succeeded")
    return code, "".join(lines)

command = [
    sys.executable, "-u", "-m", "ddpm_derm.train_classifier",
    "--variant", "C4", "--seed", "0", "--epochs", "1",
    "--img-size", "128", "--batch-size", "32", "--lr", "3e-4",
    "--weight-decay", "1e-4", "--df-target-count", "585",
    "--generated-manifest", str(CANDIDATE_MANIFEST),
    "--limit", "256", "--num-workers", "2",
    "--run-label", "c4_sqrt_balanced_validation_smoke",
    "--output-dir", str(VALIDATION_DIR),
]
_, fresh_output = run_stream(command)
assert "checkpoint_saved=last.pt" in fresh_output

last_path = VALIDATION_DIR / "checkpoints" / "C4_seed0" / "last.pt"
best_path = VALIDATION_DIR / "checkpoints" / "C4_seed0" / "best.pt"
result_path = VALIDATION_DIR / "results" / "results_C4_seed0.json"
wait_for_files((last_path, best_path, result_path))
checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
identity = checkpoint["run_identity"]
assert checkpoint["epoch"] == 1 and len(checkpoint["history"]) == 1
assert identity["run_label"] == "c4_sqrt_balanced_validation_smoke"
assert identity["candidate_manifest_sha256"] == CANDIDATE_SHA256
assert identity["source_split"] == "train" and identity["git_commit"] == EXPECTED_COMMIT

_, resume_output = run_stream(command + ["--resume"])
assert "[resume] found last.pt" in resume_output and "[skip]" in resume_output

before_hash = sha256(last_path)
before_mtime = last_path.stat().st_mtime_ns
mismatch = list(command + ["--resume"])
mismatch[mismatch.index("--batch-size") + 1] = "16"
_, mismatch_output = run_stream(mismatch, expect_success=False)
assert "resume identity mismatch" in mismatch_output
assert sha256(last_path) == before_hash and last_path.stat().st_mtime_ns == before_mtime

# The only classifier checkpoint is already on Drive. A new child process has
# no runtime-local checkpoint to depend on and restores/skips from that Drive file.
assert str(last_path).startswith("/content/drive/")
_, restore_output = run_stream(command + ["--resume"])
assert "[resume] found last.pt" in restore_output
print("same-config resume, mismatch guard, unchanged checkpoint, and Drive-only restore: OK")

# %% [markdown]
# ## 5. Stop before formal training

# %%
assert inventory(protected) == protected_before, "a frozen/formal output changed"
assert not FORMAL_RUN_DIR.exists(), "formal training path was created"
record = {
    "status": "passed",
    "formal_training_started": False,
    "git_commit": EXPECTED_COMMIT,
    "candidate_manifest_sha256": CANDIDATE_SHA256,
    "validation_output": str(VALIDATION_DIR),
    "completed_utc": datetime.now(timezone.utc).isoformat(),
}
(VALIDATION_DIR / "validation_record.json").write_text(
    json.dumps(record, indent=2), encoding="utf-8"
)
print(json.dumps(record, indent=2))
print("VALIDATION PASSED — FORMAL TRAINING NOT STARTED")
