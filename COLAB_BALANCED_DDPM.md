# Exploratory sqrt-balanced DDPM — Colab runbook

This is a post-Stage-4 exploratory run. It does not replace the frozen natural
DDPM or the formal epoch-100 synthetic dataset. The strategy is deliberately
named sqrt_balanced: each train row in class c has weight 1 / sqrt(n_c).
Sampling uses replacement and draws len(train) indices per epoch.

The trainer requires a clean Git checkout so run_metadata.json records the
code commit that actually ran.

## 1. Sync code and prepare the runtime

~~~python
from google.colab import drive
drive.mount("/content/drive")
~~~

~~~python
import os
import subprocess
from pathlib import Path

PROJECT_DIR = Path("/content/drive/MyDrive/ddpm-derm-augmentation")
subprocess.run(
    ["git", "-C", str(PROJECT_DIR), "fetch", "origin"],
    check=True,
)
subprocess.run(
    ["git", "-C", str(PROJECT_DIR), "checkout", "main"],
    check=True,
)
subprocess.run(
    ["git", "-C", str(PROJECT_DIR), "pull", "--ff-only", "origin", "main"],
    check=True,
)
status = subprocess.run(
    ["git", "-C", str(PROJECT_DIR), "status", "--short"],
    check=True, capture_output=True, text=True,
).stdout
assert not status.strip(), f"checkout is not clean:\n{status}"
commit = subprocess.run(
    ["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"],
    check=True, capture_output=True, text=True,
).stdout.strip()
print("commit:", commit)

DRIVE_DATA_DIR = PROJECT_DIR / "data"
LOCAL_DATA_DIR = Path("/content/data")
OUTPUTS_DIR = PROJECT_DIR / "outputs"
os.environ["DDPM_DERM_DATA_DIR"] = str(LOCAL_DATA_DIR)
os.environ["DDPM_DERM_OUTPUTS_DIR"] = str(OUTPUTS_DIR)
~~~

~~~python
# Copy the fixed dataset to local disk once per runtime.
if not (LOCAL_DATA_DIR / "manifests" / "class_to_idx.json").is_file():
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cp", "-a", str(DRIVE_DATA_DIR) + "/.", str(LOCAL_DATA_DIR) + "/"],
        check=True,
    )

import pandas as pd
rows = pd.concat(
    [
        pd.read_csv(LOCAL_DATA_DIR / "manifests" / f"{split}.csv")
        for split in ("train", "val", "test")
    ],
    ignore_index=True,
)
missing = [
    p for p in rows["image_path"]
    if not (LOCAL_DATA_DIR / p).is_file()
]
assert len(rows) == 10015
assert rows["image_path"].nunique() == 10015
assert not missing
print("local fixed manifests/images OK")
~~~

~~~python
!pip install -q diffusers
!cd "{PROJECT_DIR}" && python scripts/smoke_test.py
!cd "{PROJECT_DIR}" && python -m unittest tests.test_ddpm_sampler -v
!cd "{PROJECT_DIR}" && python scripts/smoke_ddpm.py
~~~

The last command must print a seed-0 sampler label histogram and pass the
replacement, num_samples == 6995, fixed-seed index sequence, and tiny DDPM
checks.

## 2. Tiny one-epoch smoke and resume guard

Use a smoke-only Drive version and a separate local mutable checkpoint:

~~~python
SMOKE_VERSION = "sqrt_balanced_seed0_smoke_v1"
SMOKE_RUN_DIR = OUTPUTS_DIR / "exploratory_balanced_ddpm" / SMOKE_VERSION
SMOKE_SNAPSHOTS = SMOKE_RUN_DIR / "checkpoints"
SMOKE_PREVIEWS = SMOKE_RUN_DIR / "previews"
SMOKE_METADATA = SMOKE_RUN_DIR / "run_metadata.json"
SMOKE_LOCAL_CKPT = Path("/content/exploratory_balanced_ddpm_smoke_ckpt")

assert not SMOKE_RUN_DIR.exists(), (
    f"{SMOKE_RUN_DIR} already exists; choose a new smoke version"
)
assert not SMOKE_LOCAL_CKPT.exists(), (
    f"{SMOKE_LOCAL_CKPT} already exists; choose a fresh local smoke path"
)
SMOKE_SNAPSHOTS.mkdir(parents=True)
SMOKE_PREVIEWS.mkdir()
assert SMOKE_SNAPSHOTS.is_dir() and SMOKE_PREVIEWS.is_dir()
~~~

~~~python
!cd "{PROJECT_DIR}/src" && python -m ddpm_derm.train_ddpm \
  --sampler-strategy sqrt_balanced \
  --seed 0 --epochs 1 --img-size 32 --batch-size 4 --lr 1e-4 \
  --ema-decay 0.999 --timesteps 50 --beta-start 1e-4 --beta-end 2e-2 \
  --limit 32 --tiny --num-workers 0 --preview-every 0 \
  --output-dir "{SMOKE_LOCAL_CKPT}" \
  --snapshot-dir "{SMOKE_SNAPSHOTS}" --snapshot-every 1 \
  --preview-dir "{SMOKE_PREVIEWS}" \
  --run-metadata-path "{SMOKE_METADATA}"
~~~

~~~python
import json
from ddpm_derm.checkpoint import load_checkpoint

smoke_ckpt_path = SMOKE_LOCAL_CKPT / "run_seed0_last.pt"
smoke_ckpt = load_checkpoint(
    smoke_ckpt_path, map_location="cpu"
)
assert smoke_ckpt["sampler_strategy"] == "sqrt_balanced"
assert smoke_ckpt["sampler_generator_state"] is not None
assert smoke_ckpt["run_metadata"]["source_split"] == "train"
assert smoke_ckpt["run_metadata"]["git_commit"] == commit
meta = json.loads(SMOKE_METADATA.read_text())
assert meta["sampler_strategy"] == "sqrt_balanced"
assert meta["class_counts"] == smoke_ckpt["run_metadata"]["class_counts"]
print(json.dumps(meta, indent=2))
~~~

Re-run the same target with --resume. It should restore global RNG and the
sqrt-sampler generator, then skip because epoch 1 is already complete:

~~~python
!cd "{PROJECT_DIR}/src" && python -m ddpm_derm.train_ddpm \
  --sampler-strategy sqrt_balanced \
  --seed 0 --epochs 1 --img-size 32 --batch-size 4 --lr 1e-4 \
  --ema-decay 0.999 --timesteps 50 --beta-start 1e-4 --beta-end 2e-2 \
  --limit 32 --tiny --num-workers 0 --preview-every 0 --resume \
  --output-dir "{SMOKE_LOCAL_CKPT}" \
  --snapshot-dir "{SMOKE_SNAPSHOTS}" --snapshot-every 1 \
  --preview-dir "{SMOKE_PREVIEWS}" \
  --run-metadata-path "{SMOKE_METADATA}"
~~~

This is not a deterministic training resume. The test confirms
strategy metadata and sampler-generator restoration; it does not prove a
bit-identical GPU loss trajectory.

## 3. Formal exploratory seed-0 run

Choose a fresh immutable version. Do not reuse the smoke version:

~~~python
RUN_VERSION = "sqrt_balanced_seed0_v1"
RUN_DIR = OUTPUTS_DIR / "exploratory_balanced_ddpm" / RUN_VERSION
SNAPSHOT_DIR = RUN_DIR / "checkpoints"
PREVIEW_DIR = RUN_DIR / "previews"
RUN_METADATA = RUN_DIR / "run_metadata.json"
LOCAL_CKPT_DIR = Path("/content/exploratory_balanced_ddpm_seed0_ckpt")

assert not RUN_DIR.exists(), f"{RUN_DIR} already exists; choose a new version"
assert not LOCAL_CKPT_DIR.exists(), (
    f"{LOCAL_CKPT_DIR} already exists; use it only for an intentional resume"
)
SNAPSHOT_DIR.mkdir(parents=True)
PREVIEW_DIR.mkdir()
assert SNAPSHOT_DIR.is_dir() and PREVIEW_DIR.is_dir()
~~~

Fresh start:

~~~python
!cd "{PROJECT_DIR}/src" && python -m ddpm_derm.train_ddpm \
  --sampler-strategy sqrt_balanced \
  --seed 0 --epochs 100 --img-size 64 --batch-size 64 --lr 1e-4 \
  --ema-decay 0.999 --timesteps 1000 --beta-start 1e-4 --beta-end 2e-2 \
  --num-workers 2 --preview-every 5 --preview-steps 50 \
  --output-dir "{LOCAL_CKPT_DIR}" \
  --snapshot-dir "{SNAPSHOT_DIR}" --snapshot-every 10 \
  --preview-dir "{PREVIEW_DIR}" \
  --run-metadata-path "{RUN_METADATA}"
~~~

After a disconnect, rerun Section 1, then reconstruct and verify the existing
paths without running the fresh-start assertions:

~~~python
RUN_VERSION = "sqrt_balanced_seed0_v1"
RUN_DIR = OUTPUTS_DIR / "exploratory_balanced_ddpm" / RUN_VERSION
SNAPSHOT_DIR = RUN_DIR / "checkpoints"
PREVIEW_DIR = RUN_DIR / "previews"
RUN_METADATA = RUN_DIR / "run_metadata.json"
LOCAL_CKPT_DIR = Path("/content/exploratory_balanced_ddpm_seed0_ckpt")

assert RUN_DIR.is_dir()
assert SNAPSHOT_DIR.is_dir()
assert PREVIEW_DIR.is_dir()
assert RUN_METADATA.is_file()
~~~

Then run the same fixed config with --resume. If the local mutable checkpoint
is absent, the trainer restores the latest immutable snapshot:

~~~python
!cd "{PROJECT_DIR}/src" && python -m ddpm_derm.train_ddpm \
  --sampler-strategy sqrt_balanced \
  --seed 0 --epochs 100 --img-size 64 --batch-size 64 --lr 1e-4 \
  --ema-decay 0.999 --timesteps 1000 --beta-start 1e-4 --beta-end 2e-2 \
  --num-workers 2 --preview-every 5 --preview-steps 50 --resume \
  --output-dir "{LOCAL_CKPT_DIR}" \
  --snapshot-dir "{SNAPSHOT_DIR}" --snapshot-every 10 \
  --preview-dir "{PREVIEW_DIR}" \
  --run-metadata-path "{RUN_METADATA}"
~~~

Never remove the run directory or use a fresh version when the intent is to
resume.

## 4. Generate a versioned candidate only after epoch 100

~~~python
EPOCH100 = SNAPSHOT_DIR / "run_seed0_epoch0100.pt"
assert EPOCH100.is_file(), f"missing completed snapshot: {EPOCH100}"

CANDIDATE_STAGING = Path(
    "/content/synthetic_df_sqrt_balanced_seed0_v1_epoch0100"
)
CANDIDATE_FINAL = RUN_DIR / "candidate_synthetic_df" / "epoch0100_seed0"
assert not CANDIDATE_STAGING.exists()
assert not CANDIDATE_FINAL.exists()
~~~

~~~python
!cd "{PROJECT_DIR}/src" && python -m ddpm_derm.sample_ddpm \
  --ckpt "{EPOCH100}" --require-epoch 100 --require-ema \
  --require-sampler-strategy sqrt_balanced \
  --n 500 --num-steps 50 --eta 0 --seed 0 \
  --out-dir "{CANDIDATE_STAGING}"
~~~

~~~python
!cd "{PROJECT_DIR}/src" && python -m ddpm_derm.publish_synthetic \
  --src "{CANDIDATE_STAGING}" --dest "{CANDIDATE_FINAL}" \
  --expect-n 500 --expect-epoch 100 --expect-seed 0 --expect-steps 50
~~~

The runbook ends here. Review the sampler histogram, run metadata, epoch-100
snapshot, previews, nearest-neighbour montage, and candidate path before any
further step. C0, C1, and C4 are not rerun here; a downstream comparison is a
separate decision.
