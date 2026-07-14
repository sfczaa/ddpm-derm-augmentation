# ddpm-derm-augmentation

Portfolio project: **can a DDPM actually help a downstream classifier** on the
imbalanced HAM10000 skin-lesion dataset? Target minority class is **df**
(dermatofibroma, only 115 images). End goal is a deployable demo
(FastAPI + Docker, with versioned assets on Hugging Face Hub).

> Status: **The formal matched-585 experiment is complete and the Stage 4
> deployment MVP is implemented.** The selected deploy candidate is C1@585
> seed 2, chosen by the highest validation df F1 among C1 seeds. Asset and API
> safety paths are locally verified. Real deployment-only checkpoint inference
> and the FastAPI runtime were exercised in Colab. The Render Free remote Docker
> build and public demo endpoints were validated on 2026-07-14; a local Docker
> build was not run because Docker CLI is unavailable on the local machine.

## What exists now

```
ddpm-derm-augmentation/
  src/ddpm_derm/
    config.py            # paths from project layout / env vars (no hard-coded personal paths)
    manifests.py         # read fixed split, build C0/C1 frames        [pure pandas]
    metrics.py           # df F1, per-class recall, macro-F1, CM        [pure numpy]
    dataset.py           # torch Dataset + transforms + dataloaders     [needs torch]
    model.py             # ResNet-18 (ImageNet) 7-way head              [needs torch]
    train_classifier.py  # CLI: train one (variant, seed), checkpoint + JSON [needs torch]
  scripts/
    smoke_test.py        # torch-free verification of data + metrics
    aggregate_results.py # mean +/- std across seeds
    export_deploy_checkpoint.py
    package_gallery_for_deploy.py
  app/                   # FastAPI service + browser UI
  deploy/                # immutable deploy metadata + class mapping
  tests/test_deploy.py   # torch-free asset/input/API tests
  Dockerfile
  Dockerfile.render
  render.yaml
  requirements-deploy.txt
  notebooks/
    colab_classifier_baseline.py   # `# %%` cell script for Colab
  requirements.txt
```

The layers are split by dependency weight on purpose: `config` / `manifests` /
`metrics` need only pandas+numpy+pillow, so the whole data path can be verified
locally before spending GPU time. `dataset` / `model` / `train_classifier` need
torch and are meant to run on Colab.

## Data

Not committed (`.gitignore` excludes it). The ISIC 2018 distribution identifies
HAM10000 as CC BY-NC 4.0, so this project and public demo are non-commercial and
must retain attribution, a license link, and an indication of modifications.
Credit: HAM10000 Dataset © ViDIR Group, Department of Dermatology, Medical
University of Vienna; Tschandl, Rosendahl & Kittler, *Scientific Data* 5,
180161 (2018), https://doi.org/10.1038/sdata.2018.161.
The dataset lives **inside the project** at `data/` (10k images + `manifests/`),
so the whole project is a single self-contained upload. The code finds a data
dir containing `manifests/class_to_idx.json` in this order
(`src/ddpm_derm/config.py`):

1. `$DDPM_DERM_DATA_DIR`
2. `<project>/data`     ← current layout (data is inside the project)
3. `<project>/../data`  ← fallback if data is kept as a sibling

`image_path` in each manifest is relative to that data dir. The
`lesion_id`-grouped train/val/test split is **fixed** — do not re-split.

## Run the smoke test (local, no GPU, no torch)

```bash
pip install pandas numpy pillow
python scripts/smoke_test.py
```

Verifies: data dir resolves; per-class counts match `split_summary.csv`; images
open; **no lesion_id/image_id leakage between train and val/test**; C0/C1 frame
construction; metric correctness. Exits non-zero on any failure.

## Train baselines (Colab T4)

The Colab entry point is `notebooks/colab_classifier_baseline.ipynb`.
Its paths cell configures the project, prepared HAM10000 data, and output
directories. Training commands run locally from `src/`:

```bash
# quick 1-epoch subset smoke of the training loop (needs torch)
python -m ddpm_derm.train_classifier --variant C0 --seed 0 --epochs 1 --limit 200

# full baselines
python -m ddpm_derm.train_classifier --variant C0 --seed 0 --epochs 20
python -m ddpm_derm.train_classifier --variant C1 --seed 0 --epochs 20 --df-target-count 585

python scripts/aggregate_results.py   # mean +/- std across seeds
```

Model selection uses **val df F1**; reported numbers are on the **test** split.
Run 3–5 seeds per variant and report mean ± std. No k-fold, no significance
tests (df is tiny → results are **suggestive**, and that is stated as such).

### Checkpoints & Colab-disconnect resume

Each run writes to `outputs/classifier/checkpoints/<variant>_seed<seed>/`:
- `best.pt` — best-on-val-df-F1 weights (use this for eval / deployment)
- `last.pt` — latest epoch + optimizer state, rewritten every epoch (atomic write)

If a Colab session drops, re-run the **same** command with `--resume` and it
continues from `last.pt`. Results JSON lands in `outputs/classifier/results/`.
Point `DDPM_DERM_OUTPUTS_DIR` at Google Drive so checkpoints survive disconnects.
See `outputs/README.md` for the output layout.

## Experiment plan (from the project brief)

| Variant | Treatment | Purpose |
|---|---|---|
| C0 | raw imbalanced train | baseline |
| C1 | duplicate real train df up to `df_target_count` | rule out "just more df exposure" |
| C4 | add DDPM-synthetic df | main result |

**Fairness knob:** C1's `--df-target-count` must equal C4's total df count
(real + synthetic) so the two only differ in *how* the extra df is produced.
The agreed composition is **585 = 85 real train df + 500 generated**, so both
C1 and C4 run with `--df-target-count 585`.

**C4 status:** the formal epoch-100 synthetic dataset and matched-585 C1/C4
runs are complete. Across three seeds, C1@585 test df F1 was 0.660 +/- 0.042
and C4@585 was 0.612 +/- 0.042. This does not support a downstream benefit
from the current synthetic data; the fixed-split, small-df result is
suggestive only. C4 still reads an explicit portable generated manifest.

## Stage 4 deployment MVP

The service provides `/health`, `/api/predict`, `/api/gallery`, OpenAPI docs,
and a browser UI. Uploads are held in memory only. Extension, MIME type,
content format, dimensions, and a 5 MB size limit are enforced before
inference. Startup fails on missing or mismatched checkpoint hash, class map,
gallery `_READY.json`, metadata, manifest, or image files.

Default local assets:

```text
DDPM_DERM_MODEL_PATH=outputs/classifier_df585/checkpoints/C1_seed2/best.pt
DDPM_DERM_MODEL_MANIFEST=deploy/model_manifest.json
DDPM_DERM_CLASS_MAP_PATH=deploy/class_to_idx.json
DDPM_DERM_GALLERY_DIR=outputs/synthetic_df/epoch0100_seed0
```

The inference transform is the training evaluation transform: RGB, resize to
128x128, tensor conversion, and ImageNet normalization. The model is ResNet-18
with the original seven-class mapping. The gallery is the first 24 rows of the
published 500-row manifest, not manually selected, and no DDPM runs per request.

Local API tests (do not load torch):

```powershell
$env:PYTHONPATH="src;."
python -m unittest discover -s tests -p "test_*.py" -v
```

Docker uses read-only model and gallery mounts; the checkpoint is never copied
into the image or committed to Git:

```powershell
docker build -t ddpm-derm-demo .
docker run --rm -p 7860:7860 `
  --mount type=bind,source="${PWD}\outputs\classifier_df585\checkpoints\C1_seed2\best.pt",target=/models/best.pt,readonly `
  --mount type=bind,source="${PWD}\outputs\synthetic_df",target=/gallery,readonly `
  ddpm-derm-demo
```

Then check `http://localhost:7860/health`, `/docs`, and the upload UI. This
repository has not yet been Docker-built locally.
See `deploy/README_SPACE.md` for the Hugging Face Spaces handoff.

Hugging Face currently requires a paid plan for Docker Spaces, so the free
public deployment route uses `Dockerfile.render` and the root `render.yaml`.
The image build downloads pinned public Hub revisions: a 42.7 MB
deployment-only checkpoint derived without retraining, and a SHA-verified 3 MB
archive containing the exact 500-image gallery. The single-archive asset path
was downloaded and safely extracted locally. See `deploy/README_RENDER.md`.

Public demo: https://ddpm-derm-augmentation-demo.onrender.com

In Colab, the deployment-only checkpoint produced seven probabilities with no
pandas import. A full Uvicorn/FastAPI health and prediction request returned
HTTP 200 with the disclaimer; measured RSS was 385.5 MB current and 408.9 MB
peak. These are user-run Colab results, not a completed Render validation.

## Deployment validation status

The Render Blueprint built commit `df75c05` on the explicit free plan. Public
checks passed for `/health`, OpenAPI, one valid prediction, MIME mismatch 415,
damaged-image 400, the 24-item gallery API, an actual PNG gallery response,
visible attribution, and the medical disclaimer. The first request during a
free-tier cold start briefly returned Render's `x-render-routing: no-server`
404; the following health request woke the service and returned 200. Local
`docker build` remains unrun and must not be described as locally validated.

## Constraints honored

- Fixed `lesion_id` split is read, never re-derived (smoke test asserts no leakage).
- No paths hard-coded to a personal machine — all via config/env.
- DDPM/classifier train on the train split only.
- df F1 is the primary metric; accuracy is never the headline.
