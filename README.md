# ddpm-derm-augmentation

Portfolio project: **can a DDPM actually help a downstream classifier** on the
imbalanced HAM10000 skin-lesion dataset? Target minority class is **df**
(dermatofibroma, only 115 images). End goal is a deployable demo
(FastAPI + Docker + Hugging Face Spaces).

> Status: **Stage 1 (data layer + C0/C1 classifier baseline) scaffolded and
> smoke-tested.** DDPM (Stage 2) and deployment (Stage 4) are not built yet.

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
  notebooks/
    colab_classifier_baseline.py   # `# %%` cell script for Colab
  requirements.txt
```

The layers are split by dependency weight on purpose: `config` / `manifests` /
`metrics` need only pandas+numpy+pillow, so the whole data path can be verified
locally before spending GPU time. `dataset` / `model` / `train_classifier` need
torch and are meant to run on Colab.

## Data

Not committed (HAM10000 is non-commercial licensed; `.gitignore` excludes it).
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

Open **`notebooks/colab_classifier_baseline.ipynb`** in Colab and run top to
bottom (edit only the paths cell). `data/` is inside the project, so uploading
the one project folder to Drive is all you need. Locally (from `src/`):

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
See `outputs/README.md` for the full layout and where to drop files you ran.

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

**C4 status:** the torch-free wiring is implemented and smoke-tested locally
(`build_classifier_frame("C4", ...)`, a portable synthetic manifest, and the
`sample_ddpm` → `publish_synthetic` staging/validate/publish flow). The formal
epoch-100 generated dataset and the GPU C1/C4 classifier runs have **not** been
executed yet. C4 reads its synthetic df from an explicit
`--generated-manifest` (relative image paths, resolved against the manifest
dir); there is no default synthetic directory.

## Not done yet (next milestones, in order)

1. **Formal epoch-100 synthetic df on Colab** — sample 500 df from the epoch-100
   EMA snapshot onto `/content` staging, then `publish_synthetic` to the
   versioned `outputs/synthetic_df/epoch0100_seed0/` (writes `_READY.json` only
   after a destination re-validation). *(Stage-1 C0/C1 baseline already done.)*
2. **Matched-585 C1/C4 on Colab** — C1 (`--df-target-count 585`) and C4
   (`--df-target-count 585 --generated-manifest .../epoch0100_seed0/synthetic_df.csv`)
   × 3 seeds into a **new** `outputs/classifier_df585/` base (Stage-1
   `outputs/classifier/` is left intact). C0 is reused from Stage 1.
3. **Combined C0/C1/C4 aggregation + figures** — `aggregate_results.py` reads
   one dir per call, so the Stage-1 C0 and the matched-585 C1/C4 are currently
   summarised as two tables; a single C0/C1/C4 view still needs a small update.
4. **Stage 4 — deployment**: FastAPI inference + gallery, Docker, HF Spaces,
   medical disclaimer, license notice.

## Constraints honored

- Fixed `lesion_id` split is read, never re-derived (smoke test asserts no leakage).
- No paths hard-coded to a personal machine — all via config/env.
- DDPM/classifier train on the train split only.
- df F1 is the primary metric; accuracy is never the headline.
