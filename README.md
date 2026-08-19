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
> build and public demo endpoints were validated on 2026-07-14; the local Docker
> build and a container smoke test were completed on 2026-08-18.

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

## Post-Stage-4 exploratory sqrt-balanced DDPM

The frozen natural-sampling DDPM and its formal outputs remain unchanged. A
separate exploratory path can reduce class imbalance more gently: a train row
in class c receives weight 1 / sqrt(n_c), then
WeightedRandomSampler(replacement=True, num_samples=len(train)) draws an
epoch. This is not fully class-balanced.

The natural strategy is the default and keeps the historical shuffled
DataLoader. sqrt_balanced must be selected explicitly and requires isolated
checkpoint, snapshot, preview, and metadata paths. Resume rejects a different
sampler strategy. The standalone metadata and checkpoint record the strategy,
train class counts, row weights, expected proportions, seed, fixed config,
train-only source, Git commit, and output paths.

All exploratory Drive artifacts live under:

~~~text
outputs/exploratory_balanced_ddpm/<version>/
~~~

See COLAB_BALANCED_DDPM.md for the tiny smoke, resume checks, 100-epoch run,
and versioned candidate-generation commands. That runbook intentionally stops
before downstream classification; the separately approved comparison is
reported below.

### Exploratory downstream comparison (completed)

The versioned sqrt-balanced candidate was evaluated as an isolated C4@585
condition with the same fixed split, classifier configuration, and seeds 0–2
used by the frozen matched-585 comparison. Each run used 20 epochs; model
selection remained based on validation df F1. Test results are mean +/-
population standard deviation across three seeds.

| Condition | df F1 (primary) | macro-F1 | df recall |
|---|---:|---:|---:|
| C4@585, sqrt-balanced DDPM candidate | 0.682 +/- 0.047 | 0.655 +/- 0.047 | 0.604 +/- 0.059 |
| C4@585, frozen natural-DDPM data | 0.612 +/- 0.042 | 0.645 +/- 0.010 | 0.521 +/- 0.029 |
| C1@585, duplicated real df | 0.660 +/- 0.042 | 0.651 +/- 0.021 | 0.604 +/- 0.029 |

The exploratory mean df F1 difference was +0.0706 versus natural C4@585 and
+0.0223 versus C1@585. This is descriptive evidence only: the test split has
16 df cases, the split is fixed, only three seeds were run, and no statistical
significance test was performed. It does not establish a validated, clinical,
or medical-effectiveness claim, and it does not automatically replace the
deployed C1 checkpoint.

### Independent frozen CoCa robustness check (completed)

A second classifier tested whether the sqrt-balanced C4 direction transferred
to a different image representation. It used the OpenCLIP
`coca_ViT-B-32` image encoder with `laion2b_s13b_b90k` weights, native 224px
preprocessing, a frozen encoder, and a trainable seven-class linear head. C1
and C4 used the same matched-585 data counts, seeds 0-2, 20 epochs, and
validation-df-F1 checkpoint selection rule.

| CoCa condition | df F1 (primary) | macro-F1 | df recall |
|---|---:|---:|---:|
| C1@585, duplicated real df | 0.000 +/- 0.000 | 0.114 +/- 0.000 | 0.000 +/- 0.000 |
| C4@585, sqrt-balanced DDPM candidate | 0.000 +/- 0.000 | 0.114 +/- 0.000 | 0.000 +/- 0.000 |

The paired C4-C1 df F1 difference was 0.000. Validation df F1 stayed at zero
for every epoch in all six runs, so each best checkpoint remained the first
epoch and predicted only the majority `nv` class on the test set. This does
not reproduce the ResNet-18 direction. It is a floor-collapse result, not
evidence that C1 and C4 are equivalent or that synthetic data is generally
ineffective. The existing C1 seed-2 deployment remains unchanged.

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

Then check `http://localhost:7860/health`, `/docs`, and the upload UI. A local
Docker build and smoke validation were completed on 2026-08-18: the image
built, `GET /health` and one `POST /api/predict` with a generated non-patient
fixture both returned HTTP 200, and the served identity was C1 seed 2. This
validates local container serving only; it does not validate model accuracy,
load/concurrency, other platforms, or public deployment.
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
404; the following health request woke the service and returned 200. Separately,
the local `docker build` was run on 2026-08-18: `GET /health` and one
`POST /api/predict` with a generated non-patient fixture each returned 200 from
the C1 seed 2 container. That covers local build and serving only, not model
accuracy, load/concurrency, other platforms, or public deployment.

## Does the synthetic data actually help? A generator diagnostic

The sqrt-balanced exploratory comparison added 500 synthetic `df` images in C4 rather than duplicating the
85 real train images as C1 did. Test df F1 moved by `+0.0223`. This is a small difference, so the first
question was whether the generator was adding anything at all.

The DDPM had only 85 real train `df` images. If it had memorised them, C4 would be functionally equivalent
to C1, and a near-zero gap would be the expected result. Memorisation therefore had to be measured before
interpreting the downstream comparison.

The original check reported a 32px nearest-neighbour distance of `min=3.20` without a reference scale, so
there was no basis for deciding whether that was close. It also compared only with unflipped originals,
while DDPM training uses `RandomHorizontalFlip`, leaving memorisation up to a mirror image undetected.

### Building a reference scale from the fixed split

The 14 validation `df` are real lesions that the generator never saw, so their distance to the train set
is a reference for a genuinely new `df`. Leave-one-out distances among the 85 real train `df` provide a
second reference. Both come from the fixed split without deriving a new one.

Distances were measured flip-aware in two independent spaces: pixels at the generator's native 64px
resolution, and penultimate features from the project's real-data-only C1 ResNet-18. The feature-space
judge therefore does not depend on synthetic training data.

All figures below describe the published epoch-100 set at
`outputs/synthetic_df/epoch0100_seed0/` (500 images, `run_seed0_epoch0100.pt`, EMA weights, DDIM 50 steps,
eta 0, seed 0), which is what the formal C4 condition trains on. An earlier revision of this section
reported the same diagnostics run against a different 500-image batch that shares the directory tree and
the filenames but not the bytes; the conclusions were unchanged but several figures were not. Each
diagnostic record now carries a content hash of the images it opened, so a result can be tied to its batch
without trusting an adjacent metadata file.

| Median nearest-neighbour distance into the 85 real train df | pixel @64px | C1 embedding |
|---|---:|---:|
| synthetic (500) | 12.04 | 0.999 |
| real val df (14) - genuinely new lesions | 7.77 | 0.368 |
| real train df - leave-one-out | 8.36 | 0.154 |

The synthetic images are farther from the training set than genuinely new real `df` are. Only `0.2%` of the
500 falls inside the closest validation `df` in the embedding and none does in pixel space, and the
probability that a synthetic image is closer to the training set than a random genuinely-new real `df` is
`0.060` and `0.038` in the two spaces, against `0.5` for indistinguishable. These measurements rule out
memorisation as the explanation for the small C4-C1 difference.

### The diagnostic instead points to distribution shift

Cosine similarity to the nearest real `df` is `0.988` for train-to-train comparisons and `0.932` for new
real `df`, but `0.501` for the synthetic set: the synthetic samples sit outside the real `df` distribution.
Nearest-class assignment says the same thing more directly. Against galleries balanced to 85 images per
class, genuinely new real `df` land nearest to `df` `71.4%` of the time; the synthetic do so `5.8%` of the
time, landing mostly on `nv` and `bcc` instead.

Within-set spacing is `0.77x` of the real set in pixel space but `1.28x` in the embedding. Those point in
opposite directions and the disagreement is itself informative: the images are tightly grouped in raw
colour while being scattered in semantic content, which is not the same failure as a mode-collapsed
generator producing near-duplicates.

A resolution ladder found a synthetic-to-new-real-`df` distance ratio of `1.550` at 64px and `1.532` at 8px.
At 8px, only coarse colour and shape remain, so the gap is not high-frequency detail and raising generator
resolution would not close it. This ruled out a costly higher-resolution retrain before it was spent.

Saturation is `0.093` against `0.188` for real `df`, contrast is `0.088` against `0.145`, and mean RGB is
approximately `0.5` in every channel, at the centre of the normalised range. This is a sample regressing
toward the data mean rather than a model that learned the wrong thing. Two controls support that reading:
passing real validation `df` through the same 64px bottleneck moves cosine similarity only from `0.932` to
`0.905`, so resolution does not reproduce the gap; and real `df` sit `0.053` from the all-class mean RGB
while the synthetic sit `0.239` from it, so this is not the generator averaging over its seven classes.

### What the sampler sweep found

The published set was drawn with DDIM at 50 steps and eta 0. A sweep over sampler settings on the same
checkpoint isolates how much of the above is a sampling artefact, and it contradicted the initial guess.
Step count was not the lever: 1000 steps at eta 0 gives saturation `0.090`, no better than 50 steps at
`0.094`. Stochasticity was. Moving eta from 0 to 1 at 50 steps raises saturation to `0.253`, `1.35x` real
`df` and therefore past it rather than onto it.

Colour is recoverable, distance is not. Every configuration tested leaves the embedding nearest-neighbour
median between `0.97` and `1.07`, against `0.368` for genuinely new real `df`. No sampler setting moved the
samples onto the real `df` manifold, which is the measurement that matters for augmentation. The epoch
sweep at the published setting shows saturation still climbing at the end of training (`0.054` at epoch 60,
`0.069` at 80, `0.094` at 100), so the run was also stopped while it was still improving.

Inspecting the sweep images rather than only its statistics changes how the eta result should be read. The
highest-saturation configuration produces frames in fluorescent cyan, magenta and flat orange that are not
skin at all, and the 1000-step deterministic setting produces colour speckle. The recovered saturation is
noise rather than restored skin tone, which is consistent with distance never improving, and it is a
reminder that a single summary statistic can move in the right direction while the underlying samples get
worse. The published setting is washed out but every image still reads as skin.

### Limits of interpretation

- The reference distribution contains only 14 validation `df`, and the test split contains only 16 real `df`. Every df-level metric here rests on very few images, and the `+0.0223` difference is well inside that noise.
- Distance in a classifier's feature space does not measure visual realism or clinical validity.
- This diagnostic is descriptive: it defines no thresholds or pass/fail rule and performs no significance testing, consistent with the project constraints.
- Distribution shift may explain the weak downstream difference, but the evidence does not demonstrate that it caused the result.

### Where this leaves the generator

Three candidate explanations have now been excluded on evidence rather than intuition: memorisation, a
high-frequency resolution deficit, and class averaging. The sweep adds a fourth, that sampler settings alone
would fix it. Colour and contrast respond to eta, but nothing moves the samples onto the real `df` manifold,
so a sampler change would produce more saturated images that are still off-distribution.

What remains consistent with all of it is a capacity or data limit: 85 real training images is very little
to learn a lesion class from, and the epoch sweep shows the run had not converged. Neither claim is
established here, and no retrain, architecture change, or new condition is proposed on this evidence alone.

A follow-up condition is pre-registered in `C4_FILTERED_EXPERIMENT_DESIGN.md`: whether the subset of
synthetic images closest to the real `df` manifold is more useful as augmentation than the pool as a whole,
with the acceptance threshold and judge fixed in advance. It is deliberately gated on a better pool, since
under the pre-registered threshold only a small fraction of the current set qualifies.

### Reproducibility

The sweep notebook is `notebooks/colab_ddpm_sampling_sweep_diagnostic.ipynb`. The supporting scripts are
read-only against the fixed train and validation manifests and never use the test split:

- `scripts/ddpm_memorization_diagnostic.py`
- `scripts/ddpm_failure_localization.py`
- `scripts/ddpm_sampling_sweep.py`

## Constraints honored

- Fixed `lesion_id` split is read, never re-derived (smoke test asserts no leakage).
- No paths hard-coded to a personal machine — all via config/env.
- DDPM/classifier train on the train split only.
- df F1 is the primary metric; accuracy is never the headline.
