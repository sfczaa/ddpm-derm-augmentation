---
title: HAM10000 Classifier Portfolio Demo
emoji: 🔬
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
license: cc-by-nc-4.0
---

# HAM10000 classifier portfolio demo

Educational portfolio demonstration only. Not for diagnosis or treatment.
The model can be wrong, and its training data and population coverage are limited.

Non-commercial use only. Data attribution: HAM10000 Dataset © ViDIR Group,
Department of Dermatology, Medical University of Vienna; distributed with the
ISIC 2018 data under CC BY-NC 4.0. Cite Tschandl, Rosendahl & Kittler,
*Scientific Data* 5, 180161 (2018), https://doi.org/10.1038/sdata.2018.161.
Project changes include a fixed lesion-level split, resizing, classifier
training, and generation of the derived synthetic gallery. No endorsement by
the dataset creators is implied.

## Space repository contents

Copy this file to `README.md` at the root of the Space repository, then upload
only `app/`, `src/`, `deploy/`, `Dockerfile`, `requirements-deploy.txt`, and
`.dockerignore`. Do not commit the `.pt`, HAM10000 data, user uploads, or the
500 gallery images to the Space Git repository.

## Publish and mount assets

1. Create a Hugging Face **model repository** and upload the selected
   `C1_seed2/best.pt` as `best.pt`.
2. Create a Hugging Face **dataset repository** and upload the complete
   `epoch0100_seed0/` directory, including `images/`, `synthetic_df.csv`,
   `metadata.json`, and `_READY.json`.
3. In the Space settings, attach the model repository as a read-only volume at
   `/models` and the dataset repository as a read-only volume at `/gallery`.
4. Add these non-secret Space variables:

```text
DDPM_DERM_MODEL_PATH=/models/best.pt
DDPM_DERM_MODEL_MANIFEST=/app/deploy/model_manifest.json
DDPM_DERM_CLASS_MAP_PATH=/app/deploy/class_to_idx.json
DDPM_DERM_GALLERY_DIR=/gallery/epoch0100_seed0
```

No token is required at application runtime when the repositories are attached
as volumes. If private assets are used, configure the volume permissions in the
Space settings; never hard-code an HF token.

## Acceptance checks after the Space builds

1. `/health` returns `status=ok`, `variant=C1`, `seed=2`, and the gallery version.
2. `/docs` shows the `/api/predict` response schema.
3. A valid JPG/PNG/WebP below 5 MB returns exactly seven finite probabilities
   that sum to approximately 1 and includes the medical disclaimer.
4. A damaged file, MIME mismatch, unsupported extension, and oversized upload
   each return a clear 4xx response.
5. The gallery renders the deterministic first 24 manifest rows.
6. Confirm the app never writes uploaded images and performs no request-time DDPM.
7. Confirm the visible page includes HAM10000/ViDIR attribution, the paper DOI,
   CC BY-NC 4.0 link, modification notice, and non-commercial-use statement.

Only after all checks pass should the public Space URL be recorded as validated.
