# Hugging Face Space handoff (Gradio on ZeroGPU)

This is the optional Gradio entry point for the same pinned checkpoint used
by Render. Confirm current account, hardware, and pricing requirements in
Hugging Face before creating a Space. The repository does not establish that
this entry point is deployed.

`deploy/space/` holds everything specific to the Space.

## Assemble the Space repository

Create the Space with **SDK: Gradio**, then set hardware to **ZeroGPU** in
Settings. Upload:

```text
README.md          <- deploy/space/README.md   (carries the Space metadata block)
app.py             <- deploy/space/app.py
requirements.txt   <- deploy/space/requirements.txt
src/ddpm_derm/     <- deploy.py, checkpoint.py, model.py, config.py and their package files
deploy/model_manifest.json
deploy/class_to_idx.json
deploy/download_render_assets.py
```

`app.py` expects `src/` and `deploy/` as siblings, exactly as in this
repository, so copying those two directories in place is enough.

**Do not upload** the `.pt` files, the HAM10000 data, or the 500 gallery images.
`app.py` fetches the checkpoint and the gallery at startup by pinned revision
and the download script hash-checks the archive; that is what keeps the Space
repository small and its provenance auditable.

No token is needed at runtime: both source repositories are public. Never
hard-code an HF token.

## Why the app looks the way it does

- Inference is the same `ClassifierService` the FastAPI deployment uses, so the
  Space cannot drift from Render in preprocessing, class order, or checkpoint
  validation.
- `@spaces.GPU` is present because the free tier is ZeroGPU-gated, not because
  the model needs a GPU. It is a 42.7 MB ResNet-18; the service stays on CPU so
  a between-call GPU deallocation cannot strand the weights on a dead device.
- The guarded `import spaces` degrades to a no-op decorator, so the module stays
  importable outside a Space.

## Acceptance checks after the Space builds

1. The Space starts and the log shows `RENDER_ASSETS_OK` with the pinned
   revisions and `gallery_pngs=500`.
2. The header reports `C1 seed 2` and the gallery version `epoch0100_seed0`.
3. Uploading a dermatoscopic image returns exactly seven finite probabilities.
   Verified locally against the pinned assets on 2026-09-06: they sum to
   `1.0`, and the first gallery image scores `nv 0.9434` / `df 0.0000` — which
   is the distribution shift the diagnostics already record, not a bug.
4. The gallery tab shows 24 images.
5. The page carries the medical disclaimer, the CC BY-NC 4.0 link, the ViDIR
   attribution, the DOI, and the statement of modifications.
6. An oversized image (over 20,000,000 pixels) is refused with a clear message.

## Boundary

A successful build proves the Space serves the pinned checkpoint. It does not
say anything about model accuracy, and nothing here changes the deployed
classifier, which remains C1 seed 2 selected on validation df F1.
