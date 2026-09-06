# Optional Hugging Face Space deployment

This replaces the earlier Docker-Space handoff. Hugging Face changed its pricing
in 2026: Gradio and Docker Spaces both run on compute and **require a paid plan
to create** — PRO for personal accounts — with one exception the docs state
plainly:

> Static Spaces are free for everyone. Gradio and Docker Spaces run on compute
> and require a paid plan to create: PRO for personal accounts, Team or
> Enterprise for organizations. **Free personal accounts in good standing can
> still host up to 2 Gradio Spaces running on ZeroGPU.**

So the free route to a Space that actually runs the model is Gradio on ZeroGPU,
not Docker. The Render deployment is unaffected and stays as it is; the two
serve the same pinned checkpoint, which `tests/test_space.py` enforces.

`deploy/space/` holds everything specific to the Space.

## Assemble the Space repository

Create the Space with **SDK: Gradio**, then set hardware to **ZeroGPU** in
Settings. Upload:

```text
README.md          <- deploy/space/README.md   (carries the Space metadata block)
app.py             <- deploy/space/app.py
requirements.txt   <- deploy/space/requirements.txt
src/ddpm_derm/     <- deploy.py, model.py, config.py and their package files
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

## Runtime

The Space uses the same `ClassifierService` as Render and performs inference on
CPU. The `spaces.GPU` decorator supports the ZeroGPU entry point; its local
fallback leaves the function unchanged.

## Acceptance checks after the Space builds

1. The Space starts and the log shows `RENDER_ASSETS_OK` with the pinned
   revisions and `gallery_pngs=500`.
2. The header reports `C1 seed 2` and the gallery version `epoch0100_seed0`.
3. A test image returns seven finite probabilities summing to one.
4. The gallery tab shows 24 images.
5. The page carries the medical disclaimer, the CC BY-NC 4.0 link, the ViDIR
   attribution, the DOI, and the statement of modifications.
6. An oversized image (over 20,000,000 pixels) is refused with a clear message.

The deployed classifier remains C1 seed 2, selected by validation df F1.
Deployment checks do not establish model accuracy.
