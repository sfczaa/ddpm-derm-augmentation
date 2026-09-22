# Render Free deployment

This route packages the FastAPI service for Render.
The public Hugging Face model and dataset repositories remain the versioned
asset source. `Dockerfile.render` downloads only pinned revisions during the
image build; no HF token is required or stored.

## Free-tier constraints

The original deployment targeted a free instance with limited memory and
CPU. Cold starts were observed after idle periods. Confirm current plan limits
and availability with Render before deploying. The filesystem is ephemeral. This demo does not save uploads or
write application state, so ephemeral storage is acceptable.

The deployment-only checkpoint was exported from the formal C1 seed-2
checkpoint without retraining. Colab measurements after a real FastAPI request
were 385.5 MB current RSS and 408.9 MB peak RSS. This is evidence of feasibility,
not a guarantee of Render runtime behavior.

## Pinned public assets

- Model: `sfczaa/ddpm-derm-c1-seed2`
- Model revision: `30b41486b5353f2a99aceecef2fc41b178c2697b`
- File: `deploy_weights.pt`
- SHA-256: `85910296186433c6cdad2d82368646c983b586ec861d605c1338317fb8306a53`
- Gallery: `sfczaa/ddpm-derm-synthetic-gallery`
- Gallery revision: `60b046e4c2ae77f1505e8c2b426c24763741c3a5`
- Gallery version: `epoch0100_seed0`, 500 manifest-ordered PNGs
- Deployment archive: `gallery_epoch0100_seed0.zip`
- Archive SHA-256: `da3d582082323728e2b0558c27e26af124c683dacf336915d1212acd8abd0bc5`

## Deploy

1. Sign in to Render and create a new Blueprint.
2. Connect `https://github.com/sfczaa/ddpm-derm-augmentation`.
3. Render reads the root `render.yaml`, selects `Dockerfile.render`, and creates
   the `ddpm-derm-augmentation-demo` web service with `plan: free`.
4. Do not add an HF token, persistent disk, GPU, or paid instance.
5. Wait for the build log to show `RENDER_ASSETS_OK` and for `/health` to pass.

## Acceptance checks

Treat the Render deployment as unvalidated until all checks pass on its public
URL: `/health`, `/docs`, one valid prediction with seven probabilities and the
medical disclaimer, invalid-upload 4xx behavior, deterministic gallery display,
visible attribution, and recovery after a free-tier cold start.

## Public validation record

Validated on 2026-07-14 at
https://ddpm-derm-augmentation-demo.onrender.com from the Stage 4 demo commit
(`d3e34e7` on `main`) on the explicit Render Free plan. `/health` reported C1
seed 2 and `epoch0100_seed0`; a synthetic-gallery upload returned seven
probabilities summing to approximately one and included the medical
disclaimer. OpenAPI, the 24-item gallery response, a real `image/png`
gallery route, attribution, MIME-mismatch 415, and damaged-image 400 checks
passed.

During a free-tier cold start, the first root request briefly returned Render's
`x-render-routing: no-server` 404. The next health request woke the service and
returned 200, after which the full acceptance suite passed. This is a hosting
limitation, not clinical validation. Docker CLI was unavailable during this
validation session; the later local Docker check is recorded in the root README.
