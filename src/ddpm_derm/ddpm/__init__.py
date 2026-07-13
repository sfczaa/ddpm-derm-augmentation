"""Stage 2: class-conditional DDPM for synthetic ``df`` generation.

Requires torch + diffusers (Colab), so nothing here is imported by the
torch-free smoke test. The generator is trained on the *train split only*
(all 7 classes) and ``df`` images are sampled by conditioning; val/test
images must never be seen by the generator.

Modules:
- ``unet``     : diffusers ``UNet2DModel`` factory (class-conditional).
- ``diffusion``: linear-schedule Gaussian diffusion + DDIM sampling.
"""
