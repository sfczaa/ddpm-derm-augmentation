"""Tiny end-to-end DDPM smoke test (requires torch + diffusers).

Proves the Stage 2 pipeline runs before spending T4 time on a real train:
builds a small class-conditional UNet, does a couple of training steps on a few
real train images, then DDIM-samples a couple of images and checks shapes and
value ranges. Runs on CPU in a few seconds.

    # from the project root, on Colab (or any machine with torch + diffusers)
    python scripts/smoke_ddpm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:
    print("SKIP: torch not installed (this smoke test is meant to run on Colab).")
    sys.exit(0)

from ddpm_derm import config, manifests  # noqa: E402
from ddpm_derm.dataset import build_ddpm_dataloader  # noqa: E402
from ddpm_derm.ddpm.diffusion import GaussianDiffusion, to_uint8_images  # noqa: E402
from ddpm_derm.ddpm.unet import build_unet  # noqa: E402

IMG = 32
N_SMOKE = 8


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    checks = []

    def check(name, ok):
        checks.append((name, bool(ok)))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    # tiny data / model / schedule
    frame = manifests.load_split("train").sample(n=N_SMOKE, random_state=0).reset_index(drop=True)
    loader = build_ddpm_dataloader(frame, img_size=IMG, batch_size=4, train=True, num_workers=0)
    diffusion = GaussianDiffusion(timesteps=50)
    model = build_unet(img_size=IMG, tiny=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    check("schedule length == timesteps", len(diffusion.betas) == 50)
    check("alphas_cumprod is decreasing",
          bool((diffusion.alphas_cumprod[1:] <= diffusion.alphas_cumprod[:-1]).all()))

    # a couple of real training steps
    losses = []
    model.train()
    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        check(f"batch images in [-1,1] (step {step})",
              float(images.min()) >= -1.001 and float(images.max()) <= 1.001)
        optimizer.zero_grad()
        loss = diffusion.p_losses(model, images, class_labels=labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if step >= 1:
            break
    check("training loss is finite", all(l == l and abs(l) < 1e6 for l in losses))

    # DDIM sampling of df
    df_labels = torch.full((2,), config.TARGET_CLASS_IDX, dtype=torch.long, device=device)
    samples = diffusion.ddim_sample(model, 2, df_labels, img_size=IMG, num_steps=8, device=device)
    check("sample shape == (2,3,IMG,IMG)", tuple(samples.shape) == (2, 3, IMG, IMG))
    check("samples are finite", bool(torch.isfinite(samples).all()))
    uint8 = to_uint8_images(samples)
    check("uint8 images shape == (2,IMG,IMG,3)", tuple(uint8.shape) == (2, IMG, IMG, 3))

    n_pass = sum(ok for _, ok in checks)
    print(f"\n{n_pass}/{len(checks)} checks passed")
    sys.exit(0 if n_pass == len(checks) else 1)


if __name__ == "__main__":
    main()
