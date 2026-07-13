"""Gaussian diffusion (DDPM training loss) + DDIM sampling. Requires torch.

Standard DDPM with a linear beta schedule and epsilon-prediction, plus a
deterministic DDIM sampler (eta=0) for fast inference (~50 steps for the demo).

Conventions
-----------
- Images are in ``[-1, 1]`` (the dataset normalizes with mean/std = 0.5).
- The model predicts the noise ``epsilon`` added at step ``t``.
- ``UNet2DModel`` is called as ``model(x_t, t, class_labels=y).sample``.

Nothing here is bespoke math beyond the textbook DDPM/DDIM equations; the point
of the module is to keep those equations in one small, readable place.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _extract(arr: torch.Tensor, t: torch.Tensor, broadcast_to: torch.Tensor) -> torch.Tensor:
    """Gather ``arr[t]`` and reshape to broadcast against ``broadcast_to``.

    ``arr`` is a 1-D schedule tensor, ``t`` a 1-D long tensor of timesteps
    (one per sample). Returns shape ``[B, 1, 1, 1]`` for image tensors.
    """
    out = arr.to(t.device).gather(0, t)
    return out.reshape(t.shape[0], *([1] * (broadcast_to.dim() - 1)))


class GaussianDiffusion:
    """Holds the noise schedule and the train/sample operations.

    Not an ``nn.Module``: it owns no learnable parameters, only buffers derived
    from the schedule. Schedule tensors live on CPU and are moved to the sample
    tensor's device on demand.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        self.timesteps = int(timesteps)
        betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # cast to float32 for use with the (float32) model
        self.betas = betas.float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).float()

    # --- forward (training) noising ------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample ``x_t ~ q(x_t | x_0)`` in closed form."""
        sqrt_ab = _extract(self.sqrt_alphas_cumprod, t, x0)
        sqrt_1m_ab = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0)
        return sqrt_ab * x0 + sqrt_1m_ab * noise

    def p_losses(self, model, x0: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """DDPM training loss: MSE between true and predicted noise.

        A timestep ``t`` is drawn uniformly per sample.
        """
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = model(x_t, t, class_labels=class_labels).sample
        return F.mse_loss(pred, noise)

    # --- reverse (sampling) --------------------------------------------------
    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        n: int,
        class_labels: torch.Tensor,
        img_size: int,
        channels: int = 3,
        num_steps: int = 50,
        eta: float = 0.0,
        device=None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Generate ``n`` images with DDIM. Returns a tensor in ``[-1, 1]``.

        ``eta=0`` (default) is the deterministic DDIM sampler. ``class_labels``
        must be a long tensor of length ``n``.
        """
        was_training = model.training
        model.eval()
        if device is None:
            device = next(model.parameters()).device
        ab = self.alphas_cumprod.to(device)

        x = torch.randn(n, channels, img_size, img_size, device=device)
        # descending timesteps, e.g. [T-1, ..., 0], num_steps of them
        seq = torch.linspace(self.timesteps - 1, 0, num_steps, device=device).round().long()

        for i in range(num_steps):
            t = int(seq[i].item())
            t_batch = torch.full((n,), t, device=device, dtype=torch.long)
            eps = model(x, t_batch, class_labels=class_labels).sample

            ab_t = ab[t]
            ab_prev = ab[int(seq[i + 1].item())] if i < num_steps - 1 else torch.tensor(1.0, device=device)

            x0_pred = (x - torch.sqrt(1.0 - ab_t) * eps) / torch.sqrt(ab_t)
            if clip_denoised:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            if eta > 0:
                sigma = eta * torch.sqrt(
                    (1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - ab_t / ab_prev)
                )
                noise = torch.randn_like(x)
            else:
                sigma = torch.zeros((), device=device)
                noise = torch.zeros_like(x)

            dir_xt = torch.sqrt(torch.clamp(1.0 - ab_prev - sigma ** 2, min=0.0)) * eps
            x = torch.sqrt(ab_prev) * x0_pred + dir_xt + sigma * noise

        if was_training:
            model.train()
        return x


def to_uint8_images(x: torch.Tensor) -> torch.Tensor:
    """Map a ``[-1, 1]`` batch to ``uint8`` ``[0, 255]`` on CPU, shape [B,H,W,C]."""
    x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
    x = (x * 255.0).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu()
