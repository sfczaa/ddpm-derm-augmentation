"""Class-conditional U-Net via HuggingFace diffusers. Requires torch + diffusers.

We use ``diffusers.UNet2DModel`` rather than a hand-rolled net so the noise
predictor is a standard, well-tested implementation (fewer places to hide a
silent shape/convergence bug, since this repo can only run torch on Colab).
Class conditioning is enabled with ``num_class_embeds``: the model learns one
embedding per class and adds it to the timestep embedding, so sampling a given
class is just passing ``class_labels``.

The predicted quantity is the noise (epsilon); see ``diffusion.py``.
"""

from __future__ import annotations

from diffusers import UNet2DModel

from .. import config


def build_unet(
    img_size: int = 64,
    num_classes: int = config.NUM_CLASSES,
    tiny: bool = False,
) -> UNet2DModel:
    """Build a class-conditional UNet2DModel.

    ``tiny=True`` returns a deliberately small net (few channels, one block per
    resolution) for the CPU smoke test; the real model uses ``tiny=False``.
    ``img_size`` must be divisible by 2**(num_resolutions-1).
    """
    if tiny:
        return UNet2DModel(
            sample_size=img_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=1,
            block_out_channels=(32, 64),
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
            num_class_embeds=num_classes,
        )
    return UNet2DModel(
        sample_size=img_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        num_class_embeds=num_classes,
    )
