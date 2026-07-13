"""Classifier backbone. Requires torch + torchvision.

ResNet-18 (ImageNet-pretrained by default) with a fresh 7-way head. Kept
deliberately small so C0/C1/C4 all train quickly on a free Colab T4 with
identical architecture and only the df-augmentation strategy differing.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from . import config


def build_model(
    num_classes: int = config.NUM_CLASSES,
    arch: str = "resnet18",
    pretrained: bool = True,
) -> nn.Module:
    if arch != "resnet18":
        raise ValueError(f"only resnet18 is wired up for now, got {arch!r}")
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
