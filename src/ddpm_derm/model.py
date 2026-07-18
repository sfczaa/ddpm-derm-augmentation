"""Classifier backbones. Requires torch + torchvision.

ResNet-18 (ImageNet-pretrained by default) with a fresh 7-way head. Kept
deliberately small so C0/C1/C4 all train quickly on a free Colab T4 with
identical architecture and only the df-augmentation strategy differing.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import torch
import torch.nn as nn
from torchvision import models

from . import config

COCA_ARCH = "coca_vit_b32"
COCA_MODEL_NAME = "coca_ViT-B-32"
COCA_PRETRAINED = "laion2b_s13b_b90k"


class FrozenCoCaClassifier(nn.Module):
    """Frozen OpenCLIP image encoder with a trainable linear classifier."""

    def __init__(self, encoder: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()
        self.head = nn.Linear(feature_dim, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images):
        self.encoder.eval()
        with torch.no_grad():
            features = self.encoder.encode_image(images)
        return self.head(features)


def _coca_feature_dim(encoder: nn.Module) -> int:
    visual = getattr(encoder, "visual", None)
    feature_dim = getattr(visual, "output_dim", None)
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ValueError("CoCa image encoder has no valid visual.output_dim")
    return feature_dim


def _transform_identity(transform) -> str:
    return repr(transform)


def _input_resolution(encoder: nn.Module) -> tuple[int, int]:
    size = getattr(getattr(encoder, "visual", None), "image_size", None)
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    raise ValueError("CoCa image encoder has no valid visual.image_size")


def trainable_parameters(model: nn.Module):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in trainable_parameters(model))
    return total, trainable


def model_identity(
    model: nn.Module, arch: str, img_size: int, pretrained: bool = True
) -> dict:
    total, trainable = parameter_counts(model)
    if arch == COCA_ARCH:
        try:
            open_clip_version = version("open_clip_torch")
        except PackageNotFoundError:
            open_clip_version = None
        return {
            "arch": COCA_ARCH,
            "model_name": model.model_name,
            "pretrained_tag": model.pretrained_tag,
            "freeze_mode": "frozen_image_encoder_linear_head",
            "preprocessing_identity": {
                "train": model.train_preprocess_identity,
                "eval": model.eval_preprocess_identity,
            },
            "input_resolution": list(model.input_resolution),
            "total_parameter_count": total,
            "trainable_parameter_count": trainable,
            "open_clip_torch_version": open_clip_version,
            "torch_version": torch.__version__,
        }
    return {
        "arch": "resnet18",
        "model_name": "torchvision_resnet18",
        "pretrained_tag": "IMAGENET1K_V1" if pretrained else None,
        "freeze_mode": "trainable",
        "preprocessing_identity": {
            "train": f"ddpm_derm.dataset.build_transforms:train:{img_size}",
            "eval": f"ddpm_derm.dataset.build_transforms:eval:{img_size}",
        },
        "input_resolution": [img_size, img_size],
        "total_parameter_count": total,
        "trainable_parameter_count": trainable,
        "open_clip_torch_version": None,
        "torch_version": torch.__version__,
    }


def build_coca_classifier(
    num_classes: int = config.NUM_CLASSES,
    pretrained: str = COCA_PRETRAINED,
    open_clip_module=None,
) -> FrozenCoCaClassifier:
    if open_clip_module is None:
        try:
            import open_clip as open_clip_module
        except ImportError as exc:
            raise ImportError(
                "coca_vit_b32 requires open_clip_torch==3.3.0"
            ) from exc
    available = set(tuple(item) for item in open_clip_module.list_pretrained())
    if (COCA_MODEL_NAME, pretrained) not in available:
        raise ValueError(
            f"unsupported OpenCLIP model/tag: {COCA_MODEL_NAME}/{pretrained}"
        )
    encoder, train_transform, eval_transform = (
        open_clip_module.create_model_and_transforms(
            COCA_MODEL_NAME,
            pretrained=pretrained,
        )
    )
    model = FrozenCoCaClassifier(
        encoder,
        feature_dim=_coca_feature_dim(encoder),
        num_classes=num_classes,
    )
    model.arch = COCA_ARCH
    model.model_name = COCA_MODEL_NAME
    model.pretrained_tag = pretrained
    model.train_preprocess = train_transform
    model.eval_preprocess = eval_transform
    model.train_preprocess_identity = _transform_identity(train_transform)
    model.eval_preprocess_identity = _transform_identity(eval_transform)
    model.input_resolution = _input_resolution(encoder)
    return model


def build_model(
    num_classes: int = config.NUM_CLASSES,
    arch: str = "resnet18",
    pretrained: bool = True,
    coca_pretrained: str = COCA_PRETRAINED,
    freeze_backbone: bool = False,
    open_clip_module=None,
) -> nn.Module:
    if arch == COCA_ARCH:
        if not freeze_backbone:
            raise ValueError("coca_vit_b32 requires --freeze-backbone")
        if not pretrained:
            raise ValueError("coca_vit_b32 requires its pinned pretrained weights")
        return build_coca_classifier(
            num_classes=num_classes,
            pretrained=coca_pretrained,
            open_clip_module=open_clip_module,
        )
    if arch != "resnet18":
        raise ValueError(f"unsupported classifier architecture: {arch!r}")
    if freeze_backbone:
        raise ValueError("--freeze-backbone is only supported for coca_vit_b32")
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
