"""Torch Dataset / transforms / dataloaders. Requires torch + torchvision.

Kept separate from ``manifests`` so the manifest and metric logic stay
importable without a deep-learning stack (see the local smoke test).
"""

from __future__ import annotations

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from . import config, ddpm_sampler

# HAM10000 images are RGB dermatoscopy photos; ImageNet stats are a fine default
# because the classifier backbone is ImageNet-pretrained.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int = 128, train: bool = True):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


class HAMDataset(Dataset):
    """Reads (image, label) pairs from a manifest frame.

    ``image_path`` values are resolved relative to the data dir via config, so
    the same frame works locally and on Colab.
    """

    def __init__(self, frame: pd.DataFrame, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        path = config.resolve_image_path(row["image_path"])
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label_idx"])
        return image, label


def build_dataloader(
    frame: pd.DataFrame,
    img_size: int = 128,
    batch_size: int = 32,
    train: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    dataset = HAMDataset(frame, transform=build_transforms(img_size, train=train))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# --- DDPM (stage 2) ----------------------------------------------------------
# The generator works in [-1, 1] (mean/std = 0.5), not ImageNet stats: it is
# trained from scratch, and DDIM sampling assumes a symmetric [-1, 1] range.
_DDPM_MEAN = (0.5, 0.5, 0.5)
_DDPM_STD = (0.5, 0.5, 0.5)


def build_ddpm_transforms(img_size: int = 64, train: bool = True):
    """Transforms for DDPM training: resize, optional h-flip, scale to [-1, 1].

    Only a horizontal flip is used for augmentation; rotations/colour jitter are
    avoided so the generator does not learn augmentation artefacts.
    """
    ops = [transforms.Resize((img_size, img_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [transforms.ToTensor(), transforms.Normalize(_DDPM_MEAN, _DDPM_STD)]
    return transforms.Compose(ops)


def build_ddpm_dataloader(
    frame: pd.DataFrame,
    img_size: int = 64,
    batch_size: int = 64,
    train: bool = True,
    num_workers: int = 2,
    sampler_strategy: str = ddpm_sampler.DEFAULT_SAMPLER_STRATEGY,
    sampler_generator=None,
) -> DataLoader:
    dataset = HAMDataset(frame, transform=build_ddpm_transforms(img_size, train=train))
    plan = ddpm_sampler.sampling_plan(sampler_strategy)
    if not train and plan["use_weighted_sampler"]:
        raise ValueError("sqrt_balanced sampler is only valid for DDPM training")
    sampler = None
    if train and plan["use_weighted_sampler"]:
        weights = ddpm_sampler.per_sample_weights(
            frame["label_idx"].tolist(), sampler_strategy
        )
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=sampler_generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and plan["shuffle"],
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )
