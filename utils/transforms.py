from __future__ import annotations

from typing import Dict

from torchvision import transforms


def build_transforms(cfg: Dict) -> Dict[str, transforms.Compose]:
    size = cfg.get("image_size", 512)
    mean = cfg.get("mean", [0.5, 0.5, 0.5])
    std = cfg.get("std", [0.25, 0.25, 0.25])
    train_aug = [transforms.Resize((size, size))]
    if cfg.get("random_flip", True):
        train_aug.append(transforms.RandomHorizontalFlip())
        train_aug.append(transforms.RandomVerticalFlip())
    if cfg.get("random_rotation", True):
        train_aug.append(transforms.RandomRotation(15))
    if cfg.get("color_jitter", False):
        train_aug.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    train_aug.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    eval_aug = [transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return {
        "train": transforms.Compose(train_aug),
        "val": transforms.Compose(eval_aug),
        "test": transforms.Compose(eval_aug),
    }
