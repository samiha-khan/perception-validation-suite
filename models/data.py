"""GTSRB / CIFAR-10 dataset and loader utilities shared across all stages.

Split design:
  - GTSRB official *train* set -> split 80/10/10 into train / val / calib
    (fixed seed, disjoint). `val` is held out from training and used for
    temperature scaling + reliability diagrams. `calib` is held out from
    training and used to fit split conformal prediction.
  - GTSRB official *test* set -> final held-out set used for accuracy,
    OOD in-distribution scores, corruption-robustness eval, and conformal
    coverage validation.
  - CIFAR-10 official *test* set -> out-of-distribution (negative) set for
    OOD benchmarking. Never used for anything else.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from models.model import IMAGENET_MEAN, IMAGENET_STD

NUM_CLASSES = 43


def base_transform(image_size: int) -> transforms.Compose:
    """Resize only -- yields a PIL image, suitable as input to corruption fns."""
    return transforms.Compose([transforms.Resize((image_size, image_size))])


def normalize_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def eval_transform(image_size: int) -> transforms.Compose:
    """Resize + normalize, no augmentation. Used for val/calib/test/OOD/corruption eval."""
    return transforms.Compose([base_transform(image_size), normalize_transform()])


def train_transform(image_size: int) -> transforms.Compose:
    """Resize + light augmentation + normalize. Used only for the training split."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        normalize_transform(),
    ])


class TransformSubset(Dataset):
    """Wraps a Subset (fixed indices) with its own transform.

    torchvision's GTSRB dataset bakes the transform in at construction time,
    but train/val/calib need *different* transforms over the *same* fixed
    split indices. This applies a transform on top of an already-selected
    subset without touching the underlying dataset's transform.
    """

    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _split_indices(n: int, fractions: list[float], seed: int) -> list[list[int]]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    sizes = [int(f * n) for f in fractions]
    sizes[-1] = n - sum(sizes[:-1])  # remainder to last split
    splits, start = [], 0
    for size in sizes:
        splits.append(perm[start:start + size])
        start += size
    return splits


def get_gtsrb_splits(root: str | Path, image_size: int, seed: int = 42,
                      fractions: list[float] | None = None):
    """Returns dict with keys: train, val, calib, test (each a Dataset)."""
    fractions = fractions or [0.8, 0.1, 0.1]
    root = str(root)

    raw_train = datasets.GTSRB(root=root, split="train", download=True, transform=None)
    raw_test = datasets.GTSRB(root=root, split="test", download=True, transform=None)

    n = len(raw_train)
    train_idx, val_idx, calib_idx = _split_indices(n, fractions, seed)

    train_ds = TransformSubset(Subset(raw_train, train_idx), train_transform(image_size))
    val_ds = TransformSubset(Subset(raw_train, val_idx), eval_transform(image_size))
    calib_ds = TransformSubset(Subset(raw_train, calib_idx), eval_transform(image_size))
    test_ds = TransformSubset(Subset(raw_test, list(range(len(raw_test)))), eval_transform(image_size))

    return {"train": train_ds, "val": val_ds, "calib": calib_ds, "test": test_ds}


def get_gtsrb_test_raw(root: str | Path, image_size: int):
    """GTSRB test set resized but NOT normalized -- for corruption pipelines
    that need to apply pixel-space corruptions before normalization."""
    root = str(root)
    raw_test = datasets.GTSRB(root=root, split="test", download=True, transform=base_transform(image_size))
    return raw_test


def get_cifar10_test(root: str | Path, image_size: int):
    """CIFAR-10 test set, resized + normalized to match the backbone's input
    distribution. Used exclusively as the OOD negative set."""
    root = str(root)
    return datasets.CIFAR10(root=root, train=False, download=True, transform=eval_transform(image_size))


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool = False, num_workers: int = 4) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, pin_memory=False)
