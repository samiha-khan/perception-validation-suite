"""Shared inference helper: run a model over a loader, collect logits/features/labels.

Used by ood/, calibration/, and robustness/ so each stage does its own single
forward pass over a given dataset rather than re-implementing this loop.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_logits_features(model, loader: DataLoader, device: str | torch.device):
    model.eval()
    all_logits, all_features, all_labels = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits, features = model(images, return_features=True)
        all_logits.append(logits.cpu().numpy())
        all_features.append(features.cpu().numpy())
        all_labels.append(np.asarray(labels))
    return (np.concatenate(all_logits), np.concatenate(all_features), np.concatenate(all_labels))


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
