"""Per-corruption accuracy degradation table vs. clean GTSRB test accuracy.

Run from repo root:
    python -m robustness.run_robustness
"""
from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from models.config import load_config, resolve_path
from models.data import get_gtsrb_test_raw, make_loader, normalize_transform
from models.inference import pick_device
from models.model import load_checkpoint
from robustness.corruptions import CORRUPTIONS


class CorruptedDataset(Dataset):
    """Wraps a resized-but-unnormalized GTSRB dataset, applying a pixel-space
    corruption (if any) before ToTensor+Normalize. severity=None/corruption_fn=None
    -> clean pass-through (still round-tripped through uint8 for a fair baseline)."""

    def __init__(self, raw_dataset, corruption_fn, severity: int):
        self.raw_dataset = raw_dataset
        self.corruption_fn = corruption_fn
        self.severity = severity
        self.normalize = normalize_transform()

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        img, label = self.raw_dataset[idx]
        arr = np.asarray(img).astype(np.float32) / 255.0
        if self.corruption_fn is not None:
            arr = self.corruption_fn(arr, self.severity)
        arr_uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        tensor = self.normalize(Image.fromarray(arr_uint8))
        return tensor, label


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    cfg = load_config()
    device = pick_device()
    rcfg = cfg["robustness"]
    data_root = resolve_path(cfg["data"]["root"])
    image_size = cfg["data"]["image_size"]
    bs, nw = cfg["model"]["batch_size"], cfg["model"]["num_workers"]

    checkpoint_path = resolve_path(cfg["model"]["checkpoint"])
    model, ckpt = load_checkpoint(checkpoint_path, device=device)
    print(f"loaded checkpoint (val_acc={ckpt.get('val_acc'):.4f})")

    raw_test = get_gtsrb_test_raw(data_root, image_size)

    clean_loader = make_loader(CorruptedDataset(raw_test, None, 0), bs, num_workers=nw)
    clean_acc = evaluate(model, clean_loader, device)
    print(f"clean test_acc={clean_acc:.4f}")

    rows = [{"corruption": "clean", "severity": 0, "accuracy": clean_acc, "degradation": 0.0}]
    for name in rcfg["corruptions"]:
        fn = CORRUPTIONS[name]
        for severity in rcfg["severities"]:
            loader = make_loader(CorruptedDataset(raw_test, fn, severity), bs, num_workers=nw)
            acc = evaluate(model, loader, device)
            degradation = clean_acc - acc
            print(f"{name:15s} severity={severity}  acc={acc:.4f}  degradation={degradation:+.4f}")
            rows.append({"corruption": name, "severity": severity, "accuracy": acc, "degradation": degradation})

    results = {"clean_accuracy": clean_acc, "rows": rows}
    results_dir = resolve_path(rcfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    table_path = resolve_path(rcfg["table_file"])
    with open(table_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {table_path}")


if __name__ == "__main__":
    main()
