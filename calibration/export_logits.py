"""Exports val/test logits + labels from the trained PyTorch checkpoint to disk.

Bridges the PyTorch model to framework-agnostic calibration tooling (e.g.
temperature_scaling_tf.py): everything downstream of this script reads only
the saved arrays and never touches the model or PyTorch again.

Run from repo root:
    python -m calibration.export_logits
"""
from __future__ import annotations

import numpy as np

from models.config import load_config, resolve_path
from models.data import get_gtsrb_splits, make_loader
from models.inference import extract_logits_features, pick_device
from models.model import load_checkpoint


def main():
    cfg = load_config()
    device = pick_device()
    ccfg = cfg["calibration"]

    data_root = resolve_path(cfg["data"]["root"])
    image_size = cfg["data"]["image_size"]
    checkpoint_path = resolve_path(cfg["model"]["checkpoint"])
    model, ckpt = load_checkpoint(checkpoint_path, device=device)
    print(f"loaded checkpoint (val_acc={ckpt.get('val_acc'):.4f})")

    splits = get_gtsrb_splits(data_root, image_size, seed=cfg["seed"],
                               fractions=cfg["data"]["train_val_calib_split"])
    bs, nw = cfg["model"]["batch_size"], cfg["model"]["num_workers"]
    val_loader = make_loader(splits["val"], bs, num_workers=nw)
    test_loader = make_loader(splits["test"], bs, num_workers=nw)

    print("extracting logits (val/test)...")
    val_logits, _, val_labels = extract_logits_features(model, val_loader, device)
    test_logits, _, test_labels = extract_logits_features(model, test_loader, device)

    results_dir = resolve_path(ccfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    logits_path = results_dir / "logits.npz"
    np.savez(
        logits_path,
        val_logits=val_logits, val_labels=val_labels,
        test_logits=test_logits, test_labels=test_labels,
    )
    print(f"saved {logits_path}")


if __name__ == "__main__":
    main()
