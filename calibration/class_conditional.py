"""Separate investigation: does class conditional (Mondrian) conformal prediction
recover coverage closer to target on the official GTSRB test set.

Fits one qhat per class on the calib split instead of a single global qhat, using
the same temperature scaled probabilities as calibration/run_calibration.py, then
compares empirical coverage on the same official test set. Writes its own results
file and does not modify calibration/run_calibration.py, calibration/conformal.py,
or their results.

Run from repo root:
    python -m calibration.class_conditional
"""
from __future__ import annotations

import json

import numpy as np

from models.config import load_config, resolve_path
from models.data import get_gtsrb_splits, make_loader
from models.inference import extract_logits_features, pick_device
from models.model import load_checkpoint
from calibration.temperature_scaling import fit_temperature, softmax_np
from calibration.conformal import calibrate_qhat, predict_sets, empirical_coverage, average_set_size


def calibrate_qhat_per_class(calib_probs: np.ndarray, calib_labels: np.ndarray,
                              num_classes: int, alpha: float) -> np.ndarray:
    fallback = calibrate_qhat(calib_probs, calib_labels, alpha)
    qhats = np.full(num_classes, fallback)
    for c in range(num_classes):
        mask = calib_labels == c
        if mask.sum() == 0:
            continue
        qhats[c] = calibrate_qhat(calib_probs[mask], calib_labels[mask], alpha)
    return qhats


def predict_sets_per_class(probs: np.ndarray, qhats: np.ndarray) -> np.ndarray:
    return probs >= (1.0 - qhats[None, :])


def per_class_coverage(pred_sets: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    coverage = np.full(num_classes, np.nan)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        coverage[c] = pred_sets[mask, c].mean()
    return coverage


def main():
    cfg = load_config()
    device = pick_device()
    ccfg = cfg["calibration"]
    num_classes = cfg["data"]["num_classes"]

    data_root = resolve_path(cfg["data"]["root"])
    checkpoint_path = resolve_path(cfg["model"]["checkpoint"])
    model, ckpt = load_checkpoint(checkpoint_path, device=device)
    print(f"loaded checkpoint (val_acc={ckpt.get('val_acc'):.4f})")

    splits = get_gtsrb_splits(data_root, cfg["data"]["image_size"], seed=cfg["seed"],
                               fractions=cfg["data"]["train_val_calib_split"])
    bs, nw = cfg["model"]["batch_size"], cfg["model"]["num_workers"]
    val_logits, _, val_labels = extract_logits_features(model, make_loader(splits["val"], bs, num_workers=nw), device)
    calib_logits, _, calib_labels = extract_logits_features(model, make_loader(splits["calib"], bs, num_workers=nw), device)
    test_logits, _, test_labels = extract_logits_features(model, make_loader(splits["test"], bs, num_workers=nw), device)

    temperature = fit_temperature(val_logits, val_labels)
    calib_probs = softmax_np(calib_logits / temperature)
    test_probs = softmax_np(test_logits / temperature)

    alpha = ccfg["conformal_alpha"]
    target_coverage = 1 - alpha

    global_qhat = calibrate_qhat(calib_probs, calib_labels, alpha)
    global_sets = predict_sets(test_probs, global_qhat)
    global_coverage = empirical_coverage(global_sets, test_labels)
    global_avg_size = average_set_size(global_sets)

    class_qhats = calibrate_qhat_per_class(calib_probs, calib_labels, num_classes, alpha)
    class_sets = predict_sets_per_class(test_probs, class_qhats)
    class_coverage = empirical_coverage(class_sets, test_labels)
    class_avg_size = average_set_size(class_sets)

    global_gap = abs(global_coverage - target_coverage)
    class_gap = abs(class_coverage - target_coverage)
    improves = class_gap < global_gap

    print(f"target coverage: {target_coverage:.4f}")
    print(f"global qhat:  coverage={global_coverage:.4f}  avg_set_size={global_avg_size:.3f}  gap={global_gap:.4f}")
    print(f"class qhat:   coverage={class_coverage:.4f}  avg_set_size={class_avg_size:.3f}  gap={class_gap:.4f}")
    print(f"class conditional {'improves' if improves else 'does not improve'} coverage toward target")

    results = {
        "alpha": alpha,
        "target_coverage": target_coverage,
        "global": {
            "qhat": global_qhat, "empirical_coverage": global_coverage,
            "avg_set_size": global_avg_size, "gap_to_target": global_gap,
        },
        "class_conditional": {
            "empirical_coverage": class_coverage, "avg_set_size": class_avg_size,
            "gap_to_target": class_gap, "improves_over_global": bool(improves),
        },
        "per_class_coverage": per_class_coverage(class_sets, test_labels, num_classes).tolist(),
        "per_class_calib_count": np.bincount(calib_labels, minlength=num_classes).tolist(),
        "n_test": int(len(test_labels)),
    }

    results_dir = resolve_path(ccfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "class_conditional_coverage.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
