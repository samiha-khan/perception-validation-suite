"""OOD benchmark: GTSRB test (in-distribution) vs CIFAR-10 test (OOD negative set).

Fits the Mahalanobis detector on the GTSRB `calib` split (held out from
training, no augmentation), scores both detectors on GTSRB test (ID) and
CIFAR-10 test (OOD), and reports AUROC/AUPR for each detector plus ROC curve
points for the report.

Run from repo root:
    python -m ood.benchmark
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from models.config import load_config, resolve_path
from models.data import get_gtsrb_splits, get_cifar10_test, make_loader
from models.inference import extract_logits_features, pick_device
from models.model import load_checkpoint
from ood.detectors import softmax_entropy_score, MahalanobisDetector


def _auroc_aupr(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Subsample the curve for a compact report payload.
    idx = np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int)
    return {
        "auroc": float(auroc),
        "aupr": float(aupr),
        "roc_fpr": fpr[idx].tolist(),
        "roc_tpr": tpr[idx].tolist(),
    }


def main():
    cfg = load_config()
    device = pick_device()
    ocfg = cfg["ood"]
    data_root = resolve_path(cfg["data"]["root"])
    image_size = cfg["data"]["image_size"]
    num_classes = cfg["data"]["num_classes"]

    checkpoint_path = resolve_path(cfg["model"]["checkpoint"])
    model, ckpt = load_checkpoint(checkpoint_path, device=device)
    print(f"loaded checkpoint (val_acc={ckpt.get('val_acc'):.4f} epoch={ckpt.get('epoch')})")

    splits = get_gtsrb_splits(data_root, image_size, seed=cfg["seed"],
                               fractions=cfg["data"]["train_val_calib_split"])
    calib_loader = make_loader(splits["calib"], cfg["model"]["batch_size"], num_workers=cfg["model"]["num_workers"])
    id_loader = make_loader(splits["test"], cfg["model"]["batch_size"], num_workers=cfg["model"]["num_workers"])

    cifar = get_cifar10_test(data_root, image_size)
    ood_loader = make_loader(cifar, cfg["model"]["batch_size"], num_workers=cfg["model"]["num_workers"])

    print("extracting calib features (for Mahalanobis fit)...")
    calib_logits, calib_features, calib_labels = extract_logits_features(model, calib_loader, device)

    print("extracting ID (GTSRB test) features...")
    id_logits, id_features, _ = extract_logits_features(model, id_loader, device)

    print("extracting OOD (CIFAR-10 test) features...")
    ood_logits, ood_features, _ = extract_logits_features(model, ood_loader, device)

    mahalanobis = MahalanobisDetector().fit(calib_features, calib_labels, num_classes=num_classes)

    id_entropy = softmax_entropy_score(id_logits)
    ood_entropy = softmax_entropy_score(ood_logits)
    id_maha = mahalanobis.score(id_features)
    ood_maha = mahalanobis.score(ood_features)

    results = {
        "in_distribution": "GTSRB test",
        "out_of_distribution": "CIFAR-10 test",
        "n_id": int(len(id_entropy)),
        "n_ood": int(len(ood_entropy)),
        "detectors": {
            "softmax_entropy": _auroc_aupr(id_entropy, ood_entropy),
            "mahalanobis": _auroc_aupr(id_maha, ood_maha),
        },
    }
    avg_auroc = np.mean([results["detectors"][d]["auroc"] for d in results["detectors"]])
    results["avg_auroc"] = float(avg_auroc)

    results_dir = resolve_path(ocfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = resolve_path(ocfg["metrics_file"])
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"softmax entropy: AUROC={results['detectors']['softmax_entropy']['auroc']:.4f} "
          f"AUPR={results['detectors']['softmax_entropy']['aupr']:.4f}")
    print(f"mahalanobis:     AUROC={results['detectors']['mahalanobis']['auroc']:.4f} "
          f"AUPR={results['detectors']['mahalanobis']['aupr']:.4f}")
    print(f"avg AUROC={avg_auroc:.4f}  -> saved {metrics_path}")


if __name__ == "__main__":
    main()
