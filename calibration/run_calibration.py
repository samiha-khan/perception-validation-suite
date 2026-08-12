"""Fit temperature scaling + split conformal prediction, validate on held-out test data.

Run from repo root:
    python -m calibration.run_calibration
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.config import load_config, resolve_path
from models.data import get_gtsrb_splits, make_loader
from models.inference import extract_logits_features, pick_device
from models.model import load_checkpoint
from calibration.temperature_scaling import fit_temperature, softmax_np, expected_calibration_error
from calibration.conformal import calibrate_qhat, predict_sets, empirical_coverage, average_set_size


def plot_reliability_diagram(bins_raw, bins_calibrated, ece_raw, ece_calibrated, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, bins, title, ece in [
        (axes[0], bins_raw, "Before temperature scaling", ece_raw),
        (axes[1], bins_calibrated, "After temperature scaling", ece_calibrated),
    ]:
        centers = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bins]
        accs = [b["accuracy"] for b in bins]
        counts = [b["count"] for b in bins]
        widths = (centers[1] - centers[0]) if len(centers) > 1 else 0.05
        ax.bar(centers, accs, width=widths * 0.9, color="#4C72B0", edgecolor="black", label="accuracy")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
        ax.set_xlabel("confidence")
        ax.set_title(f"{title}\nECE={ece:.4f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    calib_loader = make_loader(splits["calib"], bs, num_workers=nw)
    test_loader = make_loader(splits["test"], bs, num_workers=nw)

    print("extracting logits (val/calib/test)...")
    val_logits, _, val_labels = extract_logits_features(model, val_loader, device)
    calib_logits, _, calib_labels = extract_logits_features(model, calib_loader, device)
    test_logits, _, test_labels = extract_logits_features(model, test_loader, device)

    # --- Temperature scaling, fit on val, validated on test ---
    temperature = fit_temperature(val_logits, val_labels)
    print(f"fitted temperature T={temperature:.4f}")

    test_probs_raw = softmax_np(test_logits)
    test_probs_cal = softmax_np(test_logits / temperature)
    ece_raw, bins_raw = expected_calibration_error(test_probs_raw, test_labels, n_bins=ccfg["ece_bins"])
    ece_cal, bins_cal = expected_calibration_error(test_probs_cal, test_labels, n_bins=ccfg["ece_bins"])
    print(f"test ECE raw={ece_raw:.4f}  calibrated={ece_cal:.4f}")

    plot_path = resolve_path(ccfg["reliability_plot"])
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_reliability_diagram(bins_raw, bins_cal, ece_raw, ece_cal, plot_path)

    # --- Split conformal prediction, calibrated on `calib`, validated on `test` ---
    alpha = ccfg["conformal_alpha"]
    calib_probs_cal = softmax_np(calib_logits / temperature)
    qhat = calibrate_qhat(calib_probs_cal, calib_labels, alpha=alpha)

    pred_sets = predict_sets(test_probs_cal, qhat)
    coverage = empirical_coverage(pred_sets, test_labels)
    avg_size = average_set_size(pred_sets)
    target_coverage = 1 - alpha
    print(f"conformal: target_coverage={target_coverage:.3f} empirical={coverage:.4f} "
          f"avg_set_size={avg_size:.2f} qhat={qhat:.4f}")

    results = {
        "temperature": temperature,
        "ece_bins": ccfg["ece_bins"],
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        "reliability_bins_raw": bins_raw,
        "reliability_bins_calibrated": bins_cal,
        "conformal": {
            "alpha": alpha,
            "target_coverage": target_coverage,
            "qhat": qhat,
            "empirical_coverage": coverage,
            "avg_set_size": avg_size,
            "n_test": int(len(test_labels)),
        },
    }

    results_dir = resolve_path(ccfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = resolve_path(ccfg["metrics_file"])
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    temp_path = resolve_path(ccfg["temperature_file"])
    with open(temp_path, "w") as f:
        json.dump({"temperature": temperature}, f, indent=2)

    print(f"saved {metrics_path}")
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
