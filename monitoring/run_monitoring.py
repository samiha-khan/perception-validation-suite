"""Simulates a stream of production batches drifting away from the training
distribution, and flags batches where a KS drift statistic crosses a
threshold.

The first `clean_batches` batches are unmodified GTSRB test images. From
there, a corruption (reused from robustness/corruptions.py) is blended in
with linearly increasing weight, reaching full severity by the last batch,
so drift eases in rather than switching on all at once. Each batch's
prediction-confidence distribution (softmax entropy, reused from
ood/detectors.py) is compared against the val split's distribution via a
KS test.

Run from repo root:
    python -m monitoring.run_monitoring
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from models.config import load_config, resolve_path
from models.data import get_gtsrb_splits, get_gtsrb_test_raw, make_loader, normalize_transform
from models.inference import extract_logits_features, pick_device
from models.model import load_checkpoint
from ood.detectors import softmax_entropy_score
from robustness.corruptions import CORRUPTIONS
from monitoring.drift import ks_drift_score, ramp_alpha


def build_batch(raw_dataset, indices, corruption_fn, severity: int, alpha: float, normalize) -> torch.Tensor:
    tensors = []
    for idx in indices:
        img, _ = raw_dataset[idx]
        arr = np.asarray(img).astype(np.float32) / 255.0
        if alpha > 0.0:
            corrupted = corruption_fn(arr, severity)
            arr = (1 - alpha) * arr + alpha * corrupted
        arr_uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        tensors.append(normalize(Image.fromarray(arr_uint8)))
    return torch.stack(tensors)


@torch.no_grad()
def batch_logits(model, batch: torch.Tensor, device) -> np.ndarray:
    model.eval()
    return model(batch.to(device)).cpu().numpy()


def plot_drift(rows, drift_onset_batch, first_flagged_batch, threshold, out_path):
    batches = [r["batch"] for r in rows]
    stats = [r["ks_statistic"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(batches, stats, marker="o", color="#4C72B0", label="KS drift statistic")
    ax.axhline(threshold, linestyle="--", color="gray", label="flag threshold")
    ax.axvline(drift_onset_batch, linestyle=":", color="#C44E52", label="drift injected")
    if first_flagged_batch is not None:
        ax.axvline(first_flagged_batch, linestyle=":", color="#55A868", label="first flagged")
    ax.set_xlabel("batch index")
    ax.set_ylabel("KS statistic")
    ax.set_title("Drift score over batch index")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    device = pick_device()
    mcfg = cfg["monitoring"]

    data_root = resolve_path(cfg["data"]["root"])
    image_size = cfg["data"]["image_size"]
    checkpoint_path = resolve_path(cfg["model"]["checkpoint"])
    model, ckpt = load_checkpoint(checkpoint_path, device=device)
    print(f"loaded checkpoint (val_acc={ckpt.get('val_acc'):.4f})")

    splits = get_gtsrb_splits(data_root, image_size, seed=cfg["seed"],
                               fractions=cfg["data"]["train_val_calib_split"])
    val_loader = make_loader(splits["val"], cfg["model"]["batch_size"], num_workers=cfg["model"]["num_workers"])
    val_logits, _, _ = extract_logits_features(model, val_loader, device)
    reference_scores = softmax_entropy_score(val_logits)
    print(f"reference distribution: {len(reference_scores)} val samples")

    raw_test = get_gtsrb_test_raw(data_root, image_size)
    normalize = normalize_transform()

    num_batches = mcfg["num_batches"]
    batch_size = mcfg["batch_size"]
    clean_batches = mcfg["clean_batches"]
    corruption_fn = CORRUPTIONS[mcfg["drift_corruption"]]
    drift_severity = mcfg["drift_severity"]
    threshold = mcfg["drift_ks_threshold"]

    assert num_batches * batch_size <= len(raw_test), "stream longer than available test images"
    rng = np.random.default_rng(cfg["seed"])
    order = rng.permutation(len(raw_test))

    rows = []
    first_flagged_batch = None
    for i in range(num_batches):
        idx = order[i * batch_size:(i + 1) * batch_size]
        alpha = ramp_alpha(i, clean_batches, num_batches)
        batch = build_batch(raw_test, idx, corruption_fn, drift_severity, alpha, normalize)
        logits = batch_logits(model, batch, device)
        batch_scores = softmax_entropy_score(logits)
        statistic, pvalue = ks_drift_score(reference_scores, batch_scores)
        flagged = statistic > threshold
        if flagged and first_flagged_batch is None:
            first_flagged_batch = i
        print(f"batch {i:02d}  alpha={alpha:.2f}  ks={statistic:.4f}  p={pvalue:.4g}  flagged={flagged}")
        rows.append({
            "batch": i, "alpha": alpha,
            "ks_statistic": statistic, "ks_pvalue": pvalue, "flagged": flagged,
        })

    print(f"drift injected starting batch {clean_batches}, first flagged batch {first_flagged_batch}")

    results_dir = resolve_path(mcfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "drift_onset_batch": clean_batches,
        "first_flagged_batch": first_flagged_batch,
        "threshold": threshold,
        "corruption": mcfg["drift_corruption"],
        "rows": rows,
    }
    results_path = resolve_path(mcfg["results_file"])
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {results_path}")

    plot_path = resolve_path(mcfg["plot_file"])
    plot_drift(rows, clean_batches, first_flagged_batch, threshold, plot_path)
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
