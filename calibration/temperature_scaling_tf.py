"""TensorFlow reimplementation of temperature scaling (Guo et al., 2017).

Fits the same single scalar T by minimizing NLL, via TF gradient descent
instead of PyTorch's LBFGS. Operates only on the val/test logits and labels
already exported to disk by calibration/export_logits.py, never on the
PyTorch model itself, to show the calibration math is framework agnostic.

Also cross-checks both fits against the true NLL-minimizing temperature
found by scipy (independent of either framework), since the two fits are
not guaranteed to land on the same point unless both actually converge.

Run from repo root (after calibration/export_logits.py has produced
calibration/results/logits.npz):
    python -m calibration.temperature_scaling_tf
"""
from __future__ import annotations

import json

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize_scalar

from models.config import load_config, resolve_path
from calibration.temperature_scaling import softmax_np, expected_calibration_error


def fit_temperature_tf(logits: np.ndarray, labels: np.ndarray, lr: float = 0.05, max_iter: int = 500) -> float:
    logits_t = tf.constant(logits, dtype=tf.float32)
    labels_t = tf.constant(labels, dtype=tf.int64)
    # Optimize in log space so temperature stays positive with no clamping needed mid-fit.
    log_temperature = tf.Variable(0.0, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for _ in range(max_iter):
        with tf.GradientTape() as tape:
            temperature = tf.exp(log_temperature)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_t, logits=logits_t / temperature)
            )
        grads = tape.gradient(loss, [log_temperature])
        optimizer.apply_gradients(zip(grads, [log_temperature]))

    temperature = float(tf.exp(log_temperature).numpy())
    return max(temperature, 1e-2)


def apply_temperature_tf(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits_t = tf.constant(logits, dtype=tf.float32)
    return tf.nn.softmax(logits_t / temperature).numpy()


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    scaled = logits.astype(np.float64) / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(scaled).sum(axis=1))
    logp_true = scaled[np.arange(len(labels)), labels] - logsumexp
    return float(-logp_true.mean())


def true_optimal_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """NLL-minimizing temperature found by direct scalar optimization (scipy), independent
    of either PyTorch's LBFGS fit or TensorFlow's gradient descent fit. Used only as a
    reference to check which fit actually reached the optimum, not as part of either fit."""
    result = minimize_scalar(lambda t: nll(logits, labels, t), bounds=(0.01, 10), method="bounded",
                              options={"xatol": 1e-10})
    return float(result.x)


def main():
    cfg = load_config()
    ccfg = cfg["calibration"]
    results_dir = resolve_path(ccfg["results_dir"])

    logits_path = results_dir / "logits.npz"
    d = np.load(logits_path)
    val_logits, val_labels = d["val_logits"], d["val_labels"]
    test_logits, test_labels = d["test_logits"], d["test_labels"]

    with open(resolve_path(ccfg["temperature_file"])) as f:
        temperature_pt = json.load(f)["temperature"]
    temperature_tf = fit_temperature_tf(val_logits, val_labels)
    temperature_scipy = true_optimal_temperature(val_logits, val_labels)

    print(f"PyTorch temperature   T={temperature_pt:.4f}  val_nll={nll(val_logits, val_labels, temperature_pt):.6f}")
    print(f"TensorFlow temperature T={temperature_tf:.4f}  val_nll={nll(val_logits, val_labels, temperature_tf):.6f}")
    print(f"scipy reference        T={temperature_scipy:.4f}  val_nll={nll(val_logits, val_labels, temperature_scipy):.6f}")

    test_probs_pt = softmax_np(test_logits / temperature_pt)
    test_probs_tf = apply_temperature_tf(test_logits, temperature_tf)
    ece_pt, _ = expected_calibration_error(test_probs_pt, test_labels, n_bins=ccfg["ece_bins"])
    ece_tf, _ = expected_calibration_error(test_probs_tf, test_labels, n_bins=ccfg["ece_bins"])
    print(f"PyTorch test ECE calibrated={ece_pt:.4f}")
    print(f"TensorFlow test ECE calibrated={ece_tf:.4f}")

    comparison = {
        "pytorch": {
            "temperature": temperature_pt,
            "val_nll": nll(val_logits, val_labels, temperature_pt),
            "test_ece_calibrated": ece_pt,
        },
        "tensorflow": {
            "temperature": temperature_tf,
            "val_nll": nll(val_logits, val_labels, temperature_tf),
            "test_ece_calibrated": ece_tf,
        },
        "scipy_reference": {
            "temperature": temperature_scipy,
            "val_nll": nll(val_logits, val_labels, temperature_scipy),
            "note": "True NLL-minimizing temperature via direct scalar optimization, "
                    "independent of either fit above. Used to check which fit actually "
                    "converged, not used by either fit.",
        },
    }
    out_path = results_dir / "tf_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
