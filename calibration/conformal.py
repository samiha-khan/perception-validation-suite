"""Split conformal prediction (Vovk et al.) with the softmax nonconformity score.

Nonconformity score: s(x, y) = 1 - softmax(x)[y]. The calibration threshold
qhat is the (1 - alpha)-quantile (with the standard finite-sample correction)
of scores on a held-out calibration split. At test time, the prediction set
for x is every class y with softmax(x)[y] >= 1 - qhat. Under exchangeability
this gives marginal coverage >= 1 - alpha; we validate that empirically here.
"""
from __future__ import annotations

import numpy as np


def nonconformity_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return 1.0 - probs[np.arange(len(labels)), labels]


def calibrate_qhat(calib_probs: np.ndarray, calib_labels: np.ndarray, alpha: float) -> float:
    scores = nonconformity_scores(calib_probs, calib_labels)
    n = len(scores)
    level = np.clip(np.ceil((n + 1) * (1 - alpha)) / n, 0.0, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def predict_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean membership matrix (N, C): True where class is in the prediction set."""
    return probs >= (1.0 - qhat)


def empirical_coverage(pred_sets: np.ndarray, labels: np.ndarray) -> float:
    in_set = pred_sets[np.arange(len(labels)), labels]
    return float(in_set.mean())


def average_set_size(pred_sets: np.ndarray) -> float:
    return float(pred_sets.sum(axis=1).mean())
