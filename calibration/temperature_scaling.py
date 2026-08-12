"""Temperature scaling (Guo et al., 2017) + Expected Calibration Error.

A single learned scalar T rescales logits (logits / T) to better calibrate
softmax confidences without changing argmax predictions. T is fit on the
held-out `val` split by minimizing NLL with LBFGS.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def fit_temperature(logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, max_iter: int = 100) -> float:
    logits_t = torch.from_numpy(logits).float()
    labels_t = torch.from_numpy(labels).long()

    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(scaler(logits_t), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(scaler.temperature.detach().clamp(min=1e-2).item())


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict]]:
    """Returns (ECE, per-bin data for a reliability diagram)."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_data = []
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            bin_acc, bin_conf = 0.0, 0.0
        else:
            bin_acc = float(accuracies[in_bin].mean())
            bin_conf = float(confidences[in_bin].mean())
            ece += (count / n) * abs(bin_acc - bin_conf)
        bin_data.append({
            "bin_lower": float(lo), "bin_upper": float(hi),
            "accuracy": bin_acc, "confidence": bin_conf, "count": count,
        })
    return float(ece), bin_data
