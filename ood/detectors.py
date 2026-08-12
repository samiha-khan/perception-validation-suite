"""Out-of-distribution detectors.

Both detectors expose a `score(logits, features)` interface returning one
scalar per sample where *higher = more OOD*, so they plug directly into
AUROC/AUPR against a binary OOD label.
"""
from __future__ import annotations

import numpy as np


def softmax_entropy_score(logits: np.ndarray) -> np.ndarray:
    """Entropy of the softmax distribution. Higher entropy -> more uncertain -> more OOD."""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    eps = 1e-12
    entropy = -(probs * np.log(probs + eps)).sum(axis=1)
    return entropy


class MahalanobisDetector:
    """Class-conditional Gaussian OOD detector (Lee et al., 2018), tied covariance.

    Fits a per-class mean in feature space and a single shared (tied) covariance
    estimated from all training features after centering each sample by its own
    class mean. OOD score = min-over-classes squared Mahalanobis distance to the
    class means; higher distance -> further from every known class -> more OOD.
    A shrinkage term is added to the covariance for numerical stability since the
    feature dimension (512) can be comparable to or larger than per-class sample
    counts.
    """

    def __init__(self, shrinkage: float = 1e-3):
        self.shrinkage = shrinkage
        self.class_means_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray, num_classes: int) -> "MahalanobisDetector":
        feature_dim = features.shape[1]
        class_means = np.zeros((num_classes, feature_dim), dtype=np.float64)
        centered_chunks = []
        for c in range(num_classes):
            mask = labels == c
            class_feats = features[mask]
            mean_c = class_feats.mean(axis=0)
            class_means[c] = mean_c
            centered_chunks.append(class_feats - mean_c)

        centered = np.concatenate(centered_chunks, axis=0)
        cov = np.cov(centered, rowvar=False)
        cov = cov + self.shrinkage * np.eye(feature_dim)
        precision = np.linalg.inv(cov)

        self.class_means_ = class_means
        self.precision_ = precision
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        assert self.class_means_ is not None, "call fit() first"
        num_classes = self.class_means_.shape[0]
        # Per-class loop (not a full N x C x D tensor) to keep memory bounded.
        dists = np.empty((features.shape[0], num_classes), dtype=np.float64)
        for c in range(num_classes):
            diff = features - self.class_means_[c]
            dists[:, c] = np.einsum("nd,de,ne->n", diff, self.precision_, diff)
        return dists.min(axis=1)
