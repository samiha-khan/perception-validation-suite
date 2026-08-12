import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ood.detectors import softmax_entropy_score, MahalanobisDetector


def test_entropy_uniform_is_maximal():
    num_classes = 10
    uniform_logits = np.zeros((5, num_classes))
    entropy = softmax_entropy_score(uniform_logits)
    np.testing.assert_allclose(entropy, np.log(num_classes), atol=1e-6)


def test_entropy_confident_prediction_is_near_zero():
    logits = np.full((3, 5), -50.0)
    logits[:, 0] = 50.0  # one class dominates -> near-deterministic softmax
    entropy = softmax_entropy_score(logits)
    assert np.all(entropy < 1e-3)


def test_entropy_confident_lower_than_uniform():
    num_classes = 10
    uniform = softmax_entropy_score(np.zeros((1, num_classes)))
    confident_logits = np.zeros((1, num_classes))
    confident_logits[0, 0] = 20.0
    confident = softmax_entropy_score(confident_logits)
    assert confident[0] < uniform[0]


def test_mahalanobis_separates_far_cluster():
    rng = np.random.default_rng(0)
    num_classes, feature_dim, n_per_class = 4, 8, 200

    class_means = rng.normal(scale=1.0, size=(num_classes, feature_dim))
    train_feats, train_labels = [], []
    for c in range(num_classes):
        train_feats.append(rng.normal(loc=class_means[c], scale=0.1, size=(n_per_class, feature_dim)))
        train_labels.append(np.full(n_per_class, c))
    train_feats = np.concatenate(train_feats)
    train_labels = np.concatenate(train_labels)

    detector = MahalanobisDetector().fit(train_feats, train_labels, num_classes=num_classes)

    # ID: fresh samples from the same class-conditional distributions.
    id_feats = rng.normal(loc=class_means[0], scale=0.1, size=(100, feature_dim))
    # OOD: far away from every class mean.
    ood_feats = class_means.mean(axis=0) + rng.normal(scale=1.0, size=(100, feature_dim)) + 50.0

    id_scores = detector.score(id_feats)
    ood_scores = detector.score(ood_feats)

    y_true = np.concatenate([np.zeros(100), np.ones(100)])
    y_score = np.concatenate([id_scores, ood_scores])
    auroc = roc_auc_score(y_true, y_score)
    assert auroc > 0.99
    assert ood_scores.mean() > id_scores.mean()


def test_mahalanobis_requires_fit_before_score():
    detector = MahalanobisDetector()
    with pytest.raises(AssertionError):
        detector.score(np.zeros((2, 4)))
