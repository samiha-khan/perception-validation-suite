import numpy as np

from calibration.temperature_scaling import softmax_np
from calibration.conformal import (
    nonconformity_scores, calibrate_qhat, predict_sets, empirical_coverage, average_set_size,
)


def _synthetic_probs(n, num_classes, seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, num_classes))
    labels = rng.integers(0, num_classes, size=n)
    return softmax_np(logits), labels


def test_nonconformity_score_is_one_minus_true_class_prob():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    labels = np.array([0, 2])
    scores = nonconformity_scores(probs, labels)
    np.testing.assert_allclose(scores, [0.3, 0.2])


def test_predict_sets_always_contain_true_label_at_qhat_from_calibration_scores():
    probs, labels = _synthetic_probs(200, 5, seed=0)
    qhat = calibrate_qhat(probs, labels, alpha=0.1)
    # qhat is the (corrected) quantile of nonconformity on this exact set,
    # so almost all calibration points must fall inside their own prediction set.
    pred_sets = predict_sets(probs, qhat)
    coverage = empirical_coverage(pred_sets, labels)
    assert coverage >= 0.9


def test_empirical_coverage_matches_target_on_exchangeable_data():
    rng_seed_calib, rng_seed_test = 10, 11
    num_classes = 6
    calib_probs, calib_labels = _synthetic_probs(4000, num_classes, rng_seed_calib)
    test_probs, test_labels = _synthetic_probs(4000, num_classes, rng_seed_test)

    alpha = 0.1
    qhat = calibrate_qhat(calib_probs, calib_labels, alpha=alpha)
    pred_sets = predict_sets(test_probs, qhat)
    coverage = empirical_coverage(pred_sets, test_labels)

    # Marginal coverage guarantee: empirical coverage >= 1 - alpha (up to finite-sample noise).
    assert coverage >= (1 - alpha) - 0.03


def test_average_set_size_nonnegative_and_bounded_by_num_classes():
    probs, labels = _synthetic_probs(300, 5, seed=3)
    qhat = calibrate_qhat(probs, labels, alpha=0.1)
    pred_sets = predict_sets(probs, qhat)
    avg_size = average_set_size(pred_sets)
    assert 0 <= avg_size <= 5


def test_larger_alpha_gives_smaller_or_equal_set_size():
    probs, labels = _synthetic_probs(1000, 5, seed=4)
    qhat_loose = calibrate_qhat(probs, labels, alpha=0.3)  # lower target coverage
    qhat_tight = calibrate_qhat(probs, labels, alpha=0.05)  # higher target coverage
    size_loose = average_set_size(predict_sets(probs, qhat_loose))
    size_tight = average_set_size(predict_sets(probs, qhat_tight))
    assert size_loose <= size_tight
