import numpy as np

from calibration.temperature_scaling import softmax_np, expected_calibration_error, fit_temperature


def test_softmax_rows_sum_to_one():
    logits = np.random.default_rng(0).normal(size=(20, 7))
    probs = softmax_np(logits)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(20), atol=1e-6)


def test_ece_zero_when_perfectly_calibrated():
    # Confidence exactly matches empirical accuracy in every bin.
    rng = np.random.default_rng(0)
    n = 10000
    confidences = rng.uniform(0.5, 1.0, size=n)
    correct = rng.uniform(size=n) < confidences
    probs = np.zeros((n, 2))
    probs[:, 0] = confidences
    probs[:, 1] = 1 - confidences
    labels = np.where(correct, 0, 1)

    ece, bins = expected_calibration_error(probs, labels, n_bins=10)
    assert ece < 0.03
    assert len(bins) == 10


def test_ece_high_when_overconfident_and_wrong():
    n = 200
    probs = np.tile([0.99, 0.01], (n, 1))
    labels = np.ones(n, dtype=int)  # model always confidently predicts class 0, always wrong
    ece, _ = expected_calibration_error(probs, labels, n_bins=10)
    assert ece > 0.9


def test_fit_temperature_reduces_overconfidence_on_held_out_errors():
    rng = np.random.default_rng(1)
    n, num_classes = 500, 5
    labels = rng.integers(0, num_classes, size=n)
    # Overconfident logits: large margin toward a (partly wrong) fixed class.
    logits = rng.normal(scale=0.1, size=(n, num_classes))
    logits[np.arange(n), (labels + 1) % num_classes] += 8.0

    temperature = fit_temperature(logits, labels)
    assert temperature > 1.0  # softening is expected when NLL-optimal on miscalibrated logits


def test_ece_bin_counts_sum_to_total():
    rng = np.random.default_rng(2)
    n = 300
    probs = softmax_np(rng.normal(size=(n, 4)))
    labels = rng.integers(0, 4, size=n)
    _, bins = expected_calibration_error(probs, labels, n_bins=15)
    assert sum(b["count"] for b in bins) == n
