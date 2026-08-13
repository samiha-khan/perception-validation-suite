import numpy as np

from calibration.temperature_scaling_tf import fit_temperature_tf, apply_temperature_tf, nll, true_optimal_temperature


def test_apply_temperature_tf_rows_sum_to_one():
    logits = np.random.default_rng(0).normal(size=(20, 7)).astype(np.float32)
    probs = apply_temperature_tf(logits, temperature=1.5)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(20), atol=1e-5)


def test_fit_temperature_tf_reduces_overconfidence_on_held_out_errors():
    rng = np.random.default_rng(1)
    n, num_classes = 500, 5
    labels = rng.integers(0, num_classes, size=n)
    # Overconfident logits: large margin toward a (partly wrong) fixed class.
    logits = rng.normal(scale=0.1, size=(n, num_classes)).astype(np.float32)
    logits[np.arange(n), (labels + 1) % num_classes] += 8.0

    temperature = fit_temperature_tf(logits, labels)
    assert temperature > 1.0  # softening is expected when NLL-optimal on miscalibrated logits


def test_fit_temperature_tf_matches_true_optimum():
    rng = np.random.default_rng(2)
    n, num_classes = 300, 4
    labels = rng.integers(0, num_classes, size=n)
    logits = rng.normal(size=(n, num_classes)).astype(np.float32)
    logits[np.arange(n), labels] += rng.uniform(1.0, 4.0, size=n)

    fitted = fit_temperature_tf(logits, labels)
    reference = true_optimal_temperature(logits, labels)
    assert abs(fitted - reference) < 0.01


def test_nll_lower_at_true_optimum_than_at_one():
    rng = np.random.default_rng(3)
    n, num_classes = 200, 5
    labels = rng.integers(0, num_classes, size=n)
    logits = rng.normal(scale=0.1, size=(n, num_classes)).astype(np.float32)
    logits[np.arange(n), (labels + 1) % num_classes] += 6.0

    reference = true_optimal_temperature(logits, labels)
    assert nll(logits, labels, reference) <= nll(logits, labels, 1.0)
