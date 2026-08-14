import numpy as np

from monitoring.drift import ks_drift_score, ramp_alpha


def test_ks_drift_score_low_for_same_distribution():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=500)
    batch = rng.normal(size=500)
    statistic, pvalue = ks_drift_score(reference, batch)
    assert statistic < 0.15
    assert pvalue > 0.05


def test_ks_drift_score_high_for_shifted_distribution():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=500)
    shifted = rng.normal(loc=5.0, size=500)
    statistic, pvalue = ks_drift_score(reference, shifted)
    assert statistic > 0.9
    assert pvalue < 0.001


def test_ramp_alpha_zero_before_clean_batches_end():
    assert ramp_alpha(0, clean_batches=8, num_batches=25) == 0.0
    assert ramp_alpha(7, clean_batches=8, num_batches=25) == 0.0


def test_ramp_alpha_reaches_one_on_last_batch():
    assert ramp_alpha(24, clean_batches=8, num_batches=25) == 1.0


def test_ramp_alpha_monotonically_increases_after_onset():
    values = [ramp_alpha(i, clean_batches=8, num_batches=25) for i in range(8, 25)]
    assert all(b >= a for a, b in zip(values, values[1:]))
