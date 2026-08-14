"""Drift detection primitives: the statistic used to flag a shifted batch,
and the blend schedule used to simulate a gradual distribution shift.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def ks_drift_score(reference: np.ndarray, batch: np.ndarray) -> tuple[float, float]:
    """Two-sample KS test of batch against a reference distribution.

    Chosen over population stability index (PSI): PSI bins the reference
    distribution into deciles and compares bin proportions, which gets noisy
    once each bin holds only a handful of points, exactly the small-batch
    regime here (tens of samples per batch). The KS test compares empirical
    CDFs directly, no binning, and stays well behaved at these sample sizes.
    Returns (D statistic, p-value); higher D means more drift.
    """
    result = ks_2samp(reference, batch)
    return float(result.statistic), float(result.pvalue)


def ramp_alpha(batch_index: int, clean_batches: int, num_batches: int) -> float:
    """Blend weight for injected drift: 0 before clean_batches, then rises
    linearly to 1.0 by the last batch."""
    if batch_index < clean_batches:
        return 0.0
    span = max(1, num_batches - 1 - clean_batches)
    return float(min(1.0, (batch_index - clean_batches) / span))
