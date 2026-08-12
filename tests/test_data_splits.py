from models.data import _split_indices


def test_split_indices_disjoint_and_cover_all():
    n = 1000
    train, val, calib = _split_indices(n, [0.8, 0.1, 0.1], seed=42)
    assert len(train) + len(val) + len(calib) == n
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(calib)
    assert set(val).isdisjoint(calib)
    assert set(train) | set(val) | set(calib) == set(range(n))


def test_split_indices_approx_matches_fractions():
    n = 10000
    train, val, calib = _split_indices(n, [0.8, 0.1, 0.1], seed=0)
    assert abs(len(train) / n - 0.8) < 0.01
    assert abs(len(val) / n - 0.1) < 0.01
    assert abs(len(calib) / n - 0.1) < 0.01


def test_split_indices_deterministic_given_seed():
    a = _split_indices(500, [0.8, 0.1, 0.1], seed=7)
    b = _split_indices(500, [0.8, 0.1, 0.1], seed=7)
    assert a == b


def test_split_indices_differs_across_seeds():
    a = _split_indices(500, [0.8, 0.1, 0.1], seed=1)
    b = _split_indices(500, [0.8, 0.1, 0.1], seed=2)
    assert a != b
