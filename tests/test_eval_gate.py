from ci.eval_gate import compute_checks


def _cfg(auroc_min=0.85, ece_max=0.05, coverage_tol=0.05):
    return {"thresholds": {
        "ood_auroc_min": auroc_min, "ece_max": ece_max, "conformal_coverage_tolerance": coverage_tol,
    }}


def _ood(avg_auroc):
    return {"avg_auroc": avg_auroc}


def _calib(ece, empirical_coverage, target_coverage=0.9):
    return {"ece_calibrated": ece,
            "conformal": {"empirical_coverage": empirical_coverage, "target_coverage": target_coverage}}


def test_all_checks_pass_when_metrics_meet_thresholds():
    checks = compute_checks(_cfg(), _ood(0.9), _calib(ece=0.02, empirical_coverage=0.91))
    assert all(c["passed"] for c in checks)


def test_auroc_below_threshold_fails_only_that_check():
    checks = compute_checks(_cfg(), _ood(0.7), _calib(ece=0.02, empirical_coverage=0.91))
    by_name = {c["name"]: c["passed"] for c in checks}
    assert by_name["OOD avg AUROC"] is False
    assert by_name["Calibrated ECE"] is True
    assert by_name["Conformal coverage gap"] is True


def test_ece_above_threshold_fails():
    checks = compute_checks(_cfg(), _ood(0.9), _calib(ece=0.2, empirical_coverage=0.91))
    by_name = {c["name"]: c["passed"] for c in checks}
    assert by_name["Calibrated ECE"] is False


def test_coverage_gap_above_tolerance_fails():
    # target 0.9, empirical 0.5 -> gap 0.4 >> tolerance
    checks = compute_checks(_cfg(), _ood(0.9), _calib(ece=0.02, empirical_coverage=0.5))
    by_name = {c["name"]: c["passed"] for c in checks}
    assert by_name["Conformal coverage gap"] is False


def test_boundary_values_pass_inclusive():
    # AUROC/ECE compared directly against the same float literal as the
    # threshold (no subtraction), so equality is exact. The coverage gap is a
    # subtraction of two floats near 0.05, which isn't exactly representable
    # in binary -- use a gap clearly under tolerance there instead of testing
    # an exact float boundary.
    checks = compute_checks(_cfg(auroc_min=0.85, ece_max=0.05, coverage_tol=0.05),
                             _ood(0.85), _calib(ece=0.05, empirical_coverage=0.87, target_coverage=0.9))
    assert all(c["passed"] for c in checks)
