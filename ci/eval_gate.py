"""CI gate: fail the build if OOD AUROC, calibration ECE, or conformal coverage
breach the thresholds defined in config.yaml.

Reads the results files produced by `ood.benchmark` and
`calibration.run_calibration` -- run those first.

Run from repo root:
    python -m ci.eval_gate
"""
from __future__ import annotations

import json
import sys

from models.config import load_config, resolve_path


def compute_checks(cfg: dict, ood_results: dict, calib_results: dict) -> list[dict]:
    """Threshold checks shared by the CI gate and the HTML report (single
    source of truth so the report never re-derives pass/fail on its own)."""
    thresholds = cfg["thresholds"]

    avg_auroc = ood_results["avg_auroc"]
    ece = calib_results["ece_calibrated"]
    empirical_coverage = calib_results["conformal"]["empirical_coverage"]
    target_coverage = calib_results["conformal"]["target_coverage"]
    coverage_gap = abs(empirical_coverage - target_coverage)

    return [
        {"name": "OOD avg AUROC", "value": avg_auroc, "op": ">=", "threshold": thresholds["ood_auroc_min"],
         "passed": avg_auroc >= thresholds["ood_auroc_min"]},
        {"name": "Calibrated ECE", "value": ece, "op": "<=", "threshold": thresholds["ece_max"],
         "passed": ece <= thresholds["ece_max"]},
        {"name": "Conformal coverage gap", "value": coverage_gap, "op": "<=",
         "threshold": thresholds["conformal_coverage_tolerance"],
         "passed": coverage_gap <= thresholds["conformal_coverage_tolerance"]},
    ]


def load_results(cfg: dict) -> tuple[dict, dict]:
    ood_path = resolve_path(cfg["ood"]["metrics_file"])
    calib_path = resolve_path(cfg["calibration"]["metrics_file"])
    with open(ood_path) as f:
        ood_results = json.load(f)
    with open(calib_path) as f:
        calib_results = json.load(f)
    return ood_results, calib_results


def main() -> int:
    cfg = load_config()
    ood_results, calib_results = load_results(cfg)
    checks = compute_checks(cfg, ood_results, calib_results)

    print("=" * 60)
    print("CI eval gate")
    print("=" * 60)
    all_passed = True
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] {check['name']}: {check['value']:.4f} {check['op']} {check['threshold']:.4f}")
        all_passed = all_passed and check["passed"]

    print("=" * 60)
    if all_passed:
        print("All checks passed.")
        return 0
    print("One or more checks FAILED -- failing build.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
