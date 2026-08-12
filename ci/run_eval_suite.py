"""Orchestrates the full eval suite (OOD -> calibration -> robustness -> gate).

Trains a checkpoint first if none exists yet (fresh CI runner / cache miss).
Used by both local runs and the GitHub Actions workflow.

Run from repo root:
    python -m ci.run_eval_suite
"""
from __future__ import annotations

import sys

from models.config import load_config, resolve_path


def main() -> int:
    cfg = load_config()
    checkpoint_path = resolve_path(cfg["model"]["checkpoint"])

    if not checkpoint_path.exists():
        print(f"no checkpoint at {checkpoint_path}, training...")
        from models.train import main as train_main
        train_main()
    else:
        print(f"using existing checkpoint at {checkpoint_path}")

    from ood.benchmark import main as ood_main
    from calibration.run_calibration import main as calibration_main
    from robustness.run_robustness import main as robustness_main
    from ci.eval_gate import main as gate_main

    print("\n--- OOD benchmark ---")
    ood_main()
    print("\n--- Calibration ---")
    calibration_main()
    print("\n--- Robustness ---")
    robustness_main()
    print("\n--- Eval gate ---")
    return gate_main()


if __name__ == "__main__":
    sys.exit(main())
