"""Generate a single self-contained HTML report for one eval run.

Thin presentation layer only: it reads the JSON/PNG artifacts already
produced by ood.benchmark, calibration.run_calibration, and
robustness.run_robustness, plots the OOD ROC curve from the saved
fpr/tpr points, and renders everything into one HTML file. No metrics are
computed here -- pass/fail status reuses ci.eval_gate.compute_checks so
there is exactly one place that implements the threshold logic.

Run from repo root (after the other stages have produced their results):
    python -m reports.generate_report
"""
from __future__ import annotations

import base64
import datetime
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader

from models.config import load_config, resolve_path
from ci.eval_gate import compute_checks, load_results

TEMPLATE_DIR = resolve_path("reports")


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _file_to_base64(path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _plot_roc(ood_results: dict) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, detector in ood_results["detectors"].items():
        ax.plot(detector["roc_fpr"], detector["roc_tpr"],
                label=f"{name} (AUROC={detector['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"OOD ROC -- {ood_results['in_distribution']} (ID) vs "
                 f"{ood_results['out_of_distribution']} (OOD)")
    ax.legend(loc="lower right", fontsize=8)
    return _fig_to_base64(fig)


def main():
    cfg = load_config()
    ood_results, calib_results = load_results(cfg)
    checks = compute_checks(cfg, ood_results, calib_results)

    with open(resolve_path(cfg["robustness"]["table_file"])) as f:
        robustness_results = json.load(f)

    roc_b64 = _plot_roc(ood_results)
    reliability_b64 = _file_to_base64(resolve_path(cfg["calibration"]["reliability_plot"]))

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("template.html")
    html = template.render(
        generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        overall_pass=all(c["passed"] for c in checks),
        checks=checks,
        ood_results=ood_results,
        calib_results=calib_results,
        robustness_results=robustness_results,
        roc_b64=roc_b64,
        reliability_b64=reliability_b64,
    )

    out_path = resolve_path(cfg["reports"]["output_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
