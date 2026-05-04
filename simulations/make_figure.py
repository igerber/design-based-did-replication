"""Generate coverage figure for Scenario 1 (main paper Figure 1)."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# Embed TrueType (Type 42) fonts in the PDF rather than bitmap Type 3 fonts.
# arXiv requires outline fonts; matplotlib's default for PDF/PS is Type 3
# unless explicitly overridden.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def load_scenario1_coverage():
    """Load coverage data for Scenario 1 (modern estimators only, not TWFE)."""
    data = []
    estimators = {
        "cs_reg": "CS (reg)",
        "cs_dr": "CS (DR)",
        "sa": "Sun-Abraham",
    }
    for eid, label in estimators.items():
        for n in [500, 2000, 8000]:
            path = RESULTS_DIR / f"s1_{eid}_n{n}.csv"
            if path.exists():
                df = pd.read_csv(path)
                valid = df.dropna(
                    subset=["covers_naive", "covers_cluster_psu", "covers_survey"]
                )
                data.append({
                    "estimator": label,
                    "n": n,
                    "cov_naive": valid["covers_naive"].mean() * 100,
                    "cov_cluster": valid["covers_cluster_psu"].mean() * 100,
                    "cov_survey": valid["covers_survey"].mean() * 100,
                })
    return pd.DataFrame(data)


def make_coverage_figure(df):
    """Create a three-panel coverage figure: HC1 / cluster-PSU / full design."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True)
    ax1, ax2, ax3 = axes

    estimators = df["estimator"].unique()
    colors = {"CS (reg)": "#1f77b4", "CS (DR)": "#ff7f0e", "Sun-Abraham": "#2ca02c"}
    markers = {"CS (reg)": "o", "CS (DR)": "s", "Sun-Abraham": "^"}
    x_positions = np.array([500, 2000, 8000])

    panels = [
        (ax1, "(a) HC1 standard errors", "cov_naive", True),
        (ax2, "(b) Weighted point + PSU cluster", "cov_cluster", False),
        (ax3, "(c) Full design-based standard errors", "cov_survey", False),
    ]

    for ax, title, col, show_ylabel in panels:
        ax.set_title(title, fontsize=12, fontweight="bold")
        for est in estimators:
            subset = df[df["estimator"] == est]
            ax.plot(
                subset["n"], subset[col],
                marker=markers[est], color=colors[est],
                linewidth=2, markersize=8, label=est,
            )
        ax.axhline(y=95, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(600, 96.5, "Nominal 95%", fontsize=9, alpha=0.6)
        ax.set_xlabel("Sample size ($n$)", fontsize=11)
        if show_ylabel:
            ax.set_ylabel("Coverage (%)", fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_xscale("log")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["500", "2,000", "8,000"])
        ax.tick_params(axis="both", labelsize=10)

    ax1.legend(fontsize=9, loc="lower left")

    plt.tight_layout()
    return fig


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_scenario1_coverage()
    fig = make_coverage_figure(df)

    # Save as PDF (for LaTeX \includegraphics)
    out_path = FIGURES_DIR / "coverage_s1.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Figure saved to {out_path}")

    # Also save PNG for quick preview
    png_path = FIGURES_DIR / "coverage_s1.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    print(f"Preview saved to {png_path}")


if __name__ == "__main__":
    main()
