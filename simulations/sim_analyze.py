"""
Simulation Results Analysis
=============================
Loads all completed cell CSVs, computes summary statistics, and generates
tables for the paper. Outputs both console-friendly and LaTeX-ready formats.

Usage:
    python paper/simulations/sim_analyze.py
    python paper/simulations/sim_analyze.py --latex   # Also write LaTeX table files
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sim_config import (
    SCENARIOS,
    ESTIMATORS,
    RESULTS_DIR,
    SAMPLE_SIZES,
    VALID_PAIRS,
    cell_id,
    cell_csv_path,
)

TABLES_DIR = Path(__file__).parent.parent / "tables"


def load_all_results():
    """Load all cell CSVs into a dict keyed by cell_id."""
    results = {}
    for sid, estimators in VALID_PAIRS.items():
        for eid in estimators:
            for n in SAMPLE_SIZES:
                cid = cell_id(sid, eid, n)
                path = cell_csv_path(cid)
                if path.exists():
                    df = pd.read_csv(path)
                    df["scenario"] = sid
                    df["estimator"] = eid
                    df["n_units"] = n
                    results[cid] = df
    return results


def compute_cell_summary(df):
    """Compute summary statistics for one cell."""
    valid = df.dropna(subset=["att_survey", "se_survey"])
    n_valid = len(valid)
    n_total = len(df)

    if n_valid == 0:
        return {"n_valid": 0, "n_total": n_total}

    true_att = valid["true_att"].mean()
    bias_naive = valid["att_naive"].mean() - true_att
    bias_survey = valid["att_survey"].mean() - true_att
    rmse_naive = np.sqrt(((valid["att_naive"] - true_att) ** 2).mean())
    rmse_survey = np.sqrt(((valid["att_survey"] - true_att) ** 2).mean())
    coverage_naive = valid["covers_naive"].mean() * 100
    coverage_survey = valid["covers_survey"].mean() * 100
    mean_se_naive = valid["se_naive"].mean()
    mean_se_survey = valid["se_survey"].mean()
    mean_deff = valid["deff_ratio"].mean()
    median_deff = valid["deff_ratio"].median()

    # Rejection rate at 5%: fraction where CI excludes 0
    # (i.e., the estimator detects a nonzero effect)
    reject_naive = ((valid["ci_lo_naive"] > 0) | (valid["ci_hi_naive"] < 0)).mean() * 100
    reject_survey = ((valid["ci_lo_survey"] > 0) | (valid["ci_hi_survey"] < 0)).mean() * 100

    summary = {
        "n_valid": n_valid,
        "n_total": n_total,
        "true_att": true_att,
        "bias_naive": bias_naive,
        "bias_survey": bias_survey,
        "rmse_naive": rmse_naive,
        "rmse_survey": rmse_survey,
        "coverage_naive": coverage_naive,
        "coverage_survey": coverage_survey,
        "mean_se_naive": mean_se_naive,
        "mean_se_survey": mean_se_survey,
        "mean_deff": mean_deff,
        "median_deff": median_deff,
        "reject_naive": reject_naive,
        "reject_survey": reject_survey,
    }

    # Cluster-by-PSU column (added in v2 of the schema). Optional so older CSVs
    # without these columns still produce summaries (just with NaN cluster stats).
    if "covers_cluster_psu" in valid.columns:
        cluster_valid = valid.dropna(subset=["att_cluster_psu", "se_cluster_psu"])
        if len(cluster_valid) > 0:
            summary["bias_cluster_psu"] = (
                cluster_valid["att_cluster_psu"].mean() - true_att
            )
            summary["rmse_cluster_psu"] = np.sqrt(
                ((cluster_valid["att_cluster_psu"] - true_att) ** 2).mean()
            )
            summary["coverage_cluster_psu"] = (
                cluster_valid["covers_cluster_psu"].mean() * 100
            )
            summary["mean_se_cluster_psu"] = cluster_valid["se_cluster_psu"].mean()
        else:
            summary["coverage_cluster_psu"] = np.nan

    return summary


def print_scenario_table(scenario_id, results, summaries):
    """Print a console-friendly table for one scenario."""
    label = SCENARIOS[scenario_id]["label"]
    print(f"\n{'=' * 90}")
    print(f"  Scenario {scenario_id}: {label}")
    print(f"{'=' * 90}")

    header = (
        "{:<12s} {:>6s}  {:>8s} {:>8s}  {:>7s} {:>7s}  {:>6s} {:>6s}  {:>6s}"
    )
    print(header.format(
        "Estimator", "n", "Bias_n", "Bias_s",
        "Cov_n", "Cov_s", "DEFF", "RMSE_n", "RMSE_s"
    ))
    print("-" * 90)

    for eid in VALID_PAIRS[scenario_id]:
        for n in SAMPLE_SIZES:
            cid = cell_id(scenario_id, eid, n)
            s = summaries.get(cid)
            if s is None or s["n_valid"] == 0:
                continue
            elabel = ESTIMATORS[eid]["label"]
            row = "{:<12s} {:>6d}  {:>+8.4f} {:>+8.4f}  {:>6.1f}% {:>6.1f}%  {:>6.2f} {:>6.4f} {:>7.4f}"
            print(row.format(
                elabel, n,
                s["bias_naive"], s["bias_survey"],
                s["coverage_naive"], s["coverage_survey"],
                s["mean_deff"],
                s["rmse_naive"], s["rmse_survey"],
            ))


def generate_latex_table(scenario_id, summaries, out_path):
    """Write a LaTeX table for one scenario.

    Uses \small font and compact headers to fit within 6.5in text width.
    Columns are grouped: Bias (naive/design), Coverage (naive/design),
    DEFF, RMSE (naive/design).
    """
    label = SCENARIOS[scenario_id]["label"]

    # Detect whether cluster-by-PSU column is available across this scenario's cells
    has_cluster = any(
        "coverage_cluster_psu" in summaries.get(cell_id(scenario_id, eid, n), {})
        for eid in VALID_PAIRS[scenario_id]
        for n in SAMPLE_SIZES
    )

    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{Simulation results: " + label + ".}")
    lines.append("\\label{tab:sim-" + scenario_id + "}")
    if has_cluster:
        # 11 columns: estimator, n, bias_unwt, bias_wtd, cov_hc1, cov_cluster,
        # cov_design, deff, rmse_unwt, rmse_wtd. Bias/RMSE columns label by
        # the point-estimate weighting (Unwt = unweighted, Wtd = survey-weighted)
        # since Cluster and Design share the same weighted point estimate.
        lines.append("\\begin{tabular}{@{}llrrrrrrrr@{}}")
        lines.append("\\toprule")
        lines.append(
            " & & \\multicolumn{2}{c}{Bias} & "
            "\\multicolumn{3}{c}{Coverage (\\%)} & "
            " & \\multicolumn{2}{c}{RMSE} \\\\"
        )
        lines.append(
            "\\cmidrule(lr){3-4} \\cmidrule(lr){5-7} \\cmidrule(lr){9-10}"
        )
        lines.append(
            "{Estimator} & {$n$} & {Unwt} & {Wtd} & "
            "{HC1} & {Cluster} & {Design} & {DEFF} & {Unwt} & {Wtd} \\\\"
        )
    else:
        lines.append("\\begin{tabular}{@{}llrrrrrrr@{}}")
        lines.append("\\toprule")
        lines.append(
            " & & \\multicolumn{2}{c}{Bias} & "
            "\\multicolumn{2}{c}{Coverage (\\%)} & "
            " & \\multicolumn{2}{c}{RMSE} \\\\"
        )
        lines.append("\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){8-9}")
        lines.append(
            "{Estimator} & {$n$} & {Naive} & {Design} & "
            "{Naive} & {Design} & {DEFF} & {Naive} & {Design} \\\\"
        )
    lines.append("\\midrule")

    for eid in VALID_PAIRS[scenario_id]:
        elabel = ESTIMATORS[eid]["label"]
        first = True
        for n in SAMPLE_SIZES:
            cid = cell_id(scenario_id, eid, n)
            s = summaries.get(cid)
            if s is None or s["n_valid"] == 0:
                continue
            name = elabel if first else ""
            first = False
            if has_cluster:
                cov_cluster = s.get("coverage_cluster_psu", float("nan"))
                cluster_str = (
                    f"{cov_cluster:.1f}" if cov_cluster == cov_cluster else "---"
                )
                lines.append(
                    "{} & {:,} & {:+.3f} & {:+.3f} & {:.1f} & {} & {:.1f} & "
                    "{:.1f} & {:.3f} & {:.3f} \\\\".format(
                        name, n,
                        s["bias_naive"], s["bias_survey"],
                        s["coverage_naive"], cluster_str, s["coverage_survey"],
                        s["mean_deff"],
                        s["rmse_naive"], s["rmse_survey"],
                    )
                )
            else:
                lines.append(
                    "{} & {:,} & {:+.3f} & {:+.3f} & {:.1f} & {:.1f} & "
                    "{:.1f} & {:.3f} & {:.3f} \\\\".format(
                        name, n,
                        s["bias_naive"], s["bias_survey"],
                        s["coverage_naive"], s["coverage_survey"],
                        s["mean_deff"],
                        s["rmse_naive"], s["rmse_survey"],
                    )
                )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if has_cluster:
        lines.append(
            "\\\\[2pt]\\footnotesize "
            "\\emph{Notes:} HC1 uses the unweighted point estimate (Unwt) "
            "with heteroskedasticity-robust standard errors. Cluster and "
            "Design both use the survey-weighted point estimate (Wtd); "
            "Cluster applies PSU-level clustering only, while Design "
            "applies full Taylor-series linearization with strata, PSU, "
            "and FPC."
        )
    lines.append("\\end{table}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def print_headline_summary(summaries):
    """Print the key takeaways across all scenarios."""
    print(f"\n{'=' * 90}")
    print("  HEADLINE SUMMARY")
    print(f"{'=' * 90}")

    # S1: SE inflation
    print("\n  Scenario 1 - SE inflation (design vs naive):")
    for eid in ["cs_reg", "cs_dr", "sa"]:
        deffs = []
        for n in SAMPLE_SIZES:
            cid = cell_id("s1", eid, n)
            s = summaries.get(cid)
            if s and s["n_valid"] > 0:
                deffs.append(s["mean_deff"])
        if deffs:
            elabel = ESTIMATORS[eid]["label"]
            print("    {}: DEFF = {:.1f} - {:.1f}x across sample sizes".format(
                elabel, min(deffs), max(deffs)
            ))

    # S1: Coverage improvement
    print("\n  Scenario 1 - Coverage at nominal 95% (n=2000):")
    for eid in ["cs_reg", "cs_dr", "sa", "twfe"]:
        cid = cell_id("s1", eid, 2000)
        s = summaries.get(cid)
        if s and s["n_valid"] > 0:
            elabel = ESTIMATORS[eid]["label"]
            print("    {}: naive={:.1f}%, design={:.1f}%".format(
                elabel, s["coverage_naive"], s["coverage_survey"]
            ))

    # S4: Conditional PT bias reduction
    print("\n  Scenario 4 - Conditional PT (headline result, n=2000):")
    s_reg = summaries.get(cell_id("s4", "cs_reg", 2000))
    s_dr = summaries.get(cell_id("s4", "cs_dr", 2000))
    if s_reg and s_dr and s_reg["n_valid"] > 0 and s_dr["n_valid"] > 0:
        print("    CS(reg) no covariates:  bias={:+.4f}, RMSE={:.4f}, coverage={:.1f}%".format(
            s_reg["bias_survey"], s_reg["rmse_survey"], s_reg["coverage_survey"]
        ))
        print("    CS(DR) with covariates: bias={:+.4f}, RMSE={:.4f}, coverage={:.1f}%".format(
            s_dr["bias_survey"], s_dr["rmse_survey"], s_dr["coverage_survey"]
        ))
        rmse_ratio = s_reg["rmse_survey"] / s_dr["rmse_survey"]
        print("    RMSE ratio: {:.1f}x".format(rmse_ratio))

    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze simulation results")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables")
    args = parser.parse_args()

    print("Loading results...")
    results = load_all_results()
    print(f"  {len(results)} cells loaded")

    total_reps = sum(len(df) for df in results.values())
    print(f"  {total_reps:,} total replications")

    # Compute summaries
    summaries = {}
    for cid, df in results.items():
        summaries[cid] = compute_cell_summary(df)

    # Print per-scenario tables
    for sid in SCENARIOS:
        print_scenario_table(sid, results, summaries)

    # Print headlines
    print_headline_summary(summaries)

    # Write summary CSV
    rows = []
    for cid, s in summaries.items():
        row = {"cell_id": cid, **s}
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary CSV written to: {csv_path}")

    # LaTeX tables
    if args.latex:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        for sid in SCENARIOS:
            out = TABLES_DIR / f"sim_{sid}.tex"
            generate_latex_table(sid, summaries, out)
            print(f"LaTeX table written to: {out}")


if __name__ == "__main__":
    main()
