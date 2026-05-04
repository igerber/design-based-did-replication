"""Generate NHANES empirical illustration table for the paper."""

import json
from pathlib import Path

# The golden NHANES analysis JSON ships with this repo at data/nhanes/.
# It is a frozen copy of diff-diff's benchmarks/data/real/nhanes_realdata_golden.json
# (see data/README.md for provenance).
GOLDEN_PATH = (
    Path(__file__).parent.parent / "data" / "nhanes" / "nhanes_realdata_golden.json"
)
OUT_PATH = Path(__file__).parent.parent / "tables" / "nhanes.tex"


def main():
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    # Also compute naive (no survey design)
    import pandas as pd
    from diff_diff import DifferenceInDifferences

    df = pd.DataFrame(golden["_data"])
    did = DifferenceInDifferences()

    import warnings
    warnings.filterwarnings("ignore")
    result_naive = did.fit(df, "outcome", "treated", "post")

    rows = [
        {
            "label": "Naive (no weights, no design)",
            "att": result_naive.att,
            "se": result_naive.se,
            "ci_lo": result_naive.conf_int[0],
            "ci_hi": result_naive.conf_int[1],
            "df": len(df) - 4,
        },
        {
            "label": "Weights only (no clustering)",
            "att": golden["b3_weights_only"]["att"],
            "se": golden["b3_weights_only"]["se"],
            "ci_lo": golden["b3_weights_only"]["ci_lower"],
            "ci_hi": golden["b3_weights_only"]["ci_upper"],
            "df": golden["b3_weights_only"]["df"],
        },
        {
            "label": "Full design (strata + PSU + weights)",
            "att": golden["b1_strata_psu_weights"]["att"],
            "se": golden["b1_strata_psu_weights"]["se"],
            "ci_lo": golden["b1_strata_psu_weights"]["ci_lower"],
            "ci_hi": golden["b1_strata_psu_weights"]["ci_upper"],
            "df": golden["b1_strata_psu_weights"]["df"],
        },
        {
            "label": "Full design + covariates",
            "att": golden["b2_covariates"]["att"],
            "se": golden["b2_covariates"]["se"],
            "ci_lo": golden["b2_covariates"]["ci_lower"],
            "ci_hi": golden["b2_covariates"]["ci_upper"],
            "df": golden["b2_covariates"]["df"],
        },
    ]

    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(
        "\\caption{ACA dependent coverage provision: DiD estimates of the effect "
        "on health insurance coverage using NHANES data, 2007--2008 vs.\\ "
        "2015--2016. Treatment group: ages 19--25; control group: ages 27--34.}"
    )
    lines.append("\\label{tab:nhanes}")
    lines.append("\\begin{tabular}{@{}lrrrr@{}}")
    lines.append("\\toprule")
    lines.append(
        "Specification & ATT & SE & 95\\% CI & df \\\\"
    )
    lines.append("\\midrule")

    for r in rows:
        ci_str = "[{:.3f}, {:.3f}]".format(r["ci_lo"], r["ci_hi"])
        lines.append(
            "{} & {:.3f} & {:.3f} & {} & {} \\\\".format(
                r["label"], r["att"], r["se"], ci_str, r["df"]
            )
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\\\[2pt]\\footnotesize "
        "\\emph{Notes:} The Naive and Weights-only specifications report "
        "residual degrees of freedom from the underlying linear regression "
        "($n - p$). The Full design rows report survey degrees of freedom "
        "$\\sum_h n_h - H$ (sampled PSUs minus strata; see Section~"
        "\\ref{sec:theory}), which give the appropriate $t$-distribution "
        "for design-based inference."
    )
    lines.append("\\end{table}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Table written to {OUT_PATH}")

    # Also print console version
    print()
    print("{:<40s} {:>7s} {:>7s} {:>20s} {:>6s}".format(
        "Specification", "ATT", "SE", "95% CI", "df"
    ))
    print("-" * 85)
    for r in rows:
        print("{:<40s} {:>7.3f} {:>7.3f} [{:>6.3f}, {:>6.3f}] {:>6}".format(
            r["label"], r["att"], r["se"], r["ci_lo"], r["ci_hi"], r["df"]
        ))


if __name__ == "__main__":
    main()
