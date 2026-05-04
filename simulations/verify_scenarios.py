"""
Scenario Verification Script
============================
Smoke-tests all 4 simulation scenarios from the survey paper.
Generates one large dataset per scenario, fits estimators with and without
survey design, and prints diagnostics. Not a full MC study - just sanity
checks that the DGP produces expected behavior.

Usage:
    python paper/simulations/verify_scenarios.py
"""

import sys
import warnings

# Ensure diff_diff is importable from the repo root
sys.path.insert(0, ".")

from diff_diff import CallawaySantAnna, DifferenceInDifferences, SurveyDesign
from diff_diff.prep_dgp import generate_survey_did_data


def header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def fmt(label, att, se):
    if se is not None:
        print(f"  {label:40s}  ATT={att:8.4f}  SE={se:8.4f}")
    else:
        print(f"  {label:40s}  ATT={att:8.4f}")


def fit_did(df, survey=None):
    """Fit DifferenceInDifferences using formula interface."""
    did = DifferenceInDifferences()
    result = did.fit(
        df,
        formula="outcome ~ treated * post",
        survey_design=survey,
    )
    return result.att, result.se


def fit_cs(df, survey=None, covariates=None, method="reg"):
    """Fit CallawaySantAnna."""
    cs = CallawaySantAnna(estimation_method=method)
    result = cs.fit(
        df,
        outcome="outcome",
        unit="unit",
        time="period",
        first_treat="first_treat",
        covariates=covariates,
        survey_design=survey,
    )
    return result.overall_att, result.overall_se


def scenario_1():
    """Scenario 1: Unconditional PT with complex survey.

    Expected: TSL SE > naive SE; both estimators unbiased.
    """
    header("Scenario 1: Unconditional PT + Complex Survey Design")

    df = generate_survey_did_data(
        n_units=2000,
        n_periods=8,
        icc=0.1,
        weight_cv=0.5,
        treatment_effect=2.0,
        return_true_population_att=True,
        seed=42,
    )
    truth = df.attrs["dgp_truth"]
    print(f"\n  True population ATT: {truth['population_att']:.4f}")
    print(f"  Kish DEFF:           {truth['deff_kish']:.4f}")
    print(f"  Realized ICC:        {truth['icc_realized']:.4f}")

    survey = SurveyDesign(
        strata="stratum", psu="psu", fpc="fpc", weights="weight"
    )

    # Need a 'post' column for DifferenceInDifferences
    min_cohort = df.loc[df["first_treat"] > 0, "first_treat"].min()
    df["post"] = (df["period"] >= min_cohort).astype(int)

    print("\n  --- DifferenceInDifferences ---")
    att_n, se_n = fit_did(df)
    fmt("Naive (no survey)", att_n, se_n)
    att_s, se_s = fit_did(df, survey=survey)
    fmt("Design-based (TSL)", att_s, se_s)
    if se_n and se_s:
        print(f"  SE ratio (design/naive): {se_s / se_n:.2f}")

    print("\n  --- CallawaySantAnna ---")
    att_n, se_n = fit_cs(df)
    fmt("Naive (no survey)", att_n, se_n)
    att_s, se_s = fit_cs(df, survey=survey)
    fmt("Design-based (TSL)", att_s, se_s)
    if se_n and se_s:
        print(f"  SE ratio (design/naive): {se_s / se_n:.2f}")

    print("\n  Expected: Design-based SEs > naive SEs")


def scenario_2():
    """Scenario 2: Informative sampling + heterogeneous TE.

    Expected: Weighted ATT closer to true ATT than unweighted.
    """
    header("Scenario 2: Informative Sampling + Heterogeneous TE")

    df = generate_survey_did_data(
        n_units=2000,
        n_periods=8,
        informative_sampling=True,
        heterogeneous_te_by_strata=True,
        treatment_effect=2.0,
        weight_variation="high",
        return_true_population_att=True,
        seed=42,
    )
    truth = df.attrs["dgp_truth"]
    print(f"\n  True population ATT: {truth['population_att']:.4f}")
    print(f"  Stratum effects:     {truth['base_stratum_effects']}")

    survey = SurveyDesign(
        strata="stratum", psu="psu", fpc="fpc", weights="weight"
    )

    print("\n  --- CallawaySantAnna ---")
    att_naive, se_naive = fit_cs(df)
    fmt("Unweighted (no survey)", att_naive, se_naive)
    att_wtd, se_wtd = fit_cs(df, survey=survey)
    fmt("Weighted (survey design)", att_wtd, se_wtd)

    true_att = truth["population_att"]
    print(f"\n  Bias (unweighted): {abs(att_naive - true_att):.4f}")
    print(f"  Bias (weighted):   {abs(att_wtd - true_att):.4f}")
    print("  Expected: Weighted estimate closer to truth")


def scenario_3():
    """Scenario 3: Panel vs repeated cross-section.

    Expected: Both produce valid estimates.
    """
    header("Scenario 3: Panel vs Repeated Cross-Section")

    for panel_mode in [True, False]:
        label = "Panel" if panel_mode else "Repeated cross-section"
        print(f"\n  --- {label} ---")

        df = generate_survey_did_data(
            n_units=2000,
            n_periods=8,
            panel=panel_mode,
            treatment_effect=2.0,
            return_true_population_att=True,
            seed=42,
        )
        truth = df.attrs["dgp_truth"]
        print(f"  True population ATT: {truth['population_att']:.4f}")

        survey = SurveyDesign(
            strata="stratum", psu="psu", fpc="fpc", weights="weight"
        )

        cs = CallawaySantAnna(panel=panel_mode)
        result = cs.fit(
            df,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            survey_design=survey,
        )
        fmt(f"CS ({label}, design-based)", result.overall_att, result.overall_se)

    print("\n  Expected: Both modes produce valid estimates near true ATT")


def scenario_4():
    """Scenario 4: Conditional parallel trends (headline result).

    Expected: CS(dr) with covariates recovers truth; CS(reg) without is biased.
    Runs 10 seeds to show consistency, not just a lucky draw.
    """
    header("Scenario 4: Conditional PT (Most Novel Claim)")

    import numpy as np

    # Parameters: moderate signal, reduced PSU noise so effect is visible
    cpt = 1.5
    psu_sd = 0.5
    n_seeds = 10

    print(f"\n  DGP: conditional_pt={cpt}, psu_re_sd={psu_sd}")
    print(f"  Running {n_seeds} seeds to assess consistency...\n")
    print(f"  {'seed':>5s}  {'true_ATT':>8s}  {'no_cov':>8s}  {'w_cov':>8s}"
          f"  {'bias_no':>8s}  {'bias_w':>8s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    biases_no = []
    biases_w = []

    for seed in range(n_seeds):
        df = generate_survey_did_data(
            n_units=2000,
            n_periods=8,
            add_covariates=True,
            conditional_pt=cpt,
            treatment_effect=2.0,
            psu_re_sd=psu_sd,
            return_true_population_att=True,
            seed=seed,
        )
        truth = df.attrs["dgp_truth"]["population_att"]
        survey = SurveyDesign(
            strata="stratum", psu="psu", fpc="fpc", weights="weight"
        )

        att_no, _ = fit_cs(df, survey=survey, method="reg")
        att_w, _ = fit_cs(df, survey=survey, covariates=["x1", "x2"], method="dr")

        b_no = att_no - truth
        b_w = att_w - truth
        biases_no.append(b_no)
        biases_w.append(b_w)
        print(f"  {seed:5d}  {truth:8.4f}  {att_no:8.4f}  {att_w:8.4f}"
              f"  {b_no:+8.4f}  {b_w:+8.4f}")

    mean_bias_no = np.mean(biases_no)
    mean_bias_w = np.mean(biases_w)
    rmse_no = np.sqrt(np.mean(np.array(biases_no) ** 2))
    rmse_w = np.sqrt(np.mean(np.array(biases_w) ** 2))

    print(f"\n  Summary across {n_seeds} seeds:")
    print(f"  Mean bias (no covariates):   {mean_bias_no:+.4f}")
    print(f"  Mean bias (with covariates): {mean_bias_w:+.4f}")
    print(f"  RMSE (no covariates):        {rmse_no:.4f}")
    print(f"  RMSE (with covariates):      {rmse_w:.4f}")
    print(f"  RMSE ratio (no-cov / w-cov): {rmse_no / rmse_w:.1f}x")
    print("\n  Expected: no-cov estimator consistently biased upward;")
    print("  w-cov estimator approximately unbiased (centered near 0)")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    print("Survey Paper - Simulation Scenario Verification")
    print("=" * 70)

    scenario_1()
    scenario_2()
    scenario_3()
    scenario_4()

    print(f"\n{'=' * 70}")
    print("  All scenarios complete.")
    print(f"{'=' * 70}")
