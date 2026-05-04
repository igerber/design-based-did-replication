"""
Single Cell Execution
======================
Runs one (scenario, estimator, n_units) cell for N reps with checkpointing.
Results are appended to a CSV one row at a time for crash safety.
"""

import csv
import hashlib
import math
import time
import warnings

from sim_config import (
    SCENARIOS,
    ESTIMATORS,
    cell_csv_path,
    cell_id,
    make_estimator,
    get_result_values,
)

# CSV columns
COLUMNS = [
    "rep",
    "seed",
    "true_att",
    "att_naive",
    "se_naive",
    "ci_lo_naive",
    "ci_hi_naive",
    "att_cluster_psu",
    "se_cluster_psu",
    "ci_lo_cluster_psu",
    "ci_hi_cluster_psu",
    "att_survey",
    "se_survey",
    "ci_lo_survey",
    "ci_hi_survey",
    "covers_naive",
    "covers_cluster_psu",
    "covers_survey",
    "deff_ratio",
    "elapsed_s",
]


def count_existing_reps(csv_path):
    """Count completed reps in an existing CSV (excludes header)."""
    if not csv_path.exists():
        return 0
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        return sum(1 for _ in reader)


def run_cell(
    scenario_id,
    estimator_id,
    n_units,
    n_reps,
    base_dir=None,
    print_every=100,
):
    """Run a single simulation cell with checkpointing.

    Returns dict with summary stats (n_complete, n_failed, elapsed_s).
    """
    from diff_diff import SurveyDesign
    from diff_diff.prep_dgp import generate_survey_did_data

    cid = cell_id(scenario_id, estimator_id, n_units)
    csv_path = cell_csv_path(cid, base_dir=base_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: count existing rows
    start_rep = count_existing_reps(csv_path)
    if start_rep >= n_reps:
        return {"cell_id": cid, "n_complete": start_rep, "n_failed": 0, "elapsed_s": 0}

    scenario = SCENARIOS[scenario_id]
    est_cfg = ESTIMATORS[estimator_id]
    dgp_kwargs = {**scenario["dgp_kwargs"], "n_units": n_units}

    # For cross-section DGPs, CS estimator needs panel=False in constructor
    est_init_override = {}
    if dgp_kwargs.get("panel") is False and est_cfg["class"] == "CallawaySantAnna":
        est_init_override["panel"] = False

    # Seed offset: unique per cell to avoid correlation across cells.
    # Use hashlib.md5 (not Python's built-in hash) for cross-process determinism;
    # built-in hash() is randomized per interpreter unless PYTHONHASHSEED is set.
    seed_offset = int(hashlib.md5(cid.encode()).hexdigest(), 16) % 1_000_000

    n_failed = 0
    cell_start = time.time()

    # Open CSV in append mode
    is_new = start_rep == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()

        for rep in range(start_rep, n_reps):
            seed = rep * 1000 + seed_offset
            rep_start = time.time()
            row = {"rep": rep, "seed": seed}

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)

                    # Generate data
                    df = generate_survey_did_data(seed=seed, **dgp_kwargs)
                    true_att = df.attrs["dgp_truth"]["population_att"]
                    row["true_att"] = true_att

                    survey_full = SurveyDesign(
                        strata="stratum", psu="psu", fpc="fpc", weights="weight"
                    )
                    # PSU-only design: survey weights for the point estimate
                    # plus PSU-level clustering for the variance, but no strata
                    # or FPC. This is the practitioner's `cluster=psu` + survey
                    # weights pattern. Same point estimate as the full design;
                    # difference is purely in the variance estimator.
                    survey_psu_only = SurveyDesign(
                        psu="psu", weights="weight"
                    )

                    # Fit WITHOUT survey design (naive HC1)
                    est_naive = make_estimator(estimator_id, **est_init_override)
                    result_naive = est_naive.fit(df, **est_cfg["fit_kwargs"])
                    att_n, se_n, ci_lo_n, ci_hi_n = get_result_values(
                        result_naive, estimator_id
                    )
                    row["att_naive"] = att_n
                    row["se_naive"] = se_n
                    row["ci_lo_naive"] = ci_lo_n
                    row["ci_hi_naive"] = ci_hi_n
                    row["covers_naive"] = int(ci_lo_n <= true_att <= ci_hi_n)

                    # Fit with weighted point + PSU-clustered design (no
                    # strata/FPC). Variance is PSU-only.
                    est_cluster = make_estimator(estimator_id, **est_init_override)
                    result_cluster = est_cluster.fit(
                        df,
                        **est_cfg["fit_kwargs"],
                        survey_design=survey_psu_only,
                    )
                    att_c, se_c, ci_lo_c, ci_hi_c = get_result_values(
                        result_cluster, estimator_id
                    )
                    row["att_cluster_psu"] = att_c
                    row["se_cluster_psu"] = se_c
                    row["ci_lo_cluster_psu"] = ci_lo_c
                    row["ci_hi_cluster_psu"] = ci_hi_c
                    row["covers_cluster_psu"] = int(ci_lo_c <= true_att <= ci_hi_c)

                    # Fit WITH full survey design
                    est_survey = make_estimator(estimator_id, **est_init_override)
                    result_survey = est_survey.fit(
                        df, **est_cfg["fit_kwargs"], survey_design=survey_full
                    )
                    att_s, se_s, ci_lo_s, ci_hi_s = get_result_values(
                        result_survey, estimator_id
                    )
                    row["att_survey"] = att_s
                    row["se_survey"] = se_s
                    row["ci_lo_survey"] = ci_lo_s
                    row["ci_hi_survey"] = ci_hi_s
                    row["covers_survey"] = int(ci_lo_s <= true_att <= ci_hi_s)

                    # DEFF ratio
                    if se_n > 0 and math.isfinite(se_n) and math.isfinite(se_s):
                        row["deff_ratio"] = (se_s / se_n) ** 2
                    else:
                        row["deff_ratio"] = math.nan

            except Exception as e:
                # Record NaN for failed reps
                n_failed += 1
                row.update(
                    {
                        "true_att": math.nan,
                        "att_naive": math.nan,
                        "se_naive": math.nan,
                        "ci_lo_naive": math.nan,
                        "ci_hi_naive": math.nan,
                        "att_cluster_psu": math.nan,
                        "se_cluster_psu": math.nan,
                        "ci_lo_cluster_psu": math.nan,
                        "ci_hi_cluster_psu": math.nan,
                        "att_survey": math.nan,
                        "se_survey": math.nan,
                        "ci_lo_survey": math.nan,
                        "ci_hi_survey": math.nan,
                        "covers_naive": math.nan,
                        "covers_cluster_psu": math.nan,
                        "covers_survey": math.nan,
                        "deff_ratio": math.nan,
                    }
                )
                if n_failed <= 3:
                    print(f"    WARNING [{cid}] rep {rep} failed: {e}")

            row["elapsed_s"] = round(time.time() - rep_start, 4)
            writer.writerow(row)
            f.flush()

            if print_every and (rep + 1) % print_every == 0:
                elapsed = time.time() - cell_start
                rate = (rep + 1 - start_rep) / elapsed
                eta = (n_reps - rep - 1) / rate if rate > 0 else 0
                print(
                    f"    [{cid}] {rep + 1}/{n_reps} reps "
                    f"({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining, "
                    f"{n_failed} failed)"
                )

    elapsed_total = time.time() - cell_start
    failure_rate = n_failed / (n_reps - start_rep) if n_reps > start_rep else 0
    if failure_rate > 0.1:
        print(
            f"    WARNING [{cid}] high failure rate: "
            f"{n_failed}/{n_reps - start_rep} = {failure_rate:.1%}"
        )

    return {
        "cell_id": cid,
        "n_complete": n_reps,
        "n_failed": n_failed,
        "elapsed_s": round(elapsed_total, 2),
    }
