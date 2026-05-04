"""
Simulation Pipeline Validation
================================
Runs each cell with 50 reps and checks three tiers of correctness:
  Tier 1: Mechanical - no crashes, finite values, valid CSV
  Tier 2: Known-answer - results match expected statistical properties
  Tier 3: Statistical sanity - no degenerate cells

Any Tier 2 failure is a blocking signal: do NOT proceed to the full study.

Usage:
    python paper/simulations/sim_validate.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sim_config import (
    SCENARIOS,
    ESTIMATORS,
    VALIDATION_DIR,
    VALID_PAIRS,
    SAMPLE_SIZES,
    all_valid_cells,
    cell_id,
    cell_csv_path,
)
from sim_cell import run_cell

VALIDATION_REPS = 50
VALIDATION_N = 2000  # Use middle sample size for speed


def load_cell_results(cid, base_dir):
    """Load results CSV for a cell."""
    path = cell_csv_path(cid, base_dir=base_dir)
    if not path.exists():
        return None
    return pd.read_csv(path)


def tier1_mechanical(df, cid):
    """Tier 1: Mechanical checks. Returns (pass, messages)."""
    msgs = []
    ok = True

    if df is None or len(df) == 0:
        return False, [f"{cid}: No results produced"]

    # Check for NaN rate
    nan_rate = df["att_survey"].isna().mean()
    if nan_rate > 0.1:
        msgs.append(f"High NaN rate: {nan_rate:.0%} of reps failed")
        ok = False

    # Check SEs are positive where not NaN
    valid = df.dropna(subset=["se_survey"])
    if len(valid) > 0:
        if (valid["se_survey"] <= 0).any():
            msgs.append("Some SEs are <= 0")
            ok = False
    else:
        msgs.append("All SEs are NaN")
        ok = False

    # Check ATTs are finite
    valid_att = df.dropna(subset=["att_survey"])
    if len(valid_att) > 0:
        if not np.all(np.isfinite(valid_att["att_survey"])):
            msgs.append("Some ATTs are infinite")
            ok = False

    if ok:
        msgs.append("PASS")
    return ok, msgs


def tier2_known_answer(results_by_cell, scenario_id):
    """Tier 2: Known-answer checks for a scenario. Returns (pass, messages)."""
    msgs = []
    ok = True

    if scenario_id == "s1":
        # Design-based SEs should be larger than naive SEs on average
        for eid in VALID_PAIRS["s1"]:
            cid = cell_id("s1", eid, VALIDATION_N)
            df = results_by_cell.get(cid)
            if df is None or len(df) == 0:
                continue
            valid = df.dropna(subset=["se_naive", "se_survey"])
            if len(valid) < 5:
                continue

            mean_deff = valid["deff_ratio"].mean()
            if mean_deff <= 1.0:
                msgs.append(
                    f"s1/{eid}: mean DEFF={mean_deff:.2f} <= 1.0 "
                    f"(design SEs should exceed naive)"
                )
                ok = False

            # ATT should be approximately unbiased (except TWFE under staggered
            # treatment, which is known to be biased - that's the whole point)
            mean_bias = abs(valid["att_survey"].mean() - valid["true_att"].mean())
            if eid != "twfe" and mean_bias > 1.0:
                msgs.append(f"s1/{eid}: mean bias={mean_bias:.3f} > 1.0 (too biased)")
                ok = False
            elif eid == "twfe" and mean_bias < 0.1:
                msgs.append(
                    f"s1/twfe: bias={mean_bias:.3f} < 0.1 "
                    f"(TWFE should be biased under staggered treatment)"
                )
                ok = False

            # Naive coverage should be lower than survey coverage
            cov_naive = valid["covers_naive"].mean()
            cov_survey = valid["covers_survey"].mean()
            if cov_naive > cov_survey + 0.1:
                msgs.append(
                    f"s1/{eid}: naive coverage ({cov_naive:.2f}) > "
                    f"survey coverage ({cov_survey:.2f}) - unexpected"
                )
                # This is a soft check - warn but don't fail with only 50 reps
                # ok = False

    elif scenario_id == "s2":
        # Weighted ATT should be closer to truth than unweighted
        for eid in VALID_PAIRS["s2"]:
            cid = cell_id("s2", eid, VALIDATION_N)
            df = results_by_cell.get(cid)
            if df is None or len(df) == 0:
                continue
            valid = df.dropna(subset=["att_naive", "att_survey", "true_att"])
            if len(valid) < 5:
                continue

            bias_naive = abs(valid["att_naive"].mean() - valid["true_att"].mean())
            bias_survey = abs(valid["att_survey"].mean() - valid["true_att"].mean())
            # With informative sampling, the survey-weighted estimate should be
            # closer to truth. Allow some slack for 50-rep noise.
            if bias_survey > bias_naive + 0.3:
                msgs.append(
                    f"s2/{eid}: survey bias ({bias_survey:.3f}) > naive bias "
                    f"({bias_naive:.3f}) + 0.3 - weighting made things worse"
                )
                ok = False

    elif scenario_id == "s3":
        # Cross-section cells should produce valid estimates
        for eid in VALID_PAIRS["s3"]:
            cid = cell_id("s3", eid, VALIDATION_N)
            df = results_by_cell.get(cid)
            if df is None or len(df) == 0:
                msgs.append(f"s3/{eid}: no results")
                ok = False
                continue
            valid = df.dropna(subset=["att_survey"])
            if len(valid) < VALIDATION_REPS * 0.5:
                msgs.append(f"s3/{eid}: too many failures ({len(valid)}/{len(df)} valid)")
                ok = False

    elif scenario_id == "s4":
        # CS(reg) should be biased; CS(dr) should be less biased
        cid_reg = cell_id("s4", "cs_reg", VALIDATION_N)
        cid_dr = cell_id("s4", "cs_dr", VALIDATION_N)
        df_reg = results_by_cell.get(cid_reg)
        df_dr = results_by_cell.get(cid_dr)

        if df_reg is not None and df_dr is not None:
            valid_reg = df_reg.dropna(subset=["att_survey", "true_att"])
            valid_dr = df_dr.dropna(subset=["att_survey", "true_att"])

            if len(valid_reg) >= 5 and len(valid_dr) >= 5:
                bias_reg = abs(
                    valid_reg["att_survey"].mean() - valid_reg["true_att"].mean()
                )
                bias_dr = abs(
                    valid_dr["att_survey"].mean() - valid_dr["true_att"].mean()
                )

                # No-covariate estimator should show visible bias
                if bias_reg < 0.1:
                    msgs.append(
                        f"s4: CS(reg) bias={bias_reg:.3f} < 0.1 "
                        f"(expected visible bias from violated PT)"
                    )
                    ok = False

                # DR with covariates should have less bias than reg without
                if bias_dr >= bias_reg:
                    msgs.append(
                        f"s4: CS(dr) bias ({bias_dr:.3f}) >= CS(reg) bias "
                        f"({bias_reg:.3f}) - covariates didn't help"
                    )
                    ok = False
            else:
                msgs.append("s4: insufficient valid reps for comparison")
                ok = False
        else:
            msgs.append("s4: missing results for cs_reg or cs_dr")
            ok = False

    if ok and not msgs:
        msgs.append("PASS")
    elif ok:
        msgs.append("PASS (with notes)")
    return ok, msgs


def tier3_sanity(results_by_cell):
    """Tier 3: Statistical sanity across all cells. Returns (pass, messages)."""
    msgs = []
    ok = True

    # Cells where extreme coverage is expected (not a pipeline bug):
    # - TWFE under staggered treatment is biased -> can have 0% coverage
    # - s4/cs_reg is biased (no covariates, conditional PT violated) -> 0% expected
    # - s4/cs_dr may have very wide CIs -> 100% plausible with 50 reps
    KNOWN_BIASED = {"twfe", "s4_cs_reg"}

    for cid, df in results_by_cell.items():
        if df is None or len(df) == 0:
            continue
        valid = df.dropna(subset=["covers_survey"])
        if len(valid) < 5:
            continue

        cov = valid["covers_survey"].mean()
        is_known_biased = any(tag in cid for tag in KNOWN_BIASED)

        if cov == 0.0 and not is_known_biased:
            msgs.append(f"{cid}: 0% coverage (pipeline bug?)")
            ok = False
        if cov == 1.0 and not is_known_biased:
            # 100% with 50 reps is suspicious but not impossible
            msgs.append(f"{cid}: 100% coverage (note: may be OK with wide CIs)")

        # Estimates should have positive variance
        att_std = valid["att_survey"].std()
        if att_std < 1e-10:
            msgs.append(f"{cid}: zero variance in ATT estimates (all identical?)")
            ok = False

    if ok:
        msgs.append("PASS")
    return ok, msgs


def main():
    print("=" * 60)
    print("  Simulation Pipeline Validation")
    print(f"  {VALIDATION_REPS} reps per cell, n_units={VALIDATION_N}")
    print("=" * 60)

    # Run all cells at validation scale
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    cells_to_run = [
        (sid, eid, VALIDATION_N)
        for sid, estimators in VALID_PAIRS.items()
        for eid in estimators
    ]

    print(f"\nRunning {len(cells_to_run)} cells...")
    results_by_cell = {}

    for i, (sid, eid, n) in enumerate(cells_to_run):
        cid = cell_id(sid, eid, n)
        label = f"{SCENARIOS[sid]['label']} / {ESTIMATORS[eid]['label']}"
        print(f"\n  [{i + 1}/{len(cells_to_run)}] {cid} ({label})")

        result = run_cell(
            sid, eid, n, VALIDATION_REPS, base_dir=VALIDATION_DIR, print_every=0
        )
        print(
            f"    {result['n_complete']} reps, "
            f"{result['n_failed']} failed, "
            f"{result['elapsed_s']:.1f}s"
        )
        results_by_cell[cid] = load_cell_results(cid, VALIDATION_DIR)

    # === Tier 1: Mechanical ===
    print(f"\n{'=' * 60}")
    print("  TIER 1: Mechanical Checks")
    print(f"{'=' * 60}")
    tier1_pass = True
    for cid, df in results_by_cell.items():
        passed, msgs = tier1_mechanical(df, cid)
        status = "PASS" if passed else "FAIL"
        print(f"  {cid}: {status}")
        for m in msgs:
            if m != "PASS":
                print(f"    {m}")
        if not passed:
            tier1_pass = False

    # === Tier 2: Known-answer ===
    print(f"\n{'=' * 60}")
    print("  TIER 2: Known-Answer Checks")
    print(f"{'=' * 60}")
    tier2_pass = True
    for sid in SCENARIOS:
        passed, msgs = tier2_known_answer(results_by_cell, sid)
        status = "PASS" if passed else "FAIL"
        print(f"  {sid} ({SCENARIOS[sid]['label']}): {status}")
        for m in msgs:
            if m not in ("PASS", "PASS (with notes)"):
                print(f"    {m}")
        if not passed:
            tier2_pass = False

    # === Tier 3: Statistical sanity ===
    print(f"\n{'=' * 60}")
    print("  TIER 3: Statistical Sanity")
    print(f"{'=' * 60}")
    tier3_ok, tier3_msgs = tier3_sanity(results_by_cell)
    print(f"  {'PASS' if tier3_ok else 'FAIL'}")
    for m in tier3_msgs:
        if m != "PASS":
            print(f"    {m}")

    # === Summary ===
    print(f"\n{'=' * 60}")
    print("  VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Tier 1 (Mechanical):    {'PASS' if tier1_pass else 'FAIL'}")
    print(f"  Tier 2 (Known-answer):  {'PASS' if tier2_pass else '** FAIL - DO NOT PROCEED **'}")
    print(f"  Tier 3 (Sanity):        {'PASS' if tier3_ok else 'FAIL'}")

    all_pass = tier1_pass and tier2_pass and tier3_ok
    print(f"\n  Overall: {'ALL CLEAR - safe to run full study' if all_pass else 'ISSUES FOUND - investigate before proceeding'}")
    print(f"{'=' * 60}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
