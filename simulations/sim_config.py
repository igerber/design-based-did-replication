"""
Simulation Study Configuration
===============================
Grid definition for the Monte Carlo study: scenarios, estimators, sample sizes.
"""

from pathlib import Path

# --- Paths ---
RESULTS_DIR = Path(__file__).parent / "results"
VALIDATION_DIR = RESULTS_DIR / "validation"

# --- Replication settings ---
N_REPS = 2000
N_PERIODS = 8
TREATMENT_EFFECT = 2.0

# --- Sample sizes ---
SAMPLE_SIZES = [500, 2000, 8000]

# --- Scenarios ---
# Each maps scenario_id -> DGP kwargs for generate_survey_did_data()
SCENARIOS = {
    "s1": {
        "label": "Unconditional PT + complex survey",
        "dgp_kwargs": dict(
            n_periods=N_PERIODS,
            treatment_effect=TREATMENT_EFFECT,
            icc=0.1,
            weight_cv=0.5,
            add_covariates=True,
            return_true_population_att=True,
        ),
    },
    "s2": {
        "label": "Informative sampling + heterogeneous TE",
        "dgp_kwargs": dict(
            n_periods=N_PERIODS,
            treatment_effect=TREATMENT_EFFECT,
            informative_sampling=True,
            heterogeneous_te_by_strata=True,
            weight_variation="high",
            add_covariates=True,
            return_true_population_att=True,
        ),
    },
    "s3": {
        "label": "Repeated cross-section",
        "dgp_kwargs": dict(
            n_periods=N_PERIODS,
            treatment_effect=TREATMENT_EFFECT,
            icc=0.1,
            weight_cv=0.5,
            add_covariates=True,
            panel=False,
            return_true_population_att=True,
        ),
    },
    "s4": {
        "label": "Conditional PT (headline)",
        "dgp_kwargs": dict(
            n_periods=N_PERIODS,
            treatment_effect=TREATMENT_EFFECT,
            add_covariates=True,
            conditional_pt=1.5,
            psu_re_sd=0.5,
            return_true_population_att=True,
        ),
    },
}

# --- Estimators ---
# Each maps estimator_id -> class name, constructor kwargs, fit kwargs, result accessors
ESTIMATORS = {
    "cs_reg": {
        "label": "CS (reg)",
        "class": "CallawaySantAnna",
        "init_kwargs": dict(estimation_method="reg"),
        "fit_kwargs": dict(
            outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        ),
        "result_attrs": ("overall_att", "overall_se", "overall_conf_int"),
    },
    "cs_dr": {
        "label": "CS (DR)",
        "class": "CallawaySantAnna",
        "init_kwargs": dict(estimation_method="dr"),
        "fit_kwargs": dict(
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            covariates=["x1", "x2"],
        ),
        "result_attrs": ("overall_att", "overall_se", "overall_conf_int"),
    },
    "sa": {
        "label": "Sun-Abraham",
        "class": "SunAbraham",
        "init_kwargs": {},
        "fit_kwargs": dict(
            outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        ),
        "result_attrs": ("overall_att", "overall_se", "overall_conf_int"),
    },
    "twfe": {
        "label": "TWFE",
        "class": "TwoWayFixedEffects",
        "init_kwargs": {},
        "fit_kwargs": dict(
            outcome="outcome", treatment="treated", time="period", unit="unit"
        ),
        "result_attrs": ("att", "se", "conf_int"),
    },
}

# --- Valid (scenario, estimator) pairs ---
# Not all estimators work in all scenarios
VALID_PAIRS = {
    "s1": ["cs_reg", "cs_dr", "sa", "twfe"],
    "s2": ["cs_reg", "cs_dr", "sa", "twfe"],
    "s3": ["cs_reg"],  # Only CS supports panel=False natively
    "s4": ["cs_reg", "cs_dr"],  # The comparison pair: no-cov vs with-cov
}


def cell_id(scenario_id, estimator_id, n_units):
    """Unique string key for a simulation cell."""
    return f"{scenario_id}_{estimator_id}_n{n_units}"


def cell_csv_path(cid, base_dir=None):
    """Path to the results CSV for a cell."""
    d = base_dir or RESULTS_DIR
    return d / f"{cid}.csv"


def all_valid_cells():
    """Generate all valid (scenario_id, estimator_id, n_units) triples."""
    cells = []
    for sid, estimator_ids in VALID_PAIRS.items():
        for eid in estimator_ids:
            for n in SAMPLE_SIZES:
                cells.append((sid, eid, n))
    return cells


def make_estimator(estimator_id, **overrides):
    """Instantiate an estimator from its config, with optional overrides."""
    import diff_diff

    cfg = ESTIMATORS[estimator_id]
    cls = getattr(diff_diff, cfg["class"])
    kwargs = {**cfg["init_kwargs"], **overrides}
    return cls(**kwargs)


def get_result_values(result, estimator_id):
    """Extract (att, se, ci_lo, ci_hi) from a result object."""
    import math

    att_attr, se_attr, ci_attr = ESTIMATORS[estimator_id]["result_attrs"]
    att = getattr(result, att_attr, math.nan)
    se = getattr(result, se_attr, math.nan)
    ci = getattr(result, ci_attr, (math.nan, math.nan))
    return att, se, ci[0], ci[1]
