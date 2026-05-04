# Data

This directory holds inputs used by the empirical illustration in Section 6 of
the paper. All files are public-use; nothing in this repository is subject to
restricted-data agreements.

## `nhanes/nhanes_realdata_golden.json`

Frozen NHANES analysis used to produce `tables/nhanes.tex`. The file is a
copy of the cross-validation golden file shipped with the diff-diff package
(`benchmarks/data/real/nhanes_realdata_golden.json`), generated against
NHANES public-use files from the U.S. National Center for Health Statistics
(NCHS).

**Source data (upstream):** NHANES 2007-2010 cycles, public-use files from
<https://wwwn.cdc.gov/nchs/nhanes/Default.aspx>.

**Estimand:** ATT of the ACA's dependent-coverage provision on insurance
coverage for young adults (ages 19-25, treatment) versus a comparison group
(ages 26-34), pre-period 2007-2008 vs. post-period 2009-2010. See Section 6 of
the paper for the empirical specification.

**Schema:** the JSON file encodes (a) the analysis-ready DataFrame as
`_data` and (b) pre-computed reference results from R's `survey` package
under several design specifications (`b1_strata_psu_weights`,
`b2_covariates`, `b3_weights_only`). `simulations/nhanes_table.py` reads
this file directly and produces the LaTeX table without recomputing from raw
NHANES files.

**Why a frozen JSON instead of raw NHANES files?** The empirical illustration
is a single number with three variance specifications. Shipping raw NHANES
files (multiple per cycle, several MB each, with idiosyncratic variable names
and merge keys) would burden reviewers with download steps unrelated to the
paper's contribution. The golden JSON is a self-contained reproduction
artifact derived from the public NHANES files; reviewers who want to recompute
from raw NHANES files can do so via the diff-diff package's benchmark
harness at `benchmarks/R/benchmark_realdata_nhanes.R` in that repository.
