# Replication: Design-Based Variance Estimation for Modern Heterogeneity-Robust DiD

This repository contains the simulation and empirical-illustration code that
produces the numerical results in:

> Gerber, Isaac (2026). *Design-Based Variance Estimation for Modern
> Heterogeneity-Robust Difference-in-Differences Estimators.*
> [arXiv:2605.04124](https://arxiv.org/abs/2605.04124).

The companion software is the [`diff-diff`](https://github.com/igerber/diff-diff)
Python package, pinned to **v3.3.2**
([Zenodo DOI](https://doi.org/10.5281/zenodo.19803705)).

## Layout

```
.
├── simulations/
│   ├── sim_config.py          Scenario / estimator grid
│   ├── sim_cell.py            Single-cell driver (deterministic seeds via md5)
│   ├── sim_run.py             Parallel coordinator with checkpointing
│   ├── sim_analyze.py         Aggregates → console + LaTeX tables
│   ├── make_figure.py         Coverage figure (Figure 1 in the paper)
│   ├── nhanes_table.py        Empirical illustration table (NHANES + ACA)
│   ├── verify_scenarios.py    Sanity checks for the simulation DGPs
│   ├── sim_validate.py        Validation harness
│   └── results/               Cell-level CSVs (committed for diff-checking)
├── data/
│   └── nhanes/                Frozen NHANES golden JSON (with provenance)
├── tables/                    Generated LaTeX tables (sim_s{1..4}, nhanes)
├── figures/                   Generated coverage figure (PDF + PNG preview)
├── requirements.txt           Pinned Python dependencies
├── Makefile                   Reproduction targets
├── CITATION.cff
├── LICENSE
└── README.md                  This file
```

## Reproducing the simulation results

### Quick path (build outputs from committed CSVs)

If you trust the committed simulation CSVs and only want the tables and figure:

```sh
make install          # pinned dependencies into the active Python environment
make tables figures   # regenerate LaTeX tables and Figure 1 from results/*.csv
```

This takes well under a minute on a normal laptop.

### Full path (rerun simulations from scratch)

```sh
make install
make clean-results    # wipe simulations/results/
make all              # sims → tables → figures
```

Wall-clock time on a 14-core CPU: about 30 minutes for the full grid (33 cells
× 2,000 replications, three SE methods per replication: HC1, weighted point +
PSU cluster, full design-based TSL). Total CPU time is ~14 hours; the parallel
coordinator divides it across cores. CSVs are written incrementally, so the
sims can be interrupted and resumed.

### Smoke test

```sh
make smoke
```

Runs five replications of one cell and writes
`simulations/results/s1_cs_reg_n500.csv`. Use this to verify
`diff-diff==3.3.2`, the seed pipeline, and the matrix kernels all behave on
your machine before kicking off the full grid.

## Determinism

The per-cell seed is derived from a stable hash of the cell ID:

```python
seed = rep * 1000 + (md5(cell_id) % 1_000_000)
```

We use `hashlib.md5` rather than Python's built-in `hash()` because the latter
is randomized per interpreter unless `PYTHONHASHSEED` is set, which would
defeat reproducibility across machines. The Makefile targets export
`PYTHONHASHSEED=0` defensively in case other code in the call stack relies
on it.

## Software dependencies

Python 3.9.6, with the package versions in `requirements.txt`:

- `diff-diff==3.3.2` ([Zenodo](https://doi.org/10.5281/zenodo.19803705))
- `numpy==2.0.2`, `pandas==2.3.3`, `scipy==1.13.1`, `matplotlib==3.9.4`

Newer versions probably work but were not used to produce the committed
results.

## NHANES data

The NHANES illustration in Section 6 of the paper uses a frozen analytic
golden file at `data/nhanes/nhanes_realdata_golden.json`. This is a copy of
the cross-validation golden file shipped with the diff-diff package
(`benchmarks/data/real/nhanes_realdata_golden.json`), generated against
NHANES public-use files from CDC/NCHS for the 2007-2008 and 2015-2016
cycles. See `data/README.md` for full provenance.

## Citation

If you use this replication code, please cite both the paper and the
companion package:

```bibtex
@misc{gerber2026,
  author       = {Gerber, Isaac},
  title        = {Design-Based Variance Estimation for Modern Heterogeneity-Robust
                  Difference-in-Differences Estimators},
  year         = {2026},
  eprint       = {2605.04124},
  archivePrefix = {arXiv},
  primaryClass = {stat.ME},
  url          = {https://arxiv.org/abs/2605.04124}
}

@misc{diffdiff2026,
  author    = {Gerber, Isaac},
  title     = {{diff-diff: Difference-in-Differences Causal Inference for Python}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {3.3.2},
  doi       = {10.5281/zenodo.19803705}
}
```

## License

All code in this repository is released under the MIT License (see `LICENSE`).
The diff-diff package and NHANES public-use data retain their own licenses.
