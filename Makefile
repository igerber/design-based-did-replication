# Reproduction harness for the simulation results in
# Gerber (2026), "Design-Based Variance Estimation for Modern
# Heterogeneity-Robust Difference-in-Differences Estimators."
#
# All sim-related targets export PYTHONHASHSEED=0 so the per-cell seeds are
# deterministic across processes and machines.

PYTHON       ?= python3
HASHSEED     := 0

SIM_DIR      := simulations
RESULTS_DIR  := $(SIM_DIR)/results
TABLES_DIR   := tables
FIGURES_DIR  := figures

.PHONY: all install sims tables figures smoke clean-results help

help:
	@echo "Targets:"
	@echo "  install         Install pinned dependencies into the active environment"
	@echo "  sims            Run the full simulation grid (~30 min wall on 14 workers)"
	@echo "  tables          Regenerate sim tables and the NHANES table"
	@echo "  figures         Regenerate Figure 1 (three-panel coverage figure)"
	@echo "  all             sims + tables + figures"
	@echo "  smoke           5-rep smoke test on one cell"
	@echo "  clean-results   Remove all simulation outputs (forces a full rerun)"

install:
	$(PYTHON) -m pip install --upgrade -r requirements.txt

sims:
	PYTHONHASHSEED=$(HASHSEED) $(PYTHON) $(SIM_DIR)/sim_run.py

tables:
	$(PYTHON) $(SIM_DIR)/sim_analyze.py --latex
	$(PYTHON) $(SIM_DIR)/nhanes_table.py

figures:
	$(PYTHON) $(SIM_DIR)/make_figure.py

all: sims tables figures

smoke:
	rm -f $(RESULTS_DIR)/s1_cs_reg_n500.csv
	PYTHONHASHSEED=$(HASHSEED) $(PYTHON) $(SIM_DIR)/sim_run.py \
	    --cell s1_cs_reg_n500 --reps 5

clean-results:
	rm -rf $(RESULTS_DIR)
	mkdir -p $(RESULTS_DIR)
