"""
Simulation Coordinator
=======================
Iterates over the simulation grid, skips completed cells, tracks progress
via a manifest file, and supports resume after interruption.

Usage:
    python paper/simulations/sim_run.py                        # Run all pending cells
    python paper/simulations/sim_run.py --cell s1_cs_reg_n500  # Run one specific cell
    python paper/simulations/sim_run.py --status               # Print progress summary
    python paper/simulations/sim_run.py --reps 100             # Override rep count
    python paper/simulations/sim_run.py --workers 4            # Parallel workers (default: CPU count)
    python paper/simulations/sim_run.py --sequential           # Force sequential execution
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Ensure imports work from the simulations directory
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sim_config import (
    N_REPS,
    RESULTS_DIR,
    SCENARIOS,
    ESTIMATORS,
    all_valid_cells,
    cell_id,
    cell_csv_path,
)
from sim_cell import count_existing_reps, run_cell


MANIFEST_PATH = RESULTS_DIR / "manifest.json"


def _run_one_cell(args_tuple):
    """Worker function for one cell. Runs in a subprocess."""
    warnings.filterwarnings("ignore", category=UserWarning)
    sid, eid, n, n_reps = args_tuple
    return run_cell(sid, eid, n, n_reps, print_every=0)


def load_manifest():
    """Load or create the manifest."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"cells": {}, "started_at": datetime.now().isoformat()}


def save_manifest(manifest):
    """Save manifest to disk."""
    manifest["last_updated"] = datetime.now().isoformat()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def print_status(manifest, n_reps):
    """Print progress summary."""
    cells = all_valid_cells()
    total_cells = len(cells)
    total_reps = total_cells * n_reps

    complete = 0
    reps_done = 0
    for sid, eid, n in cells:
        cid = cell_id(sid, eid, n)
        csv_path = cell_csv_path(cid)
        n_existing = count_existing_reps(csv_path)
        reps_done += min(n_existing, n_reps)
        if n_existing >= n_reps:
            complete += 1

    print(f"\nSimulation Progress")
    print(f"{'=' * 50}")
    print(f"Cells:  {complete}/{total_cells} complete")
    print(f"Reps:   {reps_done:,}/{total_reps:,} ({100 * reps_done / total_reps:.1f}%)")
    print()

    # Per-scenario breakdown
    for sid in SCENARIOS:
        scenario_cells = [(s, e, n) for s, e, n in cells if s == sid]
        s_complete = sum(
            1
            for s, e, n in scenario_cells
            if count_existing_reps(cell_csv_path(cell_id(s, e, n))) >= n_reps
        )
        s_total = len(scenario_cells)
        label = SCENARIOS[sid]["label"]
        status = "DONE" if s_complete == s_total else f"{s_complete}/{s_total}"
        print(f"  {sid} ({label}): {status}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Run simulation study")
    parser.add_argument("--cell", help="Run a specific cell by ID")
    parser.add_argument("--status", action="store_true", help="Print progress")
    parser.add_argument("--reps", type=int, default=N_REPS, help="Reps per cell")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--sequential", action="store_true", help="Force sequential execution"
    )
    args = parser.parse_args()

    n_reps = args.reps
    manifest = load_manifest()

    if args.status:
        print_status(manifest, n_reps)
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.cell:
        # Run a specific cell
        parts = args.cell.rsplit("_n", 1)
        if len(parts) != 2:
            print(f"Error: invalid cell ID '{args.cell}'")
            sys.exit(1)
        scenario_est = parts[0]
        n_units = int(parts[1])
        # Find the matching (scenario, estimator)
        found = False
        for sid, eid, n in all_valid_cells():
            if cell_id(sid, eid, n) == args.cell:
                print(f"Running cell: {args.cell} ({n_reps} reps)")
                result = run_cell(sid, eid, n, n_reps)
                manifest["cells"][args.cell] = {
                    "status": "complete",
                    "reps_done": result["n_complete"],
                    "n_failed": result["n_failed"],
                    "elapsed_s": result["elapsed_s"],
                }
                save_manifest(manifest)
                found = True
                break
        if not found:
            print(f"Error: cell '{args.cell}' not in valid cells")
            sys.exit(1)
        return

    # Build list of pending cells
    cells = all_valid_cells()
    total_cells = len(cells)
    pending = []
    skipped = 0
    for sid, eid, n in cells:
        cid = cell_id(sid, eid, n)
        existing = count_existing_reps(cell_csv_path(cid))
        if existing >= n_reps:
            skipped += 1
        else:
            pending.append((sid, eid, n))

    n_workers = 1 if args.sequential else (args.workers or os.cpu_count() or 1)
    mode = "sequential" if n_workers == 1 else f"parallel ({n_workers} workers)"

    print(f"Simulation Study: {total_cells} cells x {n_reps} reps")
    print(f"Execution: {mode}")
    if skipped:
        print(f"Skipping {skipped} already-complete cells")
    print(f"Running {len(pending)} pending cells")
    print(f"{'=' * 60}")

    run_start = time.time()
    completed = [0]  # mutable counter for callback

    def _on_complete(result):
        """Callback when a cell finishes (main process only)."""
        completed[0] += 1
        cid = result["cell_id"]
        elapsed = result["elapsed_s"]
        failed = result["n_failed"]
        total_elapsed = time.time() - run_start
        print(
            f"  [{completed[0]}/{len(pending)}] {cid} done "
            f"({elapsed:.1f}s, {failed} failed) "
            f"[{total_elapsed:.0f}s total]"
        )
        manifest["cells"][cid] = {
            "status": "complete",
            "reps_done": result["n_complete"],
            "n_failed": result["n_failed"],
            "elapsed_s": result["elapsed_s"],
        }
        save_manifest(manifest)

    if n_workers == 1:
        # Sequential: run directly with per-rep logging
        for sid, eid, n in pending:
            cid = cell_id(sid, eid, n)
            label = f"{SCENARIOS[sid]['label']} / {ESTIMATORS[eid]['label']} / n={n}"
            print(f"\n  {cid} ({label})")
            result = run_cell(sid, eid, n, n_reps)
            _on_complete(result)
    else:
        # Parallel: use multiprocessing pool
        with mp.Pool(processes=n_workers) as pool:
            async_results = []
            for sid, eid, n in pending:
                ar = pool.apply_async(
                    _run_one_cell, ((sid, eid, n, n_reps),),
                    callback=_on_complete,
                )
                async_results.append(ar)
            # Wait for all to complete
            for ar in async_results:
                ar.get()

    elapsed_total = time.time() - run_start
    print(f"\n{'=' * 60}")
    print(f"All cells complete in {elapsed_total:.1f}s ({elapsed_total / 60:.1f} min)")
    print_status(manifest, n_reps)


if __name__ == "__main__":
    main()
