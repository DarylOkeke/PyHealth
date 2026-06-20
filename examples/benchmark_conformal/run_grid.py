"""Single runner for the conformal benchmark: seed, smoke, and real grid runs.

Loops dataset x tasks x models x modes (all alphas) x seeds and persists to results.csv.
Operating modes:
  * --init: (re)seed results.csv with the planned grid (all cells, results blank) and exit.
  * real (default): demo=False -> validation gate live (coverage miss -> status=flagged /
    validation_passed=false), rows stamped run_id/date_run/commit_hash, upserted into
    results.csv.
  * --demo: validation informational, results.csv NOT written -- prints per-cell status +
    cal counts. Slice with --tasks/--models/--modes (+ --dev/--epochs/--seeds) to smoke a
    code path before a full run.

MIMIC-III is deferred (not on the cluster); default dataset is mimic4.

Usage:
    python run_grid.py --init                                  # seed results.csv
    python run_grid.py --root /projects/.../mimiciv/2.2        # full mimic4 grid, 5 seeds
    python run_grid.py --root /projects/.../mimiciv/2.2 --tasks mortality   # scale-check
    python run_grid.py --root /projects/.../mimiciv/2.2 --demo --epochs 1 --seeds 0 \
        --tasks mortality --modes class-conditional            # smoke, nothing written
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import subprocess
from pathlib import Path

import framework
import grid

RESULTS_CSV = Path(__file__).parent / "results.csv"

RESULTS_COLUMNS = [
    "cell_id", "dataset", "task", "output_type", "model", "split",
    "cal_separate_from_val", "method", "mode", "alpha", "target_coverage",
    "coverage_mean", "coverage_std", "avg_set_size", "per_class_miscov",
    "worst_class_miscov", "monitor", "seeds", "status", "validation_passed",
    "run_id", "date_run", "commit_hash",
]


def planned_rows():
    """The full planned grid: every cell x alpha, results blank, status=planned."""
    seeds = f"{grid.SEEDS[0]}-{grid.SEEDS[-1]}"
    for dataset in grid.DATASETS:
        for task in grid.TASKS:
            for model in grid.MODELS:
                for mode in grid.MODES:
                    cell = f"{dataset}-{task}-{model}-{grid.METHOD}-{mode}"
                    for alpha in grid.ALPHAS:
                        yield {
                            "cell_id": cell, "dataset": dataset, "task": task,
                            "output_type": grid.OUTPUT_TYPE[task], "model": model,
                            "split": grid.SPLIT, "cal_separate_from_val": "yes",
                            "method": grid.METHOD, "mode": mode, "alpha": alpha,
                            "target_coverage": round(1 - alpha, 2),
                            "coverage_mean": "", "coverage_std": "", "avg_set_size": "",
                            "per_class_miscov": "", "worst_class_miscov": "",
                            "monitor": grid.MONITOR[task], "seeds": seeds,
                            "status": "planned", "validation_passed": "",
                            "run_id": "", "date_run": "", "commit_hash": "",
                        }


def seed_results():
    rows = list(planned_rows())
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"seeded results.csv: {len(rows)} planned rows")


def upsert_results(rows):
    """Merge result rows into results.csv, keyed by (cell_id, mode, alpha)."""
    with open(RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        table = {(r["cell_id"], r["mode"], r["alpha"]): r for r in reader}
    for row in rows:
        key = (row["cell_id"], row["mode"], str(row["alpha"]))
        table[key] = {**table.get(key, {}), **{k: row[k] for k in columns}}
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table.values())


def _commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
        ).strip()
    except Exception:
        return ""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root")
    p.add_argument("--dataset", default="mimic4", choices=grid.DATASETS)
    p.add_argument("--tasks", default=",".join(grid.TASKS))
    p.add_argument("--models", default=",".join(grid.MODELS))
    p.add_argument("--modes", default=",".join(grid.MODES))
    p.add_argument("--seeds", default=",".join(str(s) for s in grid.SEEDS))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--dev", action="store_true", help="subsample the dataset")
    p.add_argument("--demo", action="store_true",
                   help="smoke run: validation informational, results.csv not written")
    p.add_argument("--init", action="store_true",
                   help="(re)seed results.csv with the planned grid and exit")
    p.add_argument("--run-id", default=os.environ.get("SLURM_JOB_ID", ""))
    args = p.parse_args()

    if args.init:
        seed_results()
        return
    if not args.root:
        p.error("--root is required unless --init")

    tasks = args.tasks.split(",")
    models = args.models.split(",")
    modes = args.modes.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    prov = {
        "run_id": args.run_id,
        "date_run": datetime.date.today().isoformat(),
        "commit_hash": _commit_hash(),
    }
    print(f"{'SMOKE' if args.demo else 'GRID'} dataset={args.dataset} tasks={tasks} "
          f"models={models} modes={modes} seeds={seeds} epochs={args.epochs} "
          f"prov={prov}", flush=True)

    total = {}
    for task in tasks:
        for model in models:
            print(f"\n#### {args.dataset} | {task} | {model} ####", flush=True)
            rows = framework.run_cell(
                args.dataset, task, model, grid.METHOD, modes, grid.ALPHAS,
                seeds=seeds, epochs=args.epochs, root=args.root,
                dev=args.dev, demo=args.demo,
            )
            if args.demo:
                for r in rows:
                    print(f"  {r['mode']} a={r['alpha']}: {r['status']} "
                          f"cov={r['coverage_mean']} size={r['avg_set_size']} "
                          f"| {r['detail']}", flush=True)
                print(f"  cal per-class counts: {rows[0].get('cal_counts')}", flush=True)
            else:
                for r in rows:
                    r.update(prov)
                upsert_results(rows)
            cell = {}
            for r in rows:
                cell[r["status"]] = cell.get(r["status"], 0) + 1
                total[r["status"]] = total.get(r["status"], 0) + 1
            print(f"  {'(not written)' if args.demo else 'upserted'} "
                  f"{len(rows)} rows: {cell}", flush=True)

    print(f"\nDone. total: {total}"
          + ("" if args.demo else ". results.csv updated."), flush=True)
    if args.demo:
        print("FAIL -- some cells errored." if total.get("errored")
              else "OK -- no errors (smoke).", flush=True)


if __name__ == "__main__":
    main()
