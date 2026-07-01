"""Run the conformal benchmark grid (or a slice) and write results.csv.

Loops dataset x tasks x models x methods x modes (all alphas) x seeds. The model trains
once per (task, model, seed); LABEL/APS/RAPS calibrate off it. --init seeds the planned
results.csv; --demo runs without writing (validation informational).

Usage:
    python run_grid.py --init
    python run_grid.py --root /path/to/mimiciv/2.2
    python run_grid.py --root /path/to/mimiciv/2.2 --tasks mortality
    python run_grid.py --root /path/to/mimiciv/2.2 --method LABEL
    python run_grid.py --root /path/to/mimiciv/2.2 --skip-done   # resume
    python run_grid.py --root /path/to/mimiciv/2.2 --demo --epochs 1 --seeds 0
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
    "coverage_mean", "coverage_std", "avg_set_size", "rejection_rate",
    "per_class_miscov", "worst_class_miscov", "worst_class", "base_auroc", "base_f1",
    "monitor", "seeds", "n_cal", "n_test", "status", "validation_passed",
    "run_id", "date_run", "commit_hash",
]


def planned_rows():
    """The full planned grid: every cell x alpha, results blank, status=planned."""
    seeds = f"{grid.SEEDS[0]}-{grid.SEEDS[-1]}"
    for dataset in grid.DATASETS:
        for task in grid.TASKS:
            for model in grid.MODELS:
                for method in grid.METHODS:
                    modes = grid.MODES if method == "LABEL" else ["marginal"]
                    for mode in modes:
                        cell = f"{dataset}-{task}-{model}-{method}-{mode}"
                        for alpha in grid.ALPHAS:
                            yield {
                                "cell_id": cell, "dataset": dataset, "task": task,
                                "output_type": grid.OUTPUT_TYPE[task], "model": model,
                                "split": grid.SPLIT, "cal_separate_from_val": "yes",
                                "method": method, "mode": mode, "alpha": alpha,
                                "target_coverage": round(1 - alpha, 2),
                                "coverage_mean": "", "coverage_std": "",
                                "avg_set_size": "", "rejection_rate": "",
                                "per_class_miscov": "", "worst_class_miscov": "",
                                "worst_class": "", "base_auroc": "", "base_f1": "",
                                "monitor": grid.MONITOR[task], "n_cal": "", "n_test": "",
                                "seeds": seeds, "status": "planned",
                                "validation_passed": "", "run_id": "", "date_run": "",
                                "commit_hash": "",
                            }


def seed_results():
    """Add planned rows for cells not already in results.csv; never overwrite existing.

    Writes the current RESULTS_COLUMNS schema (new columns backfilled blank on old rows).
    """
    table = {}
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="") as f:
            table = {(r["cell_id"], r["mode"], r["alpha"]): r
                     for r in csv.DictReader(f)}
    added = 0
    for row in planned_rows():
        key = (row["cell_id"], row["mode"], str(row["alpha"]))
        if key not in table:
            table[key] = row
            added += 1
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table.values())
    print(f"seeded results.csv: +{added} planned rows ({len(table)} total)")


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


def _done_cells(dataset):
    """(task, model) pairs whose every results.csv row is already computed (not planned)."""
    if not RESULTS_CSV.exists():
        return set()
    complete = {}
    with open(RESULTS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["dataset"] != dataset:
                continue
            key = (r["task"], r["model"])
            complete[key] = complete.get(key, True) and r["status"] != "planned"
    return {k for k, ok in complete.items() if ok}


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
    p.add_argument("--method", default=None, choices=list(framework.SCORERS),
                   help="restrict to one method (default: all of grid.METHODS)")
    p.add_argument("--pred-cache", default=None,
                   help="dir to cache per-seed cal/test predictions (npz); skipped if unset")
    p.add_argument("--skip-done", action="store_true",
                   help="resume: skip (task,model) cells already fully computed in results.csv")
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
    methods = grid.METHODS if args.method is None else [args.method]
    label_modes = args.modes.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    prov = {
        "run_id": args.run_id,
        "date_run": datetime.date.today().isoformat(),
        "commit_hash": _commit_hash(),
    }
    print(f"{'SMOKE' if args.demo else 'GRID'} dataset={args.dataset} methods={methods} "
          f"tasks={tasks} models={models} modes={label_modes} seeds={seeds} "
          f"epochs={args.epochs} prov={prov}", flush=True)

    done = _done_cells(args.dataset) if args.skip_done else set()
    base = framework.load_base_dataset(args.dataset, args.root, args.dev)
    total = {}
    for task in tasks:
        if done and all((task, m) in done for m in models):
            print(f"\n#### {args.dataset} | {task}: skip (all models complete) ####",
                  flush=True)
            continue
        samples = base.set_task(framework.build_task(args.dataset, task))
        print(f"\n#### {args.dataset} | {task} | Samples: {len(samples)} ####", flush=True)
        for model in models:
            if (task, model) in done:
                print(f"## model={model}: skip (already complete) ##", flush=True)
                continue
            print(f"## model={model} ##", flush=True)
            rows = framework.run_cell(
                args.dataset, task, model, samples, methods, label_modes, grid.ALPHAS,
                seeds=seeds, epochs=args.epochs, demo=args.demo,
                pred_cache=args.pred_cache,
            )
            if args.demo:
                for r in rows:
                    print(f"  {r['method']}/{r['mode']} a={r['alpha']}: {r['status']} "
                          f"cov={r['coverage_mean']} size={r['avg_set_size']} "
                          f"| {r['detail']}", flush=True)
                print(f"  base auroc={rows[0]['base_auroc']} f1={rows[0]['base_f1']} "
                      f"| cal per-class counts: {rows[0].get('cal_counts')}", flush=True)
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
