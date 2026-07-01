"""Recover n_cal, n_test, rejection_rate for completed cells from cached predictions.

No GPU, no retraining: recomputes prediction sets from the saved ~/cp-preds/*.npz
(scorer -> conformal_threshold -> set = scores <= t) using the exact framework code, then
  - rejection_rate = fraction of test samples with an EMPTY set (set.sum(1)==0), mean over seeds
  - n_cal / n_test  = split sizes, mean over seeds (they vary by seed; spread is reported)
As a guard it recomputes avg_set_size and compares to results.csv; any cell whose recomputed
size disagrees, or whose npz is missing, is reported and NOT filled (never silently blank).

Writes cpbench_extra.csv keyed by (cell_id, mode, alpha) for a clean merge into results.csv.

Run from examples/benchmark_conformal/ on the cluster:
    python rescore_extra.py --pred-cache ~/cp-preds
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re

import numpy as np

import framework

NPZ_RE = re.compile(r"(mimic3|mimic4|eicu)-(los|mortality|readmission)-(\w+)-seed(\d+)\.npz$")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-cache", default=os.path.expanduser("~/cp-preds"))
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--out", default="cpbench_extra.csv")
    ap.add_argument("--tol", type=float, default=0.005, help="avg_set_size match tolerance")
    args = ap.parse_args()

    # index npz by (dataset, task, model) -> {seed: path}
    npz = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(args.pred_cache, "*.npz")):
        m = NPZ_RE.match(os.path.basename(f))
        if m:
            npz[(m.group(1), m.group(2), m.group(3))][int(m.group(4))] = f

    rows = list(csv.DictReader(open(args.results)))
    completed = [r for r in rows if r["status"] in ("done", "flagged")]

    loaded = {}

    def arrays(dtm):
        if dtm not in loaded:
            loaded[dtm] = {s: np.load(p) for s, p in npz.get(dtm, {}).items()}
        return loaded[dtm]

    out, missing, mismatch = [], [], []
    ntnc = collections.defaultdict(list)   # (ds, task) -> [(n_cal, n_test) per seed]
    recorded = set()

    for r in completed:
        dtm = (r["dataset"], r["task"], r["model"])
        data = arrays(dtm)
        seeds = sorted(data)
        if not seeds:
            missing.append((r["cell_id"], r["alpha"], "no npz for " + "-".join(dtm)))
            continue

        # split sizes (model-independent; record once per dataset/task)
        if (r["dataset"], r["task"]) not in recorded:
            for s in seeds:
                ntnc[(r["dataset"], r["task"])].append(
                    (len(data[s]["cal_y"]), len(data[s]["test_y"]))
                )
            recorded.add((r["dataset"], r["task"]))

        method, mode, alpha = r["method"], r["mode"], float(r["alpha"])
        rej, sz, ncal, ntest = [], [], [], []
        for s in seeds:
            d = data[s]
            K = d["test_prob"].shape[1]
            cal_s = framework.SCORERS[method](d["cal_prob"])
            test_s = framework.SCORERS[method](d["test_prob"])
            t = framework.conformal_threshold(cal_s, d["cal_y"], mode, alpha, K)
            predset = test_s <= t
            rej.append(float(np.mean(predset.sum(1) == 0)))
            sz.append(float(np.mean(predset.sum(1))))
            ncal.append(len(d["cal_y"]))
            ntest.append(len(d["test_y"]))

        stored = _num(r["avg_set_size"])
        recomp = float(np.mean(sz))
        if stored is not None and abs(stored - recomp) > args.tol:
            mismatch.append((r["cell_id"], r["alpha"], f"stored={stored} recomp={recomp:.4f}"))
            continue

        out.append({
            "cell_id": r["cell_id"], "mode": mode, "alpha": r["alpha"],
            "n_cal": int(round(np.mean(ncal))), "n_test": int(round(np.mean(ntest))),
            "rejection_rate": round(float(np.mean(rej)), 4),
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell_id", "mode", "alpha",
                                          "n_cal", "n_test", "rejection_rate"])
        w.writeheader()
        w.writerows(out)

    print(f"wrote {args.out}: {len(out)} cells filled")
    print("\nn_cal / n_test per (dataset, task)  [mean over seeds, ±std]:")
    print(f"  {'dataset':7s} {'task':12s} {'n_cal':>14s}  {'n_test':>14s}")
    for (ds, task), vals in sorted(ntnc.items()):
        cals = [v[0] for v in vals]
        tests = [v[1] for v in vals]
        print(f"  {ds:7s} {task:12s} "
              f"{int(np.mean(cals)):8d} (±{int(np.std(cals)):>3d})  "
              f"{int(np.mean(tests)):8d} (±{int(np.std(tests)):>3d})")

    if missing:
        print(f"\n!! MISSING npz -- {len(missing)} cells NOT filled (need a re-run with "
              f"--pred-cache to regenerate):")
        for c in missing[:30]:
            print("   ", c)
    if mismatch:
        print(f"\n!! SIZE MISMATCH -- {len(mismatch)} cells NOT filled (recompute != "
              f"results.csv; investigate before trusting):")
        for c in mismatch[:30]:
            print("   ", c)
    if not missing and not mismatch:
        print("\nOK -- every completed cell recovered; recomputed sizes match results.csv.")


if __name__ == "__main__":
    main()
