"""CPBench wrapper — conformal-prediction-on-EHR benchmark (fork).

Follows the benchmarks/cpbench/benchmark.py conventions from PR #1168 (a PROTOCOL
constant, a TASK_REGISTRY and METHOD_REGISTRY of declarative entries, `--list`, and a
JSON results record) for the EHR side of CPBench: MIMIC-III, MIMIC-IV, and eICU on
length-of-stay / mortality / readmission, with LABEL (marginal and class-conditional),
APS, and RAPS. Beyond the standard coverage/set-size metrics it also reports per-class
miscoverage, worst-class coverage, and base-model AUROC/F1 -- the metrics that make the
imbalanced-EHR story legible (accuracy alone hides the rare-class collapse).

PROTOTYPE NOTES (for merging into the shared wrapper):
  * Reuses the validated engine in examples/benchmark_conformal/ (framework.py / grid.py)
    instead of re-implementing it. Inline those before upstreaming.
  * PROTOCOL below is the fork's reference protocol (matches our existing results). The
    CPBench-v1 target is split_ratios [0.60, 0.15, 0.10, 0.15] with seeds [42..46];
    switching to it requires a re-run.
  * Our "aps"/"raps" are the genuine adaptive methods (validated set-for-set vs TorchCP),
    distinct from the wrapper's `base` = BaseConformal(score_type="aps"), which computes
    the LABEL/threshold score, not APS.

Usage:
    python benchmark.py --list
    python benchmark.py --task mimic4_mortality --method label     --data-path <root>
    python benchmark.py --task mimic4_los       --method aps       --data-path <root> --dev
    python benchmark.py --task mimic3_readmission --method label-cc --data-path <root> --output out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Reuse the reference engine (grid is light / torch-free; framework is imported lazily
# inside run_benchmark so `--list` stays fast).
_ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "examples", "benchmark_conformal")
sys.path.insert(0, _ENGINE)
import grid  # noqa: E402

PROTOCOL = {
    "version": "v1-ehr-fork",
    "split_ratios": grid.RATIOS,          # [0.6, 0.1, 0.1, 0.2] (v1 target: 0.60/0.15/0.10/0.15)
    "seeds": grid.SEEDS,                  # [0, 1, 2, 3, 4]       (v1 target: 42..46)
    "batch_size": 32,
    "standard_alphas": grid.ALPHAS,       # [0.2, 0.1, 0.05, 0.01]
}

_TASK_DESC = {
    "los": "10-class length-of-stay prediction",
    "mortality": "binary next-visit mortality prediction",
    "readmission": "binary readmission prediction",
}

TASK_REGISTRY = {
    f"{ds}_{task}": {
        "description": f"{_TASK_DESC[task]} on {ds.upper()}",
        "dataset": ds,
        "task": task,
        "output_type": grid.OUTPUT_TYPE[task],
        "monitor": grid.MONITOR[task],        # f1_macro / roc_auc_weighted_ovr, not accuracy
        "models": grid.MODELS,                # Transformer, RNN, RETAIN
        "default_epochs": 10,
    }
    for ds in grid.DATASETS for task in grid.TASKS
}

# name -> (framework scorer, mode(s)). label/label-cc share the LABEL scorer, differing
# only in marginal vs per-class calibration.
METHOD_REGISTRY = {
    "label": {
        "description": "LABEL least-ambiguous sets (Sadinle 2019), marginal",
        "framework_method": "LABEL", "modes": ["marginal"],
        "paper": "Sadinle, Lei, Wasserman. JASA (2019)",
    },
    "label-cc": {
        "description": "LABEL, class-conditional (per-class coverage guarantee)",
        "framework_method": "LABEL", "modes": ["class-conditional"],
        "paper": "Sadinle, Lei, Wasserman. JASA (2019)",
    },
    "aps": {
        "description": "APS adaptive prediction sets (Romano 2020), validated vs TorchCP",
        "framework_method": "APS", "modes": ["marginal"],
        "paper": "Romano, Sesia, Candes. NeurIPS (2020)",
    },
    "raps": {
        "description": "RAPS regularized APS (Angelopoulos 2021), validated vs TorchCP",
        "framework_method": "RAPS", "modes": ["marginal"],
        "paper": "Angelopoulos, Bates, Jordan, Malik. ICLR (2021)",
    },
}

# metric fields carried from a framework result row into the JSON record
_METRIC_KEYS = [
    "alpha", "mode", "target_coverage", "coverage_mean", "coverage_std",
    "avg_set_size", "per_class_miscov", "worst_class_miscov", "base_auroc",
    "base_f1", "validation_passed", "status",
]


def run_benchmark(task_key, method_key, data_path, models, seeds, alphas, epochs,
                  dev, pred_cache):
    """Run one (task, method) across the requested models; return {model: [row-metrics]}."""
    import framework  # heavy (torch/pyhealth); only needed for an actual run

    tcfg = TASK_REGISTRY[task_key]
    mcfg = METHOD_REGISTRY[method_key]
    dataset, task = tcfg["dataset"], tcfg["task"]
    print(f"CPBench | task={task_key} | method={method_key} | models={models}")
    print(f"  seeds={seeds}  alphas={alphas}  epochs={epochs}  dev={dev}")

    base = framework.load_base_dataset(dataset, data_path, dev)
    samples = base.set_task(framework.build_task(dataset, task))
    print(f"  samples: {len(samples)}")

    results = {}
    for model in models:
        print(f"\n## model={model} ##")
        rows = framework.run_cell(
            dataset, task, model, samples,
            [mcfg["framework_method"]], mcfg["modes"], alphas,
            seeds=seeds, epochs=epochs, demo=dev, pred_cache=pred_cache,
        )
        for r in rows:
            print(f"  {r['mode']} a={r['alpha']}: {r['status']} "
                  f"cov={r['coverage_mean']} size={r['avg_set_size']} "
                  f"worst_miscov={r['worst_class_miscov']}")
        results[model] = [{k: r.get(k) for k in _METRIC_KEYS} for r in rows]
    return results


def save_results(results, task_key, method_key, seeds, alphas, path):
    record = {
        "cpbench_protocol": PROTOCOL,
        "task": task_key,
        "method": method_key,
        "method_paper": METHOD_REGISTRY[method_key]["paper"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "alphas": alphas,
        "results_by_model": results,   # per model: aggregated (mean over seeds) rows
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"\nResults saved -> {path}")


def list_registry():
    print("\nRegistered Tasks:")
    print("-" * 70)
    for name, cfg in TASK_REGISTRY.items():
        print(f"  {name:<22} {cfg['description']}  [monitor={cfg['monitor']}]")
    print("\nRegistered CP Methods:")
    print("-" * 70)
    for name, cfg in METHOD_REGISTRY.items():
        print(f"  {name:<10} {cfg['description']}")
        print(f"  {'':10} {cfg['paper']}")
    print()


def parse_args():
    p = argparse.ArgumentParser(
        description="CPBench (EHR fork): standardized conformal prediction benchmarking")
    p.add_argument("--list", action="store_true",
                   help="List registered tasks and methods, then exit.")
    p.add_argument("--task", choices=list(TASK_REGISTRY), help="Task (see --list).")
    p.add_argument("--method", choices=list(METHOD_REGISTRY), help="CP method (see --list).")
    p.add_argument("--data-path", help="Path to the raw dataset on disk.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Models to sweep (default: all of the task's models).")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help=f"Seeds (default {PROTOCOL['seeds']}).")
    p.add_argument("--alpha", type=float, nargs="+", default=None,
                   help=f"Miscoverage rate(s) (default {PROTOCOL['standard_alphas']}).")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--pred-cache", default=None,
                   help="Dir to cache per-seed predictions (npz) for offline rescoring.")
    p.add_argument("--output", default=None, help="Path to write results JSON.")
    p.add_argument("--dev", action="store_true",
                   help="Dev mode: tiny subset, single seed, single alpha (0.1).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list:
        list_registry()
        return
    missing = [f for f in ("task", "method", "data_path") if not getattr(args, f, None)]
    if missing:
        raise SystemExit("Error: --" + ", --".join(m.replace("_", "-") for m in missing)
                         + " required (or use --list).")

    if args.dev:
        seeds, alphas = [PROTOCOL["seeds"][0]], [0.1]
    else:
        seeds = args.seeds if args.seeds is not None else PROTOCOL["seeds"]
        alphas = args.alpha if args.alpha is not None else PROTOCOL["standard_alphas"]
    models = args.models if args.models is not None else TASK_REGISTRY[args.task]["models"]

    results = run_benchmark(args.task, args.method, args.data_path, models,
                            seeds, alphas, args.epochs, args.dev, args.pred_cache)
    if args.output:
        save_results(results, args.task, args.method, seeds, alphas, args.output)


if __name__ == "__main__":
    main()
