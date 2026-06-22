"""Split-conformal benchmark engine for PyHealth EHR tasks.

Generalizes the three reference cells (full_examples/{los,mortality,readmission}_mimic4_
conformal.py) into one pipeline over dataset x task x model x method. run_cell trains the
base model once per seed and calibrates every (method, mode, alpha) from it -- LABEL, APS,
and RAPS are all downstream of one training run -- returning one row per (method, mode,
alpha); run_grid.py drives it and writes results.csv.
"""

from __future__ import annotations

import logging
import math
import os
import random
from datetime import timedelta

import numpy as np
import torch

import grid
from pyhealth.calib.predictionset.base_conformal import _query_quantile
from pyhealth.calib.utils import prepare_numpy_dataset
from pyhealth.datasets import (
    MIMIC3Dataset,
    MIMIC4Dataset,
    eICUDataset,
    get_dataloader,
    split_by_patient_conformal,
)
from pyhealth.metrics.prediction_set import (
    miscoverage_overall_ps,
    miscoverage_ps,
    size,
)
from pyhealth.models import RETAIN, RNN, Transformer
from pyhealth.tasks import (
    LengthOfStayPredictioneICU,
    LengthOfStayPredictionMIMIC3,
    LengthOfStayPredictionMIMIC4,
    MortalityPredictionEICU,
    MortalityPredictionMIMIC3,
    MortalityPredictionMIMIC4,
    ReadmissionPredictionEICU,
    ReadmissionPredictionMIMIC3,
    ReadmissionPredictionMIMIC4,
)
from pyhealth.trainer import Trainer, get_metrics_fn

logging.getLogger("pyhealth").setLevel(logging.WARNING)


# --- registries ----------------------------------------------------------

MODELS = {"Transformer": Transformer, "RNN": RNN, "RETAIN": RETAIN}

# (dataset, task) -> task class
TASK_CLASSES = {
    ("mimic3", "los"): LengthOfStayPredictionMIMIC3,
    ("mimic4", "los"): LengthOfStayPredictionMIMIC4,
    ("eicu", "los"): LengthOfStayPredictioneICU,
    ("mimic3", "mortality"): MortalityPredictionMIMIC3,
    ("mimic4", "mortality"): MortalityPredictionMIMIC4,
    ("eicu", "mortality"): MortalityPredictionEICU,
    ("mimic3", "readmission"): ReadmissionPredictionMIMIC3,
    ("mimic4", "readmission"): ReadmissionPredictionMIMIC4,
    ("eicu", "readmission"): ReadmissionPredictionEICU,
}


def aps_scores(y_prob, lam=0.0, k_reg=0):
    """APS/RAPS non-conformity scores (N, K): cumulative prob mass to each class.

    tau_k = sum of probs ranked >= class k (incl. k), plus RAPS penalty
    lam*max(0, rank_k - k_reg). lam=0 gives APS. Non-randomized.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    order = np.argsort(-y_prob, axis=1)
    ordered = np.take_along_axis(y_prob, order, axis=1)
    ranks = np.arange(1, y_prob.shape[1] + 1)
    ordered_scores = np.cumsum(ordered, axis=1) + np.maximum(0.0, lam * (ranks - k_reg))
    scores = np.empty_like(ordered_scores)
    np.put_along_axis(scores, order, ordered_scores, axis=1)
    return scores


def label_scores(y_prob):
    """LABEL non-conformity scores (N, K): 1 - p(class). Higher = less conforming."""
    return 1.0 - np.asarray(y_prob, dtype=float)


# method -> per-class non-conformity scores (N, K) from predicted probabilities
SCORERS = {
    "LABEL": label_scores,
    "APS": lambda y_prob: aps_scores(y_prob, 0.0, 0),
    "RAPS": lambda y_prob: aps_scores(y_prob, grid.RAPS_LAMBDA, grid.RAPS_K_REG),
}


def conformal_threshold(cal_scores, cal_y, mode, alpha, n_classes):
    """Split-conformal threshold from calibration scores; mirrors LABEL.calibrate.

    marginal -> one quantile of the true-class scores (scalar); class-conditional ->
    the alpha-quantile within each class (per-class vector, broadcast in the set rule).
    """
    if mode == "marginal":
        return _query_quantile(cal_scores[np.arange(len(cal_y)), cal_y], alpha)
    return np.array([
        _query_quantile(cal_scores[cal_y == k, k], alpha) for k in range(n_classes)
    ])


# --- task / dataset construction -----------------------------------------

def as_multiclass(task_cls, label_key):
    """Return a 2-class-multiclass subclass of a binary task so LABEL applies.

    Mirrors EEGAbnormalTUAB in pyhealth/tasks/temple_university_EEG_tasks.py.
    """
    return type(
        f"{task_cls.__name__}Multiclass",
        (task_cls,),
        {
            "task_name": f"{task_cls.__name__}Multiclass",
            "output_schema": {label_key: "multiclass"},
        },
    )


def build_task(dataset, task):
    """Instantiate the dataset-specific task, wrapped to multiclass if binary."""
    task_cls = TASK_CLASSES[(dataset, task)]
    if grid.OUTPUT_TYPE[task] == "binary":
        task_cls = as_multiclass(task_cls, grid.LABEL_KEY[task])
    if task == "readmission" and dataset != "eicu":
        return task_cls(window=timedelta(days=grid.READMISSION_WINDOW_DAYS))
    return task_cls()


def load_base_dataset(dataset, root, dev):
    """Load the base EHR dataset; task-independent, so built once and reused across tasks."""
    if dataset == "mimic4":
        return MIMIC4Dataset(ehr_root=root, ehr_tables=grid.TABLES[dataset], dev=dev)
    if dataset == "mimic3":
        return MIMIC3Dataset(root=root, tables=grid.TABLES[dataset], dev=dev)
    if dataset == "eicu":
        return eICUDataset(root=root, tables=grid.TABLES[dataset], dev=dev)
    raise ValueError(f"unknown dataset {dataset}")


# --- validation (coverage gate) ------------------------------------------

def coverage_tolerance(n_cal):
    """Coverage-check tolerance, scaled by calibration size."""
    if n_cal <= 0:
        return 0.25
    return min(0.25, max(0.02, 3.0 * math.sqrt(0.25 / n_cal)))


def validate_marginal(coverage, alpha, n_cal):
    """Overall coverage within tolerance of 1 - alpha."""
    tol = coverage_tolerance(n_cal)
    target = 1.0 - alpha
    ok = abs(coverage - target) <= tol
    return ok, f"cov={coverage:.3f} target={target:.3f} tol={tol:.3f}"


def validate_class_conditional(per_class_miscov, alpha, n_cal):
    """Every class's coverage (1 - miscov[k]) meets target 1 - alpha within tolerance."""
    tol = coverage_tolerance(n_cal)
    target = 1.0 - alpha
    worst_miscov = max(float(m) for m in per_class_miscov)
    ok = all((1.0 - float(m)) >= target - tol for m in per_class_miscov)
    return ok, f"worst_class_cov={1.0 - worst_miscov:.3f} target={target:.3f} tol={tol:.3f}"


# --- evaluation ----------------------------------------------------------

def evaluate(cal_scores, cal_y, test_scores, test_y, mode, alpha, n_classes):
    """Calibrate at (mode, alpha) on cached scores and score the test split.

    Set rule mirrors LABEL.forward: include class k iff score_k <= threshold (the
    per-class threshold broadcasts for class-conditional).
    Returns (coverage, avg_set_size, per_class_miscoverage).
    """
    t = conformal_threshold(cal_scores, cal_y, mode, alpha, n_classes)
    predset = test_scores <= t
    coverage = 1 - miscoverage_overall_ps(predset, test_y)
    return coverage, size(predset), miscoverage_ps(predset, test_y)


def _class_counts(split, label_key, n_classes):
    """Per-class counts for a split, length n_classes."""
    counts = [0] * n_classes
    for i in range(len(split)):
        k = int(split[i][label_key])
        if 0 <= k < n_classes:
            counts[k] += 1
    return counts


def _split_issue(name, counts):
    """Describe a split if it is empty or has < 2 classes present, else None."""
    n = sum(counts)
    present = sum(1 for c in counts if c > 0)
    if n == 0 or present < 2:
        return f"{name}(N={n}, classes={present})"
    return None


def _split_issue_fast(name, split, label_key):
    """Describe the split if it is empty or has < 2 classes present, else None."""
    n = len(split)
    if n == 0:
        return f"{name}(N=0, classes=0)"
    seen = set()
    for i in range(n):
        seen.add(int(split[i][label_key]))
        if len(seen) >= 2:
            return None
    return f"{name}(N={n}, classes={len(seen)})"


def _save_predictions(pred_cache, dataset, task, model_name, seed, cal, test):
    """Cache cal/test predictions (y_prob, y_true) so new methods/alphas need no retrain."""
    os.makedirs(pred_cache, exist_ok=True)
    path = os.path.join(pred_cache, f"{dataset}-{task}-{model_name}-seed{seed}.npz")
    np.savez_compressed(
        path,
        cal_prob=cal["y_prob"], cal_y=cal["y_true"],
        test_prob=test["y_prob"], test_y=test["y_true"],
    )


def _base_metrics(test):
    """Base model's test discriminative performance (a model property, not a method)."""
    try:
        m = get_metrics_fn("multiclass")(
            test["y_true"], test["y_prob"],
            metrics=["roc_auc_weighted_ovr", "f1_macro"],
        )
        return {"base_auroc": float(m["roc_auc_weighted_ovr"]),
                "base_f1": float(m["f1_macro"])}
    except Exception:
        return {"base_auroc": float("nan"), "base_f1": float("nan")}


def run_seed(samples, dataset, task, model_name, method_modes, alphas, seed, epochs,
             pred_cache=None):
    """Train the base model once and calibrate every (method, mode, alpha) for one seed.

    method_modes is a list of (method, mode) pairs; all share the single trained model.
    Returns (cal_counts, {(method, mode): {alpha: outcome}}, base_metrics); outcome status
    is ran / skipped / errored, with coverage / set_size / per_class_miscov / reason.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_data, val_data, cal_data, test_data = split_by_patient_conformal(
        samples, ratios=grid.RATIOS, seed=seed
    )
    label_key = grid.LABEL_KEY[task]
    n_classes = samples.output_processors[label_key].size()
    cal_counts = _class_counts(cal_data, label_key, n_classes)
    print(f"  [seed {seed}] cal per-class counts: {cal_counts} "
          f"(N train/val/test = {len(train_data)}/{len(val_data)}/{len(test_data)})")

    issues = []
    for name, split in (("train", train_data), ("val", val_data)):
        if iss := _split_issue_fast(name, split, label_key):
            issues.append(iss)
    if iss := _split_issue("cal", cal_counts):
        issues.append(iss)
    if iss := _split_issue_fast("test", test_data, label_key):
        issues.append(iss)
    if issues:
        reason = "insufficient samples (" + "; ".join(issues) + ")"
        skipped = {km: {a: {"status": "skipped", "reason": reason} for a in alphas}
                   for km in method_modes}
        return cal_counts, skipped, None

    train_loader = get_dataloader(train_data, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_data, batch_size=32, shuffle=False)

    model = MODELS[model_name](dataset=samples)
    Trainer(model=model, metrics=grid.METRICS[task]).train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor=grid.MONITOR[task],
        monitor_criterion="max",
    )

    model.eval()
    cal = prepare_numpy_dataset(model, cal_data, ["y_prob", "y_true"])
    test = prepare_numpy_dataset(model, test_data, ["y_prob", "y_true"])
    if pred_cache:
        _save_predictions(pred_cache, dataset, task, model_name, seed, cal, test)
    base = _base_metrics(test)
    present = {m for m, _ in method_modes}
    cal_scores = {m: SCORERS[m](cal["y_prob"]) for m in present}
    test_scores = {m: SCORERS[m](test["y_prob"]) for m in present}

    min_count = min(cal_counts)
    outcomes = {km: {} for km in method_modes}
    for method, mode in method_modes:
        for alpha in alphas:
            if mode == "class-conditional":
                required = math.ceil(1.0 / alpha)
                if min_count < required:
                    outcomes[(method, mode)][alpha] = {
                        "status": "skipped",
                        "reason": f"insufficient cal-class samples "
                                  f"(min_class={min_count} < ceil(1/alpha)={required})",
                    }
                    continue
            try:
                cov, set_size, per_class = evaluate(
                    cal_scores[method], cal["y_true"],
                    test_scores[method], test["y_true"],
                    mode, alpha, n_classes,
                )
                outcomes[(method, mode)][alpha] = {
                    "status": "ran",
                    "coverage": float(cov),
                    "set_size": float(set_size),
                    "per_class_miscov": [float(x) for x in per_class],
                }
            except Exception as exc:
                outcomes[(method, mode)][alpha] = {"status": "errored",
                                                   "reason": repr(exc)}
    return cal_counts, outcomes, base


# --- cell driver ---------------------------------------------------------

def _cell_id(dataset, task, model_name, method, mode):
    return f"{dataset}-{task}-{model_name}-{method}-{mode}"


def _aggregate(per_seed, key, alpha):
    """Aggregate one (method, mode) `key` at one alpha across seeds."""
    outs = [s[key][alpha] for s in per_seed]
    if {o["status"] for o in outs} != {"ran"}:
        return next(o for o in outs if o["status"] != "ran")
    covs = np.array([o["coverage"] for o in outs])
    sizes = np.array([o["set_size"] for o in outs])
    per_class = np.stack([o["per_class_miscov"] for o in outs]).mean(0)
    return {
        "status": "ran",
        "coverage_mean": float(covs.mean()),
        "coverage_std": float(covs.std()),
        "avg_set_size": float(sizes.mean()),
        "per_class_miscov": [float(x) for x in per_class],
    }


def _agg_base(bases, key):
    """Mean of a base metric across seeds; '' if none ran or all NaN."""
    vals = [b[key] for b in bases if b is not None and not math.isnan(b[key])]
    return round(float(np.mean(vals)), 4) if vals else ""


def run_cell(dataset, task, model_name, samples, methods, modes, alphas, seeds, epochs,
             demo=False, pred_cache=None):
    """Run all seeds for one (dataset, task, model); one row per (method, mode, alpha).

    Trains once per seed; the methods all calibrate off that model. LABEL uses `modes`;
    APS/RAPS are marginal only. Caches predictions per seed when pred_cache is set.
    """
    method_modes = [(method, mode)
                    for method in methods
                    for mode in (modes if method == "LABEL" else ["marginal"])]
    per_seed = []
    per_base = []
    cal_counts = None
    for seed in seeds:
        cal_counts, outcomes, base = run_seed(
            samples, dataset, task, model_name, method_modes, alphas, seed, epochs,
            pred_cache=pred_cache,
        )
        per_seed.append(outcomes)
        per_base.append(base)
    n_cal = sum(cal_counts) if cal_counts else 0
    base_auroc = _agg_base(per_base, "base_auroc")
    base_f1 = _agg_base(per_base, "base_f1")

    rows = []
    for method, mode in method_modes:
        for alpha in alphas:
            agg = _aggregate(per_seed, (method, mode), alpha)
            row = {
                "cell_id": _cell_id(dataset, task, model_name, method, mode),
                "dataset": dataset, "task": task,
                "output_type": grid.OUTPUT_TYPE[task], "model": model_name,
                "split": grid.SPLIT, "cal_separate_from_val": "yes",
                "method": method, "mode": mode, "alpha": alpha,
                "target_coverage": round(1 - alpha, 2),
                "coverage_mean": "", "coverage_std": "", "avg_set_size": "",
                "per_class_miscov": "", "worst_class_miscov": "",
                "monitor": grid.MONITOR[task], "seeds": f"{seeds[0]}-{seeds[-1]}",
                "status": agg["status"], "validation_passed": "",
                "run_id": "", "date_run": "", "commit_hash": "",
                "base_auroc": base_auroc, "base_f1": base_f1,
                "cal_counts": cal_counts, "detail": agg.get("reason", ""),
            }
            if agg["status"] == "ran":
                per_class = agg["per_class_miscov"]
                row["coverage_mean"] = round(agg["coverage_mean"], 4)
                row["coverage_std"] = round(agg["coverage_std"], 4)
                row["avg_set_size"] = round(agg["avg_set_size"], 4)
                row["per_class_miscov"] = " ".join(f"{x:.4f}" for x in per_class)
                row["worst_class_miscov"] = round(max(per_class), 4)
                if mode == "marginal":
                    ok, detail = validate_marginal(agg["coverage_mean"], alpha, n_cal)
                else:
                    ok, detail = validate_class_conditional(per_class, alpha, n_cal)
                row["validation_passed"] = "true" if ok else "false"
                row["detail"] = detail
                row["status"] = "ran" if demo else ("done" if ok else "flagged")
            rows.append(row)
    return rows
