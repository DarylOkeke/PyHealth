"""Split-conformal benchmark engine for PyHealth EHR tasks.

Generalizes the three reference cells (full_examples/{los,mortality,readmission}_mimic4_
conformal.py) into one pipeline over dataset x task x model x method. run_cell trains the
base model per seed, calibrates each (mode, alpha), and returns one result row per
(mode, alpha); run_grid.py drives it and writes results.csv.
"""

from __future__ import annotations

import logging
import math
import random
from datetime import timedelta

import numpy as np
import torch

import grid
from pyhealth.calib.predictionset import LABEL
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
from pyhealth.trainer import Trainer

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


def label_calibrator(model, alpha):
    """LABEL set predictor."""
    return LABEL(model, alpha=alpha)


METHODS = {"LABEL": label_calibrator}


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


def load_samples(dataset, task, root, dev):
    """Load the base dataset and apply the task, returning the SampleDataset."""
    if dataset == "mimic4":
        base = MIMIC4Dataset(ehr_root=root, ehr_tables=grid.TABLES[dataset], dev=dev)
    elif dataset == "mimic3":
        base = MIMIC3Dataset(root=root, tables=grid.TABLES[dataset], dev=dev)
    elif dataset == "eicu":
        base = eICUDataset(root=root, tables=grid.TABLES[dataset], dev=dev)
    else:
        raise ValueError(f"unknown dataset {dataset}")
    return base.set_task(build_task(dataset, task))


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

def _alpha_arg(mode, alpha, n_classes):
    """marginal -> float; class-conditional -> per-class array of length n_classes."""
    return alpha if mode == "marginal" else [alpha] * n_classes


def evaluate(model, method, mode, alpha, n_classes, cal_data, test_loader):
    """Calibrate `method` at `alpha`/`mode` and evaluate on the test split.

    Returns (coverage, avg_set_size, per_class_miscoverage).
    """
    calibrator = METHODS[method](model, _alpha_arg(mode, alpha, n_classes))
    calibrator.calibrate(cal_dataset=cal_data)
    y_true, _, _, extra = Trainer(model=calibrator, enable_logging=False).inference(
        test_loader, additional_outputs=["y_predset"]
    )
    predset = extra["y_predset"]
    y_true = np.asarray(y_true)
    coverage = 1 - miscoverage_overall_ps(predset, y_true)
    return coverage, size(predset), miscoverage_ps(predset, y_true)


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


def run_seed(samples, task, model_name, method, modes, alphas, seed, epochs):
    """Train the base model and calibrate every (mode, alpha) for one seed.

    Returns (cal_counts, {mode: {alpha: outcome}}); outcome status is
    ran / skipped / errored, with coverage / set_size / per_class_miscov / reason.
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
        skipped = {m: {a: {"status": "skipped", "reason": reason} for a in alphas}
                   for m in modes}
        return cal_counts, skipped

    train_loader = get_dataloader(train_data, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_data, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_data, batch_size=32, shuffle=False)

    model = MODELS[model_name](dataset=samples)
    Trainer(model=model, metrics=grid.METRICS[task]).train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor=grid.MONITOR[task],
        monitor_criterion="max",
    )

    min_count = min(cal_counts)
    outcomes = {mode: {} for mode in modes}
    for mode in modes:
        for alpha in alphas:
            if mode == "class-conditional":
                required = math.ceil(1.0 / alpha)
                if min_count < required:
                    outcomes[mode][alpha] = {
                        "status": "skipped",
                        "reason": f"insufficient cal-class samples "
                                  f"(min_class={min_count} < ceil(1/alpha)={required})",
                    }
                    continue
            try:
                cov, set_size, per_class = evaluate(
                    model, method, mode, alpha, n_classes, cal_data, test_loader
                )
                outcomes[mode][alpha] = {
                    "status": "ran",
                    "coverage": float(cov),
                    "set_size": float(set_size),
                    "per_class_miscov": [float(x) for x in per_class],
                }
            except Exception as exc:
                outcomes[mode][alpha] = {"status": "errored", "reason": repr(exc)}
    return cal_counts, outcomes


# --- cell driver ---------------------------------------------------------

def _cell_id(dataset, task, model_name, method, mode):
    return f"{dataset}-{task}-{model_name}-{method}-{mode}"


def _aggregate(per_seed, mode, alpha):
    """Aggregate one (mode, alpha) across seeds."""
    outs = [s[mode][alpha] for s in per_seed]
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


def run_cell(dataset, task, model_name, method, modes, alphas, seeds, epochs,
             root, dev, demo=False):
    """Run all seeds for one cell; return one row per (mode, alpha)."""
    samples = load_samples(dataset, task, root, dev)
    print(f"Samples: {len(samples)}")
    per_seed = []
    cal_counts = None
    for seed in seeds:
        cal_counts, outcomes = run_seed(
            samples, task, model_name, method, modes, alphas, seed, epochs
        )
        per_seed.append(outcomes)
    n_cal = sum(cal_counts) if cal_counts else 0

    rows = []
    for mode in modes:
        for alpha in alphas:
            agg = _aggregate(per_seed, mode, alpha)
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
