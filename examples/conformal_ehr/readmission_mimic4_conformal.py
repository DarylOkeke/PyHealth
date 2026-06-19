"""
Split conformal prediction for 30-day readmission on MIMIC-IV.

This example demonstrates:
1. Training a Transformer on the MIMIC-IV readmission task (30-day window), presented
   as 2-class multiclass (see ReadmissionPredictionMIMIC4Multiclass) so LABEL applies.
2. Wrapping the trained model with LABEL (split conformal prediction) in marginal and
   class-conditional modes, each with a coverage guarantee of 1 - alpha.
3. Comparing the two modes via overall coverage, average set size, and per-class
   miscoverage, averaged over multiple random seeds.

Usage:
    # Full dataset
    python readmission_mimic4_conformal.py --root /path/to/mimiciv/2.2

    # Quick smoke test on a subsampled dataset
    python readmission_mimic4_conformal.py --dev --epochs 1 --seeds 0
"""

from __future__ import annotations

import argparse
import logging
import random
from datetime import timedelta

import numpy as np
import torch

from pyhealth.calib.predictionset import LABEL
from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient_conformal,
)
from pyhealth.metrics.prediction_set import (
    miscoverage_overall_ps,
    miscoverage_ps,
    size,
)
from pyhealth.models import Transformer
from pyhealth.tasks import ReadmissionPredictionMIMIC4
from pyhealth.trainer import Trainer

# Quiet PyHealth's per-init model summary logging.
logging.getLogger("pyhealth").setLevel(logging.WARNING)

READMISSION_WINDOW = timedelta(days=30)

# Readmitted is class index 1.
READMIT_CLASS = 1


class ReadmissionPredictionMIMIC4Multiclass(ReadmissionPredictionMIMIC4):
    """Readmission presented as 2-class multiclass so LABEL applies.

    Conformal methods need a softmax (n, K) matrix, not binary's sigmoid (n, 1).
    Mirrors EEGAbnormalTUAB in pyhealth/tasks/temple_university_EEG_tasks.py.
    """

    task_name = "ReadmissionPredictionMIMIC4Multiclass"
    output_schema = {"readmission": "multiclass"}


def _count_positives(split) -> tuple[int, int]:
    """Return (n_samples, n_readmitted) for a split."""
    n = len(split)
    readmitted = sum(int(split[i]["readmission"]) for i in range(n))
    return n, readmitted


def _evaluate(model, alpha, cal_data, test_loader) -> tuple:
    """Calibrate LABEL at `alpha` and evaluate on the test split.

    Float `alpha` -> marginal coverage; per-class list -> class-conditional.
    Returns (coverage, avg_set_size, per_class_miscoverage).
    """
    cal_model = LABEL(model, alpha=alpha)
    cal_model.calibrate(cal_dataset=cal_data)
    y_true, _, _, extra = Trainer(model=cal_model, enable_logging=False).inference(
        test_loader, additional_outputs=["y_predset"]
    )
    predset = extra["y_predset"]
    y_true = np.asarray(y_true)
    coverage = 1 - miscoverage_overall_ps(predset, y_true)
    return coverage, size(predset), miscoverage_ps(predset, y_true)


def run_seed(samples, seed: int, alphas: list[float], epochs: int) -> dict:
    """Train the base model and run split conformal prediction for a single seed.

    Returns {method: {alpha: (coverage, avg_set_size, per_class_miscoverage)}} on the
    test split, for method in {"marginal", "class-conditional"}.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Train / validation / calibration / test split.
    train_data, val_data, cal_data, test_data = split_by_patient_conformal(
        samples, ratios=[0.6, 0.1, 0.1, 0.2], seed=seed
    )

    print(f"  [seed {seed}] readmission counts per split:")
    for name, split in [("train", train_data), ("val", val_data),
                        ("cal", cal_data), ("test", test_data)]:
        n, pos = _count_positives(split)
        rate = f"{pos / n:.1%}" if n else "n/a"
        print(f"    {name:>5}: {pos}/{n} readmitted ({rate})")

    train_loader = get_dataloader(train_data, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_data, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_data, batch_size=32, shuffle=False)

    model = Transformer(dataset=samples)
    Trainer(
        model=model,
        metrics=["roc_auc_weighted_ovr", "f1_macro", "accuracy"],
    ).train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor="roc_auc_weighted_ovr",
        monitor_criterion="max",
    )

    results = {"marginal": {}, "class-conditional": {}}
    for alpha in alphas:
        results["marginal"][alpha] = _evaluate(model, alpha, cal_data, test_loader)
        results["class-conditional"][alpha] = _evaluate(
            model, [alpha, alpha], cal_data, test_loader
        )
    return results


def _report(method: str, per_alpha: dict, alphas: list[float], n_seeds: int) -> None:
    """Print the coverage table and per-class miscoverage for one LABEL mode."""
    print(f"\n=== {method} LABEL (mean +/- std over {n_seeds} seeds) ===")
    print("alpha  target  coverage      avg_set_size")
    for a in alphas:
        cov = np.array([r[0] for r in per_alpha[a]])
        sizes = np.array([r[1] for r in per_alpha[a]])
        print(f"{a:.2f}    {1 - a:.0%}    {cov.mean():.2f} +/- {cov.std():.2f}   "
              f"{sizes.mean():.1f} +/- {sizes.std():.1f}")
    print(f"per-class miscoverage_ps (index {READMIT_CLASS} = readmitted):")
    for a in alphas:
        per_class = np.stack([r[2] for r in per_alpha[a]]).mean(0)
        arr = np.array2string(per_class, precision=2, floatmode="fixed")
        print(f"alpha={a:.2f}: {arr}  -> readmitted = {per_class[READMIT_CLASS]:.2f}")


def main(
    root: str,
    seeds: list[int],
    alphas: list[float],
    epochs: int,
    dev: bool,
) -> None:
    dataset = MIMIC4Dataset(
        ehr_root=root,
        ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=dev,
    )
    samples = dataset.set_task(
        ReadmissionPredictionMIMIC4Multiclass(window=READMISSION_WINDOW)
    )
    print(f"Samples: {len(samples)}")

    # Aggregate per method across seeds.
    methods = ["marginal", "class-conditional"]
    agg = {m: {a: [] for a in alphas} for m in methods}
    for seed in seeds:
        results = run_seed(samples, seed, alphas, epochs)
        for m in methods:
            for a in alphas:
                agg[m][a].append(results[m][a])

    for m in methods:
        _report(m, agg[m], alphas, len(seeds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split conformal prediction for MIMIC-IV 30-day readmission."
    )
    parser.add_argument(
        "--root",
        default="/srv/local/data/physionet.org/files/mimiciv/2.2",
        help="MIMIC-IV root (the folder containing hosp/).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs per seed.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated random seeds to average over.",
    )
    parser.add_argument(
        "--alphas",
        default="0.2,0.1,0.05,0.01",
        help="Comma-separated target miscoverage rates, e.g. '0.2,0.1,0.05,0.01'.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use a subsampled dataset for a quick smoke test.",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]
    main(args.root, seeds, alphas, args.epochs, args.dev)
