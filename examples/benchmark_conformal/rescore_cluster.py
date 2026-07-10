"""Re-score ClusterLabel (per-cluster conformal) off cached embeddings + predictions.

No GPU / no retraining. Reimplements PyHealth's ClusterLabel calibrate/forward in NumPy:
K-means on cached train+cal embeddings -> a `_query_quantile` threshold per cluster from
the cached LABEL scores (1 - p(true)) -> assign cached test embeddings to clusters -> the
prediction set uses each test point's cluster threshold. Same K-means params as PyHealth
(random_state=42, n_init=10) and the same `_query_quantile`, so it reproduces the class's
algorithm exactly (the class itself needs a live model for prepare_numpy_dataset /
extract_embeddings, so it can't run off-cache; `--validate` cross-checks the mechanism on
synthetic data and reports marginal coverage).

Runs on LOS only (the many-class, sparse-bucket case ClusterLabel targets). Requires an
embedding re-run (`--cache-embeddings`) to have populated ~/cp-preds/*-emb.npz.

Usage:
    python rescore_cluster.py --validate                     # synthetic sanity + coverage
    python rescore_cluster.py --pred-cache ~/cp-preds        # LOS re-score -> cluster_results.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re

import numpy as np
from sklearn.cluster import KMeans

import grid
from pyhealth.calib.predictionset.base_conformal import _query_quantile
from pyhealth.metrics.prediction_set import (
    miscoverage_overall_ps,
    miscoverage_ps,
    size,
)

EMB_RE = re.compile(r"(mimic3|mimic4|eicu)-los-(\w+)-seed(\d+)-emb\.npz$")


def fit_clusters(train_emb, cal_emb, test_emb, n_clusters, seed=42):
    """K-means on train+cal embeddings -- once per cell/seed (alpha-independent).
    Returns each cal and test point's cluster id."""
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(np.concatenate([train_emb, cal_emb], axis=0))
    return km.labels_[len(train_emb):], km.predict(test_emb)


def cluster_predset(cal_clusters, test_clusters, cal_prob, cal_y, test_prob,
                    alpha, n_clusters):
    """Per-cluster LABEL threshold (`_query_quantile` on that cluster's cached scores);
    each test point's set uses its own cluster's threshold. Cheap -- no clustering here."""
    cal_scores = 1.0 - cal_prob[np.arange(len(cal_y)), cal_y]        # LABEL non-conformity
    thresh = np.array([
        _query_quantile(cal_scores[cal_clusters == c], alpha)
        if np.any(cal_clusters == c) else np.inf
        for c in range(n_clusters)
    ])
    test_scores = 1.0 - test_prob                                   # (N, K)
    return test_scores <= thresh[test_clusters][:, None]           # (N, K) bool


def _metrics(predset, test_y, alpha):
    cov = 1.0 - miscoverage_overall_ps(predset, test_y)
    per_class = miscoverage_ps(predset, test_y)
    return {
        "coverage_mean": round(float(cov), 4),
        "avg_set_size": round(float(size(predset)), 4),
        "per_class_miscov": " ".join(f"{x:.4f}" for x in per_class),
        "worst_class_miscov": round(float(max(per_class)), 4),
        "worst_class": int(np.argmax(per_class)),
        "rejection_rate": round(float(np.mean(predset.sum(1) == 0)), 4),
    }


def validate(n_clusters):
    """Synthetic sanity: the mechanism runs, clusters get distinct thresholds, and marginal
    coverage lands near 1 - alpha on exchangeable data."""
    rng = np.random.default_rng(0)
    K, D = 10, 16
    def probs(n):
        z = rng.normal(size=(n, K)); e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    tr, ca, te = rng.normal(size=(3000, D)), rng.normal(size=(2000, D)), rng.normal(size=(4000, D))
    cp, tp = probs(2000), probs(4000)
    cy, ty = rng.integers(0, K, 2000), rng.integers(0, K, 4000)
    print(f"ClusterLabel synthetic check (n_clusters={n_clusters}):")
    cal_cl, test_cl = fit_clusters(tr, ca, te, n_clusters)
    for a in [0.2, 0.1, 0.05]:
        ps = cluster_predset(cal_cl, test_cl, cp, cy, tp, a, n_clusters)
        m = _metrics(ps, ty, a)
        print(f"  alpha={a}: coverage={m['coverage_mean']} (target {1-a:.2f})  "
              f"size={m['avg_set_size']}  rejection={m['rejection_rate']}")
    print("OK -- runs, per-cluster thresholds applied, coverage ~ target.")


def rescore(pred_cache, n_clusters, out):
    cells = collections.defaultdict(dict)   # (ds, model) -> {seed: prefix}
    for f in glob.glob(os.path.join(pred_cache, "*-los-*-seed*-emb.npz")):
        m = EMB_RE.match(os.path.basename(f))
        if m:
            cells[(m.group(1), m.group(2))][int(m.group(3))] = f[:-len("-emb.npz")]

    if not cells:
        raise SystemExit(f"no LOS embedding npz in {pred_cache} -- run the embedding re-run first")

    rows, missing = [], []
    for (ds, model), seedmap in sorted(cells.items()):
        per_seed = {}    # seed -> (cal_clusters, test_clusters, preds); cluster once per seed
        for seed, prefix in sorted(seedmap.items()):
            pred_path = prefix + ".npz"
            if not os.path.exists(pred_path):
                missing.append((ds, model, seed)); continue
            emb = np.load(prefix + "-emb.npz"); pred = np.load(pred_path)
            cal_cl, test_cl = fit_clusters(emb["train_emb"], emb["cal_emb"],
                                           emb["test_emb"], n_clusters)
            per_seed[seed] = (cal_cl, test_cl, pred)
        if not per_seed:
            continue
        for alpha in grid.ALPHAS:
            agg = collections.defaultdict(list)
            for cal_cl, test_cl, pred in per_seed.values():
                ps = cluster_predset(cal_cl, test_cl, pred["cal_prob"], pred["cal_y"],
                                     pred["test_prob"], alpha, n_clusters)
                m = _metrics(ps, pred["test_y"], alpha)
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        agg[k].append(v)
            rows.append({
                "cell_id": f"{ds}-los-{model}-ClusterLabel-cluster",
                "dataset": ds, "task": "los", "model": model,
                "method": "ClusterLabel", "mode": "cluster", "alpha": alpha,
                "n_clusters": n_clusters,
                "coverage_mean": round(float(np.mean(agg["coverage_mean"])), 4),
                "avg_set_size": round(float(np.mean(agg["avg_set_size"])), 4),
                "worst_class_miscov": round(float(np.mean(agg["worst_class_miscov"])), 4),
                "rejection_rate": round(float(np.mean(agg["rejection_rate"])), 4),
            })
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}: {len(rows)} ClusterLabel LOS rows")
    if missing:
        print(f"!! missing prediction npz for {len(missing)} (ds,model,seed) -- "
              f"emb present but preds absent: {missing[:10]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-cache", default=os.path.expanduser("~/cp-preds"))
    ap.add_argument("--n-clusters", type=int, default=5)
    ap.add_argument("--out", default="cluster_results.csv")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if a.validate:
        validate(a.n_clusters)
    else:
        rescore(a.pred_cache, a.n_clusters, a.out)


if __name__ == "__main__":
    main()
