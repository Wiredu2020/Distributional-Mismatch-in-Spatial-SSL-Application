"""Controlled-experiment runner: sweeps distribution mismatch, spatial
autocorrelation, and non-stationarity, fitting each SSL baseline and
recording performance + divergence metrics for every combination.

Usage:
    python -m ssl_spatial.experiments.controlled_experiment [config.yaml]
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from ssl_spatial.data.synthetic import SpatialSSLConfig, generate_spatial_ssl_dataset
from ssl_spatial.metrics.divergence import kl_divergence_kde, localized_divergence, mmd_rbf, wasserstein_marginal
from ssl_spatial.metrics.evaluation import evaluate_classifier
from ssl_spatial.models.baselines import BASELINE_METHODS

REPO_ROOT = Path(__file__).resolve().parents[3]


def run(config_path: str) -> pd.DataFrame:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    sweep = cfg["sweep"]
    methods = cfg["methods"]
    ds_cfg = cfg["dataset"]
    div_cfg = cfg["divergence"]

    combos = list(itertools.product(
        sweep["mismatch_alpha"], sweep["lengthscale"],
        sweep["nonstationarity_strength"], sweep["seeds"],
    ))

    rows = []
    start = time.time()
    for i, (alpha, lengthscale, nonstat, seed) in enumerate(combos):
        data_cfg = SpatialSSLConfig(
            mismatch_alpha=alpha,
            lengthscale=lengthscale,
            nonstationarity_strength=nonstat,
            label_anchor=tuple(ds_cfg["label_anchor"]),
            bandwidth_min=ds_cfg["bandwidth_min"],
            bandwidth_max=ds_cfg["bandwidth_max"],
            grid_resolution=ds_cfg["grid_resolution"],
            n_features=ds_cfg["n_features"],
            n_labelled=ds_cfg["n_labelled"],
            n_unlabelled=ds_cfg["n_unlabelled"],
            n_test_in=ds_cfg["n_test_in"],
            n_test_out=ds_cfg["n_test_out"],
            noise_std=ds_cfg["noise_std"],
        )
        dataset = generate_spatial_ssl_dataset(data_cfg, seed=seed)

        X_l, y_l = dataset.X[dataset.idx_labelled], dataset.y[dataset.idx_labelled]
        X_u = dataset.X[dataset.idx_unlabelled]
        X_in, y_in = dataset.X[dataset.idx_test_in], dataset.y[dataset.idx_test_in]
        X_out, y_out = dataset.X[dataset.idx_test_out], dataset.y[dataset.idx_test_out]
        c_l = dataset.coords[dataset.idx_labelled]
        c_u = dataset.coords[dataset.idx_unlabelled]

        kl = kl_divergence_kde(X_l, X_u)
        wass, _ = wasserstein_marginal(X_l, X_u)
        mmd = mmd_rbf(X_l, X_u)
        localized = localized_divergence(
            c_l, X_l, c_u, X_u,
            metric="mmd", n_bins=div_cfg["n_bins"], min_count=div_cfg["min_count"],
        )

        for method in methods:
            fit_fn = BASELINE_METHODS[method]
            clf = fit_fn(X_l, y_l, X_u, seed=seed)
            perf = evaluate_classifier(clf, X_in, y_in, X_out, y_out)

            rows.append({
                "mismatch_alpha": alpha,
                "lengthscale": lengthscale,
                "nonstationarity_strength": nonstat,
                "seed": seed,
                "method": method,
                "kl_divergence": kl,
                "wasserstein": wass,
                "mmd": mmd,
                "localized_mmd": localized["aggregate"],
                **perf,
            })

        if (i + 1) % 10 == 0 or (i + 1) == len(combos):
            elapsed = time.time() - start
            print(f"[{i + 1}/{len(combos)}] combos done ({elapsed:.1f}s elapsed)", file=sys.stderr)

    results = pd.DataFrame(rows)
    out_path = REPO_ROOT / cfg["output_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"Saved {len(results)} rows to {out_path}", file=sys.stderr)
    return results


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "configs" / "controlled_experiment.yaml")
    run(config_path)
