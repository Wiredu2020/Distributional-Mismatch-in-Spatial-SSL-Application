"""Re-validates the localised-divergence estimator (manuscript methods.tex
Section 2.3 ToDo) with the kernel-weighted fix, against the same
correlation analysis used for the original fixed-grid estimator and the
three global metrics.

Regenerates the synthetic pilot's 108 (alpha, lengthscale, nonstationarity,
seed) combinations from `configs/controlled_experiment.yaml` (same seeds,
so the same datasets), computes the new kernel-weighted local MMD for each,
and merges the result into the existing `results/controlled_experiment.csv`
by (alpha, lengthscale, nonstationarity, seed) -- the divergence metric
does not depend on which SSL method is being evaluated, so one computation
per combination covers all methods' rows.

Usage:
    python -m ssl_spatial.experiments.localized_mmd_analysis
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd
import yaml

from ssl_spatial.data.synthetic import SpatialSSLConfig, generate_spatial_ssl_dataset
from ssl_spatial.metrics.divergence import kernel_weighted_local_mmd

REPO_ROOT = Path(__file__).resolve().parents[3]


def run(config_path: str | None = None, results_csv: str | None = None) -> pd.DataFrame:
    config_path = config_path or str(REPO_ROOT / "configs" / "controlled_experiment.yaml")
    results_csv = results_csv or str(REPO_ROOT / "results" / "controlled_experiment.csv")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sweep, ds_cfg, div_cfg = cfg["sweep"], cfg["dataset"], cfg["divergence"]

    combos = list(itertools.product(
        sweep["mismatch_alpha"], sweep["lengthscale"], sweep["nonstationarity_strength"], sweep["seeds"],
    ))

    rows = []
    for i, (alpha, lengthscale, nonstat, seed) in enumerate(combos):
        data_cfg = SpatialSSLConfig(
            mismatch_alpha=alpha, lengthscale=lengthscale, nonstationarity_strength=nonstat,
            label_anchor=tuple(ds_cfg["label_anchor"]), bandwidth_min=ds_cfg["bandwidth_min"],
            bandwidth_max=ds_cfg["bandwidth_max"], grid_resolution=ds_cfg["grid_resolution"],
            n_features=ds_cfg["n_features"], n_labelled=ds_cfg["n_labelled"],
            n_unlabelled=ds_cfg["n_unlabelled"], n_test_in=ds_cfg["n_test_in"],
            n_test_out=ds_cfg["n_test_out"], noise_std=ds_cfg["noise_std"],
        )
        dataset = generate_spatial_ssl_dataset(data_cfg, seed=seed)
        X_l, X_u = dataset.X[dataset.idx_labelled], dataset.X[dataset.idx_unlabelled]
        c_l, c_u = dataset.coords[dataset.idx_labelled], dataset.coords[dataset.idx_unlabelled]

        result = kernel_weighted_local_mmd(c_l, X_l, c_u, X_u, n_grid=div_cfg["n_bins"])
        rows.append({
            "mismatch_alpha": alpha, "lengthscale": lengthscale,
            "nonstationarity_strength": nonstat, "seed": seed,
            "kernel_weighted_local_mmd": result["aggregate"],
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(combos):
            print(f"[{i + 1}/{len(combos)}] combos done", file=sys.stderr)

    new_metric = pd.DataFrame(rows)
    existing = pd.read_csv(results_csv)
    merged = existing.merge(new_metric, on=["mismatch_alpha", "lengthscale", "nonstationarity_strength", "seed"])

    correlations = merged[["kl_divergence", "wasserstein", "mmd", "localized_mmd", "kernel_weighted_local_mmd",
                            "accuracy_out_region"]].corr()["accuracy_out_region"].drop("accuracy_out_region")
    print("\nPooled Pearson correlation with out-of-region accuracy (all runs):")
    print(correlations.round(3).to_string())

    out_path = REPO_ROOT / "results" / "localized_mmd_comparison.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved merged comparison to {out_path}")
    return merged


if __name__ == "__main__":
    run()
