"""WILDS PovertyMap benchmark runner -- the manuscript's "Benchmark Dataset
Evaluation" ToDo (methods.tex): re-run the same controlled-mismatch
methodology, baseline SSL methods, and evaluation metrics used for the
synthetic pilot on an established geographic distribution-shift benchmark,
to check that the threshold-like degradation and method ranking found on
synthetic data (Table tab:h1) are not an artefact of the synthetic
generator.

Usage:
    python -m ssl_spatial.experiments.wilds_benchmark [config.yaml]

Assumes the images referenced by the sweep's pools have already been
downloaded into `<data_dir>/images/` (see notebooks/benchmark.ipynb or call
`ssl_spatial.data.wilds_poverty.download_images` directly first) -- this
script does not perform any network I/O itself.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ssl_spatial.data.wilds_poverty import (
    all_pool_indices, band_summary_features, build_feature_matrix, build_pools,
    download_metadata, labelled_indices_for_alpha, reduce_dimensionality,
)
from ssl_spatial.metrics.divergence import kl_divergence_kde, mmd_rbf, wasserstein_marginal
from ssl_spatial.metrics.evaluation import evaluate_classifier
from ssl_spatial.models.baselines import BASELINE_METHODS
from ssl_spatial.models.distribution_aware import DISTRIBUTION_AWARE_METHODS

REPO_ROOT = Path(__file__).resolve().parents[3]
ALL_METHODS = {**BASELINE_METHODS, **DISTRIBUTION_AWARE_METHODS}


def _standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def run(config_path: str) -> pd.DataFrame:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = REPO_ROOT / cfg["data_dir"]
    wcfg = cfg["wilds"]
    sweep = cfg["sweep"]
    methods = cfg["methods"]

    metadata = download_metadata(data_dir)
    pools = build_pools(
        metadata, fold=wcfg["fold"], candidate_per_country=wcfg["candidate_per_country"],
        n_unlabelled=wcfg["n_unlabelled"], n_test_in=wcfg["n_test_in"],
        n_test_out=wcfg["n_test_out"], seed=wcfg["pool_seed"],
    )
    needed = all_pool_indices(pools)
    missing = [i for i in needed if not (data_dir / "images" / f"landsat_poverty_img_{i}.npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(needed)} pool images not downloaded yet; "
            f"run ssl_spatial.data.wilds_poverty.download_images(all_pool_indices(pools), data_dir) first."
        )

    # Binary label: above/below the global median asset-wealth index, computed
    # over the full metadata table so the threshold doesn't depend on which
    # subset happens to be sampled at a given alpha.
    wealth_median = metadata["wealthpooled"].median()
    y_all = (metadata["wealthpooled"] > wealth_median).astype(int)

    # Feature cache: build once for every index referenced anywhere in the pools,
    # then standardize and PCA-reduce once so every subset drawn downstream (at
    # any alpha/seed) shares the same fixed feature space.
    X_cache, valid_idx = build_feature_matrix(needed, data_dir)
    idx_to_row = {int(idx): row for row, idx in enumerate(valid_idx)}

    mean, std = X_cache.mean(axis=0), X_cache.std(axis=0) + 1e-12
    X_std = _standardize(X_cache, mean, std)
    n_components = cfg.get("n_pca_components", 5)
    X_reduced, _pca = reduce_dimensionality(X_std, n_components=n_components)
    print(f"PCA: {X_std.shape[1]} band-summary features -> {n_components} components "
          f"({_pca.explained_variance_ratio_.sum():.1%} variance explained)", file=sys.stderr)

    def features_for(idx_array: np.ndarray) -> np.ndarray:
        rows = [idx_to_row[int(i)] for i in idx_array]
        return X_reduced[rows]

    def labels_for(idx_array: np.ndarray) -> np.ndarray:
        return y_all.iloc[idx_array].to_numpy()

    X_u = features_for(pools["unlabelled_idx"])
    X_in, y_in = features_for(pools["test_in_idx"]), labels_for(pools["test_in_idx"])
    X_out, y_out = features_for(pools["test_out_idx"]), labels_for(pools["test_out_idx"])

    rows = []
    combos = [(a, s) for a in sweep["mismatch_alpha"] for s in sweep["seeds"]]
    start = time.time()
    for i, (alpha, seed) in enumerate(combos):
        labelled_idx = labelled_indices_for_alpha(pools, alpha, sweep["n_labelled"], seed)
        X_l, y_l = features_for(labelled_idx), labels_for(labelled_idx)

        kl = kl_divergence_kde(X_l, X_u)
        wass, _ = wasserstein_marginal(X_l, X_u)
        mmd = mmd_rbf(X_l, X_u)

        for method in methods:
            fit_fn = ALL_METHODS[method]
            clf = fit_fn(X_l, y_l, X_u, seed=seed)
            perf = evaluate_classifier(clf, X_in, y_in, X_out, y_out)

            rows.append({
                "mismatch_alpha": alpha,
                "n_active_countries": len(pools["train_countries"][: _k(len(pools["train_countries"]), alpha)]),
                "seed": seed,
                "method": method,
                "kl_divergence": kl,
                "wasserstein": wass,
                "mmd": mmd,
                "n_labelled": len(labelled_idx),
                **perf,
            })

        if (i + 1) % 5 == 0 or (i + 1) == len(combos):
            elapsed = time.time() - start
            print(f"[{i + 1}/{len(combos)}] combos done ({elapsed:.1f}s elapsed)", file=sys.stderr)

    results = pd.DataFrame(rows)
    out_path = REPO_ROOT / cfg["output_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"Saved {len(results)} rows to {out_path}", file=sys.stderr)
    return results


def _k(n_train_countries: int, alpha: float) -> int:
    from ssl_spatial.data.wilds_poverty import n_active_countries
    return n_active_countries(n_train_countries, alpha)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "configs" / "wilds_benchmark.yaml")
    run(config_path)
