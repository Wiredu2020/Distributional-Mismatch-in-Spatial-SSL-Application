"""Comprehensive comparative study (manuscript methods.tex Section 2.4):
every SSL method now implemented -- the four baselines already evaluated in
the synthetic/WILDS/real-world sweeps, the two modern consistency-based
methods (Mean Teacher, FixMatch), and the three spatially explicit models
needed for H3 (GCN, spatial GNN, GNNWR), plus the Fouedjio geostatistical
SSL method -- run through the same mismatch-severity sweep across all five
datasets (synthetic, WILDS PovertyMap, housing, socio-economic, air
quality), via the uniform method-registry interface
(`ssl_spatial.models.method_registry`).

This module supplies one `get_split(alpha, seed) -> dict` closure per
dataset (reusing each dataset's existing loader/pool-construction code
unchanged) and a single sweep executor shared across all of them, so the
comparison is apples-to-apples: identical divergence/evaluation metrics,
identical severity schedule, identical seeds.

Usage:
    python -m ssl_spatial.experiments.spatial_methods_comparison configs/spatial_methods_comparison.yaml
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ssl_spatial.metrics.divergence import kl_divergence_kde, mmd_rbf, wasserstein_marginal
from ssl_spatial.metrics.evaluation import evaluate_from_proba
from ssl_spatial.models.method_registry import ALL_METHODS_UNIFORM

REPO_ROOT = Path(__file__).resolve().parents[3]


def _standardize(X: np.ndarray) -> np.ndarray:
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-12
    return (X - mean) / std


def _make_synthetic_split(dcfg: dict):
    from ssl_spatial.data.synthetic import SpatialSSLConfig, generate_spatial_ssl_dataset

    ds_cfg = dcfg.get("dataset", {})

    def get_split(alpha: float, seed: int) -> dict:
        cfg = SpatialSSLConfig(
            mismatch_alpha=alpha,
            lengthscale=ds_cfg.get("lengthscale", 0.08),
            nonstationarity_strength=ds_cfg.get("nonstationarity_strength", 0.0),
            n_labelled=ds_cfg.get("n_labelled", 150),
            n_unlabelled=ds_cfg.get("n_unlabelled", 600),
            n_test_in=ds_cfg.get("n_test_in", 300),
            n_test_out=ds_cfg.get("n_test_out", 300),
        )
        data = generate_spatial_ssl_dataset(cfg, seed=seed)
        return {
            "X_l": data.X[data.idx_labelled], "y_l": data.y[data.idx_labelled],
            "coords_l": data.coords[data.idx_labelled],
            "X_u": data.X[data.idx_unlabelled], "coords_u": data.coords[data.idx_unlabelled],
            "X_test_in": data.X[data.idx_test_in], "y_test_in": data.y[data.idx_test_in],
            "coords_test_in": data.coords[data.idx_test_in],
            "X_test_out": data.X[data.idx_test_out], "y_test_out": data.y[data.idx_test_out],
            "coords_test_out": data.coords[data.idx_test_out],
            "n_active_units": None,
        }

    return get_split


def _make_wilds_split(dcfg: dict):
    from ssl_spatial.data.wilds_poverty import (
        all_pool_indices, build_feature_matrix, build_pools, download_metadata,
        labelled_indices_for_alpha, n_active_countries, reduce_dimensionality,
    )

    data_dir = REPO_ROOT / dcfg["data_dir"]
    wcfg = dcfg["wilds"]
    metadata = download_metadata(data_dir).reset_index(drop=True)
    pools = build_pools(
        metadata, fold=wcfg["fold"], candidate_per_country=wcfg["candidate_per_country"],
        n_unlabelled=wcfg["n_unlabelled"], n_test_in=wcfg["n_test_in"],
        n_test_out=wcfg["n_test_out"], seed=wcfg["pool_seed"],
    )
    needed = all_pool_indices(pools)
    missing = [i for i in needed if not (data_dir / "images" / f"landsat_poverty_img_{i}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)}/{len(needed)} WILDS pool images not downloaded yet.")

    wealth_median = metadata["wealthpooled"].median()
    y_all = (metadata["wealthpooled"] > wealth_median).astype(int)
    X_cache, valid_idx = build_feature_matrix(needed, data_dir)
    idx_to_row = {int(idx): row for row, idx in enumerate(valid_idx)}
    X_reduced, _pca = reduce_dimensionality(_standardize(X_cache), n_components=dcfg.get("n_pca_components", 5))
    coords_all = metadata[["lat", "lon"]].to_numpy()

    def features_for(idx):
        return X_reduced[[idx_to_row[int(i)] for i in idx]]

    def labels_for(idx):
        return y_all.iloc[idx].to_numpy()

    def coords_for(idx):
        return coords_all[idx]

    X_u = features_for(pools["unlabelled_idx"])
    coords_u = coords_for(pools["unlabelled_idx"])
    X_test_in, y_test_in = features_for(pools["test_in_idx"]), labels_for(pools["test_in_idx"])
    coords_test_in = coords_for(pools["test_in_idx"])
    X_test_out, y_test_out = features_for(pools["test_out_idx"]), labels_for(pools["test_out_idx"])
    coords_test_out = coords_for(pools["test_out_idx"])
    n_train = len(pools["train_countries"])

    def get_split(alpha: float, seed: int) -> dict:
        labelled_idx = labelled_indices_for_alpha(pools, alpha, dcfg["sweep_n_labelled"], seed)
        return {
            "X_l": features_for(labelled_idx), "y_l": labels_for(labelled_idx),
            "coords_l": coords_for(labelled_idx),
            "X_u": X_u, "coords_u": coords_u,
            "X_test_in": X_test_in, "y_test_in": y_test_in, "coords_test_in": coords_test_in,
            "X_test_out": X_test_out, "y_test_out": y_test_out, "coords_test_out": coords_test_out,
            "n_active_units": n_active_countries(n_train, alpha),
        }

    return get_split


def _make_real_world_split(dataset_name: str, dcfg: dict):
    from ssl_spatial.data.region_mismatch import (
        build_region_pools, labelled_indices_for_alpha, n_active_regions,
    )
    from ssl_spatial.experiments.real_world_benchmark import _load_dataset, _train_ood_region_split

    df, feature_names = _load_dataset(dataset_name, dcfg)
    rcfg = dcfg["regions"]
    train_regions, ood_regions = _train_ood_region_split(df, rcfg["train_fraction"], rcfg["split_seed"])
    pools = build_region_pools(
        df["region"].to_numpy(), train_regions, ood_regions,
        candidate_per_region=rcfg["candidate_per_region"], n_unlabelled=rcfg["n_unlabelled"],
        n_test_in=rcfg["n_test_in"], n_test_out=rcfg["n_test_out"], seed=rcfg["pool_seed"],
    )
    X_all = _standardize(df[feature_names].to_numpy(dtype=float))
    y_all = df["label"].to_numpy()
    coords_all = df[["lat", "lon"]].to_numpy()
    n_train = len(pools["train_regions"])

    X_u, coords_u = X_all[pools["unlabelled_idx"]], coords_all[pools["unlabelled_idx"]]
    X_test_in, y_test_in = X_all[pools["test_in_idx"]], y_all[pools["test_in_idx"]]
    coords_test_in = coords_all[pools["test_in_idx"]]
    X_test_out, y_test_out = X_all[pools["test_out_idx"]], y_all[pools["test_out_idx"]]
    coords_test_out = coords_all[pools["test_out_idx"]]

    def get_split(alpha: float, seed: int) -> dict:
        labelled_idx = labelled_indices_for_alpha(pools, alpha, dcfg["sweep_n_labelled"], seed)
        return {
            "X_l": X_all[labelled_idx], "y_l": y_all[labelled_idx], "coords_l": coords_all[labelled_idx],
            "X_u": X_u, "coords_u": coords_u,
            "X_test_in": X_test_in, "y_test_in": y_test_in, "coords_test_in": coords_test_in,
            "X_test_out": X_test_out, "y_test_out": y_test_out, "coords_test_out": coords_test_out,
            "n_active_units": n_active_regions(n_train, alpha),
        }

    return get_split


_SPLIT_FACTORIES = {
    "synthetic": _make_synthetic_split,
    "wilds": _make_wilds_split,
    "housing": lambda dcfg: _make_real_world_split("housing", dcfg),
    "socioeconomic": lambda dcfg: _make_real_world_split("socioeconomic", dcfg),
    "air_quality": lambda dcfg: _make_real_world_split("air_quality", dcfg),
}


def run_dataset_sweep(dataset_name: str, get_split, sweep: dict, methods: list[str],
                       progress_every: int = 5) -> pd.DataFrame:
    rows = []
    combos = [(a, s) for a in sweep["mismatch_alpha"] for s in sweep["seeds"]]
    start = time.time()
    for i, (alpha, seed) in enumerate(combos):
        split = get_split(alpha, seed)
        kl = kl_divergence_kde(split["X_l"], split["X_u"])
        wass, _ = wasserstein_marginal(split["X_l"], split["X_u"])
        mmd = mmd_rbf(split["X_l"], split["X_u"])

        for method in methods:
            fit_predict = ALL_METHODS_UNIFORM[method]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                proba_in, proba_out = fit_predict(
                    split["X_l"], split["y_l"], split["X_u"],
                    split["X_test_in"], split["X_test_out"],
                    split["coords_l"], split["coords_u"],
                    split["coords_test_in"], split["coords_test_out"],
                    seed=seed,
                )
            perf = evaluate_from_proba(proba_in, split["y_test_in"], proba_out, split["y_test_out"])
            rows.append({
                "dataset": dataset_name, "mismatch_alpha": alpha, "seed": seed, "method": method,
                "n_active_units": split["n_active_units"],
                "kl_divergence": kl, "wasserstein": wass, "mmd": mmd,
                "n_labelled": len(split["X_l"]), **perf,
            })

        if (i + 1) % progress_every == 0 or (i + 1) == len(combos):
            print(f"[{dataset_name}] [{i + 1}/{len(combos)}] combos done "
                  f"({time.time() - start:.1f}s elapsed)", file=sys.stderr)
    return pd.DataFrame(rows)


def run(config_path: str) -> pd.DataFrame:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    all_frames = []
    for dataset_name, dcfg in cfg["datasets"].items():
        print(f"=== {dataset_name} ===", file=sys.stderr)
        get_split = _SPLIT_FACTORIES[dataset_name](dcfg)
        methods = dcfg.get("methods", cfg["methods"])
        df = run_dataset_sweep(dataset_name, get_split, cfg["sweep"], methods)
        all_frames.append(df)

    results = pd.concat(all_frames, ignore_index=True)
    out_path = REPO_ROOT / cfg["output_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"Saved {len(results)} rows to {out_path}", file=sys.stderr)
    return results


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(
        REPO_ROOT / "configs" / "spatial_methods_comparison.yaml")
    run(config_path)
