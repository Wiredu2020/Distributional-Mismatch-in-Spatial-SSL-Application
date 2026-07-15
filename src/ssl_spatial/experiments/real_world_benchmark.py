"""Real-world dataset benchmark runner (manuscript methods.tex Section
2.2.3, "Real-World Datasets" ToDo): re-runs the controlled-mismatch
methodology, baseline SSL methods, and Section 2.6 re-weighted self-training
on one of three real, geocoded, tabular datasets --

  * housing            -- California Housing block groups, region = nearest county
  * socioeconomic       -- USDA county-level poverty/income/education, region = US state
  * air_quality         -- EPA AQS PM2.5 monitors, region = US state

-- using the shared region-restriction sweep design in
`ssl_spatial.data.region_mismatch` (mismatch severity = number of source
regions the labelled set is drawn from, mirroring the WILDS PovertyMap
country-restriction design).

Usage:
    python -m ssl_spatial.experiments.real_world_benchmark configs/<name>_benchmark.yaml
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ssl_spatial.data.region_mismatch import (
    all_pool_indices, build_region_pools, labelled_indices_for_alpha,
)
from ssl_spatial.metrics.divergence import kl_divergence_kde, mmd_rbf, wasserstein_marginal
from ssl_spatial.metrics.evaluation import evaluate_classifier
from ssl_spatial.models.baselines import BASELINE_METHODS
from ssl_spatial.models.distribution_aware import DISTRIBUTION_AWARE_METHODS

REPO_ROOT = Path(__file__).resolve().parents[3]
ALL_METHODS = {**BASELINE_METHODS, **DISTRIBUTION_AWARE_METHODS}


def _load_dataset(name: str, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    data_dir = REPO_ROOT / cfg["data_dir"]
    if name == "housing":
        from ssl_spatial.data.housing import FEATURE_NAMES, load_housing
        return load_housing(data_dir), FEATURE_NAMES
    if name == "socioeconomic":
        from ssl_spatial.data.socioeconomic import FEATURE_NAMES, load_socioeconomic
        return load_socioeconomic(data_dir), FEATURE_NAMES
    if name == "air_quality":
        from ssl_spatial.data.air_quality import FEATURE_NAMES, load_air_quality
        socio_dir = REPO_ROOT / cfg["socioeconomic_data_dir"]
        return load_air_quality(data_dir, socio_dir), FEATURE_NAMES
    raise ValueError(f"unknown dataset {name!r}")


def _train_ood_region_split(df: pd.DataFrame, train_fraction: float, seed: int) -> tuple[list, list]:
    """Randomly split regions into train/out-of-distribution sets (fixed
    seed), then order the train regions by ascending geographic distance
    from an "anchor" region (the largest training region by row count) --
    nearest first, so the anchor itself is first.

    The nested mismatch-severity schedule restricts to a shrinking *prefix*
    of this order, so this ordering makes the labelled set at high severity
    a spatially *concentrated* cluster of regions around the anchor, exactly
    mirroring the synthetic generator's labelled data concentrating around a
    fixed anchor as `mismatch_alpha` increases (Section 2.2.1) -- rather
    than an arbitrary (alphabetical) or sample-size-driven ordering, neither
    of which is guaranteed to produce a spatially coherent, genuinely
    mismatched labelled region. Anchoring on the largest region also
    guarantees enough candidates to draw `n_labelled` from even at the most
    restrictive severity level.
    """
    regions = df["region"].to_numpy()
    unique_regions, counts = np.unique(regions, return_counts=True)
    size_by_region = dict(zip(unique_regions, counts))

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_regions)
    n_train = max(2, int(round(len(shuffled) * train_fraction)))
    train_regions = shuffled[:n_train].tolist()
    ood_regions = sorted(shuffled[n_train:].tolist())

    centroids = df.groupby("region")[["lat", "lon"]].mean()
    anchor_region = max(train_regions, key=lambda r: size_by_region[r])
    anchor = centroids.loc[anchor_region]
    train_regions = sorted(
        train_regions,
        key=lambda r: (centroids.loc[r, "lat"] - anchor["lat"]) ** 2
        + (centroids.loc[r, "lon"] - anchor["lon"]) ** 2,
    )
    return train_regions, ood_regions


def run(config_path: str) -> pd.DataFrame:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset"]
    rcfg = cfg["regions"]
    sweep = cfg["sweep"]
    methods = cfg["methods"]

    df, feature_names = _load_dataset(dataset_name, cfg)
    regions = df["region"].to_numpy()
    train_regions, ood_regions = _train_ood_region_split(df, rcfg["train_fraction"], rcfg["split_seed"])

    pools = build_region_pools(
        regions, train_regions, ood_regions,
        candidate_per_region=rcfg["candidate_per_region"], n_unlabelled=rcfg["n_unlabelled"],
        n_test_in=rcfg["n_test_in"], n_test_out=rcfg["n_test_out"], seed=rcfg["pool_seed"],
    )

    X_all = df[feature_names].to_numpy(dtype=float)
    y_all = df["label"].to_numpy()
    mean, std = X_all.mean(axis=0), X_all.std(axis=0) + 1e-12
    X_all = (X_all - mean) / std

    X_u = X_all[pools["unlabelled_idx"]]
    X_in, y_in = X_all[pools["test_in_idx"]], y_all[pools["test_in_idx"]]
    X_out, y_out = X_all[pools["test_out_idx"]], y_all[pools["test_out_idx"]]

    rows = []
    combos = [(a, s) for a in sweep["mismatch_alpha"] for s in sweep["seeds"]]
    start = time.time()
    for i, (alpha, seed) in enumerate(combos):
        labelled_idx = labelled_indices_for_alpha(pools, alpha, sweep["n_labelled"], seed)
        X_l, y_l = X_all[labelled_idx], y_all[labelled_idx]

        kl = kl_divergence_kde(X_l, X_u)
        wass, _ = wasserstein_marginal(X_l, X_u)
        mmd = mmd_rbf(X_l, X_u)

        n_train_regions = len(pools["train_regions"])
        from ssl_spatial.data.region_mismatch import n_active_regions
        k = n_active_regions(n_train_regions, alpha)

        for method in methods:
            fit_fn = ALL_METHODS[method]
            clf = fit_fn(X_l, y_l, X_u, seed=seed)
            perf = evaluate_classifier(clf, X_in, y_in, X_out, y_out)

            rows.append({
                "dataset": dataset_name,
                "mismatch_alpha": alpha,
                "n_active_regions": k,
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
            print(f"[{dataset_name}] [{i + 1}/{len(combos)}] combos done ({elapsed:.1f}s elapsed)", file=sys.stderr)

    results = pd.DataFrame(rows)
    out_path = REPO_ROOT / cfg["output_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"Saved {len(results)} rows to {out_path}", file=sys.stderr)
    return results


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "configs" / "housing_benchmark.yaml")
    run(config_path)
