"""Generic region-based mismatch-severity sweep, shared across the
real-world datasets (housing, urban socio-economic, environmental
monitoring) evaluated for the manuscript's "Real-World Datasets" ToDo
(methods.tex Section 2.2.3).

This generalises the country-restriction design used for the WILDS
PovertyMap benchmark (`ssl_spatial.data.wilds_poverty`) to an arbitrary
categorical "region" column: mismatch severity is operationalised by
restricting the labelled set to a shrinking, nested subset of "training"
regions (ordered so each severity level's regions are a subset of the
previous level's), while the unlabelled pool and in-region test set are
drawn uniformly across all training regions and the out-of-region test set
is drawn from a held-out set of "out-of-distribution" regions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

COUNTY_CENTROIDS_URL = (
    "https://gist.githubusercontent.com/russellsamora/"
    "12be4f9f574e92413ea3f92ce1bc58e6/raw/us_county_latlng.csv"
)


def download_us_county_centroids(cache_path) -> pd.DataFrame:
    """FIPS code, name, lat/lon centroid for every US county (~3,143 rows).
    Real data (derived from Census TIGER/Line), cached locally after first
    download; used to geocode county-level tabular datasets that don't
    ship their own coordinates.
    """
    from pathlib import Path
    cache_path = Path(cache_path)
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(COUNTY_CENTROIDS_URL, timeout=30)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
    df = pd.read_csv(cache_path, dtype={"fips_code": str})
    df["fips_code"] = df["fips_code"].str.zfill(5)
    return df


def assign_nearest_region(coords: np.ndarray, region_centroids: pd.DataFrame,
                           lat_col: str = "lat", lon_col: str = "lng",
                           label_col: str = "name") -> np.ndarray:
    """Assign each (lat, lon) row in `coords` to its nearest region centroid
    (simple Euclidean nearest-neighbour on lat/lon -- adequate at
    within-state spatial scales). Returns an array of region labels.
    """
    centroids = region_centroids[[lat_col, lon_col]].to_numpy()
    labels = region_centroids[label_col].to_numpy()
    # (N, 1, 2) - (1, M, 2) -> (N, M, 2) squared distances, chunked to bound memory
    assigned = np.empty(len(coords), dtype=labels.dtype)
    chunk = 2000
    for start in range(0, len(coords), chunk):
        end = min(start + chunk, len(coords))
        diff = coords[start:end, None, :] - centroids[None, :, :]
        dist2 = np.sum(diff ** 2, axis=-1)
        assigned[start:end] = labels[np.argmin(dist2, axis=1)]
    return assigned


def build_region_pools(regions: np.ndarray, train_regions: list, ood_regions: list,
                        candidate_per_region: int, n_unlabelled: int, n_test_in: int,
                        n_test_out: int, seed: int = 0) -> dict:
    """Build the fixed pools the mismatch-severity sweep draws from.

    `train_regions` must already be in the fixed, nested order used by
    `labelled_indices_for_alpha` (e.g. alphabetical); `ood_regions` are held
    out entirely from labelled/unlabelled/in-region-test and used only for
    the out-of-region test set.
    """
    regions = np.asarray(regions)
    all_idx = np.arange(len(regions))
    rng = np.random.default_rng(seed)

    region_candidates = {}
    used_idx = set()
    for region in train_regions:
        rows = all_idx[regions == region]
        rng.shuffle(rows)
        take = rows[:candidate_per_region]
        region_candidates[region] = take
        used_idx.update(take.tolist())

    in_region_mask = np.isin(regions, train_regions)
    remaining = all_idx[in_region_mask]
    remaining = remaining[~np.isin(remaining, list(used_idx))]
    rng.shuffle(remaining)

    if len(remaining) < n_unlabelled + n_test_in:
        raise ValueError(
            f"Not enough non-candidate in-region rows ({len(remaining)}) for "
            f"n_unlabelled + n_test_in = {n_unlabelled + n_test_in}; reduce "
            f"candidate_per_region or the pool sizes."
        )
    unlabelled_idx = remaining[:n_unlabelled]
    test_in_idx = remaining[n_unlabelled:n_unlabelled + n_test_in]

    ood_idx_all = all_idx[np.isin(regions, ood_regions)]
    rng.shuffle(ood_idx_all)
    if len(ood_idx_all) < n_test_out:
        raise ValueError(
            f"Not enough out-of-distribution rows ({len(ood_idx_all)}) for "
            f"n_test_out = {n_test_out}."
        )
    test_out_idx = ood_idx_all[:n_test_out]

    return {
        "train_regions": list(train_regions),
        "ood_regions": list(ood_regions),
        "region_candidates": region_candidates,
        "unlabelled_idx": unlabelled_idx,
        "test_in_idx": test_in_idx,
        "test_out_idx": test_out_idx,
    }


def all_pool_indices(pools: dict) -> np.ndarray:
    parts = [pools["unlabelled_idx"], pools["test_in_idx"], pools["test_out_idx"]]
    parts += list(pools["region_candidates"].values())
    return np.unique(np.concatenate(parts))


def n_active_regions(n_train_regions: int, alpha: float) -> int:
    """Number of (nested, fixed-order) regions the labelled set is drawn
    from at a given mismatch severity `alpha`: alpha=0 spans all training
    regions, alpha=1 restricts the labelled set to a single region.
    """
    k = round(n_train_regions - (n_train_regions - 1) * alpha)
    return int(np.clip(k, 1, n_train_regions))


def labelled_indices_for_alpha(pools: dict, alpha: float, n_labelled: int, seed: int) -> np.ndarray:
    regions = pools["train_regions"]
    k = n_active_regions(len(regions), alpha)
    active = regions[:k]
    pool = np.concatenate([pools["region_candidates"][r] for r in active])

    rng = np.random.default_rng(seed * 10_000 + int(round(alpha * 100)))
    n_draw = min(n_labelled, len(pool))
    return rng.choice(pool, size=n_draw, replace=False)


__all__ = [
    "download_us_county_centroids", "assign_nearest_region", "build_region_pools",
    "all_pool_indices", "n_active_regions", "labelled_indices_for_alpha",
]
