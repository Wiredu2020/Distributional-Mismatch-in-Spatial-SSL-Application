r"""WILDS PovertyMap benchmark loader for the §2.2.2 ToDo: evaluate the
controlled-mismatch methodology on an established distribution-shift
benchmark, so that the threshold-like degradation and method ranking found
on synthetic data (Section 5) are not an artefact of the synthetic
generator.

PovertyMap-WILDS \citep{koh2021wilds} pairs 224x224x8 Landsat + nighttime-light
image patches per DHS survey cluster with a continuous asset-wealth label
and an explicit country-level geographic split (`wilds.datasets.poverty_dataset`),
which is exactly the "geographic or region-based split" the ToDo calls for.

Rather than pulling the full ~13GB archive, this module talks directly to
the CodaLab bundle's file-level REST endpoint
(https://worksheets.codalab.org/rest/bundles/<id>/contents/blob/<path>),
which serves the small `dhs_metadata.csv` and individual per-cluster
`.npz` image files independently. We build a numeric feature vector per
cluster from summary statistics of each spectral/nightlight band (mean,
std, 10th/50th/90th percentile), in the spirit of the simple
satellite-derived covariates used in the poverty-mapping literature
\citep{jean2016combining} -- this keeps the resulting features compatible
with the sklearn-only SSL baselines in `models/baselines.py`, rather than
requiring a CNN feature extractor.

CodaLab's TLS certificate for worksheets.codalab.org is expired at the time
of writing (a known, publicly-documented server-side issue, not specific to
this environment); requests are made with certificate verification
disabled purely to work around that, over a connection that is otherwise
authenticated by the fixed, hardcoded bundle ID below.
"""
from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3
from wilds.datasets.poverty_dataset import BAND_ORDER, DHS_COUNTRIES, SURVEY_NAMES

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BUNDLE_ID = "0xfc0aa86ad9af4eb08c42dfc40eacf094"
BASE_URL = f"https://worksheets.codalab.org/rest/bundles/{BUNDLE_ID}/contents/blob"
N_BANDS = len(BAND_ORDER)  # 8: BLUE GREEN RED SWIR1 SWIR2 TEMP1 NIR NIGHTLIGHTS
N_STATS_PER_BAND = 5  # mean, std, p10, p50, p90
N_FEATURES = N_BANDS * N_STATS_PER_BAND


def _session() -> requests.Session:
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=0)
    s.mount("https://", adapter)
    return s


def download_metadata(data_dir: Path, force: bool = False) -> pd.DataFrame:
    """Download (or load a cached copy of) the DHS cluster metadata table:
    lat, lon, wealthpooled (continuous asset-wealth label), country, year,
    urban, nl_mean, nl_center, households. ~2.4MB, ~19,669 rows.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "dhs_metadata.csv"

    if not csv_path.exists() or force:
        delay = 2.0
        last_exc = None
        for attempt in range(5):
            try:
                resp = requests.get(f"{BASE_URL}/dhs_metadata.csv", verify=False, timeout=20)
                resp.raise_for_status()
                csv_path.write_bytes(resp.content)
                last_exc = None
                break
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(delay)
                delay = min(delay * 2, 15.0)
        if last_exc is not None:
            raise last_exc

    return pd.read_csv(csv_path)


def get_country_fold(fold: str = "A") -> dict:
    """Official WILDS train/val/test country groupings for the requested
    fold (A-E); val and test countries are geographically held out (OOD).
    """
    return SURVEY_NAMES[f"2009-17{fold}"]


def _image_url(idx: int) -> str:
    return f"{BASE_URL}/images/landsat_poverty_img_{idx}.npz"


def _download_one(idx: int, out_path: Path, session: requests.Session,
                   max_retries: int, timeout: float) -> tuple[int, bool, str]:
    if out_path.exists():
        return idx, True, "cached"

    delay = 1.5
    for attempt in range(max_retries):
        try:
            resp = session.get(_image_url(idx), verify=False, timeout=timeout)
            if resp.status_code == 200 and resp.content[:2] == b"PK":  # valid .npz (zip) magic
                tmp = out_path.with_suffix(".tmp")
                tmp.write_bytes(resp.content)
                tmp.rename(out_path)
                return idx, True, "ok"
            reason = f"http_{resp.status_code}"
        except requests.RequestException as exc:
            reason = type(exc).__name__
        time.sleep(delay + np.random.uniform(0, 0.5))
        delay = min(delay * 1.8, 20.0)
    return idx, False, reason


def download_images(idxs, data_dir: Path, max_workers: int = 6,
                     max_retries: int = 5, timeout: float = 30.0,
                     progress_every: int = 50) -> dict:
    """Download per-cluster .npz image patches for the given DHS cluster
    indices, skipping any already cached on disk. Returns a summary dict
    with the list of indices that failed after retries.

    Downloads are done with modest concurrency (default 6 workers) because
    the CodaLab bundle endpoint throttles/times out under heavy concurrent
    load (observed empirically); failures are retried with exponential
    backoff and are safe to re-run since completed files are cached.
    """
    images_dir = Path(data_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    idxs = list(dict.fromkeys(int(i) for i in idxs))  # dedupe, preserve order
    todo = [i for i in idxs if not (images_dir / f"landsat_poverty_img_{i}.npz").exists()]
    print(f"[wilds_poverty] {len(idxs) - len(todo)}/{len(idxs)} images already cached; "
          f"downloading {len(todo)} more", flush=True)

    failed = []
    done = 0
    session = _session()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_download_one, i, images_dir / f"landsat_poverty_img_{i}.npz",
                        session, max_retries, timeout): i
            for i in todo
        }
        for fut in as_completed(futures):
            idx, ok, reason = fut.result()
            done += 1
            if not ok:
                failed.append((idx, reason))
            if done % progress_every == 0 or done == len(todo):
                print(f"[wilds_poverty] {done}/{len(todo)} done, {len(failed)} failed", flush=True)

    if failed:
        warnings.warn(f"{len(failed)} images failed to download after {max_retries} retries: "
                       f"{failed[:10]}{'...' if len(failed) > 10 else ''}")

    return {"requested": len(idxs), "already_cached": len(idxs) - len(todo),
            "newly_downloaded": len(todo) - len(failed), "failed": [i for i, _ in failed]}


def band_summary_features(idx: int, data_dir: Path) -> np.ndarray | None:
    """Load one cluster's cached (8, 224, 224) image patch and reduce it to
    a 40-dim feature vector: [mean, std, p10, p50, p90] per band, in
    BAND_ORDER. Returns None if the file is not cached.
    """
    path = Path(data_dir) / "images" / f"landsat_poverty_img_{idx}.npz"
    if not path.exists():
        return None
    img = np.load(path)["x"]  # (8, 224, 224)
    feats = np.empty(N_FEATURES, dtype=np.float64)
    for b in range(N_BANDS):
        band = img[b]
        mean, std = band.mean(), band.std()
        p10, p50, p90 = np.percentile(band, [10, 50, 90])
        feats[b * N_STATS_PER_BAND:(b + 1) * N_STATS_PER_BAND] = [mean, std, p10, p50, p90]
    return feats


def build_feature_matrix(idxs, data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Build a (len(idxs), 40) feature matrix from cached images. Returns
    (features, valid_idxs) where valid_idxs excludes any index whose image
    failed to download/cache.
    """
    feats, valid = [], []
    for i in idxs:
        f = band_summary_features(i, data_dir)
        if f is not None:
            feats.append(f)
            valid.append(i)
    return np.vstack(feats) if feats else np.empty((0, N_FEATURES)), np.array(valid, dtype=int)


FEATURE_NAMES = [f"{band}_{stat}" for band in BAND_ORDER for stat in ("mean", "std", "p10", "p50", "p90")]


def reduce_dimensionality(X: np.ndarray, n_components: int = 5, seed: int = 0):
    """PCA-reduce the 40-dim band-summary features to a small number of
    components. The KDE-based KL-divergence estimator in
    `metrics.divergence` requires substantially more samples than
    dimensions to avoid a singular covariance matrix, which the raw 40-dim
    feature vector violates at the labelled-set sample sizes used here
    (n_labelled ~ 60); this mirrors why the synthetic generator itself only
    exposes 2 covariates. Returns (X_reduced, fitted_pca).
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=seed)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca


def build_pools(metadata: pd.DataFrame, fold: str = "A", candidate_per_country: int = 90,
                 n_unlabelled: int = 300, n_test_in: int = 150, n_test_out: int = 150,
                 seed: int = 0) -> dict:
    """Build the fixed pools the mismatch-severity sweep draws from, mirroring
    the synthetic generator's labelled/unlabelled/test_in/test_out roles:

      * `country_candidates` -- per (WILDS official "train"/in-distribution)
        country, a fixed random subsample of up to `candidate_per_country`
        clusters. The alpha sweep restricts *which countries'* candidates the
        labelled set is drawn from (see `labelled_indices_for_alpha`), mirroring
        how mismatch_alpha restricts the synthetic labelled sample to a
        shrinking spatial region.
      * `unlabelled_idx` / `test_in_idx` -- drawn uniformly across all
        in-distribution (train) countries, disjoint from every country's
        candidate pool, mirroring the synthetic unlabelled/in-region-test sets
        that are always drawn uniformly regardless of alpha.
      * `test_out_idx` -- drawn from the WILDS official OOD ("test") countries,
        i.e. countries never used for labelled/unlabelled data, giving a
        stronger and more realistic out-of-region test set than the synthetic
        pilot's "uniform over the same domain".

    These pools are fixed once (governed by `seed`, independent of the
    experiment seed used later to draw the labelled subsample) so that the
    total number of satellite images that need to be downloaded is bounded
    and reusable across the whole alpha x seed sweep.
    """
    metadata = metadata.reset_index(drop=True)
    folds = SURVEY_NAMES[f"2009-17{fold}"]
    train_countries = sorted(folds["train"])
    test_countries = folds["test"]

    rng = np.random.default_rng(seed)

    in_region = metadata[metadata["country"].isin(train_countries)]
    out_region = metadata[metadata["country"].isin(test_countries)]

    country_candidates = {}
    used_idx = set()
    for country in train_countries:
        rows = in_region[in_region["country"] == country].index.to_numpy().copy()
        rng.shuffle(rows)
        take = rows[:candidate_per_country]
        country_candidates[country] = take
        used_idx.update(take.tolist())

    remaining = in_region.index.to_numpy()
    remaining = remaining[~np.isin(remaining, list(used_idx))].copy()
    rng.shuffle(remaining)

    if len(remaining) < n_unlabelled + n_test_in:
        raise ValueError(
            f"Not enough non-candidate in-region clusters ({len(remaining)}) for "
            f"n_unlabelled + n_test_in = {n_unlabelled + n_test_in}; reduce "
            f"candidate_per_country or the pool sizes."
        )
    unlabelled_idx = remaining[:n_unlabelled]
    test_in_idx = remaining[n_unlabelled:n_unlabelled + n_test_in]

    out_idx_all = out_region.index.to_numpy().copy()
    rng.shuffle(out_idx_all)
    test_out_idx = out_idx_all[:n_test_out]

    return {
        "fold": fold,
        "train_countries": train_countries,
        "test_countries": test_countries,
        "country_candidates": country_candidates,
        "unlabelled_idx": unlabelled_idx,
        "test_in_idx": test_in_idx,
        "test_out_idx": test_out_idx,
    }


def all_pool_indices(pools: dict) -> np.ndarray:
    """Union of every index referenced by the pools -- exactly the set of
    images that needs to be downloaded once, up front, for the whole sweep.
    """
    parts = [pools["unlabelled_idx"], pools["test_in_idx"], pools["test_out_idx"]]
    parts += list(pools["country_candidates"].values())
    return np.unique(np.concatenate(parts))


def n_active_countries(n_train_countries: int, alpha: float) -> int:
    """Number of (nested, fixed-order) countries the labelled set is drawn
    from at a given mismatch severity `alpha`, mirroring the synthetic
    generator's bandwidth interpolation: alpha=0 spans all training
    countries (same distribution as the unlabelled pool); alpha=1 restricts
    the labelled set to a single country.
    """
    k = round(n_train_countries - (n_train_countries - 1) * alpha)
    return int(np.clip(k, 1, n_train_countries))


def labelled_indices_for_alpha(pools: dict, alpha: float, n_labelled: int, seed: int) -> np.ndarray:
    """Draw the labelled index set for a given mismatch severity `alpha` and
    experiment `seed`. Restricts to the first `k(alpha)` training countries
    (fixed alphabetical order, so higher alpha is always a *subset* of lower
    alpha's countries -- analogous to shrinking the synthetic sampling
    bandwidth around a fixed anchor) and draws `n_labelled` points at random
    from the union of those countries' candidate pools.
    """
    countries = pools["train_countries"]
    k = n_active_countries(len(countries), alpha)
    active = countries[:k]
    pool = np.concatenate([pools["country_candidates"][c] for c in active])

    rng = np.random.default_rng(seed * 10_000 + int(round(alpha * 100)))
    n_draw = min(n_labelled, len(pool))
    return rng.choice(pool, size=n_draw, replace=False)


__all__ = [
    "BUNDLE_ID", "N_FEATURES", "FEATURE_NAMES", "DHS_COUNTRIES", "SURVEY_NAMES",
    "download_metadata", "get_country_fold", "download_images",
    "band_summary_features", "build_feature_matrix",
    "build_pools", "all_pool_indices", "n_active_countries", "labelled_indices_for_alpha",
    "reduce_dimensionality",
]
