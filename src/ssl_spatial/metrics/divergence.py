"""Metrics for quantifying marginal distribution mismatch between labelled
and unlabelled data (manuscript section 6.3), plus spatial extensions that
localise the divergence to geographic sub-regions instead of a single global
number.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import gaussian_kde, wasserstein_distance
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel

_LOG_FLOOR = -50.0


def kl_divergence_kde(X_p: np.ndarray, X_q: np.ndarray) -> float:
    """Plug-in KDE estimate of KL(P || Q) using P-samples as evaluation points.

    Approximate and only reliable in low dimensions (a handful of features),
    which matches the covariate dimensionality used in the synthetic study.
    """
    kde_p = gaussian_kde(X_p.T)
    kde_q = gaussian_kde(X_q.T)
    log_p = np.clip(kde_p.logpdf(X_p.T), _LOG_FLOOR, None)
    log_q = np.clip(kde_q.logpdf(X_p.T), _LOG_FLOOR, None)
    return float(np.mean(log_p - log_q))


def wasserstein_marginal(X_p: np.ndarray, X_q: np.ndarray) -> tuple[float, np.ndarray]:
    """Average of per-feature (1D) Wasserstein distances.

    Called "marginal" because it measures mismatch feature-by-feature rather
    than jointly, mirroring the manuscript's framing of *marginal*
    distribution mismatch.
    """
    per_feature = np.array([
        wasserstein_distance(X_p[:, j], X_q[:, j]) for j in range(X_p.shape[1])
    ])
    return float(per_feature.mean()), per_feature


def mmd_rbf(X_p: np.ndarray, X_q: np.ndarray, gamma: float | None = None) -> float:
    """Squared MMD with an RBF kernel; bandwidth set by the median heuristic
    over the pooled sample unless `gamma` is given explicitly.
    """
    if gamma is None:
        pooled = np.vstack([X_p, X_q])
        dists = euclidean_distances(pooled, pooled)
        med = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
        gamma = 1.0 / (2 * med ** 2 + 1e-12)

    k_pp = rbf_kernel(X_p, X_p, gamma=gamma)
    k_qq = rbf_kernel(X_q, X_q, gamma=gamma)
    k_pq = rbf_kernel(X_p, X_q, gamma=gamma)
    mmd2 = k_pp.mean() + k_qq.mean() - 2 * k_pq.mean()
    return float(max(mmd2, 0.0))


DIVERGENCE_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "kl": kl_divergence_kde,
    "wasserstein": lambda p, q: wasserstein_marginal(p, q)[0],
    "mmd": mmd_rbf,
}


def localized_divergence(
    coords_p: np.ndarray,
    X_p: np.ndarray,
    coords_q: np.ndarray,
    X_q: np.ndarray,
    metric: str = "mmd",
    n_bins: int = 5,
    min_count: int = 5,
) -> dict:
    """Partition [0,1]^2 into an n_bins x n_bins grid and compute the chosen
    divergence metric within each cell that has enough points from both
    distributions. Returns per-cell values plus a count-weighted aggregate.

    This is the spatial extension referenced in section 6.3: it reveals
    *where* mismatch is concentrated rather than collapsing it to one number.
    """
    metric_fn = DIVERGENCE_METRICS[metric]
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    def cell_ids(coords: np.ndarray) -> np.ndarray:
        bx = np.clip(np.digitize(coords[:, 0], edges[1:-1]), 0, n_bins - 1)
        by = np.clip(np.digitize(coords[:, 1], edges[1:-1]), 0, n_bins - 1)
        return bx * n_bins + by

    cells_p = cell_ids(coords_p)
    cells_q = cell_ids(coords_q)

    per_cell = {}
    weights = []
    values = []
    for cell in np.unique(np.concatenate([cells_p, cells_q])):
        mask_p = cells_p == cell
        mask_q = cells_q == cell
        n_p, n_q = mask_p.sum(), mask_q.sum()
        if n_p < min_count or n_q < min_count:
            continue
        val = metric_fn(X_p[mask_p], X_q[mask_q])
        per_cell[int(cell)] = val
        weights.append(min(n_p, n_q))
        values.append(val)

    if not values:
        return {"per_cell": {}, "aggregate": float("nan"), "n_cells_used": 0}

    weights = np.array(weights, dtype=float)
    values = np.array(values, dtype=float)
    aggregate = float(np.average(values, weights=weights))
    return {"per_cell": per_cell, "aggregate": aggregate, "n_cells_used": len(values)}


def kernel_weighted_local_mmd(
    coords_p: np.ndarray,
    X_p: np.ndarray,
    coords_q: np.ndarray,
    X_q: np.ndarray,
    query_points: np.ndarray | None = None,
    n_grid: int = 5,
    bandwidth: float | None = None,
    feature_gamma: float | None = None,
) -> dict:
    """Spatially weighted local MMD: at each query location, every point in
    both samples contributes to the local divergence estimate with a
    Gaussian-kernel weight based on its distance to the query location,
    rather than being included only if it falls inside a hard grid cell.

    This replaces the fixed-grid localised MMD estimator, whose correlation
    with accuracy loss was found to be far weaker than the global metrics
    because too few grid cells retain enough points from both distributions
    once global mismatch is severe. Because every point always receives a
    strictly positive (if small) weight, the estimate is always defined and
    degrades smoothly as local sample density falls, rather than the cell
    simply being dropped below a minimum-count threshold. The aggregate
    across query locations is weighted by each location's effective local
    sample size (Kish's effective-sample-size formula applied to the
    kernel weights), so query locations with almost no local support
    contribute little to the aggregate rather than being all-or-nothing.
    """
    if query_points is None:
        lin = (np.arange(n_grid) + 0.5) / n_grid
        gx, gy = np.meshgrid(lin, lin)
        query_points = np.column_stack([gx.ravel(), gy.ravel()])
    if bandwidth is None:
        bandwidth = 0.5 / n_grid

    if feature_gamma is None:
        pooled = np.vstack([X_p, X_q])
        dists = euclidean_distances(pooled, pooled)
        med = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
        feature_gamma = 1.0 / (2 * med ** 2 + 1e-12)

    k_pp = rbf_kernel(X_p, X_p, gamma=feature_gamma)
    k_qq = rbf_kernel(X_q, X_q, gamma=feature_gamma)
    k_pq = rbf_kernel(X_p, X_q, gamma=feature_gamma)

    per_query, values, eff_sizes = {}, [], []
    for i, q in enumerate(query_points):
        w_p = np.exp(-np.sum((coords_p - q) ** 2, axis=1) / (2 * bandwidth ** 2))
        w_q = np.exp(-np.sum((coords_q - q) ** 2, axis=1) / (2 * bandwidth ** 2))

        eff_n_p = (w_p.sum() ** 2) / (np.sum(w_p ** 2) + 1e-12)
        eff_n_q = (w_q.sum() ** 2) / (np.sum(w_q ** 2) + 1e-12)

        w_p_norm = w_p / (w_p.sum() + 1e-12)
        w_q_norm = w_q / (w_q.sum() + 1e-12)

        mmd2 = float(max(
            w_p_norm @ k_pp @ w_p_norm + w_q_norm @ k_qq @ w_q_norm
            - 2 * w_p_norm @ k_pq @ w_q_norm,
            0.0,
        ))
        per_query[i] = mmd2
        values.append(mmd2)
        eff_sizes.append(min(eff_n_p, eff_n_q))

    values = np.array(values)
    eff_sizes = np.array(eff_sizes)
    aggregate = float(np.average(values, weights=eff_sizes + 1e-6))
    return {"per_query": per_query, "aggregate": aggregate, "effective_sizes": eff_sizes.tolist()}
