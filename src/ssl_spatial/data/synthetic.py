"""Synthetic spatial data generator for studying marginal distribution
mismatch between labelled and unlabelled data in semi-supervised learning.

Two independent axes of spatial data violation are exposed as knobs, matching
the manuscript's distinction between:

  * marginal distribution mismatch  -> controlled via `mismatch_alpha`
    (labelled data is sampled from a spatially concentrated region while
    unlabelled/test data is sampled from the full domain, so p_L(x) != p_U(x))

  * spatial non-stationarity        -> controlled via `nonstationarity_strength`
    (the relationship between features and label itself varies over space,
    so the labelling function f(x, s) is not spatially constant)

Spatial autocorrelation in the covariates is controlled via `lengthscale`
(features are generated as smoothed Gaussian random fields on a grid).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class SpatialSSLConfig:
    grid_resolution: int = 100          # grid is grid_resolution x grid_resolution over [0,1]^2
    n_features: int = 2                 # number of spatially autocorrelated covariates
    lengthscale: float = 0.08           # spatial autocorrelation strength (fraction of domain)
    nonstationarity_strength: float = 0.0  # 0 = stationary relationship, >0 = spatially varying coefficient
    mismatch_alpha: float = 0.0         # 0 = labelled sampled like unlabelled, 1 = maximally concentrated
    label_anchor: tuple = (0.25, 0.25)  # centre of the region labelled data is biased towards
    bandwidth_min: float = 0.10         # labelled-sampling kernel bandwidth at alpha=1 (concentrated)
    bandwidth_max: float = 0.75         # labelled-sampling kernel bandwidth at alpha=0 (near-uniform)
    n_labelled: int = 150
    n_unlabelled: int = 600
    n_test_in: int = 300                # test points drawn from the same biased region as labels
    n_test_out: int = 300               # test points drawn uniformly (out-of-region generalisation)
    noise_std: float = 0.3
    beta: tuple = field(default_factory=lambda: (1.5, -1.2))  # base coefficients for the first features


@dataclass
class SpatialSSLDataset:
    coords: np.ndarray        # (N, 2) coordinates of the sampled universe
    X: np.ndarray             # (N, n_features) covariates
    y: np.ndarray             # (N,) binary labels in {0, 1}
    idx_labelled: np.ndarray
    idx_unlabelled: np.ndarray
    idx_test_in: np.ndarray
    idx_test_out: np.ndarray
    config: SpatialSSLConfig


def _smoothed_random_field(rng: np.random.Generator, resolution: int, lengthscale: float) -> np.ndarray:
    """White noise smoothed by a Gaussian filter -> a spatially autocorrelated field.

    `lengthscale` is expressed as a fraction of the domain side length; larger
    values produce smoother, more strongly autocorrelated fields.
    """
    sigma_px = max(lengthscale * resolution, 1e-6)
    noise = rng.standard_normal((resolution, resolution))
    field_ = gaussian_filter(noise, sigma=sigma_px, mode="reflect")
    field_ = (field_ - field_.mean()) / (field_.std() + 1e-12)
    return field_


def _sampling_weights(coords: np.ndarray, anchor: np.ndarray, bandwidth: float) -> np.ndarray:
    sq_dist = np.sum((coords - anchor) ** 2, axis=1)
    w = np.exp(-sq_dist / (2 * bandwidth ** 2))
    return w / w.sum()


def generate_spatial_ssl_dataset(config: SpatialSSLConfig, seed: int = 0) -> SpatialSSLDataset:
    rng = np.random.default_rng(seed)
    res = config.grid_resolution

    # 1. Universe of candidate locations on a regular grid over [0, 1]^2.
    lin = (np.arange(res) + 0.5) / res
    gx, gy = np.meshgrid(lin, lin)
    coords_full = np.column_stack([gx.ravel(), gy.ravel()])  # (res*res, 2)

    # 2. Spatially autocorrelated covariates.
    features = []
    for j in range(config.n_features):
        f = _smoothed_random_field(rng, res, config.lengthscale)
        features.append(f.ravel())
    X_full = np.column_stack(features)

    # 3. Spatial non-stationarity: coefficient on feature 0 varies smoothly over space.
    #    g(s) in [-1, 1] modulates the effective coefficient away from its base value.
    g = np.sin(2 * np.pi * coords_full[:, 0]) * np.cos(2 * np.pi * coords_full[:, 1])
    beta0_eff = config.beta[0] + config.nonstationarity_strength * g

    latent = beta0_eff * X_full[:, 0]
    for j in range(1, config.n_features):
        beta_j = config.beta[j] if j < len(config.beta) else 0.0
        latent = latent + beta_j * X_full[:, j]

    latent = (latent - latent.mean()) / (latent.std() + 1e-12)
    latent = latent + rng.normal(0, config.noise_std, size=latent.shape)
    y_full = (latent > 0).astype(int)

    # 4. Controlled sampling: labelled data concentrated near `label_anchor`,
    #    with concentration set by mismatch_alpha; unlabelled/test drawn uniformly.
    anchor = np.array(config.label_anchor)
    bandwidth = config.bandwidth_max * (1 - config.mismatch_alpha) + config.bandwidth_min * config.mismatch_alpha
    p_labelled = _sampling_weights(coords_full, anchor, bandwidth)

    n_total_needed = config.n_labelled + config.n_unlabelled + config.n_test_in + config.n_test_out
    if n_total_needed > coords_full.shape[0]:
        raise ValueError("grid_resolution too small for requested sample sizes")

    all_idx = np.arange(coords_full.shape[0])

    idx_labelled = rng.choice(all_idx, size=config.n_labelled, replace=False, p=p_labelled)
    remaining = np.setdiff1d(all_idx, idx_labelled, assume_unique=False)

    idx_test_in = rng.choice(
        remaining, size=config.n_test_in, replace=False,
        p=_renormalise(p_labelled, remaining),
    )
    remaining = np.setdiff1d(remaining, idx_test_in, assume_unique=False)

    # unlabelled and out-of-region test are drawn uniformly at random (unbiased target distribution)
    idx_unlabelled = rng.choice(remaining, size=config.n_unlabelled, replace=False)
    remaining = np.setdiff1d(remaining, idx_unlabelled, assume_unique=False)

    idx_test_out = rng.choice(remaining, size=config.n_test_out, replace=False)

    return SpatialSSLDataset(
        coords=coords_full,
        X=X_full,
        y=y_full,
        idx_labelled=idx_labelled,
        idx_unlabelled=idx_unlabelled,
        idx_test_in=idx_test_in,
        idx_test_out=idx_test_out,
        config=config,
    )


def _renormalise(weights: np.ndarray, subset_idx: np.ndarray) -> np.ndarray:
    w = weights[subset_idx]
    return w / w.sum()
