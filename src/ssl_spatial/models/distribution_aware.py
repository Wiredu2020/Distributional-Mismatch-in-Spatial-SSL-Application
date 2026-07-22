"""Distribution-aware SSL framework (manuscript methods.tex Section 2.6):
the re-weighting component layered on top of self-training, implemented
first per the manuscript's own recommendation, since self-training shows
the largest H1 effect and the clearest mechanism (accumulating confidently
wrong pseudo-labels from the mismatched region) of the three baselines
(Section 3.2), giving the most direct test of H4 without requiring the
full spatial-kernel or GNN machinery.

The re-weighting module estimates a density ratio between the unlabelled
and labelled covariate distributions via a labelled-vs-unlabelled
discriminator (a standard density-ratio-estimation trick), then uses it as
an importance weight on the labelled samples during self-training, in the
spirit of the covariate-shift correction of \\citet{shimodaira2000improving}:
labelled points that look like the (mismatched) unlabelled/target
distribution are up-weighted; labelled points deep in the over-represented
labelled region are down-weighted.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def estimate_density_ratio_weights(X_l: np.ndarray, X_u: np.ndarray, seed: int = 0,
                                    clip: tuple[float, float] = (0.1, 10.0)) -> np.ndarray:
    """Estimate per-labelled-point importance weights w(x) ~ p_U(x) / p_L(x)
    via a logistic-regression discriminator trained to separate labelled
    from unlabelled points. Weights are clipped for stability and
    normalised to mean 1 so the effective labelled sample size is
    preserved.
    """
    X_disc = np.vstack([X_l, X_u])
    y_disc = np.concatenate([np.zeros(len(X_l)), np.ones(len(X_u))])
    disc = LogisticRegression(max_iter=1000, random_state=seed)
    disc.fit(X_disc, y_disc)

    p_unlabelled = disc.predict_proba(X_l)[:, list(disc.classes_).index(1.0)]
    p_unlabelled = np.clip(p_unlabelled, 0.01, 0.99)
    weights = p_unlabelled / (1 - p_unlabelled)
    weights = np.clip(weights, *clip)
    return weights / weights.mean()


def fit_reweighted_self_training(X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray, seed: int = 0,
                                  threshold: float = 0.75, max_iter: int = 10, **kwargs):
    """Self-training with labelled points re-weighted by an estimated
    density ratio (Section 2.6 H4 test). Otherwise identical to
    `models.baselines.fit_self_training`: an SVC base learner, iteratively
    fit and used to pseudo-label the most confident remaining unlabelled
    points (confidence >= `threshold`) until no more points are accepted or
    `max_iter` rounds elapse.
    """
    weights = estimate_density_ratio_weights(X_l, X_u, seed=seed)

    X_train, y_train, w_train = X_l.copy(), y_l.copy(), weights.copy()
    X_pool = X_u.copy()
    clf = SVC(probability=True, random_state=seed)

    for _ in range(max_iter):
        clf.fit(X_train, y_train, sample_weight=w_train)
        if len(X_pool) == 0:
            break
        proba = clf.predict_proba(X_pool)
        confidence = proba.max(axis=1)
        predictions = clf.classes_[proba.argmax(axis=1)]
        accept = confidence >= threshold
        if not np.any(accept):
            break
        X_train = np.vstack([X_train, X_pool[accept]])
        y_train = np.concatenate([y_train, predictions[accept]])
        # newly accepted pseudo-labelled points are drawn from the unlabelled
        # pool itself, so they already represent the target distribution and
        # are not down/up-weighted.
        w_train = np.concatenate([w_train, np.ones(int(accept.sum()))])
        X_pool = X_pool[~accept]

    return clf


def estimate_spatial_relevance_weights(coords_l: np.ndarray, coords_u: np.ndarray,
                                        bandwidth: float | None = None) -> np.ndarray:
    """Weight for each labelled point based on geographic proximity to the
    unlabelled (target) distribution's centroid: the spatial-weighting
    component of the distribution-aware framework, which incorporates
    geographic distance directly into the training objective rather than
    only into post-hoc evaluation. Complements the covariate-based density
    ratio with an explicit geographic signal.
    """
    centroid_u = coords_u.mean(axis=0)
    dist = np.sqrt(np.sum((coords_l - centroid_u) ** 2, axis=1))
    if bandwidth is None:
        bandwidth = float(np.std(dist)) + 1e-6
    weights = np.exp(-dist ** 2 / (2 * bandwidth ** 2))
    return weights / (weights.mean() + 1e-12)


def _adaptive_thresholds(coords_pool: np.ndarray, coords_l: np.ndarray, base_threshold: float,
                          bonus: float, bandwidth: float | None = None) -> np.ndarray:
    """Per-point pseudo-label acceptance threshold: lower (more lenient) for
    unlabelled points geographically close to the labelled region, where
    the model is interpolating rather than extrapolating and pseudo-labels
    are more trustworthy; higher (stricter) for points far from the
    labelled region, where self-training's core failure mode -- confidently
    wrong pseudo-labels compounding from the mismatched region -- is most
    likely.
    """
    centroid_l = coords_l.mean(axis=0)
    dist = np.sqrt(np.sum((coords_pool - centroid_l) ** 2, axis=1))
    if bandwidth is None:
        bandwidth = float(np.std(dist)) + 1e-6
    relevance = np.exp(-dist ** 2 / (2 * bandwidth ** 2))  # 1 = close/relevant, ~0 = far
    thresholds = (base_threshold - bonus) + 2 * bonus * (1 - relevance)
    return np.clip(thresholds, 0.5, 0.97)


def _fit_distribution_aware_variant(X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray, seed: int,
                                     coords_l: np.ndarray, coords_u: np.ndarray,
                                     use_spatial_weighting: bool, use_adaptive_threshold: bool,
                                     threshold: float = 0.75, threshold_bonus: float = 0.15,
                                     max_iter: int = 10):
    """Shared self-training loop for every coordinate-aware distribution-aware
    variant, parameterised by which of the two spatial components (spatial
    weighting, adaptive pseudo-labelling threshold) are switched on. Density-
    ratio re-weighting is always applied -- it is the framework's base
    component (Section 2.6) that every variant builds on. Used directly by
    `fit_full_distribution_aware` (both components on) and by the two
    ablation variants below (exactly one component on), so the H4 ablation
    isolates which added component is responsible for the full framework's
    regression relative to re-weighting alone (Supplementary Material
    Section S2.1's open question).
    """
    density_weights = estimate_density_ratio_weights(X_l, X_u, seed=seed)
    if use_spatial_weighting:
        spatial_weights = estimate_spatial_relevance_weights(coords_l, coords_u)
        combined_weights = density_weights * spatial_weights
        combined_weights = combined_weights / (combined_weights.mean() + 1e-12)
    else:
        combined_weights = density_weights

    X_train, y_train, w_train = X_l.copy(), y_l.copy(), combined_weights.copy()
    X_pool, coords_pool = X_u.copy(), coords_u.copy()
    coords_train = coords_l.copy()
    clf = SVC(probability=True, random_state=seed)

    for _ in range(max_iter):
        clf.fit(X_train, y_train, sample_weight=w_train)
        if len(X_pool) == 0:
            break
        proba = clf.predict_proba(X_pool)
        confidence = proba.max(axis=1)
        predictions = clf.classes_[proba.argmax(axis=1)]

        if use_adaptive_threshold:
            accept_threshold = _adaptive_thresholds(coords_pool, coords_train, threshold, threshold_bonus)
        else:
            accept_threshold = threshold
        accept = confidence >= accept_threshold
        if not np.any(accept):
            break

        X_train = np.vstack([X_train, X_pool[accept]])
        y_train = np.concatenate([y_train, predictions[accept]])
        coords_train = np.vstack([coords_train, coords_pool[accept]])
        w_train = np.concatenate([w_train, np.ones(int(accept.sum()))])
        X_pool, coords_pool = X_pool[~accept], coords_pool[~accept]

    return clf


def fit_full_distribution_aware(X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray, seed: int = 0,
                                 coords_l: np.ndarray | None = None, coords_u: np.ndarray | None = None,
                                 threshold: float = 0.75, threshold_bonus: float = 0.15,
                                 max_iter: int = 10, **kwargs):
    """The distribution-aware framework's three components layered on
    self-training (manuscript methods.tex Section 2.6): density-ratio
    re-weighting (as in `fit_reweighted_self_training`), a spatial
    weighting mechanism (`estimate_spatial_relevance_weights`), and
    adaptive pseudo-labelling (`_adaptive_thresholds`). Falls back to
    `fit_reweighted_self_training` if coordinates are not supplied, since
    the spatial components have nothing to act on without them.
    """
    if coords_l is None or coords_u is None:
        return fit_reweighted_self_training(X_l, y_l, X_u, seed=seed, threshold=threshold, max_iter=max_iter)
    return _fit_distribution_aware_variant(
        X_l, y_l, X_u, seed, coords_l, coords_u,
        use_spatial_weighting=True, use_adaptive_threshold=True,
        threshold=threshold, threshold_bonus=threshold_bonus, max_iter=max_iter,
    )


def fit_reweighted_spatial_self_training(X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray, seed: int = 0,
                                          coords_l: np.ndarray | None = None, coords_u: np.ndarray | None = None,
                                          threshold: float = 0.75, max_iter: int = 10, **kwargs):
    """H4 ablation variant: density-ratio re-weighting + spatial weighting,
    WITHOUT adaptive pseudo-labelling. Isolates whether spatial weighting
    alone (rather than its combination with the adaptive threshold) accounts
    for the full framework's regression relative to re-weighting alone.
    Falls back to `fit_reweighted_self_training` if coordinates are absent.
    """
    if coords_l is None or coords_u is None:
        return fit_reweighted_self_training(X_l, y_l, X_u, seed=seed, threshold=threshold, max_iter=max_iter)
    return _fit_distribution_aware_variant(
        X_l, y_l, X_u, seed, coords_l, coords_u,
        use_spatial_weighting=True, use_adaptive_threshold=False,
        threshold=threshold, max_iter=max_iter,
    )


def fit_reweighted_adaptive_self_training(X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray, seed: int = 0,
                                           coords_l: np.ndarray | None = None, coords_u: np.ndarray | None = None,
                                           threshold: float = 0.75, threshold_bonus: float = 0.15,
                                           max_iter: int = 10, **kwargs):
    """H4 ablation variant: density-ratio re-weighting + adaptive
    pseudo-labelling threshold, WITHOUT spatial weighting. Isolates whether
    the adaptive threshold alone accounts for the full framework's
    regression relative to re-weighting alone. Falls back to
    `fit_reweighted_self_training` if coordinates are absent.
    """
    if coords_l is None or coords_u is None:
        return fit_reweighted_self_training(X_l, y_l, X_u, seed=seed, threshold=threshold, max_iter=max_iter)
    return _fit_distribution_aware_variant(
        X_l, y_l, X_u, seed, coords_l, coords_u,
        use_spatial_weighting=False, use_adaptive_threshold=True,
        threshold=threshold, threshold_bonus=threshold_bonus, max_iter=max_iter,
    )


DISTRIBUTION_AWARE_METHODS = {
    "reweighted_self_training": fit_reweighted_self_training,
    "reweighted_spatial_self_training": fit_reweighted_spatial_self_training,
    "reweighted_adaptive_self_training": fit_reweighted_adaptive_self_training,
    "full_distribution_aware": fit_full_distribution_aware,
}

__all__ = [
    "estimate_density_ratio_weights", "estimate_spatial_relevance_weights",
    "fit_reweighted_self_training", "fit_reweighted_spatial_self_training",
    "fit_reweighted_adaptive_self_training", "fit_full_distribution_aware",
    "DISTRIBUTION_AWARE_METHODS",
]
