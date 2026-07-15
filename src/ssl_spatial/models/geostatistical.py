r"""Geostatistical semi-supervised learning for spatial prediction
\citep{fouedjio2022geostatistical}: the spatially explicit method added to
the comparative study in `notebooks/spatial_methods_comparison.ipynb`
(manuscript methods.tex Section 2.4, "spatially explicit models", and the
Fouedjio/Talebi paper in `Papers/`).

The original method (Algorithm 1 of the paper) is defined for a continuous
target: (1) fit a variogram to the labelled target values, (2) select
unlabelled locations within a quarter of the variogram range of a labelled
point, (3) generate an ensemble of L pseudo-label realisations at those
locations via geostatistical conditional simulation (conditioned on the
labelled data), (4) train one Random Forest per pseudo-labelled realisation
(original labelled data + one simulated realisation), and (5) aggregate the
ensemble of Random Forests by averaging.

This study's target is binary, not continuous, so steps (1)-(3) here treat
the 0/1 label as the simulated variable directly (an indicator-style
simulation: the paper's own concluding remarks note that a real
implementation for categorical targets would use sequential indicator or
plurigaussian simulation instead of Gaussian conditional simulation; we
approximate this with a single Gaussian-conditioned simulation of the
indicator variable, thresholded at 0.5, since `gstools` -- the geostatistics
library used here -- does not implement indicator simulation directly).
Step (4)-(5) use `RandomForestClassifier` and average predicted
probabilities instead of continuous predictions, otherwise following the
paper's ensemble-of-random-forests design exactly.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class _GeostatisticalSSLEnsemble:
    """Fitted ensemble wrapper exposing the standard `.predict` /
    `.predict_proba` interface via probability-averaging across the L
    pseudo-training-set Random Forests (Eq. 1 of the paper, adapted to
    classification)."""

    def __init__(self, models: list, classes_: np.ndarray):
        self.models = models
        self.classes_ = classes_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.zeros((len(X), len(self.classes_)))
        for model in self.models:
            proba = model.predict_proba(X)
            for j, c in enumerate(model.classes_):
                probs[:, list(self.classes_).index(c)] += proba[:, j]
        return probs / len(self.models)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def fit_geostatistical_ssl(X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray, seed: int = 0,
                            coords_l: np.ndarray | None = None, coords_u: np.ndarray | None = None,
                            n_realizations: int = 30, range_fraction: float = 0.25,
                            n_estimators: int = 200, **kwargs):
    """Geostatistical semi-supervised Random Forest (Fouedjio & Talebi,
    2022). Requires `coords_l`/`coords_u`; falls back to a plain supervised
    Random Forest (ignoring unlabelled data) if coordinates are not
    provided, or if the variogram cannot be estimated (e.g. too few
    labelled points, or a degenerate/pure-nugget fit).
    """
    if coords_l is None or coords_u is None or len(X_u) == 0:
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
        clf.fit(X_l, y_l)
        return clf

    import gstools as gs

    rng = np.random.default_rng(seed)
    y_indicator = y_l.astype(float)

    try:
        bin_center, gamma = gs.vario_estimate((coords_l[:, 0], coords_l[:, 1]), y_indicator)
        model = gs.Exponential(dim=2)
        model.fit_variogram(bin_center, gamma, nugget=True)
        vario_range = model.len_scale
        if not np.isfinite(vario_range) or vario_range <= 0:
            raise ValueError("degenerate variogram range")
    except Exception:
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
        clf.fit(X_l, y_l)
        return clf

    # Select unlabelled locations within `range_fraction` of the variogram
    # range from the nearest labelled point (Algorithm 1, step 2).
    dists = np.min(
        np.sqrt(((coords_u[:, None, :] - coords_l[None, :, :]) ** 2).sum(-1)), axis=1
    )
    threshold_dist = range_fraction * vario_range
    near_mask = dists <= threshold_dist
    if not np.any(near_mask):
        near_mask = np.ones(len(coords_u), dtype=bool)  # fall back to all unlabelled points
    coords_sim = coords_u[near_mask]
    X_sim = X_u[near_mask]

    krige = gs.krige.Ordinary(model, cond_pos=(coords_l[:, 0], coords_l[:, 1]), cond_val=y_indicator)
    csrf = gs.CondSRF(krige)

    classes = np.unique(y_l)
    models = []
    for l in range(n_realizations):
        sim_seed = int(rng.integers(0, 2**31 - 1))
        sim_values = csrf((coords_sim[:, 0], coords_sim[:, 1]), seed=sim_seed, mesh_type="unstructured")
        pseudo_y = (sim_values > 0.5).astype(y_l.dtype)

        X_train = np.vstack([X_l, X_sim])
        y_train = np.concatenate([y_l, pseudo_y])
        if len(np.unique(y_train)) < 2:
            continue  # degenerate realisation (all one class); skip

        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed * 1000 + l)
        rf.fit(X_train, y_train)
        models.append(rf)

    if not models:
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
        clf.fit(X_l, y_l)
        return clf

    return _GeostatisticalSSLEnsemble(models, classes)


GEOSTATISTICAL_METHODS = {"geostatistical_ssl": fit_geostatistical_ssl}

__all__ = ["fit_geostatistical_ssl", "GEOSTATISTICAL_METHODS"]
