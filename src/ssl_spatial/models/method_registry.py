"""Unified method registry for the comparative study
(`notebooks/spatial_methods_comparison.ipynb`): every method -- the four
already evaluated in the synthetic/WILDS/real-world sweeps plus the six new
ones added here (Mean Teacher, FixMatch, GCN, spatial GNN, GNNWR,
geostatistical SSL) -- exposed through one call signature:

    fit_predict(X_l, y_l, X_u, X_test_in, X_test_out,
                coords_l, coords_u, coords_test_in, coords_test_out,
                seed) -> (proba_test_in, proba_test_out)

Sklearn-style methods (the four baselines) don't need the test data at fit
time; the adapters below simply fit-then-predict internally and discard the
unused arguments. Graph-based methods are inherently transductive and use
the full node set from the start (see `ssl_spatial.models.graph_ssl`).
GNNWR is supervised-only and ignores the unlabelled data (see
`ssl_spatial.models.gnnwr`). This uniform interface is what lets one sweep
loop compare all ten methods identically.
"""
from __future__ import annotations

import numpy as np

from ssl_spatial.metrics.evaluation import _positive_class_proba
from ssl_spatial.models.baselines import BASELINE_METHODS
from ssl_spatial.models.distribution_aware import DISTRIBUTION_AWARE_METHODS
from ssl_spatial.models.geostatistical import GEOSTATISTICAL_METHODS
from ssl_spatial.models.gnnwr import GNNWR_METHODS
from ssl_spatial.models.graph_ssl import GRAPH_METHODS
from ssl_spatial.models.neural_ssl import NEURAL_SSL_METHODS

_SKLEARN_STYLE = {**BASELINE_METHODS, **DISTRIBUTION_AWARE_METHODS, **NEURAL_SSL_METHODS}


def _make_sklearn_adapter(fit_fn):
    def adapter(X_l, y_l, X_u, X_test_in, X_test_out, coords_l, coords_u,
                coords_test_in, coords_test_out, seed):
        clf = fit_fn(X_l, y_l, X_u, seed=seed, coords_l=coords_l, coords_u=coords_u)
        return _positive_class_proba(clf, X_test_in), _positive_class_proba(clf, X_test_out)
    return adapter


def _make_geostatistical_adapter(fit_fn):
    def adapter(X_l, y_l, X_u, X_test_in, X_test_out, coords_l, coords_u,
                coords_test_in, coords_test_out, seed):
        clf = fit_fn(X_l, y_l, X_u, seed=seed, coords_l=coords_l, coords_u=coords_u)
        return _positive_class_proba(clf, X_test_in), _positive_class_proba(clf, X_test_out)
    return adapter


ALL_METHODS_UNIFORM = {}
ALL_METHODS_UNIFORM.update({name: _make_sklearn_adapter(fn) for name, fn in _SKLEARN_STYLE.items()})
ALL_METHODS_UNIFORM.update({name: _make_geostatistical_adapter(fn) for name, fn in GEOSTATISTICAL_METHODS.items()})
ALL_METHODS_UNIFORM.update(GRAPH_METHODS)
ALL_METHODS_UNIFORM.update(GNNWR_METHODS)

__all__ = ["ALL_METHODS_UNIFORM"]
