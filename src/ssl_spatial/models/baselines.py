"""Baseline (non-spatial) SSL methods from manuscript section 6.4, plus a
supervised-only control. Every `fit_*` function returns a fitted classifier
exposing the standard sklearn `.predict` / `.predict_proba` interface so all
methods can be evaluated identically downstream.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.svm import SVC


def fit_supervised_only(X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray, seed: int = 0, **kwargs):
    """Ignores unlabelled data entirely -- the control condition.

    Accepts and ignores arbitrary keyword arguments (e.g. `coords_l`,
    `coords_u`) so that every method in the comparative study
    (`notebooks/spatial_methods_comparison.ipynb`) can be called through the
    same signature, whether or not it actually uses spatial coordinates.
    """
    clf = RandomForestClassifier(n_estimators=200, random_state=seed)
    clf.fit(X_l, y_l)
    return clf


def fit_self_training(X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray, seed: int = 0,
                       threshold: float = 0.75, **kwargs):
    """Pseudo-labelling: relies on the cluster/low-density-separation
    assumption, so expected to be more sensitive to distribution mismatch
    (H2)."""
    base = SVC(probability=True, random_state=seed)
    X_all = np.vstack([X_l, X_u])
    y_all = np.concatenate([y_l, np.full(len(X_u), -1)])
    clf = SelfTrainingClassifier(base, threshold=threshold)
    clf.fit(X_all, y_all)
    return clf


def fit_label_propagation(X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray, seed: int = 0,
                           gamma: float = 20.0, **kwargs):
    """Graph-based SSL relying on the manifold assumption; expected to be
    highly sensitive to spatial non-stationarity and mismatch (H2)."""
    X_all = np.vstack([X_l, X_u])
    y_all = np.concatenate([y_l, np.full(len(X_u), -1)])
    clf = LabelSpreading(kernel="rbf", gamma=gamma)
    clf.fit(X_all, y_all)
    return clf


BASELINE_METHODS = {
    "supervised_only": fit_supervised_only,
    "self_training": fit_self_training,
    "label_propagation": fit_label_propagation,
}
