"""Performance evaluation metrics from manuscript section 6.5, step 2:
accuracy/F1, calibration error, and spatial generalisation error (the gap
between in-region and out-of-region test performance).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def expected_calibration_error(y_true: np.ndarray, y_prob_pos: np.ndarray, n_bins: int = 10) -> float:
    """Standard binned ECE for a binary classifier's positive-class probability."""
    confidence = np.maximum(y_prob_pos, 1 - y_prob_pos)
    prediction = (y_prob_pos >= 0.5).astype(int)
    correct = (prediction == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        if not np.any(mask):
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def _positive_class_proba(clf, X: np.ndarray) -> np.ndarray:
    classes = list(clf.classes_)
    proba = clf.predict_proba(X)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return proba[:, -1]


def evaluate_classifier(clf, X_test_in: np.ndarray, y_test_in: np.ndarray,
                         X_test_out: np.ndarray, y_test_out: np.ndarray) -> dict:
    """Evaluate on an in-region test set (same biased region as labelled data)
    and an out-of-region test set (uniform over the domain). The gap between
    the two is the spatial generalisation error referenced in section 6.5.
    """
    pred_in = clf.predict(X_test_in)
    pred_out = clf.predict(X_test_out)
    proba_in = _positive_class_proba(clf, X_test_in)
    proba_out = _positive_class_proba(clf, X_test_out)

    acc_in = accuracy_score(y_test_in, pred_in)
    acc_out = accuracy_score(y_test_out, pred_out)

    return {
        "accuracy_in_region": acc_in,
        "accuracy_out_region": acc_out,
        "spatial_generalisation_gap": acc_in - acc_out,
        "f1_in_region": f1_score(y_test_in, pred_in, zero_division=0),
        "f1_out_region": f1_score(y_test_out, pred_out, zero_division=0),
        "ece_in_region": expected_calibration_error(y_test_in, proba_in),
        "ece_out_region": expected_calibration_error(y_test_out, proba_out),
    }
