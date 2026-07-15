r"""Simplified geographically neural network weighted regression (GNNWR)
\citep{wang2022house}: the second spatially explicit model needed for H3
(manuscript methods.tex Section 2.4), alongside the graph-based models in
`ssl_spatial.models.graph_ssl`.

GNNWR generalises geographically weighted regression (GWR) by learning the
spatial weighting kernel with a neural network instead of fixing it a
priori (e.g. a Gaussian kernel with a hand-tuned bandwidth). The full
architecture in the original paper learns a local *regression coefficient
vector* per location via weighted least squares; this implementation
simplifies that to a locally-weighted (Nadaraya-Watson-style) classifier: a
small "spatial weighting network" (SWNN) maps the relative position between
a query location and each labelled point to a non-negative weight, and the
prediction at the query location is the SWNN-weighted average of the
labelled points' 0/1 labels. The SWNN is trained end-to-end (leave-one-out
over the labelled set) to minimise binary cross-entropy, so the learned
weighting is whatever locally-weighted kernel best predicts the labelled
data, rather than a fixed distance-decay function.

Unlike every other method in the comparative study, GNNWR here is a
*supervised* spatial baseline -- it does not use the unlabelled data at all
-- included specifically to test whether spatial awareness alone, without
any semi-supervised mechanism, changes robustness to marginal distribution
mismatch (H3), which is the comparison the manuscript's spatially-explicit
model class is for.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class _SWNN(nn.Module):
    """Maps a relative spatial position (query - reference) to a
    non-negative weight."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),  # input: [dx, dy, euclidean distance]
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Softplus(),
        )

    def forward(self, rel_pos: torch.Tensor) -> torch.Tensor:
        return self.net(rel_pos).squeeze(-1)


def _relative_position_features(coords_query: torch.Tensor, coords_ref: torch.Tensor) -> torch.Tensor:
    """(n_query, n_ref, 3) tensor of [dx, dy, distance] for every
    query/reference pair."""
    diff = coords_query[:, None, :] - coords_ref[None, :, :]
    dist = torch.linalg.norm(diff, dim=-1, keepdim=True)
    return torch.cat([diff, dist], dim=-1)


class _GNNWRWrapper:
    def __init__(self, swnn: nn.Module, coords_l: torch.Tensor, y_l: torch.Tensor):
        self.swnn = swnn.eval()
        self.coords_l = coords_l
        self.y_l = y_l

    def _predict_proba_at(self, coords_query: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            rel = _relative_position_features(coords_query, self.coords_l)
            weights = self.swnn(rel)  # (n_query, n_l)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            proba = weights @ self.y_l
        return proba.numpy()


def fit_predict_gnnwr(X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray,
                       X_test_in: np.ndarray, X_test_out: np.ndarray,
                       coords_l: np.ndarray, coords_u: np.ndarray,
                       coords_test_in: np.ndarray, coords_test_out: np.ndarray,
                       seed: int = 0, hidden: int = 16, n_epochs: int = 300,
                       lr: float = 1e-2) -> tuple[np.ndarray, np.ndarray]:
    """Fits the SWNN on labelled data only, then predicts at the in-region
    and out-of-region test coordinates. Follows the same
    (proba_in, proba_out) return convention as the graph-based methods for
    a uniform comparison interface (`ssl_spatial.models.method_registry`).
    """
    torch.manual_seed(seed)
    swnn = _SWNN(hidden=hidden)
    opt = torch.optim.Adam(swnn.parameters(), lr=lr)
    bce = nn.BCELoss()

    coords_l_t = torch.as_tensor(coords_l, dtype=torch.float32)
    y_l_t = torch.as_tensor(y_l, dtype=torch.float32)
    n_l = len(y_l)
    eye_mask = ~torch.eye(n_l, dtype=torch.bool)  # leave-one-out during training

    for _ in range(n_epochs):
        swnn.train()
        opt.zero_grad()
        rel = _relative_position_features(coords_l_t, coords_l_t)
        weights = swnn(rel)
        weights = weights.masked_fill(~eye_mask, 0.0)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        proba_loo = weights @ y_l_t
        loss = bce(proba_loo.clamp(1e-6, 1 - 1e-6), y_l_t)
        loss.backward()
        opt.step()

    model = _GNNWRWrapper(swnn, coords_l_t, y_l_t)
    proba_in = model._predict_proba_at(torch.as_tensor(coords_test_in, dtype=torch.float32))
    proba_out = model._predict_proba_at(torch.as_tensor(coords_test_out, dtype=torch.float32))
    return proba_in, proba_out


GNNWR_METHODS = {"gnnwr": fit_predict_gnnwr}

__all__ = ["fit_predict_gnnwr", "GNNWR_METHODS"]
