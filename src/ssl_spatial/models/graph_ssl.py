"""Graph-based SSL methods from manuscript methods.tex Section 2.4: a plain
graph convolutional network (GCN) on a spatial k-nearest-neighbour graph
\\citep{kipf2017semi} ("modern consistency- and graph-based SSL"), and a
spatial GNN that incorporates geographic distance as edge weights rather
than a plain 0/1 adjacency ("spatially explicit models", needed for H3).

Both are transductive, following the original semi-supervised GCN
formulation: the graph spans every point whose label the study will ever
need to know about at fit time (labelled, unlabelled, in-region test, and
out-of-region test), the loss is computed on labelled nodes only, and
predictions for every other node fall out of the same forward pass. This is
why these methods use a different calling convention than the
`fit(...)`-then-`.predict(X)` sklearn-style baselines: see
`ssl_spatial.models.method_registry` for the uniform interface used to
compare all methods in the same sweep.

Manual dense-adjacency graph convolution is used instead of
`torch_geometric` (not installed, and unnecessary at the node counts
involved here -- at most a few thousand per run), following the standard
GCN propagation rule \\citep{kipf2017semi}:
    H^{(l+1)} = ReLU(D^{-1/2} (A + I) D^{-1/2} H^{(l)} W^{(l)}).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ssl_spatial.models._device import DEVICE


def _knn_adjacency(coords: np.ndarray, k: int = 10, weighted: bool = False,
                    bandwidth: float | None = None) -> np.ndarray:
    """Symmetric k-NN adjacency matrix. `weighted=False` gives the plain GCN
    (0/1 edges); `weighted=True` gives the spatial-GNN variant, with edge
    weights set by a Gaussian kernel of geographic distance (the bandwidth
    defaults to the median k-NN distance, a standard heuristic).
    """
    n = len(coords)
    dists = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dists, np.inf)
    knn_idx = np.argsort(dists, axis=1)[:, :k]

    A = np.zeros((n, n), dtype=np.float32)
    if weighted:
        if bandwidth is None:
            bandwidth = float(np.median(np.take_along_axis(dists, knn_idx, axis=1)))
        bandwidth = max(bandwidth, 1e-6)
    for i in range(n):
        for j in knn_idx[i]:
            w = np.exp(-(dists[i, j] ** 2) / (2 * bandwidth ** 2)) if weighted else 1.0
            A[i, j] = max(A[i, j], w)
            A[j, i] = max(A[j, i], w)
    return A


def _normalise_adjacency(A: np.ndarray) -> np.ndarray:
    A_hat = A + np.eye(len(A), dtype=A.dtype)
    deg = A_hat.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    np.power(deg, -0.5, where=deg > 0, out=d_inv_sqrt)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


class _GCN(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.W1 = nn.Linear(n_features, hidden, bias=True)
        self.W2 = nn.Linear(hidden, 1, bias=True)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        H = torch.relu(A_norm @ self.W1(X))
        return (A_norm @ self.W2(H)).squeeze(-1)


def fit_predict_graph_ssl(X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray,
                           X_test_in: np.ndarray, X_test_out: np.ndarray,
                           coords_l: np.ndarray, coords_u: np.ndarray,
                           coords_test_in: np.ndarray, coords_test_out: np.ndarray,
                           seed: int = 0, k: int = 10, hidden: int = 32,
                           n_epochs: int = 200, lr: float = 1e-2, weight_decay: float = 5e-4,
                           edge_weighted: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Transductive GCN (`edge_weighted=False`) or spatial GNN
    (`edge_weighted=True`). Returns (proba_test_in, proba_test_out) directly,
    per the uniform method-registry interface, since predictions for every
    node fall out of the same training forward pass.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_l, n_u = len(X_l), len(X_u)
    n_ti = len(X_test_in)
    X_all = np.vstack([X_l, X_u, X_test_in, X_test_out])
    coords_all = np.vstack([coords_l, coords_u, coords_test_in, coords_test_out])

    A = _knn_adjacency(coords_all, k=k, weighted=edge_weighted)
    A_norm = torch.as_tensor(_normalise_adjacency(A), dtype=torch.float32, device=DEVICE)
    X_all_t = torch.as_tensor(X_all, dtype=torch.float32, device=DEVICE)
    y_l_t = torch.as_tensor(y_l, dtype=torch.float32, device=DEVICE)

    model = _GCN(X_all.shape[1], hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(n_epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_all_t, A_norm)
        loss = bce(logits[:n_l], y_l_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        proba_all = torch.sigmoid(model(X_all_t, A_norm)).cpu().numpy()

    proba_test_in = proba_all[n_l + n_u: n_l + n_u + n_ti]
    proba_test_out = proba_all[n_l + n_u + n_ti:]
    return proba_test_in, proba_test_out


def fit_predict_gcn(*args, **kwargs):
    kwargs.pop("edge_weighted", None)
    return fit_predict_graph_ssl(*args, edge_weighted=False, **kwargs)


def fit_predict_spatial_gnn(*args, **kwargs):
    kwargs.pop("edge_weighted", None)
    return fit_predict_graph_ssl(*args, edge_weighted=True, **kwargs)


GRAPH_METHODS = {
    "gcn": fit_predict_gcn,
    "spatial_gnn": fit_predict_spatial_gnn,
}

__all__ = ["fit_predict_graph_ssl", "fit_predict_gcn", "fit_predict_spatial_gnn", "GRAPH_METHODS"]
