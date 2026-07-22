"""Domain-adversarial distribution-aware SSL (H4 strengthening attempt,
supervisor-revision round): motivated directly by the H4 ablation
(`notebooks/h4_ablation_v2.ipynb`), which showed that the manuscript's
existing distribution-aware framework regresses specifically because of its
*fixed-bandwidth spatial-distance* weighting term, which collapses toward
uninformative near-zero weights once every labelled point is far from the
unlabelled centroid, rather than because of adding spatial awareness per se.

This module replaces that fixed heuristic with a *learned* covariate-shift
correction -- domain-adversarial training (DANN; Ganin et al., 2016, already
cited in the manuscript's Related Work) -- combined with Mean-Teacher-style
consistency regularisation (`ssl_spatial.models.neural_ssl`, reusing its
`_ramp_up` schedule) on the unlabelled pool. A shared MLP trunk feeds two
heads: a label predictor (trained normally on labelled data) and a domain
classifier (trained to separate labelled from unlabelled points, but with a
gradient-reversal layer between it and the trunk, so the trunk is pushed to
produce features the domain classifier *cannot* separate). Because the
adversarial game is against a discriminator that adapts to the actual data
at hand rather than a fixed spatial kernel evaluated at one summary point,
it should not exhibit the same collapse-to-uninformative-weights failure
mode identified in the ablation.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ssl_spatial.models._device import DEVICE
from ssl_spatial.models.neural_ssl import _ramp_up


class _GradientReversal(torch.autograd.Function):
    """Identity in the forward pass; negates (and scales by `lambd`) the
    gradient in the backward pass, so gradient descent on the trunk's
    parameters acts as gradient *ascent* on the domain classifier's loss --
    the standard DANN mechanism for making the trunk's features
    domain-invariant without alternating optimisers.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def _grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradientReversal.apply(x, lambd)


class _DASSLModel(nn.Module):
    """Shared MLP trunk with a label-prediction head and a gradient-reversed
    domain-classification head."""

    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.label_head = nn.Linear(hidden, 1)
        self.domain_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def label_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.label_head(self.trunk(x)).squeeze(-1)

    def forward(self, x: torch.Tensor, lambd: float) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.label_head(h).squeeze(-1), self.domain_head(_grad_reverse(h, lambd)).squeeze(-1)


class _DASSLWrapper:
    """Exposes the trained (teacher) model's label head through sklearn's
    `.predict` / `.predict_proba` interface, matching
    `neural_ssl._TorchClassifierWrapper`."""

    def __init__(self, model: _DASSLModel, classes_: np.ndarray):
        self.model = model.eval()
        self.classes_ = classes_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = self.model.label_logits(torch.as_tensor(X, dtype=torch.float32, device=DEVICE))
            p1 = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def fit_domain_adversarial_ssl(X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray, seed: int = 0,
                                n_epochs: int = 150, lr: float = 1e-2, hidden: int = 32,
                                noise_std: float = 0.15, ema_decay: float = 0.97,
                                consistency_max: float = 1.0, domain_lambda_max: float = 1.0,
                                **kwargs):
    """Domain-adversarial + consistency-regularised SSL: the H4 strengthening
    attempt motivated by the ablation study. Both the domain-adversarial
    loss weight and the consistency loss weight follow the same sigmoid
    ramp-up used by Mean Teacher (`neural_ssl._ramp_up`), so the model
    relies on labelled supervision early in training before the
    domain-invariance and consistency terms are weighted heavily.
    """
    torch.manual_seed(seed)
    n_features = X_l.shape[1]
    student = _DASSLModel(n_features, hidden=hidden).to(DEVICE)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)

    X_l_t = torch.as_tensor(X_l, dtype=torch.float32, device=DEVICE)
    y_l_t = torch.as_tensor(y_l, dtype=torch.float32, device=DEVICE)
    X_u_t = torch.as_tensor(X_u, dtype=torch.float32, device=DEVICE) if len(X_u) else None

    domain_y_l = torch.zeros(len(X_l_t), device=DEVICE)
    domain_y_u = torch.ones(len(X_u_t), device=DEVICE) if X_u_t is not None else None

    opt = torch.optim.Adam(student.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        student.train()
        opt.zero_grad()
        progress = _ramp_up(epoch, n_epochs)
        lambd = domain_lambda_max * progress

        label_logits_l, domain_logits_l = student(X_l_t, lambd)
        loss = bce(label_logits_l, y_l_t) + bce(domain_logits_l, domain_y_l)

        if X_u_t is not None and len(X_u_t) > 0:
            _, domain_logits_u = student(X_u_t, lambd)
            loss = loss + bce(domain_logits_u, domain_y_u)

            consistency_weight = consistency_max * progress
            noise1 = torch.randn_like(X_u_t) * noise_std
            noise2 = torch.randn_like(X_u_t) * noise_std
            student_label_logits = student.label_logits(X_u_t + noise1)
            with torch.no_grad():
                teacher_label_logits = teacher.label_logits(X_u_t + noise2)
            consistency = nn.functional.mse_loss(
                torch.sigmoid(student_label_logits), torch.sigmoid(teacher_label_logits))
            loss = loss + consistency_weight * consistency

        loss.backward()
        opt.step()

        with torch.no_grad():
            for t_p, s_p in zip(teacher.parameters(), student.parameters()):
                t_p.mul_(ema_decay).add_(s_p, alpha=1 - ema_decay)

    return _DASSLWrapper(teacher, np.unique(y_l))


NEURAL_DISTRIBUTION_AWARE_METHODS = {"domain_adversarial_ssl": fit_domain_adversarial_ssl}

__all__ = ["fit_domain_adversarial_ssl", "NEURAL_DISTRIBUTION_AWARE_METHODS"]
