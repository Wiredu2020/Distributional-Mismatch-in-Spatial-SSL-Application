"""Modern consistency-based SSL methods from manuscript methods.tex Section
2.4 ("modern consistency- and graph-based SSL"): Mean Teacher
\\citep{tarvainen2017mean} and a FixMatch-style pseudo-labelling-plus-
consistency method \\citep{sohn2020fixmatch}, adapted from their original
image-augmentation setting to tabular spatial covariates by using additive
Gaussian noise as the "weak"/"strong" augmentation (a standard substitute
for consistency regularisation on non-image data, since there is no
canonical augmentation for arbitrary numeric features the way there is for
images). Both use a small MLP and full-batch training, since every dataset
in this study has at most a few thousand labelled+unlabelled points.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ssl_spatial.models._device import DEVICE


class _MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class _TorchClassifierWrapper:
    """Wraps a trained binary MLP to expose sklearn's `.predict` /
    `.predict_proba` interface."""

    def __init__(self, model: nn.Module, classes_: np.ndarray):
        self.model = model.eval()
        self.classes_ = classes_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = self.model(torch.as_tensor(X, dtype=torch.float32, device=DEVICE))
            p1 = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def _ramp_up(step: int, total_steps: int) -> float:
    """Sigmoid ramp-up for the consistency-loss weight (Tarvainen & Valpola,
    2017), so the model relies on supervised signal early in training before
    the unlabelled consistency term is weighted heavily."""
    progress = np.clip(step / max(total_steps, 1), 0.0, 1.0)
    return float(np.exp(-5 * (1 - progress) ** 2))


def fit_mean_teacher(X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray, seed: int = 0,
                      n_epochs: int = 150, lr: float = 1e-2, noise_std: float = 0.15,
                      ema_decay: float = 0.97, consistency_max: float = 1.0, **kwargs):
    torch.manual_seed(seed)
    n_features = X_l.shape[1]
    student = _MLP(n_features).to(DEVICE)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)

    X_l_t = torch.as_tensor(X_l, dtype=torch.float32, device=DEVICE)
    y_l_t = torch.as_tensor(y_l, dtype=torch.float32, device=DEVICE)
    X_u_t = torch.as_tensor(X_u, dtype=torch.float32, device=DEVICE) if len(X_u) else None

    opt = torch.optim.Adam(student.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        student.train()
        opt.zero_grad()
        loss = bce(student(X_l_t), y_l_t)

        if X_u_t is not None and len(X_u_t) > 0:
            weight = consistency_max * _ramp_up(epoch, n_epochs)
            noise1 = torch.randn_like(X_u_t) * noise_std
            noise2 = torch.randn_like(X_u_t) * noise_std
            student_logits = student(X_u_t + noise1)
            with torch.no_grad():
                teacher_logits = teacher(X_u_t + noise2)
            consistency = nn.functional.mse_loss(torch.sigmoid(student_logits), torch.sigmoid(teacher_logits))
            loss = loss + weight * consistency

        loss.backward()
        opt.step()

        with torch.no_grad():
            for t_p, s_p in zip(teacher.parameters(), student.parameters()):
                t_p.mul_(ema_decay).add_(s_p, alpha=1 - ema_decay)

    return _TorchClassifierWrapper(teacher, np.unique(y_l))


def fit_fixmatch(X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray, seed: int = 0,
                  n_epochs: int = 150, lr: float = 1e-2, weak_noise_std: float = 0.05,
                  strong_noise_std: float = 0.35, confidence_threshold: float = 0.9,
                  lambda_u: float = 1.0, **kwargs):
    torch.manual_seed(seed)
    n_features = X_l.shape[1]
    model = _MLP(n_features).to(DEVICE)

    X_l_t = torch.as_tensor(X_l, dtype=torch.float32, device=DEVICE)
    y_l_t = torch.as_tensor(y_l, dtype=torch.float32, device=DEVICE)
    X_u_t = torch.as_tensor(X_u, dtype=torch.float32, device=DEVICE) if len(X_u) else None

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        loss = bce(model(X_l_t), y_l_t)

        if X_u_t is not None and len(X_u_t) > 0:
            with torch.no_grad():
                weak_logits = model(X_u_t + torch.randn_like(X_u_t) * weak_noise_std)
                weak_prob = torch.sigmoid(weak_logits)
                confidence = torch.maximum(weak_prob, 1 - weak_prob)
                pseudo_label = (weak_prob > 0.5).float()
                confident = confidence >= confidence_threshold

            if confident.any():
                strong_logits = model(X_u_t[confident] + torch.randn_like(X_u_t[confident]) * strong_noise_std)
                unsup_loss = bce(strong_logits, pseudo_label[confident])
                loss = loss + lambda_u * unsup_loss

        loss.backward()
        opt.step()

    return _TorchClassifierWrapper(model, np.unique(y_l))


NEURAL_SSL_METHODS = {
    "mean_teacher": fit_mean_teacher,
    "fixmatch": fit_fixmatch,
}

__all__ = ["fit_mean_teacher", "fit_fixmatch", "NEURAL_SSL_METHODS"]
