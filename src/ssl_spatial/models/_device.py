"""Shared GPU/CPU device selection for every PyTorch-based SSL method
(`neural_ssl.py`, `graph_ssl.py`, `gnnwr.py`, `neural_distribution_aware.py`).
Picks CUDA when available (e.g. a Colab GPU runtime) and falls back to CPU
otherwise, so the exact same code runs unchanged in both environments.
"""
from __future__ import annotations

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = ["DEVICE"]
